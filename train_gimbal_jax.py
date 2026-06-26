import argparse
import time
from pathlib import Path
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
import optax
import flax.linen as nn
from flax.training.train_state import TrainState
from flax import struct

# Import JAX Gimbal Env
from gimbal_env_jax import GimbalJaxEnv, EnvState, OBSERVATION_NAMES

# =====================================================================
# FLAX NEURAL NETWORKS
# =====================================================================

class Actor(nn.Module):
    action_dim: int = 3

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(400)(x)
        x = nn.relu(x)
        x = nn.Dense(300)(x)
        x = nn.relu(x)
        x = nn.Dense(self.action_dim)(x)
        
        # action[0] = thrust, map to [0, 1] using sigmoid
        thrust = nn.sigmoid(x[..., 0:1])
        # action[1:3] = TVC yaw/pitch, map to [-1, 1] using tanh
        tvc = nn.tanh(x[..., 1:3])
        
        return jnp.concatenate([thrust, tvc], axis=-1)

class Critic(nn.Module):
    @nn.compact
    def __call__(self, obs, action):
        x = jnp.concatenate([obs, action], axis=-1)
        x = nn.Dense(400)(x)
        x = nn.relu(x)
        x = nn.Dense(300)(x)
        x = nn.relu(x)
        q = nn.Dense(1)(x)
        return q

class TwinCritic(nn.Module):
    @nn.compact
    def __call__(self, obs, action):
        q1 = Critic()(obs, action)
        q2 = Critic()(obs, action)
        return q1, q2

# =====================================================================
# REPLAY BUFFER (JAX-Native PyTree)
# =====================================================================

@struct.dataclass
class ReplayBuffer:
    obs: jnp.ndarray
    actions: jnp.ndarray
    rewards: jnp.ndarray
    next_obs: jnp.ndarray
    dones: jnp.ndarray
    ptr: int
    size: int
    max_size: int

    @classmethod
    def create(cls, max_size, obs_dim, action_dim):
        return cls(
            obs=jnp.zeros((max_size, obs_dim), dtype=jnp.float32),
            actions=jnp.zeros((max_size, action_dim), dtype=jnp.float32),
            rewards=jnp.zeros((max_size,), dtype=jnp.float32),
            next_obs=jnp.zeros((max_size, obs_dim), dtype=jnp.float32),
            dones=jnp.zeros((max_size,), dtype=jnp.float32),
            ptr=0,
            size=0,
            max_size=max_size
        )

# Functional Replay Buffer Addition
def add_batch(buffer, obs, actions, rewards, next_obs, dones):
    num_items = obs.shape[0]
    indices = (buffer.ptr + jnp.arange(num_items)) % buffer.max_size
    
    obs_new = buffer.obs.at[indices].set(obs)
    actions_new = buffer.actions.at[indices].set(actions)
    rewards_new = buffer.rewards.at[indices].set(rewards)
    next_obs_new = buffer.next_obs.at[indices].set(next_obs)
    dones_new = buffer.dones.at[indices].set(dones)
    
    ptr_new = (buffer.ptr + num_items) % buffer.max_size
    size_new = jnp.minimum(buffer.size + num_items, buffer.max_size)
    
    return buffer.replace(
        obs=obs_new,
        actions=actions_new,
        rewards=rewards_new,
        next_obs=next_obs_new,
        dones=dones_new,
        ptr=ptr_new,
        size=size_new
    )

def sample_batch(buffer, key, batch_size):
    idx = jax.random.randint(key, shape=(batch_size,), minval=0, maxval=buffer.size)
    return {
        "obs": buffer.obs[idx],
        "actions": buffer.actions[idx],
        "rewards": buffer.rewards[idx],
        "next_obs": buffer.next_obs[idx],
        "dones": buffer.dones[idx],
    }

# =====================================================================
# TD3 JIT COMPILATED UPDATE STEP
# =====================================================================

@jax.jit
def update_step(
    actor_state, critic_state, target_actor_params, target_critic_params,
    batch, step, policy_noise, noise_clip, policy_delay, gamma, tau
):
    # Split batch random key for target action noise
    key_noise, key_batch = jax.random.split(batch["rng"])
    
    # 1. Update Critic (Twin Critics)
    def critic_loss_fn(params):
        # Predict target next actions
        next_actions = Actor().apply({"params": target_actor_params}, batch["next_obs"])
        # Target action noise regularization
        noise = jax.random.normal(key_noise, shape=next_actions.shape) * policy_noise
        noise = jnp.clip(noise, -noise_clip, noise_clip)
        next_actions = jnp.clip(next_actions + noise, jnp.array([0.0, -1.0, -1.0]), jnp.array([1.0, 1.0, 1.0]))
        
        # Twin target Q-values
        target_q1, target_q2 = TwinCritic().apply({"params": target_critic_params}, batch["next_obs"], next_actions)
        target_q = jnp.minimum(target_q1, target_q2).squeeze()
        y = batch["rewards"] + (1.0 - batch["dones"]) * gamma * target_q
        
        # Current predictions
        q1, q2 = TwinCritic().apply({"params": params}, batch["obs"], batch["actions"])
        loss = jnp.mean(jnp.square(q1.squeeze() - y)) + jnp.mean(jnp.square(q2.squeeze() - y))
        return loss

    grad_fn_critic = jax.value_and_grad(critic_loss_fn)
    critic_loss, grads_critic = grad_fn_critic(critic_state.params)
    critic_state = critic_state.apply_gradients(grads=grads_critic)
    
    # 2. Delayed Actor & Target Parameters updates
    def actor_loss_fn(params):
        actions = Actor().apply({"params": params}, batch["obs"])
        q1, _ = TwinCritic().apply({"params": critic_state.params}, batch["obs"], actions)
        return -jnp.mean(q1)

    update_actor = (step % policy_delay == 0)
    
    def update_actor_fn():
        grad_fn_actor = jax.value_and_grad(actor_loss_fn)
        actor_loss, grads_actor = grad_fn_actor(actor_state.params)
        new_actor_state = actor_state.apply_gradients(grads=grads_actor)
        
        # Soft updates target actor parameters: theta_target = tau * theta + (1-tau) * theta_target
        new_target_actor_params = jax.tree_util.tree_map(
            lambda p, tp: p * tau + tp * (1.0 - tau),
            new_actor_state.params, target_actor_params
        )
        # Soft updates target critic parameters
        new_target_critic_params = jax.tree_util.tree_map(
            lambda p, tp: p * tau + tp * (1.0 - tau),
            critic_state.params, target_critic_params
        )
        return new_actor_state, new_target_actor_params, new_target_critic_params, actor_loss

    def no_update_fn():
        return actor_state, target_actor_params, target_critic_params, jnp.float32(0.0)

    actor_state, target_actor_params, target_critic_params, actor_loss = jax.lax.cond(
        update_actor,
        update_actor_fn,
        no_update_fn
    )
    
    return actor_state, critic_state, target_actor_params, target_critic_params, critic_loss, actor_loss

# =====================================================================
# JIT COMPILATION OF FULL ROLLOUT LOOP (THE KEY TO GPU SPEED)
# =====================================================================

def make_train_epoch(
    env, v_reset, v_step, num_envs, batch_size, learning_starts,
    policy_noise, noise_clip, policy_delay, gamma, tau, epoch_steps=100
):
    @jax.jit
    def train_epoch(carry, rng_epoch):
        state_batch, buffer, actor_state, critic_state, target_actor_params, target_critic_params, steps = carry
        
        def scan_step(scan_carry, _):
            state_curr, buf, act_st, crit_st, t_act, t_crit, steps_curr, key = scan_carry
            
            # Split random keys
            key, key_action, key_noise, key_reset, key_sample = jax.random.split(key, 5)
            
            # 1. Select actions for all parallel envs
            actions = Actor().apply({"params": act_st.params}, state_curr.obs)
            noise = jax.random.normal(key_action, shape=actions.shape) * 0.1
            actions_noisy = jnp.clip(actions + noise, jnp.array([0.0, -1.0, -1.0]), jnp.array([1.0, 1.0, 1.0]))
            
            # 2. Step all parallel envs
            next_obs, next_state, reward, done, _ = v_step(state_curr, actions_noisy)
            
            # 3. Add batch transitions to Replay Buffer
            buf = add_batch(buf, state_curr.obs, actions_noisy, reward, next_obs, done)
            
            # 4. Reset finished envs
            rng_res = jax.random.split(key_reset, num_envs)
            new_obs, new_state = v_reset(rng_res)
            
            obs_next = jnp.where(done[:, None], new_obs, next_obs)
            state_next = jax.tree_util.tree_map(
                lambda s, ns: jnp.where(done.reshape((done.shape[0],) + (1,) * (s.ndim - 1)), ns, s),
                next_state, new_state
            )
            state_next = state_next.replace(obs=obs_next)
            
            # 5. TD3 Updates (conditionally)
            carry_update = (act_st, crit_st, t_act, t_crit, buf, steps_curr, key_sample)
            
            def update_fn(c_up):
                a_st, c_st, ta, tc, b, s_val, r_key = c_up
                batch = sample_batch(b, r_key, batch_size)
                batch["rng"] = r_key
                return update_step(
                    a_st, c_st, ta, tc, batch, s_val,
                    policy_noise, noise_clip, policy_delay, gamma, tau
                )
                
            def no_update_fn(c_up):
                a_st, c_st, ta, tc, _, _, _ = c_up
                return a_st, c_st, ta, tc, jnp.float32(0.0), jnp.float32(0.0)
                
            act_st, crit_st, t_act, t_crit, c_loss, a_loss = jax.lax.cond(
                buf.size > learning_starts,
                update_fn,
                no_update_fn,
                carry_update
            )
            
            steps_curr = steps_curr + num_envs
            
            new_scan_carry = (state_next, buf, act_st, crit_st, t_act, t_crit, steps_curr, key)
            
            step_output = {
                "reward": reward,
                "done": done
            }
            return new_scan_carry, step_output
            
        init_scan_carry = (state_batch, buffer, actor_state, critic_state, target_actor_params, target_critic_params, steps, rng_epoch)
        final_scan_carry, outputs = jax.lax.scan(scan_step, init_scan_carry, None, length=epoch_steps)
        
        state_batch_f, buffer_f, actor_state_f, critic_state_f, target_actor_params_f, target_critic_params_f, steps_f, _ = final_scan_carry
        new_carry = (state_batch_f, buffer_f, actor_state_f, critic_state_f, target_actor_params_f, target_critic_params_f, steps_f)
        
        return new_carry, outputs

    return train_epoch

# =====================================================================
# MAIN VECTORIZED ROLLOUT AND TRAINING
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="Train JAX/MJX TD3 Agent for Rocket Stabilization")
    parser.add_argument("--timesteps", type=int, default=300_000, help="Total training steps")
    parser.add_argument("--num-envs", type=int, default=64, help="Number of parallel environments (GPU vectorization)")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--buffer-size", type=int, default=300_000, help="Replay buffer size")
    parser.add_argument("--learning-starts", type=int, default=5000, help="Timesteps before learning starts")
    parser.add_argument("--output-dir", default="runs_jax", help="Directory to save runs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Create output directory
    run_dir = Path(args.output_dir) / f"td3_gimbal_jax_envs_{args.num_envs}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving run results to {run_dir}")
    
    # Initialize Environment
    env = GimbalJaxEnv()
    
    # JAX keys
    rng = jax.random.PRNGKey(args.seed)
    rng_init, rng_train = jax.random.split(rng, 2)
    
    # Network Initialization Inputs Shape: (batch, obs_dim)
    obs_dim = len(OBSERVATION_NAMES)
    action_dim = 3
    dummy_obs = jnp.zeros((1, obs_dim))
    dummy_action = jnp.zeros((1, action_dim))
    
    # 1. Initialize Actor network
    actor = Actor()
    rng_act_init, rng_train = jax.random.split(rng_train)
    actor_params = actor.init(rng_act_init, dummy_obs)["params"]
    actor_state = TrainState.create(
        apply_fn=actor.apply,
        params=actor_params,
        tx=optax.adam(args.learning_rate)
    )
    target_actor_params = jax.tree_util.tree_map(lambda p: jnp.copy(p), actor_params)
    
    # 2. Initialize Twin Critic network
    critic = TwinCritic()
    rng_crit_init, rng_train = jax.random.split(rng_train)
    critic_params = critic.init(rng_crit_init, dummy_obs, dummy_action)["params"]
    critic_state = TrainState.create(
        apply_fn=critic.apply,
        params=critic_params,
        tx=optax.adam(args.learning_rate)
    )
    target_critic_params = jax.tree_util.tree_map(lambda p: jnp.copy(p), critic_params)
    
    # 3. Create JAX Replay Buffer
    buffer = ReplayBuffer.create(args.buffer_size, obs_dim, action_dim)
    
    # 4. Vectorized Environment Setup (reset and step)
    v_reset = jax.vmap(env.reset)
    v_step = jax.vmap(env.step)
    
    # Reset parallel envs
    rng_envs = jax.random.split(rng_init, args.num_envs)
    obs_batch, state_batch = v_reset(rng_envs)
    
    # 5. Compiled Epoch Training Function
    epoch_steps = 100
    train_epoch_fn = make_train_epoch(
        env, v_reset, v_step, args.num_envs, args.batch_size, args.learning_starts,
        policy_noise=0.2, noise_clip=0.5, policy_delay=2, gamma=args.gamma, tau=0.005,
        epoch_steps=epoch_steps
    )
    
    carry = (state_batch, buffer, actor_state, critic_state, target_actor_params, target_critic_params, jnp.int32(0))

    # Parameters
    steps = 0
    episodes_finished = 0
    last_print_steps = 0
    t0 = time.time()
    
    # Metrics logging
    episode_rewards = []
    env_rewards = np.zeros(args.num_envs)
    
    print("Starting training with JIT compiled scan loops...")
    
    while steps < args.timesteps:
        rng_train, rng_epoch = jax.random.split(rng_train)
        
        # 1. Execute JIT compiled scan epoch (100 steps of physics & learning on device)
        carry, outputs = train_epoch_fn(carry, rng_epoch)
        
        # 2. Extract step rewards/dones to CPU to update log statistics
        rewards_np = np.array(outputs["reward"])  # shape (100, num_envs)
        dones_np = np.array(outputs["done"])      # shape (100, num_envs)
        
        # Track finished episodes
        for s_idx in range(epoch_steps):
            env_rewards += rewards_np[s_idx]
            for i in range(args.num_envs):
                if dones_np[s_idx, i] > 0.5:
                    episode_rewards.append(env_rewards[i])
                    env_rewards[i] = 0.0
                    episodes_finished += 1
                    
        steps += args.num_envs * epoch_steps
        
        # Log status
        if steps - last_print_steps >= 10000:
            elapsed = time.time() - t0
            steps_per_sec = steps / elapsed
            mean_r = np.mean(episode_rewards[-20:]) if len(episode_rewards) > 0 else 0.0
            print(f"Step: {steps} / {args.timesteps} | Finished Eps: {episodes_finished} | SPS: {steps_per_sec:.1f} | Last 20 Mean Reward: {mean_r:.2f} | Time: {elapsed:.1f}s")
            last_print_steps = steps
            
            # Save periodic checkpoint of params
            actor_state_val = carry[2]
            critic_state_val = carry[3]
            if steps % 50000 == 0:
                checkpoint_path = run_dir / f"checkpoint_params_step_{steps}.npz"
                np.savez(
                    checkpoint_path,
                    actor_params=actor_state_val.params,
                    critic_params=critic_state_val.params
                )
                print(f"Saved checkpoint: {checkpoint_path}")
                
    # Save final model parameters
    actor_state_val = carry[2]
    critic_state_val = carry[3]
    final_path = run_dir / "latest_params.npz"
    np.savez(
        final_path,
        actor_params=actor_state_val.params,
        critic_params=critic_state_val.params
    )
    print(f"Training completed. Final params saved to {final_path}")

if __name__ == "__main__":
    main()
