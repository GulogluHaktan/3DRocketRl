import jax
import jax.numpy as jnp
from gimbal_env_jax import GimbalJaxEnv

def main():
    print("Testing GimbalJaxEnv...")
    env = GimbalJaxEnv()
    
    # JAX PRNG key
    key = jax.random.PRNGKey(0)
    
    # Reset env
    print("Resetting environment...")
    obs, state = env.reset(key)
    print("Observation shape:", obs.shape)
    print("State qpos:", state.pipeline_state.qpos)
    print("State qvel:", state.pipeline_state.qvel)
    
    # Step env
    print("Stepping environment...")
    action = jnp.array([0.5, 0.1, -0.1], dtype=jnp.float32)
    next_obs, next_state, reward, done, info = env.step(state, action)
    print("Next observation shape:", next_obs.shape)
    print("Reward:", reward)
    print("Done:", done)
    print("Next State step_count:", next_state.step_count)
    
    # Let's test jax.jit compilation
    print("Testing JIT compilation of step function...")
    jit_step = jax.jit(env.step)
    # First call will compile
    next_obs_jit, next_state_jit, reward_jit, done_jit, _ = jit_step(state, action)
    print("JIT step completed successfully. Next State step_count:", next_state_jit.step_count)
    
    print("\nAll environment checks passed successfully!")

if __name__ == "__main__":
    main()
