import argparse
import time
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
import mujoco
import mujoco.viewer

# Define Flax Actor (Must match train_gimbal_jax.py)
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

# Helper function for quaternion creation
def quat_from_axis_angle(axis, angle):
    axis = np.asarray(axis)
    norm = np.linalg.norm(axis)
    if norm > 1e-8:
        axis = axis / norm
    half_angle = 0.5 * angle
    return np.array([
        np.cos(half_angle),
        *(axis * np.sin(half_angle)),
    ])

def main():
    parser = argparse.ArgumentParser(description="Visualize JAX Gimbal Rocket Stabilization")
    parser.add_argument("--model", type=str, default="runs_jax/td3_gimbal_jax_envs_512/latest_params.npz", 
                        help="Path to trained parameters .npz")
    parser.add_argument("--continuous", action="store_true", help="Continuously teleport/tilt rocket every 5s of stabilization")
    parser.add_argument("--min-tilt", type=float, default=15.0, help="Min initial tilt (deg)")
    parser.add_argument("--max-tilt", type=float, default=35.0, help="Max initial tilt (deg)")
    parser.add_argument("--gimbal-limit", type=float, default=45.0, help="Mechanical gimbal fail limit (deg)")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes to run if not continuous")
    
    args = parser.parse_args()

    # 1. Load MuJoCo Model and Data
    model = mujoco.MjModel.from_xml_path("gimbal_default.xml")
    data = mujoco.MjData(model)

    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hopper")
    yaw_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "tvc_yaw_joint")
    pitch_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "tvc_pitch_joint")
    
    yaw_qpos_id = int(model.jnt_qposadr[yaw_joint_id])
    pitch_qpos_id = int(model.jnt_qposadr[pitch_joint_id])
    yaw_qvel_id = int(model.jnt_dofadr[yaw_joint_id])
    pitch_qvel_id = int(model.jnt_dofadr[pitch_joint_id])
    
    thrust_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "thrust_site")

    # Parameters
    max_thrust = 45.0
    max_tvc_angle = np.deg2rad(20.0)
    tvc_servo_speed = np.deg2rad(60.0) / 0.13
    frame_skip = 10
    angular_damping = 0.15
    target_pos = np.array([0.0, 0.0, 1.0])

    print(f"Loading parameters from {args.model}...")
    try:
        loaded_data = np.load(args.model, allow_pickle=True)
        actor_params = loaded_data["actor_params"].item()
        print("Parameters loaded successfully!")
    except Exception as e:
        print(f"Error loading parameters: {e}")
        print("Running with random/zero actions instead.")
        actor_params = None

    actor = Actor()

    # Launch passive viewer
    print("Launching MuJoCo viewer...")
    viewer = mujoco.viewer.launch_passive(model, data)

    # Passive viewer Camera positioning
    viewer.cam.lookat[:] = [0.0, 0.0, 1.3]
    viewer.cam.distance = 4.0
    viewer.cam.azimuth = 135.0
    viewer.cam.elevation = -18.0

    def apply_random_tilt(data):
        phi = np.random.uniform(0.0, 2.0 * np.pi)
        axis = np.array([np.cos(phi), np.sin(phi), 0.0])
        tilt = np.random.uniform(np.deg2rad(args.min_tilt), np.deg2rad(args.max_tilt))
        quat = quat_from_axis_angle(axis, tilt)
        
        data.qpos[0:4] = quat
        data.qpos[4:6] = 0.0
        data.qvel[0:3] = np.random.uniform(-0.3, 0.3, size=3)
        data.qvel[3:5] = 0.0
        mujoco.mj_forward(model, data)

    def get_observation(data, main_power):
        pos = data.xpos[body_id]
        rot = data.xmat[body_id].reshape(3, 3)
        lin_vel = data.cvel[body_id][3:6]
        ang_vel = data.cvel[body_id][0:3]
        
        bottom_point = pos + rot @ np.array([0.0, 0.0, -0.355])
        target_delta = target_pos - bottom_point
        
        tvc_yaw = data.qpos[yaw_qpos_id]
        tvc_pitch = data.qpos[pitch_qpos_id]
        tvc_yaw_vel = data.qvel[yaw_qvel_id]
        tvc_pitch_vel = data.qvel[pitch_qvel_id]
        
        # Simple foot height contact check (4 feet)
        feet_local = np.array([
            [0.4, 0.0, -0.355],
            [-0.4, 0.0, -0.355],
            [0.0, 0.4, -0.355],
            [0.0, -0.4, -0.355]
        ])
        feet_pos = pos + feet_local @ rot.T
        contacts = np.where(feet_pos[:, 2] < 0.025, 1.0, 0.0)
        
        obs = np.concatenate([
            target_delta,
            data.qpos[0:4],
            [bottom_point[2]],
            lin_vel,
            ang_vel,
            [tvc_yaw, tvc_pitch, tvc_yaw_vel, tvc_pitch_vel],
            contacts,
            [main_power]
        ])
        return obs

    try:
        episode = 0
        while viewer.is_running() and (args.continuous or episode < args.episodes):
            episode += 1
            print(f"\n--- Starting Episode {episode} ---")
            mujoco.mj_resetData(model, data)
            apply_random_tilt(data)
            
            step = 0
            hover_timer = 0.0
            tvc_yaw_cmd = 0.0
            tvc_pitch_cmd = 0.0
            main_power = 0.0
            has_printed_success = False
            
            timestep = model.opt.timestep
            
            while viewer.is_running():
                step += 1
                obs = get_observation(data, main_power)
                
                if actor_params is not None:
                    # Flax inference
                    obs_jnp = jnp.array(obs, dtype=jnp.float32)
                    action_jnp = actor.apply({"params": actor_params}, obs_jnp)
                    action = np.array(action_jnp)
                else:
                    action = np.array([0.45, 0.0, 0.0], dtype=np.float32)
                
                main_power = action[0]
                yaw_ctrl = max_tvc_angle * action[1]
                pitch_ctrl = max_tvc_angle * action[2]
                
                # Step physics
                max_delta = tvc_servo_speed * timestep
                for _ in range(frame_skip):
                    # Rate limit servo commands
                    tvc_yaw_cmd += np.clip(yaw_ctrl - tvc_yaw_cmd, -max_delta, max_delta)
                    tvc_pitch_cmd += np.clip(pitch_ctrl - tvc_pitch_cmd, -max_delta, max_delta)
                    
                    data.ctrl[0] = np.rad2deg(tvc_yaw_cmd)
                    data.ctrl[1] = np.rad2deg(tvc_pitch_cmd)
                    mujoco.mj_forward(model, data)
                    
                    site_pos = data.site_xpos[thrust_site_id]
                    site_xmat = data.site_xmat[thrust_site_id].reshape(3, 3)
                    thrust_dir = site_xmat @ [0.0, 0.0, 1.0]
                    thrust_force = thrust_dir * (main_power * max_thrust)
                    
                    # Apply thrust torque & damping torque
                    torque = np.cross(site_pos - data.xpos[body_id], thrust_force) - angular_damping * data.cvel[body_id][0:3]
                    data.qfrc_applied[0:3] = torque
                    
                    mujoco.mj_step(model, data)
                
                # Check status metrics
                pos = data.xpos[body_id]
                rot = data.xmat[body_id].reshape(3, 3)
                lin_vel = data.cvel[body_id][3:6]
                ang_vel = data.cvel[body_id][0:3]
                
                body_axis = rot @ [0.0, 0.0, 1.0]
                bottom_point = pos + rot @ [0.0, 0.0, -0.355]
                upright_score = np.dot(body_axis, [0.0, 0.0, 1.0])
                rel_dist = np.linalg.norm(bottom_point[:2] - target_pos[:2])
                
                # Fail conditions
                tilt_angle_deg = np.rad2deg(np.arccos(np.clip(upright_score, -1.0, 1.0)))
                fail_gimbal = tilt_angle_deg > args.gimbal_limit
                
                feet_pos = pos + np.array([[0.4, 0.0, -0.355], [-0.4, 0.0, -0.355], [0.0, 0.4, -0.355], [0.0, -0.4, -0.355]]) @ rot.T
                fail_contact = np.any(feet_pos[:, 2] < 0.025)
                
                fail = fail_gimbal | fail_contact
                
                # Hover timer tracking
                is_stable = (np.abs(bottom_point[2] - target_pos[2]) < 1.0) & (upright_score > 0.90) & (np.linalg.norm(lin_vel) < 2.0) & (np.linalg.norm(ang_vel) < 2.0)
                if is_stable:
                    hover_timer += timestep * frame_skip
                else:
                    hover_timer = 0.0
                
                success = hover_timer >= 5.0
                
                # Synced sleep
                viewer.sync()
                time.sleep(timestep * frame_skip)
                
                if fail:
                    reasons = []
                    if fail_gimbal:
                        reasons.append(f"Gimbal Limit Exceeded (tilt: {tilt_angle_deg:.1f}° > limit: {args.gimbal_limit}°)")
                    if fail_contact:
                        reasons.append(f"Ground Contact (feet heights: {feet_pos[:, 2]})")
                    print(f"Episode {episode} FAILED! Reasons: {', '.join(reasons)}")
                    break
                    
                if success:
                    if not has_printed_success:
                        print(f"Episode {episode} STABLE FOR 5 SECONDS!")
                        has_printed_success = True
                    if args.continuous:
                        print("Continuous mode active: applying random tilt disturbance...")
                        apply_random_tilt(data)
                        hover_timer = 0.0
                        has_printed_success = False
                
                if step >= 1500 and not args.continuous:
                    print(f"Episode {episode} COMPLETED (1500 steps/15s)!")
                    break
                        
    except KeyboardInterrupt:
        print("\nVisualization stopped by user.")
    finally:
        print("Closing viewer...")
        viewer.close()

if __name__ == "__main__":
    main()
