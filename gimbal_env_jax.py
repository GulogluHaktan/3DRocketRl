import jax
import jax.numpy as jnp
from flax import struct
import mujoco
from mujoco import mjx

# Default constants from V2 gimbal_env.py
OBSERVATION_NAMES = [
    "bottom_target_delta_x", "bottom_target_delta_y", "bottom_target_delta_z",
    "body_quat_w", "body_quat_x", "body_quat_y", "body_quat_z",
    "bottom_height",
    "linear_velocity_x", "linear_velocity_y", "linear_velocity_z",
    "angular_velocity_x", "angular_velocity_y", "angular_velocity_z",
    "tvc_yaw_angle", "tvc_pitch_angle",
    "tvc_yaw_velocity", "tvc_pitch_velocity",
    "foot1_contact", "foot2_contact", "foot3_contact", "foot4_contact",
    "main_motor_power"
]

@struct.dataclass
class EnvState:
    pipeline_state: mjx.Data
    obs: jnp.ndarray
    reward: jnp.float32
    done: jnp.float32
    step_count: jnp.int32
    hover_timer: jnp.float32
    current_phase: jnp.int32  # 0: recovery, 1: hover, 2: done
    tvc_yaw_cmd: jnp.float32
    tvc_pitch_cmd: jnp.float32
    last_action: jnp.ndarray  # shape (3,)
    last_vertical_velocity: jnp.float32
    last_height: jnp.float32
    last_rel_dist: jnp.float32
    fail_reason: jnp.int32    # 0: none, 1: gimbal_limit, 2: ground_contact, 3: speed_limit, 4: target_escape
    rng: jax.Array

class GimbalJaxEnv:
    def __init__(self, xml_path="gimbal_default.xml", min_tilt_deg=15.0, max_tilt_deg=35.0):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.mjx_model = mjx.put_model(self.model)
        
        self.min_tilt_deg = min_tilt_deg
        self.max_tilt_deg = max_tilt_deg
        self.max_gimbal_limit_deg = 45.0
        self.max_thrust = 45.0
        self.max_tvc_angle = jnp.deg2rad(20.0)
        self.tvc_servo_speed = jnp.deg2rad(60.0) / 0.13  # 0.13s per 60 deg -> ~8.05 rad/s
        self.frame_skip = 10
        self.max_steps = 1500
        
        self.linear_damping = 0.15
        self.angular_damping = 0.15
        
        self.target_pos = jnp.array([0.0, 0.0, 1.0], dtype=jnp.float32)
        
        # Geoms and joints lookup
        self.body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "hopper")
        self.yaw_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "tvc_yaw_joint")
        self.pitch_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "tvc_pitch_joint")
        
        self.yaw_qpos_id = int(self.model.jnt_qposadr[self.yaw_joint_id]) # 4
        self.pitch_qpos_id = int(self.model.jnt_qposadr[self.pitch_joint_id]) # 5
        self.yaw_qvel_id = int(self.model.jnt_dofadr[self.yaw_joint_id]) # 3
        self.pitch_qvel_id = int(self.model.jnt_dofadr[self.pitch_joint_id]) # 4
        
        self.thrust_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "thrust_site")
        
        # Default V2 reward weights
        self.reward_weights = {
            "failure_penalty": 20000.0,
            "success_bonus": 2500.0,
            "recovery_linear_speed": 2.4,
            "recovery_angular_speed": 2.8,
            "recovery_joint_speed": 0.15,
            "recovery_upright": 45.0,
            "recovery_low_altitude": 26.0,
            "recovery_upward_speed": 30.0,
            "recovery_downward_speed": 22.0,
            "recovery_falling_thrust": 85.0,
            "recovery_thrust_alignment": 38.0,
            "recovery_height_error": 14.0,
            "recovery_altitude_progress": 260.0,
            "recovery_below_hover_band": 55.0,
            "recovery_ground_stall": 180.0,
            "recovery_upright_climb": 35.0,
            "recovery_rel_dist": 0.0,
            "hover_linear_speed": 3.0,
            "hover_angular_speed": 6.0,
            "hover_joint_speed": 1.0,
            "hover_upright": 45.0,
            "hover_height_error": 30.0,
            "hover_vertical_speed": 15.0,
            "hover_falling_thrust": 80.0,
            "hover_low_altitude": 26.0,
            "hover_upward_speed": 28.0,
            "phase_recovery_to_hover_bonus": 1400.0,
            "time_penalty": 0.0
        }

    def _quat_from_axis_angle(self, axis, angle):
        axis_norm = jnp.linalg.norm(axis)
        axis = jnp.where(axis_norm > 1e-8, axis / axis_norm, axis)
        half_angle = 0.5 * angle
        w = jnp.cos(half_angle)
        xyz = axis * jnp.sin(half_angle)
        return jnp.array([w, xyz[0], xyz[1], xyz[2]], dtype=jnp.float32)

    def reset(self, rng):
        # 1. Initialize MJX Data
        data = mjx.make_data(self.mjx_model)
        
        # 2. Apply random initial tilt
        rng, rng_phi, rng_tilt, rng_w = jax.random.split(rng, 4)
        phi = jax.random.uniform(rng_phi, minval=0.0, maxval=2.0 * jnp.pi)
        axis = jnp.array([jnp.cos(phi), jnp.sin(phi), 0.0], dtype=jnp.float32)
        
        min_tilt_rad = jnp.deg2rad(self.min_tilt_deg)
        max_tilt_rad = jnp.deg2rad(self.max_tilt_deg)
        tilt = jax.random.uniform(rng_tilt, minval=min_tilt_rad, maxval=max_tilt_rad)
        
        quat = self._quat_from_axis_angle(axis, tilt)
        
        # Small angular velocity disturbance
        w_disturbance = jax.random.uniform(rng_w, shape=(3,), minval=-0.3, maxval=0.3)
        
        # Set positions and velocities
        qpos = data.qpos.at[0:4].set(quat)
        qpos = qpos.at[4:6].set(0.0) # TVC joints at 0
        
        qvel = data.qvel.at[0:3].set(w_disturbance)
        qvel = qvel.at[3:5].set(0.0) # TVC velocities at 0
        
        data = data.replace(qpos=qpos, qvel=qvel)
        data = mjx.forward(self.mjx_model, data)
        
        # 3. Compute initial metrics
        pos = data.xpos[self.body_id]
        rot = data.xmat[self.body_id].reshape(3, 3)
        bottom_point = pos + rot @ jnp.array([0.0, 0.0, -0.355], dtype=jnp.float32)
        relative_pos = bottom_point - self.target_pos
        rel_dist = jnp.linalg.norm(relative_pos[:2])
        lin_vel = data.cvel[self.body_id][3:6]
        
        # 4. Construct observation
        obs = self._get_obs(data, 0.0)
        
        state = EnvState(
            pipeline_state=data,
            obs=obs,
            reward=jnp.float32(0.0),
            done=jnp.float32(0.0),
            step_count=0,
            hover_timer=0.0,
            current_phase=0, # 0: recovery
            tvc_yaw_cmd=jnp.float32(0.0),
            tvc_pitch_cmd=jnp.float32(0.0),
            last_action=jnp.zeros(3, dtype=jnp.float32),
            last_vertical_velocity=lin_vel[2],
            last_height=bottom_point[2],
            last_rel_dist=rel_dist,
            fail_reason=0,
            rng=rng
        )
        
        return obs, state

    def _get_obs(self, data, main_motor_power):
        pos = data.xpos[self.body_id]
        rot = data.xmat[self.body_id].reshape(3, 3)
        lin_vel = data.cvel[self.body_id][3:6]
        ang_vel = data.cvel[self.body_id][0:3]
        
        bottom_point = pos + rot @ jnp.array([0.0, 0.0, -0.355], dtype=jnp.float32)
        target_delta = self.target_pos - bottom_point
        
        tvc_yaw = data.qpos[self.yaw_qpos_id]
        tvc_pitch = data.qpos[self.pitch_qpos_id]
        tvc_yaw_vel = data.qvel[self.yaw_qvel_id]
        tvc_pitch_vel = data.qvel[self.pitch_qvel_id]
        
        # Simple foot height contact check (4 feet)
        # Foot local positions
        feet_local = jnp.array([
            [0.4, 0.0, -0.355],
            [-0.4, 0.0, -0.355],
            [0.0, 0.4, -0.355],
            [0.0, -0.4, -0.355]
        ], dtype=jnp.float32)
        
        # Foot positions in world coordinates
        feet_pos = pos + feet_local @ rot.T
        contacts = jnp.where(feet_pos[:, 2] < 0.025, 1.0, 0.0)
        
        obs = jnp.concatenate([
            target_delta,
            data.qpos[0:4], # body quat
            jnp.array([bottom_point[2]], dtype=jnp.float32),
            lin_vel,
            ang_vel,
            jnp.array([tvc_yaw, tvc_pitch, tvc_yaw_vel, tvc_pitch_vel], dtype=jnp.float32),
            contacts,
            jnp.array([main_motor_power], dtype=jnp.float32)
        ])
        return obs

    def step(self, state, action):
        # Action clip
        action = jnp.clip(action, jnp.array([0.0, -1.0, -1.0]), jnp.array([1.0, 1.0, 1.0]))
        main = action[0]
        yaw_ctrl = self.max_tvc_angle * action[1]
        pitch_ctrl = self.max_tvc_angle * action[2]
        
        timestep = self.mjx_model.opt.timestep
        max_delta = self.tvc_servo_speed * timestep
        
        # Multi-step simulation loop inside JAX
        def sim_loop_fn(i, val_tuple):
            data_val, yaw_cmd_val, pitch_cmd_val = val_tuple
            
            # Rate limit commands
            yaw_cmd_val += jnp.clip(yaw_ctrl - yaw_cmd_val, -max_delta, max_delta)
            pitch_cmd_val += jnp.clip(pitch_ctrl - pitch_cmd_val, -max_delta, max_delta)
            
            # Position actuators target in degrees
            ctrl = data_val.ctrl.at[0].set(jnp.rad2deg(yaw_cmd_val))
            ctrl = ctrl.at[1].set(jnp.rad2deg(pitch_cmd_val))
            data_val = data_val.replace(ctrl=ctrl)
            data_val = mjx.forward(self.mjx_model, data_val)
            
            # Apply thrust forces and moments
            site_pos = data_val.site_xpos[self.thrust_site_id]
            site_xmat = data_val.site_xmat[self.thrust_site_id].reshape(3, 3)
            thrust_dir = site_xmat @ jnp.array([0.0, 0.0, 1.0])
            thrust_force = thrust_dir * (main * self.max_thrust)
            
            body_pos = data_val.xpos[self.body_id]
            ang_vel_val = data_val.cvel[self.body_id][0:3]
            
            # Apply torque and damping torque
            torque = jnp.cross(site_pos - body_pos, thrust_force) - self.angular_damping * ang_vel_val
            
            # Damping on linear velocity (simulate simple air drag)
            lin_vel_val = data_val.cvel[self.body_id][3:6]
            # Since linear motion is physically constrained by ball joint, linear damping is not strictly necessary
            # but we replicate the force application
            lin_damping_force = -self.linear_damping * lin_vel_val
            
            qfrc_applied = data_val.qfrc_applied.at[0:3].set(torque)
            data_val = data_val.replace(qfrc_applied=qfrc_applied)
            
            data_val = mjx.step(self.mjx_model, data_val)
            return data_val, yaw_cmd_val, pitch_cmd_val

        init_val = (state.pipeline_state, state.tvc_yaw_cmd, state.tvc_pitch_cmd)
        data, tvc_yaw_cmd, tvc_pitch_cmd = jax.lax.fori_loop(0, self.frame_skip, sim_loop_fn, init_val)
        
        # Compute metrics
        pos = data.xpos[self.body_id]
        rot = data.xmat[self.body_id].reshape(3, 3)
        lin_vel = data.cvel[self.body_id][3:6]
        ang_vel = data.cvel[self.body_id][0:3]
        
        body_axis = rot @ jnp.array([0.0, 0.0, 1.0], dtype=jnp.float32)
        bottom_point = pos + rot @ jnp.array([0.0, 0.0, -0.355], dtype=jnp.float32)
        relative_pos = bottom_point - self.target_pos
        rel_dist = jnp.linalg.norm(relative_pos[:2])
        upright_score = jnp.dot(body_axis, jnp.array([0.0, 0.0, 1.0], dtype=jnp.float32))
        
        # Joint speed
        joint_speed = jnp.linalg.norm(jnp.array([data.qvel[self.yaw_qvel_id], data.qvel[self.pitch_qvel_id]]))
        
        # Check hover entry (target vertical band)
        is_height_ok = jnp.abs(bottom_point[2] - self.target_pos[2]) < 6.0
        is_upright_ok = upright_score > 0.95
        transition_to_hover = (state.current_phase == 0) & is_height_ok & is_upright_ok
        
        # Update current phase and hover timer
        next_phase = jnp.where(transition_to_hover, 1, state.current_phase)
        dt = timestep * self.frame_skip
        
        # Hover timer increments if we are in hover phase and stable
        is_hover_stable = (next_phase >= 1) & (jnp.abs(bottom_point[2] - self.target_pos[2]) < 1.0) & (upright_score > 0.90) & (jnp.linalg.norm(lin_vel) < 2.0) & (jnp.linalg.norm(ang_vel) < 2.0)
        next_hover_timer = jnp.where(is_hover_stable, state.hover_timer + dt, 0.0)
        
        # Success if stable for 5s
        success = next_hover_timer >= 5.0
        next_phase = jnp.where(success, 2, next_phase) # 2: success (keeps running to maintain)
        
        # Check fail conditions
        # 1. Gimbal limit exceeded (45 degrees)
        tilt_angle_deg = jnp.rad2deg(jnp.arccos(jnp.clip(upright_score, -1.0, 1.0)))
        fail_gimbal = tilt_angle_deg > self.max_gimbal_limit_deg
        
        # 2. Foot touch ground
        feet_local = jnp.array([
            [0.4, 0.0, -0.355],
            [-0.4, 0.0, -0.355],
            [0.0, 0.4, -0.355],
            [0.0, -0.4, -0.355]
        ], dtype=jnp.float32)
        feet_pos = pos + feet_local @ rot.T
        fail_contact = jnp.any(feet_pos[:, 2] < 0.025)
        
        # 3. Speed limit exceeded in hover
        fail_speed = (next_phase == 1) & ((jnp.linalg.norm(lin_vel) > 70.0) | (jnp.linalg.norm(ang_vel) > 70.0))
        
        fail = fail_gimbal | fail_contact | fail_speed
        
        # Fail reasons
        fail_reason = jnp.where(fail_gimbal, 1, 0)
        fail_reason = jnp.where(fail_contact, 2, fail_reason)
        fail_reason = jnp.where(fail_speed, 3, fail_reason)
        
        # Calculate Reward (Legacy reward from V2 environment)
        rw = self.reward_weights
        
        # Phase rewards
        # 1. Recovery reward
        reward_recovery = 0.0
        reward_recovery -= jnp.linalg.norm(lin_vel) * rw["recovery_linear_speed"]
        reward_recovery -= jnp.linalg.norm(ang_vel) * rw["recovery_angular_speed"]
        reward_recovery -= joint_speed * rw["recovery_joint_speed"]
        reward_recovery += rw["recovery_upright"] * upright_score
        reward_recovery -= ((1.0 - upright_score) ** 2) * 3000.0 # quadratic alignment
        reward_recovery -= rel_dist * rw["recovery_rel_dist"]
        reward_recovery -= jnp.abs(bottom_point[2] - self.target_pos[2]) * rw["recovery_height_error"]
        
        # Upward velocity / falling thrust
        vertical_velocity = lin_vel[2]
        upward_speed = jnp.maximum(vertical_velocity, 0.0)
        downward_speed = jnp.maximum(-vertical_velocity, 0.0)
        reward_recovery -= downward_speed * rw["recovery_downward_speed"]
        
        below_hover_target = bottom_point[2] < self.target_pos[2]
        reward_recovery += jnp.where(below_hover_target & (vertical_velocity < 0.0), main * rw["recovery_falling_thrust"], 0.0)
        reward_recovery += jnp.where(below_hover_target, upward_speed * rw["recovery_upward_speed"] + upward_speed * jnp.maximum(upright_score, 0.0) * rw["recovery_upright_climb"], 0.0)
        
        # 2. Hover reward
        reward_hover = 0.0
        reward_hover -= jnp.linalg.norm(lin_vel) * rw["hover_linear_speed"]
        reward_hover -= jnp.linalg.norm(ang_vel) * rw["hover_angular_speed"]
        reward_hover -= joint_speed * rw["hover_joint_speed"]
        reward_hover += rw["hover_upright"] * upright_score
        reward_hover -= ((1.0 - upright_score) ** 2) * 3000.0
        reward_hover -= jnp.abs(bottom_point[2] - self.target_pos[2]) * rw["hover_height_error"]
        reward_hover -= jnp.abs(vertical_velocity) * rw["hover_vertical_speed"]
        reward_hover += jnp.where(vertical_velocity < 0.0, main * rw["hover_falling_thrust"], 0.0)
        
        # Select active phase reward
        reward = jnp.where(next_phase >= 1, reward_hover, reward_recovery)
        
        # Action rate penalty: small in recovery (0.2) to allow quick setup, large in hover (2.0) to prevent jitter
        action_diff = action - state.last_action
        action_diff_weight = jnp.where(next_phase >= 1, 2.0, 0.2)
        reward -= jnp.sum(jnp.square(action_diff)) * action_diff_weight
        
        # Control effort penalty: small in recovery (0.5), large in hover (2.0) to center TVC nozzle at 0 when stable
        control_effort_weight = jnp.where(next_phase >= 1, 2.0, 0.5)
        reward -= jnp.sum(jnp.square(action[1:3])) * control_effort_weight
        
        # Transition bonus
        reward += jnp.where(transition_to_hover, rw["phase_recovery_to_hover_bonus"], 0.0)
        
        # Success bonus / failure penalty
        # Success bonus is given only once, when first transitioning to phase 2 (success)
        is_new_success = (state.current_phase < 2) & (next_phase == 2)
        reward = jnp.where(is_new_success, reward + rw["success_bonus"], reward)
        reward = jnp.where(fail, reward - rw["failure_penalty"], reward)
        
        # Done condition
        next_step_count = state.step_count + 1
        done = fail | (next_step_count >= self.max_steps)
        
        # Construct next observation
        obs = self._get_obs(data, main)
        
        next_state = EnvState(
            pipeline_state=data,
            obs=obs,
            reward=reward,
            done=jnp.where(done, 1.0, 0.0),
            step_count=next_step_count,
            hover_timer=next_hover_timer,
            current_phase=next_phase,
            tvc_yaw_cmd=tvc_yaw_cmd,
            tvc_pitch_cmd=tvc_pitch_cmd,
            last_action=action,
            last_vertical_velocity=lin_vel[2],
            last_height=bottom_point[2],
            last_rel_dist=rel_dist,
            fail_reason=fail_reason,
            rng=state.rng
        )
        
        return obs, next_state, reward, done, {}
