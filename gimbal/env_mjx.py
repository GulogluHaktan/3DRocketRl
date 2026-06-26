from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Tuple

from brax.envs import base
from flax import struct
import jax
import jax.numpy as jp
import mujoco
from mujoco import mjx


@struct.dataclass
class GimbalStage:
    number: int
    max_start_tilt_deg: float
    max_start_rate_dps: float
    disturbances: int
    domain_randomization: bool


# JAX-friendly representation of stages
STAGE_1 = GimbalStage(1, 15.0, 10.0, 0, False)
STAGE_2 = GimbalStage(2, 30.0, 20.0, 1, False)
STAGE_3 = GimbalStage(3, 45.0, 30.0, 3, True)


@struct.dataclass
class GimbalMJXState:
    data: mjx.Data
    previous_action: jax.Array
    tvc_target: jax.Array
    tvc_rate: jax.Array
    tvc_rate_command: jax.Array
    step_count: jax.Array
    success_hold_time: jax.Array
    disturbances_applied: jax.Array
    disturbance_end_step: jax.Array
    disturbance_torque: jax.Array
    last_tilt_cost: jax.Array
    mass_scale: jax.Array
    inertia_scale: jax.Array
    thrust_scale: jax.Array
    servo_scale: jax.Array
    imu_noise_std: jax.Array
    sensor_delay_steps: jax.Array
    previous_obs: jax.Array



class GimbalMJXEnv(base.Env):
    """Pure-JAX Gimbal environment backed by MuJoCo MJX."""

    def __init__(
        self,
        stage: int = 1,
        control_rate: float = 50.0,
        fixed_thrust_power: float = 1.0,
        max_tvc_deg: float = 20.0,
        tvc_servo_sec_per_60deg: float = 0.13,
        max_tvc_rate_dps: float = 120.0,
        max_steps: int = 500,
        domain_randomization: bool | None = None,
    ):
        xml_path = Path(__file__).resolve().parent / "model.xml"
        self._host_model = mujoco.MjModel.from_xml_path(str(xml_path))
        
        # Optimize host model solver for MJX throughput
        self._host_model.opt.integrator = mujoco.mjtIntegrator.mjINT_EULER
        self._host_model.opt.solver = mujoco.mjtSolver.mjSOL_CG
        self._host_model.opt.iterations = 8
        self._host_model.opt.ls_iterations = 4
        self._host_model.opt.jacobian = mujoco.mjtJacobian.mjJAC_SPARSE
        
        self._model = mjx.put_model(self._host_model)
        
        self.stage_num = stage
        self.stage = {1: STAGE_1, 2: STAGE_2, 3: STAGE_3}[stage]
        self.control_rate = float(control_rate)
        self.control_dt = 1.0 / self.control_rate
        self.fixed_thrust_power = float(fixed_thrust_power)
        self.max_tvc_angle = jp.deg2rad(max_tvc_deg)
        self.base_servo_speed = jp.deg2rad(60.0) / tvc_servo_sec_per_60deg
        self.base_command_speed = jp.minimum(jp.deg2rad(max_tvc_rate_dps), self.base_servo_speed)
        self.angular_velocity_observation_scale = jp.deg2rad(45.0)
        self.tvc_velocity_observation_scale = jp.deg2rad(120.0)
        self.max_steps = int(max_steps)
        self.randomize_domain = (
            self.stage.domain_randomization
            if domain_randomization is None
            else bool(domain_randomization)
        )
        
        sim_dt = float(self._host_model.opt.timestep)
        self.frame_skip = int(round(self.control_dt / sim_dt))
        self.disturbance_steps = int(max(1, round(0.08 / sim_dt)))
        
        # Site and Joint indices
        self._rocket_id = mujoco.mj_name2id(self._host_model, mujoco.mjtObj.mjOBJ_BODY, "rocket")
        self._thrust_site_id = mujoco.mj_name2id(self._host_model, mujoco.mjtObj.mjOBJ_SITE, "thrust_site")
        self._roll_joint_id = mujoco.mj_name2id(self._host_model, mujoco.mjtObj.mjOBJ_JOINT, "gimbal_roll")
        self._pitch_joint_id = mujoco.mj_name2id(self._host_model, mujoco.mjtObj.mjOBJ_JOINT, "gimbal_pitch")
        self._tvc_x_joint_id = mujoco.mj_name2id(self._host_model, mujoco.mjtObj.mjOBJ_JOINT, "tvc_x")
        self._tvc_y_joint_id = mujoco.mj_name2id(self._host_model, mujoco.mjtObj.mjOBJ_JOINT, "tvc_y")
        
        self._roll_qpos = int(self._host_model.jnt_qposadr[self._roll_joint_id])
        self._pitch_qpos = int(self._host_model.jnt_qposadr[self._pitch_joint_id])
        self._roll_qvel = int(self._host_model.jnt_dofadr[self._roll_joint_id])
        self._pitch_qvel = int(self._host_model.jnt_dofadr[self._pitch_joint_id])
        self._tvc_x_qpos = int(self._host_model.jnt_qposadr[self._tvc_x_joint_id])
        self._tvc_y_qpos = int(self._host_model.jnt_qposadr[self._tvc_y_joint_id])
        self._tvc_x_qvel = int(self._host_model.jnt_dofadr[self._tvc_x_joint_id])
        self._tvc_y_qvel = int(self._host_model.jnt_dofadr[self._tvc_y_joint_id])
        
        self.base_mass = float(self._host_model.body_mass[self._rocket_id])
        self.base_inertia = self._host_model.body_inertia[self._rocket_id].copy()
        self.base_thrust = 16.68

    @property
    def backend(self) -> str:
        return "mjx"

    @property
    def observation_size(self) -> int:
        return 12

    @property
    def action_size(self) -> int:
        return 2

    @property
    def mjx_model(self):
        return self._model

    @property
    def host_model(self):
        return self._host_model

    def _body_z(self, data: mjx.Data) -> jax.Array:
        # data.xmat has shape (nbody, 3, 3) or (nbody, 9)
        # Extracted rocket rotation Z axis
        return data.xmat[self._rocket_id, :, 2]

    def _tilt_rad(self, data: mjx.Data) -> jax.Array:
        return jp.arccos(jp.clip(self._body_z(data)[2], -1.0, 1.0))

    def _angular_velocity(self, data: mjx.Data) -> jax.Array:
        roll_vel = data.qvel[self._roll_qvel]
        pitch_vel = data.qvel[self._pitch_qvel]
        roll_pos = data.qpos[self._roll_qpos]
        
        omega_x = roll_vel
        omega_y = pitch_vel * jp.cos(roll_pos)
        omega_z = pitch_vel * jp.sin(roll_pos)
        return jp.array([omega_x, omega_y, omega_z])


    def _normalized_tilt_cost(self, tilt: jax.Array) -> jax.Array:
        scale = jp.maximum(jp.deg2rad(self.stage.max_start_tilt_deg), jp.deg2rad(5.0))
        return jp.square(tilt / scale)

    def reset(self, rng: jax.Array) -> base.State:
        keys = jax.random.split(rng, 10)
        
        # Domain Randomization scales
        mass_scale = jp.where(self.randomize_domain, jax.random.uniform(keys[0], (), minval=0.9, maxval=1.1), 1.0)
        inertia_scale = jp.where(self.randomize_domain, jax.random.uniform(keys[1], (3,), minval=0.85, maxval=1.15), jp.ones(3))
        thrust_scale = jp.where(self.randomize_domain, jax.random.uniform(keys[2], (), minval=0.9, maxval=1.1), 1.0)
        servo_scale = jp.where(self.randomize_domain, jax.random.uniform(keys[3], (), minval=0.85, maxval=1.15), 1.0)
        imu_noise_std = jp.where(self.randomize_domain, jax.random.uniform(keys[4], (), minval=0.0, maxval=0.005), 0.0)
        
        # Sensor delay
        delay_ms = jp.where(self.randomize_domain, jax.random.uniform(keys[5], (), minval=0.0, maxval=20.0), 0.0)
        sensor_delay_steps = jp.where(delay_ms >= 0.5 * self.control_dt * 1000.0, 1, 0)
        
        # Initial tilt state setup
        max_tilt_deg = self.stage.max_start_tilt_deg
        max_rate_dps = self.stage.max_start_rate_dps
        min_tilt_deg = jp.minimum(5.0, max_tilt_deg)
        tilt = jp.deg2rad(jax.random.uniform(keys[6], (), minval=min_tilt_deg, maxval=max_tilt_deg))
        direction = jax.random.uniform(keys[7], (), minval=-jp.pi, maxval=jp.pi)
        
        qpos = jp.zeros((self._host_model.nq,))
        qpos = qpos.at[self._roll_qpos].set(tilt * jp.cos(direction))
        qpos = qpos.at[self._pitch_qpos].set(tilt * jp.sin(direction))
        
        max_rate = jp.deg2rad(max_rate_dps)
        qvel = jp.zeros((self._host_model.nv,))
        qvel = qvel.at[self._roll_qvel].set(jax.random.uniform(keys[8], (), minval=-max_rate, maxval=max_rate))
        qvel = qvel.at[self._pitch_qvel].set(jax.random.uniform(keys[9], (), minval=-max_rate, maxval=max_rate))
        
        # Sync initial model values
        model = self._model
        if self.randomize_domain:
            body_mass = model.body_mass.at[self._rocket_id].set(self.base_mass * mass_scale)
            body_inertia = model.body_inertia.at[self._rocket_id].set(self.base_inertia * inertia_scale)
            model = model.replace(body_mass=body_mass, body_inertia=body_inertia)
            
        data = mjx.make_data(model).replace(qpos=qpos, qvel=qvel)
        data = mjx.forward(model, data)
        
        # Initial state construction
        tilt_cost = self._normalized_tilt_cost(self._tilt_rad(data))
        
        gimbal_state = GimbalMJXState(
            data=data,
            previous_action=jp.zeros(2),
            tvc_target=jp.zeros(2),
            tvc_rate=jp.zeros(2),
            tvc_rate_command=jp.zeros(2),
            step_count=jp.array(0, dtype=jp.int32),
            success_hold_time=jp.array(0.0),
            disturbances_applied=jp.array(0, dtype=jp.int32),
            disturbance_end_step=jp.array(-1, dtype=jp.int32),
            disturbance_torque=jp.zeros(3),
            last_tilt_cost=tilt_cost,
            mass_scale=mass_scale,
            inertia_scale=inertia_scale,
            thrust_scale=thrust_scale,
            servo_scale=servo_scale,
            imu_noise_std=imu_noise_std,
            sensor_delay_steps=sensor_delay_steps,
            previous_obs=jp.zeros(12),
        )
        
        raw_obs = self._raw_observation(gimbal_state)
        gimbal_state = gimbal_state.replace(previous_obs=raw_obs)
        
        return base.State(
            pipeline_state=gimbal_state,
            obs=raw_obs,
            reward=jp.array(0.0),
            done=jp.array(0.0),
            metrics={
                "reward": jp.array(0.0),
                "tilt_deg": jp.rad2deg(self._tilt_rad(data)),
                "success": jp.array(0.0),
            },
            info={},
        )

    def _raw_observation(self, state: GimbalMJXState) -> jax.Array:
        data = state.data
        body_z = self._body_z(data)
        angular_velocity = self._angular_velocity(data)
        
        # Replicate IMU noise
        noise_key = jax.random.PRNGKey(state.step_count)
        noise = jax.random.normal(noise_key, (3,)) * state.imu_noise_std
        body_z_noisy = body_z + noise
        body_z_noisy = body_z_noisy / jp.maximum(jp.linalg.norm(body_z_noisy), 1e-8)
        angular_velocity_noisy = angular_velocity + jax.random.normal(noise_key, (3,)) * state.imu_noise_std * 10.0
        
        obs = jp.concatenate([
            body_z_noisy,
            angular_velocity_noisy / self.angular_velocity_observation_scale,
            jp.array([data.qpos[self._tvc_x_qpos], data.qpos[self._tvc_y_qpos]]) / self.max_tvc_angle,
            state.tvc_rate / self.tvc_velocity_observation_scale,
            state.previous_action,
        ])
        return jp.clip(jp.nan_to_num(obs), -1.0, 1.0)

    def _get_obs(self, state: GimbalMJXState) -> jax.Array:
        current_raw_obs = self._raw_observation(state)
        return jp.where(state.sensor_delay_steps > 0, state.previous_obs, current_raw_obs)


    def _physics_substep(self, carry, inputs) -> Tuple[Tuple[mjx.Data, jax.Array], None]:
        data, tvc_target = carry
        thrust_scale, servo_scale, step_count, dist_torque, dist_end_step = inputs
        
        # Prescribed TVC servo command sync
        qpos = data.qpos.at[self._tvc_x_qpos].set(tvc_target[0])
        qpos = qpos.at[self._tvc_y_qpos].set(tvc_target[1])
        qvel = data.qvel.at[self._tvc_x_qvel].set(0.0)
        qvel = qvel.at[self._tvc_y_qvel].set(0.0)
        data = data.replace(qpos=qpos, qvel=qvel)
        data = mjx.forward(self._model, data)
        
        # Thrust force vector
        site_rotation = data.site_xmat[self._thrust_site_id].reshape(3, 3)
        thrust_dir = site_rotation[:, 2]
        force = thrust_dir * self.fixed_thrust_power * self.base_thrust * thrust_scale
        
        # Restoring torque
        lever = data.site_xpos[self._thrust_site_id] - data.xipos[self._rocket_id]
        torque = jp.cross(lever, force)

        
        # Apply external tork disturbances
        physics_step = step_count * self.frame_skip
        is_disturbing = physics_step < dist_end_step
        applied_dist_torque = jp.where(is_disturbing, dist_torque, jp.zeros(3))
        
        xfrc = jp.zeros_like(data.xfrc_applied)
        xfrc = xfrc.at[self._rocket_id, 0:3].set(force)
        xfrc = xfrc.at[self._rocket_id, 3:6].set(torque + applied_dist_torque)
        data = data.replace(xfrc_applied=xfrc)
        
        data = mjx.step(self._model, data)
        return (data, tvc_target), None

    def step(self, state: base.State, action: jax.Array) -> base.State:
        gstate = state.pipeline_state
        action = jp.clip(action, -1.0, 1.0)
        
        # Disturbance trigger logic (curriculum Stage 2 & 3)
        # Check if disturbance needs to start
        spacing = int(2.0 * self.control_rate)
        # Disturbances timing schedule: spacing * (index + 1)
        # Max disturbances depends on stage
        max_disturbances = self.stage.disturbances
        
        rng_key = jax.random.PRNGKey(gstate.step_count)
        rng_keys = jax.random.split(rng_key, 3)
        
        # Logic to trigger a new disturbance
        should_trigger = (gstate.step_count > 0) & (gstate.step_count % spacing == 0) & (gstate.disturbances_applied < max_disturbances)
        
        # Generate random disturbance tork vector
        angle = jax.random.uniform(rng_keys[0], (), minval=0.0, maxval=2.0 * jp.pi)
        magnitude = jax.random.uniform(rng_keys[1], (), minval=0.25, maxval=0.55)
        new_torque = magnitude * jp.array([jp.cos(angle), jp.sin(angle), 0.0])
        new_end_step = gstate.step_count * self.frame_skip + self.disturbance_steps
        
        dist_torque = jp.where(should_trigger, new_torque, gstate.disturbance_torque)
        dist_end_step = jp.where(should_trigger, new_end_step, gstate.disturbance_end_step)
        dist_applied = jp.where(should_trigger, gstate.disturbances_applied + 1, gstate.disturbances_applied)
        success_hold = jp.where(should_trigger, 0.0, gstate.success_hold_time)
        
        # Actuator command updates
        command_speed = self.base_command_speed * gstate.servo_scale
        max_rate_command_delta = 0.25
        
        tvc_rate_cmd = gstate.tvc_rate_command + jp.clip(
            action - gstate.tvc_rate_command,
            -max_rate_command_delta,
            max_rate_command_delta,
        )
        requested_target = gstate.tvc_target + tvc_rate_cmd * command_speed * self.control_dt
        tvc_target = jp.clip(requested_target, -self.max_tvc_angle, self.max_tvc_angle)
        tvc_rate = (tvc_target - gstate.tvc_target) / self.control_dt
        
        # Carry out simulator frame_skip substeps via jax scan
        carry_init = (gstate.data, tvc_target)
        scan_inputs = (
            jp.repeat(gstate.thrust_scale, self.frame_skip),
            jp.repeat(gstate.servo_scale, self.frame_skip),
            jp.repeat(gstate.step_count, self.frame_skip),
            jp.repeat(dist_torque[None, :], self.frame_skip, axis=0),
            jp.repeat(dist_end_step, self.frame_skip),
        )
        
        (final_data, _), _ = jax.lax.scan(
            self._physics_substep,
            carry_init,
            scan_inputs,
            length=self.frame_skip,
        )
        
        next_step_count = gstate.step_count + 1
        
        # Rewards and Cost computation
        tilt = self._tilt_rad(final_data)
        angular_velocity = self._angular_velocity(final_data)
        tvc_angle = jp.array([final_data.qpos[self._tvc_x_qpos], final_data.qpos[self._tvc_y_qpos]])
        tvc_vel = tvc_rate
        
        tilt_cost = self._normalized_tilt_cost(tilt)
        progress = gstate.last_tilt_cost - tilt_cost
        delta_action = action - gstate.previous_action
        
        angular_cost = jp.dot(angular_velocity[:2], angular_velocity[:2]) / (jp.deg2rad(45.0) ** 2)
        tvc_angle_cost = jp.dot(tvc_angle, tvc_angle) / (self.max_tvc_angle ** 2)
        tvc_target_cost = jp.dot(tvc_target, tvc_target) / (self.max_tvc_angle ** 2)
        tvc_velocity_cost = jp.dot(tvc_vel, tvc_vel) / (self.base_command_speed ** 2)
        action_cost = jp.dot(action, action)
        action_delta_cost = jp.dot(delta_action, delta_action)
        saturation_cost = jp.sum(
            jp.square(jp.maximum(jp.abs(tvc_target) / self.max_tvc_angle - 0.85, 0.0))
        ) / (0.15 ** 2)
        
        reward_terms = {
            "progress": 25.0 * progress,
            "tilt": -4.5 * tilt_cost,
            "angular_rate": -0.25 * angular_cost,
            "tvc_angle": -0.05 * tvc_angle_cost,
            "tvc_target": -0.05 * tvc_target_cost,
            "tvc_velocity": -0.15 * tvc_velocity_cost,
            "action": -0.08 * action_cost,
            "action_delta": -0.35 * action_delta_cost,
            "saturation": -0.75 * saturation_cost,
        }
        reward = sum(reward_terms.values())
        
        # Stability criteria checking
        last_dist_step = max_disturbances * spacing
        after_last_disturbance = next_step_count > last_dist_step
        stable = (
            (tilt <= jp.deg2rad(5.0))
            & (jp.linalg.norm(angular_velocity) <= jp.deg2rad(5.0))
            & (jp.max(jp.abs(tvc_angle)) <= jp.deg2rad(3.0))
            & (jp.max(jp.abs(tvc_vel)) <= jp.deg2rad(20.0))
        )
        
        success_hold = jp.where(after_last_disturbance & stable, success_hold + self.control_dt, 0.0)
        
        # Success and termination conditions
        success = success_hold >= 1.0 # required_success_hold_sec = 1.0
        fallen = tilt >= jp.deg2rad(50.0)
        timeout = next_step_count >= self.max_steps
        
        terminated = success | fallen
        truncated = timeout & (~terminated)
        done = terminated | truncated
        
        # Reward adjustments for success/failure
        reward = jp.where(success, reward + 100.0, reward)
        reward = jp.where(fallen | (timeout & (~success)), reward - 100.0, reward)
        
        new_gstate = GimbalMJXState(
            data=final_data,
            previous_action=action,
            tvc_target=tvc_target,
            tvc_rate=tvc_rate,
            tvc_rate_command=tvc_rate_cmd,
            step_count=next_step_count,
            success_hold_time=success_hold,
            disturbances_applied=dist_applied,
            disturbance_end_step=dist_end_step,
            disturbance_torque=dist_torque,
            last_tilt_cost=tilt_cost,
            mass_scale=gstate.mass_scale,
            inertia_scale=gstate.inertia_scale,
            thrust_scale=gstate.thrust_scale,
            servo_scale=gstate.servo_scale,
            imu_noise_std=gstate.imu_noise_std,
            sensor_delay_steps=gstate.sensor_delay_steps,
            previous_obs=jp.zeros(12),
        )
        
        raw_obs = self._raw_observation(new_gstate)
        new_gstate = new_gstate.replace(previous_obs=raw_obs)
        
        obs = jp.where(gstate.sensor_delay_steps > 0, gstate.previous_obs, raw_obs)
        
        metrics = {
            "reward": reward,
            "tilt_deg": jp.rad2deg(tilt),
            "success": success.astype(jp.float32),
        }
        
        return base.State(
            pipeline_state=new_gstate,
            obs=obs,
            reward=reward,
            done=done.astype(jp.float32),
            metrics=metrics,
            info=state.info,
        )
