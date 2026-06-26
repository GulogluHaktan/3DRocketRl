from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
import time

import gymnasium as gym
import mujoco
import mujoco.viewer
import numpy as np
from gymnasium import spaces


MODEL_XML = Path(__file__).with_name("model.xml")
OBSERVATION_NAMES = (
    "body_z_x",
    "body_z_y",
    "body_z_z",
    "angular_velocity_x",
    "angular_velocity_y",
    "angular_velocity_z",
    "tvc_x_angle",
    "tvc_y_angle",
    "tvc_x_velocity",
    "tvc_y_velocity",
    "previous_action_x",
    "previous_action_y",
)


@dataclass(frozen=True)
class GimbalStage:
    number: int
    max_start_tilt_deg: float
    max_start_rate_dps: float
    disturbances: int
    domain_randomization: bool


STAGES = {
    1: GimbalStage(1, 15.0, 10.0, 0, False),
    2: GimbalStage(2, 30.0, 20.0, 1, False),
    3: GimbalStage(3, 45.0, 30.0, 3, True),
}


class GimbalEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 50}

    def __init__(
        self,
        stage: int = 1,
        control_rate: float = 50.0,
        fixed_thrust_power: float = 1.0,
        max_tvc_deg: float = 20.0,
        tvc_servo_sec_per_60deg: float = 0.13,
        max_tvc_rate_dps: float = 120.0,
        max_steps: int = 500,
        render_mode: str | None = None,
        domain_randomization: bool | None = None,
        sensor_delay_ms: float | None = None,
        imu_noise_std: float | None = None,
        training_mode: bool = False,
        success_hold_sec: float = 1.0,
    ):
        super().__init__()
        if stage not in STAGES:
            raise ValueError("stage must be 1, 2, or 3")
        if control_rate <= 0:
            raise ValueError("control_rate must be positive")

        self.stage = STAGES[stage]
        self.control_rate = float(control_rate)
        self.control_dt = 1.0 / self.control_rate
        self.fixed_thrust_power = float(np.clip(fixed_thrust_power, 0.0, 1.0))
        self.max_tvc_angle = np.deg2rad(max(float(max_tvc_deg), 0.0))
        self.base_servo_speed = np.deg2rad(60.0) / max(float(tvc_servo_sec_per_60deg), 1e-6)
        self.base_command_speed = min(
            np.deg2rad(max(float(max_tvc_rate_dps), 0.0)),
            self.base_servo_speed,
        )
        self.angular_velocity_observation_scale = np.deg2rad(45.0)
        self.tvc_velocity_observation_scale = np.deg2rad(120.0)
        self.max_steps = int(max_steps)
        self.render_mode = render_mode
        self.randomize_domain = (
            self.stage.domain_randomization
            if domain_randomization is None
            else bool(domain_randomization)
        )
        self.requested_sensor_delay_ms = sensor_delay_ms
        self.requested_imu_noise_std = imu_noise_std
        self.training_mode = bool(training_mode)
        self.required_success_hold_sec = max(float(success_hold_sec), 0.0)

        self.model = mujoco.MjModel.from_xml_path(str(MODEL_XML))
        sim_dt = float(self.model.opt.timestep)
        frame_skip = round(self.control_dt / sim_dt)
        if frame_skip < 1 or not np.isclose(frame_skip * sim_dt, self.control_dt, atol=1e-9):
            raise ValueError("control_rate must be an integer divisor of the MuJoCo timestep")
        self.frame_skip = int(frame_skip)
        self.data = mujoco.MjData(self.model)

        self.rocket_id = self._id(mujoco.mjtObj.mjOBJ_BODY, "rocket")
        self.thrust_site_id = self._id(mujoco.mjtObj.mjOBJ_SITE, "thrust_site")
        self.roll_joint_id = self._id(mujoco.mjtObj.mjOBJ_JOINT, "gimbal_roll")
        self.pitch_joint_id = self._id(mujoco.mjtObj.mjOBJ_JOINT, "gimbal_pitch")
        self.tvc_x_joint_id = self._id(mujoco.mjtObj.mjOBJ_JOINT, "tvc_x")
        self.tvc_y_joint_id = self._id(mujoco.mjtObj.mjOBJ_JOINT, "tvc_y")
        self.roll_qpos = int(self.model.jnt_qposadr[self.roll_joint_id])
        self.pitch_qpos = int(self.model.jnt_qposadr[self.pitch_joint_id])
        self.roll_qvel = int(self.model.jnt_dofadr[self.roll_joint_id])
        self.pitch_qvel = int(self.model.jnt_dofadr[self.pitch_joint_id])
        self.tvc_x_qpos = int(self.model.jnt_qposadr[self.tvc_x_joint_id])
        self.tvc_y_qpos = int(self.model.jnt_qposadr[self.tvc_y_joint_id])
        self.tvc_x_qvel = int(self.model.jnt_dofadr[self.tvc_x_joint_id])
        self.tvc_y_qvel = int(self.model.jnt_dofadr[self.tvc_y_joint_id])
        for joint_id in (self.tvc_x_joint_id, self.tvc_y_joint_id):
            self.model.jnt_range[joint_id] = (-self.max_tvc_angle, self.max_tvc_angle)

        self.base_mass = float(self.model.body_mass[self.rocket_id])
        self.base_inertia = self.model.body_inertia[self.rocket_id].copy()
        self.base_body_pos = self.model.body_pos[self.rocket_id].copy()
        self.base_thrust = 16.68
        self.max_thrust = self.base_thrust
        self.servo_speed = self.base_servo_speed
        self.command_speed = self.base_command_speed
        self.imu_noise_std = 0.0
        self.sensor_delay_steps = 0

        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(12,), dtype=np.float32)
        self.previous_action = np.zeros(2, dtype=np.float32)
        self.tvc_target = np.zeros(2, dtype=np.float64)
        self.tvc_rate = np.zeros(2, dtype=np.float64)
        self.tvc_rate_command = np.zeros(2, dtype=np.float64)
        self.max_rate_command_delta = 0.25
        self.step_count = 0
        self.success_hold_time = 0.0
        self.disturbances_applied = 0
        self.disturbance_end_step = -1
        self.disturbance_torque = np.zeros(3, dtype=np.float64)
        self.disturbance_steps = max(1, round(0.08 / sim_dt))
        self.last_disturbance_control_step = 0
        self.disturbance_schedule = ()
        self.last_tilt_cost = 0.0
        self.last_reward_terms = {}
        self.viewer = None
        self.observation_history: deque[np.ndarray] = deque(maxlen=2)

    def _id(self, object_type, name):
        object_id = mujoco.mj_name2id(self.model, object_type, name)
        if object_id < 0:
            raise RuntimeError(f"MuJoCo object not found: {name}")
        return object_id

    @property
    def tilt_rad(self):
        return float(np.arccos(np.clip(self._body_z()[2], -1.0, 1.0)))

    def _body_z(self):
        return self.data.xmat[self.rocket_id].reshape(3, 3)[:, 2].copy()

    def _angular_velocity(self):
        velocity = np.zeros(6, dtype=np.float64)
        mujoco.mj_objectVelocity(
            self.model,
            self.data,
            mujoco.mjtObj.mjOBJ_BODY,
            self.rocket_id,
            velocity,
            0,
        )
        return velocity[:3].copy()

    def _randomize_model(self):
        self.model.body_mass[self.rocket_id] = self.base_mass
        self.model.body_inertia[self.rocket_id] = self.base_inertia
        self.model.body_pos[self.rocket_id] = self.base_body_pos
        self.max_thrust = self.base_thrust
        self.servo_speed = self.base_servo_speed
        self.command_speed = self.base_command_speed
        self.imu_noise_std = 0.0
        self.sensor_delay_steps = 0
        if not self.randomize_domain:
            return
        self.model.body_mass[self.rocket_id] = self.base_mass * self.np_random.uniform(0.9, 1.1)
        self.model.body_inertia[self.rocket_id] = (
            self.base_inertia * self.np_random.uniform(0.85, 1.15, size=3)
        )
        self.model.body_pos[self.rocket_id, :2] = self.np_random.uniform(-0.005, 0.005, size=2)
        self.max_thrust = self.base_thrust * self.np_random.uniform(0.9, 1.1)
        self.servo_speed = self.base_servo_speed * self.np_random.uniform(0.85, 1.15)
        self.command_speed = min(
            self.base_command_speed * self.np_random.uniform(0.85, 1.15),
            self.servo_speed,
        )
        self.imu_noise_std = float(
            self.requested_imu_noise_std
            if self.requested_imu_noise_std is not None
            else self.np_random.uniform(0.0, 0.005)
        )
        delay_ms = (
            self.requested_sensor_delay_ms
            if self.requested_sensor_delay_ms is not None
            else self.np_random.uniform(0.0, 20.0)
        )
        self.sensor_delay_steps = int(delay_ms >= 0.5 * self.control_dt * 1000.0)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self._randomize_model()
        self.step_count = 0
        self.success_hold_time = 0.0
        self.disturbances_applied = 0
        self.disturbance_end_step = -1
        self.disturbance_torque[:] = 0.0
        self.previous_action[:] = 0.0
        self.tvc_target[:] = 0.0
        self.tvc_rate[:] = 0.0
        self.tvc_rate_command[:] = 0.0

        easy_reset_probability = {1: 0.50, 2: 0.20, 3: 0.10}[self.stage.number]
        easy_reset = (
            self.training_mode
            and self.np_random.random() < easy_reset_probability
        )
        if easy_reset:
            max_tilt_deg = min(5.0, self.stage.max_start_tilt_deg)
            max_rate_dps = min(2.0, self.stage.max_start_rate_dps)
        else:
            max_tilt_deg = self.stage.max_start_tilt_deg
            max_rate_dps = self.stage.max_start_rate_dps
        min_tilt_deg = min(5.0, max_tilt_deg)
        tilt = np.deg2rad(self.np_random.uniform(min_tilt_deg, max_tilt_deg))
        direction = self.np_random.uniform(-np.pi, np.pi)
        self.data.qpos[self.roll_qpos] = tilt * np.cos(direction)
        self.data.qpos[self.pitch_qpos] = tilt * np.sin(direction)
        max_rate = np.deg2rad(max_rate_dps)
        self.data.qvel[self.roll_qvel] = self.np_random.uniform(-max_rate, max_rate)
        self.data.qvel[self.pitch_qvel] = self.np_random.uniform(-max_rate, max_rate)
        self._sync_tvc_state()
        mujoco.mj_forward(self.model, self.data)

        if self.stage.disturbances:
            spacing = max(1, int(2.0 * self.control_rate))
            self.disturbance_schedule = tuple(
                spacing * (index + 1) for index in range(self.stage.disturbances)
            )
            self.last_disturbance_control_step = self.disturbance_schedule[-1]
        else:
            self.disturbance_schedule = ()
            self.last_disturbance_control_step = 0

        self.last_tilt_cost = self._normalized_tilt_cost(self.tilt_rad)
        self.last_reward_terms = {}
        raw_obs = self._raw_observation()
        self.observation_history = deque(
            [raw_obs.copy(), raw_obs.copy()],
            maxlen=max(2, self.sensor_delay_steps + 1),
        )
        return self._observation(), self._info()

    def _start_disturbance(self):
        angle = self.np_random.uniform(0.0, 2.0 * np.pi)
        magnitude = self.np_random.uniform(0.25, 0.55)
        self.disturbance_torque[:] = magnitude * np.array(
            [np.cos(angle), np.sin(angle), 0.0]
        )
        self.disturbance_end_step = (
            self.step_count * self.frame_skip + self.disturbance_steps
        )
        self.disturbances_applied += 1
        self.success_hold_time = 0.0

    def _apply_forces(self, physics_step):
        self.data.qfrc_applied[:] = 0.0
        self.data.xfrc_applied[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        site_pos = self.data.site_xpos[self.thrust_site_id].copy()
        site_rotation = self.data.site_xmat[self.thrust_site_id].reshape(3, 3)
        thrust_force = (
            site_rotation[:, 2] * self.fixed_thrust_power * self.max_thrust
        )
        mujoco.mj_applyFT(
            self.model,
            self.data,
            thrust_force,
            np.zeros(3),
            site_pos,
            self.rocket_id,
            self.data.qfrc_applied,
        )
        if physics_step < self.disturbance_end_step:
            self.data.xfrc_applied[self.rocket_id, 3:6] += self.disturbance_torque

    def _sync_tvc_state(self):
        self.data.qpos[self.tvc_x_qpos] = self.tvc_target[0]
        self.data.qpos[self.tvc_y_qpos] = self.tvc_target[1]
        # Servo state is prescribed; these generalized velocities must not
        # create a second, uncontrolled MuJoCo servo dynamic.
        self.data.qvel[self.tvc_x_qvel] = 0.0
        self.data.qvel[self.tvc_y_qvel] = 0.0

    def _normalized_tilt_cost(self, tilt):
        scale = max(np.deg2rad(self.stage.max_start_tilt_deg), np.deg2rad(5.0))
        return float((tilt / scale) ** 2)

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        if self.step_count in self.disturbance_schedule:
            self._start_disturbance()

        previous_tvc_target = self.tvc_target.copy()
        self.tvc_rate_command += np.clip(
            action - self.tvc_rate_command,
            -self.max_rate_command_delta,
            self.max_rate_command_delta,
        )
        requested_target = (
            previous_tvc_target
            + self.tvc_rate_command * self.command_speed * self.control_dt
        )
        self.tvc_target = np.clip(
            requested_target, -self.max_tvc_angle, self.max_tvc_angle
        )
        self.tvc_rate = (self.tvc_target - previous_tvc_target) / self.control_dt
        self._sync_tvc_state()
        mujoco.mj_forward(self.model, self.data)
        physics_start = self.step_count * self.frame_skip
        for offset in range(self.frame_skip):
            self._sync_tvc_state()
            self._apply_forces(physics_start + offset)
            mujoco.mj_step(self.model, self.data)
            self._sync_tvc_state()
        mujoco.mj_forward(self.model, self.data)

        self.step_count += 1
        tilt = self.tilt_rad
        angular_velocity = self._angular_velocity()
        tvc_angle = np.array(
            [self.data.qpos[self.tvc_x_qpos], self.data.qpos[self.tvc_y_qpos]]
        )
        tvc_velocity = self.tvc_rate.copy()
        tilt_cost = self._normalized_tilt_cost(tilt)
        progress = self.last_tilt_cost - tilt_cost
        delta_action = action - self.previous_action
        angular_cost = float(
            np.dot(angular_velocity[:2], angular_velocity[:2])
            / np.deg2rad(45.0) ** 2
        )
        tvc_angle_cost = float(
            np.dot(tvc_angle, tvc_angle) / max(self.max_tvc_angle**2, 1e-8)
        )
        tvc_target_cost = float(
            np.dot(self.tvc_target, self.tvc_target)
            / max(self.max_tvc_angle**2, 1e-8)
        )
        tvc_velocity_cost = float(
            np.dot(tvc_velocity, tvc_velocity) / max(self.command_speed**2, 1e-8)
        )
        action_cost = float(np.dot(action, action))
        action_delta_cost = float(np.dot(delta_action, delta_action))
        saturation_cost = float(
            np.sum(
                np.maximum(
                    np.abs(self.tvc_target) / max(self.max_tvc_angle, 1e-8) - 0.85,
                    0.0,
                )
                ** 2
            )
            / 0.15**2
        )
        self.last_reward_terms = {
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
        reward = float(sum(self.last_reward_terms.values()))
        self.last_tilt_cost = tilt_cost
        self.previous_action = action.copy()

        after_last_disturbance = self.step_count > self.last_disturbance_control_step
        stable = (
            tilt <= np.deg2rad(5.0)
            and np.linalg.norm(angular_velocity) <= np.deg2rad(5.0)
            and np.max(np.abs(tvc_angle)) <= np.deg2rad(3.0)
            and np.max(np.abs(tvc_velocity)) <= np.deg2rad(20.0)
        )
        if after_last_disturbance and stable:
            self.success_hold_time += self.control_dt
        else:
            self.success_hold_time = 0.0

        success = self.success_hold_time >= self.required_success_hold_sec
        fallen = tilt >= np.deg2rad(50.0)
        timeout = self.step_count >= self.max_steps
        terminated = bool(success or fallen)
        truncated = bool(timeout and not terminated)
        if success:
            reward += 100.0
        elif fallen or timeout:
            reward -= 100.0

        raw_obs = self._raw_observation()
        self.observation_history.append(raw_obs)
        if self.render_mode == "human":
            self.render()
        return self._observation(), float(reward), terminated, truncated, self._info(
            success=success, fallen=fallen, timeout=timeout
        )

    def _raw_observation(self):
        body_z = self._body_z()
        angular_velocity = self._angular_velocity()
        if self.imu_noise_std:
            body_z += self.np_random.normal(0.0, self.imu_noise_std, size=3)
            body_z /= max(np.linalg.norm(body_z), 1e-8)
            angular_velocity += self.np_random.normal(
                0.0, self.imu_noise_std * 10.0, size=3
            )
        values = np.array(
            [
                *body_z,
                *(angular_velocity / self.angular_velocity_observation_scale),
                self.data.qpos[self.tvc_x_qpos] / max(self.max_tvc_angle, 1e-8),
                self.data.qpos[self.tvc_y_qpos] / max(self.max_tvc_angle, 1e-8),
                self.tvc_rate[0] / self.tvc_velocity_observation_scale,
                self.tvc_rate[1] / self.tvc_velocity_observation_scale,
                *self.previous_action,
            ],
            dtype=np.float32,
        )
        return np.clip(np.nan_to_num(values), -1.0, 1.0)

    def _observation(self):
        index = max(0, len(self.observation_history) - 1 - self.sensor_delay_steps)
        return self.observation_history[index].copy()

    def _info(self, success=False, fallen=False, timeout=False):
        angular_velocity = self._angular_velocity()
        tvc_angle = np.array(
            [self.data.qpos[self.tvc_x_qpos], self.data.qpos[self.tvc_y_qpos]]
        )
        tvc_velocity = self.tvc_rate.copy()
        return {
            "stage": self.stage.number,
            "success": bool(success),
            "fallen": bool(fallen),
            "timeout": bool(timeout),
            "tilt_deg": float(np.rad2deg(self.tilt_rad)),
            "angular_speed_dps": float(np.rad2deg(np.linalg.norm(angular_velocity))),
            "tvc_angle_deg": np.rad2deg(tvc_angle).astype(float),
            "tvc_speed_dps": np.rad2deg(tvc_velocity).astype(float),
            "success_hold_sec": float(self.success_hold_time),
            "required_success_hold_sec": self.required_success_hold_sec,
            "disturbances_applied": self.disturbances_applied,
            "fixed_thrust_power": self.fixed_thrust_power,
            "sensor_delay_ms": self.sensor_delay_steps * self.control_dt * 1000.0,
            "tvc_rate_command": self.tvc_rate_command.astype(float).copy(),
            "reward_terms": dict(self.last_reward_terms),
        }

    def render(self):
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.viewer.sync()
        time.sleep(self.control_dt)

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
