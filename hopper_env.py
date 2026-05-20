from __future__ import annotations

import time
from pathlib import Path

import gymnasium as gym
import mujoco
import mujoco.viewer
import numpy as np
from gymnasium import spaces


MODEL_XML = Path(__file__).with_name("hopper_default.xml")


OBSERVATION_NAMES = (
    "bottom_target_delta_x",
    "bottom_target_delta_y",
    "bottom_target_delta_z",
    "body_quat_w",
    "body_quat_x",
    "body_quat_y",
    "body_quat_z",
    "bottom_height",
    "linear_velocity_x",
    "linear_velocity_y",
    "linear_velocity_z",
    "angular_velocity_x",
    "angular_velocity_y",
    "angular_velocity_z",
    "tvc_yaw_angle",
    "tvc_pitch_angle",
    "tvc_yaw_velocity",
    "tvc_pitch_velocity",
    "foot1_contact",
    "foot2_contact",
    "foot3_contact",
    "foot4_contact",
    "main_motor_power",
)


PHASES = ("climb", "flip", "recovery", "hover", "done")
FULL_FLIP_RAD = 2.0 * np.pi


DEFAULT_REWARD_WEIGHTS = {
    "climb_z_error": 1.5,
    "climb_high_z": 0.5,
    "climb_low_altitude": 1.0,
    "climb_low_altitude_sq": 0.5,
    "climb_x": 1.15,
    "climb_y": 1.25,
    "climb_rel_dist_sq": 1.5,
    "climb_angular_speed": 2.0,
    "climb_joint_speed": 0.09,
    "climb_upright": 15.0,
    "climb_speed": 3.4,
    "flip_height_error": 0.35,
    "flip_progress": 35.0,
    "flip_stall": 0.35,
    "flip_overrotate": 1.0,
    "flip_low_altitude": 2.5,
    "flip_upright_change": 30.0,
    "flip_high_angular_speed": 1.0,
    "flip_axis_alignment": 8.0,
    "flip_off_axis": 0.65,
    "flip_joint_speed": 0.12,
    "flip_side_deviation": 3.0,
    "flip_rel_dist": 3.0,
    "flip_rel_dist_sq": 1.5,
    "flip_late_angular_speed": 0.45,
    "flip_body_spin": 0.35,
    "recovery_linear_speed": 1.5,
    "recovery_angular_speed": 1.45,
    "recovery_joint_speed": 0.09,
    "recovery_upright": 30.0,
    "recovery_flip_progress": 30.0,
    "recovery_rel_dist": 30.0,
    "recovery_rel_progress": 50.0,
    "hover_linear_speed": 1.5,
    "hover_angular_speed": 1.45,
    "hover_joint_speed": 0.09,
    "hover_upright": 30.0,
    "hover_rel_dist": 30.0,
}


class HopperEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        start_z: float = 10.0,
        target_pos=(0.0, 0.0, 5.0),
        max_thrust: float | None = None,
        max_tvc_deg: float = 20.0,
        random_start_z: bool = True,
        min_start_z: float = 0.5,
        max_start_z: float = 10.0,
        reward_weights: dict[str, float] | None = None,
    ):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(str(MODEL_XML))
        self.data = mujoco.MjData(self.model)
        self.body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "hopper")
        self.thrust_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "thrust_site")
        self.yaw_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "tvc_yaw_joint")
        self.pitch_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "tvc_pitch_joint")
        self.yaw_qpos_id = self.model.jnt_qposadr[self.yaw_joint_id]
        self.pitch_qpos_id = self.model.jnt_qposadr[self.pitch_joint_id]
        self.yaw_qvel_id = self.model.jnt_dofadr[self.yaw_joint_id]
        self.pitch_qvel_id = self.model.jnt_dofadr[self.pitch_joint_id]
        self.foot_geom_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"foot{i}_geom")
            for i in range(1, 5)
        ]
        self.pad_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "pad_geom")
        self.ground_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "ground")

        self.start_z = float(start_z)
        self.random_start_z = bool(random_start_z)
        self.min_start_z = float(min_start_z)
        self.max_start_z = float(max_start_z)
        self.target_pos = np.asarray(target_pos, dtype=np.float64)
        self.reward_weights = DEFAULT_REWARD_WEIGHTS.copy()
        if reward_weights:
            self.reward_weights.update(reward_weights)
        self.max_thrust = float(max_thrust if max_thrust is not None else 3.6 * 9.80665)
        self.max_tvc_angle = np.deg2rad(float(max_tvc_deg))
        self.tvc_servo_speed = np.deg2rad(60.0 / 0.14)
        self.tvc_yaw_cmd = 0.0
        self.tvc_pitch_cmd = 0.0
        self.linear_damping = 0.08
        self.angular_damping = 0.34
        self.frame_skip = 8
        self.max_steps = 1200
        self.step_count = 0
        self.main_motor_power = 0.0
        self.viewer = None
        self.follow_camera = True
        self.camera_distance = 5.0
        self.camera_azimuth = 135.0
        self.camera_elevation = -18.0
        self.current_phase = "climb"
        self.flip_angle = 0.0
        self.flip_progress = 0.0
        self.last_step_flip_progress = 0.0
        self.last_upright_score = 1.0
        self.last_rel_dist = 0.0
        self.hover_timer = 0.0
        self.success = False
        self.fail = False
        self.fail_reason = ""
        self.current_start_z = self.start_z
        self.last_phase_bonus = 0.0

        self.action_space = spaces.Box(
            low=np.array([0.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(len(OBSERVATION_NAMES),),
            dtype=np.float32,
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self.main_motor_power = 0.0
        self.tvc_yaw_cmd = 0.0
        self.tvc_pitch_cmd = 0.0
        self.current_phase = "climb"
        self.flip_angle = 0.0
        self.flip_progress = 0.0
        self.last_step_flip_progress = 0.0
        self.last_upright_score = 1.0
        self.last_rel_dist = 0.0
        self.hover_timer = 0.0
        self.success = False
        self.fail = False
        self.fail_reason = ""
        self.last_phase_bonus = 0.0
        if self.random_start_z:
            self.current_start_z = float(self.np_random.uniform(self.min_start_z, self.max_start_z))
        else:
            self.current_start_z = self.start_z
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[0:3] = np.array([0.0, 0.0, self.current_start_z], dtype=np.float64)
        self.data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self.data.qvel[0:6] = np.array([0.0, 0.0, -0.1, 0.0, 0.0, 0.0], dtype=np.float64)
        self.data.ctrl[:] = 0.0
        self.data.qfrc_applied[:] = 0.0
        self.data.xfrc_applied[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        metrics = self._compute_metrics()
        self.last_upright_score = metrics["upright_score"]
        self.last_rel_dist = metrics["rel_dist"]
        return self.get_observation(), {}

    def step(self, action):
        self.step_count += 1
        action = np.clip(np.asarray(action, dtype=np.float32), self.action_space.low, self.action_space.high)
        main = float(action[0])
        yaw_ctrl = float(self.max_tvc_angle * action[1])
        pitch_ctrl = float(self.max_tvc_angle * action[2])
        self.main_motor_power = main

        bad_physics = False
        for _ in range(self.frame_skip):
            self._apply_controls(main, yaw_ctrl, pitch_ctrl)
            mujoco.mj_step(self.model, self.data)
            if self._has_bad_physics():
                bad_physics = True
                break
            if self.viewer is not None:
                self._update_viewer_camera()
                self.viewer.sync()
                time.sleep(self.model.opt.timestep)

        if bad_physics:
            self.fail = True
            self.fail_reason = "bad_physics"
            self.current_phase = "done"
            obs = self.get_observation()
            return obs, -1000.0, True, False, self.get_info()

        self._update_flip_progress()
        metrics = self._compute_metrics()
        reward, fail = self._compute_phase_reward(metrics)
        fail = fail or self._check_fail_conditions(metrics)
        self.fail = bool(fail)
        if self.fail and not self.fail_reason:
            self.fail_reason = self._infer_fail_reason(metrics)
        self.last_phase_bonus = self._update_phase(metrics)
        reward += self.last_phase_bonus
        terminated = bool(self.fail or self.success)
        if self.fail:
            reward -= 1000.0
        if self.success:
            reward += 2500.0

        obs = self.get_observation()
        truncated = self.step_count >= self.max_steps
        self.last_step_flip_progress = self.flip_progress
        self.last_upright_score = metrics["upright_score"]
        self.last_rel_dist = metrics["rel_dist"]
        return obs, float(reward), terminated, truncated, self.get_info()

    def get_observation(self):
        pos, rot, lin_vel, ang_vel = self.get_state()
        body_quat = self.data.qpos[3:7].copy()
        body_quat /= max(np.linalg.norm(body_quat), 1e-8)
        bottom_point = pos + rot @ np.array([0.0, 0.0, -0.355])
        target_delta = self.target_pos - bottom_point
        tvc_yaw = float(self.data.qpos[self.yaw_qpos_id])
        tvc_pitch = float(self.data.qpos[self.pitch_qpos_id])
        tvc_yaw_vel = float(self.data.qvel[self.yaw_qvel_id])
        tvc_pitch_vel = float(self.data.qvel[self.pitch_qvel_id])
        contacts = self.foot_contacts()

        obs = np.array([
            *target_delta,
            *body_quat,
            bottom_point[2],
            *lin_vel,
            *ang_vel,
            tvc_yaw,
            tvc_pitch,
            tvc_yaw_vel,
            tvc_pitch_vel,
            *contacts,
            self.main_motor_power,
        ], dtype=np.float32)
        return np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)

    def get_info(self):
        info = dict(zip(OBSERVATION_NAMES, self.get_observation().tolist()))
        metrics = self._compute_metrics()
        info.update({
            "phase": self.current_phase,
            "success": self.success,
            "fail": self.fail,
            "fail_reason": self.fail_reason,
            "phase_bonus": float(self.last_phase_bonus),
            "flip_progress": float(self.flip_progress),
            "hover_timer": float(self.hover_timer),
            "upright_score": float(metrics["upright_score"]),
            "rel_dist": float(metrics["rel_dist"]),
            "linear_speed": float(metrics["v"]),
            "angular_speed": float(metrics["w"]),
            "joint_speed": float(metrics["joint_speed"]),
            "height": float(metrics["z"]),
        })
        return info

    def get_state(self):
        pos = self.data.xpos[self.body_id].copy()
        rot = self.data.xmat[self.body_id].reshape(3, 3).copy()
        lin_vel = self.data.qvel[0:3].copy()
        ang_vel = self.data.qvel[3:6].copy()
        return pos, rot, lin_vel, ang_vel

    def _has_bad_physics(self):
        if not np.all(np.isfinite(self.data.qpos)):
            return True
        if not np.all(np.isfinite(self.data.qvel)):
            return True
        if not np.all(np.isfinite(self.data.qacc)):
            return True
        if np.linalg.norm(self.data.qvel[0:3]) > 120.0:
            return True
        if np.linalg.norm(self.data.qvel[3:6]) > 180.0:
            return True
        z = float(self.data.qpos[2])
        return bool(z < 0.05 or z > 18.0)

    def foot_contacts(self):
        contacts = np.zeros(4, dtype=np.float32)
        valid_surface_ids = {self.pad_geom_id, self.ground_geom_id}
        for i in range(self.data.ncon):
            con = self.data.contact[i]
            pair = {con.geom1, con.geom2}
            if not pair & valid_surface_ids:
                continue
            for foot_i, foot_geom_id in enumerate(self.foot_geom_ids):
                if foot_geom_id in pair:
                    contacts[foot_i] = 1.0
        return contacts

    def _apply_controls(self, main, yaw_ctrl, pitch_ctrl):
        max_delta = self.tvc_servo_speed * self.model.opt.timestep
        self.tvc_yaw_cmd += np.clip(yaw_ctrl - self.tvc_yaw_cmd, -max_delta, max_delta)
        self.tvc_pitch_cmd += np.clip(pitch_ctrl - self.tvc_pitch_cmd, -max_delta, max_delta)
        self.data.ctrl[0] = np.rad2deg(self.tvc_yaw_cmd)
        self.data.ctrl[1] = np.rad2deg(self.tvc_pitch_cmd)
        self.data.qfrc_applied[:] = 0.0
        self.data.xfrc_applied[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

        site_pos = self.data.site_xpos[self.thrust_site_id].copy()
        site_xmat = self.data.site_xmat[self.thrust_site_id].reshape(3, 3).copy()
        thrust_dir = site_xmat @ np.array([0.0, 0.0, 1.0])
        thrust_force = thrust_dir * (main * self.max_thrust)
        _, _, lin_vel, ang_vel = self.get_state()

        mujoco.mj_applyFT(
            self.model,
            self.data,
            thrust_force,
            np.zeros(3),
            site_pos,
            self.body_id,
            self.data.qfrc_applied,
        )
        self.data.xfrc_applied[self.body_id, 0:3] = -self.linear_damping * lin_vel
        self.data.xfrc_applied[self.body_id, 3:6] = -self.angular_damping * ang_vel

    def _compute_metrics(self):
        pos, rot, linear_vel, angular_vel = self.get_state()
        body_axis = rot @ np.array([0.0, 0.0, 1.0])
        bottom_point = pos + rot @ np.array([0.0, 0.0, -0.355])
        relative_pos = bottom_point - self.target_pos
        v = float(np.linalg.norm(linear_vel))
        w = float(np.linalg.norm(angular_vel))
        joint_speed = float(np.linalg.norm([
            self.data.qvel[self.yaw_qvel_id],
            self.data.qvel[self.pitch_qvel_id],
        ]))
        rel_dist = float(np.linalg.norm(relative_pos[:2]))
        upright_score = float(np.dot(body_axis, np.array([0.0, 0.0, 1.0])))
        spin_about_body_axis = float(np.dot(angular_vel, body_axis))
        flip_axis = np.array([0.0, 1.0, 0.0])
        flip_axis_rate_signed = float(np.dot(angular_vel, flip_axis))
        flip_axis_rate = abs(flip_axis_rate_signed)
        positive_flip_axis_rate = max(flip_axis_rate_signed, 0.0)
        off_axis_angular_vel = angular_vel - flip_axis_rate_signed * flip_axis
        off_axis_angular_speed = float(np.linalg.norm(off_axis_angular_vel))
        expected_flip_angle = min(self.flip_progress, 1.0) * FULL_FLIP_RAD
        expected_body_axis = np.array([
            np.sin(expected_flip_angle),
            0.0,
            np.cos(expected_flip_angle),
        ])
        expected_axis_alignment = float(np.dot(body_axis, expected_body_axis))
        return {
            "z": float(bottom_point[2]),
            "relative_pos": relative_pos,
            "linear_vel": linear_vel,
            "angular_vel": angular_vel,
            "v": v,
            "w": w,
            "joint_speed": joint_speed,
            "rel_dist": rel_dist,
            "horizontal_speed": float(np.linalg.norm(linear_vel[:2])),
            "vertical_speed": abs(float(linear_vel[2])),
            "upright_score": upright_score,
            "body_axis": body_axis,
            "flip_axis_rate": flip_axis_rate,
            "positive_flip_axis_rate": positive_flip_axis_rate,
            "off_axis_angular_speed": off_axis_angular_speed,
            "expected_axis_alignment": expected_axis_alignment,
            "spin_about_body_axis": spin_about_body_axis,
            "side_deviation": abs(float(body_axis[1])),
        }

    def _update_flip_progress(self):
        metrics = self._compute_metrics()
        flip_rate = metrics["positive_flip_axis_rate"]
        dt = self.model.opt.timestep * self.frame_skip
        if self.current_phase == "flip":
            self.flip_angle += flip_rate * dt
            self.flip_progress = self.flip_angle / FULL_FLIP_RAD

    def _compute_phase_reward(self, metrics):
        if self.current_phase == "climb":
            return self._compute_climb_reward(metrics)
        if self.current_phase == "flip":
            return self._compute_flip_reward(metrics)
        if self.current_phase == "recovery":
            return self._compute_recovery_reward(metrics)
        if self.current_phase == "hover":
            return self._compute_hover_reward(metrics)
        return 0.0, False

    def _compute_climb_reward(self, metrics):
        z = metrics["z"]
        relative_pos = metrics["relative_pos"]
        linear_vel = metrics["linear_vel"]
        w = metrics["w"]
        joint_speed = metrics["joint_speed"]
        upright_score = metrics["upright_score"]
        horizontal_speed = float(np.linalg.norm(linear_vel[:2]))
        vertical_speed = abs(float(linear_vel[2]))
        target_z = 10.0
        low_altitude_error = max(9.0 - z, 0.0)

        reward = 0.0
        rw = self.reward_weights
        if z < 10.5:
            reward -= abs(z - target_z) * rw["climb_z_error"]
        else:
            reward -= rw["climb_high_z"] * z
        reward -= low_altitude_error * rw["climb_low_altitude"]
        reward -= low_altitude_error ** 2 * rw["climb_low_altitude_sq"]
        reward -= abs(float(relative_pos[0])) * rw["climb_x"]
        reward -= abs(float(relative_pos[1])) * rw["climb_y"]
        reward -= metrics["rel_dist"] ** 2 * rw["climb_rel_dist_sq"]
        reward -= w * rw["climb_angular_speed"]
        reward -= joint_speed * rw["climb_joint_speed"]
        reward += upright_score * rw["climb_upright"]
        reward -= (horizontal_speed + vertical_speed) * rw["climb_speed"]

        fail = bool(abs(relative_pos[0]) > 3.0 or abs(relative_pos[1]) > 2.0)
        return reward, fail

    def _compute_flip_reward(self, metrics):
        z = metrics["z"]
        w = metrics["w"]
        joint_speed = metrics["joint_speed"]
        upright_score = metrics["upright_score"]
        body_axis = metrics["body_axis"]
        angular_vel = metrics["angular_vel"]
        side_deviation = abs(float(body_axis[1]))
        spin_about_body_axis = float(np.dot(angular_vel, body_axis))
        off_axis_angular_speed = metrics["off_axis_angular_speed"]
        expected_axis_alignment = metrics["expected_axis_alignment"]
        progress_delta = abs(self.flip_progress) - abs(self.last_step_flip_progress)
        rw = self.reward_weights

        reward = 0.0
        flip_height_error = abs(z - 10.0)
        reward -= flip_height_error * rw["flip_height_error"]
        reward += progress_delta * rw["flip_progress"]
        if progress_delta <= 0.001:
            reward -= rw["flip_stall"]
        if self.flip_progress > 1.0:
            reward -= self.flip_progress * rw["flip_overrotate"]
        if z < 5.0:
            reward -= z * rw["flip_low_altitude"]
        upright_change = abs(upright_score - self.last_upright_score)
        if upright_change > 0.3:
            reward -= upright_change * rw["flip_upright_change"]
        if w > 30.0:
            reward -= w * rw["flip_high_angular_speed"]
        reward += expected_axis_alignment * rw["flip_axis_alignment"]
        reward -= off_axis_angular_speed * rw["flip_off_axis"]
        reward -= joint_speed * rw["flip_joint_speed"]
        reward -= side_deviation * rw["flip_side_deviation"]
        reward -= metrics["rel_dist"] * rw["flip_rel_dist"]
        reward -= metrics["rel_dist"] ** 2 * rw["flip_rel_dist_sq"]
        if self.flip_progress > 0.75:
            reward -= w * rw["flip_late_angular_speed"]
        reward -= rw["flip_body_spin"] * abs(spin_about_body_axis)

        fail = bool((z < 3.5 and self.flip_progress < 0.15) or w > 70.0)
        return reward, fail

    def _compute_recovery_reward(self, metrics):
        v = metrics["v"]
        w = metrics["w"]
        joint_speed = metrics["joint_speed"]
        upright_score = metrics["upright_score"]
        rel_dist = metrics["rel_dist"]
        rel_progress = self.last_rel_dist - rel_dist
        rw = self.reward_weights

        reward = 0.0
        reward -= v * rw["recovery_linear_speed"]
        reward -= w * rw["recovery_angular_speed"]
        reward -= joint_speed * rw["recovery_joint_speed"]
        reward += rw["recovery_upright"] * upright_score
        reward += rw["recovery_flip_progress"] * self.flip_progress
        reward -= rel_dist * rw["recovery_rel_dist"]
        reward += rel_progress * rw["recovery_rel_progress"]

        fail = bool(v > 70.0 or w > 70.0 or rel_dist > 1.5)
        return reward, fail

    def _compute_hover_reward(self, metrics):
        v = metrics["v"]
        w = metrics["w"]
        joint_speed = metrics["joint_speed"]
        upright_score = metrics["upright_score"]
        rel_dist = metrics["rel_dist"]
        rw = self.reward_weights

        reward = 0.0
        reward -= v * rw["hover_linear_speed"]
        reward -= w * rw["hover_angular_speed"]
        reward -= joint_speed * rw["hover_joint_speed"]
        reward += rw["hover_upright"] * upright_score
        reward -= rel_dist * rw["hover_rel_dist"]

        fail = bool(rel_dist > 1.5 or v > 70.0 or w > 70.0)
        return reward, fail

    def _check_fail_conditions(self, metrics):
        relative_pos = metrics["relative_pos"]
        v = metrics["v"]
        w = metrics["w"]
        z = metrics["z"]
        rel_dist = metrics["rel_dist"]

        if self._has_bad_physics():
            self.fail_reason = "bad_physics"
            return True
        if abs(float(relative_pos[0])) > 6.0 or abs(float(relative_pos[1])) > 6.0:
            self.fail_reason = "global_xy_escape"
            return True
        if v > 45.0 or w > 60.0:
            self.fail_reason = "global_speed_limit"
            return True
        if z < 0.05 or z > 18.0:
            self.fail_reason = "global_height_limit"
            return True

        if self.current_phase == "climb":
            if abs(relative_pos[0]) > 3.0 or abs(relative_pos[1]) > 2.0:
                self.fail_reason = "climb_xy_escape"
                return True
            return False
        if self.current_phase == "flip":
            if z < 3.5 and self.flip_progress < 0.15:
                self.fail_reason = "flip_too_low_too_early"
                return True
            if w > 70.0:
                self.fail_reason = "flip_angular_speed_limit"
                return True
            return False
        if self.current_phase in {"recovery", "hover"}:
            if v > 70.0 or w > 70.0:
                self.fail_reason = f"{self.current_phase}_speed_limit"
                return True
            if rel_dist > 1.5:
                self.fail_reason = f"{self.current_phase}_target_escape"
                return True
            return False
        return False

    def _infer_fail_reason(self, metrics):
        if self.current_phase == "flip":
            if metrics["w"] > 60.0:
                return "flip_angular_speed_limit"
            if metrics["z"] < 3.5:
                return "flip_low_altitude"
        if self.current_phase in {"recovery", "hover"}:
            if metrics["rel_dist"] > 1.5:
                return f"{self.current_phase}_target_escape"
            if metrics["v"] > 45.0 or metrics["w"] > 60.0:
                return f"{self.current_phase}_speed_limit"
        if self.current_phase == "climb":
            return "climb_fail"
        return "phase_reward_fail"

    def _update_phase(self, metrics):
        if self.fail or self.success:
            self.current_phase = "done"
            return 0.0

        if self.current_phase == "climb":
            if 9.0 <= metrics["z"] <= 10.0 and metrics["upright_score"] > 0.85:
                self.current_phase = "flip"
                self.flip_angle = 0.0
                self.flip_progress = 0.0
                self.last_step_flip_progress = 0.0
                return 50.0

        if self.current_phase == "flip":
            if self.flip_progress > 0.95 and metrics["upright_score"] > 0.85:
                self.current_phase = "recovery"
                return 150.0

        if self.current_phase == "recovery":
            if self.flip_progress > 0.95 and metrics["upright_score"] > 0.95:
                self.current_phase = "hover"
                self.hover_timer = 0.0
                return 100.0

        if self.current_phase == "hover":
            dt = self.model.opt.timestep * self.frame_skip
            self.hover_timer += dt
            if self.hover_timer >= 5.0:
                self.success = True
                self.current_phase = "done"
        return 0.0

    def launch_viewer(self):
        self.close_viewer()
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self._update_viewer_camera()
        return self.viewer

    def _update_viewer_camera(self):
        if self.viewer is None or not self.follow_camera:
            return
        try:
            pos = self.data.xpos[self.body_id]
            self.viewer.cam.lookat[:] = pos
            self.viewer.cam.distance = self.camera_distance
            self.viewer.cam.azimuth = self.camera_azimuth
            self.viewer.cam.elevation = self.camera_elevation
        except AttributeError:
            return

    def close_viewer(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


if __name__ == "__main__":
    env = HopperEnv()
    env.reset()
    viewer = env.launch_viewer()
    try:
        while viewer.is_running():
            env.step(np.array([0.0, 0.0, 0.0], dtype=np.float32))
    finally:
        env.close_viewer()
