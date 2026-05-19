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
    "body_axis_x",
    "body_axis_y",
    "body_axis_z",
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
    "top_angle_to_vertical",
)


class HopperEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, start_z: float = 10.0, target_pos=(0.0, 0.0, 0.40)):
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
        self.target_pos = np.asarray(target_pos, dtype=np.float64)
        self.max_thrust = 33.0
        self.max_tvc_angle = np.deg2rad(10.0)
        self.linear_damping = 0.08
        self.angular_damping = 0.34
        self.frame_skip = 8
        self.max_steps = 1200
        self.step_count = 0
        self.main_motor_power = 0.0
        self.viewer = None

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
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[0:3] = np.array([0.0, 0.0, self.start_z], dtype=np.float64)
        self.data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self.data.qvel[0:6] = np.array([0.0, 0.0, -0.1, 0.0, 0.0, 0.0], dtype=np.float64)
        self.data.ctrl[:] = 0.0
        self.data.qfrc_applied[:] = 0.0
        self.data.xfrc_applied[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        return self.get_observation(), {}

    def step(self, action):
        self.step_count += 1
        action = np.clip(np.asarray(action, dtype=np.float32), self.action_space.low, self.action_space.high)
        main = float(action[0])
        yaw_ctrl = float(self.max_tvc_angle * action[1])
        pitch_ctrl = float(self.max_tvc_angle * action[2])
        self.main_motor_power = main

        for _ in range(self.frame_skip):
            self._apply_controls(main, yaw_ctrl, pitch_ctrl)
            mujoco.mj_step(self.model, self.data)
            if self.viewer is not None:
                self.viewer.sync()
                time.sleep(self.model.opt.timestep)

        obs = self.get_observation()
        reward = self._reward()
        terminated = self._is_terminal()
        truncated = self.step_count >= self.max_steps
        return obs, reward, terminated, truncated, self.get_info()

    def get_observation(self):
        pos, rot, lin_vel, ang_vel = self.get_state()
        body_axis = rot @ np.array([0.0, 0.0, 1.0])
        bottom_point = pos + rot @ np.array([0.0, 0.0, -0.355])
        target_delta = self.target_pos - bottom_point
        tvc_yaw = float(self.data.qpos[self.yaw_qpos_id])
        tvc_pitch = float(self.data.qpos[self.pitch_qpos_id])
        tvc_yaw_vel = float(self.data.qvel[self.yaw_qvel_id])
        tvc_pitch_vel = float(self.data.qvel[self.pitch_qvel_id])
        contacts = self.foot_contacts()
        top_angle = float(np.arccos(np.clip(body_axis[2], -1.0, 1.0)))

        return np.array([
            *target_delta,
            *body_axis,
            bottom_point[2],
            *lin_vel,
            *ang_vel,
            tvc_yaw,
            tvc_pitch,
            tvc_yaw_vel,
            tvc_pitch_vel,
            *contacts,
            self.main_motor_power,
            top_angle,
        ], dtype=np.float32)

    def get_info(self):
        return dict(zip(OBSERVATION_NAMES, self.get_observation().tolist()))

    def get_state(self):
        pos = self.data.xpos[self.body_id].copy()
        rot = self.data.xmat[self.body_id].reshape(3, 3).copy()
        lin_vel = self.data.qvel[0:3].copy()
        ang_vel = self.data.qvel[3:6].copy()
        return pos, rot, lin_vel, ang_vel

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
        self.data.ctrl[0] = np.rad2deg(yaw_ctrl)
        self.data.ctrl[1] = np.rad2deg(pitch_ctrl)
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

    def _reward(self):
        pos, rot, lin_vel, ang_vel = self.get_state()
        body_axis = rot @ np.array([0.0, 0.0, 1.0])
        xy_distance = float(np.linalg.norm(pos[:2] - self.target_pos[:2]))
        horizontal_speed = float(np.linalg.norm(lin_vel[:2]))
        vertical_speed = abs(float(lin_vel[2]))
        top_angle = float(np.arccos(np.clip(body_axis[2], -1.0, 1.0)))
        target_height_error = abs(float(pos[2] - self.target_pos[2]))
        contact_count = float(np.sum(self.foot_contacts()))

        return float(
            3.0 * max(1.0 - top_angle / 0.25, -2.0)
            + 2.0 * max(1.0 - xy_distance / 0.25, -3.0)
            - 4.0 * horizontal_speed
            - 0.8 * vertical_speed
            - 0.2 * target_height_error
            - 0.3 * np.linalg.norm(ang_vel)
            + 2.0 * (contact_count >= 2.0)
        )

    def _is_terminal(self):
        pos, rot, lin_vel, _ = self.get_state()
        top_angle = float(np.arccos(np.clip((rot @ np.array([0.0, 0.0, 1.0]))[2], -1.0, 1.0)))
        xy_distance = float(np.linalg.norm(pos[:2] - self.target_pos[:2]))
        if pos[2] < 0.20 or pos[2] > self.start_z + 2.0:
            return True
        if xy_distance > 2.5 or top_angle > np.deg2rad(75.0):
            return True
        if np.sum(self.foot_contacts()) >= 2 and abs(lin_vel[2]) < 0.4 and xy_distance < 0.25:
            return True
        return False

    def launch_viewer(self):
        self.close_viewer()
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        return self.viewer

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
