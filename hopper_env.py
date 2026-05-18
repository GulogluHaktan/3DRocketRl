from pathlib import Path
import time

import gymnasium as gym
import mujoco
import mujoco.viewer
import numpy as np
from gymnasium import spaces


MODEL_XML = Path(__file__).with_name("hopper_default.xml")
MODEL_PREFIX = "ppo_hopper"

PAD_TOP_Z = 0.02
BODY_START_Z = 0.40
TAKEOFF_TARGET_Z = 10.0
HOVER_TARGET_Z = 10.0
LANDING_START_Z = 10.0
FLIP_TARGET_Z = 3.80
FULL_FLIP_RAD = 2.0 * np.pi

# Landing platform geometry.
# pad radius = 0.50, feet are about 0.40 m from center.
# So the rocket center should stay inside roughly 0.10 m for all feet to stay on pad.
PAD_RADIUS = 0.50
FOOT_RADIAL_SPAN = 0.40
LANDING_SAFE_XY_RADIUS = PAD_RADIUS - FOOT_RADIAL_SPAN


class HopperRocketEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, task="takeoff", landing_start_z=LANDING_START_Z):
        super().__init__()

        if task not in {"takeoff", "hover", "landing", "flip"}:
            raise ValueError(f"Unknown task: {task}")

        self.task = task
        self.model = mujoco.MjModel.from_xml_path(str(MODEL_XML))
        self.data = mujoco.MjData(self.model)

        self.body_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            "hopper"
        )
        self.thrust_site_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_SITE,
            "thrust_site"
        )

        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(18,),
            dtype=np.float32
        )

        self.max_thrust = 33.0
        self.max_tvc_angle = np.deg2rad(18.0)
        self.linear_damping = 0.08
        self.angular_damping = 0.34

        self.wind_strength = 0.08
        self.wind_turbulence = 0.018
        self.wind_xy = np.zeros(2, dtype=np.float32)
        self.wind_target_xy = np.zeros(2, dtype=np.float32)
        self.wind_change_interval = 45
        self.wind_smoothing = 0.035

        self.fuel_capacity = 1.0
        self.fuel = self.fuel_capacity
        self.fuel_burn_idle = 0.0005
        self.fuel_burn_main = 0.020
        self.fuel_burn_tvc = 0.006

        self.frame_skip = 8
        self.max_steps = 900
        self.step_count = 0
        self.success_counter = 0
        self.required_success_steps = 25
        self.landing_start_z = float(landing_start_z)
        self.flip_angle = 0.0
        self.last_measured_flip_angle = 0.0
        self.viewer = None
        self.camera_follow = True
        self.camera_follow_pause_steps = 0
        self._last_auto_camera = None

        self.target_pos = np.array([0.0, 0.0, HOVER_TARGET_Z], dtype=np.float32)

    def set_task(self, task):
        if task not in {"takeoff", "hover", "landing", "flip"}:
            raise ValueError(f"Unknown task: {task}")

        self.task = task
        self.success_counter = 0

        if task == "takeoff":
            self.target_pos = np.array([0.0, 0.0, TAKEOFF_TARGET_Z], dtype=np.float32)
        elif task == "hover":
            self.target_pos = np.array([0.0, 0.0, HOVER_TARGET_Z], dtype=np.float32)
        elif task == "landing":
            self.target_pos = np.array([0.0, 0.0, BODY_START_Z], dtype=np.float32)
        else:
            self.target_pos = np.array([0.0, 0.0, FLIP_TARGET_Z], dtype=np.float32)

        return self.get_obs()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.step_count = 0
        self.success_counter = 0
        self.flip_angle = 0.0
        self.last_measured_flip_angle = 0.0
        self.fuel = self.fuel_capacity
        self.wind_xy = self._sample_wind()
        self.wind_target_xy = self._sample_wind()

        mujoco.mj_resetData(self.model, self.data)

        if self.task == "takeoff":
            start_pos = np.array([0.0, 0.0, BODY_START_Z], dtype=np.float64)
            self.target_pos = np.array([0.0, 0.0, TAKEOFF_TARGET_Z], dtype=np.float32)
            qvel = np.zeros(6, dtype=np.float64)
        elif self.task == "hover":
            start_pos = np.array([0.0, 0.0, HOVER_TARGET_Z], dtype=np.float64)
            self.target_pos = np.array([0.0, 0.0, HOVER_TARGET_Z], dtype=np.float32)
            qvel = self.np_random.uniform(-0.04, 0.04, size=6).astype(np.float64)
        elif self.task == "landing":
            start_pos = np.array([
                self.np_random.uniform(-0.03, 0.03),
                self.np_random.uniform(-0.03, 0.03),
                self.landing_start_z
            ], dtype=np.float64)
            self.target_pos = np.array([0.0, 0.0, BODY_START_Z], dtype=np.float32)
            qvel = self.np_random.uniform(
                low=np.array([-0.01, -0.01, -0.20, -0.02, -0.02, -0.02]),
                high=np.array([0.01, 0.01, -0.05, 0.02, 0.02, 0.02])
            ).astype(np.float64)
            self.wind_xy[:] = 0.0
            self.wind_target_xy[:] = 0.0
        else:
            start_pos = np.array([
                self.np_random.uniform(-0.05, 0.05),
                self.np_random.uniform(-0.05, 0.05),
                FLIP_TARGET_Z + self.np_random.uniform(0.6, 1.0)
            ], dtype=np.float64)
            self.target_pos = np.array([0.0, 0.0, FLIP_TARGET_Z], dtype=np.float32)
            qvel = np.zeros(6, dtype=np.float64)

        self.data.qpos[0:3] = start_pos
        self.data.qpos[3:7] = self._random_small_tilt_quat(max_angle_deg=4.0)
        self.data.qvel[0:6] = qvel
        self.data.ctrl[:] = 0.0
        self.data.qfrc_applied[:] = 0.0
        self.data.xfrc_applied[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

        return self.get_obs(), {}

    def step(self, action):
        self.step_count += 1
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, self.action_space.low, self.action_space.high)

        main, yaw_ctrl, pitch_ctrl, fuel_used = self._action_to_controls(action)
        old_pos, _, _, _ = self.get_state()
        old_distance = float(np.linalg.norm(self.target_pos - old_pos))

        self._update_wind()

        previous_flip_angle = self.flip_angle

        for _ in range(self.frame_skip):
            self._apply_controls(main, yaw_ctrl, pitch_ctrl)
            mujoco.mj_step(self.model, self.data)

            if self.task == "flip":
                roll_rate = max(float(self.data.qvel[3]), 0.0)
                self.flip_angle += roll_rate * self.model.opt.timestep
                self.flip_angle = min(self.flip_angle, FULL_FLIP_RAD)

            if self.viewer is not None:
                self._update_follow_camera()
                self.viewer.sync()
                time.sleep(self.model.opt.timestep)

        pos, rot, lin_vel, ang_vel = self.get_state()
        up_vector = rot @ np.array([0.0, 0.0, 1.0])
        upright_score = float(np.dot(up_vector, np.array([0.0, 0.0, 1.0])))
        distance = float(np.linalg.norm(self.target_pos - pos))
        progress = old_distance - distance
        xy_distance = float(np.linalg.norm(pos[:2] - self.target_pos[:2]))
        speed = float(np.linalg.norm(lin_vel))
        angular_speed = float(np.linalg.norm(ang_vel))
        contact = self._contact_flags()

        reward = self._reward(
            pos=pos,
            lin_vel=lin_vel,
            ang_vel=ang_vel,
            upright_score=upright_score,
            progress=progress,
            distance=distance,
            xy_distance=xy_distance,
            speed=speed,
            angular_speed=angular_speed,
            main=main,
            yaw_ctrl=yaw_ctrl,
            pitch_ctrl=pitch_ctrl,
            fuel_used=fuel_used,
            previous_flip_angle=previous_flip_angle,
            contact=contact
        )

        success = self._success(pos, lin_vel, ang_vel, upright_score, contact)

        if success:
            self.success_counter += 1
            reward += self._success_step_bonus()
        else:
            self.success_counter = 0

        success_done = self.success_counter >= self._required_success_steps()
        crash = self._crash(pos, upright_score, contact)
        too_far = self._too_far(pos, xy_distance)
        touchdown_miss = self._touchdown_miss(pos, lin_vel, ang_vel, upright_score, contact)
        out_of_fuel = self.fuel <= 1e-4 and not success_done
        terminated = bool(
            success_done
            or crash
            or too_far
            or touchdown_miss
            or out_of_fuel
        )
        truncated = bool(self.step_count >= self.max_steps)

        if success_done:
            reward += self._success_done_bonus()
        if crash:
            reward -= self._crash_penalty()
        if too_far:
            reward -= 140.0
        if touchdown_miss:
            reward -= 45.0
        if out_of_fuel:
            reward -= 40.0

        if self.task == "landing" and too_far:
            reward = min(reward, -120.0)

        reward = self._clip_reward(reward)

        info = {
            "task": self.task,
            "success": bool(success_done),
            "crash": bool(crash),
            "too_far": bool(too_far),
            "touchdown_miss": bool(touchdown_miss),
            "out_of_fuel": bool(out_of_fuel),
            "height": float(pos[2]),
            "target_z": float(self.target_pos[2]),
            "xy_distance": float(xy_distance),
            "speed": float(speed),
            "vertical_velocity": float(lin_vel[2]),
            "vertical_speed": float(abs(lin_vel[2])),
            "descent_speed": float(max(-lin_vel[2], 0.0)),
            "horizontal_speed": float(np.linalg.norm(lin_vel[:2])),
            "angular_speed": float(angular_speed),
            "upright_score": float(upright_score),
            "fuel": float(self.fuel),
            "main": float(main),
            "yaw_ctrl_deg": float(np.rad2deg(yaw_ctrl)),
            "pitch_ctrl_deg": float(np.rad2deg(pitch_ctrl)),
            "flip_angle_deg": float(np.rad2deg(self.flip_angle)),
            "landing_start_z": float(self.landing_start_z),
            "contact_pairs": contact["pairs"],
        }

        return self.get_obs(), float(reward), terminated, truncated, info

    def _action_to_controls(self, action):
        if self.task == "landing":
            main = float(np.clip(0.40 + 0.06 * action[0], 0.0, 1.0))
        elif self.task == "takeoff":
            main = float(np.clip(0.55 + 0.40 * action[0], 0.0, 1.0))
        elif self.task == "flip":
            main = float(np.clip(0.90 + 0.10 * action[0], 0.0, 1.0))
        else:
            main = float(np.clip(0.48 + 0.22 * action[0], 0.0, 1.0))

        tvc_limit = self._task_tvc_limit()
        yaw_ctrl = float(tvc_limit * action[1])
        pitch_ctrl = float(tvc_limit * action[2])

        if self.task == "flip":
            yaw_ctrl = tvc_limit

        dt = self.model.opt.timestep * self.frame_skip
        burn_rate = (
            self.fuel_burn_idle
            + self.fuel_burn_main * main
            + self.fuel_burn_tvc * (abs(action[1]) + abs(action[2]))
        )
        fuel_needed = dt * burn_rate

        if fuel_needed > self.fuel:
            scale = self.fuel / max(fuel_needed, 1e-8)
            fuel_used = self.fuel
            self.fuel = 0.0
            return main * scale, yaw_ctrl * scale, pitch_ctrl * scale, fuel_used

        self.fuel -= fuel_needed
        return main, yaw_ctrl, pitch_ctrl, fuel_needed

    def _task_tvc_limit(self):
        if self.task == "landing":
            return np.deg2rad(2.5)
        return self.max_tvc_angle

    def _apply_controls(self, main, yaw_ctrl, pitch_ctrl):
        self.data.ctrl[0] = yaw_ctrl
        self.data.ctrl[1] = pitch_ctrl
        self.data.qfrc_applied[:] = 0.0
        self.data.xfrc_applied[:] = 0.0

        mujoco.mj_forward(self.model, self.data)

        site_pos = self.data.site_xpos[self.thrust_site_id].copy()
        site_xmat = self.data.site_xmat[self.thrust_site_id].reshape(3, 3).copy()
        thrust_dir = site_xmat @ np.array([0.0, 0.0, 1.0])
        thrust_force = thrust_dir * (main * self.max_thrust)

        pos, _, lin_vel, ang_vel = self.get_state()
        turbulence = self.np_random.normal(
            loc=0.0,
            scale=self.wind_turbulence,
            size=2
        ).astype(np.float32)
        wind_xy = self.wind_xy + turbulence
        wind_force = np.array([wind_xy[0], wind_xy[1], 0.0])
        drag_force = -self.linear_damping * lin_vel
        damping_torque = -self.angular_damping * ang_vel

        mujoco.mj_applyFT(
            self.model,
            self.data,
            thrust_force,
            np.zeros(3),
            site_pos,
            self.body_id,
            self.data.qfrc_applied
        )
        self.data.xfrc_applied[self.body_id, 0:3] = wind_force + drag_force
        self.data.xfrc_applied[self.body_id, 3:6] = damping_torque

    def _reward(
        self,
        pos,
        lin_vel,
        ang_vel,
        upright_score,
        progress,
        distance,
        xy_distance,
        speed,
        angular_speed,
        main,
        yaw_ctrl,
        pitch_ctrl,
        fuel_used,
        previous_flip_angle,
        contact,
    ):
        tvc_limit = max(self._task_tvc_limit(), 1e-8)
        tvc_usage = (abs(yaw_ctrl) + abs(pitch_ctrl)) / (2.0 * tvc_limit)
        tvc_saturation = max(abs(yaw_ctrl), abs(pitch_ctrl)) / tvc_limit
        tvc_penalty = 0.05 * tvc_usage

        if self.task == "takeoff":
            grounded_penalty = 2.5 if pos[2] < BODY_START_Z + 0.10 else 0.0
            height_fraction = np.clip(pos[2] / self.target_pos[2], 0.0, 1.0)
            return (
                25.0 * height_fraction
                + 1.8 * progress
                + 0.45 * main
                + 2.2 * upright_score
                - 1.2 * xy_distance
                - 0.45 * angular_speed
                - grounded_penalty
                - tvc_penalty
                - 1.2 * fuel_used
                - 0.05
            )

        if self.task == "hover":
            upright_reward = 1.4 * np.clip((upright_score - 0.85) / 0.15, 0.0, 1.0)
            return (
                3.0
                - 4.5 * distance
                - 0.45 * speed
                - 0.55 * angular_speed
                + 1.2 * upright_score
                + upright_reward
                - 0.04 * tvc_usage
                - 1.4 * fuel_used
                - 0.01
            )

        if self.task == "landing":
            vertical_vel = float(lin_vel[2])
            descent_speed = max(-vertical_vel, 0.0)
            horizontal_speed = float(np.linalg.norm(lin_vel[:2]))
            height_above_target = max(float(pos[2] - self.target_pos[2]), 0.0)
            height_fraction = min(
                height_above_target
                / max(self.landing_start_z - BODY_START_Z, 1e-6),
                1.0
            )
            closeness = 1.0 - height_fraction
            tilt_amount = max(1.0 - upright_score, 0.0)
            descending_gate = 1.0 if vertical_vel <= 0.05 else 0.0
            upright_reward = (
                5.0
                * descending_gate
                * np.clip((upright_score - 0.90) / 0.08, 0.0, 1.0)
            )
            upright_penalty = (
                14.0 * max(0.96 - upright_score, 0.0)
                + 80.0 * max(0.90 - upright_score, 0.0) ** 2
            )

            center_radius = 0.12
            center_score = np.clip(1.0 - xy_distance / center_radius, -3.0, 1.0)
            center_reward = 5.0 * descending_gate * center_score
            xy_penalty = 10.0 * xy_distance + 40.0 * xy_distance ** 2

            desired_descent_speed = 0.14 + 0.34 * height_fraction
            descent_error = abs(descent_speed - desired_descent_speed)
            descent_reward = 1.8 * np.clip(1.0 - descent_error / 0.55, 0.0, 1.0)
            descent_penalty = (
                4.0 * max(descent_speed - desired_descent_speed - 0.20, 0.0)
                + 18.0 * max(vertical_vel, 0.0)
            )

            horizontal_penalty = 10.0 * horizontal_speed + 8.0 * horizontal_speed ** 2
            high_penalty = 0.08 * height_above_target
            ascent_escape_penalty = 16.0 * max(float(pos[2] - self.landing_start_z), 0.0)
            high_main_penalty = 7.0 * max(main - 0.43, 0.0)
            tvc_penalty_simple = 4.0 * tvc_usage + 10.0 * tvc_saturation ** 2

            soft_touch_bonus = 4.0 if (
                contact["foot_pad"]
                and not contact["body"]
                and xy_distance < center_radius
                and descent_speed < 0.48
                and horizontal_speed < 0.34
                and angular_speed < 0.65
                and upright_score > 0.96
            ) else 0.0

            return (
                0.5
                + 0.8 * progress
                + upright_reward
                + center_reward
                + descent_reward
                + soft_touch_bonus
                - upright_penalty
                - horizontal_penalty
                - xy_penalty
                - descent_penalty
                - high_penalty
                - ascent_escape_penalty
                - high_main_penalty
                - tvc_penalty_simple
                - (2.0 + 4.0 * tilt_amount) * angular_speed
                - 0.25 * fuel_used
                - 0.02
            )

        angle_progress = max(self.flip_angle - previous_flip_angle, 0.0)
        angle_progress = min(angle_progress, 0.18)
        angle_fraction = min(self.flip_angle / FULL_FLIP_RAD, 1.0)
        return (
            22.0 * angle_progress / FULL_FLIP_RAD
            + 7.0 * angle_fraction
            + 0.18 * main
            + 0.06
            - 2.0 * abs(float(pos[2] - self.target_pos[2]))
            - 2.8 * xy_distance
            - 0.35 * max(-float(lin_vel[2]), 0.0)
            - 0.05 * tvc_usage
            - 1.2 * fuel_used
            - 0.01
        )

    def _success_step_bonus(self):
        if self.task == "takeoff":
            return 1.0
        if self.task == "hover":
            return 0.4
        if self.task == "landing":
            return 3.0
        return 1.5

    def _success_done_bonus(self):
        if self.task == "takeoff":
            return 25.0 + 5.0 * self.fuel
        if self.task == "hover":
            return 12.0 + 3.0 * self.fuel
        if self.task == "landing":
            return 260.0 + 40.0 * self.fuel
        return 180.0 + 20.0 * self.fuel

    def _crash_penalty(self):
        if self.task == "landing":
            return 90.0
        return 150.0

    def _required_success_steps(self):
        if self.task == "landing":
            return 4
        return self.required_success_steps

    def _clip_reward(self, reward):
        if self.task == "landing":
            return float(np.clip(reward, -220.0, 330.0))
        if self.task == "flip":
            return float(np.clip(reward, -220.0, 220.0))
        return float(reward)

    def _success(self, pos, lin_vel, ang_vel, upright_score, contact):
        xy_distance = float(np.linalg.norm(pos[:2] - self.target_pos[:2]))
        speed = float(np.linalg.norm(lin_vel))
        angular_speed = float(np.linalg.norm(ang_vel))

        if self.task == "takeoff":
            return (
                pos[2] >= self.target_pos[2] - 0.08 and
                xy_distance < 0.35 and
                speed < 0.90 and
                upright_score > 0.90
            )

        if self.task == "hover":
            return (
                abs(float(pos[2] - self.target_pos[2])) < 0.16 and
                xy_distance < 0.30 and
                speed < 0.45 and
                angular_speed < 0.55 and
                upright_score > 0.92
            )

        if self.task == "landing":
            return (
                contact["foot_pad"] and
                not contact.get("foot_ground", False) and
                not contact["body"] and
                xy_distance < LANDING_SAFE_XY_RADIUS and
                abs(float(lin_vel[2])) < 0.42 and
                np.linalg.norm(lin_vel[:2]) < 0.34 and
                angular_speed < 0.55 and
                upright_score > 0.90
            )

        return (
            self.flip_angle >= FULL_FLIP_RAD and
            abs(float(pos[2] - self.target_pos[2])) < 1.20 and
            xy_distance < 0.75
        )

    def _crash(self, pos, upright_score, contact):
        if self.task == "takeoff":
            return contact["body"] or upright_score < 0.20
        if self.task == "hover":
            return contact["body"] or pos[2] < 0.45 or upright_score < 0.15
        if self.task == "landing":
            return contact["body"] or upright_score < 0.15
        return (
            contact["body"]
            or pos[2] < self.target_pos[2] - 3.0
            or np.linalg.norm(self.data.qvel[3:6]) > 80.0
        )

    def _too_far(self, pos, xy_distance):
        if self.task == "landing":
            height_above_target = max(float(pos[2] - self.target_pos[2]), 0.0)
            max_xy = 0.18 + 0.025 * min(height_above_target, 8.0)
            return bool(
                xy_distance > max_xy or
                pos[2] > self.landing_start_z + 0.25
            )

        if xy_distance > 3.0:
            return True

        return pos[2] > self.target_pos[2] + 2.6

    def _touchdown_miss(self, pos, lin_vel, ang_vel, upright_score, contact):
        if self.task != "landing":
            return False

        xy_distance = float(np.linalg.norm(pos[:2] - self.target_pos[:2]))
        settled = (
            (contact["foot_pad"] or contact.get("foot_ground", False))
            and not contact["body"]
            and abs(float(lin_vel[2])) < 0.35
            and float(np.linalg.norm(lin_vel[:2])) < 0.35
            and float(np.linalg.norm(ang_vel)) < 0.55
            and upright_score > 0.88
        )

        outside_safe_pad = xy_distance >= LANDING_SAFE_XY_RADIUS
        touched_ground_not_pad = (
            contact.get("foot_ground", False)
            and not contact["foot_pad"]
        )

        return bool(settled and (outside_safe_pad or touched_ground_not_pad))

    def _contact_flags(self):
        foot_names = {"foot1_geom", "foot2_geom", "foot3_geom", "foot4_geom"}
        body_names = {"main_body", "engine"}
        pad_names = {"pad_geom"}
        ground_names = {"ground"}

        foot_pad = False
        foot_ground = False
        body = False
        body_ground = False
        pairs = []

        for k in range(self.data.ncon):
            contact = self.data.contact[k]
            name1 = mujoco.mj_id2name(
                self.model,
                mujoco.mjtObj.mjOBJ_GEOM,
                contact.geom1
            )
            name2 = mujoco.mj_id2name(
                self.model,
                mujoco.mjtObj.mjOBJ_GEOM,
                contact.geom2
            )

            if name1 is None or name2 is None:
                continue

            names = {name1, name2}
            pairs.append(f"{name1}<->{name2}")

            if len(names & body_names) > 0 and len(names & pad_names) > 0:
                body = True

            if len(names & body_names) > 0 and len(names & ground_names) > 0:
                body = True
                body_ground = True

            if len(names & foot_names) > 0 and len(names & pad_names) > 0:
                foot_pad = True

            if len(names & foot_names) > 0 and len(names & ground_names) > 0:
                foot_ground = True

        return {
            "foot_pad": bool(foot_pad),
            "foot_ground": bool(foot_ground),
            "body": bool(body),
            "body_ground": bool(body_ground),
            "pairs": pairs[:16],
        }

    def get_state(self):
        pos = self.data.xpos[self.body_id].copy()
        rot = self.data.xmat[self.body_id].reshape(3, 3).copy()
        lin_vel = self.data.qvel[0:3].copy()
        ang_vel = self.data.qvel[3:6].copy()
        return pos, rot, lin_vel, ang_vel

    def get_obs(self):
        pos, rot, lin_vel, ang_vel = self.get_state()
        up_vector = rot @ np.array([0.0, 0.0, 1.0])
        target_delta = self.target_pos - pos

        return np.concatenate([
            pos,
            lin_vel,
            up_vector,
            ang_vel,
            target_delta,
            self.wind_xy,
            np.array([self.fuel], dtype=np.float32)
        ]).astype(np.float32)

    def _sample_wind(self):
        return self.np_random.uniform(
            low=np.array([-self.wind_strength, -self.wind_strength]),
            high=np.array([self.wind_strength, self.wind_strength])
        ).astype(np.float32)

    def _update_wind(self):
        if self.step_count % self.wind_change_interval == 0:
            self.wind_target_xy = self._sample_wind()

        self.wind_xy = (
            (1.0 - self.wind_smoothing) * self.wind_xy
            + self.wind_smoothing * self.wind_target_xy
        ).astype(np.float32)

    def _random_small_tilt_quat(self, max_angle_deg=5.0):
        axis = self.np_random.normal(size=3)
        axis[2] = 0.0
        norm = np.linalg.norm(axis)

        if norm < 1e-8:
            axis = np.array([1.0, 0.0, 0.0])
        else:
            axis = axis / norm

        angle = np.deg2rad(
            self.np_random.uniform(-max_angle_deg, max_angle_deg)
        )
        half = angle / 2.0
        quat = np.array([
            np.cos(half),
            axis[0] * np.sin(half),
            axis[1] * np.sin(half),
            axis[2] * np.sin(half)
        ], dtype=np.float64)
        quat /= np.linalg.norm(quat)
        return quat

    def launch_viewer(self):
        self.close_viewer()
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self._setup_follow_camera()
        return self.viewer

    def close_viewer(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
            self._last_auto_camera = None

    def set_camera_follow(self, enabled=True):
        self.camera_follow = bool(enabled)
        self.camera_follow_pause_steps = 0
        self._last_auto_camera = None

    def _camera_snapshot(self):
        if self.viewer is None:
            return None

        return (
            self.viewer.cam.lookat.copy(),
            float(self.viewer.cam.distance),
            float(self.viewer.cam.azimuth),
            float(self.viewer.cam.elevation),
        )

    def _camera_was_moved_by_user(self):
        if self._last_auto_camera is None:
            return False

        lookat, distance, azimuth, elevation = self._camera_snapshot()
        last_lookat, last_distance, last_azimuth, last_elevation = self._last_auto_camera

        return (
            np.linalg.norm(lookat - last_lookat) > 0.08
            or abs(distance - last_distance) > 0.08
            or abs(azimuth - last_azimuth) > 1.5
            or abs(elevation - last_elevation) > 1.5
        )

    def _setup_follow_camera(self):
        if self.viewer is None:
            return

        pos, _, _, _ = self.get_state()
        self.viewer.cam.lookat[:] = pos
        self.viewer.cam.lookat[2] = max(float(pos[2]), 0.6)
        self.viewer.cam.distance = 4.5
        self.viewer.cam.azimuth = 135.0
        self.viewer.cam.elevation = -18.0
        self._last_auto_camera = self._camera_snapshot()

    def _update_follow_camera(self):
        if not self.camera_follow or self.viewer is None:
            return

        if self._camera_was_moved_by_user():
            self.camera_follow_pause_steps = 120
            self._last_auto_camera = self._camera_snapshot()
            return

        if self.camera_follow_pause_steps > 0:
            self.camera_follow_pause_steps -= 1
            self._last_auto_camera = self._camera_snapshot()
            return

        pos, _, _, _ = self.get_state()
        target = np.array([
            float(pos[0]),
            float(pos[1]),
            max(float(pos[2]), 0.6),
        ])
        self.viewer.cam.lookat[:] = (
            0.86 * self.viewer.cam.lookat
            + 0.14 * target
        )
        self._last_auto_camera = self._camera_snapshot()
