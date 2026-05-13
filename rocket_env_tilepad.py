import argparse
import time

import gymnasium as gym
import mujoco
import mujoco.viewer
import numpy as np
from gymnasium import spaces

TERRAIN_N = 13
CELL_SIZE = 0.85
TERRAIN_MIN_HEIGHT = 0.035
TERRAIN_MAX_HEIGHT = 1.05
TERRAIN_HALF_THICKNESS = TERRAIN_MAX_HEIGHT / 2.0
TERRAIN_MIN_RELIEF = 0.35
TERRAIN_MIN_STD = 0.10
FOOT_CENTER_Z = -0.40
FOOT_RADIUS = 0.04
FOOT_CONTACT_TOL = 0.004
ROCKET_TOUCHDOWN_Z_OFFSET = abs(FOOT_CENTER_Z) + FOOT_RADIUS
MODEL_NAME = "ppo_freejoint_landing_terrain_tilepad_split_smallzone"

def cell_to_xy(i, j):
    center = (TERRAIN_N - 1) / 2.0
    x = (i - center) * CELL_SIZE
    y = (j - center) * CELL_SIZE
    return np.array([x, y], dtype=np.float32)

def build_terrain_geoms(height_map, landing_cells):
    geoms = []

    for i in range(TERRAIN_N):
        for j in range(TERRAIN_N):
            x, y = cell_to_xy(i, j)
            h = float(max(height_map[i, j], TERRAIN_MIN_HEIGHT))
            z = h - TERRAIN_HALF_THICKNESS

            if (i, j) in landing_cells:
                rgba = "0.0 1.0 0.0 1"
                friction = "1.5 0.01 0.0001"
                solref = "0.025 1"
                solimp = "0.88 0.97 0.001"
                margin = "0.0"
            else:
                color_t = min(h / TERRAIN_MAX_HEIGHT, 1.0)
                r = 0.18 + 0.18 * color_t
                g = 0.14 + 0.10 * color_t
                b = 0.08 + 0.05 * color_t
                rgba = f"{r:.3f} {g:.3f} {b:.3f} 1"
                friction = "1.0 0.005 0.0001"
                solref = "0.03 1"
                solimp = "0.85 0.95 0.001"
                margin = "0.0"

            geoms.append(f'''
        <geom name="terrain_{i}_{j}"
              type="box"
              pos="{x:.6f} {y:.6f} {z:.6f}"
              size="{CELL_SIZE / 2.0:.6f} {CELL_SIZE / 2.0:.6f} {TERRAIN_HALF_THICKNESS:.6f}"
              rgba="{rgba}"
              contype="1"
              conaffinity="1"
              margin="{margin}"
              friction="{friction}"
              solref="{solref}"
              solimp="{solimp}"/>''')

    return "\n".join(geoms)

def build_xml(height_map, landing_cells):
    terrain_geoms = build_terrain_geoms(height_map, landing_cells)

    return f'''
<mujoco model="freejoint_landing_terrain_tilepad_rebuild">

    <compiler angle="degree"/>
    <option timestep="0.005"
            gravity="0 0 -9.81"
            iterations="80"
            solver="Newton"/>

    <worldbody>
        <light name="main_light" pos="0 -3 6"/>

        <camera name="fixed_camera"
                pos="0 -7 4"
                xyaxes="1 0 0 0 0.55 0.85"/>

        <!-- Sadece görsel. Collision kapalı. -->
        <geom name="floor"
              type="plane"
              pos="0 0 -0.20"
              size="6 6 0.1"
              rgba="0.18 0.18 0.18 1"
              contype="0"
              conaffinity="0"/>

        <geom name="x_axis_visual"
              type="box"
              pos="0 0 0.025"
              size="3 0.015 0.015"
              rgba="0.08 0.08 0.08 1"
              contype="0"
              conaffinity="0"/>

        <geom name="y_axis_visual"
              type="box"
              pos="0 0 0.03"
              size="0.015 3 0.015"
              rgba="0.08 0.08 0.08 1"
              contype="0"
              conaffinity="0"/>

        <geom name="z_axis_visual"
              type="box"
              pos="0 0 1.5"
              size="0.015 0.015 1.5"
              rgba="0.08 0.08 0.08 1"
              contype="0"
              conaffinity="0"/>

        <!-- Landing zone ayrı geom değil. Yeşil tile da terrain geom'u. -->
        {terrain_geoms}

        <body name="rocket" pos="0 0 2.0">
            <freejoint/>

            <!-- Ana gövde collision açık; pencere/flame gibi detaylar görsel. -->
            <geom name="rocket_body"
                  type="capsule"
                  fromto="0 0 -0.18 0 0 0.25"
                  size="0.08"
                  mass="1.0"
                  contype="1"
                  conaffinity="1"
                  margin="0.0"
                  friction="0.9 0.005 0.0001"
                  solref="0.03 1"
                  solimp="0.85 0.95 0.001"
                  rgba="0.92 0.92 0.86 1"/>

            <geom name="rocket_tip"
                  type="sphere"
                  pos="0 0 0.35"
                  size="0.09"
                  mass="0.05"
                  contype="1"
                  conaffinity="1"
                  margin="0.0"
                  friction="0.9 0.005 0.0001"
                  solref="0.03 1"
                  solimp="0.85 0.95 0.001"
                  rgba="1.0 0.45 0.08 1"/>

            <geom name="rocket_window"
                  type="sphere"
                  pos="0 -0.085 0.12"
                  size="0.035"
                  mass="0.001"
                  contype="0"
                  conaffinity="0"
                  rgba="0.1 0.55 1.0 1"/>

            <geom name="engine_nozzle"
                  type="cylinder"
                  pos="0 0 -0.28"
                  size="0.055 0.07"
                  mass="0.02"
                  contype="1"
                  conaffinity="1"
                  margin="0.0"
                  friction="0.9 0.005 0.0001"
                  solref="0.03 1"
                  solimp="0.85 0.95 0.001"
                  rgba="0.12 0.12 0.12 1"/>

            <geom name="engine_flame"
                  type="sphere"
                  pos="0 0 -0.30"
                  size="0.018"
                  mass="0.001"
                  contype="0"
                  conaffinity="0"
                  rgba="1.0 0.35 0.02 0.08"/>

            <geom name="attitude_flame_x"
                  type="sphere"
                  pos="0.14 0 -0.08"
                  size="0.02"
                  mass="0.001"
                  contype="0"
                  conaffinity="0"
                  rgba="1.0 0.35 0.02 0.0"/>

            <geom name="attitude_flame_y"
                  type="sphere"
                  pos="0 0.14 -0.08"
                  size="0.02"
                  mass="0.001"
                  contype="0"
                  conaffinity="0"
                  rgba="1.0 0.35 0.02 0.0"/>

            <!-- Görsel bacaklar. Collision kapalı. -->
            <geom name="leg_front_right"
                  type="capsule"
                  fromto="0.05 -0.05 -0.12 0.22 -0.22 -0.38"
                  size="0.012"
                  mass="0.004"
                  contype="0"
                  conaffinity="0"
                  rgba="0.18 0.18 0.18 1"/>

            <geom name="leg_front_left"
                  type="capsule"
                  fromto="-0.05 -0.05 -0.12 -0.22 -0.22 -0.38"
                  size="0.012"
                  mass="0.004"
                  contype="0"
                  conaffinity="0"
                  rgba="0.18 0.18 0.18 1"/>

            <geom name="leg_back_right"
                  type="capsule"
                  fromto="0.05 0.05 -0.12 0.22 0.22 -0.38"
                  size="0.012"
                  mass="0.004"
                  contype="0"
                  conaffinity="0"
                  rgba="0.18 0.18 0.18 1"/>

            <geom name="leg_back_left"
                  type="capsule"
                  fromto="-0.05 0.05 -0.12 -0.22 0.22 -0.38"
                  size="0.012"
                  mass="0.004"
                  contype="0"
                  conaffinity="0"
                  rgba="0.18 0.18 0.18 1"/>

            <geom name="foot_front_right"
                  type="sphere"
                  pos="0.22 -0.22 -0.40"
                  size="0.04"
                  mass="0.025"
                  rgba="0.05 0.05 0.05 1"
                  contype="1"
                  conaffinity="1"
                  margin="0.0"
                  friction="1.6 0.01 0.0001"
                  solref="0.03 1"
                  solimp="0.85 0.95 0.001"/>

            <geom name="foot_front_left"
                  type="sphere"
                  pos="-0.22 -0.22 -0.40"
                  size="0.04"
                  mass="0.025"
                  rgba="0.05 0.05 0.05 1"
                  contype="1"
                  conaffinity="1"
                  margin="0.0"
                  friction="1.6 0.01 0.0001"
                  solref="0.03 1"
                  solimp="0.85 0.95 0.001"/>

            <geom name="foot_back_right"
                  type="sphere"
                  pos="0.22 0.22 -0.40"
                  size="0.04"
                  mass="0.025"
                  rgba="0.05 0.05 0.05 1"
                  contype="1"
                  conaffinity="1"
                  margin="0.0"
                  friction="1.6 0.01 0.0001"
                  solref="0.03 1"
                  solimp="0.85 0.95 0.001"/>

            <geom name="foot_back_left"
                  type="sphere"
                  pos="-0.22 0.22 -0.40"
                  size="0.04"
                  mass="0.025"
                  rgba="0.05 0.05 0.05 1"
                  contype="1"
                  conaffinity="1"
                  margin="0.0"
                  friction="1.6 0.01 0.0001"
                  solref="0.03 1"
                  solimp="0.85 0.95 0.001"/>
        </body>
    </worldbody>
</mujoco>
'''

class FreeJointRocketTerrainLandingEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self):
        super().__init__()

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

        self.terrain_n = TERRAIN_N
        self.cell_size = CELL_SIZE

        self.height_map = np.zeros((TERRAIN_N, TERRAIN_N), dtype=np.float32)
        self.landing_cell = (TERRAIN_N // 2, TERRAIN_N // 2)
        self.landing_pad_geom_names = set()
        self.landing_terrain_height = TERRAIN_MIN_HEIGHT

        # Body z where the bottom of the foot spheres exactly touches terrain.
        self.landing_z_offset = ROCKET_TOUCHDOWN_Z_OFFSET
        self.target_pos = np.array([0.0, 0.0, self.landing_z_offset], dtype=np.float32)

        self.max_main_thrust = 18.0
        self.max_torque = 0.45

        self.angular_damping = 0.8
        self.linear_damping = 0.08

        self.wind_strength = 0.18
        self.wind_turbulence = 0.035
        self.wind_xy = np.zeros(2, dtype=np.float32)
        self.wind_target_xy = np.zeros(2, dtype=np.float32)
        self.wind_change_interval = 40
        self.wind_smoothing = 0.04

        self.fuel = 1.0
        self.fuel_capacity = 1.0
        self.fuel_burn_idle = 0.0010
        self.fuel_burn_main = 0.030
        self.fuel_burn_torque = 0.012

        self.frame_skip = 8
        self.max_steps = 650
        self.step_count = 0

        self.landing_counter = 0
        self.required_landing_steps = 30

        self.viewer = None

        self._rebuild_model()

    def _landing_cells_for_xml(self):
        cells = set()

        for name in self.landing_pad_geom_names:
            # name format: terrain_i_j
            parts = name.split("_")
            if len(parts) == 3:
                cells.add((int(parts[1]), int(parts[2])))

        if len(cells) == 0:
            cells.add(self.landing_cell)

        return cells

    def _rebuild_model(self):
        xml = build_xml(self.height_map, self._landing_cells_for_xml())

        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)

        self.rocket_body_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            "rocket"
        )

        self.flame_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_GEOM,
            "engine_flame"
        )

        self.attitude_x_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_GEOM,
            "attitude_flame_x"
        )

        self.attitude_y_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_GEOM,
            "attitude_flame_y"
        )

        self.foot_geom_ids = {
            name: mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_GEOM,
                name
            )
            for name in [
                "foot_front_right",
                "foot_front_left",
                "foot_back_right",
                "foot_back_left",
            ]
        }

        self.terrain_geom_ids = {}
        for i in range(self.terrain_n):
            for j in range(self.terrain_n):
                name = f"terrain_{i}_{j}"
                self.terrain_geom_ids[(i, j)] = mujoco.mj_name2id(
                    self.model,
                    mujoco.mjtObj.mjOBJ_GEOM,
                    name
                )

        mujoco.mj_setConst(self.model, self.data)

    def close_viewer(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def launch_viewer(self):
        self.close_viewer()
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        return self.viewer

    def reset_with_viewer(self, seed=None, options=None):
        obs, info = self.reset(seed=seed, options=options)

        if self.viewer is None or not self.viewer.is_running():
            self.launch_viewer()
        else:
            self.launch_viewer()

        return obs, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.step_count = 0
        self.landing_counter = 0
        self.fuel = self.fuel_capacity

        self.wind_xy = self._sample_wind()
        self.wind_target_xy = self._sample_wind()

        self.height_map, landing_i, landing_j = self._generate_terrain()

        landing_height = float(max(
            self.height_map[landing_i, landing_j],
            TERRAIN_MIN_HEIGHT
        ))

        # Landing zone = terrain'in parçası.
        # Tek tile çok küçük kaldığı için roketin ayak açıklığına uygun küçük patch yapıyoruz.
        # Ayrı geom yok; sadece bu terrain hücreleri yeşil.
        landing_radius = 0

        self.landing_cell = (landing_i, landing_j)
        self.landing_terrain_height = landing_height
        self.landing_pad_geom_names = set()

        for di in range(-landing_radius, landing_radius + 1):
            for dj in range(-landing_radius, landing_radius + 1):
                ni = landing_i + di
                nj = landing_j + dj

                if 0 <= ni < self.terrain_n and 0 <= nj < self.terrain_n:
                    self.height_map[ni, nj] = landing_height
                    self.landing_pad_geom_names.add(f"terrain_{ni}_{nj}")

        target_xy = cell_to_xy(landing_i, landing_j)

        self.target_pos = np.array([
            target_xy[0],
            target_xy[1],
            self.landing_terrain_height + self.landing_z_offset
        ], dtype=np.float32)

        self._rebuild_model()
        mujoco.mj_resetData(self.model, self.data)

        start_pos = np.array([
            target_xy[0] + self.np_random.uniform(-0.35, 0.35),
            target_xy[1] + self.np_random.uniform(-0.35, 0.35),
            self.landing_terrain_height + self.np_random.uniform(1.8, 2.25)
        ], dtype=np.float32)

        start_quat = self._random_small_tilt_quat(max_angle_deg=8.0)

        self.data.qpos[0:3] = start_pos
        self.data.qpos[3:7] = start_quat

        self.data.qvel[0:3] = self.np_random.uniform(
            low=np.array([-0.15, -0.15, -0.15]),
            high=np.array([0.15, 0.15, 0.15])
        ).astype(np.float64)

        self.data.qvel[3:6] = self.np_random.uniform(
            low=np.array([-0.25, -0.25, -0.10]),
            high=np.array([0.25, 0.25, 0.10])
        ).astype(np.float64)

        self.data.xfrc_applied[:] = 0.0

        mujoco.mj_forward(self.model, self.data)

        return self.get_obs(), {}

    def run_empty(self, sleep=True):
        """Run the environment without RL or control forces."""
        obs, info = self.reset_with_viewer()
        print(self._reset_debug_info("Reset"))

        try:
            while self.viewer is not None and self.viewer.is_running():
                self.step_count += 1
                self.data.xfrc_applied[:] = 0.0
                self._update_engine_visuals(0.0, 0.0, 0.0)
                mujoco.mj_step(self.model, self.data)
                self.viewer.sync()

                if sleep:
                    time.sleep(self.model.opt.timestep)

                pos, _, _, _ = self.get_state()
                terrain_under_rocket, _ = self._terrain_height_at_xy(pos[:2])

                if terrain_under_rocket is None:
                    terrain_under_rocket = self.landing_terrain_height

                too_low = pos[2] < terrain_under_rocket - 1.0
                timed_out = self.step_count >= self.max_steps

                if too_low or timed_out:
                    print(self._reset_debug_info("Episode bitti"))
                    time.sleep(0.5)
                    obs, info = self.reset_rocket_on_same_terrain()
                    self.viewer.sync()
                    print(self._reset_debug_info("Roket reset"))

        except KeyboardInterrupt:
            print("Ctrl+C ile çıkıldı.")

        finally:
            self.close_viewer()
            print("Viewer kapatıldı.")

    def _reset_debug_info(self, prefix):
        contact_flags = self._get_foot_contact_flags()
        foot_samples = self._foot_ground_samples()
        max_penetration = 0.0

        if len(foot_samples) > 0:
            max_penetration = max(
                sample["penetration"]
                for sample in foot_samples
            )

        return {
            "event": prefix,
            "landing_cell": self.landing_cell,
            "landing_height": round(float(self.landing_terrain_height), 3),
            "terrain_max": round(float(self.height_map.max()), 3),
            "rocket_start_z": round(float(self.data.qpos[2]), 3),
            "max_foot_penetration": round(float(max_penetration), 4),
            "num_contacts": int(contact_flags["num_foot_contacts"]),
            "contact_pairs": contact_flags["contact_pairs"][:4],
        }

    def reset_rocket_on_same_terrain(self):
        """Viewer kapanmadan aynı terrain üstünde sadece roketi tekrar başlatır."""
        self.step_count = 0
        self.landing_counter = 0
        self.fuel = self.fuel_capacity

        self.wind_xy = self._sample_wind()
        self.wind_target_xy = self._sample_wind()

        mujoco.mj_resetData(self.model, self.data)

        self.data.qpos[0:3] = np.array([
            self.target_pos[0] + self.np_random.uniform(-0.30, 0.30),
            self.target_pos[1] + self.np_random.uniform(-0.30, 0.30),
            self.target_pos[2] + self.np_random.uniform(1.2, 1.8)
        ], dtype=np.float64)

        self.data.qpos[3:7] = self._random_small_tilt_quat(max_angle_deg=8.0)

        self.data.qvel[0:3] = self.np_random.uniform(
            low=np.array([-0.15, -0.15, -0.15]),
            high=np.array([0.15, 0.15, 0.15])
        ).astype(np.float64)

        self.data.qvel[3:6] = self.np_random.uniform(
            low=np.array([-0.25, -0.25, -0.10]),
            high=np.array([0.25, 0.25, 0.10])
        ).astype(np.float64)

        self.data.xfrc_applied[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

        return self.get_obs(), {}

    def step(self, action):
        self.step_count += 1

        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, self.action_space.low, self.action_space.high)

        main_thrust = float(np.clip(0.62 + 0.25 * action[0], 0.0, 1.0))
        torque_x = float(action[1])
        torque_y = float(action[2])

        main_thrust, torque_x, torque_y, fuel_used = self._apply_fuel_limits(
            main_thrust,
            torque_x,
            torque_y
        )

        self._update_engine_visuals(main_thrust, torque_x, torque_y)

        old_pos, _, _, _ = self.get_state()
        old_distance = np.linalg.norm(self.target_pos - old_pos)

        self._update_wind()

        for _ in range(self.frame_skip):
            self._apply_rocket_forces(main_thrust, torque_x, torque_y)
            mujoco.mj_step(self.model, self.data)

            if self.viewer is not None:
                self.viewer.sync()
                time.sleep(self.model.opt.timestep)

        pos, rot, lin_vel, ang_vel = self.get_state()

        up_vector = rot @ np.array([0.0, 0.0, 1.0])
        upright_score = float(np.dot(up_vector, np.array([0.0, 0.0, 1.0])))

        distance = np.linalg.norm(self.target_pos - pos)
        progress = old_distance - distance

        speed = np.linalg.norm(lin_vel)
        angular_speed = np.linalg.norm(ang_vel)

        reward = self._compute_reward(
            progress=progress,
            distance=distance,
            speed=speed,
            angular_speed=angular_speed,
            upright_score=upright_score,
            fuel_used=fuel_used
        )

        xy_distance = np.linalg.norm(pos[:2] - self.target_pos[:2])
        vertical_speed = abs(lin_vel[2])
        horizontal_speed = np.linalg.norm(lin_vel[:2])

        contact_flags = self._get_foot_contact_flags()

        touchdown = contact_flags["touchdown"]
        foot_pad_contact = contact_flags["foot_pad_contact"]
        foot_terrain_contact = contact_flags["foot_terrain_contact"]
        foot_floor_contact = contact_flags["foot_floor_contact"]
        body_terrain_contact = contact_flags["body_terrain_contact"]
        body_floor_contact = contact_flags["body_floor_contact"]
        num_foot_contacts = contact_flags["num_foot_contacts"]

        terrain_under_rocket, _ = self._terrain_height_at_xy(pos[:2])

        if terrain_under_rocket is None:
            terrain_under_rocket = self.landing_terrain_height

        local_touchdown_z = terrain_under_rocket + self.landing_z_offset
        low_enough = pos[2] <= local_touchdown_z + 0.06
        landed = bool(touchdown)

        safe_landing_state = (
            foot_pad_contact and
            not foot_terrain_contact and
            xy_distance < 0.30 and
            vertical_speed < 0.48 and
            horizontal_speed < 0.45 and
            angular_speed < 0.60 and
            upright_score > 0.90
        )

        # Yeşil tile dışındaki terrain'e ayak değerse kötü, ama anında episode bitmesin.
        wrong_tile_touch = bool(foot_terrain_contact and not foot_pad_contact)

        if wrong_tile_touch:
            reward -= 25.0

        hard_crash = (
            foot_floor_contact or
            body_terrain_contact or
            body_floor_contact or
            (
                landed and (
                    vertical_speed > 1.25 or
                    horizontal_speed > 1.05 or
                    upright_score < 0.55
                )
            )
        )

        bad_touchdown = landed and not safe_landing_state

        if safe_landing_state:
            self.landing_counter += 1
            reward += 1.0
        else:
            self.landing_counter = 0

        success = self.landing_counter >= self.required_landing_steps
        crash = hard_crash

        out_of_fuel = self.fuel <= 1e-4 and pos[2] > self.target_pos[2] + 0.20

        too_far = (
            np.linalg.norm(pos[:2] - self.target_pos[:2]) > 3.0 or
            pos[2] > self.target_pos[2] + 3.0
        )

        upside_down = upright_score < 0.15

        terminated = bool(
            success or
            crash or
            too_far or
            upside_down or
            out_of_fuel
        )

        truncated = bool(self.step_count >= self.max_steps)

        if success:
            reward += 210.0 + 25.0 * self.fuel

        if crash:
            reward -= 160.0

        if too_far:
            reward -= 60.0

        if upside_down:
            reward -= 80.0

        if out_of_fuel:
            reward -= 80.0

        info = {
            "distance": float(distance),
            "xy_distance": float(xy_distance),
            "speed": float(speed),
            "vertical_speed": float(vertical_speed),
            "horizontal_speed": float(horizontal_speed),
            "angular_speed": float(angular_speed),
            "upright_score": float(upright_score),
            "fuel": float(self.fuel),
            "fuel_used": float(fuel_used),
            "success": bool(success),
            "crash": bool(crash),
            "too_far": bool(too_far),
            "upside_down": bool(upside_down),
            "out_of_fuel": bool(out_of_fuel),
            "landed": bool(landed),
            "touchdown": bool(touchdown),
            "low_enough": bool(low_enough),
            "foot_pad_contact": bool(foot_pad_contact),
            "foot_terrain_contact": bool(foot_terrain_contact),
            "foot_floor_contact": bool(foot_floor_contact),
            "body_terrain_contact": bool(body_terrain_contact),
            "body_floor_contact": bool(body_floor_contact),
            "num_foot_contacts": int(num_foot_contacts),
            "wrong_tile_touch": bool(wrong_tile_touch),
            "contact_pairs": contact_flags["contact_pairs"],
            "landing_counter": int(self.landing_counter),
            "safe_landing_state": bool(safe_landing_state),
            "bad_touchdown": bool(bad_touchdown),
            "terrain_height": float(self.landing_terrain_height),
            "terrain_under_rocket": float(terrain_under_rocket),
            "landing_tile": str(self.landing_cell),
            "rocket_z": float(pos[2]),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
        }

        return self.get_obs(), float(reward), terminated, truncated, info

    def _generate_terrain(self):
        n = self.terrain_n

        for _ in range(8):
            height_map = np.zeros((n, n), dtype=np.float32)

            num_hills = int(self.np_random.integers(8, 15))

            for _ in range(num_hills):
                ci = int(self.np_random.integers(1, n - 1))
                cj = int(self.np_random.integers(1, n - 1))

                hill_height = float(self.np_random.uniform(0.12, 0.90))
                radius = float(self.np_random.uniform(0.9, 3.1))

                for i in range(n):
                    for j in range(n):
                        d = np.sqrt((i - ci) ** 2 + (j - cj) ** 2)

                        if d < radius:
                            falloff = 0.5 * (1.0 + np.cos(np.pi * d / radius))
                            height_map[i, j] += hill_height * falloff

            roughness = self.np_random.uniform(0.0, 0.09, size=(n, n))
            height_map += roughness.astype(np.float32)
            height_map = np.clip(height_map, 0.0, TERRAIN_MAX_HEIGHT).astype(np.float32)
            height_map = self._smooth_height_map(height_map)
            height_map = np.clip(height_map, 0.0, TERRAIN_MAX_HEIGHT).astype(np.float32)

            relief = float(height_map.max() - height_map.min())
            std = float(height_map.std())

            if relief >= TERRAIN_MIN_RELIEF and std >= TERRAIN_MIN_STD:
                break

        landing_i, landing_j = self._choose_landing_cell(height_map)

        return height_map, landing_i, landing_j

    def _smooth_height_map(self, height_map):
        n = height_map.shape[0]
        new_map = height_map.copy()

        for i in range(n):
            for j in range(n):
                values = []

                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        ni = i + di
                        nj = j + dj

                        if 0 <= ni < n and 0 <= nj < n:
                            values.append(height_map[ni, nj])

                new_map[i, j] = np.mean(values)

        return new_map.astype(np.float32)

    def _choose_landing_cell(self, height_map):
        n = height_map.shape[0]
        candidates = []

        for i in range(2, n - 2):
            for j in range(2, n - 2):
                h = float(height_map[i, j])

                if h < 0.70:
                    candidates.append((i, j))

        if len(candidates) == 0:
            return n // 2, n // 2

        idx = int(self.np_random.integers(0, len(candidates)))
        return candidates[idx]

    def _compute_reward(
        self,
        progress,
        distance,
        speed,
        angular_speed,
        upright_score,
        fuel_used
    ):
        pos, rot, lin_vel, ang_vel = self.get_state()

        xy_distance = np.linalg.norm(pos[:2] - self.target_pos[:2])
        z_error = abs(pos[2] - self.target_pos[2])

        vertical_speed = abs(lin_vel[2])
        horizontal_speed = np.linalg.norm(lin_vel[:2])

        reward = 0.0

        reward += 5.0 * progress
        reward -= 4.8 * xy_distance
        reward -= 0.50 * z_error
        reward -= 0.45 * speed
        reward -= 0.55 * angular_speed
        reward += 2.0 * upright_score
        reward -= 4.0 * (1.0 - upright_score)

        if pos[2] < self.target_pos[2] + 0.85:
            reward -= 3.0 * vertical_speed
            reward -= 1.3 * horizontal_speed

        reward -= 1.5 * fuel_used
        reward -= 0.01

        return reward

    def get_state(self):
        pos = self.data.xpos[self.rocket_body_id].copy()
        rot = self.data.xmat[self.rocket_body_id].reshape(3, 3).copy()
        lin_vel = self.data.qvel[0:3].copy()
        ang_vel = self.data.qvel[3:6].copy()

        return pos, rot, lin_vel, ang_vel

    def get_obs(self):
        pos, rot, lin_vel, ang_vel = self.get_state()
        up_vector = rot @ np.array([0.0, 0.0, 1.0])
        target_delta = self.target_pos - pos

        obs = np.concatenate([
            pos,
            lin_vel,
            up_vector,
            ang_vel,
            target_delta,
            self.wind_xy,
            np.array([self.fuel], dtype=np.float32)
        ]).astype(np.float32)

        return obs

    def _apply_rocket_forces(self, main_thrust, torque_x, torque_y):
        pos, rot, lin_vel, ang_vel = self.get_state()

        local_z_world = rot @ np.array([0.0, 0.0, 1.0])
        thrust_force = local_z_world * (main_thrust * self.max_main_thrust)

        turbulence = self.np_random.normal(
            loc=0.0,
            scale=self.wind_turbulence,
            size=2
        ).astype(np.float32)

        wind_xy = self.wind_xy + turbulence
        wind_force = np.array([wind_xy[0], wind_xy[1], 0.0])

        drag_force = -self.linear_damping * lin_vel

        total_force = thrust_force + wind_force + drag_force

        torque_body = np.array([
            torque_x * self.max_torque,
            torque_y * self.max_torque,
            0.0
        ])

        torque_world = rot @ torque_body
        damping_torque = -self.angular_damping * ang_vel
        total_torque = torque_world + damping_torque

        self.data.xfrc_applied[:] = 0.0
        self.data.xfrc_applied[self.rocket_body_id, 0:3] = total_force
        self.data.xfrc_applied[self.rocket_body_id, 3:6] = total_torque

    def _apply_fuel_limits(self, main_thrust, torque_x, torque_y):
        if self.fuel <= 0.0:
            return 0.0, 0.0, 0.0, 0.0

        dt = self.model.opt.timestep * self.frame_skip

        burn_rate = (
            self.fuel_burn_idle
            + self.fuel_burn_main * max(main_thrust, 0.0)
            + self.fuel_burn_torque * (abs(torque_x) + abs(torque_y))
        )

        fuel_needed = dt * burn_rate

        if fuel_needed <= self.fuel:
            self.fuel -= fuel_needed
            return main_thrust, torque_x, torque_y, fuel_needed

        scale = self.fuel / max(fuel_needed, 1e-8)
        fuel_used = self.fuel
        self.fuel = 0.0

        return main_thrust * scale, torque_x * scale, torque_y * scale, fuel_used

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

    def _xy_to_cell(self, xy):
        center = (self.terrain_n - 1) / 2.0
        i = int(np.floor(float(xy[0]) / self.cell_size + center + 0.5))
        j = int(np.floor(float(xy[1]) / self.cell_size + center + 0.5))

        if i < 0 or i >= self.terrain_n or j < 0 or j >= self.terrain_n:
            return None

        return i, j

    def _terrain_height_at_xy(self, xy):
        cell = self._xy_to_cell(xy)

        if cell is None:
            return None, None

        return float(max(self.height_map[cell], TERRAIN_MIN_HEIGHT)), cell

    def _foot_ground_samples(self):
        samples = []

        for name, geom_id in self.foot_geom_ids.items():
            foot_pos = self.data.geom_xpos[geom_id].copy()
            terrain_height, cell = self._terrain_height_at_xy(foot_pos[:2])

            if terrain_height is None:
                continue

            foot_bottom_z = float(foot_pos[2] - FOOT_RADIUS)
            penetration = terrain_height - foot_bottom_z

            samples.append({
                "name": name,
                "cell": cell,
                "terrain_height": terrain_height,
                "foot_bottom_z": foot_bottom_z,
                "penetration": penetration,
            })

        return samples

    def _get_foot_contact_flags(self):
        foot_geoms = {
            "foot_front_right",
            "foot_front_left",
            "foot_back_right",
            "foot_back_left",
        }
        body_geoms = {
            "rocket_body",
            "rocket_tip",
            "engine_nozzle",
        }

        foot_pad_contact = False
        foot_terrain_contact = False
        foot_floor_contact = False
        body_terrain_contact = False
        body_floor_contact = False
        num_foot_contacts = 0
        contact_pairs = []

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
            contact_pairs.append(f"{name1}<->{name2}")

            has_foot = len(names & foot_geoms) > 0
            has_body = len(names & body_geoms) > 0

            if has_body:
                if "floor" in names:
                    body_floor_contact = True

                if any(name.startswith("terrain_") for name in names):
                    body_terrain_contact = True

            if not has_foot:
                continue

            if "floor" in names:
                foot_floor_contact = True
                num_foot_contacts += 1
                continue

            terrain_names = {
                name
                for name in names
                if name.startswith("terrain_")
            }

            if len(terrain_names) == 0:
                continue

            num_foot_contacts += 1

            if len(terrain_names & self.landing_pad_geom_names) > 0:
                foot_pad_contact = True
            else:
                foot_terrain_contact = True

        for sample in self._foot_ground_samples():
            if sample["penetration"] < -FOOT_CONTACT_TOL:
                continue

            num_foot_contacts += 1
            terrain_name = f"terrain_{sample['cell'][0]}_{sample['cell'][1]}"
            contact_pairs.append(f"{sample['name']}<->{terrain_name}:height")

            if terrain_name in self.landing_pad_geom_names:
                foot_pad_contact = True
            else:
                foot_terrain_contact = True

        touchdown = foot_pad_contact or foot_terrain_contact or foot_floor_contact

        return {
            "touchdown": bool(touchdown),
            "foot_pad_contact": bool(foot_pad_contact),
            "foot_terrain_contact": bool(foot_terrain_contact),
            "foot_floor_contact": bool(foot_floor_contact),
            "body_terrain_contact": bool(body_terrain_contact),
            "body_floor_contact": bool(body_floor_contact),
            "num_foot_contacts": int(num_foot_contacts),
            "contact_pairs": contact_pairs[:16],
        }

    def _update_engine_visuals(self, main_thrust, torque_x, torque_y):
        main = float(np.clip(main_thrust, 0.0, 1.0))
        tx = abs(float(torque_x))
        ty = abs(float(torque_y))

        self.model.geom_size[self.flame_id, 0] = 0.010 + 0.030 * main
        self.model.geom_rgba[self.flame_id] = np.array([
            1.0,
            0.25 + 0.45 * main,
            0.02,
            0.10 + 0.70 * main
        ])

        self.model.geom_size[self.attitude_x_id, 0] = 0.015 + 0.05 * tx
        self.model.geom_rgba[self.attitude_x_id] = np.array([
            1.0,
            0.30 + 0.50 * tx,
            0.02,
            0.05 + 0.70 * tx
        ])

        self.model.geom_size[self.attitude_y_id, 0] = 0.015 + 0.05 * ty
        self.model.geom_rgba[self.attitude_y_id] = np.array([
            1.0,
            0.30 + 0.50 * ty,
            0.02,
            0.05 + 0.70 * ty
        ])

    def _random_small_tilt_quat(self, max_angle_deg=10.0):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["empty"],
        default="empty"
    )
    parser.add_argument(
        "--no-sleep",
        action="store_true",
        help="Simülasyonu gerçek zamana yakın yavaşlatmadan çalıştır."
    )

    args = parser.parse_args()

    env = FreeJointRocketTerrainLandingEnv()

    if args.mode == "empty":
        env.run_empty(sleep=not args.no_sleep)


if __name__ == "__main__":
    main()
