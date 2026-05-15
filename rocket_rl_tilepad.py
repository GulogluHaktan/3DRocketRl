import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from rocket_env_tilepad import (
    FreeJointRocketTerrainLandingEnv,
    MODEL_NAME,
)

TAKEOFF_MODEL_NAME = "ppo_freejoint_takeoff"
HOVER_MODEL_NAME = "ppo_freejoint_hover"
FLIP_MODEL_NAME = "ppo_freejoint_flip"
HOVER_SECONDS = 5.0
MISSION_HOVER_HEIGHT = 3.40
FULL_FLIP_RAD = 2.0 * np.pi
RIGHT_TORQUE_BASE = 1.0
RIGHT_TORQUE_ACTION_SCALE = 0.0


def load_first_model(candidates, label):
    for model_name in candidates:
        if Path(f"{model_name}.zip").exists():
            print(f"{label} modeli yüklendi: {model_name}.zip")
            return PPO.load(model_name, device="cpu"), model_name

    names = ", ".join(f"{name}.zip" for name in candidates)
    raise FileNotFoundError(f"{label} modeli bulunamadı. Arananlar: {names}")


def load_landing_model():
    return load_first_model(
        [
            MODEL_NAME,
            "ppo_freejoint_landing_terrain",
            "ppo_freejoint_landing",
            HOVER_MODEL_NAME,
        ],
        "Landing"
    )


def load_takeoff_model():
    return load_first_model(
        [
            TAKEOFF_MODEL_NAME,
            HOVER_MODEL_NAME,
        ],
        "Takeoff"
    )


def load_hover_model():
    return load_first_model(
        [
            HOVER_MODEL_NAME,
            "ppo_hover",
        ],
        "Hover"
    )


def load_flip_model():
    return load_first_model(
        [
            FLIP_MODEL_NAME,
            TAKEOFF_MODEL_NAME,
            HOVER_MODEL_NAME,
        ],
        "Flip"
    )


def set_rocket_pose(env, pos, quat=None):
    if quat is None:
        quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    env.data.qpos[0:3] = np.asarray(pos, dtype=np.float64)
    env.data.qpos[3:7] = np.asarray(quat, dtype=np.float64)
    env.data.qvel[:] = 0.0
    env.data.xfrc_applied[:] = 0.0
    mujoco.mj_forward(env.model, env.data)


def prepare_launch_pose(env, landing_target, hover_target):
    env.target_pos = hover_target.astype(np.float32)
    env.landing_counter = 0
    set_rocket_pose(
        env,
        np.array([
            landing_target[0],
            landing_target[1],
            env.landing_terrain_height + env.landing_z_offset
        ], dtype=np.float64)
    )
    return env.get_obs()


def apply_policy_action(
    env,
    model,
    main_bias=0.62,
    main_scale=0.25,
    deterministic=True
):
    obs = env.get_obs()
    action, _ = model.predict(obs, deterministic=deterministic)
    action = np.asarray(action, dtype=np.float32)
    action = np.clip(action, env.action_space.low, env.action_space.high)

    main_thrust = float(np.clip(main_bias + main_scale * action[0], 0.0, 1.0))
    torque_x = float(action[1])
    torque_y = float(action[2])

    main_thrust, torque_x, torque_y, _ = env._apply_fuel_limits(
        main_thrust,
        torque_x,
        torque_y
    )
    env._update_engine_visuals(main_thrust, torque_x, torque_y)
    env._update_wind()

    for _ in range(env.frame_skip):
        env._apply_rocket_forces(main_thrust, torque_x, torque_y)
        mujoco.mj_step(env.model, env.data)

        if env.viewer is not None:
            env.viewer.sync()
            time.sleep(env.model.opt.timestep)

    env.step_count += 1
    return env.get_obs()


def apply_flip_policy_action(env, model, deterministic=True):
    obs = env.get_obs()
    action, _ = model.predict(obs, deterministic=deterministic)
    action = np.asarray(action, dtype=np.float32)
    action = np.clip(action, env.action_space.low, env.action_space.high)

    main_thrust = float(np.clip(0.92 + 0.08 * action[0], 0.0, 1.0))
    torque_x = float(np.clip(
        RIGHT_TORQUE_BASE + RIGHT_TORQUE_ACTION_SCALE * action[1],
        0.0,
        1.0
    ))
    torque_y = float(0.35 * action[2])

    main_thrust, torque_x, torque_y, _ = env._apply_fuel_limits(
        main_thrust,
        torque_x,
        torque_y
    )
    env._update_engine_visuals(main_thrust, torque_x, torque_y)
    env._update_wind()

    for _ in range(env.frame_skip):
        env._apply_rocket_forces(main_thrust, torque_x, torque_y)
        mujoco.mj_step(env.model, env.data)

        if env.viewer is not None:
            env.viewer.sync()
            time.sleep(env.model.opt.timestep)

    env.step_count += 1
    return env.get_obs()


def phase_failed(env, hover_target):
    pos, rot, lin_vel, ang_vel = env.get_state()
    up_vector = rot @ np.array([0.0, 0.0, 1.0])
    upright_score = float(np.dot(up_vector, np.array([0.0, 0.0, 1.0])))
    flags = env._get_foot_contact_flags()

    too_far = np.linalg.norm(pos[:2] - hover_target[:2]) > 2.5
    too_high = pos[2] > hover_target[2] + 1.4
    body_hit = flags["body_terrain_contact"] or flags["body_floor_contact"]
    upside_down = upright_score < 0.15

    return bool(too_far or too_high or body_hit or upside_down)


def run_takeoff_phase(env, model, hover_target, max_steps=360):
    env.target_pos = hover_target.astype(np.float32)
    print("Faz: takeoff")

    for _ in range(max_steps):
        if env.viewer is None or not env.viewer.is_running():
            return False

        apply_policy_action(env, model, main_bias=0.58, main_scale=0.36)
        pos, _, lin_vel, _ = env.get_state()

        if phase_failed(env, hover_target):
            return False

        if pos[2] >= hover_target[2] - 0.18 and abs(lin_vel[2]) < 1.15:
            return True

    return True


def run_hover_phase(env, model, hover_target, seconds=HOVER_SECONDS):
    env.target_pos = hover_target.astype(np.float32)
    phase_steps = int(seconds / (env.model.opt.timestep * env.frame_skip))
    print(f"Faz: hover ({seconds:.1f} saniye)")

    for _ in range(phase_steps):
        if env.viewer is None or not env.viewer.is_running():
            return False

        apply_policy_action(env, model, main_bias=0.62, main_scale=0.25)

        if phase_failed(env, hover_target):
            return False

    return True


def run_flip_phase(env, model, hover_target, max_steps=420):
    env.target_pos = hover_target.astype(np.float32)
    flip_angle = 0.0
    print("Faz: flip")

    for _ in range(max_steps):
        if env.viewer is None or not env.viewer.is_running():
            return False

        apply_flip_policy_action(env, model)
        roll_rate = max(float(env.data.qvel[3]), 0.0)
        flip_angle += roll_rate * env.model.opt.timestep * env.frame_skip

        if phase_failed(env, hover_target):
            return False

        if flip_angle >= FULL_FLIP_RAD:
            print({"flip_angle_deg": float(np.rad2deg(flip_angle))})
            return True

    print({"flip_angle_deg": float(np.rad2deg(flip_angle))})
    return flip_angle >= 0.85 * FULL_FLIP_RAD


def run_landing_phase(env, model, landing_target):
    env.target_pos = landing_target.astype(np.float32)
    env.landing_counter = 0
    obs = env.get_obs()
    print("Faz: landing")

    while env.viewer is not None and env.viewer.is_running():
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            print("Landing episode bitti:", info)
            return bool(info.get("success", False))

    return False

def train(total_timesteps):
    env = FreeJointRocketTerrainLandingEnv()
    check_env(env, warn=True)

    terrain_model_path = Path(f"{MODEL_NAME}.zip")
    pretrained_terrain_model_path = Path("ppo_freejoint_landing_terrain.zip")
    flat_landing_model_path = Path("ppo_freejoint_landing.zip")
    hover_model_path = Path("ppo_freejoint_hover.zip")

    if terrain_model_path.exists():
        print("Terrain tile-pad modeli bulundu, üstüne eğitime devam ediliyor.")
        model = PPO.load(
            MODEL_NAME,
            env=env,
            device="cpu"
        )

    elif pretrained_terrain_model_path.exists():
        print("Önceden eğitilmiş terrain modeli bulundu, onun üstünden devam ediliyor.")
        model = PPO.load(
            "ppo_freejoint_landing_terrain",
            env=env,
            device="cpu"
        )

    elif flat_landing_model_path.exists():
        print("Düz zemin landing modeli bulundu, terrain tile-pad eğitimi onun üstünden başlıyor.")
        model = PPO.load(
            "ppo_freejoint_landing",
            env=env,
            device="cpu"
        )

    elif hover_model_path.exists():
        print("Hover modeli bulundu, terrain tile-pad eğitimi hover modelinden başlıyor.")
        model = PPO.load(
            "ppo_freejoint_hover",
            env=env,
            device="cpu"
        )

    else:
        print("Model bulunamadı, sıfırdan terrain tile-pad eğitimi başlıyor.")
        model = PPO(
            policy="MlpPolicy",
            env=env,
            verbose=1,
            learning_rate=3e-4,
            n_steps=1024,
            batch_size=64,
            gamma=0.99,
            device="cpu"
        )

    model.learn(
        total_timesteps=total_timesteps,
        reset_num_timesteps=False
    )

    model.save(MODEL_NAME)
    print(f"Kaydedildi: {MODEL_NAME}.zip")


def watch_landing():
    env = FreeJointRocketTerrainLandingEnv()
    model, _ = load_landing_model()

    obs, info = env.reset_with_viewer()
    viewer = env.viewer

    try:
        while viewer.is_running():
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                print("Episode bitti:", info)
                time.sleep(1.0)

                # Full reset terrain'i ve roket başlangıcını birlikte yeniler.
                obs, info = env.reset_with_viewer()
                viewer = env.viewer

    except KeyboardInterrupt:
        print("Ctrl+C ile çıkıldı.")

    finally:
        viewer.close()
        print("Viewer kapatıldı.")


def watch_takeoff():
    env = FreeJointRocketTerrainLandingEnv()
    model, _ = load_takeoff_model()
    obs, info = env.reset_with_viewer()
    viewer = env.viewer

    landing_target = env.target_pos.copy()
    hover_target = np.array([
        landing_target[0],
        landing_target[1],
        env.landing_terrain_height + MISSION_HOVER_HEIGHT
    ], dtype=np.float32)
    prepare_launch_pose(env, landing_target, hover_target)

    try:
        while viewer.is_running():
            ok = run_takeoff_phase(env, model, hover_target)
            print("Takeoff bitti:", {"ok": ok})
            time.sleep(0.8)

            obs, info = env.reset_with_viewer()
            viewer = env.viewer
            landing_target = env.target_pos.copy()
            hover_target = np.array([
                landing_target[0],
                landing_target[1],
                env.landing_terrain_height + MISSION_HOVER_HEIGHT
            ], dtype=np.float32)
            prepare_launch_pose(env, landing_target, hover_target)

    except KeyboardInterrupt:
        print("Ctrl+C ile çıkıldı.")

    finally:
        viewer.close()
        print("Viewer kapatıldı.")


def watch_hover():
    env = FreeJointRocketTerrainLandingEnv()
    model, _ = load_hover_model()
    obs, info = env.reset_with_viewer()
    viewer = env.viewer

    landing_target = env.target_pos.copy()
    hover_target = np.array([
        landing_target[0],
        landing_target[1],
        env.landing_terrain_height + MISSION_HOVER_HEIGHT
    ], dtype=np.float32)
    env.target_pos = hover_target
    set_rocket_pose(env, hover_target)

    try:
        while viewer.is_running():
            apply_policy_action(env, model, main_bias=0.62, main_scale=0.25)

            if phase_failed(env, hover_target):
                print("Hover reset")
                time.sleep(0.8)
                obs, info = env.reset_with_viewer()
                viewer = env.viewer
                landing_target = env.target_pos.copy()
                hover_target = np.array([
                    landing_target[0],
                    landing_target[1],
                    env.landing_terrain_height + MISSION_HOVER_HEIGHT
                ], dtype=np.float32)
                env.target_pos = hover_target
                set_rocket_pose(env, hover_target)

    except KeyboardInterrupt:
        print("Ctrl+C ile çıkıldı.")

    finally:
        viewer.close()
        print("Viewer kapatıldı.")


def watch_flip():
    env = FreeJointRocketTerrainLandingEnv()
    model, _ = load_flip_model()
    obs, info = env.reset_with_viewer()
    viewer = env.viewer

    landing_target = env.target_pos.copy()
    hover_target = np.array([
        landing_target[0],
        landing_target[1],
        env.landing_terrain_height + MISSION_HOVER_HEIGHT
    ], dtype=np.float32)
    env.target_pos = hover_target
    set_rocket_pose(env, hover_target)

    try:
        while viewer.is_running():
            ok = run_flip_phase(env, model, hover_target)
            print("Flip bitti:", {"ok": ok})
            time.sleep(0.8)
            obs, info = env.reset_with_viewer()
            viewer = env.viewer
            landing_target = env.target_pos.copy()
            hover_target = np.array([
                landing_target[0],
                landing_target[1],
                env.landing_terrain_height + MISSION_HOVER_HEIGHT
            ], dtype=np.float32)
            env.target_pos = hover_target
            set_rocket_pose(env, hover_target)

    except KeyboardInterrupt:
        print("Ctrl+C ile çıkıldı.")

    finally:
        viewer.close()
        print("Viewer kapatıldı.")


def normal_mission():
    env = FreeJointRocketTerrainLandingEnv()
    takeoff_model, _ = load_takeoff_model()
    hover_model, _ = load_hover_model()
    landing_model, _ = load_landing_model()

    obs, info = env.reset_with_viewer()
    viewer = env.viewer

    try:
        while viewer.is_running():
            landing_target = env.target_pos.copy()
            hover_target = np.array([
                landing_target[0],
                landing_target[1],
                env.landing_terrain_height + MISSION_HOVER_HEIGHT
            ], dtype=np.float32)

            prepare_launch_pose(env, landing_target, hover_target)
            print({
                "mission": "start",
                "landing_cell": env.landing_cell,
                "landing_height": float(env.landing_terrain_height),
                "hover_target": hover_target.tolist(),
            })

            takeoff_ok = run_takeoff_phase(env, takeoff_model, hover_target)

            if takeoff_ok:
                hover_ok = run_hover_phase(env, hover_model, hover_target)
            else:
                hover_ok = False

            if takeoff_ok and hover_ok:
                landing_ok = run_landing_phase(env, landing_model, landing_target)
            else:
                landing_ok = False

            print({
                "mission": "end",
                "takeoff_ok": bool(takeoff_ok),
                "hover_ok": bool(hover_ok),
                "landing_ok": bool(landing_ok),
            })

            time.sleep(1.0)

            if viewer.is_running():
                obs, info = env.reset_with_viewer()
                viewer = env.viewer

    except KeyboardInterrupt:
        print("Ctrl+C ile çıkıldı.")

    finally:
        viewer.close()
        print("Viewer kapatıldı.")


def stunt_mission():
    env = FreeJointRocketTerrainLandingEnv()
    takeoff_model, _ = load_takeoff_model()
    flip_model, _ = load_flip_model()
    landing_model, _ = load_landing_model()

    obs, info = env.reset_with_viewer()
    viewer = env.viewer

    try:
        while viewer.is_running():
            landing_target = env.target_pos.copy()
            hover_target = np.array([
                landing_target[0],
                landing_target[1],
                env.landing_terrain_height + MISSION_HOVER_HEIGHT
            ], dtype=np.float32)

            prepare_launch_pose(env, landing_target, hover_target)
            print({
                "mission": "stunt_start",
                "landing_cell": env.landing_cell,
                "landing_height": float(env.landing_terrain_height),
                "hover_target": hover_target.tolist(),
            })

            takeoff_ok = run_takeoff_phase(env, takeoff_model, hover_target)

            if takeoff_ok:
                flip_ok = run_flip_phase(env, flip_model, hover_target)
            else:
                flip_ok = False

            if takeoff_ok and flip_ok:
                landing_ok = run_landing_phase(env, landing_model, landing_target)
            else:
                landing_ok = False

            print({
                "mission": "stunt_end",
                "takeoff_ok": bool(takeoff_ok),
                "flip_ok": bool(flip_ok),
                "landing_ok": bool(landing_ok),
            })

            time.sleep(1.0)

            if viewer.is_running():
                obs, info = env.reset_with_viewer()
                viewer = env.viewer

    except KeyboardInterrupt:
        print("Ctrl+C ile çıkıldı.")

    finally:
        viewer.close()
        print("Viewer kapatıldı.")


def drop_test():
    """RL ve motor kullanmadan sadece ayak/terrain temas testi."""
    env = FreeJointRocketTerrainLandingEnv()
    obs, info = env.reset()

    def place_over_tile():
        env.data.qpos[0:3] = np.array([
            env.target_pos[0],
            env.target_pos[1],
            env.target_pos[2] + 0.75
        ], dtype=np.float64)

        env.data.qpos[3:7] = np.array(
            [1.0, 0.0, 0.0, 0.0],
            dtype=np.float64
        )

        env.data.qvel[:] = 0.0
        env.data.xfrc_applied[:] = 0.0
        mujoco.mj_forward(env.model, env.data)

    place_over_tile()

    viewer = mujoco.viewer.launch_passive(env.model, env.data)
    env.viewer = viewer

    step_count = 0

    try:
        while viewer.is_running():
            step_count += 1

            env.data.xfrc_applied[:] = 0.0
            mujoco.mj_step(env.model, env.data)
            viewer.sync()
            time.sleep(env.model.opt.timestep)

            flags = env._get_foot_contact_flags()
            pos, rot, lin_vel, ang_vel = env.get_state()

            if step_count % 20 == 0 or flags["touchdown"]:
                print({
                    "step": step_count,
                    "rocket_z": float(pos[2]),
                    "target_z": float(env.target_pos[2]),
                    "landing_cell": env.landing_cell,
                    "touchdown": flags["touchdown"],
                    "foot_pad_contact": flags["foot_pad_contact"],
                    "foot_terrain_contact": flags["foot_terrain_contact"],
                    "foot_floor_contact": flags["foot_floor_contact"],
                    "num_foot_contacts": flags["num_foot_contacts"],
                    "contact_pairs": flags["contact_pairs"],
                })

            if pos[2] < env.landing_terrain_height - 2.0:
                print("Aşağı düştü, contact yok:", {
                    "rocket_z": float(pos[2]),
                    "target_z": float(env.target_pos[2]),
                    "landing_cell": env.landing_cell,
                    "contact_pairs": flags["contact_pairs"],
                })
                time.sleep(1.0)
                place_over_tile()
                step_count = 0

    except KeyboardInterrupt:
        print("Ctrl+C ile çıkıldı.")

    finally:
        viewer.close()
        print("Viewer kapatıldı.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=[
            "train",
            "normal",
            "stunt",
            "landing",
            "takeoff",
            "hover",
            "flip",
            "watch",
            "drop_test",
        ],
        default="normal"
    )

    parser.add_argument(
        "--timesteps",
        type=int,
        default=300_000
    )

    args = parser.parse_args()

    if args.mode == "train":
        train(args.timesteps)

    elif args.mode == "normal":
        normal_mission()

    elif args.mode == "stunt":
        stunt_mission()

    elif args.mode == "landing":
        watch_landing()

    elif args.mode == "takeoff":
        watch_takeoff()

    elif args.mode == "hover":
        watch_hover()

    elif args.mode == "flip":
        watch_flip()

    elif args.mode == "watch":
        watch_landing()

    elif args.mode == "drop_test":
        drop_test()
