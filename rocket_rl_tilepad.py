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


def watch():
    env = FreeJointRocketTerrainLandingEnv()

    if Path(f"{MODEL_NAME}.zip").exists():
        model = PPO.load(MODEL_NAME, device="cpu")
    elif Path("ppo_freejoint_landing_terrain.zip").exists():
        print("Tile-pad modeli yok, terrain modeli ile izleniyor.")
        model = PPO.load("ppo_freejoint_landing_terrain", device="cpu")
    elif Path("ppo_freejoint_landing.zip").exists():
        print("Terrain tile-pad modeli yok, düz zemin landing modeli ile izleniyor.")
        model = PPO.load("ppo_freejoint_landing", device="cpu")
    else:
        print("Terrain/landing modeli yok, hover modeli ile izleniyor.")
        model = PPO.load("ppo_freejoint_hover", device="cpu")

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
        choices=["train", "watch", "drop_test"],
        default="train"
    )

    parser.add_argument(
        "--timesteps",
        type=int,
        default=300_000
    )

    args = parser.parse_args()

    if args.mode == "train":
        train(args.timesteps)

    elif args.mode == "watch":
        watch()

    elif args.mode == "drop_test":
        drop_test()
