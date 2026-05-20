import argparse
import time
from pathlib import Path

import numpy as np

from hopper_env import HopperEnv, OBSERVATION_NAMES


ALGORITHMS = {
    "sac": ("stable_baselines3", "SAC"),
    "ppo": ("stable_baselines3", "PPO"),
    "td3": ("stable_baselines3", "TD3"),
}


def load_model(algo, model_path, env):
    try:
        module_name, class_name = ALGORITHMS[algo]
        module = __import__(module_name, fromlist=[class_name])
        Algorithm = getattr(module, class_name)
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "stable_baselines3 kurulu degil. Model izlemek icin "
            "`pip install stable-baselines3` gerekli."
        ) from exc

    return Algorithm.load(str(model_path), env=env, device="cpu")


def print_observation(obs):
    values = ", ".join(
        f"{name}={value:.3f}"
        for name, value in zip(OBSERVATION_NAMES, obs)
    )
    print(values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--algo",
        choices=sorted(ALGORITHMS),
        default="sac",
        help="Yuklenecek modelin algoritmasi.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Train dosyasinin kaydettigi model zip yolu.",
    )
    parser.add_argument(
        "--random",
        action="store_true",
        help="Model yuklemeden random action ile goster.",
    )
    parser.add_argument(
        "--zero",
        action="store_true",
        help="Model yuklemeden sifir action ile goster.",
    )
    parser.add_argument(
        "--print-obs",
        action="store_true",
        help="Observation degerlerini terminale yazdir.",
    )
    parser.add_argument(
        "--start-z",
        type=float,
        default=10.0,
        help="Baslangic yuksekligi.",
    )
    parser.add_argument(
        "--fixed-start-z",
        action="store_true",
        help="Random reset yuksekligi yerine --start-z kullan.",
    )
    parser.add_argument(
        "--min-start-z",
        type=float,
        default=0.5,
        help="Random reset minimum yuksekligi.",
    )
    parser.add_argument(
        "--max-start-z",
        type=float,
        default=10.0,
        help="Random reset maksimum yuksekligi.",
    )
    parser.add_argument(
        "--max-thrust",
        type=float,
        default=None,
        help="Maksimum thrust Newton. Varsayilan: 3.6 kgf.",
    )
    parser.add_argument(
        "--max-tvc-deg",
        type=float,
        default=20.0,
        help="Maksimum TVC acisi derece.",
    )
    args = parser.parse_args()

    env = HopperEnv(
        start_z=args.start_z,
        max_thrust=args.max_thrust,
        max_tvc_deg=args.max_tvc_deg,
        random_start_z=not args.fixed_start_z,
        min_start_z=args.min_start_z,
        max_start_z=args.max_start_z,
    )
    obs, _ = env.reset()
    model = None

    if not args.random and not args.zero:
        model_path = Path(args.model or f"{args.algo}_hopper_latest.zip")
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model bulunamadi: {model_path}\n"
                "Random deneme icin: python visualize.py --random"
            )
        model = load_model(args.algo, model_path, env)
        print(f"{args.algo.upper()} modeli yuklendi: {model_path}")

    viewer = env.launch_viewer()
    step_count = 0

    try:
        while viewer.is_running():
            if model is not None:
                action, _ = model.predict(obs, deterministic=True)
            elif args.random:
                action = env.action_space.sample()
            else:
                action = np.array([0.0, 0.0, 0.0], dtype=np.float32)

            obs, reward, terminated, truncated, info = env.step(action)
            step_count += 1

            if args.print_obs and step_count % 30 == 0:
                print_observation(obs)

            if terminated or truncated:
                print("Episode bitti:", {k: round(v, 3) for k, v in info.items()})
                time.sleep(0.5)
                obs, _ = env.reset()
                step_count = 0

    except KeyboardInterrupt:
        print("Ctrl+C ile cikildi.")
    finally:
        env.close_viewer()


if __name__ == "__main__":
    main()
