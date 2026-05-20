from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np

from hopper_env import HopperEnv, OBSERVATION_NAMES


EXTRA_INFO_KEYS = (
    "phase",
    "success",
    "fail",
    "fail_reason",
    "phase_bonus",
    "flip_progress",
    "hover_timer",
    "upright_score",
    "rel_dist",
    "linear_speed",
    "angular_speed",
    "joint_speed",
    "height",
)


def require_sac():
    try:
        from stable_baselines3 import SAC
        from stable_baselines3.common.callbacks import BaseCallback
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "SAC icin stable-baselines3 gerekli: pip install stable-baselines3"
        ) from exc
    return SAC, BaseCallback


def make_run_dir(base_dir="runs"):
    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path(base_dir) / f"sac_hopper_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    return run_dir


def make_step_callback(csv_path, chunk_index):
    _, BaseCallback = require_sac()

    class StepCSVCallback(BaseCallback):
        def __init__(self):
            super().__init__()
            self.file = None
            self.writer = None

        def _on_training_start(self):
            self.file = csv_path.open("w", newline="")
            fields = (
                "timestep",
                "chunk",
                "reward",
                "done",
                *OBSERVATION_NAMES,
                *EXTRA_INFO_KEYS,
            )
            self.writer = csv.DictWriter(self.file, fieldnames=fields)
            self.writer.writeheader()

        def _on_step(self):
            infos = self.locals.get("infos", [])
            rewards = self.locals.get("rewards", [])
            dones = self.locals.get("dones", [])

            if not infos:
                return True

            info = infos[0]
            row = {
                "timestep": self.num_timesteps,
                "chunk": chunk_index,
                "reward": float(rewards[0]) if len(rewards) else 0.0,
                "done": bool(dones[0]) if len(dones) else False,
            }
            for key in OBSERVATION_NAMES:
                row[key] = info.get(key, "")
            for key in EXTRA_INFO_KEYS:
                row[key] = info.get(key, "")

            self.writer.writerow(row)
            return True

        def _on_training_end(self):
            if self.file is not None:
                self.file.close()

    return StepCSVCallback()


def train(args):
    SAC, _ = require_sac()
    run_dir = Path(args.run_dir) if args.run_dir else make_run_dir(args.runs_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)

    env = HopperEnv(
        start_z=args.start_z,
        max_thrust=args.max_thrust,
        max_tvc_deg=args.max_tvc_deg,
        random_start_z=not args.fixed_start_z,
        min_start_z=args.min_start_z,
        max_start_z=args.max_start_z,
    )

    if args.resume:
        model = SAC.load(args.resume, env=env, device="cpu")
        print(f"Model devam ediyor: {args.resume}")
    else:
        model = SAC(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=args.learning_rate,
            buffer_size=args.buffer_size,
            batch_size=args.batch_size,
            gamma=args.gamma,
            tau=args.tau,
            train_freq=1,
            gradient_steps=1,
            learning_starts=args.learning_starts,
            device="cpu",
        )
        print("SAC modeli sifirdan basliyor.")

    remaining = args.timesteps
    chunk_index = 0

    while remaining > 0:
        chunk_steps = min(args.chunk_steps, remaining)
        chunk_index += 1
        csv_path = run_dir / f"steps_chunk_{chunk_index:03d}.csv"
        print(f"\n=== CHUNK {chunk_index} | {chunk_steps} step | {csv_path} ===")
        callback = make_step_callback(csv_path, chunk_index)
        model.learn(
            total_timesteps=chunk_steps,
            reset_num_timesteps=False,
            callback=callback,
            log_interval=10,
        )
        checkpoint = run_dir / "checkpoints" / f"sac_hopper_{model.num_timesteps}.zip"
        model.save(checkpoint)
        model.save(run_dir / "sac_hopper_latest.zip")
        model.save("sac_hopper_latest.zip")
        print(f"Checkpoint kaydedildi: {checkpoint}")
        remaining -= chunk_steps

    plot_run(run_dir)
    env.close_viewer()
    print(f"Run klasoru: {run_dir}")


def read_step_rows(run_dir):
    rows = []
    for csv_path in sorted(Path(run_dir).glob("steps_chunk_*.csv")):
        with csv_path.open(newline="") as file:
            rows.extend(csv.DictReader(file))
    return rows


def to_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def plot_run(run_dir):
    rows = read_step_rows(run_dir)
    if not rows:
        print("Grafik icin CSV bulunamadi.")
        return

    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print("matplotlib yok, grafik olusturulmadi.")
        return

    run_dir = Path(run_dir)
    timesteps = np.array([to_float(row["timestep"]) for row in rows])

    plots = {
        "reward.png": ("reward", "Reward"),
        "flip_progress.png": ("flip_progress", "Flip progress"),
        "rel_dist.png": ("rel_dist", "Relative XY distance"),
        "speeds.png": (("linear_speed", "angular_speed"), "Speeds"),
        "upright.png": ("upright_score", "Upright score"),
        "height.png": ("bottom_height", "Bottom height"),
    }

    for filename, (keys, title) in plots.items():
        plt.figure(figsize=(10, 4))
        if isinstance(keys, tuple):
            for key in keys:
                plt.plot(timesteps, [to_float(row.get(key)) for row in rows], label=key)
            plt.legend()
        else:
            plt.plot(timesteps, [to_float(row.get(keys)) for row in rows])
        plt.title(title)
        plt.xlabel("timestep")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        output_path = run_dir / filename
        plt.savefig(output_path)
        plt.close()
        print(f"Grafik kaydedildi: {output_path}")


def watch(args):
    SAC, _ = require_sac()
    env = HopperEnv(
        start_z=args.start_z,
        max_thrust=args.max_thrust,
        max_tvc_deg=args.max_tvc_deg,
        random_start_z=not args.fixed_start_z,
        min_start_z=args.min_start_z,
        max_start_z=args.max_start_z,
    )
    obs, _ = env.reset()
    model = SAC.load(args.model, env=env, device="cpu")
    viewer = env.launch_viewer()
    csv_file = None
    writer = None

    if args.csv:
        csv_path = Path(args.csv)
        csv_file = csv_path.open("w", newline="")
        writer = csv.DictWriter(
            csv_file,
            fieldnames=("step", "reward", *OBSERVATION_NAMES, *EXTRA_INFO_KEYS),
        )
        writer.writeheader()

    step_count = 0
    try:
        while viewer.is_running():
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            step_count += 1

            if writer is not None:
                row = {"step": step_count, "reward": reward}
                for key in OBSERVATION_NAMES:
                    row[key] = info.get(key, "")
                for key in EXTRA_INFO_KEYS:
                    row[key] = info.get(key, "")
                writer.writerow(row)

            if terminated or truncated:
                print("Episode bitti:", {
                    "phase": info.get("phase"),
                    "success": info.get("success"),
                    "fail": info.get("fail"),
                    "fail_reason": info.get("fail_reason"),
                    "flip_progress": round(float(info.get("flip_progress", 0.0)), 3),
                    "rel_dist": round(float(info.get("rel_dist", 0.0)), 3),
                    "height": round(float(info.get("height", 0.0)), 3),
                    "upright": round(float(info.get("upright_score", 0.0)), 3),
                    "linear_speed": round(float(info.get("linear_speed", 0.0)), 3),
                    "angular_speed": round(float(info.get("angular_speed", 0.0)), 3),
                    "joint_speed": round(float(info.get("joint_speed", 0.0)), 3),
                })
                obs, _ = env.reset()
                step_count = 0
    finally:
        if csv_file is not None:
            csv_file.close()
        env.close_viewer()


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)

    train_parser = sub.add_parser("train")
    train_parser.add_argument("--timesteps", type=int, default=250_000)
    train_parser.add_argument("--chunk-steps", type=int, default=25_000)
    train_parser.add_argument("--runs-dir", default="runs")
    train_parser.add_argument("--run-dir", default=None)
    train_parser.add_argument("--resume", default=None)
    train_parser.add_argument("--start-z", type=float, default=10.0)
    train_parser.add_argument("--fixed-start-z", action="store_true")
    train_parser.add_argument("--min-start-z", type=float, default=0.5)
    train_parser.add_argument("--max-start-z", type=float, default=10.0)
    train_parser.add_argument("--max-thrust", type=float, default=None)
    train_parser.add_argument("--max-tvc-deg", type=float, default=20.0)
    train_parser.add_argument("--learning-rate", type=float, default=3e-4)
    train_parser.add_argument("--buffer-size", type=int, default=300_000)
    train_parser.add_argument("--batch-size", type=int, default=256)
    train_parser.add_argument("--gamma", type=float, default=0.99)
    train_parser.add_argument("--tau", type=float, default=0.005)
    train_parser.add_argument("--learning-starts", type=int, default=5_000)

    watch_parser = sub.add_parser("watch")
    watch_parser.add_argument("--model", default="sac_hopper_latest.zip")
    watch_parser.add_argument("--start-z", type=float, default=10.0)
    watch_parser.add_argument("--fixed-start-z", action="store_true")
    watch_parser.add_argument("--min-start-z", type=float, default=0.5)
    watch_parser.add_argument("--max-start-z", type=float, default=10.0)
    watch_parser.add_argument("--max-thrust", type=float, default=None)
    watch_parser.add_argument("--max-tvc-deg", type=float, default=20.0)
    watch_parser.add_argument("--csv", default=None)

    plot_parser = sub.add_parser("plot")
    plot_parser.add_argument("run_dir")

    args = parser.parse_args()

    if args.mode == "train":
        train(args)
    elif args.mode == "watch":
        watch(args)
    elif args.mode == "plot":
        plot_run(args.run_dir)


if __name__ == "__main__":
    main()
