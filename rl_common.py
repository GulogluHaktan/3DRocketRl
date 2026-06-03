from __future__ import annotations

import csv
import time
from pathlib import Path

import numpy as np

from hopper_env import HopperEnv


EXTRA_INFO_KEYS = (
    "phase",
    "success",
    "fail",
    "fail_reason",
    "phase_bonus",
    "flip_progress",
    "hover_timer",
    "climb_ready_timer",
    "flip_low_altitude_stall_timer",
    "recovery_low_altitude_timer",
    "upright_score",
    "rel_dist",
    "linear_speed",
    "angular_speed",
    "joint_speed",
    "height",
    "main_thrust_newton",
)


def make_env(args, reward_weights=None, env_kwargs=None):
    env_kwargs = env_kwargs or {}
    config = {
        "start_z": args.start_z,
        "max_thrust": args.max_thrust,
        "max_tvc_deg": args.max_tvc_deg,
        "random_start_z": not args.fixed_start_z,
        "min_start_z": args.min_start_z,
        "max_start_z": args.max_start_z,
        "reward_weights": reward_weights,
    }
    config.update(env_kwargs)
    return HopperEnv(**config)


def make_run_dir(algo_name, base_dir="runs"):
    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path(base_dir) / f"{algo_name}_hopper_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    return run_dir


def make_step_callback(
    csv_path,
    chunk_index,
    BaseCallback,
    observation_names,
    run_start_timesteps,
    run_total_timesteps,
):
    class StepCSVCallback(BaseCallback):
        def __init__(self):
            super().__init__()
            self.file = None
            self.writer = None
            self.next_progress_update = 0

        def _on_training_start(self):
            self.file = csv_path.open("w", newline="")
            fields = (
                "timestep",
                "chunk",
                "reward",
                "done",
                *observation_names,
                *EXTRA_INFO_KEYS,
            )
            self.writer = csv.DictWriter(self.file, fieldnames=fields)
            self.writer.writeheader()
            self.next_progress_update = self.num_timesteps

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
            for key in observation_names:
                row[key] = info.get(key, "")
            for key in EXTRA_INFO_KEYS:
                row[key] = info.get(key, "")

            self.writer.writerow(row)
            self._print_progress()
            return True

        def _on_training_end(self):
            self._print_progress(force=True)
            if self.file is not None:
                self.file.close()

        def _print_progress(self, force=False):
            if not force and self.num_timesteps < self.next_progress_update:
                return

            completed = max(0, self.num_timesteps - run_start_timesteps)
            completed = min(completed, run_total_timesteps)
            pct = 100.0 * completed / max(run_total_timesteps, 1)
            width = 30
            filled = int(width * pct / 100.0)
            bar = "#" * filled + "-" * (width - filled)
            print(
                f"\rTrain progress [{bar}] {pct:6.2f}% "
                f"({completed}/{run_total_timesteps})",
                end="",
                flush=True,
            )
            self.next_progress_update = self.num_timesteps + 1000

    return StepCSVCallback()


def train_loop(
    args,
    algo_name,
    Algorithm,
    BaseCallback,
    model_kwargs,
    reward_weights=None,
    env_kwargs=None,
):
    run_dir = Path(args.run_dir) if args.run_dir else make_run_dir(algo_name, args.runs_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)

    env = make_env(args, reward_weights=reward_weights, env_kwargs=env_kwargs)
    observation_names = tuple(env.observation_names)
    root_model = f"{algo_name}_hopper_latest.zip"
    run_model = run_dir / root_model

    if args.resume:
        model = Algorithm.load(args.resume, env=env, device="cpu")
        print(f"{algo_name.upper()} modeli devam ediyor: {args.resume}")
    else:
        model = Algorithm("MlpPolicy", env, verbose=1, device="cpu", **model_kwargs)
        print(f"{algo_name.upper()} modeli sifirdan basliyor.")

    remaining = args.timesteps
    chunk_index = 0
    run_start_timesteps = model.num_timesteps

    while remaining > 0:
        chunk_steps = min(args.chunk_steps, remaining)
        chunk_index += 1
        csv_path = run_dir / f"steps_chunk_{chunk_index:03d}.csv"
        print(f"\n=== {algo_name.upper()} CHUNK {chunk_index} | {chunk_steps} step | {csv_path} ===")
        callback = make_step_callback(
            csv_path,
            chunk_index,
            BaseCallback,
            observation_names,
            run_start_timesteps,
            args.timesteps,
        )
        model.learn(
            total_timesteps=chunk_steps,
            reset_num_timesteps=False,
            callback=callback,
            log_interval=10,
        )
        checkpoint = run_dir / "checkpoints" / f"{algo_name}_hopper_{model.num_timesteps}.zip"
        model.save(checkpoint)
        model.save(run_model)
        model.save(root_model)
        print(f"\nCheckpoint kaydedildi: {checkpoint}")
        remaining -= chunk_steps

    plot_run(run_dir)
    env.close_viewer()
    print(f"Run klasoru: {run_dir}")


def watch_model(args, algo_name, Algorithm, reward_weights=None, env_kwargs=None):
    env = make_env(args, reward_weights=reward_weights, env_kwargs=env_kwargs)
    observation_names = tuple(env.observation_names)
    obs, _ = env.reset()
    model_path = args.model or f"{algo_name}_hopper_latest.zip"
    model = Algorithm.load(model_path, env=env, device="cpu")
    viewer = env.launch_viewer()
    csv_file = None
    writer = None

    if args.csv:
        csv_path = Path(args.csv)
        csv_file = csv_path.open("w", newline="")
        writer = csv.DictWriter(
            csv_file,
            fieldnames=("step", "reward", *observation_names, *EXTRA_INFO_KEYS),
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
                for key in observation_names:
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
                    "thrust_N": round(float(info.get("main_thrust_newton", 0.0)), 3),
                })
                obs, _ = env.reset()
                step_count = 0
    finally:
        if csv_file is not None:
            csv_file.close()
        env.close_viewer()


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
