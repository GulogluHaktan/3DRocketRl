from __future__ import annotations

import csv
import time
from pathlib import Path

import numpy as np

from hopper_env import HopperEnv, OBSERVATION_NAMES
from telegram_notifier import TelegramNotifier


ETA_REPORT_STEPS = 5_000


EXTRA_INFO_KEYS = (
    "phase",
    "success",
    "fail",
    "fail_reason",
    "transition_bonus",
    "flip_progress",
    "surface_contact",
    "flip_surface_contact",
    "hover_timer",
    "upright_score",
    "rel_dist",
    "linear_speed",
    "horizontal_speed",
    "angular_speed",
    "joint_speed",
    "vertical_velocity",
    "vertical_velocity_change",
    "flip_progress_delta",
    "flip_axis_rate",
    "positive_flip_axis_rate",
    "off_axis_angular_speed",
    "spin_about_body_axis",
    "world_z_spin",
    "expected_axis_alignment",
    "physics_linear_speed_delta",
    "physics_angular_speed_delta",
    "height",
)


def make_env(args, reward_weights=None):
    return HopperEnv(
        start_z=args.start_z,
        max_thrust=args.max_thrust,
        max_tvc_deg=args.max_tvc_deg,
        random_start_z=args.random_start_z or not args.fixed_start_z,
        min_start_z=args.min_start_z,
        max_start_z=args.max_start_z,
        start_phase=args.start_phase,
        reward_weights=reward_weights,
    )


def make_run_dir(algo_name, base_dir="runs"):
    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path(base_dir) / f"{algo_name}_hopper_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    return run_dir


def format_duration(seconds):
    seconds = max(int(seconds), 0)
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def make_step_callback(
    csv_path,
    chunk_index,
    BaseCallback,
    algo_name,
    run_dir,
    session_start_steps,
    session_total_steps,
    session_start_time,
    telegram_notifier=None,
    telegram_every=10_000,
    callback_state=None,
):
    class EpisodeCSVCallback(BaseCallback):
        def __init__(self):
            super().__init__()
            self.callback_state = callback_state if callback_state is not None else {}
            self.file = None
            self.writer = None
            self.next_eta_step = None
            self.next_telegram_step = None
            self.final_eta_printed = False
            self.episode_index = int(self.callback_state.get("episode_index", 0))
            self.episode_reward = float(self.callback_state.get("episode_reward", 0.0))
            self.episode_length = int(self.callback_state.get("episode_length", 0))
            self.best_score_path = Path(run_dir) / "best_score.txt"
            self.best_model_path = Path(run_dir) / f"{algo_name}_hopper_best.zip"
            self.best_checkpoint_path = Path(run_dir) / "checkpoints" / f"{algo_name}_hopper_best.zip"
            self.best_episode_reward = self._load_best_episode_reward()

        def _load_best_episode_reward(self):
            try:
                return float(self.best_score_path.read_text().strip())
            except (OSError, ValueError):
                return -float("inf")

        def _on_training_start(self):
            self.file = csv_path.open("w", newline="")
            fields = (
                "timestep",
                "chunk",
                "episode",
                "reward",
                "episode_reward",
                "episode_length",
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
            reward = float(rewards[0]) if len(rewards) else 0.0
            done = bool(dones[0]) if len(dones) else False
            self.episode_reward += reward
            self.episode_length += 1

            if done:
                self.episode_index += 1
                row = {
                    "timestep": self.num_timesteps,
                    "chunk": chunk_index,
                    "episode": self.episode_index,
                    "reward": self.episode_reward,
                    "episode_reward": self.episode_reward,
                    "episode_length": self.episode_length,
                    "done": True,
                }
                for key in OBSERVATION_NAMES:
                    row[key] = info.get(key, "")
                for key in EXTRA_INFO_KEYS:
                    row[key] = info.get(key, "")

                self.writer.writerow(row)
                self.file.flush()
                if self.episode_reward > self.best_episode_reward:
                    self.best_episode_reward = self.episode_reward
                    self.model.save(self.best_model_path)
                    self.model.save(self.best_checkpoint_path)
                    self.best_score_path.write_text(f"{self.best_episode_reward:.12g}\n")
                    print(
                        f"\n[BEST] episode_reward={self.best_episode_reward:.2f} | "
                        f"model={self.best_model_path}",
                        flush=True,
                    )
                    if telegram_notifier is not None:
                        telegram_notifier.send(
                            f"[{algo_name.upper()} BEST]\n"
                            f"reward: {self.best_episode_reward:.2f}\n"
                            f"step: {self.num_timesteps}\n"
                            f"model: {self.best_model_path}"
                        )
                self.episode_reward = 0.0
                self.episode_length = 0

            completed = max(self.num_timesteps - session_start_steps, 0)
            if self.next_eta_step is None:
                self.next_eta_step = (
                    (completed // ETA_REPORT_STEPS) + 1
                ) * ETA_REPORT_STEPS
            report_now = completed >= self.next_eta_step
            final_report = completed >= session_total_steps and not self.final_eta_printed
            if report_now or final_report:
                elapsed = max(time.monotonic() - session_start_time, 1e-9)
                steps_per_second = completed / elapsed
                remaining_steps = max(session_total_steps - completed, 0)
                eta_seconds = remaining_steps / max(steps_per_second, 1e-9)
                percent = 100.0 * completed / max(session_total_steps, 1)
                print(
                    f"\n[ETA] {completed}/{session_total_steps} step "
                    f"(%{percent:.1f}) | {steps_per_second:.1f} step/s | "
                    f"kalan {format_duration(eta_seconds)} | "
                    f"gecen {format_duration(elapsed)}",
                    flush=True,
                )
                while self.next_eta_step <= completed:
                    self.next_eta_step += ETA_REPORT_STEPS
                self.final_eta_printed = final_report

            if telegram_notifier is not None and telegram_every > 0:
                if self.next_telegram_step is None:
                    self.next_telegram_step = (
                        (completed // telegram_every) + 1
                    ) * telegram_every
                if completed >= self.next_telegram_step or final_report:
                    elapsed = max(time.monotonic() - session_start_time, 1e-9)
                    steps_per_second = completed / elapsed
                    remaining_steps = max(session_total_steps - completed, 0)
                    eta_seconds = remaining_steps / max(steps_per_second, 1e-9)
                    percent = 100.0 * completed / max(session_total_steps, 1)
                    telegram_notifier.send(
                        f"[{algo_name.upper()} TRAIN]\n"
                        f"step: {completed}/{session_total_steps} (%{percent:.1f})\n"
                        f"speed: {steps_per_second:.1f} step/s\n"
                        f"kalan: {format_duration(eta_seconds)}\n"
                        f"gecen: {format_duration(elapsed)}\n"
                        f"best reward: {self.best_episode_reward:.2f}"
                    )
                    while self.next_telegram_step <= completed:
                        self.next_telegram_step += telegram_every
            self.callback_state["episode_index"] = self.episode_index
            self.callback_state["episode_reward"] = self.episode_reward
            self.callback_state["episode_length"] = self.episode_length
            return True

        def _on_training_end(self):
            self.callback_state["episode_index"] = self.episode_index
            self.callback_state["episode_reward"] = self.episode_reward
            self.callback_state["episode_length"] = self.episode_length
            if self.file is not None:
                self.file.close()

    return EpisodeCSVCallback()


def train_loop(args, algo_name, Algorithm, BaseCallback, model_kwargs, reward_weights=None):
    run_dir = Path(args.run_dir) if args.run_dir else make_run_dir(algo_name, args.runs_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)

    env = make_env(args, reward_weights=reward_weights)
    telegram_notifier = None
    if not args.no_telegram:
        telegram_notifier = TelegramNotifier.from_config(args.telegram_config)
        if telegram_notifier is None:
            print(
                f"[Telegram] kapali: {args.telegram_config} icinde bot_token/chat_id yok.",
                flush=True,
            )
        else:
            print(
                f"[Telegram] acik: her {args.telegram_every} stepte bildirim.",
                flush=True,
            )
    root_model = f"{algo_name}_hopper_latest.zip"
    run_model = run_dir / root_model

    if args.resume:
        model = Algorithm.load(args.resume, env=env, device="cpu")
        print(f"{algo_name.upper()} modeli devam ediyor: {args.resume}")
    else:
        model = Algorithm("MlpPolicy", env, verbose=1, device="cpu", **model_kwargs)
        print(f"{algo_name.upper()} modeli sifirdan basliyor.")

    remaining = args.timesteps
    existing_chunks = sorted(run_dir.glob("steps_chunk_*.csv"))
    chunk_index = 0
    chunk_numbers = [
        int(path.stem.rsplit("_", 1)[-1])
        for path in existing_chunks
        if path.stem.rsplit("_", 1)[-1].isdigit()
    ]
    if chunk_numbers:
        chunk_index = max(chunk_numbers)
    session_start_steps = model.num_timesteps
    session_start_time = time.monotonic()
    callback_state = {
        "episode_index": 0,
        "episode_reward": 0.0,
        "episode_length": 0,
    }
    if telegram_notifier is not None:
        telegram_notifier.send(
            f"[{algo_name.upper()} TRAIN START]\n"
            f"timesteps: {args.timesteps}\n"
            f"chunk: {args.chunk_steps}\n"
            f"run: {run_dir}"
        )

    while remaining > 0:
        chunk_steps = min(args.chunk_steps, remaining)
        chunk_index += 1
        csv_path = run_dir / f"steps_chunk_{chunk_index:03d}.csv"
        print(f"\n=== {algo_name.upper()} CHUNK {chunk_index} | {chunk_steps} step | {csv_path} ===")
        callback = make_step_callback(
            csv_path,
            chunk_index,
            BaseCallback,
            algo_name=algo_name,
            run_dir=run_dir,
            session_start_steps=session_start_steps,
            session_total_steps=args.timesteps,
            session_start_time=session_start_time,
            telegram_notifier=telegram_notifier,
            telegram_every=max(int(args.telegram_every), 0),
            callback_state=callback_state,
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
        print(f"Checkpoint kaydedildi: {checkpoint}")
        remaining -= chunk_steps

    plot_run(run_dir)
    env.close_viewer()
    print(f"Run klasoru: {run_dir}")
    if telegram_notifier is not None:
        telegram_notifier.send(f"[{algo_name.upper()} TRAIN DONE]\nrun: {run_dir}")


def watch_model(args, algo_name, Algorithm, reward_weights=None):
    env = make_env(args, reward_weights=reward_weights)
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
            fieldnames=(
                "episode",
                "step",
                "reward",
                "episode_reward",
                "episode_length",
                "done",
                *OBSERVATION_NAMES,
                *EXTRA_INFO_KEYS,
            ),
        )
        writer.writeheader()

    step_count = 0
    episode_index = 0
    episode_reward = 0.0
    try:
        while viewer.is_running():
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            step_count += 1
            episode_reward += float(reward)
            done = bool(terminated or truncated)

            if writer is not None:
                row = {
                    "episode": episode_index + 1,
                    "step": step_count,
                    "reward": float(reward),
                    "episode_reward": episode_reward,
                    "episode_length": step_count,
                    "done": done,
                }
                for key in OBSERVATION_NAMES:
                    row[key] = info.get(key, "")
                for key in EXTRA_INFO_KEYS:
                    row[key] = info.get(key, "")
                writer.writerow(row)
                csv_file.flush()

            if done:
                episode_index += 1
                print("Episode bitti:", {
                    "phase": info.get("phase"),
                    "success": info.get("success"),
                    "fail": info.get("fail"),
                    "fail_reason": info.get("fail_reason"),
                    "transition_bonus": round(float(info.get("transition_bonus", 0.0)), 3),
                    "flip_progress": round(float(info.get("flip_progress", 0.0)), 3),
                    "surface_contact": info.get("surface_contact"),
                    "flip_surface_contact": info.get("flip_surface_contact"),
                    "rel_dist": round(float(info.get("rel_dist", 0.0)), 3),
                    "height": round(float(info.get("height", 0.0)), 3),
                    "upright": round(float(info.get("upright_score", 0.0)), 3),
                    "linear_speed": round(float(info.get("linear_speed", 0.0)), 3),
                    "horizontal_speed": round(float(info.get("horizontal_speed", 0.0)), 3),
                    "angular_speed": round(float(info.get("angular_speed", 0.0)), 3),
                    "joint_speed": round(float(info.get("joint_speed", 0.0)), 3),
                    "main": round(float(info.get("main_motor_power", 0.0)), 3),
                    "flip_axis_rate": round(float(info.get("flip_axis_rate", 0.0)), 3),
                    "world_z_spin": round(float(info.get("world_z_spin", 0.0)), 3),
                    "physics_linear_jump": round(float(info.get("physics_linear_speed_delta", 0.0)), 3),
                    "physics_angular_jump": round(float(info.get("physics_angular_speed_delta", 0.0)), 3),
                })
                obs, _ = env.reset()
                step_count = 0
                episode_reward = 0.0
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
        "speeds.png": (("linear_speed", "horizontal_speed", "angular_speed"), "Speeds"),
        "flip_axis.png": (
            ("flip_axis_rate", "positive_flip_axis_rate", "world_z_spin"),
            "Flip axis rates",
        ),
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
