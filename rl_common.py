from __future__ import annotations

import csv
import glob
import json
import time
from pathlib import Path

import gymnasium as gym
import numpy as np

from hopper_env import HopperEnv, REWARD_BREAKDOWN_KEYS
from telegram_notifier import TelegramNotifier


ETA_REPORT_STEPS = 5_000


EXTRA_INFO_KEYS = (
    "phase",
    "specialist_phase",
    "specialist_success",
    "specialist_handoff_phase",
    "handoff_used",
    "handoff_attempt",
    "handoff_steps",
    "handoff_phase",
    "handoff_reward",
    "handoff_active_model_sequence",
    "success",
    "fail",
    "fail_reason",
    "transition_bonus",
    "flip_progress",
    "surface_contact",
    "flip_surface_contact",
    "hover_timer",
    "climb_ready_timer",
    "flip_low_altitude_stall_timer",
    "recovery_low_altitude_timer",
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
    "main_thrust_newton",
    "ready_for_flip",
    "ready_for_recovery",
    "ready_for_hover",
    "hover_stable",
    "attitude_recovery_test",
    "attitude_upright_hold_timer",
    "attitude_tilt_deg",
    "reward_mode",
    *REWARD_BREAKDOWN_KEYS,
)


def unique_fields(fields):
    return tuple(dict.fromkeys(fields))


def make_env(args, reward_weights=None, env_kwargs=None):
    config = {
        "start_z": args.start_z,
        "max_thrust": args.max_thrust,
        "max_tvc_deg": args.max_tvc_deg,
        "tvc_servo_sec_per_60deg": getattr(args, "tvc_servo_sec_per_60deg", 0.13),
        "random_start_z": args.random_start_z or not args.fixed_start_z,
        "min_start_z": args.min_start_z,
        "max_start_z": args.max_start_z,
        "start_phase": args.start_phase,
        "specialist_phase": getattr(args, "specialist_phase", None),
        "reward_weights": reward_weights,
        "attitude_recovery_test": getattr(args, "attitude_recovery_test", False),
        "max_start_tilt_deg": getattr(args, "max_start_tilt_deg", 45.0),
        "min_start_tilt_deg": getattr(args, "min_start_tilt_deg", 5.0),
        "upright_success_deg": getattr(args, "upright_success_deg", 5.0),
        "upright_hold_sec": getattr(args, "upright_hold_sec", 1.0),
    }
    if env_kwargs:
        config.update(env_kwargs)
    for arg_name in (
        "flip_target_z",
        "flip_start_min_z",
        "flip_start_max_z",
        "climb_ready_min_z",
    ):
        value = getattr(args, arg_name, None)
        if value is not None:
            config[arg_name] = value
    phase_start_roughness = float(getattr(args, "phase_start_roughness", 0.0) or 0.0)
    if phase_start_roughness > 0.0:
        if args.start_phase == "flip":
            config["flip_start_roughness"] = phase_start_roughness
        elif args.start_phase == "recovery":
            config["recovery_start_roughness"] = phase_start_roughness
        elif args.start_phase == "hover":
            config["hover_start_roughness"] = phase_start_roughness
    return HopperEnv(**config)


class HandoffStartWrapper(gym.Wrapper):
    def __init__(
        self,
        env,
        handoff_model,
        target_phase,
        max_steps=600,
        attempts=20,
    ):
        super().__init__(env)
        self.handoff_model = handoff_model
        self.target_phase = target_phase
        self.max_steps = int(max_steps)
        self.attempts = int(attempts)
        self.last_handoff_info = {}

    def reset(self, **kwargs):
        last_info = {}
        for attempt in range(1, max(self.attempts, 1) + 1):
            reset_kwargs = dict(kwargs)
            if attempt > 1:
                reset_kwargs.pop("seed", None)
            obs, info = self.env.reset(**reset_kwargs)
            if self.env.current_phase == self.target_phase:
                self.last_handoff_info = {
                    "handoff_used": False,
                    "handoff_attempt": attempt,
                    "handoff_steps": 0,
                    "handoff_phase": self.env.current_phase,
                    "handoff_reward": 0.0,
                    "handoff_active_model_sequence": self.env.current_phase,
                }
                return obs, {**info, **self.last_handoff_info}

            episode_reward = 0.0
            for step in range(1, max(self.max_steps, 1) + 1):
                action, _ = self.handoff_model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, last_info = self.env.step(action)
                episode_reward += float(reward)
                if self.env.current_phase == self.target_phase and not terminated and not truncated:
                    self.last_handoff_info = {
                        "handoff_used": True,
                        "handoff_attempt": attempt,
                        "handoff_steps": step,
                        "handoff_phase": self.env.current_phase,
                        "handoff_reward": episode_reward,
                        "handoff_active_model_sequence": self.env.current_phase,
                    }
                    return obs, {
                        **info,
                        **self.last_handoff_info,
                    }
                if terminated or truncated:
                    break

        raise RuntimeError(
            f"Handoff target phase '{self.target_phase}' bulunamadi. "
            f"Son phase={self.env.current_phase!r}, info={last_info}"
        )

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if self.last_handoff_info:
            info = {
                **info,
                **self.last_handoff_info,
            }
        return obs, reward, terminated, truncated, info


class PhaseChainHandoffStartWrapper(gym.Wrapper):
    def __init__(
        self,
        env,
        phase_models,
        phase_paths,
        target_phase,
        max_steps=1200,
        attempts=20,
    ):
        super().__init__(env)
        self.phase_models = phase_models
        self.phase_paths = phase_paths
        self.target_phase = target_phase
        self.max_steps = int(max_steps)
        self.attempts = int(attempts)
        self.last_handoff_info = {}

    def reset(self, **kwargs):
        last_info = {}
        for attempt in range(1, max(self.attempts, 1) + 1):
            reset_kwargs = dict(kwargs)
            if attempt > 1:
                reset_kwargs.pop("seed", None)
            obs, info = self.env.reset(**reset_kwargs)
            active_model_sequence = []
            if self.env.current_phase == self.target_phase:
                self.last_handoff_info = {
                    "handoff_used": False,
                    "handoff_attempt": attempt,
                    "handoff_steps": 0,
                    "handoff_phase": self.env.current_phase,
                    "handoff_reward": 0.0,
                    "handoff_active_model_sequence": self.env.current_phase,
                }
                return obs, {**info, **self.last_handoff_info}

            episode_reward = 0.0
            for step in range(1, max(self.max_steps, 1) + 1):
                phase = self.env.current_phase
                if phase == "done":
                    break
                if phase not in self.phase_models:
                    last_info = {"phase": phase, "error": "phase model yok"}
                    break
                append_sequence_value(active_model_sequence, phase)
                action, _ = self.phase_models[phase].predict(obs, deterministic=True)
                obs, reward, terminated, truncated, last_info = self.env.step(action)
                episode_reward += float(reward)
                if self.env.current_phase == self.target_phase and not terminated and not truncated:
                    self.last_handoff_info = {
                        "handoff_used": True,
                        "handoff_attempt": attempt,
                        "handoff_steps": step,
                        "handoff_phase": self.env.current_phase,
                        "handoff_reward": episode_reward,
                        "handoff_active_model_sequence": ">".join(active_model_sequence),
                    }
                    return obs, {**info, **self.last_handoff_info}
                if terminated or truncated:
                    break

        raise RuntimeError(
            f"Phase-chain handoff target phase '{self.target_phase}' bulunamadi. "
            f"Son phase={self.env.current_phase!r}, info={last_info}"
        )

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if self.last_handoff_info:
            info = {
                **info,
                **self.last_handoff_info,
            }
        return obs, reward, terminated, truncated, info


def generate_headless_video(env, model, video_path, duration_sec=8.0, fps=30):
    import imageio
    import mujoco
    import numpy as np

    renderer = mujoco.Renderer(env.model, height=480, width=640)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance = 5.0
    cam.azimuth = 135.0
    cam.elevation = -18.0

    frames = []
    obs, _ = env.reset()
    dt = env.model.opt.timestep * env.frame_skip
    total_steps = int(duration_sec / dt)
    step_interval = int(round(1.0 / (fps * dt)))
    if step_interval < 1:
        step_interval = 1

    step_count = 0
    while step_count < total_steps:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        if step_count % step_interval == 0:
            pos = env.data.xpos[env.body_id]
            cam.lookat[:] = pos
            renderer.update_scene(env.data, camera=cam)
            pixels = renderer.render()
            frames.append(pixels)

        step_count += 1
        if terminated or truncated:
            break

    video_path = Path(video_path)
    video_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(video_path), frames, fps=fps)


def make_run_dir(algo_name, base_dir="runs", specialist_phase=None):
    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_name = f"{algo_name}_{specialist_phase}_{stamp}" if specialist_phase else f"{algo_name}_hopper_{stamp}"
    run_dir = Path(base_dir) / run_name
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
    observation_names,
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
            fields = unique_fields((
                "timestep",
                "chunk",
                "episode",
                "reward",
                "episode_reward",
                "episode_length",
                "done",
                *observation_names,
                *EXTRA_INFO_KEYS,
            ))
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
                for key in observation_names:
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


def train_loop(
    args,
    algo_name,
    Algorithm,
    BaseCallback,
    model_kwargs,
    reward_weights=None,
    env_kwargs=None,
):
    run_dir = (
        Path(args.run_dir)
        if args.run_dir
        else make_run_dir(algo_name, args.runs_dir, getattr(args, "specialist_phase", None))
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)

    env = make_env(args, reward_weights=reward_weights, env_kwargs=env_kwargs)
    handoff_model = None
    handoff_phase_models = None
    handoff_phase_paths = None
    if getattr(args, "handoff_model", None) and getattr(args, "handoff_phase_models_config", None):
        raise ValueError("--handoff-model ve --handoff-phase-models-config ayni anda kullanilamaz.")
    if getattr(args, "handoff_model", None):
        if getattr(args, "specialist_phase", None) is None:
            raise ValueError("--handoff-model icin --specialist-phase gerekli.")
        handoff_model = Algorithm.load(args.handoff_model, env=env, device="cpu")
        env = HandoffStartWrapper(
            env,
            handoff_model=handoff_model,
            target_phase=args.specialist_phase,
            max_steps=getattr(args, "handoff_max_steps", 600),
            attempts=getattr(args, "handoff_attempts", 20),
        )
        print(
            "[Handoff] acik: "
            f"{args.start_phase} -> {args.specialist_phase} | "
            f"model={args.handoff_model}",
            flush=True,
        )
    elif getattr(args, "handoff_phase_models_config", None):
        if getattr(args, "specialist_phase", None) is None:
            raise ValueError("--handoff-phase-models-config icin --specialist-phase gerekli.")
        handoff_phase_models, handoff_phase_paths = load_phase_models(
            args.handoff_phase_models_config,
            Algorithm,
            env,
        )
        env = PhaseChainHandoffStartWrapper(
            env,
            phase_models=handoff_phase_models,
            phase_paths=handoff_phase_paths,
            target_phase=args.specialist_phase,
            max_steps=getattr(args, "handoff_max_steps", 1200),
            attempts=getattr(args, "handoff_attempts", 20),
        )
        print(
            "[Phase Handoff] acik: "
            f"{args.start_phase} -> {args.specialist_phase} | "
            f"config={args.handoff_phase_models_config}",
            flush=True,
        )
    observation_names = tuple(env.unwrapped.observation_names)
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
    specialist_phase = getattr(args, "specialist_phase", None)
    root_model = (
        f"{algo_name}_{specialist_phase}_latest.zip"
        if specialist_phase
        else f"{algo_name}_hopper_latest.zip"
    )
    run_model = run_dir / root_model
    run_compat_model = run_dir / f"{algo_name}_hopper_latest.zip"

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

    last_video_step = 0
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
            observation_names=observation_names,
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
        if run_compat_model != run_model:
            model.save(run_compat_model)
        
        # Preserve golden flip model if it exists in root
        if root_model == f"{algo_name}_flip_latest.zip" and Path(root_model).exists():
            golden_backup = Path(root_model).parent / f"{algo_name}_flip_latest_golden.zip"
            if not golden_backup.exists():
                import shutil
                try:
                    shutil.copy2(root_model, golden_backup)
                    print(f"Golden flip model backed up to: {golden_backup}", flush=True)
                except Exception as exc:
                    print(f"Golden flip model backup hatasi: {exc}", flush=True)

        model.save(root_model)
        print(f"\nCheckpoint kaydedildi: {checkpoint}")

        # Headless video generation
        video_every = getattr(args, "telegram_video_every", 100_000)
        no_video = getattr(args, "no_telegram_video", False)
        if not no_video and video_every > 0:
            completed_steps = model.num_timesteps
            if (completed_steps // video_every) > (last_video_step // video_every):
                last_video_step = completed_steps
                print(f"\n[Video] Headless watch videosu uretiliyor... (step {completed_steps})", flush=True)
                video_name = f"watch_{specialist_phase or 'hopper'}_{completed_steps}.mp4"
                video_path = run_dir / "videos" / video_name
                try:
                    import copy
                    video_args = copy.deepcopy(args)
                    video_args.fixed_start_z = True
                    # Create env
                    video_env = make_env(video_args, reward_weights=reward_weights, env_kwargs=env_kwargs)
                    if handoff_model is not None:
                        video_env = HandoffStartWrapper(
                            video_env,
                            handoff_model=handoff_model,
                            target_phase=args.specialist_phase,
                            max_steps=getattr(args, "handoff_max_steps", 600),
                            attempts=getattr(args, "handoff_attempts", 20),
                        )
                    elif handoff_phase_models is not None:
                        video_env = PhaseChainHandoffStartWrapper(
                            video_env,
                            phase_models=handoff_phase_models,
                            phase_paths=handoff_phase_paths,
                            target_phase=args.specialist_phase,
                            max_steps=getattr(args, "handoff_max_steps", 1200),
                            attempts=getattr(args, "handoff_attempts", 20),
                        )
                    # Generate video
                    generate_headless_video(
                        env=video_env,
                        model=model,
                        video_path=video_path,
                        duration_sec=8.0,
                        fps=30
                    )
                    video_env.close()
                    print(f"[Video] Video kaydedildi: {video_path}", flush=True)

                    if telegram_notifier is not None and telegram_notifier.enabled:
                        caption = (
                            f"[{algo_name.upper()} WATCH VIDEO]\n"
                            f"Phase: {specialist_phase or 'hopper'}\n"
                            f"Step: {completed_steps}\n"
                            f"Run: {run_dir.name}"
                        )
                        print(f"[Telegram] Video gonderiliyor: {video_path}", flush=True)
                        telegram_notifier.send_video(video_path, caption)
                except Exception as exc:
                    print(f"[Video] Video uretimi veya gonderimi sirasinda hata: {exc}", flush=True)
                    import traceback
                    traceback.print_exc()

        remaining -= chunk_steps

    plot_run(run_dir)
    env.unwrapped.close_viewer()
    print(f"Run klasoru: {run_dir}")
    if telegram_notifier is not None:
        telegram_notifier.send(f"[{algo_name.upper()} TRAIN DONE]\nrun: {run_dir}")


PHASE_MODEL_KEYS = ("climb", "flip", "recovery", "hover")


def load_phase_models(config_path, Algorithm, env):
    config_path = Path(config_path)
    with config_path.open() as file:
        config = json.load(file)

    phase_paths = {}
    for phase in PHASE_MODEL_KEYS:
        path = config.get(phase)
        fallback_flip = config.get("fallback_flip")
        if phase == "flip" and fallback_flip and (not path or not Path(path).exists()):
            path = fallback_flip
        if not path:
            raise ValueError(f"{config_path} icinde '{phase}' modeli yok.")
        phase_paths[phase] = Path(path)

    if config.get("fallback_flip"):
        phase_paths["fallback_flip"] = Path(config["fallback_flip"])

    missing = [str(path) for phase, path in phase_paths.items() if phase != "fallback_flip" and not path.exists()]
    if missing:
        raise FileNotFoundError("Phase model bulunamadi: " + ", ".join(missing))

    cache = {}
    phase_models = {}
    for phase in PHASE_MODEL_KEYS:
        path = phase_paths[phase]
        if path not in cache:
            cache[path] = Algorithm.load(path, env=env, device="cpu")
        phase_models[phase] = cache[path]
    return phase_models, phase_paths


def make_phase_models_config(
    output_path,
    runs_dir="runs",
    algo_name="sac",
    fallback_flip=None,
    require_all=False,
    model_overrides=None,
):
    runs_dir = Path(runs_dir)
    model_overrides = model_overrides or {}
    config = {}
    missing = []
    for phase in PHASE_MODEL_KEYS:
        override = model_overrides.get(phase)
        latest = Path(override) if override else find_latest_specialist_model(runs_dir, algo_name, phase)
        if latest is None:
            if phase == "flip" and fallback_flip:
                config[phase] = str(Path(fallback_flip))
            else:
                missing.append(phase)
                config[phase] = str(
                    runs_dir / f"{algo_name}_{phase}_<run>" / f"{algo_name}_hopper_latest.zip"
                )
        else:
            config[phase] = str(latest)

    if fallback_flip:
        config["fallback_flip"] = str(Path(fallback_flip))

    if require_all and missing:
        raise FileNotFoundError("Eksik specialist model: " + ", ".join(missing))

    output_path = Path(output_path)
    output_path.write_text(json.dumps(config, indent=2) + "\n")
    return config, missing


def find_latest_specialist_model(runs_dir, algo_name, phase):
    candidates = []
    for run_dir in runs_dir.glob(f"{algo_name}_{phase}_*"):
        model_path = run_dir / f"{algo_name}_hopper_latest.zip"
        if model_path.exists():
            candidates.append(model_path)
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def phase_model_for_env(env, phase_models, phase_paths):
    phase = env.current_phase
    if phase not in phase_models:
        phase = "hover"
    return phase, phase_models[phase], str(phase_paths[phase])


def append_sequence_value(sequence, value):
    if value and (not sequence or sequence[-1] != value):
        sequence.append(value)


def watch_model(args, algo_name, Algorithm, reward_weights=None, env_kwargs=None):
    env = make_env(args, reward_weights=reward_weights, env_kwargs=env_kwargs)
    observation_names = tuple(env.observation_names)
    obs, _ = env.reset()
    phase_models = None
    phase_paths = None
    model_path = args.model or f"{algo_name}_hopper_latest.zip"
    if getattr(args, "phase_models_config", None):
        phase_models, phase_paths = load_phase_models(args.phase_models_config, Algorithm, env)
        model = None
    else:
        model = Algorithm.load(model_path, env=env, device="cpu")
    viewer = env.launch_viewer()
    csv_file = None
    writer = None

    if args.csv:
        csv_path = Path(args.csv)
        csv_file = csv_path.open("w", newline="")
        writer = csv.DictWriter(
            csv_file,
            fieldnames=unique_fields((
                "episode",
                "step",
                "reward",
                "episode_reward",
                "episode_length",
                "done",
                "active_model",
                "active_model_path",
                *observation_names,
                *EXTRA_INFO_KEYS,
            )),
        )
        writer.writeheader()

    step_count = 0
    episode_index = 0
    episode_reward = 0.0
    try:
        while viewer.is_running():
            if phase_models is not None:
                active_phase, active_model, active_model_path = phase_model_for_env(env, phase_models, phase_paths)
            else:
                active_phase, active_model, active_model_path = env.current_phase, model, str(model_path)
            action, _ = active_model.predict(obs, deterministic=True)
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
                    "active_model": active_phase,
                    "active_model_path": active_model_path,
                }
                for key in observation_names:
                    row[key] = info.get(key, "")
                for key in EXTRA_INFO_KEYS:
                    row[key] = info.get(key, "")
                writer.writerow(row)
                csv_file.flush()

            if done:
                episode_index += 1
                print("Episode bitti:", {
                    "active_model": active_phase,
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


def evaluate_model(args, algo_name, Algorithm, reward_weights=None, env_kwargs=None):
    env = make_env(args, reward_weights=reward_weights, env_kwargs=env_kwargs)
    observation_names = tuple(env.observation_names)
    phase_models = None
    phase_paths = None
    if getattr(args, "phase_models_config", None):
        phase_models, phase_paths = load_phase_models(args.phase_models_config, Algorithm, env)
        model_paths = [Path(args.phase_models_config)]
    else:
        model_paths = resolve_eval_model_paths(args, algo_name)
    csv_file = None
    writer = None

    if args.csv:
        csv_path = Path(args.csv)
        csv_file = csv_path.open("w", newline="")
        writer = csv.DictWriter(
            csv_file,
            fieldnames=unique_fields((
                "model",
                "episode",
                "step",
                "reward",
                "episode_reward",
                "episode_length",
                "done",
                "active_model",
                "active_model_path",
                *observation_names,
                *EXTRA_INFO_KEYS,
            )),
        )
        writer.writeheader()

    summaries = []
    try:
        for model_path in model_paths:
            model_path = Path(model_path)
            model = None if phase_models is not None else Algorithm.load(model_path, env=env, device="cpu")
            model_summaries = []
            for episode_index in range(1, args.episodes + 1):
                obs, _ = env.reset()
                episode_reward = 0.0
                max_flip_progress = 0.0
                max_axis_rate = 0.0
                max_rel_dist = 0.0
                min_height = float("inf")
                final_info = {}
                final_step = 0
                phase_sequence = [env.current_phase]
                active_model_sequence = []
                for step_count in range(1, args.max_steps + 1):
                    if phase_models is not None:
                        active_phase, active_model, active_model_path = phase_model_for_env(env, phase_models, phase_paths)
                    else:
                        active_phase, active_model, active_model_path = env.current_phase, model, str(model_path)
                    append_sequence_value(active_model_sequence, active_phase)
                    action, _ = active_model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = env.step(action)
                    reward = float(reward)
                    episode_reward += reward
                    done = bool(terminated or truncated)
                    max_flip_progress = max(max_flip_progress, float(info.get("flip_progress", 0.0)))
                    max_axis_rate = max(max_axis_rate, float(info.get("positive_flip_axis_rate", 0.0)))
                    max_rel_dist = max(max_rel_dist, float(info.get("rel_dist", 0.0)))
                    min_height = min(min_height, float(info.get("height", 0.0)))
                    final_info = info
                    final_step = step_count
                    append_sequence_value(phase_sequence, info.get("phase"))

                    if writer is not None:
                        row = {
                            "model": str(model_path),
                            "episode": episode_index,
                            "step": step_count,
                            "reward": reward,
                            "episode_reward": episode_reward,
                            "episode_length": step_count,
                            "done": done,
                            "active_model": active_phase,
                            "active_model_path": active_model_path,
                        }
                        for key in observation_names:
                            row[key] = info.get(key, "")
                        for key in EXTRA_INFO_KEYS:
                            row[key] = info.get(key, "")
                        writer.writerow(row)

                    if done:
                        break

                if csv_file is not None:
                    csv_file.flush()

                summary = {
                    "model": (
                        f"phase-router:{model_path}"
                        if phase_models is not None
                        else str(model_path)
                    ),
                    "episode": episode_index,
                    "steps": final_step,
                    "reward": round(episode_reward, 3),
                    "phase": final_info.get("phase"),
                    "success": final_info.get("success"),
                    "specialist_success": final_info.get("specialist_success"),
                    "specialist_handoff_phase": final_info.get("specialist_handoff_phase"),
                    "fail": final_info.get("fail"),
                    "fail_reason": final_info.get("fail_reason"),
                    "phase_sequence": ">".join(phase_sequence),
                    "active_model_sequence": ">".join(active_model_sequence),
                    "max_flip_progress": round(max_flip_progress, 3),
                    "final_flip_progress": round(float(final_info.get("flip_progress", 0.0)), 3),
                    "max_axis_rate": round(max_axis_rate, 3),
                    "max_rel_dist": round(max_rel_dist, 3),
                    "min_height": round(min_height, 3),
                    "final_height": round(float(final_info.get("height", 0.0)), 3),
                    "final_rel_dist": round(float(final_info.get("rel_dist", 0.0)), 3),
                    "overrotate_penalty": round(float(final_info.get("penalty_overrotate", 0.0)), 3),
                }
                summaries.append(summary)
                model_summaries.append(summary)
                print("Eval episode:", summary, flush=True)

            print_eval_summary("Eval model summary", model_summaries)

        if summaries:
            print_eval_summary("Eval summary", summaries)
    finally:
        if csv_file is not None:
            csv_file.close()
        env.close()


def resolve_eval_model_paths(args, algo_name):
    if args.models_glob:
        model_paths = [Path(path) for path in sorted(glob.glob(args.models_glob))]
    elif args.run_dir:
        run_dir = Path(args.run_dir)
        checkpoint_paths = sorted(
            (run_dir / "checkpoints").glob(f"{algo_name}_hopper_*.zip"),
            key=lambda path: checkpoint_step(path, algo_name),
        )
        latest_path = run_dir / f"{algo_name}_hopper_latest.zip"
        model_paths = checkpoint_paths or ([latest_path] if latest_path.exists() else [])
    else:
        model_paths = [Path(args.model or f"{algo_name}_hopper_latest.zip")]

    model_paths = [path for path in model_paths if path.exists()]
    if not model_paths:
        raise FileNotFoundError("Eval icin model bulunamadi.")
    return model_paths


def checkpoint_step(path, algo_name):
    stem = path.stem
    prefix = f"{algo_name}_hopper_"
    suffix = stem[len(prefix):] if stem.startswith(prefix) else stem
    return int(suffix) if suffix.isdigit() else 10**18


def print_eval_summary(label, summaries):
    if not summaries:
        return
    successes = sum(1 for item in summaries if item["success"])
    specialist_successes = sum(1 for item in summaries if item.get("specialist_success"))
    print(
        f"{label}:",
        {
            "models": len({item["model"] for item in summaries}),
            "episodes": len(summaries),
            "successes": successes,
            "specialist_successes": specialist_successes,
            "best_flip_progress": max(item["max_flip_progress"] for item in summaries),
            "avg_reward": round(sum(item["reward"] for item in summaries) / len(summaries), 3),
            "fail_reasons": {
                reason: sum(1 for item in summaries if item["fail_reason"] == reason)
                for reason in sorted({item["fail_reason"] for item in summaries})
            },
            "phase_sequences": {
                sequence: sum(1 for item in summaries if item.get("phase_sequence") == sequence)
                for sequence in sorted({item.get("phase_sequence", "") for item in summaries})
            },
            "active_model_sequences": {
                sequence: sum(1 for item in summaries if item.get("active_model_sequence") == sequence)
                for sequence in sorted({item.get("active_model_sequence", "") for item in summaries})
            },
            "specialist_handoffs": {
                phase: sum(1 for item in summaries if item.get("specialist_handoff_phase") == phase)
                for phase in sorted({item.get("specialist_handoff_phase", "") for item in summaries})
                if phase
            },
        },
        flush=True,
    )


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
