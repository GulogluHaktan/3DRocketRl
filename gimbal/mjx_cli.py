from __future__ import annotations

import csv
import functools
import json
import math
from pathlib import Path
import time

from gimbal.env_mjx import GimbalMJXEnv
from mjx.cli import _latest_checkpoint, _require_jax


def _resolve_checkpoint(model_path):
    path = Path(model_path).resolve()
    if path.is_file() and path.name == "manifest.json":
        return Path(json.loads(path.read_text())["best_checkpoint"])
    if path.is_dir() and (path / "manifest.json").exists():
        return Path(json.loads((path / "manifest.json").read_text())["best_checkpoint"])
    return _latest_checkpoint(path)


def train(args):
    jax, _, _, networks, sac_train = _require_jax()
    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.run_dir or Path(args.runs_dir) / f"mjx_gimbal_sac_{stamp}").resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    stage_steps = _stage_steps(args.total_env_steps, args.curriculum_stages)
    config = vars(args).copy()
    config["stage_steps"] = stage_steps
    config["jax_devices"] = [str(device) for device in jax.devices()]
    (run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    metrics_csv = run_dir / "metrics.csv"
    restore_path = str(Path(args.resume).resolve()) if args.resume else None
    best_reward = -float("inf")
    best_checkpoint = None

    with metrics_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("stage", "step", "metrics_json"))
        writer.writeheader()
        for stage_index, num_steps in enumerate(stage_steps):
            stage_num = stage_index + 1
            stage_dir = run_dir / f"stage_{stage_num}"
            stage_dir.mkdir(parents=True, exist_ok=True)
            env = GimbalMJXEnv(
                stage=stage_num,
                control_rate=args.control_rate,
                fixed_thrust_power=args.fixed_thrust_power,
                max_tvc_deg=args.max_tvc_deg,
                tvc_servo_sec_per_60deg=args.tvc_servo_sec_per_60deg,
                max_tvc_rate_dps=args.max_tvc_rate_dps,
                max_steps=args.max_episode_steps,
            )
            network_factory = functools.partial(
                networks.make_sac_networks,
                hidden_layer_sizes=(256, 256),
            )

            def progress(step, metrics, current_stage=stage_num):
                nonlocal best_reward, best_checkpoint
                scalar_metrics = {
                    key: float(value)
                    for key, value in metrics.items()
                    if getattr(value, "shape", ()) == ()
                }
                writer.writerow({
                    "stage": current_stage,
                    "step": int(step),
                    "metrics_json": json.dumps(scalar_metrics, sort_keys=True),
                })
                handle.flush()
                reward = scalar_metrics.get("eval/episode_reward", -float("inf"))
                if reward > best_reward:
                    best_reward = reward
                    try:
                        best_checkpoint = str(_latest_checkpoint(stage_dir / "checkpoints"))
                    except FileNotFoundError:
                        pass
                print(
                    json.dumps(
                        {
                            "stage": current_stage,
                            "step": int(step),
                            "eval_reward": reward,
                            "eval_success": scalar_metrics.get("eval/episode_success"),
                            "eval_tilt_deg": scalar_metrics.get("eval/episode_tilt_deg"),
                        }
                    ),
                    flush=True,
                )

            num_evals = max(2, math.ceil(num_steps / args.checkpoint_every) + 1)
            sac_train.train(
                environment=env,
                num_timesteps=num_steps,
                episode_length=args.max_episode_steps,
                action_repeat=1,
                num_envs=args.num_envs,
                num_eval_envs=min(args.num_eval_envs, args.num_envs),
                learning_rate=args.learning_rate,
                discounting=args.discounting,
                seed=args.seed,
                batch_size=args.batch_size,
                num_evals=num_evals,
                normalize_observations=True,
                reward_scaling=args.reward_scaling,
                tau=args.tau,
                min_replay_size=args.min_replay_size,
                max_replay_size=args.max_replay_size,
                grad_updates_per_step=args.grad_updates_per_step,
                deterministic_eval=True,
                network_factory=network_factory,
                progress_fn=progress,
                checkpoint_logdir=str(stage_dir / "checkpoints"),
                restore_checkpoint_path=restore_path,
            )
            restore_path = str(_latest_checkpoint(stage_dir / "checkpoints"))
            (stage_dir / "latest_checkpoint.txt").write_text(restore_path + "\n")

    manifest = {
        "final_checkpoint": restore_path,
        "best_checkpoint": best_checkpoint or restore_path,
        "best_eval_reward": best_reward,
        "stages": stage_steps,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"MJX gimbal run: {run_dir}")
    print(f"Final checkpoint: {restore_path}")
    return run_dir


def _stage_steps(total_env_steps, curriculum_stages):
    if curriculum_stages == 1:
        return [int(total_env_steps)]
    if curriculum_stages == 2:
        first = int(total_env_steps * 0.40)
        return [first, int(total_env_steps) - first]
    first = int(total_env_steps * 0.25)
    second = int(total_env_steps * 0.35)
    return [first, second, int(total_env_steps) - first - second]


def evaluate(args):
    jax, jp, checkpoint, _, _ = _require_jax()
    checkpoint_path = _resolve_checkpoint(args.model)
    policy = checkpoint.load_policy(checkpoint_path, deterministic=True)
    env = GimbalMJXEnv(
        stage=args.stage,
        control_rate=args.control_rate,
        fixed_thrust_power=args.fixed_thrust_power,
        max_tvc_deg=args.max_tvc_deg,
        tvc_servo_sec_per_60deg=args.tvc_servo_sec_per_60deg,
        max_tvc_rate_dps=args.max_tvc_rate_dps,
        max_steps=args.max_episode_steps,
    )
    keys = jax.random.split(jax.random.key(args.seed), args.episodes)
    state = jax.jit(jax.vmap(env.reset))(keys)
    step_fn = jax.jit(jax.vmap(env.step))
    policy_keys = jax.random.split(jax.random.key(args.seed + 1), args.episodes)
    active = jp.ones((args.episodes,), dtype=jp.bool_)
    ever_success = jp.zeros((args.episodes,), dtype=jp.bool_)
    final_tilt = state.metrics["tilt_deg"]

    for _ in range(args.max_episode_steps):
        action, _ = policy(state.obs, policy_keys)
        next_state = step_fn(state, action)
        ever_success = ever_success | (next_state.metrics["success"] > 0.5)
        final_tilt = jp.where(active, next_state.metrics["tilt_deg"], final_tilt)
        active = active & ~next_state.done.astype(jp.bool_)
        state = next_state
        if not bool(jp.any(active)):
            break
    jax.block_until_ready(state.obs)

    report = {
        "backend": "mjx-gimbal",
        "checkpoint": str(checkpoint_path),
        "stage": args.stage,
        "episodes": args.episodes,
        "success_rate": float(jp.mean(ever_success.astype(jp.float32))),
        "mean_final_tilt_deg": float(jp.mean(final_tilt)),
        "mean_reward": float(jp.mean(state.reward)),
    }
    report["accepted"] = report["success_rate"] >= args.acceptance_success_rate
    print(json.dumps(report, indent=2))
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    return report


def watch(args):
    jax, jp, checkpoint, _, _ = _require_jax()
    import numpy as np
    from gimbal.env import GimbalEnv

    checkpoint_path = _resolve_checkpoint(args.model)
    policy = checkpoint.load_policy(checkpoint_path, deterministic=True)
    env = GimbalEnv(
        stage=args.stage,
        control_rate=args.control_rate,
        fixed_thrust_power=args.fixed_thrust_power,
        max_tvc_deg=args.max_tvc_deg,
        tvc_servo_sec_per_60deg=args.tvc_servo_sec_per_60deg,
        max_tvc_rate_dps=args.max_tvc_rate_dps,
        max_steps=args.max_episode_steps,
        render_mode="human",
    )
    try:
        for episode in range(args.episodes):
            obs, _ = env.reset(seed=args.seed + episode)
            key = jax.random.key(args.seed + episode)
            done = False
            while not done:
                action, _ = policy(jp.asarray(obs)[None, :], key)
                obs, _, terminated, truncated, info = env.step(np.asarray(action[0]))
                done = terminated or truncated
            print(
                {
                    "episode": episode + 1,
                    "success": info.get("success"),
                    "tilt_deg": info.get("tilt_deg"),
                    "fallen": info.get("fallen"),
                },
                flush=True,
            )
    finally:
        env.close()
