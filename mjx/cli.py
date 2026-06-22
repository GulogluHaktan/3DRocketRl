from __future__ import annotations

import csv
import copy
import functools
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time

from .env import RocketMJXEnv


def _require_jax():
    try:
        import jax
        import jax.numpy as jp
        from brax.training.agents.sac import checkpoint
        from brax.training.agents.sac import networks
        from brax.training.agents.sac import train as sac_train

        # Monkeypatch _unpmap to handle numpy arrays and non-replicated JAX arrays
        original_unpmap = sac_train._unpmap
        def patched_unpmap(v):
            def leaf_unpmap(x):
                if hasattr(x, "addressable_shards"):
                    return x.addressable_shards[0].data.squeeze(0)
                return x
            return jax.tree_util.tree_map(leaf_unpmap, v)
        sac_train._unpmap = patched_unpmap

        # Monkeypatch sac_train.train to replicate restored checkpoint parameters
        original_train = sac_train.train
        def patched_train(*args, **kwargs):
            restore_path = kwargs.get("restore_checkpoint_path")
            if restore_path is not None:
                original_load = checkpoint.load
                def patched_load(path):
                    params = original_load(path)
                    max_devices = kwargs.get("max_devices_per_host")
                    local_devices = jax.local_device_count()
                    if max_devices is not None:
                        local_devices = min(local_devices, max_devices)
                    return jax.device_put_replicated(params, jax.local_devices()[:local_devices])
                
                checkpoint.load = patched_load
                try:
                    return original_train(*args, **kwargs)
                finally:
                    checkpoint.load = original_load
            else:
                return original_train(*args, **kwargs)

        sac_train.train = patched_train

    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "MJX bagimliliklari yok. `.venv-mjx/bin/python rl.py mjx-doctor` kullan."
        ) from exc
    return jax, jp, checkpoint, networks, sac_train


def _run_command(*command):
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        return f"unavailable: {exc}"


def doctor(args):
    jax, jp, _, _, _ = _require_jax()
    import brax
    import mujoco
    import mujoco.mjx as mujoco_mjx

    env = RocketMJXEnv(curriculum_stage=0)
    reset = jax.jit(env.reset)
    step = jax.jit(env.step)
    state = reset(jax.random.key(args.seed))
    jax.block_until_ready(state.obs)
    state = step(state, jp.zeros(3))
    jax.block_until_ready(state.obs)
    parity = _single_step_parity(env, args.seed)
    devices = [str(device) for device in jax.devices()]
    result = {
        "ok": True,
        "jax": jax.__version__,
        "mujoco": mujoco.__version__,
        "mujoco_mjx": getattr(mujoco_mjx, "__version__", mujoco.__version__),
        "brax": brax.__version__,
        "devices": devices,
        "gpu": _run_command(
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader",
        ),
        "observation_size": int(state.obs.shape[-1]),
        "action_size": env.action_size,
        "height_after_step": float(state.metrics["height"]),
        "single_step_parity": parity,
    }
    print(json.dumps(result, indent=2))
    return result


def _single_step_parity(mjx_env, seed):
    import numpy as np
    import mujoco
    from mujoco import mjx as mujoco_mjx
    from hopper_env import HopperEnv

    classic = HopperEnv(
        start_phase="climb",
        start_z=5.0,
        random_start_z=False,
        include_task_state_observation=True,
        max_thrust=45.0,
    )
    try:
        classic.reset(seed=seed)
        classic.data.qpos[:] = 0.0
        classic.data.qpos[2] = 5.355
        classic.data.qpos[3] = 1.0
        classic.data.qvel[:] = 0.0
        mujoco.mj_forward(classic.model, classic.data)
        classic.step(np.array([0.5, 0.0, 0.0], dtype=np.float32))

        jax, jp, _, _, _ = _require_jax()
        state = mjx_env.reset(jax.random.key(seed))
        qpos = jp.zeros_like(state.pipeline_state.data.qpos)
        qpos = qpos.at[2].set(5.355)
        qpos = qpos.at[3].set(1.0)
        qvel = jp.zeros_like(state.pipeline_state.data.qvel)
        data = state.pipeline_state.data.replace(qpos=qpos, qvel=qvel)
        data = mujoco_mjx.forward(mjx_env.mjx_model, data)
        rocket = state.pipeline_state.replace(
            data=data,
            phase=jp.array(0),
            tvc_cmd=jp.zeros(2),
            last_height=jp.array(5.0),
            last_vertical_velocity=jp.array(0.0),
        )
        state = state.replace(pipeline_state=rocket)
        state = jax.jit(mjx_env.step)(state, jp.zeros(3))
        jax.block_until_ready(state.pipeline_state.data.qpos)
        qpos_delta = np.max(
            np.abs(
                np.asarray(state.pipeline_state.data.qpos)
                - classic.data.qpos
            )
        )
        qvel_delta = np.max(
            np.abs(
                np.asarray(state.pipeline_state.data.qvel)
                - classic.data.qvel
            )
        )
        return {
            "max_qpos_abs_delta": float(qpos_delta),
            "max_qvel_abs_delta": float(qvel_delta),
            "within_behavioral_tolerance": bool(
                qpos_delta <= 0.03 and qvel_delta <= 0.75
            ),
        }
    finally:
        classic.close()


def _benchmark_profile(num_envs, steps, seed):
    jax, jp, _, _, _ = _require_jax()
    env = RocketMJXEnv(curriculum_stage=0)
    keys = jax.random.split(jax.random.key(seed), num_envs)
    reset = jax.jit(jax.vmap(env.reset))
    batched_step = jax.vmap(env.step)

    def rollout(state):
        def body(carry, _):
            action = jp.zeros((num_envs, 3))
            return batched_step(carry, action), None
        return jax.lax.scan(body, state, None, length=steps)[0]

    rollout = jax.jit(rollout)
    state = reset(keys)
    jax.block_until_ready(state.obs)
    compile_start = time.perf_counter()
    state = rollout(state)
    jax.block_until_ready(state.obs)
    compile_sec = time.perf_counter() - compile_start
    start = time.perf_counter()
    state = rollout(state)
    jax.block_until_ready(state.obs)
    elapsed = time.perf_counter() - start
    return {
        "num_envs": num_envs,
        "rollout_steps": steps,
        "jit_compile_sec": compile_sec,
        "elapsed_sec": elapsed,
        "environment_steps_per_sec": num_envs * steps / max(elapsed, 1e-9),
    }


def benchmark(args):
    profiles = []
    for num_envs in args.num_envs:
        try:
            result = _benchmark_profile(num_envs, args.steps, args.seed)
            result["ok"] = True
        except Exception as exc:  # GPU OOM and compile failures are reported per profile.
            result = {"num_envs": num_envs, "ok": False, "error": repr(exc)}
        profiles.append(result)
        print(json.dumps(result, indent=2), flush=True)
    valid = [item for item in profiles if item["ok"]]
    recommendation = max(
        valid,
        key=lambda item: item["environment_steps_per_sec"],
        default=None,
    )
    report = {"profiles": profiles, "recommended": recommendation}
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print("MJX benchmark recommendation:", recommendation)
    return report


def _latest_checkpoint(path):
    path = Path(path).resolve()
    if (path / "sac_network_config.json").exists():
        return path
    candidates = [
        candidate
        for candidate in path.rglob("*")
        if candidate.is_dir()
        and candidate.name.isdigit()
        and (candidate / "sac_network_config.json").exists()
    ]
    if not candidates:
        raise FileNotFoundError(f"MJX checkpoint bulunamadi: {path}")
    return max(candidates, key=lambda item: int(item.name))


def _write_run_metadata(run_dir, config):
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (run_dir / "gpu.txt").write_text(_run_command("nvidia-smi") + "\n")
    (run_dir / "git_diff.patch").write_text(_run_command("git", "diff") + "\n")
    (run_dir / "requirements-mjx.lock").write_text(
        _run_command(str(Path(".venv-mjx/bin/python")), "-m", "pip", "freeze") + "\n"
    )


def train(args):
    if args.seeds:
        base_run_dir = Path(
            args.run_dir
            or Path(args.runs_dir) / f"mjx_sac_chain_{time.strftime('%Y%m%d_%H%M%S')}"
        ).resolve()
        runs = []
        for seed in args.seeds:
            seed_args = copy.copy(args)
            seed_args.seed = seed
            seed_args.seeds = None
            seed_args.run_dir = str(base_run_dir / f"seed_{seed}")
            runs.append(str(_train_single(seed_args)))
        print(json.dumps({"seed_runs": runs}, indent=2))
        return base_run_dir
    return _train_single(args)


def _is_oom_error(exc):
    text = repr(exc).lower()
    return any(token in text for token in ("out of memory", "resource_exhausted", "oom"))


def _train_single(args):
    jax, _, _, networks, sac_train = _require_jax()
    from tensorboardX import SummaryWriter

    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.run_dir or Path(args.runs_dir) / f"mjx_sac_chain_{stamp}").resolve()
    all_stage_steps = [
        int(args.total_env_steps * 0.20),
        int(args.total_env_steps * 0.40),
        args.total_env_steps - int(args.total_env_steps * 0.60),
    ]
    if args.curriculum_stages == 1:
        stage_steps = [args.total_env_steps]
    elif args.curriculum_stages == 2:
        stage_steps = [
            int(args.total_env_steps / 3),
            args.total_env_steps - int(args.total_env_steps / 3),
        ]
    else:
        stage_steps = all_stage_steps
    config = vars(args).copy()
    config["stage_steps"] = stage_steps
    config["jax_devices"] = [str(device) for device in jax.devices()]
    _write_run_metadata(run_dir, config)
    metrics_csv = run_dir / "metrics.csv"
    writer = SummaryWriter(str(run_dir / "tensorboard"))
    restore_path = str(Path(args.resume).resolve()) if args.resume else None
    best_reward = -float("inf")
    best_checkpoint = None

    with metrics_csv.open("w", newline="") as handle:
        csv_writer = csv.DictWriter(
            handle, fieldnames=("stage", "step", "metrics_json")
        )
        csv_writer.writeheader()
        for stage, num_steps in enumerate(stage_steps):
            stage_dir = run_dir / f"stage_{stage + 1}"
            stage_dir.mkdir(parents=True, exist_ok=True)
            env = RocketMJXEnv(
                curriculum_stage=stage,
                episode_length=args.episode_length,
                max_thrust=args.max_thrust,
                max_tvc_deg=args.max_tvc_deg,
            )
            network_factory = functools.partial(
                networks.make_sac_networks,
                hidden_layer_sizes=(256, 256),
            )

            def progress(step, metrics, stage_index=stage):
                nonlocal best_reward, best_checkpoint
                scalar_metrics = {
                    key: float(value)
                    for key, value in metrics.items()
                    if getattr(value, "shape", ()) == ()
                }
                flat = {
                    "stage": stage_index + 1,
                    "step": int(step),
                    **scalar_metrics,
                }
                csv_writer.writerow({
                    "stage": stage_index + 1,
                    "step": int(step),
                    "metrics_json": json.dumps(scalar_metrics, sort_keys=True),
                })
                handle.flush()
                for key, value in flat.items():
                    if key not in {"stage", "step"}:
                        writer.add_scalar(f"stage_{stage_index + 1}/{key}", value, step)
                episode_reward = flat.get("eval/episode_reward", -float("inf"))
                if episode_reward > best_reward:
                    best_reward = episode_reward
                    try:
                        best_checkpoint = str(_latest_checkpoint(stage_dir / "checkpoints"))
                    except FileNotFoundError:
                        pass
                print("[MJX]", flat, flush=True)

            num_evals = max(2, math.ceil(num_steps / args.checkpoint_every) + 1)
            env_candidates = []
            for candidate in (args.num_envs, 512, 256, 128):
                if candidate <= args.num_envs and candidate not in env_candidates:
                    env_candidates.append(candidate)
            last_error = None
            for effective_num_envs in env_candidates:
                effective_replay = (
                    args.max_replay_size
                    if effective_num_envs >= 256
                    else min(args.max_replay_size, 131_072)
                )
                try:
                    _, _, _ = sac_train.train(
                        environment=env,
                        num_timesteps=num_steps,
                        episode_length=args.episode_length,
                        action_repeat=1,
                        num_envs=effective_num_envs,
                        num_eval_envs=min(args.num_eval_envs, effective_num_envs),
                        learning_rate=args.learning_rate,
                        discounting=args.discounting,
                        seed=args.seed,
                        batch_size=args.batch_size,
                        num_evals=num_evals,
                        normalize_observations=True,
                        reward_scaling=args.reward_scaling,
                        tau=args.tau,
                        min_replay_size=args.min_replay_size,
                        max_replay_size=effective_replay,
                        grad_updates_per_step=args.grad_updates_per_step,
                        deterministic_eval=True,
                        network_factory=network_factory,
                        progress_fn=progress,
                        checkpoint_logdir=str(stage_dir / "checkpoints"),
                        restore_checkpoint_path=restore_path,
                    )
                    break
                except Exception as exc:
                    last_error = exc
                    if not _is_oom_error(exc) or effective_num_envs == env_candidates[-1]:
                        raise
                    print(
                        f"[MJX OOM] num_envs={effective_num_envs}; daha kucuk profile geciliyor.",
                        flush=True,
                    )
            if last_error is not None and not (stage_dir / "checkpoints").exists():
                raise last_error
            restore_path = str(_latest_checkpoint(stage_dir / "checkpoints"))
            (stage_dir / "latest_checkpoint.txt").write_text(restore_path + "\n")

    writer.close()
    final_checkpoint = restore_path
    manifest = {
        "final_checkpoint": final_checkpoint,
        "best_checkpoint": best_checkpoint or final_checkpoint,
        "best_eval_reward": best_reward,
        "stages": stage_steps,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"MJX run: {run_dir}")
    print(f"Final checkpoint: {final_checkpoint}")
    return run_dir


def _resolve_checkpoint(model_path):
    path = Path(model_path).resolve()
    if path.is_file() and path.name == "manifest.json":
        manifest = json.loads(path.read_text())
        return Path(manifest["best_checkpoint"])
    if path.is_dir() and (path / "manifest.json").exists():
        manifest = json.loads((path / "manifest.json").read_text())
        return Path(manifest["best_checkpoint"])
    return _latest_checkpoint(path)


def evaluate(args):
    if args.classic:
        return _evaluate_classic(args)
    jax, jp, checkpoint, _, _ = _require_jax()
    checkpoint_path = _resolve_checkpoint(args.model)
    policy = checkpoint.load_policy(checkpoint_path, deterministic=True)
    env = RocketMJXEnv(curriculum_stage=3, episode_length=args.episode_length)
    num_envs = int(args.episodes)
    keys = jax.random.split(jax.random.key(args.seed), num_envs)
    state = jax.jit(jax.vmap(env.reset))(keys)
    step_fn = jax.jit(jax.vmap(env.step))
    policy_keys = jax.random.split(jax.random.key(args.seed + 1), num_envs)
    active = jp.ones((num_envs,), dtype=jp.bool_)
    ever_success = jp.zeros((num_envs,), dtype=jp.bool_)
    transition_max = jp.zeros((num_envs, 4))

    for _ in range(args.episode_length):
        action, _ = policy(state.obs, policy_keys)
        next_state = step_fn(state, action)
        ever_success = ever_success | next_state.pipeline_state.success
        transition_max = jp.maximum(
            transition_max, next_state.pipeline_state.transition_mask
        )
        active = active & ~next_state.done.astype(jp.bool_)
        state = next_state
        if not bool(jp.any(active)):
            break
    jax.block_until_ready(state.obs)
    report = {
        "checkpoint": str(checkpoint_path),
        "episodes": num_envs,
        "chain_success_rate": float(jp.mean(ever_success.astype(jp.float32))),
        "climb_handoff_rate": float(jp.mean(transition_max[:, 0])),
        "flip_handoff_rate": float(jp.mean(transition_max[:, 1])),
        "recovery_handoff_rate": float(jp.mean(transition_max[:, 2])),
        "hover_success_rate": float(jp.mean(transition_max[:, 3])),
        "mean_final_height": float(jp.mean(state.metrics["height"])),
        "mean_max_tvc_deg": float(jp.rad2deg(jp.mean(state.metrics["max_tvc"]))),
        "mean_max_tvc_rate_dps": float(jp.rad2deg(jp.mean(state.metrics["max_tvc_speed"]))),
        "mean_settling_time_sec": float(
            jp.nanmean(
                jp.where(
                    state.pipeline_state.settling_time >= 0.0,
                    state.pipeline_state.settling_time,
                    jp.nan,
                )
            )
        ),
        "mean_transition_height": [
            float(value)
            for value in jp.nanmean(
                state.pipeline_state.transition_height, axis=0
            )
        ],
        "mean_transition_linear_speed": [
            float(value)
            for value in jp.nanmean(
                state.pipeline_state.transition_linear_speed, axis=0
            )
        ],
        "mean_transition_angular_speed": [
            float(value)
            for value in jp.nanmean(
                state.pipeline_state.transition_angular_speed, axis=0
            )
        ],
    }
    report["accepted"] = bool(
        report["climb_handoff_rate"] >= 0.90
        and report["flip_handoff_rate"] >= 0.85
        and report["recovery_handoff_rate"] >= 0.85
        and report["hover_success_rate"] >= 0.90
        and report["chain_success_rate"] >= 0.80
    )
    print(json.dumps(report, indent=2))
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    return report


def _make_classic_env(args):
    from hopper_env import HopperEnv
    return HopperEnv(
        start_phase="climb",
        start_z=2.0,
        random_start_z=True,
        min_start_z=0.5,
        max_start_z=8.0,
        include_task_state_observation=True,
        reward_mode="dense",
        max_thrust=args.max_thrust,
        max_tvc_deg=args.max_tvc_deg,
        flip_target_z=10.0,
        climb_ready_min_z=10.0,
        flip_start_min_z=10.0,
        flip_start_max_z=None,
        flip_start_max_linear_speed=2.0,
        flip_start_max_horizontal_speed=2.0,
        flip_start_min_vertical_velocity=-2.0,
        flip_start_max_vertical_velocity=2.0,
        flip_start_max_angular_speed=0.5,
        flip_start_min_upright=0.97,
        flip_start_max_tvc_angle=0.10,
        flip_start_max_joint_speed=0.8,
    )


def _evaluate_classic(args):
    jax, jp, checkpoint, _, _ = _require_jax()
    import numpy as np
    checkpoint_path = _resolve_checkpoint(args.model)
    policy = checkpoint.load_policy(checkpoint_path, deterministic=True)
    env = _make_classic_env(args)
    successes = 0
    phase_successes = dict(climb=0, flip=0, recovery=0, hover=0)
    fail_reasons = {}
    try:
        for episode in range(args.episodes):
            obs, _ = env.reset(seed=args.seed + episode)
            seen = {env.current_phase}
            final_info = {}
            key = jax.random.key(args.seed + episode)
            for _ in range(args.episode_length):
                action, _ = policy(jp.asarray(obs)[None, :], key)
                action = np.asarray(action[0])
                classic_action = np.array([
                    np.clip((action[0] + 1.0) * 0.5, 0.0, 1.0),
                    action[1],
                    action[2],
                ], dtype=np.float32)
                obs, _, terminated, truncated, final_info = env.step(classic_action)
                seen.add(final_info.get("phase", ""))
                if terminated or truncated:
                    break
            successes += int(bool(final_info.get("success")))
            phase_successes["climb"] += int("flip" in seen)
            phase_successes["flip"] += int("recovery" in seen)
            phase_successes["recovery"] += int("hover" in seen)
            phase_successes["hover"] += int(bool(final_info.get("success")))
            reason = final_info.get("fail_reason", "") or "none"
            fail_reasons[reason] = fail_reasons.get(reason, 0) + 1
    finally:
        env.close()
    report = {
        "backend": "classic-mujoco",
        "checkpoint": str(checkpoint_path),
        "episodes": args.episodes,
        "chain_success_rate": successes / max(args.episodes, 1),
        **{
            f"{phase}_handoff_rate": count / max(args.episodes, 1)
            for phase, count in phase_successes.items()
        },
        "fail_reasons": fail_reasons,
    }
    report["accepted"] = report["chain_success_rate"] >= 0.75
    print(json.dumps(report, indent=2))
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    return report


def watch(args):
    jax, jp, checkpoint, _, _ = _require_jax()
    import numpy as np
    from hopper_env import HopperEnv

    checkpoint_path = _resolve_checkpoint(args.model)
    policy = checkpoint.load_policy(checkpoint_path, deterministic=True)
    env = _make_classic_env(args)
    env.random_start_z = False
    env.start_z = args.start_z
    obs, _ = env.reset(seed=args.seed)
    viewer = env.launch_viewer()
    key = jax.random.key(args.seed)
    try:
        while viewer.is_running():
            action, _ = policy(jp.asarray(obs)[None, :], key)
            action = np.asarray(action[0])
            classic_action = np.array([
                np.clip((action[0] + 1.0) * 0.5, 0.0, 1.0),
                action[1],
                action[2],
            ], dtype=np.float32)
            obs, _, terminated, truncated, info = env.step(classic_action)
            if terminated or truncated:
                print({
                    "success": info.get("success"),
                    "fail": info.get("fail"),
                    "fail_reason": info.get("fail_reason"),
                    "height": info.get("height"),
                    "phase": info.get("phase"),
                })
                obs, _ = env.reset()
    finally:
        env.close_viewer()
