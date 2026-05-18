import argparse
import csv
import shutil
import time
from pathlib import Path

import numpy as np

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_checker import check_env
    from stable_baselines3.common.logger import configure
    from stable_baselines3.common.monitor import Monitor
except ModuleNotFoundError:
    PPO = None
    check_env = None
    configure = None
    Monitor = None

BODY_START_Z = 0.40
HOVER_TARGET_Z = 10.0
MODEL_PREFIX = "ppo_hopper"
TASKS = ("takeoff", "hover", "landing")
PHASE_TASKS = ("takeoff", "hover", "landing")
PHASE_TIMESTEPS = 250_000
HOVER_SECONDS = 5.0
DEFAULT_LOG_DIR = "training_logs"
STALE_MODEL_DIR = Path("archive_unused") / "stale_hopper_models"
MONITOR_INFO_KEYS = (
    "success",
    "crash",
    "too_far",
    "touchdown_miss",
    "out_of_fuel",
    "height",
    "target_z",
    "xy_distance",
    "speed",
    "vertical_velocity",
    "vertical_speed",
    "descent_speed",
    "horizontal_speed",
    "angular_speed",
    "upright_score",
    "fuel",
    "main",
    "yaw_ctrl_deg",
    "pitch_ctrl_deg",
    "flip_angle_deg",
    "landing_start_z",
)


def model_name(task):
    return f"{MODEL_PREFIX}_{task}"


def transfer_candidates(task):
    if task == "takeoff":
        return []
    if task == "hover":
        return ["takeoff"]
    if task == "landing":
        return ["hover", "takeoff"]
    if task == "flip":
        return ["hover", "takeoff"]
    return []


def require_sb3():
    if PPO is None:
        raise ModuleNotFoundError(
            "stable_baselines3 bulunamadı. Train/watch için requirements.txt "
            "kurulu venv'i aktif etmen lazım."
        )


def make_raw_env(task, landing_start_z=10.0):
    if task == "takeoff":
        from hopper_env_takeoff import HopperTakeoffEnv

        return HopperTakeoffEnv(landing_start_z=landing_start_z)

    if task == "hover":
        from hopper_env_hover import HopperHoverEnv

        return HopperHoverEnv(landing_start_z=landing_start_z)

    if task == "landing":
        from hopper_env_landing import HopperLandingEnv

        return HopperLandingEnv(landing_start_z=landing_start_z)

    raise ValueError(f"Unknown task: {task}")


def load_policy_for_task(task, env=None, required=True, allow_stale=False):
    require_sb3()
    primary = Path(f"{model_name(task)}.zip")

    if primary.exists():
        print(f"{task} modeli yüklendi: {primary.name}")
        return PPO.load(model_name(task), env=env, device="cpu")

    stale = STALE_MODEL_DIR / primary.name
    if allow_stale and stale.exists():
        print(f"{task} eski arşiv modeli yüklendi: {stale}")
        return PPO.load(str(stale), env=env, device="cpu")

    if required:
        message = (
            f"Model bulunamadı: {primary.name}\n"
            f"Eğitmek için: python hopper_rl.py --mode train --task {task} --timesteps 250000\n"
            f"Eski arşiv modelini denemek için: python hopper_rl.py --mode watch --task {task} --allow-stale"
        )
        raise FileNotFoundError(message)

    return None


def make_model(env):
    require_sb3()
    return PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        device="cpu",
    )


def task_log_dir(task, log_dir=DEFAULT_LOG_DIR):
    return Path(log_dir) / model_name(task)


def make_training_env(task, log_dir=DEFAULT_LOG_DIR, landing_start_z=10.0):
    require_sb3()
    raw_env = make_raw_env(task, landing_start_z=landing_start_z)
    check_env(raw_env, warn=True)

    log_path = task_log_dir(task, log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    return Monitor(
        raw_env,
        filename=str(log_path / "episodes"),
        info_keywords=MONITOR_INFO_KEYS,
    )


def set_training_logger(model, task, log_dir=DEFAULT_LOG_DIR):
    require_sb3()
    log_path = task_log_dir(task, log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    model.set_logger(configure(str(log_path), ["stdout", "csv"]))
    print(f"CSV log klasörü: {log_path}")


def reset_task_logs(task, log_dir=DEFAULT_LOG_DIR):
    log_path = task_log_dir(task, log_dir)
    if not log_path.exists():
        return

    for filename in ("progress.csv", "episodes.monitor.csv", "monitor.csv"):
        path = log_path / filename
        if path.exists():
            path.unlink()
            print(f"Log silindi: {path}")


def backup_existing_model(task):
    model_path = Path(f"{model_name(task)}.zip")
    if not model_path.exists():
        return

    backup_dir = Path("model_backups")
    backup_dir.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    backup_path = backup_dir / f"{model_name(task)}_{stamp}.zip"
    shutil.copy2(model_path, backup_path)
    print(f"Model yedeği alındı: {backup_path}")


def train_task(
    task,
    total_timesteps,
    fresh=False,
    transfer=False,
    log_dir=DEFAULT_LOG_DIR,
    landing_start_z=10.0,
    reset_log=False,
):
    if reset_log:
        reset_task_logs(task, log_dir=log_dir)

    env = make_training_env(
        task,
        log_dir=log_dir,
        landing_start_z=landing_start_z,
    )

    existing = Path(f"{model_name(task)}.zip")
    backup_existing_model(task)

    if existing.exists() and not fresh:
        print(f"{existing.name} bulundu, üstüne eğitime devam ediliyor.")
        model = PPO.load(model_name(task), env=env, device="cpu")
    else:
        model = None
        if transfer and not fresh:
            for candidate in transfer_candidates(task):
                candidate_path = Path(f"{model_name(candidate)}.zip")
                if candidate_path.exists():
                    print(
                        f"{task} eğitimi {candidate_path.name} üstünden başlıyor."
                    )
                    model = PPO.load(model_name(candidate), env=env, device="cpu")
                    break

        if model is None:
            print(f"{task} modeli sıfırdan eğitiliyor.")
            model = make_model(env)

    set_training_logger(model, task, log_dir=log_dir)
    model.learn(total_timesteps=total_timesteps, reset_num_timesteps=False)
    model.save(model_name(task))
    print(f"Kaydedildi: {model_name(task)}.zip")


def train_all(
    total_timesteps,
    fresh=False,
    transfer=False,
    log_dir=DEFAULT_LOG_DIR,
    landing_start_z=10.0,
    reset_log=False,
):
    for task in PHASE_TASKS:
        print(f"\n=== {task.upper()} TRAIN ===")
        train_task(
            task,
            total_timesteps,
            fresh=fresh,
            transfer=transfer,
            log_dir=log_dir,
            landing_start_z=landing_start_z,
            reset_log=reset_log,
        )


def read_csv_rows(path, skip_comment_lines=False):
    if not path.exists():
        return []

    with path.open(newline="") as csv_file:
        if skip_comment_lines:
            lines = [line for line in csv_file if not line.startswith("#")]
            return list(csv.DictReader(lines))

        return list(csv.DictReader(csv_file))


def to_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def to_bool(value):
    return str(value).lower() in {"true", "1", "1.0"}


def summarize_logs(task, log_dir=DEFAULT_LOG_DIR, tail=20):
    log_path = task_log_dir(task, log_dir)
    progress_rows = read_csv_rows(log_path / "progress.csv")
    monitor_rows = read_csv_rows(
        log_path / "episodes.monitor.csv",
        skip_comment_lines=True,
    )

    if not monitor_rows:
        monitor_rows = read_csv_rows(log_path / "monitor.csv", skip_comment_lines=True)

    print(f"Log klasörü: {log_path}")

    if progress_rows:
        last = progress_rows[-1]
        fields = [
            "time/total_timesteps",
            "rollout/ep_rew_mean",
            "rollout/ep_len_mean",
            "train/approx_kl",
            "train/explained_variance",
            "train/value_loss",
            "train/std",
        ]
        print("Son progress:")
        for field in fields:
            if field in last:
                print(f"  {field}: {last[field]}")
    else:
        print("progress.csv bulunamadı.")

    if monitor_rows:
        recent = monitor_rows[-tail:]
        success_rate = sum(to_bool(row.get("success")) for row in recent) / len(recent)
        crash_rate = sum(to_bool(row.get("crash")) for row in recent) / len(recent)
        too_far_rate = sum(to_bool(row.get("too_far")) for row in recent) / len(recent)
        touchdown_miss_rate = (
            sum(to_bool(row.get("touchdown_miss")) for row in recent)
            / len(recent)
        )
        avg_reward = sum(to_float(row.get("r")) for row in recent) / len(recent)
        avg_len = sum(to_float(row.get("l")) for row in recent) / len(recent)
        avg_height = sum(to_float(row.get("height")) for row in recent) / len(recent)
        avg_fuel = sum(to_float(row.get("fuel")) for row in recent) / len(recent)
        avg_main = sum(to_float(row.get("main")) for row in recent) / len(recent)
        avg_yaw = sum(abs(to_float(row.get("yaw_ctrl_deg"))) for row in recent) / len(recent)
        avg_pitch = sum(abs(to_float(row.get("pitch_ctrl_deg"))) for row in recent) / len(recent)

        print(f"Son {len(recent)} episode:")
        print(f"  success_rate: {success_rate:.2f}")
        print(f"  crash_rate: {crash_rate:.2f}")
        print(f"  too_far_rate: {too_far_rate:.2f}")
        print(f"  touchdown_miss_rate: {touchdown_miss_rate:.2f}")
        print(f"  avg_reward: {avg_reward:.2f}")
        print(f"  avg_len: {avg_len:.1f}")
        print(f"  avg_height: {avg_height:.2f}")
        print(f"  avg_fuel: {avg_fuel:.2f}")
        print(f"  avg_main: {avg_main:.2f}")
        print(f"  avg_abs_yaw_deg: {avg_yaw:.2f}")
        print(f"  avg_abs_pitch_deg: {avg_pitch:.2f}")
    else:
        print("monitor.csv bulunamadı.")


def watch_task(
    task,
    deterministic=True,
    allow_stale=False,
    landing_start_z=10.0,
):
    env = make_raw_env(task, landing_start_z=landing_start_z)
    model = load_policy_for_task(task, allow_stale=allow_stale)
    obs, _ = env.reset()
    viewer = env.launch_viewer()

    try:
        while viewer.is_running():
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                print("Episode bitti:", info)
                time.sleep(0.8)
                obs, _ = env.reset()

    except KeyboardInterrupt:
        print("Ctrl+C ile çıkıldı.")

    finally:
        env.close_viewer()
        print("Viewer kapatıldı.")


def smoke(task, landing_start_z=10.0):
    env = make_raw_env(task, landing_start_z=landing_start_z)
    obs, _ = env.reset()
    viewer = env.launch_viewer()

    fixed_actions = {
        "takeoff": np.array([0.85, 0.0, 0.0], dtype=np.float32),
        "hover": np.array([0.0, 0.0, 0.0], dtype=np.float32),
        "landing": np.array([-0.15, 0.0, 0.0], dtype=np.float32),
        "flip": np.array([1.0, 1.0, 0.0], dtype=np.float32),
    }
    action = fixed_actions[task]
    step_count = 0

    try:
        while viewer.is_running():
            obs, reward, terminated, truncated, info = env.step(action)
            step_count += 1
            if step_count % 60 == 0:
                print("Smoke:", info)
            if terminated or truncated:
                print("Smoke episode bitti:", info)
                time.sleep(0.8)
                obs, _ = env.reset()
                step_count = 0

    except KeyboardInterrupt:
        print("Ctrl+C ile çıkıldı.")

    finally:
        env.close_viewer()
        print("Viewer kapatıldı.")


def run_model_steps(env, model, task, max_steps, stop_on_success=True):
    obs = env.set_task(task)

    for _ in range(max_steps):
        if env.viewer is None or not env.viewer.is_running():
            return False, obs, {}

        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        if info.get("success", False) and stop_on_success:
            return True, obs, info

        if terminated or truncated:
            if info.get("success", False) and not stop_on_success:
                env.success_counter = 0
                continue
            return bool(info.get("success", False)), obs, info

    return True, obs, {"task": task, "timeout": True}


def mission(allow_stale=False, landing_start_z=10.0):
    takeoff_model = load_policy_for_task("takeoff", allow_stale=allow_stale)
    hover_model = load_policy_for_task("hover", allow_stale=allow_stale)
    landing_model = load_policy_for_task("landing", allow_stale=allow_stale)

    env = make_raw_env(task="takeoff", landing_start_z=landing_start_z)
    obs, _ = env.reset()
    viewer = env.launch_viewer()

    hover_steps = int(HOVER_SECONDS / (env.model.opt.timestep * env.frame_skip))

    try:
        while viewer.is_running():
            print("Faz: takeoff")
            takeoff_ok, obs, info = run_model_steps(
                env,
                takeoff_model,
                "takeoff",
                max_steps=900,
            )

            print("Takeoff:", {"ok": takeoff_ok, **info})

            if takeoff_ok:
                print(f"Faz: hover ({HOVER_SECONDS:.1f} saniye)")
                env.target_pos = np.array(
                    [0.0, 0.0, HOVER_TARGET_Z],
                    dtype=np.float32,
                )
                hover_ok, obs, info = run_model_steps(
                    env,
                    hover_model,
                    "hover",
                    max_steps=hover_steps,
                    stop_on_success=False,
                )
            else:
                hover_ok = False
                info = {}

            print("Hover:", {"ok": hover_ok, **info})

            if takeoff_ok and hover_ok:
                print("Faz: landing")
                env.target_pos = np.array(
                    [0.0, 0.0, BODY_START_Z],
                    dtype=np.float32,
                )
                landing_ok, obs, info = run_model_steps(
                    env,
                    landing_model,
                    "landing",
                    max_steps=900,
                )
            else:
                landing_ok = False
                info = {}

            print("Landing:", {"ok": landing_ok, **info})
            time.sleep(1.0)

            if viewer.is_running():
                obs, _ = env.reset()

    except KeyboardInterrupt:
        print("Ctrl+C ile çıkıldı.")

    finally:
        env.close_viewer()
        print("Viewer kapatıldı.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["train", "train_all", "watch", "mission", "smoke", "summary"],
        default="watch",
    )
    parser.add_argument(
        "--task",
        choices=TASKS,
        default="takeoff",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=PHASE_TIMESTEPS,
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Ignore saved and transfer models, then start a fresh policy.",
    )
    parser.add_argument(
        "--transfer",
        action="store_true",
        help="When no same-task model exists, initialize from a previous phase model.",
    )
    parser.add_argument(
        "--log-dir",
        default=DEFAULT_LOG_DIR,
        help="Directory for SB3 progress CSV and episode monitor CSV logs.",
    )
    parser.add_argument(
        "--tail",
        type=int,
        default=20,
        help="Episode count used by summary mode.",
    )
    parser.add_argument(
        "--landing-start-z",
        type=float,
        default=10.0,
        help="Landing reset altitude. Use 3, 6, then 10 for curriculum.",
    )
    parser.add_argument(
        "--reset-log",
        action="store_true",
        help="Delete this task's CSV logs before training.",
    )
    parser.add_argument(
        "--allow-stale",
        action="store_true",
        help="Allow loading archived stale hopper models for quick visualization.",
    )
    args = parser.parse_args()

    if args.mode == "train":
        train_task(
            args.task,
            args.timesteps,
            fresh=args.fresh,
            transfer=args.transfer,
            log_dir=args.log_dir,
            landing_start_z=args.landing_start_z,
            reset_log=args.reset_log,
        )
    elif args.mode == "train_all":
        train_all(
            args.timesteps,
            fresh=args.fresh,
            transfer=args.transfer,
            log_dir=args.log_dir,
            landing_start_z=args.landing_start_z,
            reset_log=args.reset_log,
        )
    elif args.mode == "mission":
        mission(
            allow_stale=args.allow_stale,
            landing_start_z=args.landing_start_z,
        )
    elif args.mode == "smoke":
        smoke(args.task, landing_start_z=args.landing_start_z)
    elif args.mode == "summary":
        summarize_logs(args.task, log_dir=args.log_dir, tail=args.tail)
    else:
        try:
            watch_task(
                args.task,
                allow_stale=args.allow_stale,
                landing_start_z=args.landing_start_z,
            )
        except FileNotFoundError as exc:
            print(exc)


if __name__ == "__main__":
    main()
