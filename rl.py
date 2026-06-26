from __future__ import annotations

import argparse
import importlib
from pathlib import Path

ALGORITHMS = {
    "sac": "algorithms.sac",
    "ppo": "algorithms.ppo",
    "td3": "algorithms.td3",
}


def load_algo(name):
    try:
        return importlib.import_module(ALGORITHMS[name])
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "RL algoritmalari icin stable-baselines3 gerekli: "
            "pip install stable-baselines3"
        ) from exc


def add_env_args(parser):
    parser.add_argument("--start-z", type=float, default=10.0)
    parser.add_argument("--fixed-start-z", action="store_true")
    parser.add_argument("--random-start-z", action="store_true")
    parser.add_argument("--min-start-z", type=float, default=0.5)
    parser.add_argument("--max-start-z", type=float, default=10.0)
    parser.add_argument("--start-phase", choices=("climb", "flip", "recovery", "hover"), default="climb")
    parser.add_argument("--max-thrust", type=float, default=None)
    parser.add_argument("--max-tvc-deg", type=float, default=20.0)
    parser.add_argument("--tvc-servo-sec-per-60deg", type=float, default=0.13)
    parser.add_argument("--phase-start-roughness", type=float, default=0.0)
    parser.add_argument(
        "--reset-profile",
        choices=("legacy", "handoff-mix"),
        default="legacy",
    )
    parser.add_argument("--legacy-reset-probability", type=float, default=0.30)
    parser.add_argument("--flip-target-z", type=float, default=None)
    parser.add_argument("--flip-start-min-z", type=float, default=None)
    parser.add_argument("--flip-start-max-z", type=float, default=None)
    parser.add_argument("--climb-ready-min-z", type=float, default=None)


def add_train_args(parser):
    parser.add_argument("--algo", choices=sorted(ALGORITHMS), default="sac")
    parser.add_argument("--timesteps", type=int, default=250_000)
    parser.add_argument("--chunk-steps", type=int, default=25_000)
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--specialist-phase", choices=("climb", "flip", "recovery", "hover"), default=None)
    parser.add_argument("--handoff-model", default=None)
    parser.add_argument("--handoff-phase-models-config", default=None)
    parser.add_argument("--handoff-max-steps", type=int, default=1500)
    parser.add_argument("--handoff-attempts", type=int, default=20)
    parser.add_argument("--telegram-config", default="configs/telegram_secrets.json")
    parser.add_argument("--telegram-every", type=int, default=10_000)
    parser.add_argument("--no-telegram", action="store_true")
    parser.add_argument("--telegram-video-every", type=int, default=100_000)
    parser.add_argument("--no-telegram-video", action="store_true")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=0.97)

    # Off-policy algorithm args.
    parser.add_argument("--buffer-size", type=int, default=300_000)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--learning-starts", type=int, default=5_000)
    parser.add_argument("--sac-ent-coef", default="auto_0.02")
    parser.add_argument(
        "--publish-root-model",
        action="store_true",
        help="Kabul edilen modeli repo kokundeki *_latest.zip yoluna da yazar.",
    )
    parser.add_argument("--acceptance-episodes", type=int, default=100)
    parser.add_argument("--acceptance-eval-every", type=int, default=25_000)

    # PPO args.
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.0)

    add_env_args(parser)


def add_watch_args(parser):
    parser.add_argument("--algo", choices=sorted(ALGORITHMS), default="sac")
    parser.add_argument("--model", default=None)
    parser.add_argument("--phase-models-config", default=None)
    parser.add_argument("--specialist-phase", choices=("climb", "flip", "recovery", "hover"), default=None)
    parser.add_argument("--csv", default=None)
    add_env_args(parser)
    parser.set_defaults(fixed_start_z=True)


def add_eval_args(parser):
    parser.add_argument("--algo", choices=sorted(ALGORITHMS), default="sac")
    parser.add_argument("--model", default=None)
    parser.add_argument("--phase-models-config", default=None)
    parser.add_argument("--specialist-phase", choices=("climb", "flip", "recovery", "hover"), default=None)
    parser.add_argument("--models-glob", default=None)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=1200)
    parser.add_argument("--csv", default=None)
    add_env_args(parser)
    parser.set_defaults(fixed_start_z=True)


def add_mjx_common_args(parser):
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episode-length", type=int, default=1200)
    parser.add_argument("--max-thrust", type=float, default=45.0)
    parser.add_argument("--max-tvc-deg", type=float, default=20.0)
    parser.add_argument(
        "--specialist-phase",
        choices=("climb", "flip", "recovery", "hover"),
        default=None,
        help="Specialist phase to train or evaluate. Terminates the episode with success upon completion."
    )


def add_mjx_train_args(parser):
    add_mjx_common_args(parser)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--total-env-steps", type=int, default=50_000_000)
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--num-eval-envs", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--min-replay-size", type=int, default=32_768)
    parser.add_argument("--max-replay-size", type=int, default=262_144)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--discounting", type=float, default=0.995)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--grad-updates-per-step", type=int, default=16)
    parser.add_argument("--reward-scaling", type=float, default=1.0)
    parser.add_argument("--checkpoint-every", type=int, default=2_000_000)
    parser.add_argument(
        "--curriculum-stages", type=int, choices=(1, 2, 3), default=3
    )
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--resume", default=None)


def add_gimbal_mjx_common_args(parser):
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stage", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--control-rate", type=float, default=50.0)
    parser.add_argument("--fixed-thrust-power", type=float, default=1.0)
    parser.add_argument("--max-tvc-deg", type=float, default=20.0)
    parser.add_argument("--tvc-servo-sec-per-60deg", type=float, default=0.13)
    parser.add_argument("--max-tvc-rate-dps", type=float, default=120.0)
    parser.add_argument("--max-episode-steps", type=int, default=500)


def add_gimbal_mjx_train_args(parser):
    add_gimbal_mjx_common_args(parser)
    parser.add_argument("--total-env-steps", type=int, default=8_000_000)
    parser.add_argument("--curriculum-stages", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--num-envs", type=int, default=512)
    parser.add_argument("--num-eval-envs", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--min-replay-size", type=int, default=8_192)
    parser.add_argument("--max-replay-size", type=int, default=262_144)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--discounting", type=float, default=0.995)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--grad-updates-per-step", type=int, default=16)
    parser.add_argument("--reward-scaling", type=float, default=1.0)
    parser.add_argument("--checkpoint-every", type=int, default=100_000)
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--resume", default=None)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)

    train_parser = sub.add_parser("train")
    add_train_args(train_parser)

    train_specialists_parser = sub.add_parser("train-specialists")
    add_train_args(train_specialists_parser)
    train_specialists_parser.add_argument("--timesteps-per-phase", type=int, default=None)
    train_specialists_parser.add_argument(
        "--phases",
        nargs="+",
        choices=("climb", "flip", "recovery", "hover"),
        default=("climb", "flip", "recovery", "hover"),
    )
    for phase in ("climb", "flip", "recovery", "hover"):
        train_specialists_parser.add_argument(
            f"--{phase}-model",
            default=None,
            help=f"{phase} fine-tune baslangic checkpoint'i.",
        )
    train_specialists_parser.set_defaults(
        reset_profile="handoff-mix",
        learning_rate=3e-5,
        chunk_steps=25_000,
    )

    watch_parser = sub.add_parser("watch")
    add_watch_args(watch_parser)

    eval_parser = sub.add_parser("eval")
    add_eval_args(eval_parser)

    mjx_doctor_parser = sub.add_parser("mjx-doctor")
    add_mjx_common_args(mjx_doctor_parser)

    mjx_benchmark_parser = sub.add_parser("benchmark-mjx")
    mjx_benchmark_parser.add_argument(
        "--num-envs", type=int, nargs="+", default=(128, 256, 512)
    )
    mjx_benchmark_parser.add_argument("--steps", type=int, default=64)
    mjx_benchmark_parser.add_argument("--seed", type=int, default=0)
    mjx_benchmark_parser.add_argument(
        "--output", default="runs/mjx_benchmark.json"
    )

    mjx_train_parser = sub.add_parser("train-mjx")
    add_mjx_train_args(mjx_train_parser)

    gimbal_mjx_train_parser = sub.add_parser("train-gimbal-mjx")
    add_gimbal_mjx_train_args(gimbal_mjx_train_parser)

    gimbal_mjx_eval_parser = sub.add_parser("eval-gimbal-mjx")
    add_gimbal_mjx_common_args(gimbal_mjx_eval_parser)
    gimbal_mjx_eval_parser.add_argument("--model", required=True)
    gimbal_mjx_eval_parser.add_argument("--episodes", type=int, default=100)
    gimbal_mjx_eval_parser.add_argument("--output", default=None)
    gimbal_mjx_eval_parser.add_argument("--acceptance-success-rate", type=float, default=0.80)

    gimbal_mjx_watch_parser = sub.add_parser("watch-gimbal-mjx")
    add_gimbal_mjx_common_args(gimbal_mjx_watch_parser)
    gimbal_mjx_watch_parser.add_argument("--model", required=True)
    gimbal_mjx_watch_parser.add_argument("--episodes", type=int, default=5)

    mjx_eval_parser = sub.add_parser("eval-mjx")
    add_mjx_common_args(mjx_eval_parser)
    mjx_eval_parser.add_argument("--model", required=True)
    mjx_eval_parser.add_argument("--episodes", type=int, default=1000)
    mjx_eval_parser.add_argument("--output", default=None)
    mjx_eval_parser.add_argument(
        "--classic",
        action="store_true",
        help="Policy'yi klasik MuJoCo HopperEnv uzerinde degerlendirir.",
    )

    mjx_watch_parser = sub.add_parser("watch-mjx")
    add_mjx_common_args(mjx_watch_parser)
    mjx_watch_parser.add_argument("--model", required=True)
    mjx_watch_parser.add_argument("--start-z", type=float, default=2.0)

    plot_parser = sub.add_parser("plot")
    plot_parser.add_argument("run_dir")

    phase_config_parser = sub.add_parser("make-phase-config")
    phase_config_parser.add_argument("--output", default="phase_models.json")
    phase_config_parser.add_argument("--runs-dir", default="runs")
    phase_config_parser.add_argument("--climb-model", default=None)
    phase_config_parser.add_argument("--flip-model", default=None)
    phase_config_parser.add_argument("--recovery-model", default=None)
    phase_config_parser.add_argument("--hover-model", default=None)
    phase_config_parser.add_argument(
        "--fallback-flip",
        default=None,
    )
    phase_config_parser.add_argument("--require-all", action="store_true")

    args = parser.parse_args()

    if getattr(args, "handoff_model", None) and getattr(args, "handoff_phase_models_config", None):
        parser.error("--handoff-model ve --handoff-phase-models-config ayni anda kullanilamaz.")

    if (
        hasattr(args, "specialist_phase")
        and args.specialist_phase is not None
        and hasattr(args, "start_phase")
        and args.start_phase != args.specialist_phase
        and not getattr(args, "handoff_model", None)
        and not getattr(args, "handoff_phase_models_config", None)
    ):
        parser.error(
            "--specialist-phase ile --start-phase ayni olmali "
            f"({args.specialist_phase!r} != {args.start_phase!r})."
        )

    if args.mode == "train":
        load_algo(args.algo).train(args)
    elif args.mode == "train-specialists":
        import copy
        start_z_map = {
            "climb": 2.0,
            "flip": 11.0,
            "recovery": 9.0,
            "hover": 5.0,
        }
        timesteps_map = {
            "climb": 100_000,
            "flip": 150_000,
            "recovery": 200_000,
            "hover": 150_000,
        }
        algo_mod = load_algo(args.algo)
        for phase in args.phases:
            print(f"\n=========================================")
            print(f"STARTING PHASE SPECIALIST TRAINING: {phase.upper()}")
            print(f"=========================================\n")
            phase_args = copy.deepcopy(args)
            phase_args.specialist_phase = phase
            phase_model_arg = getattr(args, f"{phase}_model")
            default_phase_model = str(
                Path("models/current") / f"{args.algo}_{phase}_latest.zip"
            )
            phase_args.resume = phase_model_arg or (
                default_phase_model
                if Path(default_phase_model).exists()
                else args.resume
            )
            use_real_handoffs = (
                phase != "climb"
                and bool(phase_args.handoff_phase_models_config)
            )
            if phase == "climb":
                phase_args.handoff_phase_models_config = None
                phase_args.handoff_model = None
            phase_args.start_phase = "climb" if use_real_handoffs else phase
            phase_args.start_z = start_z_map[phase]
            if phase == "climb" or use_real_handoffs:
                phase_args.fixed_start_z = False
                phase_args.random_start_z = True
            else:
                phase_args.fixed_start_z = True
                phase_args.random_start_z = False
            if phase_args.max_thrust is None:
                phase_args.max_thrust = 45.0
            phase_args.timesteps = args.timesteps_per_phase or timesteps_map[phase]
            phase_args.run_dir = None
            algo_mod.train(phase_args)
    elif args.mode == "watch":
        load_algo(args.algo).watch(args)
    elif args.mode == "eval":
        load_algo(args.algo).evaluate(args)
    elif args.mode == "mjx-doctor":
        from mjx import cli as mjx_cli
        mjx_cli.doctor(args)
    elif args.mode == "benchmark-mjx":
        from mjx import cli as mjx_cli
        mjx_cli.benchmark(args)
    elif args.mode == "train-mjx":
        from mjx import cli as mjx_cli
        mjx_cli.train(args)
    elif args.mode == "train-gimbal-mjx":
        from gimbal import mjx_cli
        mjx_cli.train(args)
    elif args.mode == "eval-gimbal-mjx":
        from gimbal import mjx_cli
        mjx_cli.evaluate(args)
    elif args.mode == "watch-gimbal-mjx":
        from gimbal import mjx_cli
        mjx_cli.watch(args)
    elif args.mode == "eval-mjx":
        from mjx import cli as mjx_cli
        mjx_cli.evaluate(args)
    elif args.mode == "watch-mjx":
        from mjx import cli as mjx_cli
        mjx_cli.watch(args)
    elif args.mode == "plot":
        from rl_common import plot_run
        plot_run(args.run_dir)
    elif args.mode == "make-phase-config":
        from rl_common import make_phase_models_config
        try:
            config, missing = make_phase_models_config(
                args.output,
                runs_dir=args.runs_dir,
                algo_name="sac",
                fallback_flip=args.fallback_flip,
                require_all=args.require_all,
                model_overrides={
                    "climb": args.climb_model,
                    "flip": args.flip_model,
                    "recovery": args.recovery_model,
                    "hover": args.hover_model,
                },
            )
        except FileNotFoundError as exc:
            parser.exit(1, f"{exc}\n")
        print(f"Phase config yazildi: {args.output}")
        if missing:
            print("Eksik specialist modeller:", ", ".join(missing))
        for phase, path in config.items():
            print(f"{phase}: {path}")


if __name__ == "__main__":
    main()
