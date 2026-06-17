from __future__ import annotations

import argparse
import importlib

from rl_common import make_phase_models_config, plot_run


ALGORITHMS = {
    "sac": "sac",
    "ppo": "ppo",
    "td3": "td3",
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
    parser.add_argument("--telegram-config", default="telegram_secrets.json")
    parser.add_argument("--telegram-every", type=int, default=10_000)
    parser.add_argument("--no-telegram", action="store_true")
    parser.add_argument("--telegram-video-every", type=int, default=100_000)
    parser.add_argument("--no-telegram-video", action="store_true")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=0.97)

    # SAC/TD3 args.
    parser.add_argument("--buffer-size", type=int, default=300_000)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--learning-starts", type=int, default=5_000)
    parser.add_argument("--sac-ent-coef", default="auto_0.02")

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


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)

    train_parser = sub.add_parser("train")
    add_train_args(train_parser)

    train_specialists_parser = sub.add_parser("train-specialists")
    add_train_args(train_specialists_parser)
    train_specialists_parser.add_argument("--timesteps-per-phase", type=int, default=500_000)
    train_specialists_parser.add_argument(
        "--phases",
        nargs="+",
        choices=("climb", "flip", "recovery", "hover"),
        default=("climb", "flip", "recovery", "hover"),
    )

    watch_parser = sub.add_parser("watch")
    add_watch_args(watch_parser)

    eval_parser = sub.add_parser("eval")
    add_eval_args(eval_parser)

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
        default="runs/sac_hopper_20260603_152230/sac_hopper_latest.zip",
    )
    phase_config_parser.add_argument("--require-all", action="store_true")

    args = parser.parse_args()

    if getattr(args, "handoff_model", None) and getattr(args, "handoff_phase_models_config", None):
        parser.error("--handoff-model ve --handoff-phase-models-config ayni anda kullanilamaz.")

    if (
        hasattr(args, "specialist_phase")
        and args.specialist_phase is not None
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
        algo_mod = load_algo(args.algo)
        for phase in args.phases:
            print(f"\n=========================================")
            print(f"STARTING PHASE SPECIALIST TRAINING: {phase.upper()}")
            print(f"=========================================\n")
            phase_args = copy.deepcopy(args)
            phase_args.specialist_phase = phase
            phase_args.start_phase = phase
            phase_args.start_z = start_z_map[phase]
            if phase == "climb":
                phase_args.fixed_start_z = False
                phase_args.random_start_z = True
            else:
                phase_args.fixed_start_z = True
                phase_args.random_start_z = False
            if phase_args.max_thrust is None:
                phase_args.max_thrust = 45.0
            phase_args.timesteps = args.timesteps_per_phase
            phase_args.run_dir = None
            algo_mod.train(phase_args)
    elif args.mode == "watch":
        load_algo(args.algo).watch(args)
    elif args.mode == "eval":
        load_algo(args.algo).evaluate(args)
    elif args.mode == "plot":
        plot_run(args.run_dir)
    elif args.mode == "make-phase-config":
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
