from __future__ import annotations

import argparse
import importlib

from rl_common import plot_run


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
    parser.add_argument("--min-start-z", type=float, default=0.5)
    parser.add_argument("--max-start-z", type=float, default=10.0)
    parser.add_argument("--max-thrust", type=float, default=None)
    parser.add_argument("--max-tvc-deg", type=float, default=20.0)


def add_train_args(parser):
    parser.add_argument("--algo", choices=sorted(ALGORITHMS), default="sac")
    parser.add_argument("--timesteps", type=int, default=250_000)
    parser.add_argument("--chunk-steps", type=int, default=25_000)
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=0.99)

    # SAC/TD3 args.
    parser.add_argument("--buffer-size", type=int, default=300_000)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--learning-starts", type=int, default=5_000)

    # PPO args.
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.0)

    add_env_args(parser)


def add_watch_args(parser):
    parser.add_argument("--algo", choices=sorted(ALGORITHMS), default="sac")
    parser.add_argument("--model", default=None)
    parser.add_argument("--csv", default=None)
    add_env_args(parser)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)

    train_parser = sub.add_parser("train")
    add_train_args(train_parser)

    watch_parser = sub.add_parser("watch")
    add_watch_args(watch_parser)

    plot_parser = sub.add_parser("plot")
    plot_parser.add_argument("run_dir")

    args = parser.parse_args()

    if args.mode == "train":
        load_algo(args.algo).train(args)
    elif args.mode == "watch":
        load_algo(args.algo).watch(args)
    elif args.mode == "plot":
        plot_run(args.run_dir)


if __name__ == "__main__":
    main()
