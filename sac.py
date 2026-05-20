from __future__ import annotations

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback

from rl_common import train_loop, watch_model


ALGO_NAME = "sac"
REWARD_WEIGHTS = {
    # SAC reward denemelerini burada override edebilirsin.
}


def train(args):
    model_kwargs = {
        "learning_rate": args.learning_rate,
        "buffer_size": args.buffer_size,
        "batch_size": args.batch_size,
        "gamma": args.gamma,
        "tau": args.tau,
        "train_freq": 1,
        "gradient_steps": 1,
        "learning_starts": args.learning_starts,
    }
    train_loop(args, ALGO_NAME, SAC, BaseCallback, model_kwargs, REWARD_WEIGHTS)


def watch(args):
    watch_model(args, ALGO_NAME, SAC, REWARD_WEIGHTS)
