from __future__ import annotations

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from rl_common import train_loop, watch_model


ALGO_NAME = "ppo"
REWARD_WEIGHTS = {
    # PPO reward denemelerini burada override edebilirsin.
}


def train(args):
    model_kwargs = {
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "gamma": args.gamma,
        "n_steps": args.n_steps,
        "gae_lambda": args.gae_lambda,
        "clip_range": args.clip_range,
        "ent_coef": args.ent_coef,
    }
    train_loop(args, ALGO_NAME, PPO, BaseCallback, model_kwargs, REWARD_WEIGHTS)


def watch(args):
    watch_model(args, ALGO_NAME, PPO, REWARD_WEIGHTS)
