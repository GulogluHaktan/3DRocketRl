from __future__ import annotations

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback

from rl_common import evaluate_model, train_loop, watch_model


ALGO_NAME = "sac"
REWARD_WEIGHTS = {
    # SAC dense reward mode: small continuous shaping signals plus modest
    # terminal safety costs. Milestone/progress-pressure rewards stay out.
    "time_penalty": 0.01,
    "failure_penalty": 150.0,  # Increased from 25.0 to suppress spin exploit
    "success_bonus": 200.0,     # Increased from 50.0 to balance high failure penalty
    "fail_penalty": 150.0,     # Increased from 25.0 to match failure_penalty
    "flip_rel_dist_limit": 5.0,
    "flip_xy_escape_penalty": 100.0,         # Increased from 30.0
    "flip_surface_contact_penalty": 120.0,   # Increased from 40.0

    "dense_position": 1.0,
    "dense_height": 0.8,
    "dense_upright": 0.7,
    "dense_velocity": 0.5,
    "dense_flip_axis": 2.0,
    "dense_flip_upright_recovery": 1.8,
    "dense_hover_stability": 1.0,
    "dense_flip_progress_delta": 8.0,
    "dense_drift": 0.45,
    "dense_angular": 0.25,
    "dense_off_axis": 0.45,
    "dense_yaw_spin": 0.65,
    "dense_control_effort": 0.05,
    "dense_action_smoothness": 0.08,
    "dense_safety": 3.0,                     # Increased from 1.2 to penalize low altitude during flip more heavily
    "dense_overrotate": 2.5,                 # Increased from 1.6 to prevent multiple spins
}
ENV_KWARGS = {
    "reward_mode": "dense",
    "include_task_state_observation": True,  # Enabled to give agent visibility of phases and flip progress
}


def parse_ent_coef(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


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
        "ent_coef": parse_ent_coef(args.sac_ent_coef),
    }
    train_loop(args, ALGO_NAME, SAC, BaseCallback, model_kwargs, REWARD_WEIGHTS, ENV_KWARGS)


def watch(args):
    watch_model(args, ALGO_NAME, SAC, REWARD_WEIGHTS, ENV_KWARGS)


def evaluate(args):
    evaluate_model(args, ALGO_NAME, SAC, REWARD_WEIGHTS, ENV_KWARGS)
