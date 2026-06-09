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
    "dense_upright": 0.8,
    "dense_upright_hold": 0.9,
    "dense_velocity": 0.5,
    "dense_flip_axis": 2.0,
    "dense_flip_upright_recovery": 1.8,
    "dense_climb_target": 1.2,
    "dense_climb_upright": 0.8,
    "dense_climb_velocity": 0.9,
    "dense_hover_height": 1.2,
    "dense_hover_upright": 1.0,
    "dense_hover_velocity": 1.1,
    "dense_hover_stability": 1.2,
    "dense_flip_progress_delta": 8.0,
    "dense_drift": 0.45,
    "dense_angular": 0.25,
    "dense_off_axis": 0.45,
    "dense_yaw_spin": 0.65,
    "dense_control_effort": 0.05,
    "dense_action_smoothness": 0.14,
    "dense_tvc_angle": 0.70,
    "dense_tvc_velocity": 1.25,
    "dense_recovery_descent_speed": 3.0,
    "dense_recovery_speed": 1.1,
    "dense_safety": 3.0,                     # Increased from 1.2 to penalize low altitude during flip more heavily
    "dense_overrotate": 2.5,                 # Increased from 1.6 to prevent multiple spins
    "dense_falling_thrust": 1.6,             # Encourage thrusting to break the fall in climb/recovery
    "dense_upward_climb": 2.0,               # Encourage sustained ascent toward the next phase target
    "climb_ready_dwell": 25.0,
}
ENV_KWARGS = {
    "reward_mode": "dense",
    "include_task_state_observation": True,  # Enabled to give agent visibility of phases and flip progress
    "climb_fail_x_limit": 6.0,               # Give rocket more room to drift/correct horizontally in climb phase
    "climb_fail_y_limit": 6.0,               # Give rocket more room to drift/correct horizontally in climb phase
    "hover_max_rel_dist": 1.2,
    "hover_entry_max_height_error": 0.8,
    "hover_entry_max_downward_speed": 2.0,
    "recovery_start_roughness": 0.75,        # Rough post-flip handoffs, staged below full chaos for learning
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
