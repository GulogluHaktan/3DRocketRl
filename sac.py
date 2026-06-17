from __future__ import annotations

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback

from rl_common import evaluate_model, train_loop, watch_model


ALGO_NAME = "sac"
REWARD_WEIGHTS = {
    # SAC dense reward mode: small continuous shaping signals plus modest
    # terminal safety costs. Milestone/progress-pressure rewards stay out.
    "time_penalty": 0.01,
    "failure_penalty": 220.0,  # Keep failed long dense episodes from looking profitable
    "success_bonus": 320.0,     # Make real handoff beat timeout survival clearly
    "fail_penalty": 220.0,     # Match failure_penalty
    "flip_rel_dist_limit": 5.0,
    "flip_xy_escape_penalty": 100.0,         # Increased from 30.0
    "flip_surface_contact_penalty": 120.0,   # Increased from 40.0

    "dense_position": 1.0,
    "dense_height": 0.8,
    "dense_upright": 0.8,
    "dense_upright_hold": 0.9,
    "dense_velocity": 0.5,
    "dense_flip_axis": 3.0,
    "dense_flip_upright_recovery": 2.4,
    "dense_climb_target": 1.2,
    "dense_climb_upright": 0.8,
    "dense_climb_velocity": 0.9,
    "dense_climb_handoff": 4.0,
    "dense_climb_tvc_cleanup": 1.8,
    "dense_climb_excess_upward_speed": 2.8,
    "dense_climb_descent": 4.0,
    "dense_hover_height": 1.2,
    "dense_hover_upright": 1.0,
    "dense_hover_velocity": 1.1,
    "dense_hover_stability": 1.2,
    "dense_recovery_handoff": 2.0,
    "dense_flip_progress_delta": 14.0,
    "dense_drift": 0.45,
    "dense_angular": 0.25,
    "dense_off_axis": 0.45,
    "dense_yaw_spin": 0.65,
    "dense_control_effort": 0.05,
    "dense_action_smoothness": 0.14,
    "dense_tvc_angle": 0.85,
    "dense_tvc_velocity": 1.55,
    "dense_recovery_descent_speed": 3.0,
    "dense_recovery_speed": 1.1,
    "dense_recovery_handoff_speed": 2.0,
    "dense_safety": 3.0,                     # Increased from 1.2 to penalize low altitude during flip more heavily
    "dense_overrotate": 2.5,                 # Increased from 1.6 to prevent multiple spins
    "dense_falling_thrust": 2.0,             # Encourage thrusting to break the fall in climb/recovery
    "dense_upward_climb": 2.4,               # Encourage sustained ascent toward the next phase target
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
    "hover_entry_min_upright": 0.95,
    "hover_entry_max_horizontal_speed": 1.5,
    "hover_entry_max_angular_speed": 1.5,
    "hover_entry_max_vertical_speed": 1.5,
    "flip_start_min_upright": 0.97,
    "flip_start_min_z": 8.0,
    "flip_start_max_rel_dist": 1.0,
    "flip_start_max_horizontal_speed": 0.30,
    "flip_start_min_vertical_velocity": -0.50,
    "flip_start_max_vertical_velocity": 0.50,
    "flip_start_max_angular_speed": 0.30,
    "flip_start_max_tvc_angle": 0.05,
    "flip_start_max_joint_speed": 0.40,
    "flip_complete_progress": 0.95,
    "flip_complete_min_upright": 0.85,
    "flip_complete_max_rel_dist": 2.0,
    "flip_complete_max_horizontal_speed": 4.0,
    "flip_complete_min_z": 5.0,
    "flip_complete_min_vertical_velocity": -8.0,
    "flip_complete_max_world_z_spin": 2.0,
    "flip_progress_schedule_grace_steps": 35,
    "flip_progress_schedule_min_final": 0.88,
    "flip_start_roughness": 0.5,
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
