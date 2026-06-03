from __future__ import annotations

import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import NormalActionNoise

from rl_common import train_loop, watch_model


ALGO_NAME = "td3"
REWARD_WEIGHTS = {
    "climb_z_error": 3.0,
    "climb_low_altitude": 3.0,
    "climb_low_altitude_sq": 1.0,
    "climb_x": 2.0,
    "climb_y": 3.0,
    "climb_rel_dist_sq": 2.5,
    "climb_angular_speed": 2.2,
    "climb_joint_speed": 0.05,
    "climb_upright": 16.0,
    "climb_speed": 2.4,
    "climb_altitude_progress": 140.0,
    "climb_upward_speed": 3.5,
    "climb_ready_dwell": 25.0,
    "flip_progress": 105.0,
    "flip_stall": 3.0,
    "flip_axis_rate": 11.0,
    "flip_slow_axis_rate": 14.0,
    "flip_excess_axis_rate": 2.0,
    "flip_upward_speed": 2.0,
    "flip_downward_speed": 4.0,
    "flip_falling_thrust": 14.0,
    "flip_recovery_falling_thrust": 70.0,
    "flip_recovery_upward_speed": 10.0,
    "flip_recovery_downward_speed": 12.0,
    "flip_low_altitude_upward_speed": 8.0,
    "flip_low_altitude_margin": 12.0,
    "flip_high_angular_speed": 0.8,
    "flip_axis_alignment": 16.0,
    "flip_off_axis": 1.0,
    "flip_rel_dist": 5.0,
    "flip_rel_dist_sq": 2.5,
    "flip_late_angular_speed": 0.25,
    "flip_body_spin": 0.8,
    "recovery_linear_speed": 2.4,
    "recovery_angular_speed": 2.8,
    "recovery_upright": 45.0,
    "recovery_flip_progress": 35.0,
    "recovery_overrotate": 90.0,
    "recovery_rel_dist": 55.0,
    "recovery_low_altitude": 26.0,
    "recovery_upward_speed": 30.0,
    "recovery_downward_speed": 22.0,
    "recovery_falling_thrust": 85.0,
    "recovery_target_closing_speed": 18.0,
    "recovery_thrust_alignment": 38.0,
    "recovery_height_error": 14.0,
    "recovery_altitude_progress": 260.0,
    "recovery_below_hover_band": 55.0,
    "recovery_ground_stall": 180.0,
    "recovery_upright_climb": 35.0,
    "hover_low_altitude": 26.0,
    "hover_upward_speed": 28.0,
    "hover_falling_thrust": 80.0,
    "hover_upright": 45.0,
    "hover_height_error": 30.0,
    "hover_vertical_speed": 10.0,
    "phase_climb_to_flip_bonus": 350.0,
    "phase_flip_to_recovery_bonus": 900.0,
    "phase_recovery_to_hover_bonus": 1400.0,
    "success_bonus": 4500.0,
}
ENV_KWARGS = {
    "flip_target_z": 10.0,
    "hover_target_z": 5.0,
    "include_task_state_observation": True,
    "use_corrected_flip_low_altitude_penalty": True,
    "max_climb_ready_time": 1.2,
    "climb_ready_min_z": 8.7,
    "flip_start_min_z": 8.4,
    "flip_start_max_z": 10.6,
    "flip_start_min_upright": 0.65,
    "climb_fail_x_limit": 3.5,
    "climb_fail_y_limit": 2.8,
    "use_world_angular_damping": True,
    "use_world_linear_damping": True,
    "flip_low_altitude_threshold": 3.5,
    "flip_low_altitude_margin": 5.0,
    "recovery_max_rel_dist": 6.0,
    "hover_max_rel_dist": 2.0,
    "hover_entry_max_height_error": 1.5,
    "hover_entry_max_downward_speed": 0.8,
    "hover_stable_max_height_error": 1.0,
    "hover_stable_min_upright": 0.9,
    "hover_stable_max_linear_speed": 2.0,
    "hover_stable_max_angular_speed": 2.0,
    "flip_complete_progress": 0.97,
    "flip_complete_min_upright": 0.85,
    "flip_upright_recovery_progress": 0.82,
    "flip_upright_recovery_min_upright": 0.95,
    "flip_upright_recovery_max_z": 3.5,
    "flip_low_altitude_stall_progress": 0.75,
    "flip_low_altitude_stall_z": 2.0,
    "flip_low_altitude_stall_time": 0.8,
    "flip_low_altitude_stall_min_upward_speed": 0.25,
    "recovery_low_altitude_fail_z": 2.0,
    "recovery_low_altitude_fail_time": 1.0,
    "recovery_low_altitude_min_upward_speed": 0.25,
}


def train(args):
    action_noise = NormalActionNoise(
        mean=np.zeros(3),
        sigma=np.array([0.08, 0.25, 0.25]),
    )
    model_kwargs = {
        "learning_rate": args.learning_rate,
        "buffer_size": args.buffer_size,
        "batch_size": args.batch_size,
        "gamma": args.gamma,
        "tau": args.tau,
        "train_freq": 1,
        "gradient_steps": 1,
        "learning_starts": args.learning_starts,
        "action_noise": action_noise,
    }
    train_loop(args, ALGO_NAME, TD3, BaseCallback, model_kwargs, REWARD_WEIGHTS, ENV_KWARGS)


def watch(args):
    watch_model(args, ALGO_NAME, TD3, REWARD_WEIGHTS, ENV_KWARGS)
