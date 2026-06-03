from __future__ import annotations

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback

from rl_common import train_loop, watch_model


ALGO_NAME = "sac"
REWARD_WEIGHTS = {
    # SAC buyuk reward araliklarinda entropi dengesini cabuk kaybedebiliyor.
    # Bu set terminal odulleri yumusatip her adima kucuk zaman cezasi ekler.
    "time_penalty": 0.03,
    "failure_penalty": 60.0,
    "success_bonus": 120.0,

    "flip_height_reward": 0.08,
    "flip_progress": 55.0,
    "flip_axis_rate": 1.2,
    "flip_progress_late_scale": 2.5,
    "flip_completion_pressure": 2.2,
    "flip_completion_pressure_late_scale": 1.2,
    "flip_no_progress": 1.4,
    "flip_no_progress_late_scale": 3.2,
    "flip_altitude_progress": 18.0,
    "flip_descent_progress": 18.0,
    "flip_low_axis_rate": 1.8,
    "flip_overrotate": 8.0,
    "flip_low_altitude": 0.8,
    "flip_low_altitude_descent": 0.9,
    "flip_airtime_floor": 0.5,
    "flip_descent_speed": 1.8,
    "flip_descent_speed_sq": 0.16,
    "flip_thrust_while_falling": 2.8,
    "flip_no_thrust_while_falling": 2.8,
    "flip_low_thrust_descent": 4.5,
    "flip_low_altitude_low_thrust": 6.0,
    "flip_rel_dist": 3.8,
    "flip_rel_dist_sq": 1.4,
    "flip_rel_progress": 16.0,
    "flip_rel_away": 28.0,
    "flip_rel_boundary_start": 1.0,
    "flip_rel_boundary": 12.0,
    "flip_rel_boundary_sq": 28.0,
    "flip_horizontal_speed": 3.0,
    "flip_horizontal_speed_sq": 0.35,
    "flip_boundary_horizontal_speed": 6.0,
    "flip_boundary_thrust": 28.0,
    "flip_away_thrust": 42.0,
    "flip_rel_dist_limit": 5.0,
    "flip_xy_escape_penalty": 120.0,
    "flip_surface_contact_penalty": 160.0,
    "flip_high_angular_speed": 0.4,
    "flip_axis_alignment": 3.0,
    "flip_off_axis": 0.45,
    "flip_joint_speed": 0.05,
    "flip_side_deviation": 1.0,
    "flip_late_angular_speed": 0.18,
    "flip_body_spin": 0.12,
    "flip_world_z_spin": 0.35,
    "flip_completion_bonus": 30.0,

    "recovery_linear_speed": 0.45,
    "recovery_angular_speed": 0.45,
    "recovery_upright": 8.0,
    "recovery_flip_progress": 8.0,
    "recovery_rel_dist": 8.0,
    "recovery_rel_progress": 12.0,
    "hover_linear_speed": 0.45,
    "hover_angular_speed": 0.45,
    "hover_upright": 8.0,
    "hover_rel_dist": 8.0,
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
