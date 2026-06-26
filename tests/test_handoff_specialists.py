import numpy as np
import pytest

pytest.importorskip("gymnasium")
pytest.importorskip("mujoco")

import mujoco

from hopper_env import FULL_FLIP_RAD, HopperEnv


def make_handoff_env(phase, **kwargs):
    return HopperEnv(
        start_phase=phase,
        specialist_phase=phase,
        reset_profile="handoff-mix",
        legacy_reset_probability=0.0,
        include_task_state_observation=True,
        **kwargs,
    )


def test_handoff_reset_preserves_observation_shape():
    legacy = HopperEnv(include_task_state_observation=True, reset_profile="legacy")
    handoff = make_handoff_env("hover")
    try:
        legacy_obs, _ = legacy.reset(seed=1)
        handoff_obs, _ = handoff.reset(seed=1)
        assert legacy_obs.shape == handoff_obs.shape
        assert legacy.observation_space.shape == handoff.observation_space.shape
    finally:
        legacy.close()
        handoff.close()


def test_climb_handoff_height_range():
    env = make_handoff_env("climb")
    try:
        heights = []
        for seed in range(50):
            env.reset(seed=seed)
            heights.append(env._compute_metrics()["z"])
        assert min(heights) >= 0.5 - 1e-6
        assert max(heights) <= 8.0 + 1e-6
    finally:
        env.close()


def test_flip_handoff_ranges():
    env = make_handoff_env("flip")
    try:
        for seed in range(50):
            env.reset(seed=seed)
            metrics = env._compute_metrics()
            assert 8.0 <= metrics["z"] <= 12.0
            assert np.max(np.abs(metrics["linear_vel"][:2])) <= 0.5 + 1e-8
            assert abs(metrics["linear_vel"][2]) <= 1.0 + 1e-8
            assert np.max(np.abs(metrics["angular_vel"])) <= 0.3 + 1e-8
            assert metrics["tvc_angle"] <= np.sqrt(2.0) * env.max_tvc_angle + 1e-8
            assert metrics["joint_speed"] <= np.sqrt(2.0) * 0.8 + 1e-8
    finally:
        env.close()


def test_recovery_progress_orientation_and_spin_are_consistent():
    env = make_handoff_env("recovery")
    try:
        forward_spins = 0
        for seed in range(100):
            env.reset(seed=seed)
            metrics = env._compute_metrics()
            assert 0.75 <= env.flip_progress <= 0.90
            assert env.flip_angle == pytest.approx(env.flip_progress * FULL_FLIP_RAD)
            assert metrics["expected_axis_alignment"] >= 0.95
            if metrics["positive_flip_axis_rate"] >= 0.5:
                forward_spins += 1
        assert forward_spins >= 80
    finally:
        env.close()


def test_hover_handoff_ranges():
    env = make_handoff_env("hover")
    try:
        for seed in range(50):
            env.reset(seed=seed)
            metrics = env._compute_metrics()
            assert 3.5 <= metrics["z"] <= 6.5
            assert np.max(np.abs(metrics["linear_vel"][:2])) <= 0.75 + 1e-8
            assert abs(metrics["linear_vel"][2]) <= 1.0 + 1e-8
            assert np.max(np.abs(metrics["angular_vel"])) <= 0.5 + 1e-8
    finally:
        env.close()


def test_climb_requires_ten_meters_and_bounded_speeds():
    env = HopperEnv(
        flip_start_min_z=10.0,
        flip_start_max_z=None,
        flip_start_max_linear_speed=2.0,
        flip_start_max_horizontal_speed=2.0,
        flip_start_min_vertical_velocity=-2.0,
        flip_start_max_vertical_velocity=2.0,
        flip_start_max_angular_speed=0.5,
        flip_start_min_upright=0.97,
        flip_start_max_tvc_angle=0.10,
        flip_start_max_joint_speed=0.8,
    )
    try:
        env.reset(seed=0)

        def metrics_at(height, linear_speed=0.0, angular_speed=0.0):
            env.data.qpos[0:3] = [0.0, 0.0, height + 0.355]
            env.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
            env.data.qpos[env.yaw_qpos_id] = 0.0
            env.data.qpos[env.pitch_qpos_id] = 0.0
            env.data.qvel[:] = 0.0
            env.data.qvel[2] = linear_speed
            env.data.qvel[3] = angular_speed
            mujoco.mj_forward(env.model, env.data)
            return env._compute_metrics()

        assert not env._is_ready_for_flip(metrics_at(9.99))
        assert not env._is_ready_for_flip(metrics_at(10.0, linear_speed=2.01))
        assert not env._is_ready_for_flip(metrics_at(10.0, angular_speed=0.51))
        assert env._is_ready_for_flip(metrics_at(10.0, linear_speed=1.9, angular_speed=0.49))
    finally:
        env.close()
