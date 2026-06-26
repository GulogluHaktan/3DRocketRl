import numpy as np
import pytest


mujoco = pytest.importorskip("mujoco")
pytest.importorskip("gymnasium")

from gimbal.env import GimbalEnv


def test_spaces_and_control_timing():
    env = GimbalEnv(stage=1, control_rate=50)
    obs, _ = env.reset(seed=1)
    assert env.frame_skip == 10
    assert env.action_space.shape == (2,)
    assert env.observation_space.shape == (12,)
    assert env.observation_space.contains(obs)
    env.close()


def test_pivot_has_no_translation_or_yaw_dof():
    env = GimbalEnv(stage=1)
    joint_names = {
        mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_JOINT, index)
        for index in range(env.model.njnt)
    }
    assert joint_names == {"gimbal_roll", "gimbal_pitch", "tvc_x", "tvc_y"}
    assert env.model.nq == 4
    assert env.model.nv == 4
    env.close()


def test_action_and_servo_limits():
    env = GimbalEnv(stage=1)
    env.reset(seed=2)
    for _ in range(100):
        obs, _, terminated, truncated, _ = env.step(np.array([4.0, -4.0]))
        assert env.observation_space.contains(obs)
        assert abs(env.data.qpos[env.tvc_x_qpos]) <= env.max_tvc_angle + 1e-9
        assert abs(env.data.qpos[env.tvc_y_qpos]) <= env.max_tvc_angle + 1e-9
        if terminated or truncated:
            break
    assert np.max(np.abs(env.tvc_target)) <= env.max_tvc_angle + 1e-9
    env.close()


def test_sensor_delay_is_bounded_to_one_control_step():
    env = GimbalEnv(stage=3, domain_randomization=True, sensor_delay_ms=20.0)
    env.reset(seed=3)
    assert env.sensor_delay_steps == 1
    env.close()


def test_positive_roll_uses_positive_tvc_x_to_reduce_tilt_cost():
    def next_tilt(action_x):
        env = GimbalEnv(stage=1)
        env.reset(seed=4)
        env.data.qpos[env.roll_qpos] = np.deg2rad(10.0)
        env.data.qpos[env.pitch_qpos] = 0.0
        env.data.qvel[:] = 0.0
        mujoco.mj_forward(env.model, env.data)
        for _ in range(5):
            env.step(np.array([action_x, 0.0], dtype=np.float32))
        tilt = env.tilt_rad
        env.close()
        return tilt

    assert next_tilt(1.0) < next_tilt(-1.0)


def test_reward_penalizes_servo_saturation_and_action_reversal():
    env = GimbalEnv(stage=1)
    env.reset(seed=5)
    env.tvc_target[:] = env.max_tvc_angle
    env.previous_action[:] = -1.0
    env.step(np.array([1.0, 1.0], dtype=np.float32))
    assert env.last_reward_terms["saturation"] < 0.0
    assert env.last_reward_terms["action_delta"] <= -2.0
    env.close()


def test_servo_rate_and_rate_command_slew_are_bounded():
    env = GimbalEnv(stage=1)
    env.reset(seed=6)
    previous_command = env.tvc_rate_command.copy()
    for action in (
        np.ones(2, dtype=np.float32),
        -np.ones(2, dtype=np.float32),
        np.ones(2, dtype=np.float32),
    ):
        env.step(action)
        assert np.max(np.abs(env.tvc_rate)) <= env.command_speed + 1e-9
        assert (
            np.max(np.abs(env.tvc_rate_command - previous_command))
            <= env.max_rate_command_delta + 1e-9
        )
        previous_command = env.tvc_rate_command.copy()
    env.close()


def test_success_region_rates_are_visible_in_observation():
    env = GimbalEnv(stage=1)
    env.reset(seed=7)
    env.data.qvel[:] = 0.0
    env.data.qvel[env.roll_qvel] = np.deg2rad(5.0)
    env.tvc_rate[0] = np.deg2rad(20.0)
    mujoco.mj_forward(env.model, env.data)
    obs = env._raw_observation()
    assert obs[3] == pytest.approx(5.0 / 45.0)
    assert obs[8] == pytest.approx(20.0 / 120.0)
    env.close()


def test_training_easy_resets_stay_inside_stage_limits():
    env = GimbalEnv(stage=1, training_mode=True)
    tilts = []
    rates = []
    for seed in range(100):
        _, info = env.reset(seed=seed)
        tilts.append(info["tilt_deg"])
        rates.append(info["angular_speed_dps"])
    assert max(tilts) <= 15.0 + 1e-6
    assert max(rates) <= np.sqrt(2.0) * 10.0 + 1e-6
    assert sum(tilt <= 5.0 for tilt in tilts) > 40
    env.close()


def test_training_hold_curriculum_does_not_change_default_success_rule():
    default_env = GimbalEnv(stage=1)
    curriculum_env = GimbalEnv(stage=1, training_mode=True, success_hold_sec=0.25)
    assert default_env.required_success_hold_sec == 1.0
    assert curriculum_env.required_success_hold_sec == 0.25
    default_env.close()
    curriculum_env.close()
