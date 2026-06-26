import pytest

jax = pytest.importorskip("jax")
jp = pytest.importorskip("jax.numpy")
pytest.importorskip("brax")
pytest.importorskip("mujoco.mjx")

from mjx.env import (
    FULL_FLIP,
    PHASE_CLIMB,
    PHASE_FLIP,
    PHASE_HOVER,
    PHASE_RECOVERY,
    RocketMJXEnv,
)


def test_mjx_reset_and_step_are_jittable():
    env = RocketMJXEnv(curriculum_stage=0)
    state = jax.jit(env.reset)(jax.random.key(0))
    assert state.obs.shape == (31,)
    state = jax.jit(env.step)(state, jp.zeros(3))
    assert state.obs.shape == (31,)
    assert jp.all(jp.isfinite(state.obs))


def test_mjx_reset_ranges_and_recovery_consistency():
    env = RocketMJXEnv(curriculum_stage=0)
    states = jax.jit(jax.vmap(env.reset))(
        jax.random.split(jax.random.key(10), 128)
    )
    phase = states.pipeline_state.phase
    height = states.metrics["height"]
    assert jp.all(jp.where(phase == PHASE_CLIMB, (height >= 0.5) & (height <= 8.0), True))
    assert jp.all(jp.where(phase == PHASE_FLIP, (height >= 8.0) & (height <= 12.0), True))
    assert jp.all(jp.where(phase == PHASE_RECOVERY, (height >= 5.5) & (height <= 10.5), True))
    assert jp.all(jp.where(phase == PHASE_HOVER, (height >= 3.5) & (height <= 6.5), True))
    recovery_progress = states.pipeline_state.flip_progress
    assert jp.all(
        jp.where(
            phase == PHASE_RECOVERY,
            (recovery_progress >= 0.75) & (recovery_progress <= 0.90),
            True,
        )
    )
    assert jp.allclose(
        states.pipeline_state.flip_angle,
        states.pipeline_state.flip_progress * FULL_FLIP,
    )


def test_mjx_full_chain_eval_always_starts_climb():
    env = RocketMJXEnv(curriculum_stage=3)
    states = jax.jit(jax.vmap(env.reset))(
        jax.random.split(jax.random.key(20), 32)
    )
    assert jp.all(states.pipeline_state.phase == PHASE_CLIMB)


def test_mjx_climb_handoff_requires_height_and_bounded_speed():
    env = RocketMJXEnv(curriculum_stage=3)
    state = env.reset(jax.random.key(30))

    def transition_at(height, vertical_speed, angular_speed):
        qpos = state.pipeline_state.data.qpos.at[2].set(height + 0.355)
        qpos = qpos.at[0:2].set(jp.zeros(2))
        qpos = qpos.at[3:7].set(jp.array([1.0, 0.0, 0.0, 0.0]))
        qpos = qpos.at[7:9].set(jp.zeros(2))
        qvel = jp.zeros_like(state.pipeline_state.data.qvel)
        qvel = qvel.at[2].set(vertical_speed)
        qvel = qvel.at[3].set(angular_speed)
        data = state.pipeline_state.data.replace(qpos=qpos, qvel=qvel)
        from mujoco import mjx as mujoco_mjx
        data = mujoco_mjx.forward(env.mjx_model, data)
        rocket = state.pipeline_state.replace(data=data, phase=jp.array(PHASE_CLIMB))
        return env._reward_and_transition(rocket, jp.zeros(3))[1]

    assert int(transition_at(9.99, 0.0, 0.0)) == PHASE_CLIMB
    assert int(transition_at(10.0, 2.01, 0.0)) == PHASE_CLIMB
    assert int(transition_at(10.0, 0.0, 0.51)) == PHASE_CLIMB
    assert int(transition_at(10.0, 1.9, 0.49)) == PHASE_FLIP


def test_mjx_fail_reason_codes_cover_speed_limits():
    env = RocketMJXEnv(curriculum_stage=3)
    state = env.reset(jax.random.key(31))
    qpos = state.pipeline_state.data.qpos.at[2].set(5.0)
    qpos = qpos.at[3:7].set(jp.array([1.0, 0.0, 0.0, 0.0]))
    qvel = jp.zeros_like(state.pipeline_state.data.qvel)
    qvel = qvel.at[0].set(71.0)

    from mujoco import mjx as mujoco_mjx
    data = state.pipeline_state.data.replace(qpos=qpos, qvel=qvel)
    data = mujoco_mjx.forward(env.mjx_model, data)
    rocket = state.pipeline_state.replace(data=data, phase=jp.array(PHASE_CLIMB))

    result = env._reward_and_transition(rocket, jp.zeros(3))
    assert bool(result[7])
    assert float(result[8]) == 10.0
