import pytest

jax = pytest.importorskip("jax")
jp = pytest.importorskip("jax.numpy")
pytest.importorskip("brax")
pytest.importorskip("mujoco.mjx")

from gimbal.env_mjx import GimbalMJXEnv


def test_gimbal_mjx_reset_and_step_are_jittable():
    env = GimbalMJXEnv(stage=1)
    state = jax.jit(env.reset)(jax.random.key(0))
    assert state.obs.shape == (12,)
    state = jax.jit(env.step)(state, jp.zeros(2))
    assert state.obs.shape == (12,)
    assert jp.all(jp.isfinite(state.obs))


def test_gimbal_mjx_stage_three_uses_domain_randomization():
    env = GimbalMJXEnv(stage=3)
    states = jax.jit(jax.vmap(env.reset))(jax.random.split(jax.random.key(1), 16))
    assert jp.any(states.pipeline_state.mass_scale != 1.0)
    assert jp.any(states.pipeline_state.servo_scale != 1.0)
