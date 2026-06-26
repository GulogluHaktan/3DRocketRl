from __future__ import annotations

from pathlib import Path
from typing import Any

from brax.envs import base
from flax import struct
import jax
import jax.numpy as jp
import mujoco
from mujoco import mjx


PHASE_CLIMB = 0
PHASE_FLIP = 1
PHASE_RECOVERY = 2
PHASE_HOVER = 3
PHASE_DONE = 4
FULL_FLIP = 2.0 * jp.pi


@struct.dataclass
class RocketState:
    data: Any
    phase: jax.Array
    flip_angle: jax.Array
    flip_progress: jax.Array
    hover_timer: jax.Array
    climb_timer: jax.Array
    tvc_cmd: jax.Array
    last_action: jax.Array
    last_height: jax.Array
    last_vertical_velocity: jax.Array
    episode_step: jax.Array
    success: jax.Array
    fail: jax.Array
    fail_reason_code: jax.Array
    transition_mask: jax.Array
    transition_height: jax.Array
    transition_linear_speed: jax.Array
    transition_angular_speed: jax.Array
    settling_time: jax.Array
    max_tvc_seen: jax.Array
    max_tvc_speed_seen: jax.Array
    thrust_scale: jax.Array
    servo_scale: jax.Array
    damping_scale: jax.Array


def quat_mul(left: jax.Array, right: jax.Array) -> jax.Array:
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    q = jp.array([
        lw * rw - lx * rx - ly * ry - lz * rz,
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
    ])
    return q / jp.maximum(jp.linalg.norm(q), 1e-8)


def axis_angle(axis: jax.Array, angle: jax.Array) -> jax.Array:
    axis = axis / jp.maximum(jp.linalg.norm(axis), 1e-8)
    half = 0.5 * angle
    return jp.concatenate((jp.cos(half)[None], axis * jp.sin(half)))


def quat_to_mat(q: jax.Array) -> jax.Array:
    q = q / jp.maximum(jp.linalg.norm(q), 1e-8)
    w, x, y, z = q
    return jp.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


class RocketMJXEnv(base.Env):
    """Pure-JAX four-phase rocket environment backed by MuJoCo MJX."""

    def __init__(
        self,
        curriculum_stage: int = 0,
        episode_length: int = 1200,
        max_thrust: float = 45.0,
        max_tvc_deg: float = 20.0,
        frame_skip: int = 8,
        specialist_phase: str | None = None,
    ):
        xml = Path(__file__).resolve().parents[1] / "assets" / "hopper_default.xml"
        self._host_model = mujoco.MjModel.from_xml_path(str(xml))
        # The classic model deliberately uses a high-accuracy implicit solver.
        # MJX throughput is dramatically better with a GPU-oriented solver
        # profile; geometry, mass, actuators and timestep stay unchanged.
        self._host_model.opt.integrator = mujoco.mjtIntegrator.mjINT_EULER
        self._host_model.opt.solver = mujoco.mjtSolver.mjSOL_CG
        self._host_model.opt.iterations = 8
        self._host_model.opt.ls_iterations = 4
        self._host_model.opt.jacobian = mujoco.mjtJacobian.mjJAC_SPARSE
        # Landing/contact is terminal in the flight-chain task.  Solving every
        # possible foot/ground constraint in every batched MJX step is wasted
        # work, so MJX uses geometric foot-height termination instead.
        self._host_model.geom_contype[:] = 0
        self._host_model.geom_conaffinity[:] = 0
        self._model = mjx.put_model(self._host_model)
        self._stage = int(curriculum_stage)
        if specialist_phase is None:
            self._specialist_phase = -1
        else:
            phase_map = {"climb": 0, "flip": 1, "recovery": 2, "hover": 3}
            self._specialist_phase = phase_map[specialist_phase]
        self._episode_length = int(episode_length)
        self._max_thrust = float(max_thrust)
        self._max_tvc = jp.deg2rad(max_tvc_deg)
        self._servo_speed = jp.deg2rad(60.0 / 0.13)
        self._frame_skip = int(frame_skip)
        self._dt = float(self._host_model.opt.timestep * frame_skip)
        self._physics_dt = float(self._host_model.opt.timestep)
        self._body_id = mujoco.mj_name2id(
            self._host_model, mujoco.mjtObj.mjOBJ_BODY, "hopper"
        )
        self._site_id = mujoco.mj_name2id(
            self._host_model, mujoco.mjtObj.mjOBJ_SITE, "thrust_site"
        )
        self._yaw_qpos = int(self._host_model.jnt_qposadr[
            mujoco.mj_name2id(
                self._host_model, mujoco.mjtObj.mjOBJ_JOINT, "tvc_yaw_joint"
            )
        ])
        self._pitch_qpos = int(self._host_model.jnt_qposadr[
            mujoco.mj_name2id(
                self._host_model, mujoco.mjtObj.mjOBJ_JOINT, "tvc_pitch_joint"
            )
        ])
        self._yaw_qvel = int(self._host_model.jnt_dofadr[
            mujoco.mj_name2id(
                self._host_model, mujoco.mjtObj.mjOBJ_JOINT, "tvc_yaw_joint"
            )
        ])
        self._pitch_qvel = int(self._host_model.jnt_dofadr[
            mujoco.mj_name2id(
                self._host_model, mujoco.mjtObj.mjOBJ_JOINT, "tvc_pitch_joint"
            )
        ])
        self._foot_geom_ids = jp.array([
            mujoco.mj_name2id(
                self._host_model, mujoco.mjtObj.mjOBJ_GEOM, f"foot{i}_geom"
            )
            for i in range(1, 5)
        ])

    @property
    def backend(self) -> str:
        return "mjx"

    @property
    def observation_size(self) -> int:
        return 31

    @property
    def action_size(self) -> int:
        return 3

    @property
    def mjx_model(self):
        return self._model

    @property
    def host_model(self):
        return self._host_model

    def _sample_phase(self, key: jax.Array) -> jax.Array:
        if self._stage >= 3:
            return jp.array(PHASE_CLIMB, dtype=jp.int32)
        uniform_phase = jax.random.randint(key, (), 0, 4)
        coin = jax.random.uniform(key)
        if self._stage == 0:
            return uniform_phase
        if self._stage == 1:
            return jp.where(coin < 0.50, PHASE_CLIMB, uniform_phase)
        return jp.where(coin < 0.80, PHASE_CLIMB, uniform_phase)

    def reset(self, rng: jax.Array) -> base.State:
        keys = jax.random.split(rng, 16)
        phase = self._sample_phase(keys[0])
        hard_enabled = jp.asarray(self._stage >= 2)
        qpos = jp.zeros((self._host_model.nq,), dtype=jp.float32)
        qvel = jp.zeros((self._host_model.nv,), dtype=jp.float32)

        climb_z = jax.random.uniform(keys[1], (), minval=0.5, maxval=8.0)
        flip_z = jax.random.uniform(keys[2], (), minval=8.0, maxval=12.0)
        recovery_z = jax.random.uniform(keys[3], (), minval=5.5, maxval=10.5)
        hover_z = jax.random.uniform(keys[4], (), minval=3.5, maxval=6.5)
        bottom_z = jp.array([climb_z, flip_z, recovery_z, hover_z])[phase]
        xy_scale = jp.array([0.0, 0.25, 1.0, 0.35])[phase]
        xy = jax.random.uniform(keys[5], (2,), minval=-xy_scale, maxval=xy_scale)

        flip_progress = jax.random.uniform(keys[6], (), minval=0.75, maxval=0.90)
        base_recovery_q = axis_angle(jp.array([0.0, 1.0, 0.0]), -flip_progress * FULL_FLIP)
        small_angle = jp.deg2rad(jp.where(phase == PHASE_RECOVERY, 8.0, 5.0))
        roll = jax.random.uniform(keys[7], (), minval=-small_angle, maxval=small_angle)
        pitch = jax.random.uniform(keys[8], (), minval=-small_angle, maxval=small_angle)
        yaw = jax.random.uniform(keys[9], (), minval=-small_angle, maxval=small_angle)
        perturb_q = quat_mul(
            axis_angle(jp.array([0.0, 0.0, 1.0]), yaw),
            quat_mul(
                axis_angle(jp.array([0.0, 1.0, 0.0]), pitch),
                axis_angle(jp.array([1.0, 0.0, 0.0]), roll),
            ),
        )
        quat = jp.where(
            phase == PHASE_CLIMB,
            jp.array([1.0, 0.0, 0.0, 0.0]),
            jp.where(
                phase == PHASE_RECOVERY,
                quat_mul(base_recovery_q, perturb_q),
                perturb_q,
            ),
        )
        qpos = qpos.at[0:2].set(xy)
        qpos = qpos.at[2].set(bottom_z + 0.355)
        qpos = qpos.at[3:7].set(quat)

        flip_linear = jax.random.uniform(keys[10], (3,), minval=jp.array([-0.5, -0.5, -1.0]), maxval=jp.array([0.5, 0.5, 1.0]))
        recovery_linear = jax.random.uniform(keys[10], (3,), minval=jp.array([-1.5, -1.5, -5.0]), maxval=jp.array([1.5, 1.5, 1.0]))
        hover_linear = jax.random.uniform(keys[10], (3,), minval=jp.array([-0.75, -0.75, -1.0]), maxval=jp.array([0.75, 0.75, 1.0]))
        linear = jp.where(
            phase == PHASE_FLIP,
            flip_linear,
            jp.where(
                phase == PHASE_RECOVERY,
                recovery_linear,
                jp.where(phase == PHASE_HOVER, hover_linear, jp.array([0.0, 0.0, -0.1])),
            ),
        )
        small_angular = jax.random.uniform(keys[11], (3,), minval=-0.3, maxval=0.3)
        recovery_rate = jp.where(
            jax.random.uniform(keys[12]) < 0.9,
            -jax.random.uniform(keys[13], (), minval=0.5, maxval=4.0),
            jax.random.uniform(keys[13], (), minval=-0.5, maxval=0.5),
        )
        recovery_angular = small_angular.at[1].set(recovery_rate)
        angular = jp.where(
            phase == PHASE_RECOVERY,
            recovery_angular,
            jp.where(phase == PHASE_CLIMB, jp.zeros(3), small_angular),
        )
        qvel = qvel.at[0:3].set(linear)
        qvel = qvel.at[3:6].set(angular)

        handoff_tvc = jp.deg2rad(5.0)
        tvc_limit = jp.where(
            (phase == PHASE_RECOVERY)
            | (hard_enabled & (jax.random.uniform(keys[14]) < 0.2)),
            self._max_tvc,
            jp.where(phase == PHASE_HOVER, jp.deg2rad(8.5), handoff_tvc),
        )
        tvc = jax.random.uniform(keys[14], (2,), minval=-tvc_limit, maxval=tvc_limit)
        tvc = jp.where(phase == PHASE_CLIMB, jp.zeros(2), tvc)
        qpos = qpos.at[self._yaw_qpos].set(tvc[0])
        qpos = qpos.at[self._pitch_qpos].set(tvc[1])

        data = mjx.make_data(self._model).replace(qpos=qpos, qvel=qvel)
        data = mjx.forward(self._model, data)
        thrust_scale = jax.random.uniform(keys[15], (), minval=0.9, maxval=1.1)
        servo_scale = jax.random.uniform(keys[13], (), minval=0.85, maxval=1.15)
        damping_scale = jax.random.uniform(keys[12], (), minval=0.85, maxval=1.15)
        thrust_scale = jp.where(hard_enabled, thrust_scale, 1.0)
        servo_scale = jp.where(hard_enabled, servo_scale, 1.0)
        damping_scale = jp.where(hard_enabled, damping_scale, 1.0)

        rocket = RocketState(
            data=data,
            phase=phase,
            flip_angle=jp.where(phase == PHASE_RECOVERY, flip_progress * FULL_FLIP, jp.where(phase == PHASE_HOVER, FULL_FLIP, 0.0)),
            flip_progress=jp.where(phase == PHASE_RECOVERY, flip_progress, jp.where(phase == PHASE_HOVER, 1.0, 0.0)),
            hover_timer=jp.array(0.0),
            climb_timer=jp.array(0.0),
            tvc_cmd=tvc,
            last_action=jp.zeros(3),
            last_height=bottom_z,
            last_vertical_velocity=linear[2],
            episode_step=jp.array(0, dtype=jp.int32),
            success=jp.array(False),
            fail=jp.array(False),
            fail_reason_code=jp.array(0.0, dtype=jp.float32),
            transition_mask=jp.zeros(4),
            transition_height=jp.full(3, jp.nan),
            transition_linear_speed=jp.full(3, jp.nan),
            transition_angular_speed=jp.full(3, jp.nan),
            settling_time=jp.array(-1.0),
            max_tvc_seen=jp.linalg.norm(tvc, ord=jp.inf),
            max_tvc_speed_seen=jp.array(0.0),
            thrust_scale=thrust_scale,
            servo_scale=servo_scale,
            damping_scale=damping_scale,
        )
        obs, metrics = self._observe(rocket, jp.array(0.0))
        return base.State(
            pipeline_state=rocket,
            obs=obs,
            reward=jp.array(0.0),
            done=jp.array(0.0),
            metrics=metrics,
            info={},
        )

    def _physics_substep(self, carry, inputs):
        data, tvc_cmd = carry
        main, tvc_target, thrust_scale, servo_scale, damping_scale = inputs
        max_delta = self._servo_speed * servo_scale * self._physics_dt
        tvc_cmd = tvc_cmd + jp.clip(tvc_target - tvc_cmd, -max_delta, max_delta)
        ctrl = jp.rad2deg(tvc_cmd)
        data = data.replace(ctrl=ctrl)
        data = mjx.forward(self._model, data)
        thrust_dir = data.site_xmat[self._site_id] @ jp.array([0.0, 0.0, 1.0])
        force = thrust_dir * main * self._max_thrust * thrust_scale
        lever = data.site_xpos[self._site_id] - data.xpos[self._body_id]
        torque = jp.cross(lever, force)
        force = force - 0.08 * damping_scale * data.qvel[0:3]
        torque = torque - 0.34 * damping_scale * data.qvel[3:6]
        xfrc = jp.zeros_like(data.xfrc_applied)
        xfrc = xfrc.at[self._body_id, 0:3].set(force)
        xfrc = xfrc.at[self._body_id, 3:6].set(torque)
        data = data.replace(xfrc_applied=xfrc)
        data = mjx.step(self._model, data)
        return (data, tvc_cmd), None

    def _observe(self, state: RocketState, reward: jax.Array):
        data = state.data
        q = data.qpos[3:7] / jp.maximum(jp.linalg.norm(data.qpos[3:7]), 1e-8)
        rot = quat_to_mat(q)
        body_axis = rot @ jp.array([0.0, 0.0, 1.0])
        bottom = data.qpos[0:3] + rot @ jp.array([0.0, 0.0, -0.355])
        target = jp.array([0.0, 0.0, 5.0])
        target_delta = target - bottom
        linear = data.qvel[0:3]
        angular = data.qvel[3:6]
        tvc_angles = jp.array([data.qpos[self._yaw_qpos], data.qpos[self._pitch_qpos]])
        tvc_vel = jp.array([data.qvel[self._yaw_qvel], data.qvel[self._pitch_qvel]])
        foot_z = data.geom_xpos[self._foot_geom_ids, 2]
        contacts = (foot_z <= 0.03).astype(jp.float32)
        phase_one_hot = jax.nn.one_hot(state.phase, 5)
        phase_target_z = jp.where(state.phase <= PHASE_FLIP, 10.0, 5.0)
        main_power = jp.clip((state.last_action[0] + 1.0) * 0.5, 0.0, 1.0)
        obs = jp.concatenate([
            target_delta,
            q,
            bottom[2:3],
            linear,
            angular,
            tvc_angles,
            tvc_vel,
            contacts,
            main_power[None],
            phase_one_hot,
            jp.clip(state.flip_progress, 0.0, 1.25)[None],
            jp.clip(state.hover_timer / 5.0, 0.0, 1.0)[None],
            (phase_target_z - bottom[2])[None],
        ])
        flip_axis = rot @ jp.array([0.0, 1.0, 0.0])
        signed_flip_rate = jp.dot(angular, flip_axis)
        positive_flip_rate = jp.maximum(-signed_flip_rate, 0.0)
        metrics = {
            "reward": reward,
            "phase": state.phase.astype(jp.float32),
            "height": bottom[2],
            "upright": body_axis[2],
            "linear_speed": jp.linalg.norm(linear),
            "angular_speed": jp.linalg.norm(angular),
            "horizontal_speed": jp.linalg.norm(linear[:2]),
            "vertical_velocity": linear[2],
            "rel_dist": jp.linalg.norm(bottom[:2]),
            "flip_progress": state.flip_progress,
            "positive_flip_rate": positive_flip_rate,
            "hover_timer": state.hover_timer,
            "success": state.success.astype(jp.float32),
            "fail": state.fail.astype(jp.float32),
            "fail_reason_code": state.fail_reason_code,
            "transition_climb": state.transition_mask[0],
            "transition_flip": state.transition_mask[1],
            "transition_recovery": state.transition_mask[2],
            "transition_hover": state.transition_mask[3],
            "max_tvc": state.max_tvc_seen,
            "max_tvc_speed": state.max_tvc_speed_seen,
            "settling_time": state.settling_time,
            "climb_ready_timer": state.climb_timer,
        }
        return jp.nan_to_num(obs), metrics

    def _reward_and_transition(self, state: RocketState, action: jax.Array):
        data = state.data
        q = data.qpos[3:7] / jp.maximum(jp.linalg.norm(data.qpos[3:7]), 1e-8)
        rot = quat_to_mat(q)
        body_axis = rot @ jp.array([0.0, 0.0, 1.0])
        bottom = data.qpos[0:3] + rot @ jp.array([0.0, 0.0, -0.355])
        linear, angular = data.qvel[0:3], data.qvel[3:6]
        v, w = jp.linalg.norm(linear), jp.linalg.norm(angular)
        rel = jp.linalg.norm(bottom[:2])
        tvc = jp.linalg.norm(jp.array([data.qpos[self._yaw_qpos], data.qpos[self._pitch_qpos]]))
        joint_speed = jp.linalg.norm(jp.array([data.qvel[self._yaw_qvel], data.qvel[self._pitch_qvel]]))
        flip_axis = rot @ jp.array([0.0, 1.0, 0.0])
        signed_rate = jp.dot(angular, flip_axis)
        positive_rate = jp.maximum(-signed_rate, 0.0)
        off_axis = jp.linalg.norm(angular - signed_rate * flip_axis)
        progress_delta = jp.where(state.phase == PHASE_FLIP, positive_rate * self._dt / FULL_FLIP, 0.0)
        flip_angle = state.flip_angle + progress_delta * FULL_FLIP
        flip_progress = flip_angle / FULL_FLIP

        ready_flip = (
            (bottom[2] >= 10.0)
            & (v <= 2.0)
            & (w <= 0.5)
            & (body_axis[2] >= 0.97)
            & (rel <= 1.0)
            & (tvc <= 0.10)
            & (joint_speed <= 0.8)
        )
        ready_recovery = (
            (flip_progress >= 0.95)
            & (body_axis[2] >= 0.85)
            & (rel <= 2.0)
            & (jp.linalg.norm(linear[:2]) <= 4.0)
            & (bottom[2] >= 5.0)
            & (linear[2] >= -8.0)
            & (jp.abs(angular[2]) <= 2.0)
        )
        ready_hover = (
            (flip_progress >= 0.95)
            & (body_axis[2] >= 0.95)
            & (jp.abs(bottom[2] - 5.0) <= 0.8)
            & (rel <= 1.2)
            & (jp.linalg.norm(linear[:2]) <= 1.5)
            & (jp.abs(linear[2]) <= 1.5)
            & (w <= 1.5)
        )
        hover_stable = (
            (jp.abs(bottom[2] - 5.0) <= 1.0)
            & (body_axis[2] >= 0.90)
            & (rel <= 1.2)
            & (v <= 2.0)
            & (w <= 2.0)
        )

        climb_reward = (
            2.0 * jp.clip((bottom[2] - state.last_height) / 0.05, -1.0, 1.0)
            + 1.2 * jp.exp(-jp.square((10.0 - bottom[2]) / 3.0))
            + 0.8 * body_axis[2]
            - 0.35 * rel
            - 0.15 * w
            - 0.3 * jp.square(jp.maximum(linear[2] - jp.clip((10.0 - bottom[2]) / 3.0, 0.0, 1.6), 0.0))
        )
        expected = jp.array([-jp.sin(jp.minimum(flip_progress, 1.0) * FULL_FLIP), 0.0, jp.cos(jp.minimum(flip_progress, 1.0) * FULL_FLIP)])
        flip_reward = (
            14.0 * progress_delta
            + 3.0 * jp.tanh(positive_rate / 6.0)
            + 2.4 * jp.clip((flip_progress - 0.65) / 0.35, 0.0, 1.0) ** 2 * jp.dot(body_axis, expected)
            - 0.45 * off_axis / (off_axis + 5.0)
            - 0.65 * jp.abs(angular[2]) / (jp.abs(angular[2]) + 3.0)
            - 0.4 * rel
        )
        recovery_reward = (
            2.5 * body_axis[2]
            + 1.5 * jp.exp(-jp.square((bottom[2] - 5.0) / 1.5))
            + 1.0 * jp.exp(-jp.square(v / 3.0))
            + 1.0 * jp.exp(-jp.square(w / 3.0))
            - 0.5 * rel
            - 0.4 * jp.maximum(-linear[2] - 2.0, 0.0)
        )
        hover_reward = (
            2.0 * jp.exp(-jp.square((bottom[2] - 5.0) / 0.8))
            + 1.5 * body_axis[2]
            + 1.2 * jp.exp(-jp.square(v / 1.0))
            + 1.2 * jp.exp(-jp.square(w / 0.8))
            - 0.5 * rel
        )
        non_flip_phase = (state.phase != PHASE_FLIP).astype(jp.float32)
        tvc_angle_norm = tvc / jp.maximum(self._max_tvc, 1e-8)
        tvc_angle_penalty = non_flip_phase * (tvc_angle_norm * tvc_angle_norm) / (tvc_angle_norm * tvc_angle_norm + 1.0)
        tvc_velocity_norm = joint_speed / 12.0
        tvc_velocity_penalty = non_flip_phase * (tvc_velocity_norm * tvc_velocity_norm) / (tvc_velocity_norm * tvc_velocity_norm + 1.0)

        is_climb_ready = (state.phase == PHASE_CLIMB) & (bottom[2] >= 10.0)
        climb_timer = jp.where(is_climb_ready, state.climb_timer + self._dt, 0.0)

        reward = jp.array([
            climb_reward,
            flip_reward,
            recovery_reward,
            hover_reward,
        ])[jp.minimum(state.phase, PHASE_HOVER)]
        reward -= 1.2 * tvc_angle_penalty
        reward -= 2.0 * tvc_velocity_penalty
        reward -= 25.0 * climb_timer
        reward -= 0.05 * jp.mean(jp.square(action[1:]))
        reward -= 0.15 * jp.mean(jp.square(action - state.last_action))
        reward -= 0.01

        next_phase = state.phase
        next_phase = jp.where((state.phase == PHASE_CLIMB) & ready_flip, PHASE_FLIP, next_phase)
        next_phase = jp.where((state.phase == PHASE_FLIP) & ready_recovery, PHASE_RECOVERY, next_phase)
        next_phase = jp.where((state.phase == PHASE_RECOVERY) & ready_hover, PHASE_HOVER, next_phase)
        hover_timer = jp.where(
            next_phase == PHASE_HOVER,
            jp.where(hover_stable, state.hover_timer + self._dt, 0.0),
            0.0,
        )
        # Check transition successes
        climb_success = (state.phase == PHASE_CLIMB) & ready_flip
        flip_success = (state.phase == PHASE_FLIP) & ready_recovery
        recovery_success = (state.phase == PHASE_RECOVERY) & ready_hover
        hover_success = (next_phase == PHASE_HOVER) & (hover_timer >= 5.0)

        success = jp.select(
            [
                self._specialist_phase == 0,
                self._specialist_phase == 1,
                self._specialist_phase == 2,
                self._specialist_phase == 3,
            ],
            [
                climb_success,
                flip_success,
                recovery_success,
                hover_success,
            ],
            default=hover_success
        )

        bad_physics = ~jp.all(jp.isfinite(data.qpos)) | ~jp.all(jp.isfinite(data.qvel))
        xy_escape = (jp.abs(bottom[0]) > 6.0) | (jp.abs(bottom[1]) > 6.0)
        speed_limit = (v > 70.0) | (w > 70.0)
        surface = bottom[2] <= 0.05

        fail = surface | xy_escape | speed_limit | bad_physics

        fail_code = jp.where(bad_physics, 1.0, 0.0)
        fail_code = jp.where(
            (fail_code == 0.0) & surface,
            jp.where(state.phase == PHASE_FLIP, 4.0, 9.0),
            fail_code
        )
        fail_code = jp.where(
            (fail_code == 0.0) & xy_escape,
            jp.select(
                [state.phase == PHASE_CLIMB, state.phase == PHASE_FLIP, state.phase == PHASE_RECOVERY, state.phase == PHASE_HOVER],
                [2.0, 5.0, 6.0, 7.0],
                default=0.0
            ),
            fail_code
        )
        fail_code = jp.where(
            (fail_code == 0.0) & speed_limit,
            jp.select(
                [state.phase == PHASE_CLIMB, state.phase == PHASE_FLIP, state.phase == PHASE_RECOVERY, state.phase == PHASE_HOVER],
                [10.0, 11.0, 12.0, 8.0],
                default=13.0,
            ),
            fail_code,
        )

        reward += jp.where(ready_flip & (state.phase == PHASE_CLIMB), 40.0, 0.0)
        reward += jp.where(ready_recovery & (state.phase == PHASE_FLIP), 80.0, 0.0)
        reward += jp.where(ready_hover & (state.phase == PHASE_RECOVERY), 60.0, 0.0)
        reward += jp.where(success, 320.0, 0.0)
        reward -= jp.where(fail, 220.0, 0.0)
        next_phase = jp.where(success | fail, PHASE_DONE, next_phase)
        transitions = state.transition_mask
        transitions = transitions.at[0].set(jp.maximum(transitions[0], ready_flip.astype(jp.float32)))
        transitions = transitions.at[1].set(jp.maximum(transitions[1], ready_recovery.astype(jp.float32)))
        transitions = transitions.at[2].set(jp.maximum(transitions[2], ready_hover.astype(jp.float32)))
        transitions = transitions.at[3].set(jp.maximum(transitions[3], success.astype(jp.float32)))
        events = jp.array([
            ready_flip & (state.phase == PHASE_CLIMB),
            ready_recovery & (state.phase == PHASE_FLIP),
            ready_hover & (state.phase == PHASE_RECOVERY),
        ])
        transition_height = jp.where(
            events & jp.isnan(state.transition_height),
            bottom[2],
            state.transition_height,
        )
        transition_linear_speed = jp.where(
            events & jp.isnan(state.transition_linear_speed),
            v,
            state.transition_linear_speed,
        )
        transition_angular_speed = jp.where(
            events & jp.isnan(state.transition_angular_speed),
            w,
            state.transition_angular_speed,
        )
        return (
            reward,
            next_phase,
            flip_angle,
            flip_progress,
            hover_timer,
            climb_timer,
            success,
            fail,
            fail_code,
            transitions,
            transition_height,
            transition_linear_speed,
            transition_angular_speed,
        )

    def step(self, state: base.State, action: jax.Array) -> base.State:
        rocket: RocketState = state.pipeline_state
        action = jp.clip(action, -1.0, 1.0)
        main = jp.clip((action[0] + 1.0) * 0.5, 0.0, 1.0)
        tvc_target = action[1:3] * self._max_tvc
        inputs = (
            jp.repeat(main[None], self._frame_skip, axis=0),
            jp.repeat(tvc_target[None, :], self._frame_skip, axis=0),
            jp.repeat(rocket.thrust_scale[None], self._frame_skip, axis=0),
            jp.repeat(rocket.servo_scale[None], self._frame_skip, axis=0),
            jp.repeat(rocket.damping_scale[None], self._frame_skip, axis=0),
        )
        (data, tvc_cmd), _ = jax.lax.scan(
            self._physics_substep,
            (rocket.data, rocket.tvc_cmd),
            inputs,
        )
        candidate = rocket.replace(data=data, tvc_cmd=tvc_cmd)
        (
            reward,
            phase,
            flip_angle,
            flip_progress,
            hover_timer,
            climb_timer,
            success,
            fail,
            fail_code,
            transitions,
            transition_height,
            transition_linear_speed,
            transition_angular_speed,
        ) = self._reward_and_transition(candidate, action)
        episode_step = rocket.episode_step + 1
        timeout = episode_step >= self._episode_length
        done = success | fail | timeout
        q = data.qpos[3:7] / jp.maximum(jp.linalg.norm(data.qpos[3:7]), 1e-8)
        bottom_height = data.qpos[2] + (quat_to_mat(q) @ jp.array([0.0, 0.0, -0.355]))[2]
        current_tvc = jp.max(jp.abs(jp.array([
            data.qpos[self._yaw_qpos], data.qpos[self._pitch_qpos]
        ])))
        current_tvc_speed = jp.max(jp.abs(jp.array([
            data.qvel[self._yaw_qvel], data.qvel[self._pitch_qvel]
        ])))
        settling_time = jp.where(
            (rocket.settling_time < 0.0) & (hover_timer > 0.0),
            episode_step * self._dt,
            rocket.settling_time,
        )
        rocket = candidate.replace(
            phase=jp.where(done, PHASE_DONE, phase),
            flip_angle=flip_angle,
            flip_progress=flip_progress,
            hover_timer=hover_timer,
            climb_timer=climb_timer,
            last_action=action,
            last_height=bottom_height,
            last_vertical_velocity=data.qvel[2],
            episode_step=episode_step,
            success=success,
            fail=fail,
            fail_reason_code=fail_code,
            transition_mask=transitions,
            transition_height=transition_height,
            transition_linear_speed=transition_linear_speed,
            transition_angular_speed=transition_angular_speed,
            settling_time=settling_time,
            max_tvc_seen=jp.maximum(rocket.max_tvc_seen, current_tvc),
            max_tvc_speed_seen=jp.maximum(
                rocket.max_tvc_speed_seen, current_tvc_speed
            ),
        )
        obs, metrics = self._observe(rocket, reward)
        return base.State(
            pipeline_state=rocket,
            obs=obs,
            reward=reward,
            done=done.astype(jp.float32),
            metrics=metrics,
            info=state.info,
        )
