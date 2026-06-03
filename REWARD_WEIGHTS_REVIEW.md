# Reward Weights Review

This project now has two reward modes:

- `legacy`: the original phase-specific reward formulas used by TD3 and PPO.
- `dense`: the SAC-only dense shaping reward selected through `ENV_KWARGS = {"reward_mode": "dense"}` in `sac.py`.

`hopper_env.py` still owns the physics metrics, phase state, terminal conditions, and shared logging. Algorithm-specific weight values stay in each algorithm file.

## SAC Dense Reward

SAC no longer uses milestone bonuses, completion pressure, no-progress penalties, or phase transition rewards. The phase machine still changes `climb -> flip -> recovery -> hover`, but `transition_bonus` is forced to `0.0` in dense mode.

### Positive Dense Components

| Log key | Weight key | Meaning |
| --- | --- | --- |
| `reward_position_score` | `dense_position` | Smooth horizontal target closeness. |
| `reward_height_score` | `dense_height` | Smooth closeness to the current phase target height. |
| `reward_upright_score` | `dense_upright` | Normalized uprightness from 0 to 1. |
| `reward_velocity_score` | `dense_velocity` | Rewards controlled, lower linear speed. |
| `reward_flip_axis_score` | `dense_flip_axis`, `dense_flip_progress_delta` | Rewards intended flip-axis rotation and only positive progress delta. |
| `reward_flip_upright_recovery_score` | `dense_flip_upright_recovery` | Smoothly rewards becoming upright, low-spin, and near target after most of the flip is complete. |
| `reward_hover_stability_score` | `dense_hover_stability` | Rewards upright, low-speed, low-angular-speed stability near hover target. |

### Continuous Penalties

| Log key | Weight key | Meaning |
| --- | --- | --- |
| `penalty_drift` | `dense_drift` | Smooth lateral drift penalty. |
| `penalty_angular` | `dense_angular` | Smooth excessive angular speed penalty. |
| `penalty_off_axis` | `dense_off_axis` | Penalizes rotation away from the intended flip axis. |
| `penalty_yaw_spin` | `dense_yaw_spin` | Penalizes world-z yaw spin exploit. |
| `penalty_control_effort` | `dense_control_effort` | Penalizes large action magnitude. |
| `penalty_action_smoothness` | `dense_action_smoothness` | Penalizes abrupt action changes. |
| `penalty_safety` | `dense_safety` | Penalizes unsafe low-altitude states continuously. |
| `penalty_overrotate` | `dense_overrotate` | Smoothly penalizes rotating well past one flip. |

`reward_dense_total` is the sum of these dense components before terminal success/failure and `time_penalty`.

In `flip` phase, static "stay calm/upright" rewards are intentionally reduced so the policy does not learn to hover at the flip start:

- `reward_upright_score` is disabled during flip.
- `reward_position_score` is scaled to 20 percent during flip.
- `reward_height_score` is scaled to 25 percent during flip.
- `reward_velocity_score` is scaled to 12 percent during flip.
- `reward_flip_axis_score` remains the main positive signal during flip.
- `reward_flip_upright_recovery_score` opens smoothly near the end of the flip so the policy learns to stop the rotation instead of spinning forever.
- `penalty_angular` scales up after roughly 65 percent flip progress, so braking late rotation is rewarded without suppressing early spin exploration.
- `penalty_safety` also penalizes climbing above the flip corridor, so full-thrust vertical escape is not a better alternative to rotating.
- `penalty_overrotate` grows after one flip, reducing the value of endless spin.

### SAC Safety And Entropy

| Key | SAC value | Notes |
| --- | ---: | --- |
| `time_penalty` | 0.01 | Small anti-stall pressure. |
| `failure_penalty` | 25.0 | Modest terminal failure cost. |
| `success_bonus` | 50.0 | Modest terminal success reward. |
| `flip_xy_escape_penalty` | 30.0 | Extra terminal cost for lateral flip escape. |
| `flip_surface_contact_penalty` | 40.0 | Extra terminal cost for flip ground contact. |
| `flip_rel_dist_limit` | 5.0 | Hard flip escape limit used by fail logic. |
| `--sac-ent-coef` | `auto_0.02` | Passed directly to Stable-Baselines3 SAC `ent_coef`. |

## Legacy TD3/PPO Reward

TD3 and PPO keep the existing legacy mode. In this mode, reward is selected by current phase:

- `climb`: target altitude, uprightness, drift, angular speed, and climb progress shaping.
- `flip`: progress, axis rate, completion pressure, no-progress penalty, altitude/descent shaping, lateral escape penalties, off-axis/yaw penalties.
- `recovery`: upright recovery, target distance, altitude, descent, thrust alignment, and recovery progress.
- `hover`: upright hover, target height, velocity damping, and target-distance shaping.

Legacy phase transition bonuses are still active only in legacy mode:

- `phase_climb_to_flip_bonus`
- `phase_flip_to_recovery_bonus`
- `phase_recovery_to_hover_bonus`
- `flip_completion_bonus`

## Shared Terminal Conditions

These fail/success mechanics still apply to both reward modes:

| Condition | Phase | Effect |
| --- | --- | --- |
| `bad_physics` | Any | Terminates on non-finite values or emergency physics jumps. |
| `surface_contact` | Any | Terminates on unexpected ground/pad contact. |
| `flip_surface_contact` | Flip | Terminates flip on surface contact. |
| `flip_xy_escape` | Flip | Terminates when `rel_dist > flip_rel_dist_limit`. |
| `flip_too_low_too_early` | Flip | Terminates if altitude is below 3.5 m while progress is below 0.15. |
| `recovery_target_escape` | Recovery | Terminates when `rel_dist > recovery_max_rel_dist`. |
| `hover_target_escape` | Hover | Terminates when `rel_dist > hover_max_rel_dist`. |
| `hover_speed_limit` | Hover | Terminates when linear or angular speed exceeds 70. |
| `success` | Hover | Requires stable hover for 5 seconds. |

## Logging

`rl_common.py` writes all dense breakdown keys into train CSV and watch CSV through `EXTRA_INFO_KEYS`. In legacy mode these fields are present but zero, so TD3/PPO logs remain schema-compatible.

Use headless evaluation after training to compare checkpoints without opening the viewer:

```bash
python rl.py eval --algo sac --model runs/<run>/sac_hopper_latest.zip --start-phase flip --fixed-start-z --start-z 11 --max-thrust 45 --episodes 5 --csv eval_dense_sac.csv
```

To evaluate all checkpoints in a run:

```bash
python rl.py eval --algo sac --run-dir runs/<run> --start-phase flip --fixed-start-z --start-z 11 --max-thrust 45 --episodes 3 --csv eval_dense_sac_checkpoints.csv
```

The eval summary prints max flip progress, max axis rate, max target drift, min height, final fail reason, and overrotate penalty.

Use these columns to detect dominance:

- Dense reward too positive everywhere: lower `dense_position`, `dense_height`, or `dense_upright`.
- Too passive: lower `dense_control_effort`, `dense_action_smoothness`, or raise `--sac-ent-coef`.
- Sideways flip exploit: raise `dense_drift`, `dense_yaw_spin`, or `dense_off_axis`.
- Falls before rotating: raise `dense_safety`, `dense_flip_axis`, or `dense_flip_progress_delta`.
- Spins past one flip: raise `dense_overrotate` or `dense_flip_upright_recovery`.
