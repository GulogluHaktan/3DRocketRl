# SAC Specialist Training Runbook

This runbook trains four 31-observation SAC specialists without overwriting the golden flip model.

Use the project venv:

```bash
PY=/home/haktan/Documents/RL/MuJoCo/.venv/bin/python
```

## 1. Train Specialists

Train `climb` first:

```bash
$PY rl.py train --algo sac --specialist-phase climb --timesteps 500000 --chunk-steps 25000 --start-phase climb --fixed-start-z --start-z 2 --max-thrust 45
```

Use the suite command to train all four single-motor SAC specialists overnight:

```bash
$PY rl.py train-specialists --algo sac --timesteps-per-phase 500000 --chunk-steps 25000 --max-thrust 45 --telegram-every 100000 --telegram-video-every 100000
```

## Resume Existing Single-Motor Specialists

Use new run directories when adapting copied specialists to the stricter phase
handoff conditions:

```bash
CLIMB=runs/sac_climb_20260609_204157/sac_hopper_latest.zip
FLIP=runs/sac_flip_20260610_023105/sac_hopper_latest.zip
RECOVERY=runs/sac_recovery_20260610_085854/sac_hopper_latest.zip
HOVER=runs/sac_hover_20260610_135859/sac_hopper_latest.zip
```

```bash
$PY rl.py train --algo sac --resume "$CLIMB" --specialist-phase climb --start-phase climb --fixed-start-z --start-z 2 --max-thrust 45 --timesteps 300000 --chunk-steps 25000 --telegram-every 100000
$PY rl.py train --algo sac --resume "$FLIP" --specialist-phase flip --start-phase flip --fixed-start-z --start-z 11 --max-thrust 45 --timesteps 300000 --chunk-steps 25000 --telegram-every 100000
$PY rl.py train --algo sac --resume "$RECOVERY" --specialist-phase recovery --start-phase recovery --fixed-start-z --start-z 9 --max-thrust 45 --timesteps 200000 --chunk-steps 25000 --telegram-every 100000
$PY rl.py train --algo sac --resume "$HOVER" --specialist-phase hover --start-phase hover --fixed-start-z --start-z 5 --max-thrust 45 --timesteps 150000 --chunk-steps 25000 --telegram-every 100000
```

## Real Handoff Resume

Use this when specialists work alone but the router handoff distribution breaks
them. The teacher specialists in `phase_models.json` fly from `climb` until the
target phase is reached, then the resumed specialist learns from that exact
state.

```bash
$PY rl.py train --algo sac --resume "$CLIMB" --specialist-phase climb --start-phase climb --fixed-start-z --start-z 2 --max-thrust 45 --timesteps 150000 --chunk-steps 25000 --telegram-every 50000

$PY rl.py train --algo sac --resume "$FLIP" --specialist-phase flip --start-phase climb --handoff-phase-models-config phase_models.json --fixed-start-z --start-z 2 --max-thrust 45 --timesteps 200000 --chunk-steps 25000 --telegram-every 50000

$PY rl.py train --algo sac --resume "$RECOVERY" --specialist-phase recovery --start-phase climb --handoff-phase-models-config phase_models.json --fixed-start-z --start-z 2 --max-thrust 45 --timesteps 200000 --chunk-steps 25000 --telegram-every 50000

$PY rl.py train --algo sac --resume "$HOVER" --specialist-phase hover --start-phase climb --handoff-phase-models-config phase_models.json --fixed-start-z --start-z 2 --max-thrust 45 --timesteps 150000 --chunk-steps 25000 --telegram-every 50000
```

For standalone noisy reset adaptation, add for example:

```bash
--phase-start-roughness 0.3
```

Train `recovery` second:

```bash
$PY rl.py train --algo sac --specialist-phase recovery --timesteps 500000 --chunk-steps 25000 --start-phase recovery --fixed-start-z --start-z 9 --max-thrust 45
```

SAC recovery uses a rough post-flip reset distribution. Even with `--fixed-start-z`,
the reset randomizes height, drift, attitude, linear velocity, angular velocity, and
`flip_progress` around a messy handoff so the specialist does not overfit to a clean
upright recovery state.

Train `hover` third:

```bash
$PY rl.py train --algo sac --specialist-phase hover --timesteps 500000 --chunk-steps 25000 --start-phase hover --fixed-start-z --start-z 5 --max-thrust 45
```

Keep the golden flip model as fallback. Train a new flip specialist only after climb/recovery/hover are usable:

```bash
$PY rl.py train --algo sac --specialist-phase flip --timesteps 500000 --chunk-steps 25000 --start-phase flip --fixed-start-z --start-z 11 --max-thrust 45
```

## 2. Evaluate Each Specialist

Use specialist eval so phase handoff success is counted.

```bash
$PY rl.py eval --algo sac --model runs/<climb_run>/sac_hopper_latest.zip --specialist-phase climb --start-phase climb --fixed-start-z --start-z 2 --max-thrust 45 --episodes 10 --csv eval_climb.csv
$PY rl.py eval --algo sac --model runs/<recovery_run>/sac_hopper_latest.zip --specialist-phase recovery --start-phase recovery --fixed-start-z --start-z 9 --max-thrust 45 --episodes 10 --csv eval_recovery.csv
$PY rl.py eval --algo sac --model runs/<hover_run>/sac_hopper_latest.zip --specialist-phase hover --start-phase hover --fixed-start-z --start-z 5 --max-thrust 45 --episodes 10 --csv eval_hover.csv
```

Look for:

- `specialist_successes` increasing.
- `specialist_handoffs` matching the next phase: `flip`, `recovery`, or `hover`.
- Per-step handoff flags becoming true before specialist success:
  `ready_for_flip`, `ready_for_recovery`, `ready_for_hover`, and `hover_stable`.
- Low `fail_reasons` for target escape and surface contact.

## 3. Build Router Config

Generate config from latest specialist runs:

```bash
$PY rl.py make-phase-config --output phase_models.json --fallback-flip runs/sac_hopper_20260603_152230/sac_hopper_latest.zip
```

Require all specialists before full router eval:

```bash
$PY rl.py make-phase-config --output phase_models.json --require-all --fallback-flip runs/sac_hopper_20260603_152230/sac_hopper_latest.zip
```

## 4. Full Router Eval

```bash
$PY rl.py eval --algo sac --phase-models-config phase_models.json --start-phase climb --fixed-start-z --start-z 2 --max-thrust 45 --episodes 5 --csv router_eval.csv
```

Good signal:

- `phase_sequence` reaches `climb>flip>recovery>hover`.
- `active_model_sequence` follows the same phase order.
- Golden flip fallback still reaches about `flip_progress=0.95+` from `start-phase flip`.
