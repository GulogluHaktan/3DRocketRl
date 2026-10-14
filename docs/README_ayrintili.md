# 3D Rocket MJX RL

GPU-native MJX training workspace for a TVC rocket phase chain and a table-gimbal stabilization task.

Main rocket chain:

```text
climb -> flip -> recovery -> hover
```

The public workflow is MJX-first. Classic MuJoCo/SB3 files remain as reference utilities for comparison and visualization, but new training should start from the MJX commands below.

## Project Layout

```text
3drocket/
├── rl.py                 # Unified CLI
├── mjx/                  # MJX rocket environment and SAC train/eval helpers
├── gimbal/               # Table-gimbal environments and MJX/SB3 training code
├── assets/               # Rocket MuJoCo XML assets
├── algorithms/           # Legacy SB3 algorithm wrappers
├── tests/                # Pytest coverage for MJX, gimbal, and handoff rules
├── configs/              # Example local configs
├── docs/                 # Technical notes
├── models/               # Optional published model placeholders
└── runs/                 # Generated training outputs, ignored by git
```

Do not commit run directories, checkpoints, generated plots, local secrets, or machine-specific configs.

## MJX Setup

Use Python 3.11 for the pinned JAX/Brax/MuJoCo stack:

```bash
python3.11 -m venv .venv-mjx
.venv-mjx/bin/python -m pip install -r requirements-mjx.lock
```

Verify the environment:

```bash
.venv-mjx/bin/python rl.py mjx-doctor
.venv-mjx/bin/python rl.py benchmark-mjx --num-envs 128 256 512
```

The lock file pins JAX, Brax, MuJoCo, and MJX versions together. Do not upgrade only one of these packages without rerunning the doctor and tests.

## Rocket MJX Training

Short smoke run:

```bash
.venv-mjx/bin/python rl.py train-mjx \
  --total-env-steps 100000 \
  --curriculum-stages 1 \
  --num-envs 128 \
  --run-dir runs/mjx_sac_smoke
```

Main training profile:

```bash
.venv-mjx/bin/python rl.py train-mjx \
  --total-env-steps 50000000 \
  --curriculum-stages 3 \
  --num-envs 512 \
  --num-eval-envs 128 \
  --batch-size 512 \
  --min-replay-size 32768 \
  --max-replay-size 262144 \
  --grad-updates-per-step 16
```

Evaluate and visualize:

```bash
.venv-mjx/bin/python rl.py eval-mjx --model runs/mjx_sac_chain_<run> --episodes 1000
.venv-mjx/bin/python rl.py watch-mjx --model runs/mjx_sac_chain_<run>
```

Acceptance targets are `climb_handoff_rate >= 0.90`, `flip_handoff_rate >= 0.85`, `recovery_handoff_rate >= 0.85`, `hover_success_rate >= 0.90`, and `chain_success_rate >= 0.80`.

## Table Gimbal MJX

Train the table-gimbal SAC curriculum:

```bash
.venv-mjx/bin/python rl.py train-gimbal-mjx \
  --total-env-steps 8000000 \
  --curriculum-stages 3 \
  --num-envs 512 \
  --run-dir runs/mjx_gimbal_sac_main
```

Smoke profile:

```bash
.venv-mjx/bin/python rl.py train-gimbal-mjx \
  --total-env-steps 1000 \
  --curriculum-stages 1 \
  --num-envs 8 \
  --num-eval-envs 4 \
  --run-dir runs/mjx_gimbal_smoke
```

Evaluate and watch:

```bash
.venv-mjx/bin/python rl.py eval-gimbal-mjx --model runs/mjx_gimbal_sac_main --episodes 100
.venv-mjx/bin/python rl.py watch-gimbal-mjx --model runs/mjx_gimbal_sac_main --episodes 3
```

The watch command replays the JAX policy in the classic rendered `GimbalEnv` so the table-gimbal behavior can be inspected visually.

## Tests

Run the active checks from the MJX environment:

```bash
.venv-mjx/bin/python -m pytest -q tests/test_mjx_env.py
.venv-mjx/bin/python -m pytest -q tests/test_gimbal_env.py
.venv-mjx/bin/python -m py_compile rl.py mjx/*.py gimbal/*.py
```

The MJX rocket handoff tests use the 10 m climb-to-flip standard.
