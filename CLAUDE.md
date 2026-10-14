# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GPU-native MJX (MuJoCo XLA / JAX) reinforcement learning workspace for two tasks:

1. **Rocket TVC (thrust-vector control) phase chain**: `climb -> flip -> recovery -> hover`, trained with SAC on GPU via Brax.
2. **Table-gimbal stabilization**: a separate simulated task with its own curriculum, sharing the same MJX/Brax training machinery.

The public workflow is **MJX-first**. Classic MuJoCo/Gymnasium + Stable-Baselines3 code remains only as reference/visualization tooling; new training work should go through the MJX commands below, not the legacy SB3 path.

## Build & Environment Setup

Two separate virtualenvs are used — do not mix them:

**Legacy/reference utilities** (SB3, classic MuJoCo, plotting):
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**MJX GPU stack** (pinned Python 3.11, JAX/Brax/MuJoCo-MJX):
```bash
python3.11 -m venv .venv-mjx
.venv-mjx/bin/python -m pip install -r requirements-mjx.lock
```
`requirements-mjx.lock` pins JAX, Brax, MuJoCo, and MJX versions together — never upgrade only one of these packages without rerunning `mjx-doctor` and the test suite. `requirements-mjx.in` is the unpinned source; regenerate the lock rather than hand-editing it.

Verify the MJX environment after any dependency change:
```bash
.venv-mjx/bin/python rl.py mjx-doctor
.venv-mjx/bin/python rl.py benchmark-mjx --num-envs 128 256 512
```
`mjx-doctor` also runs a single-step parity check between the classic MuJoCo `HopperEnv` and the JAX `RocketMJXEnv` (see `mjx/cli.py::_single_step_parity`) — a useful sanity check any time env dynamics change.

## Common Commands

Everything routes through the single CLI entry point `rl.py` (argparse subcommands dispatch to `algorithms/*.py` or `mjx/`/`gimbal/` modules — see the `main()` dispatcher at the bottom of `rl.py`).

Rocket MJX training/eval:
```bash
.venv-mjx/bin/python rl.py train-mjx --total-env-steps 100000 --curriculum-stages 1 --num-envs 128 --run-dir runs/mjx_sac_smoke   # smoke run
.venv-mjx/bin/python rl.py eval-mjx --model runs/mjx_sac_chain_<run> --episodes 1000
.venv-mjx/bin/python rl.py watch-mjx --model runs/mjx_sac_chain_<run>
```

Table-gimbal MJX training/eval:
```bash
.venv-mjx/bin/python rl.py train-gimbal-mjx --total-env-steps 8000000 --curriculum-stages 3 --num-envs 512 --run-dir runs/mjx_gimbal_sac_main
.venv-mjx/bin/python rl.py eval-gimbal-mjx --model runs/mjx_gimbal_sac_main --episodes 100
.venv-mjx/bin/python rl.py watch-gimbal-mjx --model runs/mjx_gimbal_sac_main --episodes 3   # replays the JAX policy in the classic rendered GimbalEnv
```

Legacy SB3 path (reference only): `rl.py train --algo sac|ppo|td3`, `rl.py watch`, `rl.py eval`, `rl.py train-specialists` (per-phase specialist fine-tuning).

## Tests

```bash
python -m pytest -q                              # full suite (pytest.ini restricts discovery to tests/)
.venv-mjx/bin/python -m pytest -q tests/test_mjx_env.py
.venv-mjx/bin/python -m pytest -q tests/test_gimbal_env.py
python -m py_compile rl.py mjx/*.py gimbal/*.py rl_common.py hopper_env.py algorithms/*.py   # fast syntax check
```
Some MJX/gimbal tests require the `.venv-mjx` environment (JAX/Brax/MuJoCo-MJX); classic-env tests (`test_handoff_specialists.py`) only need `gymnasium`/`mujoco` and use `pytest.importorskip` to skip cleanly if unavailable. Name new test files `test_*.py` and functions `test_*`; add focused tests for environment dynamics, router/phase handoffs, and MJX behavior whenever those surfaces change.

Acceptance targets for the rocket chain: `climb_handoff_rate >= 0.90`, `flip_handoff_rate >= 0.85`, `recovery_handoff_rate >= 0.85`, `hover_success_rate >= 0.90`, `chain_success_rate >= 0.80`. The handoff tests use the 10m climb-to-flip standard.

## Architecture

### Phase chain and handoffs (the central concept)

The rocket task is a 4-phase state machine — `PHASE_CLIMB=0, PHASE_FLIP=1, PHASE_RECOVERY=2, PHASE_HOVER=3` (see `mjx/env.py`) — mirrored in the classic env in `hopper_env.py`. Policies can be trained either:
- **end-to-end** across the full chain, or
- as **per-phase specialists**, each starting mid-chain from a distribution of states representative of what the previous phase hands off (`--reset-profile handoff-mix`, `HandoffStartWrapper` / `PhaseChainHandoffStartWrapper` in `rl_common.py`).

A trained specialist chain is wired together via a **phase-models config** (JSON mapping phase name -> checkpoint path, see `configs/phase_models.example.json`), produced by `rl.py make-phase-config` and consumed via `--phase-models-config` / `--handoff-phase-models-config`. `rl_common.load_phase_models` / `phase_model_for_env` do the runtime lookup. Changing phase thresholds, handoff ranges, or reset distributions requires keeping `mjx/env.py` and `hopper_env.py` in sync (the parity check in `mjx-doctor` and `tests/test_handoff_specialists.py` / `tests/test_mjx_env.py` guard against drift).

### Two independent tasks sharing one training stack

- `mjx/` — rocket: `env.py` (JAX `RocketMJXEnv`, a Brax `base.Env`, holds `RocketState`, phase logic, quaternion helpers), `cli.py` (doctor/benchmark/train/eval/watch, SAC via Brax, includes monkeypatches on `brax.training.agents.sac.train` for checkpoint restore/replication across devices — see `_require_jax()`).
- `gimbal/` — table-gimbal: `env.py` (classic Gymnasium `GimbalEnv` for rendering/inspection), `env_mjx.py` (JAX `GimbalMJXEnv`/`GimbalMJXState`, mirrors the rocket env's structure), `mjx_cli.py` (train/evaluate/watch), `model.xml` (MuJoCo model). `watch-gimbal-mjx` deliberately replays a JAX-trained policy inside the classic rendered env for visual inspection — that cross-env replay path is intentional, not a leftover.
- `algorithms/` — thin SB3 wrappers (`sac.py`, `ppo.py`, `td3.py`) used only by the legacy `rl.py train/watch/eval` path; loaded lazily via `load_algo()` in `rl.py` so SB3 isn't a hard dependency of the MJX path.
- `rl_common.py` — shared, algorithm-agnostic infrastructure used by the legacy path: env factory (`make_env`), the handoff wrappers, the full `train_loop`, eval/watch helpers, phase-config helpers, CSV/plotting (`plot_run`). This is the largest shared module — check here before adding new cross-cutting training/eval logic.
- `hopper_env.py` — the classic (non-JAX) Gymnasium rocket environment; source of truth for phase/handoff semantics that `mjx/env.py` must stay in parity with.
- `telegram_notifier.py` — optional training-progress notifications (`--telegram-config`, `--telegram-every`, `--no-telegram`); driven by a local, gitignored secrets file (see `configs/telegram_secrets.example.json`).
- `compare_tvc.py` / `compare_tvc_mjx.py` / `visualize.py` — standalone comparison/inspection scripts against checkpoints, not part of the CLI subcommand tree.

### Outputs

Training writes to `runs/<run_dir>/` (checkpoints, step CSVs, videos). `runs/` is git-ignored — never commit run directories, checkpoints, generated plots, or machine-specific configs. `models/` holds optional published model placeholders only.

## Coding Conventions

Python 3, 4-space indentation, snake_case for functions/variables/CLI flags, PascalCase reserved for classes. Keep new CLI flags explicit and consistent with the existing patterns in `mjx/cli.py`/`rl.py` (argparse subparsers, `add_*_args` helper functions per subcommand family). Prefer small helper functions colocated in the owning module over new shared abstractions.

## Commit Style

History mixes Conventional Commit prefixes (`feat:`, `fix:`, `docs:`) with occasional short `push` commits — prefer the descriptive prefix style. PRs should describe the training/runtime behavior changed, list commands run, and include plots/screenshots when visual behavior changes.
