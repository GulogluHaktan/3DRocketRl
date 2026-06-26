# Repository Guidelines

## Project Structure & Module Organization

This repository trains and evaluates MJX rocket control agents, plus a table-gimbal simulation workflow. Core CLI entry points live at the root in `rl.py`. Rocket MJX code is in `mjx/`; table-gimbal simulation and MJX helpers are in `gimbal/`. Legacy SB3 wrappers are in `algorithms/` for comparison. MuJoCo XML assets live in `assets/` and `gimbal/model.xml`. Tests live in `tests/`. Training outputs belong in `runs/` and must not be committed.

## Build, Test, and Development Commands

Create the standard environment for legacy utilities:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the active test suite:

```bash
python -m pytest -q
```

Check syntax for the main modules:

```bash
python -m py_compile rl.py rl_common.py hopper_env.py algorithms/*.py gimbal/*.py
```

MJX GPU work uses the pinned Python 3.11 environment:

```bash
python3.11 -m venv .venv-mjx
.venv-mjx/bin/python -m pip install -r requirements-mjx.lock
.venv-mjx/bin/python rl.py mjx-doctor
```

Common MJX commands include `.venv-mjx/bin/python rl.py train-mjx ...`, `.venv-mjx/bin/python rl.py eval-mjx ...`, and `.venv-mjx/bin/python rl.py train-gimbal-mjx ...`.

## Coding Style & Naming Conventions

Use Python 3 style with 4-space indentation, snake_case for functions, variables, and CLI options, and PascalCase only for classes. Keep flags explicit and consistent with existing `rl.py` patterns. Prefer small helper functions in the owning module. Do not commit generated checkpoints, large logs, or new root-level artifacts.

## Testing Guidelines

Tests use `pytest`; `pytest.ini` restricts discovery to `tests/`. Name files `test_*.py` and functions `test_*`. Add focused tests for environment dynamics, router handoffs, and MJX behavior when those surfaces change.

## Commit & Pull Request Guidelines

Recent history mixes Conventional Commit prefixes (`feat:`, `fix:`, `docs:`) with short `push` commits; prefer the descriptive prefix style, for example `fix: resolve checkpoint path handling`. Pull requests should describe the training or runtime behavior changed, list commands run, link related issues or experiment notes, and include screenshots or generated plots when visual comparisons change.

## Security & Configuration Tips

Keep local secrets out of git. Store real tokens only in local ignored files. Treat model paths and run directories as machine-specific configuration.
