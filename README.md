# MuJoCo SAC Hopper Rocket

MuJoCo + Gymnasium environment for a TVC hopper rocket trained with SAC. The
current task uses a phase-based reward system:

```text
climb -> flip -> recovery -> hover -> done
```

## Files

- `hopper_default.xml`: MuJoCo rocket/hopper model.
- `hopper_env.py`: Gymnasium environment, observations, physics controls, reward, fail logic, and viewer camera follow.
- `rl.py`: SAC train/watch/plot entrypoint.
- `visualize.py`: Lightweight viewer for a trained SAC model, random actions, or zero actions.
- `RL_RewardFunction.pdf`: Original reward draft.
- `RL_RewardFunction_updated.docx`: Updated reward document matching the current environment.
- `requirements.txt`: Python dependencies.

Training outputs are written under `runs/` and are ignored by Git.

## Setup

Python 3.10-3.12 is recommended for MuJoCo and Stable-Baselines3.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train

```bash
python rl.py train --timesteps 250000 --chunk-steps 25000
```

The trainer saves:

```text
runs/sac_hopper_<timestamp>/steps_chunk_*.csv
runs/sac_hopper_<timestamp>/checkpoints/sac_hopper_*.zip
runs/sac_hopper_<timestamp>/sac_hopper_latest.zip
sac_hopper_latest.zip
```

## Watch

Watch the latest root model:

```bash
python rl.py watch --model sac_hopper_latest.zip
```

Watch a specific run:

```bash
python rl.py watch --model runs/sac_hopper_<timestamp>/sac_hopper_latest.zip
```

Start from a fixed height:

```bash
python rl.py watch --model sac_hopper_latest.zip --fixed-start-z --start-z 9.3
```

## Visualize

```bash
python visualize.py --model sac_hopper_latest.zip
python visualize.py --random
python visualize.py --zero
```

## Notes

- Action space: `[main_thrust, yaw_tvc, pitch_tvc]`.
- Max thrust defaults to `3.6 kgf`.
- Max TVC angle defaults to `20 deg`.
- TVC servo speed is limited to `60 deg / 0.14 s`.
- Viewer camera follows the rocket body.
