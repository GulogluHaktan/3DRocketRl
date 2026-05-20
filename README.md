# MuJoCo Hopper Rocket

MuJoCo + Gymnasium environment for a TVC hopper rocket. The same environment can
be trained with SAC, PPO, or TD3 through the shared `rl.py` entrypoint. The
goal is to teach the rocket to climb to a safe altitude, perform a controlled
360 degree flip, recover its attitude, and finish by hovering near the target
position. The current task uses a phase-based reward system:

```text
climb -> flip -> recovery -> hover -> done
```

## Files

- `hopper_default.xml`: MuJoCo rocket/hopper model.
- `hopper_env.py`: Gymnasium environment, observations, physics controls, reward, fail logic, and viewer camera follow.
- `rl.py`: train/watch/plot entrypoint and algorithm router.
- `rl_common.py`: shared training loop, CSV logging, model watching, and plot generation.
- `sac.py`: SAC-specific Stable-Baselines3 setup.
- `ppo.py`: PPO-specific Stable-Baselines3 setup.
- `td3.py`: TD3-specific Stable-Baselines3 setup.
- `visualize.py`: Lightweight viewer for a trained model, random actions, or zero actions.
- `RL_RewardFunction.pdf`: Original reward draft.
- `RL_RewardFunction_updated.docx`: Updated reward document matching the current environment.
- `requirements.txt`: Python dependencies.

Training outputs are written under `runs/` and are ignored by Git.

Each algorithm file has a `REWARD_WEIGHTS` dictionary. Leave it empty to use
the shared defaults from `hopper_env.py`, or override only the reward weights
you want to test for that algorithm.

## Setup

Python 3.10-3.12 is recommended for MuJoCo and Stable-Baselines3.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train

```bash
python rl.py train --algo sac --timesteps 250000 --chunk-steps 25000
python rl.py train --algo ppo --timesteps 250000 --chunk-steps 25000
python rl.py train --algo td3 --timesteps 250000 --chunk-steps 25000
```

The trainer saves:

```text
runs/<algo>_hopper_<timestamp>/steps_chunk_*.csv
runs/<algo>_hopper_<timestamp>/checkpoints/<algo>_hopper_*.zip
runs/<algo>_hopper_<timestamp>/<algo>_hopper_latest.zip
<algo>_hopper_latest.zip
```

## Watch

Watch the latest root model:

```bash
python rl.py watch --algo sac
python rl.py watch --algo ppo
python rl.py watch --algo td3
```

Watch a specific run:

```bash
python rl.py watch --algo sac --model runs/sac_hopper_<timestamp>/sac_hopper_latest.zip
```

Start from a fixed height:

```bash
python rl.py watch --algo sac --fixed-start-z --start-z 9.3
```

## Visualize

```bash
python visualize.py --algo sac
python visualize.py --algo ppo
python visualize.py --algo td3
python visualize.py --random
python visualize.py --zero
```

## Notes

- Action space: `[main_thrust, yaw_tvc, pitch_tvc]`.
- Max thrust defaults to `3.6 kgf`.
- Max TVC angle defaults to `20 deg`.
- TVC servo speed is limited to `60 deg / 0.14 s`.
- Viewer camera follows the rocket body.
