# MuJoCo Rocket Landing Terrain

Free-joint rocket landing environment built with MuJoCo and Gymnasium. The rocket learns to land on a small green tile-pad placed inside randomly generated terrain.

## Included Files

- `rocket_env_tilepad.py`: Gymnasium environment and standalone empty MuJoCo viewer.
- `rocket_rl_tilepad.py`: PPO train/watch/drop-test runner.
- `ppo_freejoint_landing_terrain.zip`: included pretrained terrain landing model.
- `requirements.txt`: Python dependencies.

## Setup

Python 3.10-3.12 is recommended for the smoothest MuJoCo / Stable-Baselines3 install.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run Without RL

This opens the environment with no PPO model and no control forces:

```bash
python rocket_env_tilepad.py
```

## Watch The Model

```bash
python rocket_rl_tilepad.py --mode watch
```

`rocket_rl_tilepad.py` looks for models in this order:

1. `ppo_freejoint_landing_terrain_tilepad_split_smallzone.zip`
2. `ppo_freejoint_landing_terrain.zip`
3. `ppo_freejoint_landing.zip`
4. `ppo_freejoint_hover.zip`

## Train

```bash
python rocket_rl_tilepad.py --mode train --timesteps 300000
```

Training saves the tile-pad model as:

```text
ppo_freejoint_landing_terrain_tilepad_split_smallzone.zip
```

## Drop Test

Use this for simple foot / terrain contact debugging without RL:

```bash
python rocket_rl_tilepad.py --mode drop_test
```
