# MuJoCo Rocket Landing Terrain

Free-joint rocket landing environment built with MuJoCo and Gymnasium. The rocket learns to land on a small green tile-pad placed inside randomly generated terrain.

## Included Files

- `rocket_env_tilepad.py`: Gymnasium environment and standalone empty MuJoCo viewer.
- `rocket_rl_tilepad.py`: PPO train/watch/drop-test runner.
- `rocket_env_takeoff.py`: Separate takeoff environment.
- `rocket_rl_takeoff.py`: PPO train/watch runner for takeoff.
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
python rocket_rl_tilepad.py --mode landing
```

`rocket_rl_tilepad.py` looks for models in this order:

1. `ppo_freejoint_landing_terrain_tilepad_split_smallzone.zip`
2. `ppo_freejoint_landing_terrain.zip`
3. `ppo_freejoint_landing.zip`
4. `ppo_freejoint_hover.zip`

## Mission Mode

Run the terrain mission with three policies chained together:

1. takeoff model
2. hover model for 5 seconds
3. landing model

```bash
python rocket_rl_tilepad.py --mode normal
```

Run each phase by itself inside the terrain environment:

```bash
python rocket_rl_tilepad.py --mode takeoff
python rocket_rl_tilepad.py --mode hover
python rocket_rl_tilepad.py --mode flip
python rocket_rl_tilepad.py --mode landing
```

Run the stunt mission: takeoff, 360 flip, then landing:

```bash
python rocket_rl_tilepad.py --mode stunt
```

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

## Takeoff Model

Train the separate takeoff model:

```bash
python rocket_rl_takeoff.py --mode train --timesteps 300000
```

Watch the takeoff policy:

```bash
python rocket_rl_takeoff.py --mode watch
```

Run a fixed-thrust smoke test without RL:

```bash
python rocket_rl_takeoff.py --mode thrust_test
```

## Flip Model

Train the separate 360-roll model:

```bash
python rocket_rl_flip.py --mode train --timesteps 300000
```

Watch the flip policy:

```bash
python rocket_rl_flip.py --mode watch
```

Run a fixed right-motor smoke test:

```bash
python rocket_rl_flip.py --mode right_motor_test
```
