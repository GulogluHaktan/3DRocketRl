# MuJoCo Hopper Rocket RL

MuJoCo + Gymnasium environments for a TVC hopper rocket. The active workflow is
split into takeoff, hover, and landing phases so each task can be trained and
edited separately.

## Included Files

- `hopper_default.xml`: Real hopper MuJoCo model.
- `hopper_env.py`: shared physics, observations, contacts, camera, and reward helpers.
- `hopper_env_takeoff.py`: takeoff task entrypoint.
- `hopper_env_hover.py`: hover task entrypoint.
- `hopper_env_landing.py`: landing task entrypoint.
- `hopper_rl.py`: PPO train/watch/mission runner for the hopper tasks.
- `archive_unused/`: old experiments and stale models kept for reference.
- `requirements.txt`: Python dependencies.

## Setup

Python 3.10-3.12 is recommended for the smoothest MuJoCo / Stable-Baselines3 install.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The `hopper_*` files use `hopper_default.xml` as the rocket model. The action
space is:

```text
[main_thrust, yaw_tvc, pitch_tvc]
```

Train the three normal flight phases for the real model:

```bash
python hopper_rl.py --mode train_all --timesteps 250000
```

Start each phase from a previous phase model when possible:

```bash
python hopper_rl.py --mode train_all --timesteps 250000 --transfer
```

Training writes CSV logs under `training_logs/ppo_hopper_<task>/`:

```text
progress.csv
episodes.monitor.csv
```

Read the current training status:

```bash
python hopper_rl.py --mode summary --task landing
```

If an old model behaves badly after environment/reward changes, start a fresh
policy:

```bash
python hopper_rl.py --mode train_all --timesteps 250000 --fresh
```

Train phases one by one:

```bash
python hopper_rl.py --mode train --task takeoff --timesteps 250000
python hopper_rl.py --mode train --task hover --timesteps 250000
python hopper_rl.py --mode train --task landing --timesteps 250000
```

Landing is easier to train as a curriculum:

```bash
python hopper_rl.py --mode train --task landing --timesteps 150000 --fresh --landing-start-z 3
python hopper_rl.py --mode train --task landing --timesteps 150000 --landing-start-z 6
python hopper_rl.py --mode train --task landing --timesteps 250000 --landing-start-z 10
```

Watch a single model:

```bash
python hopper_rl.py --mode watch --task takeoff
python hopper_rl.py --mode watch --task hover
python hopper_rl.py --mode watch --task landing
```

Run the chained mission after takeoff, hover, and landing models exist:

```bash
python hopper_rl.py --mode mission
```

Run a no-RL smoke test:

```bash
python hopper_rl.py --mode smoke --task takeoff
```
