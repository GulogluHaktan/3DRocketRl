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
Root-level generated watch CSV files and model zip files are also ignored.

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

Flip-focused SAC training from a fixed start height:

```bash
python rl.py train --algo sac --timesteps 500000 --chunk-steps 25000 --start-phase flip --fixed-start-z --start-z 11 --max-thrust 45
```

### Specialist Gece Eğitimi (`train-specialists`)

Tüm specialist aşamalarını (`climb → flip → recovery → hover`) sırayla ve her aşama için optimize edilmiş başlangıç yükseklikleriyle tek komutla çalıştırmak için:

```bash
.venv/bin/python rl.py train-specialists \
  --algo sac \
  --timesteps-per-phase 500000 \
  --chunk-steps 25000 \
  --max-thrust 45 \
  --telegram-every 100000 \
  --telegram-video-every 100000
```

**Varsayılan Aşamalar ve Yükseklik Başlangıçları:**
- `climb`: `--start-z 2`
- `flip`: `--start-z 11`
- `recovery`: `--start-z 9`
- `hover`: `--start-z 5`
- Tüm aşamalarda `--fixed-start-z` ve `--max-thrust 45` otomatik olarak uygulanır.

**Opsiyonel Parametreler:**
- `--phases`: Eğitilecek aşamaları seçmek için (Örn: `--phases climb recovery hover`). Varsayılan: `climb flip recovery hover`.
- `--no-telegram-video`: Telegram video gönderimini devre dışı bırakır.
- `--tvc-servo-sec-per-60deg 0.13`: Gerçek servo tepki hızını saniye cinsinden yapılandırır.

**Yedekleme Güvencesi:**
- Flip eğitimi başladığında, root dizindeki mevcut `sac_flip_latest.zip` (Golden Model) ezilmez; otomatik olarak `sac_flip_latest_golden.zip` adı altında yedeklenir.

Telegram notifications are optional. Copy the example file, fill it locally,
and do not commit the real secret file:

```bash
cp telegram_secrets.example.json telegram_secrets.json
```

You can also provide credentials through `TELEGRAM_BOT_TOKEN` and
`TELEGRAM_CHAT_ID`. The default notification interval is 10000 training steps;
override it with `--telegram-every`. Her 100k stepte bir aktif uzman checkpoint'inden üretilen headless watch videosu Telegram'a otomatik gönderilir.

The trainer saves:

```text
runs/<algo>_hopper_<timestamp>/steps_chunk_*.csv
runs/<algo>_hopper_<timestamp>/checkpoints/<algo>_hopper_*.zip
runs/<algo>_hopper_<timestamp>/<algo>_hopper_latest.zip
runs/<algo>_hopper_<timestamp>/videos/watch_*.mp4
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

Watch a flip-focused SAC checkpoint:

```bash
python rl.py watch --algo sac --model runs/sac_hopper_<timestamp>/checkpoints/sac_hopper_500000.zip --start-phase flip --fixed-start-z --start-z 11 --max-thrust 45 --csv watch.csv
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
- TVC servo hızı `--tvc-servo-sec-per-60deg` parametresine bağlıdır (Varsayılan: `0.13` saniyede `60°` dönme hızı, yani yaklaşık `461.5 deg/s`).
- Viewer camera follows the rocket body.
