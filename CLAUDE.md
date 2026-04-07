# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

HN Signal is an automated ML pipeline that predicts which Hacker News stories will hit the front page. It collects from `/newstories` (not `/topstories`) to avoid survivorship bias, trains a sentence-transformer + LogisticRegression model, and delivers predictions via a Telegram bot running on EC2.

## Running the System

**Start all services (EC2 / production-like):**
```bash
cp .env.example .env   # fill in POSTGRES_PASSWORD, TELEGRAM_TOKEN, TELEGRAM_CHAT_ID
docker-compose up -d
docker-compose logs -f <service>   # collector | snapshotter | predictor | bot | monitor
```

**Train a model (local GPU machine, requires SSH tunnel to EC2 Postgres):**
```bash
ssh -L 5432:localhost:5432 ubuntu@<ec2-host>   # tunnel in a separate terminal

python -m venv venv && source venv/bin/activate
pip install -r trainer/requirements.txt

# Train, evaluate against eval gate, and scp to EC2 if it passes:
python trainer/train.py \
  --min-precision 0.35 --min-recall 0.35 \
  --C 1.0 --threshold 0.5 --test-size 0.2 \
  --ec2 ubuntu@<ec2-host>:/models/production.pkl

# Train + MLflow logging only (no deploy):
python trainer/train.py --min-precision 0.35 --min-recall 0.35
```

**Run a single service manually:**
```bash
python collector/collect.py
python snapshotter/snapshot.py
python predictor/predict.py
python monitor/monitor.py
```

## Architecture

The pipeline has six services with strict separation:

| Service | Where | Schedule | Role |
|---------|-------|----------|------|
| **collector** | EC2 | hourly :00 | Fetches `/newstories`, freezes features at capture |
| **snapshotter** | EC2 | hourly :00 | Re-fetches story scores at 1h/6h/12h/24h after capture |
| **predictor** | EC2 | hourly :05 | Scores unpredicted stories using `production.pkl` |
| **monitor** | EC2 | daily 09:00 UTC | Computes precision/recall, sends Telegram alert if degraded |
| **bot** | EC2 | always-on | Responds to `/top`, `/status`, `/help` in Telegram |
| **trainer** | local GPU | on-demand | Trains model, evaluates against eval gate, scp to EC2 |

Data flow: `HN API → stories table → snapshots table → predictions table → Telegram bot`

### Shared Modules (used by all services)

- **`shared/db.py`** — `get_conn()` using env vars, returns `RealDictCursor` connections
- **`shared/features.py`** — single source of truth for feature engineering; `build_feature_dict(story)` and `feature_names()` must stay in sync between trainer and predictor
- **`shared/model.py`** — `HNModel` class: sentence-transformer (all-MiniLM-L6-v2, 384-dim) + tabular features → StandardScaler → LogisticRegression; `save_artifact` / `load_artifact` for pickling

### Critical Design Constraints

- **Features are frozen at capture time.** The predictor reads only from the `stories` table — it never re-fetches live HN data. Changing what the collector captures requires a matching change to `shared/features.py`.
- **No data leakage.** The trainer uses only features available at the moment a story was first seen. Do not add features derived from post-capture scores.
- **Eval gate before deployment.** `trainer/train.py` compares new model metrics against the current `production.pkl` on a holdout set; the model is only deployed if it wins.
- **Snapshots are the labels.** The trainer joins `stories` with `snapshots WHERE age_hours = 24` to determine ground truth (default: `score ≥ 200` = "blew up"). You cannot relabel without re-running the snapshotter.

## Database Schema

Postgres 16. Initialized via `db/init.sql`.

- **`stories`** — one row per HN story; features frozen at capture (`score_at_capture`, `comments_at_capture`, `hour_of_day`, `domain`, `type`, etc.)
- **`snapshots`** — one row per (story, age_hours); `UNIQUE(hn_id, age_hours)` prevents double-inserts; age buckets: 1, 6, 12, 24
- **`predictions`** — one row per (story, model_version); stores `predicted_prob` (0.0–1.0)

All DB writes use `ON CONFLICT DO NOTHING` — every script is safe to re-run.

## Key Environment Variables

```
POSTGRES_USER / POSTGRES_PASSWORD / POSTGRES_DB / POSTGRES_HOST
TELEGRAM_TOKEN / TELEGRAM_CHAT_ID
MIN_PRECISION=0.30          # monitor alert threshold
MIN_RECALL=0.30
BLOWUP_SCORE_THRESHOLD=200  # 24h score to count as "blew up"
PROB_THRESHOLD=0.5          # prediction cutoff for precision/recall
EVAL_WINDOW_DAYS=7
```

## Scheduling

Each service's `crontab` file is read by **supercronic** inside the container. The format is standard 5-field cron. Collector and snapshotter run at :00; predictor runs at :05 to ensure new stories are already collected.
