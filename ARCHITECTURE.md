# HN Predictor — Architecture

## What this is
An automated ML system that predicts which Hacker News stories will hit the front page, retrains on real outcomes, and delivers early picks via Telegram.

## Infrastructure
- **EC2 t3.small** — always on, runs all services via Docker Compose (~$15/month)
- **Your PC (GPU)** — on-demand training, SSH tunnel into EC2 Postgres

## Components

### 1. Collector (EC2, cron every 15 min)
- Hits HN `/newstories` endpoint (ALL submissions, not just top stories — avoids survivorship bias)
- Writes each story to Postgres with features frozen at capture time
- Features captured: title, url, domain, author (by), score, descendants (comments), type, time posted, hn_id

### 2. Snapshotter (EC2, scheduled)
- Checks back on stories at 1h, 6h, 12h, 24h after capture
- Records each snapshot as its own row (story_id, snapshot_age, score, descendants)
- This gives flexible labeling — "blew up" can be defined as 200+ points at 24h, top 100 at 6h, etc.
- No need to re-collect data when changing label definition

### 3. Predictor (EC2, runs after each Collector cycle)
- Loads current production model
- Scores new stories using ONLY frozen capture-time features from Postgres (never re-fetches from API — prevents data leakage)
- Writes predictions back to Postgres (story_id, model_version, predicted_prob)

### 4. Monitor (EC2, daily cron)
- Compares predictions against actual snapshot outcomes
- Tracks precision and recall separately (not just accuracy — catches "predict no for everything" failure)
- Sends Telegram alert when performance degrades past threshold

### 5. Telegram Bot (EC2, always on)
- Shows top predicted stories on /top command
- No personal re-ranking, no swipe buttons — just the model's picks
- Also receives monitor alerts

### 6. Trainer (Your PC, on demand)
- SSH tunnel into EC2 Postgres, pulls labeled data
- Trains locally with GPU
- MLflow tracks experiments
- Eval gate: new model must beat current production metrics on holdout set
- Only then scp model artifact to EC2
- Start with: frozen sentence-transformer embeddings + LogisticRegression
- Later: fine-tune the transformer, add content features, try XGBoost/neural nets

## Database Schema (Postgres)

### stories
```sql
CREATE TABLE stories (
    hn_id       INTEGER PRIMARY KEY,
    title       TEXT NOT NULL,
    url         TEXT,
    domain      TEXT,              -- extracted from url
    author      TEXT,
    type        TEXT,              -- story, show_hn, ask_hn, job
    score_at_capture  INTEGER,
    comments_at_capture INTEGER,
    time_posted TIMESTAMP,         -- HN's unix timestamp converted
    captured_at TIMESTAMP DEFAULT NOW()
);
```

### snapshots
```sql
CREATE TABLE snapshots (
    id          SERIAL PRIMARY KEY,
    hn_id       INTEGER REFERENCES stories(hn_id),
    age_hours   INTEGER NOT NULL,  -- 1, 6, 12, or 24
    score       INTEGER,
    descendants INTEGER,
    captured_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(hn_id, age_hours)
);
```

### predictions
```sql
CREATE TABLE predictions (
    id              SERIAL PRIMARY KEY,
    hn_id           INTEGER REFERENCES stories(hn_id),
    model_version   TEXT,
    predicted_prob  FLOAT,
    predicted_at    TIMESTAMP DEFAULT NOW()
);
```

## Key Design Decisions
1. Collect from /newstories not /topstories — full distribution, no survivorship bias
2. Multi-snapshot labeling — flexible target definition without re-collecting
3. Features frozen at capture time — prevents data leakage in prediction
4. Track precision/recall not just accuracy — catches degenerate models
5. Eval gate before model push — prevents shipping bad models
6. Train local, serve remote — cheap always-on serving, free GPU training
7. Postgres shared_buffers capped at 256MB — leaves room for other containers on 2GB RAM

## Build Order
1. Docker Compose + Postgres schema
2. Collector
3. Snapshotter
4. Predictor (needs a baseline model — start with random or simple heuristic)
5. Telegram Bot
6. Monitor
7. Trainer (on PC)

## Tech Stack
- Python 3.11+
- Postgres 16
- Docker + Docker Compose
- sentence-transformers (all-MiniLM-L6-v2)
- scikit-learn (LogisticRegression to start)
- MLflow (experiment tracking)
- python-telegram-bot
- cron (via system crontab or supercrond in Docker)
- psycopg2 or asyncpg for Postgres access
