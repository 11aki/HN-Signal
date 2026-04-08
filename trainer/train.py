"""
Trainer — runs on EC2 as a Docker service (daily 03:00 UTC via supercronic).

Pipeline:
  1. Pulls labeled data from Postgres
  2. Builds features (same shared.features module used in production)
  3. Trains sentence-transformer embeddings + LogisticRegression pipeline
  4. Evaluates on holdout set against precision/recall gates
  5. If gate passes: saves model to /models/production.pkl and sends Telegram notification
"""
import argparse
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import requests
from sklearn.metrics import classification_report, precision_score, recall_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shared.db import get_conn
from shared.features import build_feature_dict, feature_names
from shared.model import HNModel, save_artifact

BLOWUP_SCORE_THRESHOLD = int(os.environ.get("BLOWUP_SCORE_THRESHOLD", "200"))
SENTENCE_MODEL = "all-MiniLM-L6-v2"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

LABELED_DATA_SQL = """
SELECT
    s.hn_id,
    s.title,
    s.url,
    s.domain,
    s.author,
    s.type,
    s.score_at_capture,
    s.comments_at_capture,
    s.time_posted,
    s.captured_at,
    sn.score AS score_24h
FROM stories s
JOIN snapshots sn ON sn.hn_id = s.hn_id AND sn.age_hours = 24
ORDER BY s.captured_at DESC
"""


def load_labeled_data() -> list[dict]:
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(LABELED_DATA_SQL)
            rows = list(cur.fetchall())
        log.info("Loaded %d labeled stories", len(rows))
        return rows
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Feature building
# ---------------------------------------------------------------------------

def build_dataset(rows: list[dict]):
    """Returns (X_tabular, X_titles, y)."""
    names = feature_names()
    X_tab = np.array(
        [[build_feature_dict(r).get(n, 0) for n in names] for r in rows],
        dtype=float,
    )
    titles = [r["title"] for r in rows]
    y = np.array(
        [(r["score_24h"] or 0) >= BLOWUP_SCORE_THRESHOLD for r in rows],
        dtype=int,
    )
    return X_tab, titles, y


# ---------------------------------------------------------------------------
# Eval gate
# ---------------------------------------------------------------------------

def evaluate(model: HNModel, X_tab, titles, y, threshold: float = 0.5):
    probs = model.predict_proba(X_tab, titles)
    preds = (probs >= threshold).astype(int)
    precision = precision_score(y, preds, zero_division=0)
    recall    = recall_score(y, preds, zero_division=0)
    log.info("Holdout evaluation:\n%s", classification_report(y, preds, zero_division=0))
    return precision, recall


# ---------------------------------------------------------------------------
# Artifact save / deploy
# ---------------------------------------------------------------------------

def make_version() -> str:
    return datetime.now(timezone.utc).strftime("v%Y%m%d-%H%M%S")


def send_telegram(message: str):
    token = os.environ.get("TELEGRAM_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        log.warning("TELEGRAM_TOKEN/CHAT_ID not set — skipping notification")
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": message, "parse_mode": "Markdown"},
            timeout=10,
        ).raise_for_status()
    except Exception as e:
        log.warning("Telegram notification failed: %s", e)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train HN Signal model")
    p.add_argument("--min-precision", type=float, default=0.30)
    p.add_argument("--min-recall",    type=float, default=0.30)
    p.add_argument("--threshold",     type=float, default=0.5)
    p.add_argument("--C",             type=float, default=1.0)
    p.add_argument("--test-size",     type=float, default=0.2)
    return p.parse_args()


def run():
    args = parse_args()

    rows = load_labeled_data()
    if len(rows) < 100:
        log.error("Only %d labeled examples — need at least 100 to train", len(rows))
        sys.exit(1)

    X_tab, titles, y = build_dataset(rows)
    log.info("Label distribution: %d positive / %d negative", y.sum(), (y == 0).sum())

    idx = np.arange(len(rows))
    tr_idx, te_idx = train_test_split(idx, test_size=args.test_size, stratify=y, random_state=42)
    X_tr, X_te = X_tab[tr_idx], X_tab[te_idx]
    t_tr = [titles[i] for i in tr_idx]
    t_te = [titles[i] for i in te_idx]
    y_tr, y_te = y[tr_idx], y[te_idx]

    model = HNModel(C=args.C)
    model.fit(X_tr, t_tr, y_tr)

    precision, recall = evaluate(model, X_te, t_te, y_te, args.threshold)

    if precision < args.min_precision:
        log.error("Precision %.3f < threshold %.3f — not deploying", precision, args.min_precision)
        send_telegram(
            f"*HN Signal — Training Failed* \u274c\n"
            f"Eval gate not met\n"
            f"Precision: `{precision:.3f}` (min {args.min_precision})\n"
            f"Recall: `{recall:.3f}`\n"
            f"Training stories: {len(rows)}"
        )
        sys.exit(1)
    if recall < args.min_recall:
        log.error("Recall %.3f < threshold %.3f — not deploying", recall, args.min_recall)
        send_telegram(
            f"*HN Signal — Training Failed* \u274c\n"
            f"Eval gate not met\n"
            f"Precision: `{precision:.3f}`\n"
            f"Recall: `{recall:.3f}` (min {args.min_recall})\n"
            f"Training stories: {len(rows)}"
        )
        sys.exit(1)

    log.info("Eval gate passed (precision=%.3f, recall=%.3f)", precision, recall)

    version = make_version()
    artifact_path = Path("/models/production.pkl")
    save_artifact(model, version, artifact_path)

    send_telegram(
        f"*HN Signal — Model Updated* \u2705\n"
        f"Version: `{version}`\n"
        f"Precision: `{precision:.3f}`\n"
        f"Recall: `{recall:.3f}`\n"
        f"Training stories: {len(rows)} ({int(y.sum())} positive)"
    )

    log.info("Training complete — version %s", version)


if __name__ == "__main__":
    run()
