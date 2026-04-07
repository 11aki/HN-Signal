"""
Predictor — scores new stories using the production model.

Runs every hour (5 min after the collector so new stories are already in the DB).
Loads the model from /models/production.pkl and writes a predicted_prob (0.0–1.0)
to the predictions table for every story that hasn't been scored yet.

If no model file exists (day 1, before first training run), falls back to a
simple heuristic so the Telegram bot has something to show from the start.

Key design rule: features are built from the frozen stories row only —
we never re-fetch from the HN API here. This prevents data leakage
(you can't use a story's future score to predict its future score).
"""
import logging
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shared.db import get_conn
from shared.features import build_feature_dict, feature_names

# Where to look for the production model file on disk.
# MODEL_DIR is set in docker-compose.yml and maps to a named Docker volume.
MODEL_DIR = Path(os.environ.get("MODEL_DIR", "/models"))
PRODUCTION_MODEL = MODEL_DIR / "production.pkl"

# Version label used in the predictions table when no real model exists yet
BASELINE_VERSION = "heuristic-v0"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model():
    """
    Load whichever model is available and return a (predict_fn, version) pair.

    predict_fn signature: (rows: list[dict]) -> np.ndarray of shape (N,)
    Each value in the array is a probability between 0.0 and 1.0.
    """
    if PRODUCTION_MODEL.exists():
        # Lazy import — sentence-transformers and torch are large libraries.
        # Only import them when a real model actually exists, so the container
        # starts fast on day 1 without needing those packages installed.
        from shared.model import load_artifact
        model, version = load_artifact(PRODUCTION_MODEL)
        log.info("Loaded production model %s", version)

        def predict_fn(rows):
            # Build the tabular feature matrix from stored DB rows
            names = feature_names()
            X_tab = np.array(
                [[build_feature_dict(r).get(n, 0) for n in names] for r in rows],
                dtype=float,
            )
            titles = [r["title"] for r in rows]
            # Model combines tabular features + title embeddings internally
            return model.predict_proba(X_tab, titles)

        return predict_fn, version
    else:
        log.warning("No production model found at %s — using heuristic baseline", PRODUCTION_MODEL)
        return _heuristic_predict, BASELINE_VERSION


def _heuristic_predict(rows: list[dict]) -> np.ndarray:
    """
    Simple rule-based fallback used before the first real model is trained.

    Logic: stories with more points and comments at capture time are more
    likely to keep climbing. We apply a sigmoid so output stays in [0, 1].
    This is not a trained model — it's just a reasonable starting point.
    """
    scores = []
    for r in rows:
        s = (r.get("score_at_capture") or 0) * 0.05      # weight for score
        c = (r.get("comments_at_capture") or 0) * 0.02   # weight for comments
        scores.append(s + c)
    arr = np.array(scores, dtype=float)
    # Sigmoid squashes any value to (0, 1): σ(x) = 1 / (1 + e^-x)
    return 1.0 / (1.0 + np.exp(-arr))


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

# Find stories that haven't been scored by the current model version yet.
# This means: if you deploy a new model, it will re-score all existing stories.
# LIMIT 1000 prevents one giant batch from blocking the DB too long.
UNPREDICTED_STORIES = """
SELECT s.*
FROM stories s
WHERE NOT EXISTS (
    SELECT 1 FROM predictions p
    WHERE p.hn_id = s.hn_id
      AND p.model_version = %(version)s
)
ORDER BY s.captured_at DESC
LIMIT 1000
"""

INSERT_PREDICTION = """
INSERT INTO predictions (hn_id, model_version, predicted_prob)
VALUES (%(hn_id)s, %(model_version)s, %(predicted_prob)s)
ON CONFLICT DO NOTHING
"""


def fetch_unpredicted(conn, version: str) -> list[dict]:
    """Return stories that don't yet have a prediction from the given model version."""
    with conn.cursor() as cur:
        cur.execute(UNPREDICTED_STORIES, {"version": version})
        return list(cur.fetchall())


def write_predictions(conn, rows: list[dict], probs: np.ndarray, version: str):
    """Write one prediction row per story into the predictions table."""
    with conn.cursor() as cur:
        for row, prob in zip(rows, probs):
            cur.execute(INSERT_PREDICTION, {
                "hn_id": row["hn_id"],
                "model_version": version,
                "predicted_prob": float(prob),
            })
    conn.commit()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run():
    log.info("Predictor starting")

    # Load model (real or heuristic fallback)
    predict_fn, version = load_model()

    conn = get_conn()
    try:
        # Find stories not yet scored by this model version
        rows = fetch_unpredicted(conn, version)
        if not rows:
            log.info("No new stories to score")
            return

        log.info("Scoring %d stories with model %s", len(rows), version)
        # Run all stories through the model in one batch
        probs = predict_fn(rows)
        write_predictions(conn, rows, probs, version)
        log.info("Done — wrote %d predictions", len(rows))
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    run()
