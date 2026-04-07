"""
Monitor — daily accuracy check for the production model.

Label definition: a story "blew up" if its 24h snapshot score >= 200.
Computes precision and recall (not just accuracy) to catch degenerate models.
Sends a Telegram alert if either metric drops below threshold.

Runs daily at 09:00 UTC.
"""
import logging
import os
import sys
from dataclasses import dataclass

import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shared.db import get_conn

TELEGRAM_TOKEN   = os.environ["TELEGRAM_TOKEN"]
TELEGRAM_CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]

# Alert thresholds — tune after first few weeks of data
MIN_PRECISION = float(os.environ.get("MIN_PRECISION", "0.30"))
MIN_RECALL    = float(os.environ.get("MIN_RECALL",    "0.30"))

# Label: blew up if 24h score >= this
BLOWUP_SCORE_THRESHOLD = int(os.environ.get("BLOWUP_SCORE_THRESHOLD", "200"))

# Classification threshold on predicted_prob
PROB_THRESHOLD = float(os.environ.get("PROB_THRESHOLD", "0.5"))

# Evaluate stories captured in the last N days (must be >=1 so 24h snapshots exist).
# Default 7 gives a week of data — enough for stable precision/recall estimates.
EVAL_WINDOW_DAYS = int(os.environ.get("EVAL_WINDOW_DAYS", "7"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@dataclass
class Metrics:
    model_version: str
    n_evaluated: int
    n_positive: int
    precision: float
    recall: float
    f1: float

    def summary(self) -> str:
        return (
            f"Model: {self.model_version}\n"
            f"Evaluated: {self.n_evaluated} stories "
            f"({self.n_positive} positives)\n"
            f"Precision: {self.precision:.1%}  "
            f"Recall: {self.recall:.1%}  "
            f"F1: {self.f1:.1%}"
        )


def compute_metrics(conn) -> Metrics | None:
    """
    Joins predictions with 24h snapshots for stories captured in the last
    EVAL_WINDOW_DAYS days (but at least 24h ago so snapshots exist).
    """
    sql = """
    SELECT
        p.model_version,
        p.predicted_prob,
        sn.score AS score_24h
    FROM predictions p
    JOIN stories s ON s.hn_id = p.hn_id
    JOIN snapshots sn ON sn.hn_id = p.hn_id AND sn.age_hours = 24
    WHERE s.captured_at BETWEEN
          NOW() - INTERVAL '%(window)s days' - INTERVAL '1 day'
          AND NOW() - INTERVAL '1 day'
    ORDER BY p.predicted_at DESC
    """
    with conn.cursor() as cur:
        cur.execute(sql, {"window": EVAL_WINDOW_DAYS})
        rows = cur.fetchall()

    if not rows:
        log.warning("No labeled data available for evaluation")
        return None

    # Use the most recent model version if multiple exist
    version = rows[0]["model_version"]
    rows = [r for r in rows if r["model_version"] == version]

    tp = fp = fn = tn = 0
    for r in rows:
        predicted_pos = r["predicted_prob"] >= PROB_THRESHOLD
        actual_pos    = (r["score_24h"] or 0) >= BLOWUP_SCORE_THRESHOLD
        if predicted_pos and actual_pos:
            tp += 1
        elif predicted_pos and not actual_pos:
            fp += 1
        elif not predicted_pos and actual_pos:
            fn += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    return Metrics(
        model_version=version,
        n_evaluated=len(rows),
        n_positive=tp + fn,
        precision=precision,
        recall=recall,
        f1=f1,
    )


# ---------------------------------------------------------------------------
# Alerting
# ---------------------------------------------------------------------------

def send_telegram(message: str):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    resp = requests.post(url, json={
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "Markdown",
    }, timeout=10)
    resp.raise_for_status()


def alert_degraded(metrics: Metrics):
    issues = []
    if metrics.precision < MIN_PRECISION:
        issues.append(
            f"Precision {metrics.precision:.1%} < threshold {MIN_PRECISION:.1%}"
        )
    if metrics.recall < MIN_RECALL:
        issues.append(
            f"Recall {metrics.recall:.1%} < threshold {MIN_RECALL:.1%}"
        )
    if not issues:
        return

    msg = (
        "*HN Signal — Model Alert*\n\n"
        + "\n".join(f"- {i}" for i in issues)
        + "\n\n"
        + metrics.summary()
    )
    log.warning("Sending degradation alert: %s", issues)
    send_telegram(msg)


def send_daily_report(metrics: Metrics):
    msg = "*HN Signal — Daily Report*\n\n" + metrics.summary()
    send_telegram(msg)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run():
    log.info("Monitor starting")
    conn = get_conn()
    try:
        metrics = compute_metrics(conn)
        if metrics is None:
            log.info("Nothing to evaluate yet")
            return

        log.info(metrics.summary())
        send_daily_report(metrics)
        alert_degraded(metrics)
        log.info("Monitor done")
    finally:
        conn.close()


if __name__ == "__main__":
    run()
