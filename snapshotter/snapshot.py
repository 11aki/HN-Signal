"""
Snapshotter — checks back on stories at 1h, 6h, 12h, 24h after capture.

Why take multiple snapshots?
  A story's score at 1h is a very different signal than its score at 24h.
  By recording both, we can later define "blew up" however we want
  (e.g. 200+ points at 24h, or top 10 at 6h) without re-collecting data.

How it works:
  On each run, for each age bucket (1h, 6h, 12h, 24h), we find stories
  whose capture time was approximately that many hours ago and record
  their current score as a snapshot — but only if we haven't already
  snapshotted them at that age.

Runs every hour. Tolerance window is ±30 min so no stories fall through the gap.
"""
import logging
import os
import sys
import time

import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shared.db import get_conn

HN_API = "https://hacker-news.firebaseio.com/v0"
REQUEST_TIMEOUT = 10
RETRY_DELAY = 2

# Age buckets and their tolerance windows.
# Format: (age_in_hours, tolerance_in_minutes)
#
# Running hourly: a ±30 min window guarantees we always catch each story
# in exactly one run per bucket (windows of consecutive runs are adjacent
# and non-overlapping). The UNIQUE(hn_id, age_hours) constraint in the DB
# prevents any double-inserts at the boundary.
SNAPSHOT_AGES = [
    (1,  30),   # 1h  ± 30 min  → catches stories captured 30-90 min ago
    (6,  30),   # 6h  ± 30 min  → catches stories captured 5.5-6.5h ago
    (12, 30),   # 12h ± 30 min  → catches stories captured 11.5-12.5h ago
    (24, 30),   # 24h ± 30 min  → catches stories captured 23.5-24.5h ago
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# HN API
# ---------------------------------------------------------------------------

def fetch_item(hn_id: int) -> dict | None:
    """Fetch current state of a story from HN. Retries up to 3 times."""
    for attempt in range(3):
        try:
            resp = requests.get(f"{HN_API}/item/{hn_id}.json", timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            if attempt == 2:
                log.warning("Failed to fetch item %d: %s", hn_id, exc)
                return None
            time.sleep(RETRY_DELAY)
    return None


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

# Find stories that are due for a snapshot at a given age bucket.
# A story is due if:
#   - It was captured approximately `age` hours ago (within ±tolerance minutes)
#   - It doesn't already have a snapshot row for this age bucket
STORIES_DUE = """
SELECT hn_id
FROM stories
WHERE captured_at BETWEEN NOW() - %(age)s * INTERVAL '1 hour' - %(tol)s * INTERVAL '1 minute'
                      AND NOW() - %(age)s * INTERVAL '1 hour' + %(tol)s * INTERVAL '1 minute'
  AND hn_id NOT IN (
      SELECT hn_id FROM snapshots WHERE age_hours = %(age)s
  )
"""

INSERT_SNAPSHOT = """
INSERT INTO snapshots (hn_id, age_hours, score, descendants)
VALUES (%(hn_id)s, %(age_hours)s, %(score)s, %(descendants)s)
ON CONFLICT (hn_id, age_hours) DO NOTHING
"""


def stories_due(conn, age_hours: int, tolerance_minutes: int) -> list[int]:
    """Return IDs of stories that need a snapshot taken at the given age bucket."""
    with conn.cursor() as cur:
        cur.execute(STORIES_DUE, {"age": age_hours, "tol": tolerance_minutes})
        return [row["hn_id"] for row in cur.fetchall()]


def insert_snapshot(conn, hn_id: int, age_hours: int, item: dict) -> bool:
    """
    Record the current score/comments for a story at a given age.
    Returns True if a new row was inserted, False if it already existed.
    If the item fetch failed (item is None), score/comments are stored as NULL.
    """
    with conn.cursor() as cur:
        cur.execute(INSERT_SNAPSHOT, {
            "hn_id": hn_id,
            "age_hours": age_hours,
            "score": item.get("score", 0) if item else None,
            "descendants": item.get("descendants", 0) if item else None,
        })
        return cur.rowcount == 1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run():
    log.info("Snapshotter starting")
    conn = get_conn()
    try:
        total = 0
        # Process each age bucket in order (1h, 6h, 12h, 24h)
        for age_hours, tolerance_minutes in SNAPSHOT_AGES:
            due = stories_due(conn, age_hours, tolerance_minutes)
            if not due:
                continue  # nothing due for this bucket right now
            log.info("Age %dh: %d stories due for snapshot", age_hours, len(due))
            for hn_id in due:
                # Re-fetch the story from HN to get its current score
                item = fetch_item(hn_id)
                inserted = insert_snapshot(conn, hn_id, age_hours, item)
                if inserted:
                    total += 1
                # Commit after each story so partial progress is saved on crash
                conn.commit()

        log.info("Done — %d snapshots written", total)
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    run()
