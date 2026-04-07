"""
Collector — scrapes HN /newstories every hour.

Why /newstories and not /topstories?
  /topstories only shows stories already doing well — if we trained on that,
  we'd only see winners and never learn what separates a future winner from
  a story that flopped. /newstories gives us the full distribution.

On each run it fetches the newest 200 story IDs, skips any already in the DB,
fetches full item data for the new ones, and writes them to the stories table.
Features (score, comments, etc.) are frozen at the moment of capture — we never
update them later, which prevents data leakage during model training.
"""
import logging
import os
import sys
import time
from datetime import datetime, timezone
from urllib.parse import urlparse

import requests

# Add parent directory to path so we can import from shared/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shared.db import get_conn

HN_API = "https://hacker-news.firebaseio.com/v0"
MAX_STORIES = 200      # /newstories returns up to 500; we take the newest 200
REQUEST_TIMEOUT = 10   # seconds before giving up on an API call
RETRY_DELAY = 2        # seconds to wait between retries on failure

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# HN API helpers
# ---------------------------------------------------------------------------

def fetch_new_story_ids() -> list[int]:
    """
    Fetch the list of newest story IDs from HN.
    Returns up to MAX_STORIES IDs, newest first.
    """
    resp = requests.get(f"{HN_API}/newstories.json", timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    ids = resp.json()
    return ids[:MAX_STORIES]


def fetch_item(hn_id: int) -> dict | None:
    """
    Fetch a single HN item by ID. Retries up to 3 times on failure.
    Returns None if all attempts fail.
    """
    for attempt in range(3):
        try:
            resp = requests.get(f"{HN_API}/item/{hn_id}.json", timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            if attempt == 2:
                log.warning("Failed to fetch item %d after 3 attempts: %s", hn_id, exc)
                return None
            time.sleep(RETRY_DELAY)
    return None


# ---------------------------------------------------------------------------
# Feature extraction (frozen at capture time)
# ---------------------------------------------------------------------------

def extract_domain(url: str | None) -> str | None:
    """
    Pull the domain from a URL, stripping www.
    e.g. "https://www.github.com/foo" → "github.com"
    Returns None if no URL is given.
    """
    if not url:
        return None
    try:
        hostname = urlparse(url).hostname or ""
        return hostname.removeprefix("www.") or None
    except Exception:
        return None


def classify_type(item: dict) -> str:
    """
    Classify a story into one of four types based on its title prefix or item type.
    HN uses title conventions: "Show HN:" and "Ask HN:" prefix user projects/questions.
    """
    title = (item.get("title") or "").lower()
    item_type = item.get("type", "story")
    if item_type == "job":
        return "job"
    if title.startswith("show hn"):
        return "show_hn"
    if title.startswith("ask hn"):
        return "ask_hn"
    return "story"


def item_to_row(item: dict) -> dict | None:
    """
    Convert a raw HN API item dict into a DB row dict ready for insertion.
    Returns None if the item should be skipped (non-story, deleted, etc.).
    """
    if item is None:
        return None
    # Skip polls, comments — only store story-like items
    if item.get("type") not in ("story", "job"):
        return None
    # Dead or deleted items have no title
    if not item.get("title"):
        return None

    url = item.get("url")
    posted_ts = item.get("time")  # HN gives Unix timestamp in seconds

    return {
        "hn_id": item["id"],
        "title": item["title"],
        "url": url,
        "domain": extract_domain(url),
        "author": item.get("by"),
        "type": classify_type(item),
        # Score and comments frozen at capture time — never updated
        "score_at_capture": item.get("score", 0),
        "comments_at_capture": item.get("descendants", 0),
        # Convert Unix timestamp to Python datetime in UTC
        "time_posted": (
            datetime.fromtimestamp(posted_ts, tz=timezone.utc) if posted_ts else None
        ),
    }


# ---------------------------------------------------------------------------
# Database writes
# ---------------------------------------------------------------------------

INSERT_STORY = """
INSERT INTO stories
    (hn_id, title, url, domain, author, type,
     score_at_capture, comments_at_capture, time_posted)
VALUES
    (%(hn_id)s, %(title)s, %(url)s, %(domain)s, %(author)s, %(type)s,
     %(score_at_capture)s, %(comments_at_capture)s, %(time_posted)s)
ON CONFLICT (hn_id) DO NOTHING
"""
# ON CONFLICT DO NOTHING means: if this story is already in the DB, skip it silently.
# This makes the collector idempotent — safe to run multiple times.


def already_collected(conn, ids: list[int]) -> set[int]:
    """
    Given a list of HN IDs, return the subset already present in the stories table.
    Used to avoid fetching full item data for stories we already have.
    """
    if not ids:
        return set()
    with conn.cursor() as cur:
        cur.execute(
            "SELECT hn_id FROM stories WHERE hn_id = ANY(%s)", (ids,)
        )
        return {row["hn_id"] for row in cur.fetchall()}


def insert_story(conn, row: dict) -> bool:
    """Insert a story row. Returns True if a new row was inserted (False if duplicate)."""
    with conn.cursor() as cur:
        cur.execute(INSERT_STORY, row)
        return cur.rowcount == 1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run():
    log.info("Collector starting")

    # Step 1: get the list of newest story IDs from HN
    ids = fetch_new_story_ids()
    log.info("Fetched %d new story IDs from HN", len(ids))

    conn = get_conn()
    try:
        # Step 2: check which IDs we already have — no need to re-fetch those
        existing = already_collected(conn, ids)
        new_ids = [i for i in ids if i not in existing]
        log.info("%d already in DB, fetching %d new items", len(existing), len(new_ids))

        inserted = 0
        skipped = 0
        for hn_id in new_ids:
            # Step 3: fetch full item data and insert into DB
            item = fetch_item(hn_id)
            row = item_to_row(item)
            if row is None:
                skipped += 1
                continue
            if insert_story(conn, row):
                inserted += 1
            # Commit after each story so partial progress is saved on crash
            conn.commit()

        log.info("Done — inserted %d, skipped %d non-story items", inserted, skipped)
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    run()
