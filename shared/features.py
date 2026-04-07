"""
Feature engineering — uses ONLY columns frozen at capture time.

Called by both the Predictor (EC2 serving) and the Trainer (local GPU).
Never touches the snapshots table — that would be data leakage
(you can't know a story's future score when predicting it).

A "feature" is just a number the model uses to make a prediction.
We turn raw story data (title, score, time posted, etc.) into a flat
list of numbers that the model understands.
"""
import re
from datetime import timezone


# ---------------------------------------------------------------------------
# Hour-of-day / day-of-week buckets (UTC)
# HN has strong posting patterns — stories posted in the morning
# US time (afternoon UTC) tend to do better.
# ---------------------------------------------------------------------------

def _hour_of_day(dt) -> int | None:
    """Return 0-23 UTC hour from a datetime, or None if missing."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        # Assume UTC if no timezone info is attached
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.hour


def _day_of_week(dt) -> int | None:
    """Return 0 (Monday) to 6 (Sunday), or None if missing."""
    if dt is None:
        return None
    return dt.weekday()


# ---------------------------------------------------------------------------
# Title signals
# Certain words in titles correlate strongly with front-page performance.
# ---------------------------------------------------------------------------

# Pre-compiled regexes for Show HN / Ask HN detection
_SHOW_HN = re.compile(r"^show hn", re.IGNORECASE)
_ASK_HN  = re.compile(r"^ask hn",  re.IGNORECASE)

# Keywords that tend to appear in high-scoring stories.
# Each becomes a 0/1 feature: 1 if the keyword is in the title, 0 if not.
TITLE_KEYWORDS = [
    "gpt", "llm", "ai", "rust", "python", "open source", "github",
    "launch", "we built", "i built", "i made", "release",
    "postgres", "linux", "security", "privacy", "free",
]


def _title_flags(title: str) -> dict:
    """
    Turn a title string into a dict of 0/1 keyword flags plus length features.

    Example output for "Show HN: I built a Rust compiler":
      {"kw_rust": 1, "kw_i_built": 1, "has_show_hn": 1, "title_len": 34, ...}
    """
    tl = title.lower()
    # One feature per keyword: 1 if keyword appears anywhere in title, else 0
    flags = {f"kw_{kw.replace(' ', '_')}": int(kw in tl) for kw in TITLE_KEYWORDS}
    flags["has_show_hn"] = int(bool(_SHOW_HN.search(title)))
    flags["has_ask_hn"]  = int(bool(_ASK_HN.search(title)))
    flags["title_len"]   = len(title)         # character count
    flags["title_words"] = len(title.split())  # word count
    return flags


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_feature_dict(story: dict) -> dict:
    """
    Build a flat feature dict from a stories-table row.

    This is the single source of truth for features — both the trainer
    and the predictor call this exact function, so they always agree on
    what the inputs mean.

    story keys used: title, domain, author, type, score_at_capture,
                     comments_at_capture, time_posted, captured_at
    """
    feats: dict = {}

    # How many points / comments the story had the moment we first saw it.
    # A story with 5 points at capture is already more interesting than one with 0.
    feats["score_at_capture"]    = story.get("score_at_capture") or 0
    feats["comments_at_capture"] = story.get("comments_at_capture") or 0

    # When was the story posted? Stories posted during peak HN hours do better.
    # Use explicit None check — hour=0 (midnight) and day=0 (Monday) are valid
    # values that would be incorrectly skipped by a simple `or` short-circuit.
    tp = story.get("time_posted")
    feats["hour_of_day"] = _hour_of_day(tp) if tp is not None else 12  # default: noon
    feats["day_of_week"] = _day_of_week(tp) if tp is not None else 0   # default: Monday

    # Story type as one-hot encoding (only one of these will be 1 at a time).
    # One-hot means: instead of storing "show_hn" as a string, we store
    # type_story=0, type_show_hn=1, type_ask_hn=0, type_job=0.
    stype = story.get("type") or "story"
    for t in ("story", "show_hn", "ask_hn", "job"):
        feats[f"type_{t}"] = int(stype == t)

    # Does the story link to an external URL, or is it a self-post (Ask HN)?
    feats["has_url"] = int(bool(story.get("url")))

    # Add all the keyword/length features from the title
    feats.update(_title_flags(story.get("title") or ""))

    return feats


def feature_names() -> list[str]:
    """
    Return the ordered list of feature names.

    The order must be consistent between training and prediction — the model
    expects columns in the same order it was trained on.
    We derive this by running build_feature_dict on a dummy story so we never
    have to maintain a separate hardcoded list.
    """
    dummy = {
        "title": "", "domain": None, "author": None, "type": "story",
        "score_at_capture": 0, "comments_at_capture": 0,
        "time_posted": None, "captured_at": None, "url": None,
    }
    return list(build_feature_dict(dummy).keys())
