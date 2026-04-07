"""
Telegram Bot — serves /top predictions and receives monitor alerts.

Commands:
  /top      — top 10 predicted stories from the last 6 hours
  /top N    — top N stories (max 20)
  /status   — system health: db counts, last collection, model version
  /help     — show available commands
"""
import logging
import os
import sys

from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shared.db import get_conn

TELEGRAM_TOKEN = os.environ["TELEGRAM_TOKEN"]
DEFAULT_TOP_N = 10
MAX_TOP_N = 20

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Database queries
# ---------------------------------------------------------------------------

TOP_STORIES_SQL = """
SELECT
    s.hn_id,
    s.title,
    s.url,
    s.type,
    s.score_at_capture,
    s.comments_at_capture,
    p.predicted_prob,
    p.model_version
FROM predictions p
JOIN stories s ON s.hn_id = p.hn_id
WHERE p.predicted_at >= NOW() - INTERVAL '6 hours'
ORDER BY p.predicted_prob DESC
LIMIT %(limit)s
"""


def fetch_status() -> dict:
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM stories")
            story_count = cur.fetchone()["count"]

            cur.execute("SELECT COUNT(*) FROM snapshots")
            snapshot_count = cur.fetchone()["count"]

            cur.execute("SELECT MAX(captured_at) FROM stories")
            last_collected = cur.fetchone()["max"]

            cur.execute(
                "SELECT model_version, MAX(predicted_at) as last_run, COUNT(*) as pred_count "
                "FROM predictions GROUP BY model_version ORDER BY last_run DESC LIMIT 1"
            )
            pred_row = cur.fetchone()

        return {
            "story_count": story_count,
            "snapshot_count": snapshot_count,
            "last_collected": last_collected,
            "model_version": pred_row["model_version"] if pred_row else None,
            "last_predicted": pred_row["last_run"] if pred_row else None,
            "pred_count": pred_row["pred_count"] if pred_row else 0,
        }
    finally:
        conn.close()


def fetch_top_stories(n: int) -> list[dict]:
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(TOP_STORIES_SQL, {"limit": n})
            return list(cur.fetchall())
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def format_story(rank: int, story: dict) -> str:
    title = story["title"]
    hn_id = story["hn_id"]
    prob  = story["predicted_prob"]
    score = story["score_at_capture"]
    comments = story["comments_at_capture"]
    url   = story.get("url") or f"https://news.ycombinator.com/item?id={hn_id}"
    hn_link = f"https://news.ycombinator.com/item?id={hn_id}"

    label = f"{rank}. [{title}]({url})"
    meta  = f"   score:{score} comments:{comments} prob:{prob:.0%} — [HN]({hn_link})"
    return f"{label}\n{meta}"


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------

async def cmd_top(update: Update, context: ContextTypes.DEFAULT_TYPE):
    n = DEFAULT_TOP_N
    if context.args:
        try:
            n = max(1, min(int(context.args[0]), MAX_TOP_N))
        except ValueError:
            await update.message.reply_text("Usage: /top [number]")
            return

    try:
        stories = fetch_top_stories(n)
    except Exception as e:
        log.exception("fetch_top_stories failed")
        await update.message.reply_text(f"Error fetching stories: {e}")
        return

    if not stories:
        await update.message.reply_text("No predictions available yet.")
        return

    version = stories[0]["model_version"]
    lines = [f"*Top {len(stories)} predicted stories* (model: `{version}`)\n"]
    for i, s in enumerate(stories, 1):
        lines.append(format_story(i, s))

    await update.message.reply_text(
        "\n".join(lines),
        parse_mode="Markdown",
        disable_web_page_preview=True,
    )


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        s = fetch_status()
    except Exception as e:
        log.exception("fetch_status failed")
        await update.message.reply_text(f"Error fetching status: {e}")
        return

    def fmt_time(dt):
        if not dt:
            return "never"
        now = __import__("datetime").datetime.now(dt.tzinfo)
        diff = now - dt
        mins = int(diff.total_seconds() // 60)
        if mins < 60:
            return f"{mins}m ago"
        return f"{mins // 60}h {mins % 60}m ago"

    model_line = (
        f"Model: `{s['model_version']}` — last run {fmt_time(s['last_predicted'])}, {s['pred_count']} total predictions"
        if s["model_version"]
        else "Model: no model deployed yet"
    )

    text = (
        "*HN Signal Status*\n\n"
        f"Stories collected: {s['story_count']:,}\n"
        f"Snapshots: {s['snapshot_count']:,}\n"
        f"Last collection: {fmt_time(s['last_collected'])}\n\n"
        f"{model_line}"
    )
    await update.message.reply_text(text, parse_mode="Markdown")


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (
        "*HN Signal Bot*\n\n"
        "/top — show top 10 predicted stories\n"
        "/top N — show top N stories (max 20)\n"
        "/status — system health and collection stats\n"
        "/help — show this message"
    )
    await update.message.reply_text(text, parse_mode="Markdown")


async def unknown_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    cmd = update.message.text.split()[0] if update.message.text else "that"
    await update.message.reply_text(
        f"`{cmd}` did nothing — try /help to see available commands.",
        parse_mode="Markdown",
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run():
    log.info("Bot starting")
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("top", cmd_top))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(MessageHandler(filters.COMMAND, unknown_command))
    log.info("Polling for updates")
    app.run_polling()


if __name__ == "__main__":
    run()
