"""SQLite-backed conversation history for the Telegram bot, keyed by session_id
(the Telegram chat id). Lives under data/state/ so a Docker volume mount keeps
it across container redeploys — see docker-compose.yml.
"""
import sqlite3

from _common import PROJECT_ROOT

DB_PATH = PROJECT_ROOT / "data" / "state" / "conversations.db"
MAX_STORED_TURNS_PER_SESSION = 50


def _connect():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS turns ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "session_id TEXT NOT NULL, "
        "question TEXT NOT NULL, "
        "answer TEXT NOT NULL, "
        "created_at TEXT DEFAULT CURRENT_TIMESTAMP)"
    )
    return conn


def get_recent_turns(session_id, limit):
    if not session_id or limit <= 0:
        return []
    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT question, answer FROM turns WHERE session_id = ? "
            "ORDER BY id DESC LIMIT ?",
            (session_id, limit),
        ).fetchall()
    finally:
        conn.close()
    return [{"question": q, "answer": a} for q, a in reversed(rows)]


def save_turn(session_id, question, answer):
    if not session_id:
        return
    conn = _connect()
    try:
        conn.execute(
            "INSERT INTO turns (session_id, question, answer) VALUES (?, ?, ?)",
            (session_id, question, answer),
        )
        conn.execute(
            "DELETE FROM turns WHERE session_id = ? AND id NOT IN ("
            "SELECT id FROM turns WHERE session_id = ? ORDER BY id DESC LIMIT ?)",
            (session_id, session_id, MAX_STORED_TURNS_PER_SESSION),
        )
        conn.commit()
    finally:
        conn.close()
