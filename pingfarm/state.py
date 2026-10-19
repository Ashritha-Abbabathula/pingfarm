"""Tiny SQLite store: who the user is (location, crop, language) and which messages we've seen.

Why SQLite: one file, no server, plenty for a pilot with a few hundred farmers.
Why dedupe: WhatsApp's servers retry a message if we are slow to answer. Without a record of
message ids, the farmer would get the same reply three times.
"""
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class UserProfile:
    user_id: str
    lat: float | None = None
    lon: float | None = None
    crop: str | None = None
    state: str | None = None


class State:
    def __init__(self, path: str | Path = ":memory:"):
        self.db = sqlite3.connect(str(path), check_same_thread=False)
        self.db.executescript(
            """
            CREATE TABLE IF NOT EXISTS users(
                user_id TEXT PRIMARY KEY, lat REAL, lon REAL, crop TEXT, state TEXT, updated REAL);
            CREATE TABLE IF NOT EXISTS seen(msg_id TEXT PRIMARY KEY, ts REAL);
            CREATE TABLE IF NOT EXISTS history(id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT, role TEXT, text TEXT, ts REAL);
            """
        )

    def first_time_seeing(self, msg_id: str) -> bool:
        """True the first time, False for a retry of the same message."""
        cur = self.db.execute("INSERT OR IGNORE INTO seen(msg_id, ts) VALUES (?, ?)", (msg_id, time.time()))
        self.db.commit()
        return cur.rowcount == 1

    def get_user(self, user_id: str) -> UserProfile:
        row = self.db.execute("SELECT lat, lon, crop, state FROM users WHERE user_id=?", (user_id,)).fetchone()
        return UserProfile(user_id, *row) if row else UserProfile(user_id)

    def update_user(self, user_id: str, **fields):
        current = self.get_user(user_id)
        for k, v in fields.items():
            setattr(current, k, v)
        self.db.execute(
            "INSERT OR REPLACE INTO users(user_id, lat, lon, crop, state, updated) VALUES (?,?,?,?,?,?)",
            (user_id, current.lat, current.lon, current.crop, current.state, time.time()),
        )
        self.db.commit()

    def add_history(self, user_id: str, role: str, text: str):
        self.db.execute("INSERT INTO history(user_id, role, text, ts) VALUES (?,?,?,?)",
                        (user_id, role, text, time.time()))
        self.db.commit()

    def recent_history(self, user_id: str, n: int = 6) -> list[tuple[str, str]]:
        rows = self.db.execute(
            "SELECT role, text FROM history WHERE user_id=? ORDER BY id DESC LIMIT ?", (user_id, n)
        ).fetchall()
        return rows[::-1]
