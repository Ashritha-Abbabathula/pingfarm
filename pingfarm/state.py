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
    district: str | None = None
    soil: str | None = None            # red | black | sandy | loamy | clay | laterite | alluvial
    planted_on: str | None = None      # ISO date
    alerts: bool = False               # opt-in to proactive messages


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
            CREATE TABLE IF NOT EXISTS alerts_sent(user_id TEXT, day TEXT, kind TEXT, ts REAL,
                PRIMARY KEY(user_id, day, kind));
            CREATE TABLE IF NOT EXISTS geocache(name TEXT PRIMARY KEY, lat REAL, lon REAL);
            """
        )
        # add newer columns to an existing database (a tiny "migration")
        have = {r[1] for r in self.db.execute("PRAGMA table_info(users)")}
        for col, typ in (("district", "TEXT"), ("soil", "TEXT"), ("planted_on", "TEXT"),
                         ("alerts", "INTEGER DEFAULT 0"), ("last_inbound", "REAL")):
            if col not in have:
                self.db.execute(f"ALTER TABLE users ADD COLUMN {col} {typ}")
        self.db.commit()

    def first_time_seeing(self, msg_id: str) -> bool:
        """True the first time, False for a retry of the same message."""
        cur = self.db.execute("INSERT OR IGNORE INTO seen(msg_id, ts) VALUES (?, ?)", (msg_id, time.time()))
        self.db.commit()
        return cur.rowcount == 1

    def get_user(self, user_id: str) -> UserProfile:
        row = self.db.execute(
            "SELECT lat, lon, crop, state, district, soil, planted_on, alerts FROM users WHERE user_id=?", (user_id,)
        ).fetchone()
        if not row:
            return UserProfile(user_id)
        return UserProfile(user_id, *row[:7], alerts=bool(row[7]))

    def update_user(self, user_id: str, **fields):
        current = self.get_user(user_id)
        for k, v in fields.items():
            setattr(current, k, v)
        self.db.execute(
            "INSERT OR REPLACE INTO users(user_id, lat, lon, crop, state, district, soil, planted_on, alerts, updated, last_inbound) "
            "VALUES (?,?,?,?,?,?,?,?,?,?, (SELECT last_inbound FROM users WHERE user_id=?))",
            (user_id, current.lat, current.lon, current.crop, current.state, current.district, current.soil,
             current.planted_on, int(current.alerts), time.time(), user_id),
        )
        self.db.commit()

    def touch_inbound(self, user_id: str):
        """Remember when the farmer last wrote to us: WhatsApp only lets us message freely within 24 hours of that."""
        self.update_user(user_id)
        self.db.execute("UPDATE users SET last_inbound=? WHERE user_id=?", (time.time(), user_id))
        self.db.commit()

    def last_inbound(self, user_id: str) -> float | None:
        row = self.db.execute("SELECT last_inbound FROM users WHERE user_id=?", (user_id,)).fetchone()
        return row[0] if row else None

    def alert_users(self) -> list[UserProfile]:
        ids = [r[0] for r in self.db.execute("SELECT user_id FROM users WHERE alerts=1 AND lat IS NOT NULL AND crop IS NOT NULL")]
        return [self.get_user(i) for i in ids]

    def already_alerted(self, user_id: str, day: str, kind: str) -> bool:
        return self.db.execute("SELECT 1 FROM alerts_sent WHERE user_id=? AND day=? AND kind=?", (user_id, day, kind)).fetchone() is not None

    def mark_alerted(self, user_id: str, day: str, kind: str):
        self.db.execute("INSERT OR IGNORE INTO alerts_sent(user_id, day, kind, ts) VALUES (?,?,?,?)", (user_id, day, kind, time.time()))
        self.db.commit()

    def delete_user(self, user_id: str):
        """The farmer's right to be forgotten: remove the profile, history and alert log."""
        for table in ("users", "history", "alerts_sent"):
            self.db.execute(f"DELETE FROM {table} WHERE user_id=?", (user_id,))
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
