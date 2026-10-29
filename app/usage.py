"""Daily (UTC) token budget persisted in ``data/usage.sqlite``.

Only the date and the number of tokens are stored: no question, no IP, no key.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = "CREATE TABLE IF NOT EXISTS usage (day TEXT PRIMARY KEY, tokens INTEGER NOT NULL DEFAULT 0)"


def utc_day(now: datetime | None = None) -> str:
    return (now or datetime.now(timezone.utc)).strftime("%Y-%m-%d")


class UsageStore:
    """One row per UTC day; the cap is checked before calling the model."""

    def __init__(self, path: Path | str, daily_cap: int) -> None:
        self.path = Path(path)
        self.daily_cap = int(daily_cap)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as conn:
            conn.execute(SCHEMA)

    def tokens_today(self, day: str | None = None) -> int:
        with sqlite3.connect(self.path) as conn:
            row = conn.execute("SELECT tokens FROM usage WHERE day = ?", (day or utc_day(),)).fetchone()
        return int(row[0]) if row else 0

    def add(self, tokens: int, day: str | None = None) -> int:
        """Add ``tokens`` to today's total and return the new total."""
        if tokens <= 0:
            return self.tokens_today(day)
        key = day or utc_day()
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                "INSERT INTO usage (day, tokens) VALUES (?, ?)"
                " ON CONFLICT(day) DO UPDATE SET tokens = tokens + excluded.tokens",
                (key, int(tokens)),
            )
        return self.tokens_today(key)

    def exhausted(self, day: str | None = None) -> bool:
        return self.tokens_today(day) >= self.daily_cap
