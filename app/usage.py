"""Monthly (UTC) spending budget in USD, persisted in ``data/usage.sqlite``.

The cost of one completion is ``prompt_tokens * price_in / 1e6 +
completion_tokens * price_out / 1e6`` and it is accumulated per UTC month
(``YYYY-MM``).  Only the month, the token counts and the cost are stored: no
question, no IP, no key.

A database written by an older version (one row per UTC day with a token
count) is migrated in place: the daily rows are folded into their month, the
legacy table is renamed instead of deleted, and the migration is idempotent.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS budget (
    month TEXT PRIMARY KEY,
    cost_usd REAL NOT NULL DEFAULT 0,
    prompt_tokens INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0
)
"""
LEGACY_TABLE = "usage"
LEGACY_MIGRATED = "usage_daily_legacy"
#: Precision kept in the database and reported: a millionth of a dollar is enough.
DIGITS = 6


def utc_month(now: datetime | None = None) -> str:
    return (now or datetime.now(timezone.utc)).strftime("%Y-%m")


def cost_usd(
    prompt_tokens: int,
    completion_tokens: int,
    *,
    input_per_million: float,
    output_per_million: float,
) -> float:
    """USD cost of one completion at prices per million tokens."""
    cost = (
        int(prompt_tokens) * float(input_per_million)
        + int(completion_tokens) * float(output_per_million)
    ) / 1_000_000
    return round(cost, DIGITS)


def _tables(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
    return {row[0] for row in rows}


class UsageStore:
    """One row per UTC month; the budget is checked before calling the model."""

    def __init__(
        self,
        path: Path | str,
        budget_usd_per_month: float,
        *,
        input_price_per_m: float = 0.0,
        output_price_per_m: float = 0.0,
    ) -> None:
        self.path = Path(path)
        self.budget_usd_per_month = float(budget_usd_per_month)
        self.input_price_per_m = float(input_price_per_m)
        self.output_price_per_m = float(output_price_per_m)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as conn:
            conn.execute(SCHEMA)
            self._migrate(conn)

    # ------------------------------------------------------------ migration
    def _migrate(self, conn: sqlite3.Connection) -> None:
        """Fold the legacy daily token table into the monthly cost table once.

        The old rows only carried a token count, so their cost is priced at the
        input rate of the current configuration; the rows stay on disk under
        :data:`LEGACY_MIGRATED`.
        """
        tables = _tables(conn)
        if LEGACY_TABLE not in tables:
            return
        if LEGACY_MIGRATED in tables:  # already migrated: the new schema is authoritative
            conn.execute(f"DROP TABLE {LEGACY_TABLE}")
            return
        for day, tokens in conn.execute(f"SELECT day, tokens FROM {LEGACY_TABLE}").fetchall():
            tokens = int(tokens)
            conn.execute(
                "INSERT INTO budget (month, cost_usd, prompt_tokens) VALUES (?, ?, ?)"
                " ON CONFLICT(month) DO UPDATE SET"
                " cost_usd = cost_usd + excluded.cost_usd,"
                " prompt_tokens = prompt_tokens + excluded.prompt_tokens",
                (
                    str(day)[:7],
                    cost_usd(tokens, 0, input_per_million=self.input_price_per_m, output_per_million=0.0),
                    tokens,
                ),
            )
        conn.execute(f"ALTER TABLE {LEGACY_TABLE} RENAME TO {LEGACY_MIGRATED}")

    # --------------------------------------------------------------- reads
    def _row(self, month: str) -> tuple[float, int, int]:
        with sqlite3.connect(self.path) as conn:
            row = conn.execute(
                "SELECT cost_usd, prompt_tokens, completion_tokens FROM budget WHERE month = ?",
                (month,),
            ).fetchone()
        return (float(row[0]), int(row[1]), int(row[2])) if row else (0.0, 0, 0)

    def spent(self, month: str | None = None) -> float:
        """USD spent in ``month`` (``YYYY-MM``, default: this UTC month)."""
        return self._row(month or utc_month())[0]

    def tokens(self, month: str | None = None) -> tuple[int, int]:
        """``(prompt, completion)`` tokens recorded in ``month``."""
        cost, prompt, completion = self._row(month or utc_month())
        del cost
        return prompt, completion

    # -------------------------------------------------------------- writes
    def add(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        month: str | None = None,
    ) -> float:
        """Add one completion to its month and return the new month total."""
        prompt_tokens = max(0, int(prompt_tokens))
        completion_tokens = max(0, int(completion_tokens))
        key = month or utc_month()
        if not (prompt_tokens or completion_tokens):
            return self.spent(key)
        cost = cost_usd(
            prompt_tokens,
            completion_tokens,
            input_per_million=self.input_price_per_m,
            output_per_million=self.output_price_per_m,
        )
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                "INSERT INTO budget (month, cost_usd, prompt_tokens, completion_tokens)"
                " VALUES (?, ?, ?, ?)"
                " ON CONFLICT(month) DO UPDATE SET"
                " cost_usd = cost_usd + excluded.cost_usd,"
                " prompt_tokens = prompt_tokens + excluded.prompt_tokens,"
                " completion_tokens = completion_tokens + excluded.completion_tokens",
                (key, cost, prompt_tokens, completion_tokens),
            )
        return self.spent(key)

    # -------------------------------------------------------------- budget
    def exhausted(self, month: str | None = None) -> bool:
        """True when the month already spent its budget (``0`` allows no spend)."""
        return self.spent(month) >= self.budget_usd_per_month

    def status(self, month: str | None = None) -> dict:
        """Public budget state: month, spending, cap and prices.  No secrets."""
        key = month or utc_month()
        cost, prompt, completion = self._row(key)
        return {
            "month": key,
            "spent_usd": round(cost, DIGITS),
            "budget_usd": round(self.budget_usd_per_month, DIGITS),
            "remaining_usd": round(max(0.0, self.budget_usd_per_month - cost), DIGITS),
            "exhausted": cost >= self.budget_usd_per_month,
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "prices_usd_per_million": {
                "input": self.input_price_per_m,
                "output": self.output_price_per_m,
            },
        }
