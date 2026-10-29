"""Monthly USD budget: cost per answer, month buckets, legacy migration."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

import pytest

from app.usage import UsageStore, cost_usd, utc_month


def test_cost_uses_both_prices_per_million_tokens() -> None:
    assert cost_usd(1_000_000, 0, input_per_million=0.15, output_per_million=0.6) == 0.15
    assert cost_usd(0, 1_000_000, input_per_million=0.15, output_per_million=0.6) == 0.6
    assert cost_usd(2_000, 500, input_per_million=1.0, output_per_million=2.0) == 0.003
    assert cost_usd(10, 10, input_per_million=0.0, output_per_million=0.0) == 0.0


def test_utc_month_is_the_calendar_month() -> None:
    assert utc_month(datetime(2026, 1, 31, 23, 59, tzinfo=timezone.utc)) == "2026-01"
    assert utc_month(datetime(2026, 2, 1, 0, 0, tzinfo=timezone.utc)) == "2026-02"


def test_answers_accumulate_inside_a_month(tmp_path) -> None:
    store = UsageStore(tmp_path / "usage.sqlite", 1.0, input_price_per_m=1.0, output_price_per_m=2.0)
    assert store.spent("2026-03") == 0.0
    assert store.add(1_000, 0, "2026-03") == pytest.approx(0.001)
    assert store.add(1_000, 500, "2026-03") == pytest.approx(0.003)
    assert store.tokens("2026-03") == (2_000, 500)
    assert store.status("2026-03")["remaining_usd"] == pytest.approx(0.997)


def test_a_new_month_starts_at_zero(tmp_path) -> None:
    store = UsageStore(tmp_path / "usage.sqlite", 0.0005, input_price_per_m=1.0)
    store.add(1_000, 0, "2026-03")
    assert store.exhausted("2026-03") is True
    assert store.exhausted("2026-04") is False
    assert store.spent("2026-04") == 0.0


def test_a_zero_budget_allows_no_spend(tmp_path) -> None:
    store = UsageStore(tmp_path / "usage.sqlite", 0.0)
    assert store.exhausted() is True


def test_status_carries_no_secret(tmp_path) -> None:
    store = UsageStore(tmp_path / "usage.sqlite", 0.5, input_price_per_m=0.1, output_price_per_m=0.2)
    store.add(1_000, 500, "2026-03")
    status = store.status("2026-03")
    assert set(status) == {
        "month",
        "spent_usd",
        "budget_usd",
        "remaining_usd",
        "exhausted",
        "prompt_tokens",
        "completion_tokens",
        "prices_usd_per_million",
    }
    assert status["month"] == "2026-03"
    assert status["budget_usd"] == 0.5
    assert status["spent_usd"] == pytest.approx(0.0002)
    assert status["prices_usd_per_million"] == {"input": 0.1, "output": 0.2}


def legacy_database(path, rows: list[tuple[str, int]]) -> None:
    """A database exactly like the one the daily token cap used to write."""
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE usage (day TEXT PRIMARY KEY, tokens INTEGER NOT NULL DEFAULT 0)")
        conn.executemany("INSERT INTO usage (day, tokens) VALUES (?, ?)", rows)


def test_a_fresh_database_is_created_from_scratch(tmp_path) -> None:
    path = tmp_path / "nested" / "usage.sqlite"
    store = UsageStore(path, 0.5)
    assert path.is_file()
    assert store.status()["spent_usd"] == 0.0
    with sqlite3.connect(path) as conn:
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    assert tables == {"budget"}


def test_the_legacy_daily_table_is_migrated_without_losing_rows(tmp_path) -> None:
    path = tmp_path / "usage.sqlite"
    legacy_database(path, [("2026-03-01", 1_000), ("2026-03-02", 500), ("2026-04-01", 2_000)])
    store = UsageStore(path, 0.5, input_price_per_m=1.0)

    assert store.tokens("2026-03") == (1_500, 0)  # both March days, in one month
    assert store.spent("2026-03") == pytest.approx(0.0015)  # priced at the input rate
    assert store.tokens("2026-04") == (2_000, 0)

    with sqlite3.connect(path) as conn:
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
        kept = conn.execute("SELECT day, tokens FROM usage_daily_legacy ORDER BY day").fetchall()
    assert tables == {"budget", "usage_daily_legacy"}
    assert kept == [("2026-03-01", 1_000), ("2026-03-02", 500), ("2026-04-01", 2_000)]


def test_the_migration_runs_once(tmp_path) -> None:
    path = tmp_path / "usage.sqlite"
    legacy_database(path, [("2026-03-01", 1_000)])
    UsageStore(path, 0.5, input_price_per_m=1.0)
    again = UsageStore(path, 0.5, input_price_per_m=1.0)  # second start: no double count
    assert again.tokens("2026-03") == (1_000, 0)
    assert again.spent("2026-03") == pytest.approx(0.001)

    # A stray legacy table next to the migrated one is dropped, never recounted.
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE usage (day TEXT PRIMARY KEY, tokens INTEGER NOT NULL DEFAULT 0)")
        conn.execute("INSERT INTO usage (day, tokens) VALUES ('2026-03-03', 9999)")
    third = UsageStore(path, 0.5, input_price_per_m=1.0)
    assert third.tokens("2026-03") == (1_000, 0)
