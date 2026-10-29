"""Threshold calibration logic (pure: no retrieval, no model)."""

from __future__ import annotations

from generation.calibrate import candidate_thresholds, choose, sweep


def rows(signal: str) -> list[dict]:
    # Answerable questions score high, unanswerable ones low, with a few overlaps.
    return [
        {"id": f"a{i}", "lang": "en", "answerable": True, signal: value}
        for i, value in enumerate([0.90, 0.88, 0.86, 0.84, 0.83, 0.82, 0.80])
    ] + [
        {"id": f"u{i}", "lang": "en", "answerable": False, signal: value}
        for i, value in enumerate([0.81, 0.79, 0.70, 0.60])
    ]


def test_candidate_thresholds_bracket_every_value() -> None:
    candidates = candidate_thresholds([0.2, 0.4])
    assert candidates[0] < 0.2
    assert candidates[-1] > 0.4
    assert 0.2 < candidates[1] < 0.4
    assert candidate_thresholds([]) == [0.0]


def test_sweep_respects_the_max_lost_budget() -> None:
    best, table = sweep(rows("vector_cosine"), "vector_cosine", max_lost=1)
    assert best["answerable_lost"] <= 1
    # The threshold just above 0.82 abstains 0.81/0.79/0.70/0.60 (4 of 4) losing 1 answerable.
    assert best["unanswerable_abstained"] == 4
    assert best["answerable_lost"] == 1
    assert len(table) == len(candidate_thresholds([row["vector_cosine"] for row in rows("vector_cosine")]))


def test_sweep_takes_the_highest_threshold_on_ties() -> None:
    best, table = sweep(rows("rrf"), "rrf", max_lost=2)
    feasible = [row for row in table if row["answerable_lost"] <= 2]
    top = max(row["unanswerable_abstained"] for row in feasible)
    assert best["unanswerable_abstained"] == top
    assert best["threshold"] == max(row["threshold"] for row in feasible if row["unanswerable_abstained"] == top)


def test_choose_prefers_the_signal_that_abstains_more() -> None:
    good = {row["threshold"]: row for row in sweep(rows("vector_cosine"), "vector_cosine", max_lost=2)[1]}
    bad = {row["threshold"]: row for row in sweep(rows("rrf"), "rrf", max_lost=2)[1]}
    table = {
        "vector_cosine": list(good.values()),
        "rrf": [dict(row, unanswerable_abstained=0) for row in bad.values()],
    }
    signal, chosen = choose(table)
    assert signal == "vector_cosine"
    assert chosen["unanswerable_abstained"] > 0


def test_choose_falls_back_when_one_signal_is_degenerate() -> None:
    table = {
        "vector_cosine": [
            {"signal": "vector_cosine", "threshold": 0.5, "answerable_lost": 2, "unanswerable_abstained": 3}
        ],
        "rrf": [{"signal": "rrf", "threshold": 0.01, "answerable_lost": 0, "unanswerable_abstained": 0}],
    }
    signal, chosen = choose(table)
    assert signal == "vector_cosine"
    assert chosen["threshold"] == 0.5
