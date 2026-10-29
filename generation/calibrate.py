"""Calibrate the pre-model abstention threshold (retrieval only, no LLM).

``python -m generation.calibrate`` retrieves the 50 answerable and 30
unanswerable questions of ``eval/questions.jsonl``, computes two candidate
signals (top-1 vector cosine and top-1 RRF) and picks, per signal, the highest
threshold that abstains the most unanswerable questions while losing at most
``--max-lost`` answerable ones.  It prints the sweep table and writes
``generation/threshold.json``.

The threshold is calibrated on the same 80 questions used to report metrics, so
the numbers are optimistically biased: it is a smoke-test set, not a benchmark.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

from retrieval.search import get_embedder, get_store, hybrid_search

from .generate import SIGNAL_NAMES, THRESHOLD_PATH, retrieval_signals

QUESTIONS_PATH = Path(__file__).resolve().parent.parent / "eval" / "questions.jsonl"
DEFAULT_MAX_LOST = 2


def load_questions(path: Path | str = QUESTIONS_PATH) -> list[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def measure_signals(
    questions: list[dict],
    store=None,
    embedder=None,
    k: int = 8,
    log=None,
) -> list[dict]:
    """One row per question with its retrieval signals (no model call)."""
    store = store or get_store()
    embedder = embedder or get_embedder()
    rows: list[dict] = []
    for position, question in enumerate(questions, start=1):
        results = hybrid_search(question["question"], k=k, store=store, embedder=embedder)
        signals = retrieval_signals(question["question"], results, store, embedder)
        rows.append(
            {
                "id": question["id"],
                "lang": question["lang"],
                "answerable": bool(question["answerable"]),
                **signals,
            }
        )
        if log and position % 10 == 0:
            log(f"  retrieved {position}/{len(questions)}")
    return rows


def candidate_thresholds(values: list[float]) -> list[float]:
    """Thresholds between consecutive signal values (plus the two extremes)."""
    ordered = sorted(set(values))
    if not ordered:
        return [0.0]
    candidates = [ordered[0] - 1e-6]
    for lower, upper in zip(ordered, ordered[1:]):
        candidates.append((lower + upper) / 2.0)
    candidates.append(ordered[-1] + 1e-6)
    return candidates


def sweep(rows: list[dict], signal: str, max_lost: int = DEFAULT_MAX_LOST) -> tuple[dict, list[dict]]:
    """Sweep thresholds; return the chosen row and the whole table.

    Abstention rule: ``signal < threshold``.  The chosen threshold maximizes the
    unanswerable abstentions subject to ``answerable_lost <= max_lost``; ties go
    to the higher threshold.
    """
    answerable = [row for row in rows if row["answerable"]]
    unanswerable = [row for row in rows if not row["answerable"]]
    table: list[dict] = []
    for threshold in candidate_thresholds([row[signal] for row in rows]):
        lost = sum(1 for row in answerable if row[signal] < threshold)
        abstained = sum(1 for row in unanswerable if row[signal] < threshold)
        table.append(
            {
                "signal": signal,
                "threshold": round(threshold, 6),
                "answerable_lost": lost,
                "unanswerable_abstained": abstained,
            }
        )

    feasible = [row for row in table if row["answerable_lost"] <= max_lost]
    best = max(feasible, key=lambda row: (row["unanswerable_abstained"], row["threshold"]))
    return best, table


def choose(table: dict[str, list[dict]]) -> tuple[str, dict]:
    """Signal that separates best: most unanswerable abstained, fewer losses."""
    best_signal = None
    best_row = None
    for signal in SIGNAL_NAMES:
        feasible = [row for row in table.get(signal, []) if row["answerable_lost"] <= DEFAULT_MAX_LOST]
        if not feasible:
            continue
        row = max(feasible, key=lambda entry: (entry["unanswerable_abstained"], entry["threshold"]))
        key = (row["unanswerable_abstained"], -row["answerable_lost"], row["threshold"])
        if best_row is None or key > (best_row["unanswerable_abstained"], -best_row["answerable_lost"], best_row["threshold"]):
            best_signal, best_row = signal, row
    if best_row is None or best_signal is None:
        raise SystemExit("no feasible threshold found")
    return best_signal, best_row


def print_table(table: list[dict], signal: str, best: dict) -> None:
    header = f"{'threshold':>12}  {'ans.lost':>8}  {'unans.abst':>10}"
    print(f"signal: {signal}")
    print(header)
    print("-" * len(header))
    seen: set[tuple[int, int]] = set()
    for row in table:
        state = (row["answerable_lost"], row["unanswerable_abstained"])
        if state in seen and row is not best:
            continue
        seen.add(state)
        marker = "  <- chosen" if row is best else ""
        print(f"{row['threshold']:>12.6f}  {row['answerable_lost']:>8}  {row['unanswerable_abstained']:>10}{marker}")
    print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="generation.calibrate", description="Calibrate the abstention threshold")
    parser.add_argument("--file", default=str(QUESTIONS_PATH), help="question set (default: eval/questions.jsonl)")
    parser.add_argument(
        "--max-lost",
        type=int,
        default=DEFAULT_MAX_LOST,
        help="answerable questions the threshold may abstain (default: 2 of 50)",
    )
    parser.add_argument("--out", default=str(THRESHOLD_PATH), help="threshold file to write")
    parser.add_argument("--k", type=int, default=8, help="retrieved URLs per question (default: 8)")
    args = parser.parse_args(argv)

    questions = load_questions(Path(args.file))
    answerable = sum(1 for question in questions if question["answerable"])
    print(f"questions: {len(questions)} ({answerable} answerable, {len(questions) - answerable} not)")
    print("retrieving (no LLM is called)...", flush=True)
    rows = measure_signals(questions, k=args.k, log=lambda message: print(message, flush=True))

    table = {signal: [row for row in sweep(rows, signal, args.max_lost)[1]] for signal in SIGNAL_NAMES}
    best: dict[str, dict] = {}
    for signal in SIGNAL_NAMES:
        best_row, _ = sweep(rows, signal, args.max_lost)
        best[signal] = best_row
        print_table(table[signal], signal, best_row)

    signal, chosen = choose(table)
    print("summary")
    for name in SIGNAL_NAMES:
        row = best[name]
        print(
            f"  {name:<14} threshold={row['threshold']:.6f}  "
            f"answerable lost={row['answerable_lost']}/{answerable}  "
            f"unanswerable abstained={row['unanswerable_abstained']}/{len(questions) - answerable}"
        )
    print(f"  chosen: {signal} @ {chosen['threshold']:.6f}")

    payload = {
        "signal": signal,
        "value": chosen["threshold"],
        "answerable_lost": chosen["answerable_lost"],
        "answerable_total": answerable,
        "unanswerable_abstained": chosen["unanswerable_abstained"],
        "unanswerable_total": len(questions) - answerable,
        "max_lost": args.max_lost,
        "calibrated_on": str(Path(args.file)),
        "calibrated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "note": "calibrated on the same 80-question eval set used for reporting: optimistically biased",
        "signals": {
            name: {
                "threshold": best[name]["threshold"],
                "answerable_lost": best[name]["answerable_lost"],
                "unanswerable_abstained": best[name]["unanswerable_abstained"],
            }
            for name in SIGNAL_NAMES
        },
    }
    out = Path(args.out)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
