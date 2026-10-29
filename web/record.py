"""``python -m web.record``: write ``web/recorded.json`` (the offline mode).

For the four example questions plus four extra ones (two answerable, two not,
half EN and half ES) it stores the question, its language and the **real**
sources that ``hybrid_search`` returns.  The embedding model is loaded once.

If a **real** generation report (``eval/reports/<date>-generate.json``) exists
and holds that question, its ``abstain`` / ``sentences`` / ``sources`` are
copied verbatim.  A ``-mock.json`` report is never read: without a real one the
entry keeps ``"answer": null`` and ``"status": "retrieval_only"``, so the web UI
can never replay a simulated answer as if it came from the model.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from generation.validate import normalize

REPO_ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = Path(__file__).resolve().parent
QUESTIONS_PATH = REPO_ROOT / "eval" / "questions.jsonl"
REPORTS_DIR = REPO_ROOT / "eval" / "reports"
OUT_PATH = WEB_DIR / "recorded.json"
K = 8

#: The four examples shown in the form (two EN, two ES), then two extra
#: answerable and two extra unanswerable questions (one per language each).
QUESTION_IDS = ("a01", "a15", "a16", "a06", "a33", "a42", "u01", "u30")


def load_questions(path: Path | str = QUESTIONS_PATH) -> dict[str, dict]:
    with Path(path).open(encoding="utf-8") as handle:
        return {row["id"]: row for row in (json.loads(line) for line in handle if line.strip())}


def latest_real_report(reports_dir: Path | str = REPORTS_DIR) -> Path | None:
    """Newest ``*-generate.json`` that is not a mock report."""
    reports_dir = Path(reports_dir)
    candidates = [
        path for path in sorted(reports_dir.glob("*-generate.json")) if not path.name.endswith("-generate-mock.json")
    ]
    return candidates[-1] if candidates else None


def report_answers(report_path: Path | None) -> tuple[dict[str, dict], str | None]:
    """Map normalised question -> row of a real report (never a mock one)."""
    if report_path is None:
        return {}, None
    data = json.loads(Path(report_path).read_text(encoding="utf-8"))
    rows = {
        normalize(row["question"]): row
        for row in data.get("questions", [])
        if isinstance(row, dict) and isinstance(row.get("question"), str)
    }
    return rows, f"eval-report:{Path(report_path).name}"


def build_entry(question: dict, sources: list[dict], answers: dict[str, dict], source_label: str | None) -> dict:
    recorded_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    entry = {
        "id": question["id"],
        "lang": question["lang"],
        "question": question["question"],
        "recorded_at": recorded_at,
        "source": "retrieval-only",
        "status": "retrieval_only",
        "sources": sources,
        "answer": None,
    }
    row = answers.get(normalize(question["question"]))
    if row is not None and isinstance(row.get("sentences"), list):
        entry["source"] = source_label or "retrieval-only"
        entry["status"] = "abstained" if row.get("abstain") else "answered"
        entry["answer"] = {
            "abstain": bool(row.get("abstain")),
            "reason": row.get("reason"),
            "sentences": row["sentences"],
            "sources": row.get("sources") or [],
        }
    return entry


def record(
    *,
    question_ids: tuple[str, ...] = QUESTION_IDS,
    report_path: Path | None | str = "auto",
    out_path: Path | str = OUT_PATH,
    log=None,
) -> dict:
    """Build and write ``recorded.json``; returns the document."""
    from retrieval.search import get_embedder, get_store, hybrid_search

    questions = load_questions()
    missing = [question_id for question_id in question_ids if question_id not in questions]
    if missing:
        raise KeyError(f"question ids not found in {QUESTIONS_PATH}: {', '.join(missing)}")

    if report_path == "auto":
        resolved: Path | None = latest_real_report()
    else:
        resolved = Path(report_path) if report_path else None
        if resolved is not None and resolved.name.endswith("-generate-mock.json"):
            raise ValueError("refusing a mock report: recorded.json never replays simulated answers")

    answers, source_label = report_answers(resolved)
    store = get_store()
    embedder = get_embedder()  # loaded once for every question

    entries = []
    for question_id in question_ids:
        question = questions[question_id]
        results = hybrid_search(question["question"], k=K, store=store, embedder=embedder)
        sources = [
            {"n": position, "url": result["url"], "title": result["page_title"]}
            for position, result in enumerate(results, start=1)
        ]
        entry = build_entry(question, sources, answers, source_label)
        entries.append(entry)
        if log:
            log(
                f"{entry['id']:<4} {entry['lang']} {entry['status']:<14} "
                f"sources={len(entry['sources'])} source={entry['source']}"
            )

    document = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        # Only the file name: the recorded document is served publicly and must
        # not publish a path of the machine that recorded it.
        "report": resolved.name if resolved is not None else None,
        "entries": entries,
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return document


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="web.record", description="Write web/recorded.json (no model, no network)")
    parser.add_argument("--out", default=str(OUT_PATH), help="output file (default: web/recorded.json)")
    parser.add_argument(
        "--report",
        "--from-report",
        dest="report",
        default="auto",
        help=(
            "real generation report to replay ('auto' = newest *-generate.json, '' = none); "
            "the answers are copied from the report, the model is never called"
        ),
    )
    args = parser.parse_args(argv)

    try:
        document = record(report_path=args.report, out_path=args.out, log=lambda message: print(message, flush=True))
    except (KeyError, ValueError) as error:
        print(f"ERROR: {error}", flush=True)
        return 2
    print()
    print(f"questions: {len(document['entries'])}")
    print(f"report: {document['report'] or 'none (retrieval only)'}")
    print(f"written: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
