"""CLI: ``python -m eval`` (recall@5 and MRR over ``eval/questions.jsonl``)."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from retrieval.search import get_embedder, get_store, hybrid_search

QUESTIONS_PATH = Path(__file__).resolve().parent / "questions.jsonl"
RECALL_K = 5
MRR_K = 8


def load_questions(path: Path = QUESTIONS_PATH) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def url_matches(result_url: str, expected_url: str) -> bool:
    """An expected URL without ``#`` matches any chunk of that page."""
    if "#" not in expected_url:
        return result_url.split("#", 1)[0] == expected_url.split("#", 1)[0]
    return result_url == expected_url


def first_hit_rank(urls: list[str], expected_urls: list[str]) -> int | None:
    for rank, url in enumerate(urls, start=1):
        if any(url_matches(url, expected) for expected in expected_urls):
            return rank
    return None


def evaluate(questions: list[dict], k: int = MRR_K) -> dict:
    store = get_store()
    embedder = get_embedder()
    rows: list[dict] = []
    started = time.perf_counter()

    for question in questions:
        results = hybrid_search(question["question"], k=k, store=store, embedder=embedder)
        urls = [result["url"] for result in results]
        rank = first_hit_rank(urls, question.get("expected_urls", []))
        rows.append(
            {
                "id": question["id"],
                "lang": question["lang"],
                "question": question["question"],
                "answerable": question["answerable"],
                "expected_urls": question.get("expected_urls", []),
                "hit_rank": rank,
                "top_urls": urls[:RECALL_K],
            }
        )

    elapsed = time.perf_counter() - started
    answerable = [row for row in rows if row["answerable"]]
    recall_hits = sum(1 for row in answerable if row["hit_rank"] is not None and row["hit_rank"] <= RECALL_K)
    reciprocal = sum(1.0 / row["hit_rank"] for row in answerable if row["hit_rank"] is not None)
    summary = {
        "questions": len(rows),
        "answerable": len(answerable),
        "unanswerable": len(rows) - len(answerable),
        f"recall@{RECALL_K}": round(recall_hits / len(answerable), 4) if answerable else 0.0,
        f"mrr@{k}": round(reciprocal / len(answerable), 4) if answerable else 0.0,
        "seconds": round(elapsed, 2),
        "mean_ms_per_question": round(elapsed * 1000 / max(len(rows), 1), 1),
    }
    return {"summary": summary, "rows": rows}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="eval", description="Evaluate hybrid retrieval")
    parser.add_argument("--json", action="store_true", help="print the full report as JSON")
    parser.add_argument("--questions", default=str(QUESTIONS_PATH), help="path to questions.jsonl")
    args = parser.parse_args(argv)

    report = evaluate(load_questions(Path(args.questions)))
    summary = report["summary"]
    rows = report["rows"]

    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0

    for row in rows:
        status = "n/a" if not row["answerable"] else ("HIT" if row["hit_rank"] and row["hit_rank"] <= RECALL_K else "miss")
        rank = row["hit_rank"] if row["hit_rank"] is not None else "-"
        print(f"[{row['id']}] ({row['lang']}) {status} rank={rank} :: {row['question']}")
        for expected in row["expected_urls"]:
            found = any(url_matches(url, expected) for url in row["top_urls"])
            print(f"    expected: {expected}  ({'in top-5' if found else 'not in top-5'})")
        for position, url in enumerate(row["top_urls"], start=1):
            print(f"    {position}. {url}")
        print()

    print("summary")
    print(f"  questions: {summary['questions']} ({summary['answerable']} answerable, {summary['unanswerable']} not)")
    print(f"  recall@{RECALL_K}: {summary[f'recall@{RECALL_K}']}")
    print(f"  mrr@{MRR_K}: {summary[f'mrr@{MRR_K}']}")
    print(f"  time: {summary['seconds']}s ({summary['mean_ms_per_question']} ms/question)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
