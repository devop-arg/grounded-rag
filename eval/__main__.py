"""CLI: ``python -m eval`` (recall@5 and MRR over a labelled question set).

Besides the single-policy report, ``--compare`` runs every fusion policy over the
same set and prints recall@5 / MRR per policy and per question language.
Expected URLs that do not exist in the corpus are reported up front and dropped
from the metrics (``--strict`` turns that into a non-zero exit).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from retrieval.lang import detect_lang
from retrieval.search import DEFAULT_POLICY, POLICIES, get_embedder, get_store, hybrid_search
from retrieval.store import iter_chunks

QUESTIONS_PATH = Path(__file__).resolve().parent / "questions.jsonl"
RECALL_K = 5
MRR_K = 8


class CachedEmbedder:
    """Encode each query once, so ``--compare`` does not embed four times."""

    def __init__(self, embedder) -> None:
        self._embedder = embedder
        self._cache: dict[str, object] = {}

    def encode_query(self, query: str):
        if query not in self._cache:
            self._cache[query] = self._embedder.encode_query(query)
        return self._cache[query]


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


def corpus_urls(chunks_path: Path | str | None = None) -> tuple[set[str], set[str]]:
    """Return ``(pages, urls)``: page URLs without anchor and full chunk URLs."""
    pages: set[str] = set()
    urls: set[str] = set()
    for chunk in iter_chunks(chunks_path) if chunks_path else iter_chunks():
        urls.add(chunk["url"])
        pages.add(chunk["url"].split("#", 1)[0])
    return pages, urls


def url_exists(expected_url: str, pages: set[str], urls: set[str]) -> bool:
    if "#" in expected_url:
        return expected_url in urls
    return expected_url in pages


def validate_urls(
    questions: list[dict], pages: set[str], urls: set[str]
) -> tuple[list[dict], list[tuple[str, str]]]:
    """Drop expected URLs missing from the corpus; flag questions left without any.

    Returns the questions with filtered ``expected_urls`` (and an ``invalid_urls``
    key plus ``excluded`` when nothing valid is left) and the list of problems.
    """
    problems: list[tuple[str, str]] = []
    cleaned: list[dict] = []
    for question in questions:
        expected = question.get("expected_urls", [])
        valid = []
        for url in expected:
            if url_exists(url, pages, urls):
                valid.append(url)
            else:
                problems.append((question["id"], url))
        row = dict(question)
        row["invalid_urls"] = [url for url in expected if url not in valid]
        row["expected_urls"] = valid
        row["excluded"] = bool(expected) and not valid
        cleaned.append(row)
    return cleaned, problems


def summarize(rows: list[dict], k: int = MRR_K) -> dict:
    answerable = [row for row in rows if row["answerable"] and not row.get("excluded")]
    hits = sum(1 for row in answerable if row["hit_rank"] is not None and row["hit_rank"] <= RECALL_K)
    reciprocal = sum(1.0 / row["hit_rank"] for row in answerable if row["hit_rank"] is not None)
    return {
        "n": len(answerable),
        f"recall@{RECALL_K}": round(hits / len(answerable), 4) if answerable else 0.0,
        f"mrr@{k}": round(reciprocal / len(answerable), 4) if answerable else 0.0,
        f"hits@{RECALL_K}": hits,
    }


def evaluate(
    questions: list[dict],
    k: int = MRR_K,
    policy: str | None = None,
    store=None,
    embedder=None,
    lexical_weight: float | None = None,
) -> dict:
    store = store or get_store()
    embedder = embedder or CachedEmbedder(get_embedder())
    rows: list[dict] = []
    started = time.perf_counter()

    for question in questions:
        results = hybrid_search(
            question["question"],
            k=k,
            policy=policy,
            store=store,
            embedder=embedder,
            lexical_weight=lexical_weight,
        )
        urls = [result["url"] for result in results]
        rank = first_hit_rank(urls, question.get("expected_urls", []))
        rows.append(
            {
                "id": question["id"],
                "lang": question["lang"],
                "detected_lang": detect_lang(question["question"]),
                "question": question["question"],
                "answerable": question["answerable"],
                "excluded": question.get("excluded", False),
                "expected_urls": question.get("expected_urls", []),
                "invalid_urls": question.get("invalid_urls", []),
                "hit_rank": rank,
                "top_urls": urls[:RECALL_K],
            }
        )

    elapsed = time.perf_counter() - started
    answerable = [row for row in rows if row["answerable"] and not row["excluded"]]
    metrics = summarize(rows, k)
    summary = {
        "policy": policy or DEFAULT_POLICY,
        "questions": len(rows),
        "answerable": len(answerable),
        "unanswerable": sum(1 for row in rows if not row["answerable"]),
        "excluded": sum(1 for row in rows if row["excluded"]),
        f"recall@{RECALL_K}": metrics[f"recall@{RECALL_K}"],
        f"mrr@{k}": metrics[f"mrr@{k}"],
        "seconds": round(elapsed, 2),
        "mean_ms_per_question": round(elapsed * 1000 / max(len(rows), 1), 1),
    }
    return {"summary": summary, "rows": rows}


def compare(
    questions: list[dict],
    policies: tuple[str, ...] = POLICIES,
    k: int = MRR_K,
    lexical_weight: float | None = None,
) -> list[dict]:
    """Evaluate every policy on the same questions; one row per (policy, lang)."""
    store = get_store()
    embedder = CachedEmbedder(get_embedder())
    table: list[dict] = []
    for policy in policies:
        report = evaluate(
            questions, k=k, policy=policy, store=store, embedder=embedder, lexical_weight=lexical_weight
        )
        langs = sorted({row["lang"] for row in report["rows"]})
        for lang in ["all", *langs]:
            rows = report["rows"] if lang == "all" else [row for row in report["rows"] if row["lang"] == lang]
            table.append({"policy": policy, "lang": lang, **summarize(rows, k)})
    return table


def print_compare_table(table: list[dict], label: str) -> None:
    print(f"file: {label}")
    header = f"{'policy':<18}  {'lang':<5}  {'n':>3}  {'recall@5':>9}  {'mrr@8':>7}"
    print(header)
    print("-" * len(header))
    for row in table:
        print(
            f"{row['policy']:<18}  {row['lang']:<5}  {row['n']:>3}  "
            f"{row[f'recall@{RECALL_K}']:>9.4f}  {row[f'mrr@{MRR_K}']:>7.4f}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="eval", description="Evaluate hybrid retrieval")
    parser.add_argument("--json", action="store_true", help="print the full report as JSON")
    parser.add_argument("--file", default=str(QUESTIONS_PATH), help="question set (default: eval/questions.jsonl)")
    parser.add_argument(
        "--policy",
        choices=POLICIES,
        default=DEFAULT_POLICY,
        help=f"fusion policy (default: {DEFAULT_POLICY})",
    )
    parser.add_argument("--compare", action="store_true", help="evaluate every policy and print one table")
    parser.add_argument(
        "--lexical-weight",
        type=float,
        default=None,
        help="override the lexical weight used by the downweight policies",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit with status 1 if an expected URL is missing from the corpus",
    )
    parser.add_argument(
        "--generate",
        action="store_true",
        help="run the full generation pipeline (retrieval + LLM + citation validation) and write a report",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="with --generate: use the deterministic offline model (no network, no key)",
    )
    parser.add_argument("--limit", type=int, default=None, help="with --generate: only the first N questions")
    args = parser.parse_args(argv)

    if args.mock and not args.generate:
        print("ERROR: --mock only makes sense with --generate", flush=True)
        return 2
    if args.limit is not None and not args.generate:
        print("ERROR: --limit only makes sense with --generate", flush=True)
        return 2

    llm = None
    config = None
    if args.generate:
        from generation.config import LLMNotConfigured, load_config

        config = load_config()
        if args.mock:
            from generation.mock import MockLLM

            llm = MockLLM()
        else:
            # Fail fast and clearly before loading the embedding model.
            try:
                config.require_llm()
            except LLMNotConfigured as error:
                print(f"ERROR: {error}", flush=True)
                print("       set the variables in ~/.config/grounded-rag/env, or use --generate --mock", flush=True)
                return 2

    questions = load_questions(Path(args.file))
    pages, urls = corpus_urls()
    questions, problems = validate_urls(questions, pages, urls)

    print(f"corpus: {len(pages)} pages, {len(urls)} chunk URLs")
    if problems:
        print(f"expected URLs not in the corpus: {len(problems)} (excluded from the metrics)")
        for question_id, url in problems:
            print(f"  [{question_id}] {url}")
    else:
        print("expected URLs not in the corpus: 0")
    excluded = [row["id"] for row in questions if row["excluded"]]
    if excluded:
        print(f"questions left without any valid expected URL: {len(excluded)} ({', '.join(excluded)})")
    print()
    if problems and args.strict:
        print("strict: refusing to report metrics with invalid expected URLs")
        return 1

    if args.generate:
        from eval.generate import print_summary, run_generate, write_report
        from retrieval.search import get_embedder, get_store

        report = run_generate(
            questions,
            store=get_store(),
            embedder=get_embedder(),
            llm=llm,
            config=config,
            limit=args.limit,
            log=lambda message: print(message, flush=True),
        )
        path = write_report(report, mock=args.mock)
        print()
        print_summary(report)
        print(f"report: {path}")
        return 0

    if args.compare:
        table = compare(questions, k=MRR_K, lexical_weight=args.lexical_weight)
        if args.json:
            print(json.dumps({"file": args.file, "table": table}, indent=2, ensure_ascii=False))
            return 0
        print_compare_table(table, args.file)
        return 0

    report = evaluate(questions, k=MRR_K, policy=args.policy, lexical_weight=args.lexical_weight)
    summary = report["summary"]
    rows = report["rows"]

    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0

    for row in rows:
        if not row["answerable"]:
            status = "n/a"
        elif row["excluded"]:
            status = "skipped (no valid expected URL)"
        else:
            status = "HIT" if row["hit_rank"] and row["hit_rank"] <= RECALL_K else "miss"
        rank = row["hit_rank"] if row["hit_rank"] is not None else "-"
        print(f"[{row['id']}] ({row['lang']}, detected {row['detected_lang']}) {status} rank={rank} :: {row['question']}")
        for expected in row["expected_urls"]:
            found = any(url_matches(url, expected) for url in row["top_urls"])
            print(f"    expected: {expected}  ({'in top-5' if found else 'not in top-5'})")
        for url in row["invalid_urls"]:
            print(f"    ignored (not in corpus): {url}")
        for position, url in enumerate(row["top_urls"], start=1):
            print(f"    {position}. {url}")
        print()

    print("summary")
    print(f"  policy: {summary['policy']}")
    print(f"  questions: {summary['questions']} ({summary['answerable']} answerable, {summary['unanswerable']} not)")
    if summary["excluded"]:
        print(f"  excluded: {summary['excluded']} (no valid expected URL)")
    print(f"  recall@{RECALL_K}: {summary[f'recall@{RECALL_K}']}")
    print(f"  mrr@{MRR_K}: {summary[f'mrr@{MRR_K}']}")
    print(f"  time: {summary['seconds']}s ({summary['mean_ms_per_question']} ms/question)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
