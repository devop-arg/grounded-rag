"""CLI: ``python -m eval`` (recall@5 and MRR over a labelled question set).

Besides the single-policy report, ``--compare`` runs every fusion policy over the
same set and prints recall@5 / MRR per policy and per question language.
``--compare-remote`` measures the optional remote retrieval parts (rerank,
remote embeddings, translation) against the local baseline and writes
``eval/reports/<date>-compare-remote.json`` (see :mod:`eval.remote_compare`).
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


def group_urls(result: dict) -> list[str]:
    """Every fragment URL of a search result, representative first.

    ``hybrid_search`` groups by page, so a result stands for several anchors; the
    group's ``urls`` holds them all.  A result without it (an older shape) is
    scored by its representative URL only.
    """
    urls = result.get("urls") or []
    return list(urls) or [result["url"]]


def first_hit(results: list[dict], expected_urls: list[str]) -> tuple[int | None, str | None]:
    """Rank of the first result holding an expected URL, plus the URL that matched.

    An expected ``#anchor`` matches when any fragment grouped into the result has
    that anchor, not only the representative; without ``#`` the page must match.
    """
    for rank, result in enumerate(results, start=1):
        for url in group_urls(result):
            if any(url_matches(url, expected) for expected in expected_urls):
                return rank, url
    return None, None


def first_hit_rank(results: list[dict], expected_urls: list[str]) -> int | None:
    return first_hit(results, expected_urls)[0]


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
        rank, hit_url = first_hit(results, question.get("expected_urls", []))
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
                "hit_url": hit_url,
                "top_urls": urls[:RECALL_K],
                "top_group_urls": [group_urls(result) for result in results[:RECALL_K]],
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


def compare_remote_main(questions: list[dict], args) -> int:
    """``--compare-remote``: every remote part against the local baseline.

    The models come from the flags, which default to the ``GROUNDED_RAG_*``
    variables of the run: nothing is hardcoded here.
    """
    from dataclasses import replace

    from eval import remote_compare
    from generation.config import load_config
    from retrieval.search import get_embedder, get_store

    config = load_config()
    overrides: dict = {}
    if args.rerank_model is not None:
        overrides["rerank_model"] = args.rerank_model
    if args.translate_model is not None:
        overrides["translate_model"] = args.translate_model
    if overrides:
        config = replace(config, **overrides)

    indexes = [name.strip() for name in (args.embed_index or "").split(",") if name.strip()]
    dims = [value.strip() for value in (args.embed_dims or "").split(",")]
    models = [
        model.strip()
        for chunk in (args.embed_model or [])
        for model in chunk.split(",")
        if model.strip()
    ]
    variants: list[tuple[str, str, int]] = []
    for position, model in enumerate(models):
        name = indexes[position] if position < len(indexes) else ""
        width = dims[position] if position < len(dims) and dims[position] else "0"
        try:
            width_value = int(width)
        except ValueError:
            print(f"ERROR: --embed-dims must be integers (got {width!r})", flush=True)
            return 2
        index_name = replace(config, embed_model=model, embed_index=name).embed_index_name
        variants.append((model, index_name, width_value))

    if not (config.rerank_model or config.translate_model or variants):
        print("ERROR: nothing to compare: pass --rerank-model, --embed-model or --translate-model", flush=True)
        print("       (or set GROUNDED_RAG_RERANK_MODEL / _EMBED_MODEL / _TRANSLATE_MODEL)", flush=True)
        return 2

    selected = questions[: args.limit] if args.limit else questions
    print(f"remote comparison over {len(selected)} questions (policy {args.policy})")
    print(f"  rerank model: {config.rerank_model.rsplit('/', 1)[-1] or '(off)'}")
    print(f"  embedding models: {', '.join(model.rsplit('/', 1)[-1] for model, _, _ in variants) or '(off)'}")
    print(f"  translation model: {config.translate_model.rsplit('/', 1)[-1] or '(off)'}")
    print(f"  prices per env: embed={config.embed_price_per_m}/M rerank={config.rerank_price_per_search}/search "
          f"translate={config.translate_price_input_per_m}/{config.translate_price_output_per_m} per M")
    print()

    report = remote_compare.run(
        selected,
        config=config,
        policy=args.policy,
        store=get_store(),
        embedder=get_embedder(),
        url_matches=url_matches,
        rerank_model=config.rerank_model,
        embeddings=variants,
        translate_model=config.translate_model,
        k=MRR_K,
        log=lambda message: print(message, flush=True),
    )
    print()
    print(remote_compare.format_table(report))
    path = remote_compare.write_report(report)
    print(f"report: {path}")
    return 0


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
        "--compare-remote",
        action="store_true",
        help="compare the local baseline against the configured remote parts and write a report",
    )
    parser.add_argument(
        "--rerank-model",
        default=None,
        help="with --compare-remote: rerank model (default: GROUNDED_RAG_RERANK_MODEL)",
    )
    parser.add_argument(
        "--embed-model",
        action="append",
        default=None,
        metavar="MODEL[,MODEL]",
        help="with --compare-remote: remote embedding model(s), repeatable or comma separated",
    )
    parser.add_argument(
        "--embed-index",
        default=None,
        metavar="NAME[,NAME]",
        help="with --compare-remote: the vector index of each --embed-model, in the same order",
    )
    parser.add_argument(
        "--embed-dims",
        default=None,
        metavar="N[,N]",
        help="with --compare-remote: output dimensions of each --embed-model (empty = model default)",
    )
    parser.add_argument(
        "--translate-model",
        default=None,
        help="with --compare-remote: translation model (default: GROUNDED_RAG_TRANSLATE_MODEL)",
    )
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
    parser.add_argument(
        "--out",
        default=None,
        help=(
            "with --generate: write the report here instead of "
            "eval/reports/<date>-generate[-remote].json"
        ),
    )
    args = parser.parse_args(argv)

    if args.mock and not args.generate:
        print("ERROR: --mock only makes sense with --generate", flush=True)
        return 2
    if args.out is not None and not args.generate:
        print("ERROR: --out only makes sense with --generate", flush=True)
        return 2
    if args.limit is not None and not (args.generate or args.compare_remote):
        print("ERROR: --limit only makes sense with --generate or --compare-remote", flush=True)
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
        from generation.generate import load_thresholds
        from retrieval.remote import build_remote
        from retrieval.search import get_embedder, get_store

        # The remote parts come from the configuration of the run (never from a
        # default here); the mock run stays fully offline.
        remote = None if args.mock else build_remote(config)
        if config.remote_active and remote is None and not args.mock:
            print("remote retrieval configured but unavailable: this run stays local", flush=True)
        remote_run = bool(remote is not None and remote.active)
        if remote_run:
            print(f"retrieval remote: {json.dumps(remote.describe(), ensure_ascii=False)}", flush=True)
            print(flush=True)

        report = run_generate(
            questions,
            store=get_store(),
            # With a usable remote index the local model is only built if a
            # remote embedding fails and the local search is the fallback.
            embedder=None if (remote_run and remote.remote_vectors) else get_embedder(),
            llm=llm,
            config=config,
            limit=args.limit,
            remote=remote,
            thresholds=load_thresholds(),
            log=lambda message: print(message, flush=True),
        )
        path = write_report(report, mock=args.mock, remote=remote_run, out=args.out)
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

    if args.compare_remote:
        return compare_remote_main(questions, args)

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
            found = any(
                url_matches(url, expected)
                for group in row["top_group_urls"]
                for url in group
            )
            print(f"    expected: {expected}  ({'in top-5' if found else 'not in top-5'})")
        for url in row["invalid_urls"]:
            print(f"    ignored (not in corpus): {url}")
        for position, url in enumerate(row["top_urls"], start=1):
            group = row["top_group_urls"][position - 1]
            extra = len(group) - 1
            suffix = f"  (+{extra} more fragment{'s' if extra != 1 else ''} of the page)" if extra else ""
            marker = "  <- matched" if url == row["hit_url"] else ""
            print(f"    {position}. {url}{suffix}{marker}")
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
