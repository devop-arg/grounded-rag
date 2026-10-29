"""CLI: ``python -m retrieval "question"``."""

from __future__ import annotations

import argparse
import json
import sys
import time

from .search import get_embedder, get_store, hybrid_search


def _print_table(query: str, results: list[dict], latency_ms: float) -> None:
    print(f'query: "{query}"')
    print(f"results: {len(results)}   latency: {latency_ms:.1f} ms")
    print()
    header = f"{'#':>2}  {'rrf':>7}  {'lex':>4}  {'vec':>4}  url"
    print(header)
    print("-" * min(len(header) + 60, 160))
    for position, result in enumerate(results, start=1):
        lex = result["lexical_rank"] if result["lexical_rank"] is not None else "-"
        vec = result["vector_rank"] if result["vector_rank"] is not None else "-"
        print(f"{position:>2}  {result['rrf']:.5f}  {str(lex):>4}  {str(vec):>4}  {result['url']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="retrieval", description="Hybrid search over PostgreSQL 17 docs")
    parser.add_argument("query", nargs="+", help="natural language question")
    parser.add_argument("-k", type=int, default=8, help="number of results (default: 8)")
    parser.add_argument("--json", action="store_true", help="print JSON instead of a table")
    args = parser.parse_args(argv)

    query = " ".join(args.query)

    started = time.perf_counter()
    store = get_store()
    embedder = get_embedder()
    load_ms = (time.perf_counter() - started) * 1000

    started = time.perf_counter()
    results = hybrid_search(query, k=args.k, store=store, embedder=embedder)
    latency_ms = (time.perf_counter() - started) * 1000

    if args.json:
        print(
            json.dumps(
                {
                    "query": query,
                    "latency_ms": round(latency_ms, 2),
                    "model_load_ms": round(load_ms, 2),
                    "results": results,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    print(f"store + model load: {load_ms:.1f} ms (excluded from latency)")
    _print_table(query, results, latency_ms)
    print()
    for position, result in enumerate(results, start=1):
        path = " > ".join(result["section_path"])
        print(f"[{position}] {result['page_title']} :: {path}")
        print(f"    {result['text'][:200]}...")
    return 0


if __name__ == "__main__":
    sys.exit(main())
