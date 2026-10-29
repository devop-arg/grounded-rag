"""CLI: ``python -m retrieval "question"``."""

from __future__ import annotations

import argparse
import json
import sys
import time

from .search import DEFAULT_POLICY, POLICIES, get_embedder, get_store, hybrid_search, resolve_policy


def _print_table(
    query: str, results: list[dict], latency_ms: float, lang: str, policy: str, mode: str, stopwords: bool
) -> None:
    print(f'query: "{query}"')
    print(f"lang: {lang}   policy: {policy}   lexical index: {mode}   stopwords dropped: {'yes' if stopwords else 'no'}")
    print(f"results: {len(results)}   latency: {latency_ms:.1f} ms")
    print()
    header = f"{'#':>2}  {'rrf':>7}  {'lex':>4}  {'vec':>4}  {'frags':>5}  url"
    print(header)
    print("-" * min(len(header) + 60, 160))
    for position, result in enumerate(results, start=1):
        lex = result["lexical_rank"] if result["lexical_rank"] is not None else "-"
        vec = result["vector_rank"] if result["vector_rank"] is not None else "-"
        print(
            f"{position:>2}  {result['rrf']:.5f}  {str(lex):>4}  {str(vec):>4}  "
            f"{result['n_chunks']:>5}  {result['url']}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="retrieval", description="Hybrid search over PostgreSQL 17 docs")
    parser.add_argument("query", nargs="+", help="natural language question")
    parser.add_argument("-k", type=int, default=8, help="number of results (default: 8)")
    parser.add_argument(
        "--policy",
        choices=POLICIES,
        default=DEFAULT_POLICY,
        help=f"lexical fusion policy (default: {DEFAULT_POLICY})",
    )
    parser.add_argument("--json", action="store_true", help="print JSON instead of a table")
    parser.add_argument(
        "--remote",
        action="store_true",
        help="use the configured remote parts (rerank, embeddings, translation) instead of the local ones",
    )
    args = parser.parse_args(argv)

    query = " ".join(args.query)
    config = resolve_policy(query, args.policy)

    remote = None
    if args.remote:
        from generation.config import load_config
        from .remote import build_remote

        remote = build_remote(load_config())
        if remote is None:
            print("remote: nothing configured (the local path is used)", flush=True)
        else:
            print(f"remote: {json.dumps(remote.describe())}", flush=True)

    started = time.perf_counter()
    store = get_store()
    embedder = None if (remote is not None and remote.remote_vectors) else get_embedder()
    load_ms = (time.perf_counter() - started) * 1000

    started = time.perf_counter()
    results = hybrid_search(
        query, k=args.k, policy=args.policy, store=store, embedder=embedder, remote=remote
    )
    latency_ms = (time.perf_counter() - started) * 1000
    if remote is not None:
        print(f"remote stats: {json.dumps(remote.stats.as_dict())}", flush=True)

    if args.json:
        print(
            json.dumps(
                {
                    "query": query,
                    "lang": config.lang,
                    "policy": args.policy,
                    "lexical_mode": config.lexical_mode,
                    "weights": config.weights,
                    "drop_stopwords": config.drop_stopwords,
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
    _print_table(query, results, latency_ms, config.lang, args.policy, config.lexical_mode, config.drop_stopwords)
    print()
    for position, result in enumerate(results, start=1):
        path = " > ".join(result["section_path"])
        print(f"[{position}] {result['page_title']} :: {path}")
        print(f"    grouped fragments: {result['n_chunks']} ({', '.join(result['chunk_ids'])})")
        print(f"    {result['text'][:200]}...")
    return 0


if __name__ == "__main__":
    sys.exit(main())
