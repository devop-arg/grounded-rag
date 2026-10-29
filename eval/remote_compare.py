"""``python -m eval --compare-remote``: local retrieval against the remote parts.

Every combination is measured on the same questions with the same fusion policy
and the same ``k``: the local baseline, the baseline plus one remote part
(rerank, each embedding index, translation) and — when more than one part is
worth keeping — the combination of the winners.

The models are **never** chosen here: they come from the command line or from
the ``GROUNDED_RAG_*`` variables of the run.  The report keeps only the part of
each model name after the last ``/``, no endpoint and no path, and one block per
combination with recall@5 / MRR@8 (all and per language), the per-question
latency (median and p95), the remote call counters and the USD cost the endpoint
reported (or the configured prices when it reported none).
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from retrieval.lang import detect_lang
from retrieval.remote import (
    RemoteEmbeddings,
    RemoteReranker,
    RemoteRetrieval,
    RemoteStats,
    RemoteTranslator,
    short_model,
)
from retrieval.search import hybrid_search
from retrieval.store import VectorIndex, vector_ids_path_for, vectors_path_for

REPORTS_DIR = Path(__file__).resolve().parent / "reports"
RECALL_K = 5
MRR_K = 8
BASE_NAME = "local base"
#: Name of the combination that keeps every part worth keeping.
COMBINED_NAME = "best combination"


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return float(ordered[lower] + (ordered[upper] - ordered[lower]) * weight)


def summarize(rows: list[dict], recall_k: int = RECALL_K, mrr_k: int = MRR_K) -> dict:
    answerable = [row for row in rows if row["answerable"] and not row.get("excluded")]
    hits = sum(1 for row in answerable if row["hit_rank"] is not None and row["hit_rank"] <= recall_k)
    reciprocal = sum(1.0 / row["hit_rank"] for row in answerable if row["hit_rank"] is not None)
    return {
        "n": len(answerable),
        f"recall@{recall_k}": round(hits / len(answerable), 4) if answerable else 0.0,
        f"mrr@{mrr_k}": round(reciprocal / len(answerable), 4) if answerable else 0.0,
        f"hits@{recall_k}": hits,
    }


def first_hit(results: list[dict], expected_urls: list[str], url_matches) -> tuple[int | None, str | None]:
    for rank, result in enumerate(results, start=1):
        for url in result.get("urls") or [result["url"]]:
            if any(url_matches(url, expected) for expected in expected_urls):
                return rank, url
    return None, None


def make_remote(
    *,
    config,
    rerank_model: str = "",
    embed: tuple[str, str, int] | None = None,
    translate_model: str = "",
    data_dir: Path | str | None = None,
) -> RemoteRetrieval | None:
    """The parts named here, on the configured endpoint and key (or ``None``).

    ``embed`` is ``(model, index name, dimensions)``; a missing index or an
    unresolvable key returns ``None`` instead of a half-configured client.
    """
    from generation.config import LLMNotConfigured, resolve_llm_api_key
    from retrieval.remote import TIMEOUT_S

    if not (rerank_model or embed or translate_model):
        return None
    try:
        api_key = resolve_llm_api_key(config)
    except LLMNotConfigured:
        return None
    common = dict(
        base_url=config.llm_base_url,
        api_key=api_key,
        timeout=TIMEOUT_S,
        referer=config.llm_referer,
        title=config.llm_title,
    )
    embeddings = None
    index = None
    if embed:
        model, name, dims = embed
        embeddings = RemoteEmbeddings(
            model=model,
            price_per_m=config.embed_price_per_m,
            dimensions=dims or None,
            **common,
        )
        base_dir = Path(data_dir) if data_dir is not None else vectors_path_for(name).parent
        index = VectorIndex(vectors_path_for(name, base_dir), vector_ids_path_for(name, base_dir))
        if not index.exists():
            embeddings, index = None, None
    reranker = (
        RemoteReranker(model=rerank_model, price_per_search=config.rerank_price_per_search, **common)
        if rerank_model
        else None
    )
    translator = (
        RemoteTranslator(
            model=translate_model,
            price_input_per_m=config.translate_price_input_per_m,
            price_output_per_m=config.translate_price_output_per_m,
            **common,
        )
        if translate_model
        else None
    )
    if not (embeddings or reranker or translator):
        return None
    return RemoteRetrieval(embeddings=embeddings, reranker=reranker, translator=translator, index=index)


def models_of(remote: RemoteRetrieval | None) -> dict[str, str]:
    """The public model names of a combination (last path segment only)."""
    if remote is None:
        return {"embeddings": "", "rerank": "", "translate": "", "index": ""}
    return {
        "embeddings": short_model(remote.embeddings.model) if remote.embeddings else "",
        "rerank": short_model(remote.reranker.model) if remote.reranker else "",
        "translate": short_model(remote.translator.model) if remote.translator else "",
        "index": remote.index.vectors_path.stem.removeprefix("vectors-") if remote.index else "",
    }


def run_combination(
    name: str,
    questions: list[dict],
    *,
    store,
    embedder,
    remote: RemoteRetrieval | None,
    policy: str,
    url_matches: Callable[[str, str], bool],
    k: int = MRR_K,
    log=None,
) -> dict:
    """Measure one combination over every question and return its report block."""
    stats = RemoteStats()
    rows: list[dict] = []
    latencies: list[float] = []
    for question in questions:
        started = time.perf_counter()
        results = hybrid_search(
            question["question"],
            k=k,
            policy=policy,
            store=store,
            embedder=embedder,
            remote=remote,
            stats=stats,
        )
        latencies.append((time.perf_counter() - started) * 1000)
        rank, _ = first_hit(results, question.get("expected_urls", []), url_matches)
        rows.append(
            {
                "id": question["id"],
                "lang": question.get("lang"),
                "detected_lang": detect_lang(question["question"]),
                "answerable": question["answerable"],
                "excluded": question.get("excluded", False),
                "hit_rank": rank,
            }
        )
        if log is not None:
            log(f"  [{name}] {question['id']:<4} rank={rank if rank is not None else '-'}")

    languages = sorted({row["lang"] for row in rows if row.get("lang")})
    summary = {"all": summarize(rows, mrr_k=k)}
    for lang in languages:
        summary[lang] = summarize([row for row in rows if row.get("lang") == lang], mrr_k=k)
    remote_block = stats.as_dict()
    status = "ok"
    if remote is not None and remote_block["calls"] and remote_block["errors"] == remote_block["calls"]:
        status = "unavailable (every remote call failed)"
    return {
        "name": name,
        "models": models_of(remote),
        "status": status,
        "summary": summary,
        "latency_ms": {
            "median": round(percentile(latencies, 0.5), 1),
            "p95": round(percentile(latencies, 0.95), 1),
        },
        "remote": remote_block,
        "cost_usd": round(stats.cost_usd, 6),
    }


def improved(candidate: dict, base: dict) -> bool:
    """A part is worth keeping when it raises the MRR without lowering recall@5."""
    return (
        candidate["status"] == "ok"
        and candidate["summary"]["all"][f"recall@{RECALL_K}"] >= base["summary"]["all"][f"recall@{RECALL_K}"]
        and candidate["summary"]["all"][f"mrr@{MRR_K}"] > base["summary"]["all"][f"mrr@{MRR_K}"]
    )


def choose(combinations: list[dict]) -> tuple[dict, str]:
    """The combination with the best MRR that does not lower recall@5."""
    base = next(item for item in combinations if item["name"] == BASE_NAME)
    base_recall = base["summary"]["all"][f"recall@{RECALL_K}"]
    best = base
    for item in combinations:
        if item is base or item["status"] != "ok":
            continue
        if item["summary"]["all"][f"recall@{RECALL_K}"] < base_recall:
            continue
        if item["summary"]["all"][f"mrr@{MRR_K}"] > best["summary"]["all"][f"mrr@{MRR_K}"]:
            best = item
    if best is base:
        return base, "no remote part raised the MRR without lowering recall@5: every part stays off"
    return best, (
        f"{best['name']} has the best MRR@{MRR_K} among the combinations that do not lower"
        f" recall@{RECALL_K} against the local baseline"
    )


def format_table(report: dict) -> str:
    header = (
        f"{'combination':<30}  {'lang':<5}  {'n':>3}  {'recall@5':>9}  {'mrr@8':>7}"
        f"  {'p50 ms':>8}  {'p95 ms':>8}  {'errors':>6}  {'cost usd':>9}"
    )
    lines = [header, "-" * len(header)]
    for combination in report["combinations"]:
        latency = combination["latency_ms"]
        for lang, block in combination["summary"].items():
            lines.append(
                f"{combination['name']:<30}  {lang:<5}  {block['n']:>3}  "
                f"{block[f'recall@{RECALL_K}']:>9.4f}  {block[f'mrr@{MRR_K}']:>7.4f}  "
                f"{latency['median']:>8.1f}  {latency['p95']:>8.1f}  "
                f"{combination['remote']['errors']:>6}  {combination['cost_usd']:>9.6f}"
            )
    lines.append("")
    lines.append(f"chosen: {report['chosen']}  ({report['reason']})")
    lines.append(f"total cost of the run: ${report['cost_usd']:.6f}")
    return "\n".join(lines)


def report_path(when: str | None = None, reports_dir: Path | str = REPORTS_DIR) -> Path:
    stamp = when or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return Path(reports_dir) / f"{stamp}-compare-remote.json"


def write_report(report: dict, reports_dir: Path | str = REPORTS_DIR) -> Path:
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = report_path(reports_dir=reports_dir)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def run(
    questions: list[dict],
    *,
    config,
    policy: str,
    store,
    embedder,
    url_matches: Callable[[str, str], bool],
    rerank_model: str = "",
    embeddings: list[tuple[str, str, int]] | None = None,
    translate_model: str = "",
    data_dir: Path | str | None = None,
    k: int = MRR_K,
    factory: Callable[..., RemoteRetrieval | None] | None = None,
    log=None,
) -> dict:
    """Run every combination (local base, one part each, the winners) and return the report.

    ``factory`` builds a combination from keyword arguments; it defaults to
    :func:`make_remote` and is injectable so the pipeline can be tested without
    a key or a network.
    """
    build = factory or make_remote
    embed_variants = list(embeddings or [])
    parts: list[tuple[str, dict[str, Any]]] = []
    if rerank_model:
        parts.append(("+rerank", {"rerank_model": rerank_model}))
    for model, name, dims in embed_variants:
        label = f"+embeddings {short_model(model)}" + (f"@{dims}" if dims else "")
        parts.append((label, {"embed": (model, name, dims)}))
    if translate_model:
        parts.append(("+translation", {"translate_model": translate_model}))

    combinations: list[dict] = []

    def measure(name: str, remote: RemoteRetrieval | None) -> dict:
        if log is not None:
            log(f"combination {name}: {json.dumps(models_of(remote), ensure_ascii=False)}")
        block = run_combination(
            name,
            questions,
            store=store,
            embedder=embedder,
            remote=remote,
            policy=policy,
            url_matches=url_matches,
            k=k,
            log=log,
        )
        combinations.append(block)
        return block

    base = measure(BASE_NAME, None)

    winners: list[tuple[str, dict[str, Any], dict]] = []
    for label, kwargs in parts:
        remote = build(config=config, data_dir=data_dir, **kwargs)
        if remote is None:
            if log is not None:
                embed = kwargs.get("embed") or ("", "", 0)
                log(f"combination {label}: skipped (no index {embed[1]!r} or no key)")
            continue
        block = measure(label, remote)
        if improved(block, base):
            winners.append((label, kwargs, block))

    reason = "no remote part raised the MRR without lowering recall@5: every part stays off"
    if winners:
        if len(winners) == 1:
            reason = f"only {winners[0][0]} raised the MRR@{MRR_K} without lowering recall@{RECALL_K}"
        # One vector index only: two embedding variants are alternatives, not
        # additions, so the combined run keeps the one with the best MRR.
        merged: dict[str, Any] = {}
        best_embed: tuple[str, dict[str, Any], dict] | None = None
        for label, kwargs, block in winners:
            if "embed" in kwargs:
                if best_embed is None or (
                    block["summary"]["all"][f"mrr@{MRR_K}"] > best_embed[2]["summary"]["all"][f"mrr@{MRR_K}"]
                ):
                    best_embed = (label, kwargs, block)
            else:
                merged.update(kwargs)
        if best_embed is not None:
            merged.update(best_embed[1])
        combined = measure(COMBINED_NAME, build(config=config, data_dir=data_dir, **merged))
        if len(winners) > 1 and combined["status"] == "ok":
            reason = (
                f"{COMBINED_NAME} keeps every part that raised the MRR (one embedding index)"
                f" and keeps recall@{RECALL_K}"
            )

    chosen, chosen_reason = choose(combinations)
    return {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "questions": len(questions),
        "answerable": sum(1 for question in questions if question["answerable"]),
        "policy": policy,
        "k": k,
        "models": {
            "rerank": short_model(rerank_model) if rerank_model else "",
            "embeddings": [
                short_model(model) + (f"@{dims}" if dims else "") for model, _, dims in embed_variants
            ],
            "translate": short_model(translate_model) if translate_model else "",
        },
        "combinations": combinations,
        "chosen": chosen["name"],
        "chosen_models": chosen["models"],
        "reason": chosen_reason if not winners else f"{chosen_reason}; {reason}",
        "cost_usd": round(sum(item["cost_usd"] for item in combinations), 6),
    }
