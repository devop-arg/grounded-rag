"""``python -m eval --generate``: grounded answer metrics over a question set.

Runs the full pipeline (retrieval -> threshold -> model -> citation validation)
one question at a time (concurrency 1) and writes a versioned JSON report with
the aggregate metrics plus one row per question.  No API key, header or secret
reaches the report.
"""

from __future__ import annotations

import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

from generation.config import Config
from generation.generate import answer_question, load_threshold
from generation.llm import ChatModel

REPORTS_DIR = Path(__file__).resolve().parent / "reports"
K = 8


def percentile(values: list[float], fraction: float) -> float:
    """Linear-interpolated percentile (``fraction`` in ``[0, 1]``)."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] + (ordered[upper] - ordered[lower]) * weight)


def ratio(numerator: float, denominator: float) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


def summarize_rows(rows: list[dict]) -> dict:
    """Aggregate metrics for a list of per-question rows."""
    answerable = [row for row in rows if row["answerable"]]
    unanswerable = [row for row in rows if not row["answerable"]]
    abstained = [row for row in rows if row["abstain"]]
    true_positive = sum(1 for row in abstained if not row["answerable"])
    false_positive = sum(1 for row in abstained if row["answerable"])
    false_negative = sum(1 for row in unanswerable if not row["abstain"])

    retrieval = [row["retrieval_ms"] for row in rows]
    llm = [row["llm_ms"] for row in rows if row["llm_ms"]]
    tokens = [row["tokens"] for row in rows]

    reasons: dict[str, int] = {}
    for row in abstained:
        reasons[row["reason"] or "unknown"] = reasons.get(row["reason"] or "unknown", 0) + 1

    return {
        "questions": len(rows),
        "answerable": len(answerable),
        "unanswerable": len(unanswerable),
        "answered": sum(1 for row in answerable if not row["abstain"]),
        "answer_rate_answerable": ratio(sum(1 for row in answerable if not row["abstain"]), len(answerable)),
        "abstention": {
            "abstained_total": len(abstained),
            "precision": ratio(true_positive, true_positive + false_positive),
            "recall": ratio(true_positive, true_positive + false_negative),
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "reasons": reasons,
        },
        "citations": {
            "valid": sum(row["citations_valid"] for row in rows),
            "total": sum(row["citations_total"] for row in rows),
            "validity_rate": ratio(
                sum(row["citations_valid"] for row in rows), sum(row["citations_total"] for row in rows)
            ),
            "sentences_total": sum(row["sentences_total"] for row in rows),
            "sentences_with_valid_citation": sum(row["sentences_with_valid"] for row in rows),
            "sentence_citation_rate": ratio(
                sum(row["sentences_with_valid"] for row in rows), sum(row["sentences_total"] for row in rows)
            ),
        },
        "tokens": {
            "total": sum(tokens),
            "mean_per_question": round(sum(tokens) / len(tokens), 1) if tokens else 0.0,
        },
        "latency_ms": {
            "retrieval": {
                "median": round(percentile(retrieval, 0.5), 1),
                "p95": round(percentile(retrieval, 0.95), 1),
            },
            "llm": {
                "median": round(percentile(llm, 0.5), 1),
                "p95": round(percentile(llm, 0.95), 1),
            },
        },
    }


def run_generate(
    questions: list[dict],
    *,
    store,
    embedder,
    llm: ChatModel,
    config: Config,
    limit: int | None = None,
    log=None,
) -> dict:
    """Answer every question (concurrency 1) and return the report dict."""
    threshold = load_threshold()
    selected = questions[:limit] if limit else questions
    rows: list[dict] = []
    started = time.perf_counter()
    for position, question in enumerate(selected, start=1):
        answer = answer_question(
            question["question"],
            question.get("lang"),
            config=config,
            store=store,
            embedder=embedder,
            llm=llm,
            k=K,
            threshold=threshold,
            use_threshold=threshold is not None,
        )
        row = {
            "id": question["id"],
            "lang": question["lang"],
            "question": question["question"],
            "answerable": bool(question["answerable"]),
            "abstain": bool(answer["abstain"]),
            "reason": answer["reason"],
            "outcome": answer.get("outcome"),
            "n_sentences": len(answer["sentences"]),
            "citations_valid": answer["stats"]["citations_valid"],
            "citations_total": answer["stats"]["citations_total"],
            "sentences_total": answer["stats"]["sentences_total"],
            "sentences_with_valid": answer["stats"]["sentences_with_valid_citation"],
            "prompt_tokens": answer["prompt_tokens"],
            "completion_tokens": answer["completion_tokens"],
            "tokens": answer["prompt_tokens"] + answer["completion_tokens"],
            "retrieval_ms": answer["retrieval_ms"],
            "llm_ms": answer["llm_ms"],
            "signals": {name: round(value, 6) for name, value in answer["signals"].items()},
            "cited_urls": [source["url"] for source in answer["sources"]],
            # Kept per question so ``web.record`` can replay a real answer
            # (never a mock one) without calling the model again.
            "sentences": [
                {
                    "text": sentence["text"],
                    "citations": [
                        {"n": int(citation["n"]), "quote": citation["quote"]}
                        for citation in sentence["citations"]
                    ],
                }
                for sentence in answer["sentences"]
            ],
            "sources": [
                {"n": int(source["n"]), "url": source["url"], "title": source["title"]}
                for source in answer["sources"]
            ],
        }
        rows.append(row)
        if log:
            log(
                f"[{position:>2}/{len(selected)}] {row['id']:<4} {row['lang']} "
                f"abstain={str(row['abstain']):<5} reason={str(row['reason']):<12} "
                f"sentences={row['n_sentences']} citations={row['citations_valid']}/{row['citations_total']} "
                f"tokens={row['tokens']}"
            )

    languages = sorted({row["lang"] for row in rows})
    return {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "seconds": round(time.perf_counter() - started, 1),
        "threshold": threshold,
        "config": config.public(),
        "summary": summarize_rows(rows),
        "by_lang": {lang: summarize_rows([row for row in rows if row["lang"] == lang]) for lang in languages},
        "questions": rows,
    }


def report_path(when: str | None = None, mock: bool = False) -> Path:
    stamp = when or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    suffix = "-generate-mock.json" if mock else "-generate.json"
    return REPORTS_DIR / f"{stamp}{suffix}"


def write_report(report: dict, mock: bool = False) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = report_path(mock=mock)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def print_summary(report: dict) -> None:
    summary = report["summary"]
    abstention = summary["abstention"]
    citations = summary["citations"]
    tokens = summary["tokens"]
    latency = summary["latency_ms"]

    print("summary")
    print(f"  questions: {summary['questions']} ({summary['answerable']} answerable, {summary['unanswerable']} not)")
    print(f"  answered (answerable): {summary['answered']}/{summary['answerable']} ({summary['answer_rate_answerable']})")
    print(
        f"  abstention: precision={abstention['precision']} recall={abstention['recall']} "
        f"(tp={abstention['true_positive']} fp={abstention['false_positive']} fn={abstention['false_negative']})"
    )
    print(f"  abstain reasons: {abstention['reasons']}")
    print(
        f"  citations: {citations['valid']}/{citations['total']} valid ({citations['validity_rate']}); "
        f"sentences with a valid citation: {citations['sentences_with_valid_citation']}/{citations['sentences_total']} "
        f"({citations['sentence_citation_rate']})"
    )
    print(f"  tokens: total={tokens['total']} mean/question={tokens['mean_per_question']}")
    print(
        f"  latency: retrieval median={latency['retrieval']['median']}ms p95={latency['retrieval']['p95']}ms; "
        f"llm median={latency['llm']['median']}ms p95={latency['llm']['p95']}ms"
    )
    for lang in sorted(report["by_lang"]):
        block = report["by_lang"][lang]["abstention"]
        print(
            f"  [{lang}] answer rate={report['by_lang'][lang]['answer_rate_answerable']} "
            f"abstain precision={block['precision']} recall={block['recall']}"
        )
