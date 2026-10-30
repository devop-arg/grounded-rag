"""Metrics for the web UI: retrieval, abstention threshold and generation.

Every block follows the index the configuration uses (the same rule as
:func:`active_threshold`): a remote embedding model means the remote index, and
the numbers come from the reports of that index, never from the other one.
Everything is read from disk on every request (reports change only when a run
regenerates them):

* local index: ``eval/reports/retrieval.json`` — the ``python -m eval --strict
  --json`` output, versioned: recall@5 / MRR@8 overall and per question language;
* remote index: the newest ``eval/reports/<date>-compare-remote.json``, read from
  the combination that comparison chose (``chosen``; the same one the "Why RAG"
  section quotes).  Its summary only measures answerable questions, so the
  per-language blocks carry recall@5 / MRR@8 and no invented question counts;
* ``generation/threshold.json`` or ``generation/threshold-remote.json``: the
  calibrated abstention threshold of the index that is active under the running
  configuration;
* ``eval/reports/<date>-generate.json`` (local) or
  ``eval/reports/<date>-generate-remote.json`` (remote) — a **real** generation
  report of that index.

The generation block never shows the simulated model's numbers: a report whose
name ends in ``-mock.json`` is counted, never summarised, so a mock run can
never be presented as a real evaluation.  Each block names its own report and
carries ``index`` (``local`` or ``remote``), so the panel can show where its
numbers come from.  The monthly budget block is added by the API endpoint,
which owns the ``UsageStore``.
"""

from __future__ import annotations

import json
from pathlib import Path

from eval.__main__ import summarize as summarize_retrieval

REPO_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = REPO_ROOT / "eval" / "reports"
RETRIEVAL_REPORT = "retrieval.json"
COMPARE_REMOTE_SUFFIX = "-compare-remote.json"
THRESHOLD_PATH = REPO_ROOT / "generation" / "threshold.json"
REMOTE_THRESHOLD_PATH = REPO_ROOT / "generation" / "threshold-remote.json"
MOCK_SUFFIX = "-generate-mock.json"
REAL_SUFFIX = "-generate.json"
REMOTE_REAL_SUFFIX = "-generate-remote.json"
WEB_DIR = REPO_ROOT / "web"


def _read_json(path: Path) -> dict | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def retrieval_metrics(path: Path | str = RETRIEVAL_REPORT) -> dict | None:
    """Recall@5 / MRR@8 overall and per language, from the eval report."""
    path = Path(path)
    data = _read_json(path)
    if data is None:
        return None
    summary = data.get("summary")
    rows = data.get("rows")
    if not isinstance(summary, dict) or not isinstance(rows, list):
        return None
    by_lang: dict[str, dict] = {}
    languages = sorted({row.get("lang") for row in rows if isinstance(row, dict) and row.get("lang")})
    for language in languages:
        selected = [row for row in rows if isinstance(row, dict) and row.get("lang") == language]
        block = summarize_retrieval(selected)
        by_lang[language] = {
            "questions": block["n"],
            "recall@5": block["recall@5"],
            "mrr@8": block["mrr@8"],
        }
    return {
        "index": "local",
        "report": path.name,
        "policy": summary.get("policy"),
        "questions": summary.get("questions"),
        "answerable": summary.get("answerable"),
        "unanswerable": summary.get("unanswerable"),
        "recall@5": summary.get("recall@5"),
        "mrr@8": summary.get("mrr@8"),
        "by_lang": by_lang,
    }


def _chosen_combination(data: dict) -> dict | None:
    """The combination the comparison chose, or the best measured one.

    ``chosen`` names it; a report without that name (or pointing at a missing or
    failed combination) falls back to the measured combination with the best
    MRR@8, which is the same rule the comparison used.  Numbers are never
    invented: a report with no usable combination simply yields nothing.
    """
    combinations = [item for item in data.get("combinations") or [] if isinstance(item, dict)]
    if not combinations:
        return None
    name = data.get("chosen")

    def summary_of(item: dict) -> dict:
        summary = item.get("summary")
        return summary if isinstance(summary, dict) else {}

    for item in combinations:
        if item.get("name") == name and isinstance(summary_of(item).get("all"), dict):
            return item
    measured = [
        item
        for item in combinations
        if item.get("status") == "ok" and isinstance(summary_of(item).get("all"), dict)
    ]
    if not measured:
        return None

    def mrr_of(item: dict) -> float:
        value = (summary_of(item).get("all") or {}).get("mrr@8")
        return float(value) if isinstance(value, (int, float)) else 0.0

    return max(measured, key=mrr_of)


def compare_retrieval_metrics(path: Path | str) -> dict | None:
    """Recall@5 / MRR@8 of the combination a ``--compare-remote`` run chose.

    The remote comparison measures the answerable questions only, so each
    per-language block carries recall@5 and MRR@8 and no question count: the
    ``n`` of the report is the answerable count, not the size of the set.
    """
    path = Path(path)
    data = _read_json(path)
    if data is None:
        return None
    combination = _chosen_combination(data)
    if combination is None:
        return None
    summary = combination.get("summary") or {}
    overall = summary.get("all")
    if not isinstance(overall, dict):
        return None
    by_lang: dict[str, dict] = {}
    for language in sorted(summary):
        block = summary[language]
        if language == "all" or not isinstance(block, dict):
            continue
        by_lang[language] = {"recall@5": block.get("recall@5"), "mrr@8": block.get("mrr@8")}
    return {
        "index": "remote",
        "report": path.name,
        "generated_at": data.get("generated_at"),
        "combination": combination.get("name"),
        "models": combination.get("models"),
        "policy": data.get("policy"),
        "questions": data.get("questions"),
        "answerable": data.get("answerable"),
        "recall@5": overall.get("recall@5"),
        "mrr@8": overall.get("mrr@8"),
        "by_lang": by_lang,
    }


def remote_retrieval_metrics(reports_dir: Path | str = REPORTS_DIR) -> dict | None:
    """The newest ``*-compare-remote.json``; nothing when the run does not exist."""
    reports_dir = Path(reports_dir)
    candidates = sorted(reports_dir.glob(f"*{COMPARE_REMOTE_SUFFIX}"))
    if not candidates:
        return None
    return compare_retrieval_metrics(candidates[-1])


def threshold_metrics(path: Path | str = THRESHOLD_PATH) -> dict | None:
    """The calibrated abstention threshold, including its known bias note."""
    data = _read_json(Path(path))
    if data is None or not isinstance(data.get("value"), (int, float)):
        return None
    return {
        "signal": data.get("signal"),
        "value": data.get("value"),
        "answerable_lost": data.get("answerable_lost"),
        "answerable_total": data.get("answerable_total"),
        "unanswerable_abstained": data.get("unanswerable_abstained"),
        "unanswerable_total": data.get("unanswerable_total"),
        "calibrated_at": data.get("calibrated_at"),
        "note": data.get("note"),
    }


def active_index(config=None) -> str:
    """``"remote"`` when the configuration embeds queries remotely, else ``"local"``.

    Same rule as :meth:`generation.generate.Thresholds.for_index`: without
    settings the local index is assumed, so a caller with no configuration gets
    the historical behaviour.
    """
    return "remote" if (config is not None and config.remote_embeddings) else "local"


def active_threshold(
    config=None,
    *,
    local_path: Path | str = THRESHOLD_PATH,
    remote_path: Path | str = REMOTE_THRESHOLD_PATH,
) -> dict | None:
    """The threshold of the index the configuration uses (never a path).

    A remote embedding model configured means the remote index, whose cosine
    lives on its own scale and has its own calibration; anything else is the
    local one.  An absent file leaves the block empty, exactly like a server
    running without a threshold.  Without ``config`` the local index is assumed,
    so a caller that has no settings gets the historical behaviour.
    """
    index = active_index(config)
    block = threshold_metrics(remote_path if index == "remote" else local_path)
    return None if block is None else {**block, "index": index}


def generation_metrics(reports_dir: Path | str = REPORTS_DIR, *, index: str = "local") -> dict:
    """The last **real** generation report of ``index``, or an explicit ``pending``.

    ``index`` picks the family of report: the local run (``<date>-generate.json``)
    or the run whose retrieval used remote parts (``<date>-generate-remote.json``).
    The other index's report is never read, so the panel cannot quote the numbers
    of an index the configuration does not use.  Mock reports are only counted
    (``simulated_reports``); their metrics are never copied into the response.
    """
    reports_dir = Path(reports_dir)
    remote = index == "remote"
    suffix = REMOTE_REAL_SUFFIX if remote else REAL_SUFFIX
    real = sorted(path for path in reports_dir.glob(f"*{suffix}") if not path.name.endswith(MOCK_SUFFIX))
    simulated = sorted(reports_dir.glob(f"*{MOCK_SUFFIX}"))
    label = "remote" if remote else "local"
    if not real:
        return {
            "index": label,
            "status": "pending",
            "reason": "no_real_report",
            "report": None,
            "simulated_reports": len(simulated),
        }
    path = real[-1]
    data = _read_json(path)
    if data is None or not isinstance(data.get("summary"), dict):
        return {
            "index": label,
            "status": "pending",
            "reason": "unreadable_report",
            "report": path.name,
            "simulated_reports": len(simulated),
        }
    summary = data["summary"]
    abstention = summary.get("abstention") or {}
    citations = summary.get("citations") or {}
    latency = (summary.get("latency_ms") or {}).get("retrieval") or {}
    return {
        "index": label,
        "status": "measured",
        "reason": None,
        "report": path.name,
        "generated_at": data.get("generated_at"),
        "questions": summary.get("questions"),
        "answerable": summary.get("answerable"),
        "answered": summary.get("answered"),
        "abstention": {
            "precision": abstention.get("precision"),
            "recall": abstention.get("recall"),
        },
        "citations": {
            "valid": citations.get("valid"),
            "total": citations.get("total"),
            "validity_rate": citations.get("validity_rate"),
        },
        "retrieval_ms": {"median": latency.get("median"), "p95": latency.get("p95")},
        "simulated_reports": len(simulated),
    }


def build_metrics(
    *,
    reports_dir: Path | str = REPORTS_DIR,
    threshold_path: Path | str = THRESHOLD_PATH,
    remote_threshold_path: Path | str = REMOTE_THRESHOLD_PATH,
    config=None,
) -> dict:
    """Everything the "How it is evaluated" panel shows, for the active index."""
    reports_dir = Path(reports_dir)
    index = active_index(config)
    retrieval = (
        remote_retrieval_metrics(reports_dir)
        if index == "remote"
        else retrieval_metrics(reports_dir / RETRIEVAL_REPORT)
    )
    return {
        "retrieval": retrieval,
        "threshold": active_threshold(
            config, local_path=threshold_path, remote_path=remote_threshold_path
        ),
        "generation": generation_metrics(reports_dir, index=index),
    }
