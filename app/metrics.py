"""Metrics for the web UI: retrieval, abstention threshold and generation.

Three sources, all read from disk on every request (they change only when a
report is regenerated):

* ``eval/reports/retrieval.json`` — the ``python -m eval --strict --json``
  output, versioned: recall@5 / MRR@8 overall and per question language;
* ``generation/threshold.json`` — the calibrated abstention threshold;
* ``eval/reports/<date>-generate.json`` — a **real** generation report.

The generation block never shows the simulated model's numbers: a report whose
name ends in ``-mock.json`` is counted, never summarised, so a mock run can
never be presented as a real evaluation.  The monthly budget block is added by
the API endpoint, which owns the ``UsageStore``.
"""

from __future__ import annotations

import json
from pathlib import Path

from eval.__main__ import summarize as summarize_retrieval

REPO_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = REPO_ROOT / "eval" / "reports"
RETRIEVAL_REPORT = "retrieval.json"
THRESHOLD_PATH = REPO_ROOT / "generation" / "threshold.json"
MOCK_SUFFIX = "-generate-mock.json"
REAL_SUFFIX = "-generate.json"
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
        "report": path.name,
        "policy": summary.get("policy"),
        "questions": summary.get("questions"),
        "answerable": summary.get("answerable"),
        "unanswerable": summary.get("unanswerable"),
        "recall@5": summary.get("recall@5"),
        "mrr@8": summary.get("mrr@8"),
        "by_lang": by_lang,
    }


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


def generation_metrics(reports_dir: Path | str = REPORTS_DIR) -> dict:
    """The last **real** generation report, or an explicit ``pending`` status.

    Mock reports are only counted (``simulated_reports``); their metrics are
    never copied into the response.
    """
    reports_dir = Path(reports_dir)
    real = sorted(path for path in reports_dir.glob(f"*{REAL_SUFFIX}") if not path.name.endswith(MOCK_SUFFIX))
    simulated = sorted(reports_dir.glob(f"*{MOCK_SUFFIX}"))
    if not real:
        return {
            "status": "pending",
            "reason": "no_real_report",
            "report": None,
            "simulated_reports": len(simulated),
        }
    path = real[-1]
    data = _read_json(path)
    if data is None or not isinstance(data.get("summary"), dict):
        return {
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
) -> dict:
    """Everything the "How it is evaluated" panel shows."""
    reports_dir = Path(reports_dir)
    return {
        "retrieval": retrieval_metrics(reports_dir / RETRIEVAL_REPORT),
        "threshold": threshold_metrics(threshold_path),
        "generation": generation_metrics(reports_dir),
    }
