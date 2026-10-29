"""Web interface: static files behind the API, security headers, metrics, recording.

No network and no model: every report is a fake written to ``tmp_path``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from app.metrics import build_metrics, generation_metrics, retrieval_metrics
from app.main import CONTENT_SECURITY_POLICY
from web.record import build_entry, latest_real_report, report_answers

from helpers import make_client

REPO_ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = REPO_ROOT / "web"
CSP_DIRECTIVES = ("default-src 'self'", "frame-ancestors 'none'", "base-uri 'none'", "connect-src 'self'")


# --------------------------------------------------------------------- fakes
def retrieval_report() -> dict:
    rows = [
        {"id": "a01", "lang": "en", "answerable": True, "excluded": False, "hit_rank": 1},
        {"id": "a02", "lang": "es", "answerable": True, "excluded": False, "hit_rank": 3},
        {"id": "u01", "lang": "en", "answerable": False, "excluded": False, "hit_rank": None},
    ]
    return {
        "summary": {
            "policy": "downweight",
            "questions": 3,
            "answerable": 2,
            "unanswerable": 1,
            "excluded": 0,
            "recall@5": 1.0,
            "mrr@8": 0.6667,
            "seconds": 0.1,
            "mean_ms_per_question": 1.0,
        },
        "rows": rows,
    }


def generation_report(answered: int, precision: float) -> dict:
    return {
        "generated_at": "2026-10-29T00:00:00Z",
        "threshold": None,
        "config": {"llm_api_key": True},
        "summary": {
            "questions": 80,
            "answerable": 50,
            "answered": answered,
            "abstention": {"precision": precision, "recall": 0.5},
            "citations": {"valid": 60, "total": 61, "validity_rate": 0.9836},
            "latency_ms": {"retrieval": {"median": 84.2, "p95": 128.6}, "llm": {"median": 1.0, "p95": 2.0}},
        },
    }


THRESHOLD = {
    "signal": "vector_cosine",
    "value": 0.820859,
    "answerable_lost": 2,
    "answerable_total": 50,
    "unanswerable_abstained": 10,
    "unanswerable_total": 30,
    "note": "calibrated on the same eval set",
}


@pytest.fixture
def reports(tmp_path: Path) -> Path:
    directory = tmp_path / "reports"
    directory.mkdir()
    (directory / "retrieval.json").write_text(json.dumps(retrieval_report()), encoding="utf-8")
    (tmp_path / "threshold.json").write_text(json.dumps(THRESHOLD), encoding="utf-8")
    return directory


# ------------------------------------------------------------------- statics
def test_root_serves_the_page_with_the_security_headers(tmp_path) -> None:
    client, _ = make_client(tmp_path)
    with client:
        response = client.get("/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "grounded-rag" in response.text
    assert response.headers["content-security-policy"] == CONTENT_SECURITY_POLICY
    for directive in CSP_DIRECTIVES:
        assert directive in response.headers["content-security-policy"]
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["referrer-policy"] == "no-referrer"


def test_static_assets_are_served(tmp_path) -> None:
    client, _ = make_client(tmp_path)
    with client:
        script = client.get("/app.js")
        styles = client.get("/app.css")
        recorded = client.get("/recorded.json")
    assert script.status_code == 200 and "javascript" in script.headers["content-type"]
    assert styles.status_code == 200 and styles.headers["content-type"].startswith("text/css")
    assert recorded.status_code == 200 and recorded.headers["content-type"].startswith("application/json")


def test_api_routes_win_over_the_static_mount(tmp_path) -> None:
    client, _ = make_client(tmp_path)
    with client:
        health = client.get("/api/health")
        metrics = client.get("/api/metrics")
    assert health.status_code == 200
    assert health.headers["content-type"].startswith("application/json")
    assert health.headers["content-security-policy"] == CONTENT_SECURITY_POLICY
    assert metrics.status_code == 200
    assert metrics.headers["content-type"].startswith("application/json")


def test_missing_static_file_is_404(tmp_path) -> None:
    client, _ = make_client(tmp_path)
    with client:
        assert client.get("/nope.html").status_code == 404
        assert client.get("/nope/deeper.js").status_code == 404


def test_the_recorder_module_is_not_served(tmp_path) -> None:
    client, _ = make_client(tmp_path)
    with client:
        assert client.get("/record.py").status_code == 404
        assert client.get("/__pycache__/record.cpython-312.pyc").status_code == 404


def test_the_assets_are_csp_clean() -> None:
    """CSP forbids inline script, inline style and ``on*`` handlers."""
    page = (WEB_DIR / "index.html").read_text(encoding="utf-8")
    assert "<style" not in page.lower()
    assert "<script>" not in page.lower()
    assert not re.search(r'\sstyle="', page)
    assert not re.search(r"\son[a-z]+\s*=", page)
    script = (WEB_DIR / "app.js").read_text(encoding="utf-8")
    assert ".style." not in script and "setAttribute('style'" not in script


def test_the_web_switches_to_recorded_mode_when_the_budget_is_spent() -> None:
    """No JS runner here: this pins the wiring the Playwright run exercises."""
    script = (WEB_DIR / "app.js").read_text(encoding="utf-8")
    assert script.count("'banner.budget':") == 2  # English and Spanish
    assert script.count("t('banner.budget')") == 2  # the 503 path and the health path
    assert "'budget_exhausted'" in script
    assert "health.budget && health.budget.exhausted" in script
    assert "state.budgetExhausted = true" in script


# ------------------------------------------------------------------- metrics
def test_retrieval_metrics_are_read_per_language(reports: Path) -> None:
    metrics = retrieval_metrics(reports / "retrieval.json")
    assert metrics["policy"] == "downweight"
    assert metrics["recall@5"] == 1.0
    assert metrics["mrr@8"] == 0.6667
    assert metrics["by_lang"]["en"] == {"questions": 1, "recall@5": 1.0, "mrr@8": 1.0}
    assert metrics["by_lang"]["es"] == {"questions": 1, "recall@5": 1.0, "mrr@8": 0.3333}


def test_missing_retrieval_report_is_null(tmp_path, reports: Path) -> None:
    metrics = build_metrics(reports_dir=tmp_path / "empty", threshold_path=tmp_path / "threshold.json")
    assert metrics["retrieval"] is None
    assert metrics["threshold"]["value"] == 0.820859


def test_a_mock_report_is_never_presented_as_real(reports: Path) -> None:
    (reports / "2026-10-29-generate-mock.json").write_text(
        json.dumps(generation_report(48, 0.8333)), encoding="utf-8"
    )
    generation = generation_metrics(reports)
    assert generation["status"] == "pending"
    assert generation["simulated_reports"] == 1
    assert generation["report"] is None
    assert "answered" not in generation and "citations" not in generation
    dumped = json.dumps(generation)
    assert "0.8333" not in dumped and "48" not in dumped


def test_a_real_report_wins_and_the_mock_is_only_counted(reports: Path) -> None:
    (reports / "2026-10-29-generate-mock.json").write_text(
        json.dumps(generation_report(48, 0.8333)), encoding="utf-8"
    )
    (reports / "2026-10-30-generate.json").write_text(
        json.dumps(generation_report(49, 0.9)), encoding="utf-8"
    )
    generation = generation_metrics(reports)
    assert generation["status"] == "measured"
    assert generation["report"] == "2026-10-30-generate.json"
    assert generation["answered"] == 49
    assert generation["abstention"]["precision"] == 0.9
    assert generation["simulated_reports"] == 1
    assert "2026-10-29-generate-mock.json" != generation["report"]


def test_build_metrics_has_the_three_blocks(reports: Path) -> None:
    (reports / "2026-10-30-generate.json").write_text(json.dumps(generation_report(49, 0.9)), encoding="utf-8")
    metrics = build_metrics(reports_dir=reports, threshold_path=reports.parent / "threshold.json")
    assert set(metrics) == {"retrieval", "threshold", "generation"}
    assert metrics["retrieval"]["questions"] == 3
    assert metrics["threshold"]["unanswerable_abstained"] == 10
    assert metrics["generation"]["status"] == "measured"


def test_metrics_endpoint_uses_the_versioned_reports(tmp_path) -> None:
    client, _ = make_client(tmp_path)
    with client:
        body = client.get("/api/metrics").json()
    assert set(body) == {"retrieval", "threshold", "generation", "budget"}
    assert body["retrieval"]["report"] == "retrieval.json"
    assert isinstance(body["retrieval"]["recall@5"], float)
    assert body["budget"]["budget_usd"] == 0.5
    assert body["generation"]["status"] in {"pending", "measured"}
    if body["generation"]["status"] == "pending":
        assert body["generation"]["report"] is None


# ------------------------------------------------------------------ recorded
def test_recorded_json_is_valid_and_mock_free() -> None:
    document = json.loads((WEB_DIR / "recorded.json").read_text(encoding="utf-8"))
    entries = document["entries"]
    assert document["generated_at"].endswith("Z")
    assert [entry["id"] for entry in entries] == ["a01", "a15", "a16", "a06", "a33", "a42", "u01", "u30"]
    assert sum(1 for entry in entries if entry["lang"] == "en") == 4
    assert sum(1 for entry in entries if entry["lang"] == "es") == 4
    assert {entry["id"] for entry in entries} & {"u01", "u30"} == {"u01", "u30"}
    for entry in entries:
        assert entry["question"].strip()
        assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", entry["recorded_at"])
        assert entry["source"] == "retrieval-only" or entry["source"].startswith("eval-report:")
        assert "mock" not in entry["source"]
        assert entry["status"] in {"retrieval_only", "answered", "abstained"}
        assert [source["n"] for source in entry["sources"]] == list(range(1, len(entry["sources"]) + 1))
        assert all(source["url"].startswith("https://www.postgresql.org/docs/17/") for source in entry["sources"])
        assert all(source["title"] for source in entry["sources"])
        if entry["status"] == "retrieval_only":
            assert entry["answer"] is None
        else:  # pragma: no cover - only when a real report exists
            assert isinstance(entry["answer"]["sentences"], list)


def test_a_mock_report_is_never_replayed(reports: Path) -> None:
    mock = reports / "2026-10-29-generate-mock.json"
    mock.write_text(json.dumps(generation_report(48, 0.8333)), encoding="utf-8")
    assert latest_real_report(reports) is None
    assert report_answers(None) == ({}, None)
    with pytest.raises(ValueError):
        from web.record import record

        record(report_path=mock, out_path=reports / "out.json")


def test_latest_real_report_ignores_mock_files(reports: Path) -> None:
    (reports / "2026-10-29-generate-mock.json").write_text("{}", encoding="utf-8")
    real = reports / "2026-10-28-generate.json"
    real.write_text("{}", encoding="utf-8")
    assert latest_real_report(reports) == real


def test_build_entry_keeps_a_real_answer_and_otherwise_stays_retrieval_only() -> None:
    sources = [{"n": 1, "url": "https://www.postgresql.org/docs/17/x.html#s", "title": "X"}]
    question = {"id": "a01", "lang": "en", "question": "How do partial indexes work?", "answerable": True}

    retrieval_only = build_entry(question, sources, {}, None)
    assert retrieval_only["status"] == "retrieval_only"
    assert retrieval_only["source"] == "retrieval-only"
    assert retrieval_only["answer"] is None

    rows = {
        "how do partial indexes work?": {
            "abstain": False,
            "reason": None,
            "sentences": [{"text": "It restricts rows.", "citations": [{"n": 1, "quote": "a partial index"}]}],
            "sources": [{"n": 1, "url": "https://www.postgresql.org/docs/17/x.html#s", "title": "X"}],
        }
    }
    answered = build_entry(question, sources, rows, "eval-report:2026-10-30-generate.json")
    assert answered["status"] == "answered"
    assert answered["source"] == "eval-report:2026-10-30-generate.json"
    assert answered["answer"]["sentences"][0]["text"] == "It restricts rows."

    rows["how do partial indexes work?"]["abstain"] = True
    assert build_entry(question, sources, rows, "eval-report:x.json")["status"] == "abstained"
