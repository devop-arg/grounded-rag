"""Generation report: metrics, per-question rows, file naming, secret-free output."""

from __future__ import annotations

import json

import pytest

import eval.generate as eval_generate
from eval.generate import (
    percentile,
    public_config,
    ratio,
    refresh_report,
    report_path,
    run_generate,
    summarize_rows,
    write_report,
)
from eval.__main__ import main as eval_main
from eval.sanitize import main as sanitize_main

from helpers import FakeEmbedder, FakeStore, ScriptedLLM, make_settings

GOOD_QUOTE = "A partial index is an index built over a subset of a table"
QUESTIONS = [
    {"id": "a01", "lang": "en", "question": "How do partial indexes work?", "answerable": True},
    {"id": "u01", "lang": "es", "question": "¿Cuál es el mejor restaurante?", "answerable": False},
]


def test_percentile_interpolates() -> None:
    values = [10, 20, 30, 40]
    assert percentile(values, 0.5) == 25.0
    assert percentile(values, 0.0) == 10
    assert percentile(values, 1.0) == 40
    assert percentile([], 0.5) == 0.0
    assert percentile([7], 0.95) == 7.0


def test_ratio_handles_zero_denominator() -> None:
    assert ratio(1, 2) == 0.5
    assert ratio(0, 0) is None


def test_summarize_rows_computes_abstention_and_citations() -> None:
    rows = [
        {
            "answerable": True,
            "abstain": False,
            "reason": None,
            "citations_valid": 2,
            "citations_total": 2,
            "sentences_total": 2,
            "sentences_with_valid": 2,
            "prompt_tokens": 80,
            "completion_tokens": 20,
            "tokens": 100,
            "retrieval_ms": 10.0,
            "llm_ms": 200.0,
        },
        {
            "answerable": True,
            "abstain": True,
            "reason": "threshold",
            "citations_valid": 0,
            "citations_total": 1,
            "sentences_total": 1,
            "sentences_with_valid": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "tokens": 0,
            "retrieval_ms": 12.0,
            "llm_ms": 0.0,
        },
        {
            "answerable": False,
            "abstain": True,
            "reason": "model",
            "citations_valid": 0,
            "citations_total": 0,
            "sentences_total": 0,
            "sentences_with_valid": 0,
            "prompt_tokens": 40,
            "completion_tokens": 10,
            "tokens": 50,
            "retrieval_ms": 11.0,
            "llm_ms": 100.0,
        },
        {
            "answerable": False,
            "abstain": False,
            "reason": None,
            "citations_valid": 1,
            "citations_total": 1,
            "sentences_total": 1,
            "sentences_with_valid": 1,
            "prompt_tokens": 40,
            "completion_tokens": 10,
            "tokens": 50,
            "retrieval_ms": 13.0,
            "llm_ms": 120.0,
        },
    ]
    summary = summarize_rows(rows, (1.0, 2.0))
    assert summary["answerable"] == 2 and summary["unanswerable"] == 2
    assert summary["answer_rate_answerable"] == 0.5
    assert summary["abstention"] == {
        "abstained_total": 2,
        "precision": 0.5,
        "recall": 0.5,
        "true_positive": 1,
        "false_positive": 1,
        "false_negative": 1,
        "reasons": {"threshold": 1, "model": 1},
    }
    assert summary["citations"]["validity_rate"] == 0.75
    assert summary["citations"]["sentence_citation_rate"] == round(3 / 4, 4)
    assert summary["tokens"]["mean_per_question"] == 50.0
    assert (summary["tokens"]["prompt"], summary["tokens"]["completion"]) == (160, 40)
    assert summary["tokens"]["total"] == 200
    assert summary["cost_usd"] == round((160 * 1.0 + 40 * 2.0) / 1_000_000, 6)
    assert summary["latency_ms"]["llm"]["median"] == 120.0
    assert summary["latency_ms"]["retrieval"]["median"] == 11.5


def test_report_paths_and_mock_suffix(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(eval_generate, "REPORTS_DIR", tmp_path)
    assert report_path("2026-10-28").name == "2026-10-28-generate.json"
    assert report_path("2026-10-28", mock=True).name == "2026-10-28-generate-mock.json"
    path = write_report({"a": 1}, mock=True)
    assert path == tmp_path / report_path(mock=True).name
    assert json.loads(path.read_text()) == {"a": 1}


def test_run_generate_builds_rows_without_secrets(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(eval_generate, "REPORTS_DIR", tmp_path)
    monkeypatch.setattr(eval_generate, "load_threshold", lambda *a, **k: None)
    llm = ScriptedLLM(
        [
            {"abstain": False, "sentences": [{"text": "Ok.", "citations": [{"n": 1, "quote": GOOD_QUOTE}]}]},
            {"abstain": True, "sentences": []},
        ]
    )
    settings = make_settings(tmp_path, llm_api_key="sk-hidden-value", llm_model="m", llm_base_url="u")
    report = run_generate(
        QUESTIONS,
        store=FakeStore(),
        embedder=FakeEmbedder(),
        llm=llm,
        config=settings,
    )
    assert report["summary"]["questions"] == 2
    rows = report["questions"]
    assert rows[0]["id"] == "a01" and rows[0]["abstain"] is False
    assert rows[0]["cited_urls"] == ["https://www.postgresql.org/docs/17/indexes-partial.html"]
    assert rows[1]["reason"] == "model"
    assert report["config"] == {
        "llm_model": "m",
        "llm_price_input_per_m": 0.0,
        "llm_price_output_per_m": 0.0,
        "budget_usd_per_month": 0.5,
        "generation": True,
    }
    # Two scripted completions of 100 + 20 tokens, prices at zero.
    assert report["summary"]["tokens"] == {
        "prompt": 200,
        "completion": 40,
        "total": 240,
        "mean_per_question": 120.0,
    }
    assert report["summary"]["cost_usd"] == 0.0
    assert report["by_lang"]["en"]["tokens"]["prompt"] == 100
    assert report["by_lang"]["es"]["tokens"]["completion"] == 20
    dumped = json.dumps(report)
    assert "sk-hidden-value" not in dumped
    assert "Authorization" not in dumped
    assert "llm_base_url" not in dumped  # the endpoint is not published either
    assert set(report["config"]) == {"llm_model", "llm_price_input_per_m", "llm_price_output_per_m", "budget_usd_per_month", "generation"}


def test_run_generate_honours_the_limit(tmp_path) -> None:
    llm = ScriptedLLM([{"abstain": True, "sentences": []}])
    settings = make_settings(tmp_path)
    report = run_generate(
        QUESTIONS,
        store=FakeStore(),
        embedder=FakeEmbedder(),
        llm=llm,
        config=settings,
        limit=1,
    )
    assert report["summary"]["questions"] == 1
    assert [row["id"] for row in report["questions"]] == ["a01"]


def test_public_config_drops_the_host_the_paths_and_the_model_route() -> None:
    raw = {
        "llm_base_url": "https://provider.example/api/v1",
        "llm_api_key": True,
        "llm_key_file": "/somewhere/auth.json",
        "llm_key_path": "provider.key",
        "llm_model": "deepseek/deepseek-v4.1-flash",
        "llm_price_input_per_m": 0.045,
        "llm_price_output_per_m": 0.5,
        "budget_usd_per_month": 0.5,
        "generation": True,
        "rate_per_ip": "10/hour",
        "ip_salt": True,
        "port": 8790,
        "usage_db": "/somewhere/usage.sqlite",
    }
    assert public_config(raw) == {
        "llm_model": "deepseek-v4.1-flash",
        "llm_price_input_per_m": 0.045,
        "llm_price_output_per_m": 0.5,
        "budget_usd_per_month": 0.5,
        "generation": True,
    }
    # A model without "/" is kept whole and a missing price falls back to 0.
    assert public_config({"llm_model": "m"})["llm_model"] == "m"
    assert public_config({}) == {
        "llm_model": "",
        "llm_price_input_per_m": 0.0,
        "llm_price_output_per_m": 0.0,
        "budget_usd_per_month": 0.0,
        "generation": True,
    }


def test_refresh_report_keeps_the_answers_and_adds_cost() -> None:
    rows = [
        {
            "id": "a01",
            "lang": "en",
            "question": "How do partial indexes work?",
            "answerable": True,
            "abstain": False,
            "reason": None,
            "citations_valid": 1,
            "citations_total": 1,
            "sentences_total": 1,
            "sentences_with_valid": 1,
            "prompt_tokens": 1_000_000,
            "completion_tokens": 1_000_000,
            "tokens": 2_000_000,
            "retrieval_ms": 10.0,
            "llm_ms": 100.0,
            "sentences": [{"text": "Ok.", "citations": [{"n": 1, "quote": GOOD_QUOTE}]}],
            "sources": [{"n": 1, "url": "https://www.postgresql.org/docs/17/x.html", "title": "X"}],
        }
    ]
    old = {
        "generated_at": "2026-10-29T00:00:00Z",
        "seconds": 1.0,
        "threshold": None,
        "config": {"llm_model": "deepseek/deepseek-v4.1-flash", "llm_base_url": "https://x/v1",
                   "llm_price_input_per_m": 0.045, "llm_price_output_per_m": 0.5, "generation": True},
        "summary": {"questions": 1},
        "by_lang": {},
        "questions": rows,
    }
    refreshed = refresh_report(old)
    assert refreshed["questions"] == rows  # copied verbatim: no model, no rewrite
    assert refreshed["generated_at"] == "2026-10-29T00:00:00Z"
    assert refreshed["config"]["llm_model"] == "deepseek-v4.1-flash"
    assert "llm_base_url" not in refreshed["config"]
    assert refreshed["summary"]["cost_usd"] == 0.545
    assert refreshed["summary"]["tokens"] == {
        "prompt": 1_000_000,
        "completion": 1_000_000,
        "total": 2_000_000,
        "mean_per_question": 2_000_000.0,
    }
    assert refreshed["by_lang"]["en"]["cost_usd"] == 0.545


def test_sanitize_cli_rewrites_the_report_in_place(tmp_path, capsys) -> None:
    path = tmp_path / "2026-10-29-generate.json"
    path.write_text(
        json.dumps(
            {
                "generated_at": "2026-10-29T00:00:00Z",
                "config": {"llm_base_url": "https://provider.example/v1",
                           "llm_model": "deepseek/deepseek-v4.1-flash",
                           "llm_price_input_per_m": 1.0, "llm_price_output_per_m": 1.0},
                "summary": {},
                "questions": [
                    {
                        "id": "a01", "lang": "en", "question": "Q?", "answerable": True,
                        "abstain": False, "reason": None, "citations_valid": 0, "citations_total": 0,
                        "sentences_total": 0, "sentences_with_valid": 0,
                        "prompt_tokens": 1_000_000, "completion_tokens": 0, "tokens": 1_000_000,
                        "retrieval_ms": 1.0, "llm_ms": 2.0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    assert sanitize_main([str(path)]) == 0
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["config"] == {
        "llm_model": "deepseek-v4.1-flash",
        "llm_price_input_per_m": 1.0,
        "llm_price_output_per_m": 1.0,
        "budget_usd_per_month": 0.0,
        "generation": True,
    }
    assert report["summary"]["cost_usd"] == 1.0
    assert "dropped config keys" in capsys.readouterr().out
    assert sanitize_main([str(tmp_path / "nope.json")]) == 2


def test_generate_without_a_key_exits_2_and_writes_nothing(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(eval_generate, "REPORTS_DIR", tmp_path)
    monkeypatch.setenv("GROUNDED_RAG_LLM_API_KEY", "")
    monkeypatch.setenv("GROUNDED_RAG_LLM_BASE_URL", "")
    monkeypatch.setenv("GROUNDED_RAG_LLM_MODEL", "")
    code = eval_main(["--generate", "--file", "eval/questions.jsonl"])
    captured = capsys.readouterr()
    assert code == 2
    assert "not configured" in captured.out
    assert list(tmp_path.glob("*.json")) == []


def test_mock_without_generate_is_rejected() -> None:
    assert eval_main(["--mock"]) == 2
    assert eval_main(["--limit", "1"]) == 2
