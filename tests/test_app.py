"""API: validation, rate limit, daily cap, generation switch, 502 and no IP leaks."""

from __future__ import annotations

import logging

import pytest

from app.usage import UsageStore
from generation.generate import Thresholds
from generation.llm import LLMError

from helpers import (
    PARTIAL_INDEX_TEXT,
    FailingLLM,
    FakeEmbedder,
    FakeStore,
    ScriptedLLM,
    chunk,
    make_client,
    make_settings,
)

GOOD_QUOTE = "A partial index is an index built over a subset of a table"
VALID_ANSWER = {
    "abstain": False,
    "sentences": [{"text": "It restricts the rows.", "citations": [{"n": 1, "quote": GOOD_QUOTE}]}],
}


def test_health(tmp_path) -> None:
    client, _ = make_client(tmp_path)
    with client:
        response = client.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["generation"] is True
    assert body["llm_configured"] is True
    assert set(body["budget"]) == {
        "month",
        "spent_usd",
        "budget_usd",
        "remaining_usd",
        "exhausted",
        "prompt_tokens",
        "completion_tokens",
        "prices_usd_per_million",
    }
    assert body["budget"]["budget_usd"] == 0.5
    assert body["budget"]["spent_usd"] == 0.0
    assert body["budget"]["exhausted"] is False


def test_ask_returns_validated_answer(tmp_path) -> None:
    client, _ = make_client(tmp_path, llm=ScriptedLLM([VALID_ANSWER]))
    with client:
        response = client.post("/api/ask", json={"question": "How do partial indexes work?"})
    assert response.status_code == 200
    body = response.json()
    assert body["abstain"] is False
    assert body["sentences"][0]["citations"] == [{"n": 1, "quote": GOOD_QUOTE}]
    assert [source["n"] for source in body["sources"]] == [1]
    assert "reason" not in body


def test_ask_abstention_carries_a_reason(tmp_path) -> None:
    client, _ = make_client(tmp_path, llm=ScriptedLLM([{"abstain": True, "sentences": []}]))
    with client:
        response = client.post("/api/ask", json={"question": "What is the airspeed of a swallow?"})
    assert response.status_code == 200
    assert response.json() == {"abstain": True, "sentences": [], "sources": [], "reason": "model"}


def test_empty_question_is_422(tmp_path) -> None:
    client, _ = make_client(tmp_path, llm=ScriptedLLM([VALID_ANSWER]))
    with client:
        for question in ("", "   ", "\n\t"):
            assert client.post("/api/ask", json={"question": question}).status_code == 422
        assert client.post("/api/ask", json={}).status_code == 422


def test_too_long_question_is_422(tmp_path) -> None:
    client, _ = make_client(tmp_path, llm=ScriptedLLM([VALID_ANSWER]))
    with client:
        assert client.post("/api/ask", json={"question": "a" * 501}).status_code == 422
        assert client.post("/api/ask", json={"question": "a" * 500}).status_code == 200


def test_unknown_language_is_422(tmp_path) -> None:
    client, _ = make_client(tmp_path, llm=ScriptedLLM([VALID_ANSWER]))
    with client:
        assert client.post("/api/ask", json={"question": "q", "lang": "fr"}).status_code == 422


def test_rate_limit_per_ip_is_429(tmp_path) -> None:
    client, _ = make_client(tmp_path, llm=ScriptedLLM([VALID_ANSWER]), rate_per_ip="2/hour")
    with client:
        assert client.post("/api/ask", json={"question": "q1"}).status_code == 200
        assert client.post("/api/ask", json={"question": "q2"}).status_code == 200
        response = client.post("/api/ask", json={"question": "q3"})
    assert response.status_code == 429
    assert "slow down" in response.json()["detail"]


def test_rate_limit_is_per_hashed_ip(tmp_path) -> None:
    client, application = make_client(tmp_path, llm=ScriptedLLM([VALID_ANSWER]), rate_per_ip="1/hour")
    limiter = application.state.limiter
    assert limiter.allow("1.2.3.4") is True
    assert limiter.allow("1.2.3.4") is False
    assert limiter.allow("5.6.7.8") is True
    assert "1.2.3.4" not in limiter._hits  # only the digest is kept


def test_budget_exhausted_is_503_with_reason_and_skips_the_model(tmp_path) -> None:
    llm = ScriptedLLM([VALID_ANSWER])
    client, _ = make_client(tmp_path, llm=llm, budget_usd_per_month=0.0)
    with client:
        response = client.post("/api/ask", json={"question": "How do partial indexes work?"})
    assert response.status_code == 503
    body = response.json()
    assert body["reason"] == "budget_exhausted"
    assert "budget" in body["detail"]
    assert llm.calls == 0


def test_a_month_of_answers_stops_at_the_budget(tmp_path) -> None:
    """Every answer costs 120 tokens × 1 USD/M = 0.00012 USD, capped at 0.0002."""
    llm = ScriptedLLM([VALID_ANSWER])
    client, application = make_client(
        tmp_path,
        llm=llm,
        budget_usd_per_month=0.0002,
        llm_price_input_per_m=1.0,
        llm_price_output_per_m=1.0,
    )
    with client:
        assert client.post("/api/ask", json={"question": "q1"}).status_code == 200
        assert client.post("/api/ask", json={"question": "q2"}).status_code == 200
        response = client.post("/api/ask", json={"question": "q3"})
    assert response.status_code == 503
    assert response.json()["reason"] == "budget_exhausted"
    assert llm.calls == 2
    assert application.state.usage.spent() == pytest.approx(0.00024)
    assert application.state.usage.exhausted() is True


def test_metrics_and_health_expose_the_budget(tmp_path) -> None:
    client, application = make_client(tmp_path, llm=ScriptedLLM([VALID_ANSWER]))
    with client:
        assert client.post("/api/ask", json={"question": "q"}).status_code == 200
        health = client.get("/api/health").json()
        metrics = client.get("/api/metrics").json()
    spent = application.state.usage.spent()
    assert health["budget"]["spent_usd"] == round(spent, 6)
    assert metrics["budget"] == health["budget"]
    dumped = repr(metrics["budget"])
    assert "key" not in dumped and "salt" not in dumped


def test_tokens_are_recorded_in_the_usage_db(tmp_path) -> None:
    client, application = make_client(tmp_path, llm=ScriptedLLM([VALID_ANSWER]))
    with client:
        assert client.post("/api/ask", json={"question": "q"}).status_code == 200
    assert application.state.usage.tokens() == (100, 20)


def test_generation_off_returns_sources_without_calling_the_model(tmp_path) -> None:
    llm = ScriptedLLM([VALID_ANSWER])
    store = FakeStore()
    client, _ = make_client(tmp_path, llm=llm, store=store, generation=False)
    with client:
        response = client.post("/api/ask", json={"question": "How do partial indexes work?"})
    assert response.status_code == 200
    body = response.json()
    assert body["abstain"] is True
    assert body["reason"] == "generation_disabled"
    assert body["sentences"] == []
    assert len(body["sources"]) == 2
    assert set(body["sources"][0]) == {"n", "url", "title"}
    assert llm.calls == 0


def test_llm_error_is_502_without_internal_details(tmp_path) -> None:
    client, _ = make_client(tmp_path, llm=FailingLLM())
    with client:
        response = client.post("/api/ask", json={"question": "How do partial indexes work?"})
    assert response.status_code == 502
    assert response.json() == {"detail": "generation failed"}


def test_missing_key_is_503_not_a_crash(tmp_path) -> None:
    client, _ = make_client(tmp_path, llm=None, llm_fake="")
    with client:
        response = client.post("/api/ask", json={"question": "q"})
    assert response.status_code == 503
    assert "not configured" in response.json()["detail"]


def test_threshold_abstention_skips_the_model(tmp_path) -> None:
    llm = ScriptedLLM([VALID_ANSWER])
    client, _ = make_client(
        tmp_path,
        llm=llm,
        thresholds=Thresholds(local={"signal": "vector_cosine", "value": 0.99, "note": "test"}),
    )
    with client:
        response = client.post("/api/ask", json={"question": "q"})
    assert response.status_code == 200
    assert response.json()["reason"] == "threshold"
    assert llm.calls == 0


def test_empty_ip_salt_still_limits_and_is_not_brute_forceable(tmp_path) -> None:
    import hashlib

    client, application = make_client(
        tmp_path, llm=ScriptedLLM([VALID_ANSWER]), rate_per_ip="1/hour", ip_salt=""
    )
    with client:
        assert client.post("/api/ask", json={"question": "q1"}).status_code == 200
        assert client.post("/api/ask", json={"question": "q2"}).status_code == 429
    limiter = application.state.limiter
    assert limiter.key("testclient") != hashlib.sha256(b"testclient").hexdigest()


def test_client_ip_is_neither_logged_nor_stored(tmp_path, caplog) -> None:
    client, application = make_client(tmp_path, llm=ScriptedLLM([VALID_ANSWER]), rate_per_ip="1/hour")
    caplog.set_level(logging.DEBUG)
    with client:
        client.post("/api/ask", json={"question": "q"})
        client.post("/api/ask", json={"question": "q"})  # 429
    assert "testclient" not in caplog.text
    raw = application.state.usage.path.read_bytes()
    assert b"testclient" not in raw
    assert application.state.limiter.key("testclient").encode() not in raw
    assert b"Authorization" not in raw


def test_threshold_rejection_does_not_consume_the_budget(tmp_path) -> None:
    client, application = make_client(
        tmp_path,
        llm=ScriptedLLM([VALID_ANSWER]),
        thresholds=Thresholds(local={"signal": "rrf", "value": 1.0}),
    )
    with client:
        client.post("/api/ask", json={"question": "q"})
    assert application.state.usage.spent() == 0.0


def test_usage_store_is_per_utc_month(tmp_path) -> None:
    store = UsageStore(tmp_path / "usage.sqlite", 0.0001, input_price_per_m=1.0, output_price_per_m=0.0)
    store.add(40_000, 0, "2026-01")
    store.add(10_000, 0, "2026-01")
    assert store.tokens("2026-01") == (50_000, 0)
    assert store.spent("2026-01") == pytest.approx(0.05)
    assert store.spent("2026-02") == 0.0
    assert store.exhausted("2026-01") is True
    assert store.exhausted("2026-02") is False


def test_app_refuses_a_non_loopback_host(monkeypatch) -> None:
    from app.__main__ import main

    monkeypatch.setenv("GROUNDED_RAG_HOST", "0.0.0.0")
    assert main([]) == 2
    assert main(["--host", "192.168.1.10"]) == 2
