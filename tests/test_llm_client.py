"""OpenAI-compatible client: headers, JSON-mode fallback, retry, no leaked key."""

from __future__ import annotations

import json

import httpx
import pytest

from generation.config import Config, LLMNotConfigured
from generation.llm import (
    USER_AGENT,
    LLMError,
    OpenAICompatibleClient,
    build_llm,
)

KEY = "sk-test-secret-value"
BASE_URL = "https://llm.example/v1"
MODEL = "deepseek-v4.1-flash"
BODY = {
    "choices": [{"message": {"content": '{"abstain": true, "sentences": []}'}}],
    "usage": {"prompt_tokens": 11, "completion_tokens": 3},
    "model": "test-model",
}

MESSAGES = [{"role": "user", "content": "hi"}]


def make_client(handler, sleeps: list | None = None, **overrides) -> OpenAICompatibleClient:
    sleeps = sleeps if sleeps is not None else []
    return OpenAICompatibleClient(
        "https://llm.example/v1/",
        KEY,
        "test-model",
        transport=httpx.MockTransport(handler),
        sleep=sleeps.append,
        timeout=5.0,
        **overrides,
    )


def test_happy_path_headers_and_usage() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=BODY)

    client = make_client(handler)
    result = client.complete(MESSAGES)

    assert result.text == '{"abstain": true, "sentences": []}'
    assert (result.prompt_tokens, result.completion_tokens) == (11, 3)
    request = seen[0]
    assert str(request.url) == "https://llm.example/v1/chat/completions"
    assert request.headers["Authorization"] == f"Bearer {KEY}"
    assert request.headers["User-Agent"] == USER_AGENT
    # Optional attribution headers are off unless configured: none is invented.
    assert "HTTP-Referer" not in request.headers
    assert "X-Title" not in request.headers
    assert KEY not in "".join(f"{name}:{value}" for name, value in request.headers.items() if name != "authorization")
    payload = json.loads(request.content)
    assert payload["response_format"] == {"type": "json_object"}
    assert payload["temperature"] == 0


def test_attribution_headers_are_sent_only_when_configured() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=BODY)

    client = make_client(handler, referer="https://example.com/docs", title="example")
    client.complete(MESSAGES)
    assert seen[0].headers["HTTP-Referer"] == "https://example.com/docs"
    assert seen[0].headers["X-Title"] == "example"


def test_repr_hides_the_key() -> None:
    client = make_client(lambda request: httpx.Response(200, json=BODY))
    assert KEY not in repr(client)


def test_400_on_response_format_retries_once_without_it() -> None:
    payloads: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        payloads.append(payload)
        if "response_format" in payload:
            return httpx.Response(400, json={"error": "response_format unsupported"})
        return httpx.Response(200, json=BODY)

    client = make_client(handler)
    result = client.complete(MESSAGES)
    assert result.prompt_tokens == 11
    assert len(payloads) == 2
    assert "response_format" not in payloads[1]


def test_429_is_retried_once_with_backoff() -> None:
    statuses = [429, 200]
    sleeps: list = []

    def handler(request: httpx.Request) -> httpx.Response:
        status = statuses.pop(0)
        return httpx.Response(status, json=BODY if status == 200 else {"error": "slow down"})

    client = make_client(handler, sleeps)
    assert client.complete(MESSAGES).text.startswith("{")
    assert sleeps == [1.0]


def test_5xx_after_the_retry_raises_a_key_free_error() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(503, text=f"upstream said: {KEY}")

    client = make_client(handler)
    with pytest.raises(LLMError) as error:
        client.complete(MESSAGES)
    assert calls["n"] == 2
    assert "HTTP 503" in str(error.value)
    assert KEY not in str(error.value)


def test_client_errors_are_not_retried() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(401, json={"error": "bad key"})

    client = make_client(handler)
    with pytest.raises(LLMError, match="HTTP 401"):
        client.complete(MESSAGES)
    assert calls["n"] == 1


def test_empty_choices_raise() -> None:
    client = make_client(lambda request: httpx.Response(200, json={"choices": []}))
    with pytest.raises(LLMError, match="no choices"):
        client.complete(MESSAGES)


def test_error_carries_the_provider_message() -> None:
    client = make_client(
        lambda request: httpx.Response(502, json={"error": {"message": "model over capacity"}})
    )
    with pytest.raises(LLMError) as error:
        client.complete(MESSAGES)
    text = str(error.value)
    assert "HTTP 502" in text
    assert "model over capacity" in text


def test_error_message_is_capped_at_200_chars() -> None:
    client = make_client(lambda request: httpx.Response(400, json={"error": {"message": "y" * 500}}))
    with pytest.raises(LLMError) as error:
        client.complete(MESSAGES)
    assert str(error.value) == f"LLM request failed with HTTP 400: {'y' * 200}"


def test_error_never_quotes_headers_or_the_key() -> None:
    client = make_client(
        lambda request: httpx.Response(
            401,
            json={"error": {"message": f"bad key {KEY}"}},
            headers={"X-Upstream-Debug": "header-secret-value"},
        )
    )
    with pytest.raises(LLMError) as error:
        client.complete(MESSAGES)
    text = str(error.value)
    assert "HTTP 401" in text
    assert KEY not in text
    assert "header-secret-value" not in text
    assert "Authorization" not in text


def test_error_body_without_error_message_is_status_only() -> None:
    client = make_client(lambda request: httpx.Response(500, json={"error": "boom"}))
    with pytest.raises(LLMError) as error:
        client.complete(MESSAGES)
    assert str(error.value) == "LLM request failed with HTTP 500"


def test_build_llm_reads_the_key_file_at_call_time(tmp_path) -> None:
    path = tmp_path / "auth.json"
    path.write_text(json.dumps({"llm": {"key": KEY}}), encoding="utf-8")
    config = Config(
        llm_base_url=BASE_URL,
        llm_model=MODEL,
        llm_key_file=str(path),
        llm_key_path="llm.key",
        llm_referer="https://example.com/docs",
        llm_title="example",
    )
    assert config.llm_api_key == ""

    client = build_llm(config)
    try:
        assert isinstance(client, OpenAICompatibleClient)
        assert client.base_url == BASE_URL  # nothing is hardcoded
        assert client.model == MODEL
        assert (client.referer, client.title) == ("https://example.com/docs", "example")
        assert KEY not in repr(client)
    finally:
        client.close()

    # The file is read per call, not cached in the config: deleting it breaks the next build.
    path.unlink()
    with pytest.raises(LLMNotConfigured):
        build_llm(config)
