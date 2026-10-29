"""OpenAI-compatible client: headers, JSON-mode fallback, retry, no leaked key."""

from __future__ import annotations

import json

import httpx
import pytest

from generation.llm import SESSION_ID, USER_AGENT, LLMError, OpenAICompatibleClient

KEY = "sk-test-secret-value"
BODY = {
    "choices": [{"message": {"content": '{"abstain": true, "sentences": []}'}}],
    "usage": {"prompt_tokens": 11, "completion_tokens": 3},
    "model": "test-model",
}

MESSAGES = [{"role": "user", "content": "hi"}]


def make_client(handler, sleeps: list | None = None) -> OpenAICompatibleClient:
    sleeps = sleeps if sleeps is not None else []
    return OpenAICompatibleClient(
        "https://llm.example/v1/",
        KEY,
        "test-model",
        transport=httpx.MockTransport(handler),
        sleep=sleeps.append,
        timeout=5.0,
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
    assert request.headers["x-session-id"] == SESSION_ID
    payload = json.loads(request.content)
    assert payload["response_format"] == {"type": "json_object"}
    assert payload["temperature"] == 0


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
