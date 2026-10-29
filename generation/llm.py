"""Generic OpenAI-compatible chat client (``POST {base_url}/chat/completions``).

No provider SDK: one ``httpx`` call, a 60 s timeout, one retry on ``429``/``5xx``
and one fallback when the server rejects ``response_format`` with a ``400``.
The client never logs the API key or the request headers, and the process-wide
``x-session-id`` id makes calls from one run traceable without leaking
anything.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import httpx

from .config import Config, LLMNotConfigured

USER_AGENT = "grounded-rag/0.2 (+https://github.com/devop-arg/grounded-rag)"
SESSION_ID = uuid.uuid4().hex
TIMEOUT_S = 60.0
MAX_RETRIES = 1


class LLMError(RuntimeError):
    """The model call failed.  Messages carry status codes, never headers."""


@dataclass
class LLMResult:
    """Text plus token usage of one completion."""

    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    model: str = ""


@runtime_checkable
class ChatModel(Protocol):
    """The injectable interface used by the generator and the eval pipeline."""

    def complete(self, messages: list[dict[str, str]], *, json_mode: bool = True) -> LLMResult: ...


class OpenAICompatibleClient:
    """``/chat/completions`` client with JSON mode, retry and bearer auth."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        *,
        timeout: float = TIMEOUT_S,
        transport: httpx.BaseTransport | None = None,
        sleep=time.sleep,
        max_retries: int = MAX_RETRIES,
    ) -> None:
        if not base_url or not api_key or not model:
            raise LLMNotConfigured("base URL, model and API key are required")
        self.base_url = base_url.rstrip("/")
        self._api_key = api_key
        self.model = model
        self.timeout = timeout
        self.sleep = sleep
        self.max_retries = max_retries
        self._json_mode = True
        self._client = httpx.Client(timeout=timeout, transport=transport)

    def __repr__(self) -> str:  # the key is never part of repr
        return f"OpenAICompatibleClient(base_url={self.base_url!r}, model={self.model!r})"

    @property
    def session_id(self) -> str:
        return SESSION_ID

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "User-Agent": USER_AGENT,
            "x-session-id": SESSION_ID,
        }

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "OpenAICompatibleClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _payload(self, messages: list[dict[str, str]], json_mode: bool) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": 0,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        return payload

    def complete(self, messages: list[dict[str, str]], *, json_mode: bool = True) -> LLMResult:
        use_json = bool(json_mode and self._json_mode)
        attempt = 0
        while True:
            try:
                response = self._client.post(
                    f"{self.base_url}/chat/completions",
                    json=self._payload(messages, use_json),
                    headers=self._headers(),
                )
            except httpx.HTTPError as error:
                raise LLMError(f"LLM request failed: {type(error).__name__}") from error

            status = response.status_code
            if status == 200:
                return self._parse(response)
            if status == 400 and use_json:
                # Some servers do not implement response_format: retry once without it.
                use_json = False
                self._json_mode = False
                attempt += 1
                if attempt > self.max_retries + 1:
                    raise LLMError("LLM request failed with HTTP 400")
                continue
            if status == 429 or 500 <= status < 600:
                if attempt < self.max_retries:
                    attempt += 1
                    self.sleep(float(attempt))
                    continue
                raise LLMError(f"LLM request failed with HTTP {status}")
            raise LLMError(f"LLM request failed with HTTP {status}")

    def _parse(self, response: httpx.Response) -> LLMResult:
        try:
            data = response.json()
        except ValueError as error:
            raise LLMError("LLM returned a non-JSON response") from error
        choices = data.get("choices") or []
        if not choices:
            raise LLMError("LLM returned no choices")
        message = choices[0].get("message") or {}
        content = message.get("content")
        if not isinstance(content, str):
            raise LLMError("LLM returned an empty message")
        usage = data.get("usage") or {}
        return LLMResult(
            text=content,
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
            model=str(data.get("model") or self.model),
        )


def build_llm(config: Config) -> ChatModel:
    """Client for ``config``; raises :class:`LLMNotConfigured` without a key.

    ``GROUNDED_RAG_LLM_FAKE=mock`` swaps in the deterministic offline model, so
    the app and the eval pipeline can be exercised end to end without network.
    """
    if config.llm_fake == "mock":
        from .mock import MockLLM

        return MockLLM()
    config.require_llm()
    return OpenAICompatibleClient(config.llm_base_url, config.llm_api_key, config.llm_model)
