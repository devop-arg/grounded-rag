"""Generic OpenAI-compatible chat client (``POST {base_url}/chat/completions``).

No provider SDK: one ``httpx`` call, a 60 s timeout, one retry on ``429``/``5xx``
and one fallback when the server rejects ``response_format`` with a ``400``.
The default endpoint is OpenRouter; ``GROUNDED_RAG_LLM_BASE_URL`` points it
anywhere OpenAI-compatible.  The client never logs the API key or the request
headers, and its two optional attribution headers (``HTTP-Referer``,
``X-Title``) are plain project constants that other providers simply ignore.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import httpx

from .config import MASK, Config, LLMNotConfigured, resolve_llm_api_key

USER_AGENT = "grounded-rag/0.2 (+https://github.com/devop-arg/grounded-rag)"
#: Optional attribution headers used by OpenRouter; harmless elsewhere.
HTTP_REFERER = "https://github.com/devop-arg/grounded-rag"
X_TITLE = "grounded-rag"
#: Upstream error messages are quoted back for diagnosis, up to this many chars.
MAX_ERROR_DETAIL = 200
TIMEOUT_S = 60.0
MAX_RETRIES = 1


class LLMError(RuntimeError):
    """The model call failed.

    Messages carry the status code and at most
    :data:`MAX_ERROR_DETAIL` characters of the provider's JSON
    ``error.message``; request/response headers are never included and the API
    key is redacted.
    """


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

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "User-Agent": USER_AGENT,
            "HTTP-Referer": HTTP_REFERER,
            "X-Title": X_TITLE,
        }

    def _redact(self, text: str) -> str:
        """Never echo the key back, even if an upstream error message quotes it."""
        return text.replace(self._api_key, MASK) if self._api_key else text

    def _error_detail(self, response: httpx.Response) -> str:
        """``error.message`` from the JSON body (≤200 chars); headers never used."""
        try:
            data = response.json()
        except ValueError:
            return ""
        if not isinstance(data, dict):
            return ""
        error = data.get("error")
        message = error.get("message") if isinstance(error, dict) else None
        if not isinstance(message, str):
            return ""
        collapsed = self._redact(" ".join(message.split()))
        return collapsed[:MAX_ERROR_DETAIL]

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "OpenAICompatibleClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _http_error(self, status: int, response: httpx.Response) -> LLMError:
        """``LLMError`` with the status and, if the body has one, ``error.message``."""
        message = f"LLM request failed with HTTP {status}"
        detail = self._error_detail(response)
        if detail:
            message = f"{message}: {detail}"
        return LLMError(message)

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
                    raise self._http_error(status, response)
                continue
            if status == 429 or 500 <= status < 600:
                if attempt < self.max_retries:
                    attempt += 1
                    self.sleep(float(attempt))
                    continue
                raise self._http_error(status, response)
            raise self._http_error(status, response)

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

    The key is resolved here, at call time: ``GROUNDED_RAG_LLM_API_KEY`` wins,
    otherwise it is read from the JSON file named by
    ``GROUNDED_RAG_LLM_KEY_FILE`` at the dotted path ``GROUNDED_RAG_LLM_KEY_PATH``
    and handed straight to the client (never stored on ``Config``).
    ``GROUNDED_RAG_LLM_FAKE=mock`` swaps in the deterministic offline model, so
    the app and the eval pipeline can be exercised end to end without network.
    """
    if config.llm_fake == "mock":
        from .mock import MockLLM

        return MockLLM()
    config.require_llm()
    return OpenAICompatibleClient(
        config.llm_base_url, resolve_llm_api_key(config), config.llm_model
    )
