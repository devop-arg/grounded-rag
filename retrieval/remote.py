"""Optional remote services for retrieval: embeddings, rerank and translation.

All three talk to the same OpenAI-compatible endpoint as the generation client
(``GROUNDED_RAG_LLM_BASE_URL``) with the same key, resolved at call time through
:func:`generation.config.resolve_llm_api_key` and never stored, printed or
logged.  Each part is switched on by its own variable and is off while that
variable is empty, so the shipped configuration changes nothing:

* ``GROUNDED_RAG_RERANK_MODEL`` — ``POST {base}/rerank``;
* ``GROUNDED_RAG_EMBED_MODEL`` plus ``GROUNDED_RAG_EMBED_INDEX`` — the named
  vector index built by ``python -m retrieval.reindex``;
* ``GROUNDED_RAG_TRANSLATE_MODEL`` — a short ``POST {base}/chat/completions``
  call that turns a non-English question into an English one.

Every call is bounded by a 10 s timeout and one retry on ``429``/``5xx``.  Any
failure (network, 4xx/5xx, timeout, a malformed body) returns ``None``: the
caller keeps the local behaviour for that part, :class:`RemoteStats` counts the
error and the answer is never failed because of it.  Costs come from
``usage.cost`` when the endpoint reports it, otherwise from the prices per env
(``GROUNDED_RAG_EMBED_PRICE_PER_M``, ``GROUNDED_RAG_RERANK_PRICE_PER_SEARCH``,
``GROUNDED_RAG_TRANSLATE_PRICE_INPUT_PER_M``/``_OUTPUT_PER_M``), default 0.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Sequence

import httpx
import numpy as np

from generation.config import MASK, Config, LLMNotConfigured, resolve_llm_api_key
from generation.llm import MAX_ERROR_DETAIL, USER_AGENT

from .embed import l2_normalize
from .store import DATA_DIR, VectorIndex, vector_ids_path_for, vectors_path_for

logger = logging.getLogger("grounded_rag.remote")

TIMEOUT_S = 10.0
MAX_RETRIES = 1
#: ``max_tokens`` of the translation call: a question, not an essay.
MAX_TRANSLATION_TOKENS = 100
#: A translation longer than this is not a translation of a question: ignore it.
MAX_TRANSLATION_CHARS = 600
TRANSLATE_SYSTEM = (
    "Translate the user question into English. Reply with the translation only: "
    "one line, no quotes, no explanation."
)
#: Characters of each candidate sent to the reranker (bounded request size).
RERANK_DOCUMENT_CHARS = 1200


class RemoteError(RuntimeError):
    """A remote call failed; the message never carries the key or a header."""


@dataclass
class RemoteStats:
    """Counters and USD cost of the remote calls made through one tracker."""

    embed_calls: int = 0
    embed_errors: int = 0
    embed_tokens: int = 0
    rerank_calls: int = 0
    rerank_errors: int = 0
    translate_calls: int = 0
    translate_errors: int = 0
    translate_prompt_tokens: int = 0
    translate_completion_tokens: int = 0
    #: Times a failed remote call was replaced by the local path.
    fallbacks: int = 0
    cost_usd: float = 0.0

    def add_cost(self, amount: float) -> None:
        if amount:
            self.cost_usd = round(self.cost_usd + float(amount), 9)

    def delta(self, before: "RemoteStats") -> "RemoteStats":
        """This tracker minus ``before``: the cost of one request."""
        change = RemoteStats()
        for item in fields(self):
            value = getattr(self, item.name) - getattr(before, item.name)
            setattr(change, item.name, round(value, 9) if item.name == "cost_usd" else int(round(value)))
        return change

    @property
    def errors(self) -> int:
        return self.embed_errors + self.rerank_errors + self.translate_errors

    @property
    def calls(self) -> int:
        return self.embed_calls + self.rerank_calls + self.translate_calls

    def as_dict(self) -> dict[str, Any]:
        return {
            "embed_calls": self.embed_calls,
            "embed_errors": self.embed_errors,
            "embed_tokens": self.embed_tokens,
            "rerank_calls": self.rerank_calls,
            "rerank_errors": self.rerank_errors,
            "translate_calls": self.translate_calls,
            "translate_errors": self.translate_errors,
            "translate_prompt_tokens": self.translate_prompt_tokens,
            "translate_completion_tokens": self.translate_completion_tokens,
            "calls": self.calls,
            "errors": self.errors,
            "fallbacks": self.fallbacks,
            "cost_usd": round(self.cost_usd, 6),
        }


def short_model(model: str) -> str:
    """The part after the last ``/``: the only model shape meant for logs/reports."""
    return str(model).rsplit("/", 1)[-1]


def uses_e5_prefixes(model: str) -> bool:
    """``True`` for the E5 family, whose texts must carry ``query:``/``passage:``."""
    return "e5" in str(model).lower()


class RemoteService:
    """Bearer-authenticated JSON client for one remote part."""

    name = "remote"

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
        referer: str = "",
        title: str = "",
    ) -> None:
        if not base_url or not api_key or not model:
            raise LLMNotConfigured("base URL, model and API key are required")
        self.base_url = base_url.rstrip("/")
        self._api_key = api_key
        self.model = model
        self.timeout = float(timeout)
        self.sleep = sleep
        self.max_retries = int(max_retries)
        self.referer = referer
        self.title = title
        self._client = httpx.Client(timeout=self.timeout, transport=transport)

    def __repr__(self) -> str:  # the key is never part of repr
        return f"{type(self).__name__}(model={short_model(self.model)!r})"

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "RemoteService":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ------------------------------------------------------------------ http
    def _headers(self) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "User-Agent": USER_AGENT,
        }
        if self.referer:
            headers["HTTP-Referer"] = self.referer
        if self.title:
            headers["X-Title"] = self.title
        return headers

    def _redact(self, text: str) -> str:
        return text.replace(self._api_key, MASK) if self._api_key else text

    def _error_detail(self, response: httpx.Response) -> str:
        """``error.message`` of the JSON body (capped, redacted); headers never used."""
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
        return self._redact(" ".join(message.split()))[:MAX_ERROR_DETAIL]

    def _http_error(self, status: int, response: httpx.Response) -> RemoteError:
        message = f"{self.name} request failed with HTTP {status}"
        detail = self._error_detail(response)
        if detail:
            message = f"{message}: {detail}"
        return RemoteError(message)

    def _post(self, path: str, payload: dict) -> dict:
        """One JSON POST with a bounded timeout and one retry on 429/5xx."""
        attempt = 0
        while True:
            try:
                response = self._client.post(
                    f"{self.base_url}{path}", json=payload, headers=self._headers()
                )
            except httpx.TimeoutException as error:
                raise RemoteError(
                    f"{self.name} request timed out after {self.timeout:g} s"
                ) from error
            except httpx.HTTPError as error:
                raise RemoteError(f"{self.name} request failed: {type(error).__name__}") from error

            status = response.status_code
            if status == 200:
                try:
                    data = response.json()
                except ValueError as error:
                    raise RemoteError(f"{self.name} returned a non-JSON response") from error
                if not isinstance(data, dict):
                    raise RemoteError(f"{self.name} returned an unexpected response")
                return data
            if (status == 429 or 500 <= status < 600) and attempt < self.max_retries:
                attempt += 1
                self.sleep(float(attempt))
                continue
            raise self._http_error(status, response)


def _usage_cost(body: dict) -> float | None:
    """``usage.cost`` when the endpoint reports it, else ``None``."""
    usage = body.get("usage")
    if not isinstance(usage, dict):
        return None
    cost = usage.get("cost")
    if isinstance(cost, (int, float)) and not isinstance(cost, bool):
        return float(cost)
    return None


def _usage_tokens(body: dict, name: str = "prompt_tokens") -> int:
    usage = body.get("usage")
    if not isinstance(usage, dict):
        return 0
    value = usage.get(name)
    return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0


class RemoteEmbeddings(RemoteService):
    """``POST {base}/embeddings``: ``{"model", "input": [...]}``."""

    name = "embeddings"

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        *,
        price_per_m: float = 0.0,
        dimensions: int | None = None,
        **kwargs,
    ) -> None:
        super().__init__(base_url, api_key, model, **kwargs)
        self.price_per_m = float(price_per_m)
        self.dimensions = int(dimensions) if dimensions else None
        self.dim: int | None = None
        #: E5 checkpoints need the prefixes; every other model gets none.
        self.passage_prefix = "passage: " if uses_e5_prefixes(model) else ""
        self.query_prefix = "query: " if uses_e5_prefixes(model) else ""

    def embed(
        self, texts: Sequence[str], *, prefix: str = "", stats: RemoteStats | None = None
    ) -> np.ndarray | None:
        """L2-normalised ``(len(texts), dim)`` matrix, or ``None`` on any failure."""
        texts = list(texts)
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        payload: dict[str, Any] = {"model": self.model, "input": [prefix + text for text in texts]}
        if self.dimensions:
            payload["dimensions"] = self.dimensions
        if stats is not None:
            stats.embed_calls += 1
        try:
            body = self._post("/embeddings", payload)
            matrix, tokens, cost = self._parse(body, len(texts))
        except RemoteError as error:
            if stats is not None:
                stats.embed_errors += 1
            logger.warning("embeddings: remote call failed, using the local model (%s)", error)
            return None
        if stats is not None:
            stats.embed_tokens += tokens
            stats.add_cost(cost if cost is not None else tokens * self.price_per_m / 1_000_000)
        self.dim = int(matrix.shape[1])
        return matrix

    def encode_query(self, query: str, stats: RemoteStats | None = None) -> np.ndarray | None:
        matrix = self.embed([query], prefix=self.query_prefix, stats=stats)
        return None if matrix is None else matrix[0]

    def encode_passages(self, texts: Sequence[str], stats: RemoteStats | None = None) -> np.ndarray | None:
        return self.embed(texts, prefix=self.passage_prefix, stats=stats)

    def _parse(self, body: dict, count: int) -> tuple[np.ndarray, int, float | None]:
        data = body.get("data")
        if not isinstance(data, list) or len(data) != count:
            raise RemoteError(f"embeddings returned {len(data) if isinstance(data, list) else 'no'} vectors for {count} inputs")
        items: list[tuple[int, Any]] = []
        for position, item in enumerate(data):
            if not isinstance(item, dict):
                raise RemoteError("embeddings returned a malformed item")
            vector = item.get("embedding")
            if not isinstance(vector, list) or not vector:
                raise RemoteError("embeddings returned an item without a vector")
            index = item.get("index", position)
            if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < count:
                raise RemoteError("embeddings returned an invalid index")
            items.append((index, vector))
        items.sort(key=lambda pair: pair[0])
        if [index for index, _ in items] != list(range(count)):
            raise RemoteError("embeddings returned duplicate or missing indexes")
        try:
            matrix = np.asarray([vector for _, vector in items], dtype=np.float32)
        except (TypeError, ValueError) as error:
            raise RemoteError("embeddings returned non-numeric vectors") from error
        if matrix.ndim != 2 or matrix.shape[0] != count or matrix.shape[1] == 0:
            raise RemoteError("embeddings returned a malformed matrix")
        if not np.isfinite(matrix).all():
            raise RemoteError("embeddings returned non-finite values")
        return l2_normalize(matrix), _usage_tokens(body), _usage_cost(body)


class RemoteReranker(RemoteService):
    """``POST {base}/rerank``: ``{"model", "query", "documents", "top_n"}``."""

    name = "rerank"

    def __init__(self, base_url: str, api_key: str, model: str, *, price_per_search: float = 0.0, **kwargs) -> None:
        super().__init__(base_url, api_key, model, **kwargs)
        self.price_per_search = float(price_per_search)
        self._logged_shape = False

    def rerank(
        self,
        query: str,
        documents: Sequence[str],
        *,
        top_n: int | None = None,
        stats: RemoteStats | None = None,
    ) -> list[tuple[int, float]] | None:
        """``[(document_index, relevance)]`` best first, or ``None`` on any failure."""
        documents = list(documents)
        if not documents:
            return []
        payload: dict[str, Any] = {"model": self.model, "query": query, "documents": documents}
        if top_n is not None:
            payload["top_n"] = int(top_n)
        if stats is not None:
            stats.rerank_calls += 1
        try:
            body = self._post("/rerank", payload)
            ranked = self._parse(body, len(documents))
        except RemoteError as error:
            if stats is not None:
                stats.rerank_errors += 1
            logger.warning("rerank: remote call failed, keeping the fused order (%s)", error)
            return None
        self._log_shape(body, len(ranked))
        if stats is not None:
            cost = _usage_cost(body)
            stats.add_cost(cost if cost is not None else self.price_per_search)
        return ranked

    def _parse(self, body: dict, count: int) -> list[tuple[int, float]]:
        items = body.get("results", body.get("data"))
        if isinstance(items, dict):
            items = items.get("results", items.get("data"))
        if not isinstance(items, list) or not items:
            raise RemoteError("rerank returned no results")
        ranked: list[tuple[int, float]] = []
        seen: set[int] = set()
        for item in items:
            if not isinstance(item, dict):
                raise RemoteError("rerank returned a malformed result")
            index = _first_int(item, ("index", "document_index", "document", "idx"))
            score = _first_number(item, ("relevance_score", "score", "relevance", "relevanceScore"))
            if index is None or score is None:
                raise RemoteError("rerank returned a result without an index or a score")
            if not 0 <= index < count or index in seen:
                continue
            seen.add(index)
            ranked.append((index, float(score)))
        if not ranked:
            raise RemoteError("rerank returned no usable scores")
        ranked.sort(key=lambda pair: (-pair[1], pair[0]))
        return ranked

    def _log_shape(self, body: dict, parsed: int) -> None:
        """Log *what the endpoint returned* once: keys and types, never values."""
        if self._logged_shape:
            return
        items = body.get("results", body.get("data"))
        if isinstance(items, dict):
            items = items.get("results", items.get("data"))
        first = items[0] if isinstance(items, list) and items and isinstance(items[0], dict) else {}
        logger.info(
            "rerank response shape: top-level keys=%s, items=%s, item keys=%s, parsed=%d, model=%s",
            sorted(body),
            len(items) if isinstance(items, list) else type(items).__name__,
            sorted(first),
            parsed,
            short_model(self.model),
        )
        self._logged_shape = True


def _first_int(item: dict, names: Sequence[str]) -> int | None:
    for name in names:
        value = item.get(name)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return None


def _first_number(item: dict, names: Sequence[str]) -> float | None:
    for name in names:
        value = item.get(name)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    return None


class RemoteTranslator(RemoteService):
    """One short chat call: the question in English, for the lexical search only."""

    name = "translate"

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        *,
        price_input_per_m: float = 0.0,
        price_output_per_m: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__(base_url, api_key, model, **kwargs)
        self.price_input_per_m = float(price_input_per_m)
        self.price_output_per_m = float(price_output_per_m)

    def translate(self, query: str, stats: RemoteStats | None = None) -> str | None:
        """The English question, or ``None`` when the call failed or the answer is unusable."""
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": TRANSLATE_SYSTEM},
                {"role": "user", "content": query},
            ],
            "temperature": 0,
            "max_tokens": MAX_TRANSLATION_TOKENS,
        }
        if stats is not None:
            stats.translate_calls += 1
        try:
            body = self._post("/chat/completions", payload)
            text = self._parse(body)
        except RemoteError as error:
            if stats is not None:
                stats.translate_errors += 1
            logger.warning("translate: remote call failed, keeping the original lexical query (%s)", error)
            return None
        if stats is not None:
            prompt_tokens = _usage_tokens(body)
            completion_tokens = _usage_tokens(body, "completion_tokens")
            stats.translate_prompt_tokens += prompt_tokens
            stats.translate_completion_tokens += completion_tokens
            cost = _usage_cost(body)
            stats.add_cost(
                cost
                if cost is not None
                else (prompt_tokens * self.price_input_per_m + completion_tokens * self.price_output_per_m)
                / 1_000_000
            )
        return text

    def _parse(self, body: dict) -> str:
        choices = body.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise RemoteError("translate returned no choices")
        message = choices[0].get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str):
            raise RemoteError("translate returned an empty message")
        text = " ".join(content.split()).strip().strip("\"'`")
        if not text:
            raise RemoteError("translate returned an empty translation")
        if len(text) > MAX_TRANSLATION_CHARS:
            raise RemoteError("translate returned a translation that is too long")
        if not any(char.isalpha() for char in text):
            raise RemoteError("translate returned a translation without words")
        return text


@dataclass
class RemoteRetrieval:
    """The remote parts that are on, the index they use and their counters."""

    embeddings: RemoteEmbeddings | None = None
    reranker: RemoteReranker | None = None
    translator: RemoteTranslator | None = None
    index: VectorIndex | None = None
    stats: RemoteStats = field(default_factory=RemoteStats)

    @property
    def active(self) -> bool:
        return bool(self.embeddings or self.reranker or self.translator)

    @property
    def remote_vectors(self) -> bool:
        """True when the remote embedding model and its index are both usable."""
        return self.embeddings is not None and self.index is not None

    def describe(self) -> dict[str, Any]:
        """Booleans plus the model names' last segment (never a route or a key)."""
        return {
            "embeddings": self.remote_vectors,
            "rerank": self.reranker is not None,
            "translate": self.translator is not None,
            "embed_model": short_model(self.embeddings.model) if self.embeddings else "",
            "rerank_model": short_model(self.reranker.model) if self.reranker else "",
            "translate_model": short_model(self.translator.model) if self.translator else "",
        }

    def close(self) -> None:
        for service in (self.embeddings, self.reranker, self.translator):
            if service is not None:
                service.close()


def build_remote(
    config: Config,
    *,
    transport: httpx.BaseTransport | None = None,
    sleep=time.sleep,
    data_dir=None,
    dims: int | None = None,
) -> RemoteRetrieval | None:
    """Build the configured remote parts; ``None`` when nothing is configured.

    A model whose vector index is missing disables only the embeddings (the rest
    of the remote parts keep working) and says which command builds it.  The key
    is resolved here, at call time: without one every part stays off and the
    local path is untouched.
    """
    if not config.remote_active:
        return None
    try:
        api_key = resolve_llm_api_key(config)
    except LLMNotConfigured as error:
        logger.info("remote retrieval disabled: %s", error)
        return None

    common = dict(
        base_url=config.llm_base_url,
        api_key=api_key,
        timeout=TIMEOUT_S,
        transport=transport,
        sleep=sleep,
        referer=config.llm_referer,
        title=config.llm_title,
    )

    embeddings = None
    index = None
    if config.embed_model:
        embeddings = RemoteEmbeddings(
            model=config.embed_model,
            price_per_m=config.embed_price_per_m,
            dimensions=dims if dims is not None else (config.embed_dims or None),
            **common,
        )
        name = config.embed_index_name
        base_dir = Path(data_dir) if data_dir is not None else DATA_DIR
        candidate = VectorIndex(vectors_path_for(name, base_dir), vector_ids_path_for(name, base_dir))
        if candidate.exists():
            index = candidate
        else:
            logger.warning(
                "remote embeddings disabled: no vector index %r (build it with"
                " `python -m retrieval.reindex --index %s`)",
                name,
                name,
            )
            embeddings = None

    reranker = (
        RemoteReranker(model=config.rerank_model, price_per_search=config.rerank_price_per_search, **common)
        if config.rerank_model
        else None
    )
    translator = (
        RemoteTranslator(
            model=config.translate_model,
            price_input_per_m=config.translate_price_input_per_m,
            price_output_per_m=config.translate_price_output_per_m,
            **common,
        )
        if config.translate_model
        else None
    )
    if not (embeddings or reranker or translator):
        return None
    return RemoteRetrieval(embeddings=embeddings, reranker=reranker, translator=translator, index=index)
