"""Shared offline fakes: no network, no embedding model, no real LLM."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from app.limits import RateLimiter
from app.main import create_app
from app.usage import UsageStore
from generation.config import Config
from generation.llm import LLMResult


def chunk(chunk_id: str, url: str, text: str, title: str = "Page", path: list[str] | None = None) -> dict:
    return {
        "chunk_id": chunk_id,
        "url": url,
        "page_title": title,
        "section_path": path or ["Section"],
        "text": text,
    }


PARTIAL_INDEX_TEXT = (
    "A partial index is an index built over a subset of a table; the subset is defined by a "
    "conditional expression. It is useful for queries that only need to scan a few rows."
)
BRIN_TEXT = (
    "BRIN indexes store summaries of the values in each block range and are small, which makes "
    "them a good fit for very large tables with naturally ordered data."
)


def default_chunks() -> list[dict]:
    return [
        chunk("c1", "https://www.postgresql.org/docs/17/indexes-partial.html", PARTIAL_INDEX_TEXT, "Partial Indexes"),
        chunk("c2", "https://www.postgresql.org/docs/17/brin.html", BRIN_TEXT, "BRIN Indexes"),
    ]


class FakeEmbedder:
    """Returns a fixed, normalized 2-d query vector."""

    def __init__(self) -> None:
        self.queries: list[str] = []

    def encode_query(self, query: str) -> np.ndarray:
        self.queries.append(query)
        return np.asarray([0.6, 0.8], dtype=np.float32)


class FakeStore:
    """Deterministic :class:`retrieval.store.Store` over an in-memory corpus."""

    def __init__(
        self,
        chunks: list[dict] | None = None,
        lexical: list[str] | None = None,
        vector_scores: dict[str, float] | None = None,
    ) -> None:
        self.chunks = {row["chunk_id"]: row for row in (chunks if chunks is not None else default_chunks())}
        order = list(self.chunks)
        self.lexical_order = lexical if lexical is not None else order
        if vector_scores is None:
            vector_scores = {chunk_id: 0.9 - 0.1 * index for index, chunk_id in enumerate(order)}
        self.vector_scores = vector_scores

    @property
    def size(self) -> int:
        return len(self.chunks)

    def lexical(self, query: str, k: int, mode: str = "stem") -> list[tuple[str, float]]:
        return [(chunk_id, 1.0 - 0.01 * index) for index, chunk_id in enumerate(self.lexical_order[:k])]

    def vector(self, qvec, k: int) -> list[tuple[str, float]]:
        ordered = sorted(self.vector_scores.items(), key=lambda pair: (-pair[1], pair[0]))
        return ordered[:k]

    def get(self, ids: list[str]) -> list[dict]:
        return [self.chunks[chunk_id] for chunk_id in ids if chunk_id in self.chunks]


class ScriptedLLM:
    """Returns the scripted payloads in order and records the prompts it got."""

    def __init__(self, payloads: list[dict | str]) -> None:
        self.payloads = list(payloads)
        self.messages: list[list[dict]] = []
        self.calls = 0

    def complete(self, messages: list[dict[str, str]], *, json_mode: bool = True) -> LLMResult:
        self.calls += 1
        self.messages.append(messages)
        index = min(self.calls - 1, len(self.payloads) - 1)
        payload = self.payloads[index]
        text = payload if isinstance(payload, str) else json.dumps(payload)
        return LLMResult(text=text, prompt_tokens=100, completion_tokens=20, model="scripted")


class FailingLLM:
    """Raises :class:`generation.llm.LLMError` (what the API turns into a 502)."""

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, messages, *, json_mode: bool = True):
        from generation.llm import LLMError

        self.calls += 1
        raise LLMError("LLM request failed with HTTP 500")


def make_settings(tmp_path: Path, **overrides) -> Config:
    base = dict(
        llm_base_url="",
        llm_api_key="",
        llm_model="",
        generation=True,
        budget_usd_per_month=0.50,
        llm_price_input_per_m=0.0,
        llm_price_output_per_m=0.0,
        rate_per_ip="10/hour",
        ip_salt="pepper",
        port=8790,
        llm_fake="mock",
        usage_db=tmp_path / "usage.sqlite",
    )
    base.update(overrides)
    return Config(**base)


def make_client(tmp_path: Path, *, llm=None, store=None, embedder=None, thresholds=None, **overrides):
    """TestClient plus the injected fakes, so no model or network is touched."""
    from fastapi.testclient import TestClient

    from generation.generate import Thresholds

    config = make_settings(tmp_path, **overrides)
    limit, window = config.rate_limit()
    if thresholds is None:
        thresholds = Thresholds(local={"signal": "vector_cosine", "value": 0.0})
    application = create_app(
        config,
        store=store or FakeStore(),
        embedder=embedder or FakeEmbedder(),
        llm=llm,
        usage=UsageStore(
            config.usage_db,
            config.budget_usd_per_month,
            input_price_per_m=config.llm_price_input_per_m,
            output_price_per_m=config.llm_price_output_per_m,
        ),
        limiter=RateLimiter(limit, window, salt=config.ip_salt),
        thresholds=thresholds,
    )
    return TestClient(application), application
