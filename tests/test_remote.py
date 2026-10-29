"""Remote retrieval parts, offline: parsing, fallbacks, costs and secret hygiene.

Every HTTP call is served by ``httpx.MockTransport``: no test touches the
network, no test needs a key file and no test loads an embedding model.
"""

from __future__ import annotations

import json
import logging

import httpx
import numpy as np
import pytest

from generation.config import Config
from retrieval.remote import (
    RemoteEmbeddings,
    RemoteError,
    RemoteReranker,
    RemoteRetrieval,
    RemoteStats,
    RemoteTranslator,
    build_remote,
    short_model,
    uses_e5_prefixes,
)
from retrieval.search import hybrid_search
from retrieval.store import VectorIndex, vector_ids_path_for, vectors_path_for

KEY = "sk-remote-test-value"
BASE_URL = "https://llm.example/v1"
MODEL = "vendor/embed-model"


# --------------------------------------------------------------------- helpers
def embeddings_client(handler, **overrides) -> RemoteEmbeddings:
    return RemoteEmbeddings(
        BASE_URL,
        KEY,
        overrides.pop("model", MODEL),
        transport=httpx.MockTransport(handler),
        sleep=overrides.pop("sleep", lambda seconds: None),
        **overrides,
    )


def reranker(handler, **overrides) -> RemoteReranker:
    return RemoteReranker(
        BASE_URL,
        KEY,
        overrides.pop("model", "vendor/rerank-model"),
        transport=httpx.MockTransport(handler),
        sleep=lambda seconds: None,
        **overrides,
    )


def translator(handler, **overrides) -> RemoteTranslator:
    return RemoteTranslator(
        BASE_URL,
        KEY,
        overrides.pop("model", "vendor/translate-model"),
        transport=httpx.MockTransport(handler),
        sleep=lambda seconds: None,
        **overrides,
    )


def embeddings_body(vectors: list[list[float]], tokens: int = 12, cost: float | None = None) -> dict:
    usage = {"prompt_tokens": tokens}
    if cost is not None:
        usage["cost"] = cost
    return {
        "object": "list",
        "data": [
            {"object": "embedding", "index": index, "embedding": vector}
            for index, vector in enumerate(vectors)
        ],
        "usage": usage,
    }


def make_chunk(chunk_id: str, url: str, text: str) -> dict:
    return {
        "chunk_id": chunk_id,
        "url": url,
        "page_title": "Page",
        "section_path": ["Section"],
        "text": text,
    }


class RecordingStore:
    """A store that records the lexical query it saw and its mode."""

    def __init__(self, chunks: list[dict], lexical: list[str] | None = None, vector: list[str] | None = None):
        self.chunks = {chunk["chunk_id"]: chunk for chunk in chunks}
        order = list(self.chunks)
        self.lexical_ids = lexical if lexical is not None else order
        self.vector_ids = vector if vector is not None else []
        self.lexical_queries: list[str] = []
        self.modes: list[str] = []

    def lexical(self, query: str, k: int, mode: str = "stem") -> list[tuple[str, float]]:
        self.lexical_queries.append(query)
        self.modes.append(mode)
        if not query.strip():
            return []
        return [(chunk_id, 1.0 - 0.01 * index) for index, chunk_id in enumerate(self.lexical_ids[:k])]

    def vector(self, qvec, k: int) -> list[tuple[str, float]]:
        return [(chunk_id, 1.0 - 0.01 * index) for index, chunk_id in enumerate(self.vector_ids[:k])]

    def get(self, ids: list[str]) -> list[dict]:
        return [self.chunks[chunk_id] for chunk_id in ids if chunk_id in self.chunks]


class RecordingEmbedder:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def encode_query(self, query: str) -> np.ndarray:
        self.queries.append(query)
        return np.asarray([0.6, 0.8], dtype=np.float32)


def write_index(directory, name: str, ids: list[str], vectors: np.ndarray) -> VectorIndex:
    directory.mkdir(parents=True, exist_ok=True)
    np.save(vectors_path_for(name, directory), vectors.astype(np.float32))
    vector_ids_path_for(name, directory).write_text(json.dumps(ids))
    return VectorIndex(vectors_path_for(name, directory), vector_ids_path_for(name, directory))


# ------------------------------------------------------------------ embeddings
def test_embeddings_parse_the_vectors_the_usage_and_the_reported_cost() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=embeddings_body([[3.0, 4.0], [0.0, 2.0]], tokens=21, cost=0.00042))

    stats = RemoteStats()
    client = embeddings_client(handler, price_per_m=1.0)
    matrix = client.encode_passages(["a", "b"], stats)

    assert matrix is not None
    assert matrix.shape == (2, 2)
    # L2-normalised, so a dot product is a cosine.
    assert matrix[0] == pytest.approx([0.6, 0.8])
    assert matrix[1] == pytest.approx([0.0, 1.0])
    assert (stats.embed_calls, stats.embed_errors, stats.embed_tokens) == (1, 0, 21)
    # usage.cost wins over the configured price.
    assert stats.cost_usd == pytest.approx(0.00042)
    request = seen[0]
    assert str(request.url) == f"{BASE_URL}/embeddings"
    assert request.headers["Authorization"] == f"Bearer {KEY}"
    payload = json.loads(request.content)
    assert payload["model"] == MODEL
    assert payload["input"] == ["a", "b"]


def test_embeddings_price_per_million_is_used_when_the_endpoint_reports_no_cost() -> None:
    client = embeddings_client(
        lambda request: httpx.Response(200, json=embeddings_body([[1.0, 0.0]])), price_per_m=2.5
    )
    stats = RemoteStats()
    assert client.encode_query("question", stats) is not None
    assert stats.cost_usd == pytest.approx(12 * 2.5 / 1_000_000)


def test_e5_models_get_the_query_and_passage_prefixes() -> None:
    assert uses_e5_prefixes("vendor/multilingual-e5-large")
    assert not uses_e5_prefixes("vendor/other-model")
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, json=embeddings_body([[1.0, 0.0]]))

    client = embeddings_client(handler, model="vendor/multilingual-e5-large")
    client.encode_query("how do I index?")
    client.encode_passages(["a passage"])
    assert seen[0]["input"] == ["query: how do I index?"]
    assert seen[1]["input"] == ["passage: a passage"]


def test_embeddings_5xx_after_the_retry_returns_none_and_counts(caplog) -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(503, text=f"upstream said: {KEY}")

    stats = RemoteStats()
    client = embeddings_client(handler)
    with caplog.at_level(logging.WARNING):
        assert client.encode_query("question", stats) is None
    # One retry, like the generation client.
    assert calls["n"] == 2
    assert (stats.embed_calls, stats.embed_errors) == (1, 1)
    assert stats.cost_usd == 0.0
    assert KEY not in caplog.text


@pytest.mark.parametrize(
    "body",
    [
        {"data": []},  # no vectors
        {"data": [{"embedding": [1.0, 0.0]}] * 2},  # duplicate indexes
        {"data": [{"index": 0, "embedding": ["x"]}]},  # non-numeric
        {"data": [{"index": 0, "embedding": []}]},  # empty vector
        {"usage": {}},  # no data at all
        [1, 2, 3],  # not an object
    ],
)
def test_embeddings_malformed_bodies_return_none(body) -> None:
    stats = RemoteStats()
    client = embeddings_client(lambda request: httpx.Response(200, json=body))
    assert client.encode_query("question", stats) is None
    assert stats.embed_errors == 1


def test_embeddings_timeout_returns_none_with_a_key_free_message(caplog) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    stats = RemoteStats()
    client = embeddings_client(handler)
    with caplog.at_level(logging.WARNING):
        assert client.encode_query("question", stats) is None
    assert "timed out" in caplog.text
    assert KEY not in caplog.text
    assert stats.embed_errors == 1


def test_embeddings_4xx_is_not_retried() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(401, json={"error": {"message": f"bad key {KEY}"}})

    stats = RemoteStats()
    client = embeddings_client(handler)
    assert client.encode_query("question", stats) is None
    assert calls["n"] == 1


def test_the_key_never_reaches_an_error_message(caplog) -> None:
    client = embeddings_client(
        lambda request: httpx.Response(400, json={"error": {"message": f"invalid token {KEY}"}})
    )
    with caplog.at_level(logging.WARNING):
        assert client.encode_query("question", RemoteStats()) is None
    assert KEY not in caplog.text
    assert "***" in caplog.text
    assert KEY not in repr(client)


# ---------------------------------------------------------------------- rerank
def test_rerank_parses_index_and_relevance_and_sorts_best_first() -> None:
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={"results": [{"index": 2, "relevance_score": 0.2}, {"index": 0, "relevance_score": 0.9}]},
        )

    client = reranker(handler)
    ranked = client.rerank("q", ["a", "b", "c"], top_n=3)
    assert ranked == [(0, 0.9), (2, 0.2)]
    assert seen[0]["top_n"] == 3
    assert seen[0]["documents"] == ["a", "b", "c"]


def test_rerank_accepts_an_alternative_response_shape() -> None:
    """The parser adapts to what the endpoint returns: ``data``/``score``/``document_index``."""
    client = reranker(
        lambda request: httpx.Response(200, json={"data": [{"document_index": 1, "score": 3.5}]})
    )
    assert client.rerank("q", ["a", "b"]) == [(1, 3.5)]


def test_rerank_logs_the_shape_of_the_response_without_the_key(caplog) -> None:
    client = reranker(
        lambda request: httpx.Response(
            200, json={"results": [{"index": 0, "relevance_score": 1.0}], "usage": {"cost": 0.0001}}
        )
    )
    with caplog.at_level(logging.INFO):
        client.rerank("q", ["a"])
    assert "rerank response shape" in caplog.text
    assert "relevance_score" in caplog.text
    assert KEY not in caplog.text


def test_rerank_failure_returns_none_and_counts() -> None:
    stats = RemoteStats()
    client = reranker(lambda request: httpx.Response(500, json={"error": {"message": "boom"}}))
    assert client.rerank("q", ["a", "b"], stats=stats) is None
    assert stats.rerank_errors == 1
    assert stats.rerank_calls == 1


def test_rerank_uses_usage_cost_or_the_price_per_search() -> None:
    with_cost = RemoteStats()
    reranker(lambda request: httpx.Response(200, json={"results": [{"index": 0, "score": 1}], "usage": {"cost": 0.0007}})).rerank("q", ["a"], stats=with_cost)
    assert with_cost.cost_usd == pytest.approx(0.0007)

    without_cost = RemoteStats()
    reranker(lambda request: httpx.Response(200, json={"results": [{"index": 0, "score": 1}]}), price_per_search=0.002).rerank("q", ["a"], stats=without_cost)
    assert without_cost.cost_usd == pytest.approx(0.002)


# ------------------------------------------------------------------ translation
def chat_body(content: str, prompt_tokens: int = 30, completion_tokens: int = 9, cost: float | None = None) -> dict:
    usage = {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens}
    if cost is not None:
        usage["cost"] = cost
    return {"choices": [{"message": {"role": "assistant", "content": content}}], "usage": usage}


def test_translation_is_a_short_deterministic_chat_call() -> None:
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, json=chat_body("How do I create a partial index?", cost=0.00005))

    stats = RemoteStats()
    client = translator(handler)
    text = client.translate("¿Cómo creo un índice parcial?", stats)

    assert text == "How do I create a partial index?"
    assert seen[0]["temperature"] == 0
    assert seen[0]["max_tokens"] == 100
    assert (stats.translate_calls, stats.translate_errors) == (1, 0)
    assert (stats.translate_prompt_tokens, stats.translate_completion_tokens) == (30, 9)
    assert stats.cost_usd == pytest.approx(0.00005)


def test_translation_cost_falls_back_to_the_prices_per_million_tokens() -> None:
    stats = RemoteStats()
    client = translator(
        lambda request: httpx.Response(200, json=chat_body("translated")),
        price_input_per_m=1.0,
        price_output_per_m=10.0,
    )
    assert client.translate("q", stats) == "translated"
    assert stats.cost_usd == pytest.approx((30 * 1.0 + 9 * 10.0) / 1_000_000)


@pytest.mark.parametrize(
    "body",
    [
        {"choices": []},
        {"choices": [{"message": {"content": ""}}]},
        {"choices": [{"message": {"content": "x" * 2000}}]},
        {"choices": [{"message": {"content": None}}]},
    ],
)
def test_unusable_translations_return_none(body) -> None:
    stats = RemoteStats()
    assert translator(lambda request: httpx.Response(200, json=body)).translate("q", stats) is None
    assert stats.translate_errors == 1


def test_translation_5xx_returns_none() -> None:
    stats = RemoteStats()
    client = translator(lambda request: httpx.Response(502, json={"error": {"message": "provider error"}}))
    assert client.translate("q", stats) is None
    assert (stats.translate_calls, stats.translate_errors) == (1, 1)


# --------------------------------------------------------------- build_remote
def test_build_remote_is_none_when_nothing_is_configured(tmp_path) -> None:
    assert build_remote(Config(llm_base_url=BASE_URL, llm_api_key=KEY), data_dir=tmp_path) is None


def test_build_remote_needs_the_key(tmp_path) -> None:
    config = Config(llm_base_url=BASE_URL, llm_api_key="", translate_model="vendor/t")
    assert build_remote(config, data_dir=tmp_path) is None


def test_build_remote_disables_only_the_embeddings_without_an_index(tmp_path, caplog) -> None:
    config = Config(
        llm_base_url=BASE_URL,
        llm_api_key=KEY,
        embed_model="vendor/embed-model",
        embed_index="missing-index",
        translate_model="vendor/translate-model",
    )
    with caplog.at_level(logging.WARNING):
        remote = build_remote(config, data_dir=tmp_path)
    assert remote is not None
    assert remote.embeddings is None and remote.index is None
    assert remote.remote_vectors is False
    assert remote.translator is not None
    assert "reindex" in caplog.text
    assert KEY not in caplog.text


def test_build_remote_reads_the_index_and_describes_itself_without_a_route(tmp_path) -> None:
    write_index(tmp_path, "alt", ["c1", "c2"], np.asarray([[1.0, 0.0], [0.0, 1.0]]))
    config = Config(
        llm_base_url=BASE_URL,
        llm_api_key=KEY,
        embed_model="vendor/embed-model",
        embed_index="alt",
        rerank_model="vendor/rerank-model",
        translate_model="vendor/translate-model",
    )
    remote = build_remote(config, data_dir=tmp_path)
    assert remote is not None and remote.remote_vectors
    described = remote.describe()
    assert described["embeddings"] is True and described["rerank"] is True
    assert described["embed_model"] == "embed-model"
    assert "vendor" not in json.dumps(described)
    assert KEY not in json.dumps(described)
    remote.close()


# ------------------------------------------------------- search integration
def test_rerank_reorders_the_fragments_before_grouping() -> None:
    chunks = [
        make_chunk("a1", "https://example.test/a.html#A", "alpha"),
        make_chunk("b1", "https://example.test/b.html#B", "beta"),
    ]
    store = RecordingStore(chunks, lexical=["a1", "b1"], vector=[])
    remote = RemoteRetrieval(
        reranker=reranker(
            lambda request: httpx.Response(
                200, json={"results": [{"index": 1, "relevance_score": 9.0}, {"index": 0, "relevance_score": 1.0}]}
            )
        )
    )
    results = hybrid_search("question", policy="baseline", store=store, embedder=RecordingEmbedder(), remote=remote)
    assert [result["page_url"] for result in results] == [
        "https://example.test/b.html",
        "https://example.test/a.html",
    ]
    # The group score is still the RRF score, not the reranker relevance.
    assert results[0]["rrf"] == pytest.approx(1 / 62)
    assert results[0]["rerank_score"] == pytest.approx(9.0)


def test_rerank_failure_keeps_the_fused_order_and_counts_a_fallback() -> None:
    chunks = [
        make_chunk("a1", "https://example.test/a.html#A", "alpha"),
        make_chunk("b1", "https://example.test/b.html#B", "beta"),
    ]
    store = RecordingStore(chunks, lexical=["a1", "b1"], vector=[])
    remote = RemoteRetrieval(reranker=reranker(lambda request: httpx.Response(500, json={})))
    results = hybrid_search("question", policy="baseline", store=store, embedder=RecordingEmbedder(), remote=remote)
    assert [result["page_url"] for result in results] == [
        "https://example.test/a.html",
        "https://example.test/b.html",
    ]
    assert remote.stats.fallbacks == 1
    assert remote.stats.rerank_errors == 1


def test_remote_embeddings_use_their_own_index_and_never_load_the_local_model(tmp_path, monkeypatch) -> None:
    write_index(tmp_path, "alt", ["remote1", "remote2"], np.asarray([[1.0, 0.0], [0.0, 1.0]]))
    chunks = [
        make_chunk("remote1", "https://example.test/r1.html#A", "one"),
        make_chunk("remote2", "https://example.test/r2.html#B", "two"),
    ]
    store = RecordingStore(chunks, lexical=[], vector=[])

    def forbidden():
        raise AssertionError("the local embedding model must not be loaded")

    monkeypatch.setattr("retrieval.search.get_embedder", forbidden)
    remote = RemoteRetrieval(
        embeddings=embeddings_client(
            lambda request: httpx.Response(200, json=embeddings_body([[1.0, 0.0]], cost=0.0)), dimensions=None
        ),
        index=VectorIndex(vectors_path_for("alt", tmp_path), vector_ids_path_for("alt", tmp_path)),
    )
    info: dict = {}
    results = hybrid_search(
        "question", policy="baseline", store=store, embedder=None, remote=remote, info=info
    )
    assert [result["chunk_ids"] for result in results] == [["remote1"], ["remote2"]]
    assert results[0]["page_url"] == "https://example.test/r1.html"
    assert info == {"embedding": "remote", "vector_cosine": pytest.approx(1.0)}


def test_a_failed_remote_embedding_falls_back_to_the_local_index(monkeypatch) -> None:
    chunks = [make_chunk("c1", "https://example.test/a.html#A", "one")]
    store = RecordingStore(chunks, lexical=[], vector=["c1"])
    embedder = RecordingEmbedder()
    remote = RemoteRetrieval(
        embeddings=embeddings_client(lambda request: httpx.Response(500, json={})),
        index=VectorIndex("does-not-exist.npy", "does-not-exist.json"),
    )
    remote.index = type("Index", (), {"search": lambda self, qvec, k: [("never", 1.0)]})()
    info: dict = {}
    results = hybrid_search(
        "question", policy="baseline", store=store, embedder=embedder, remote=remote, info=info
    )
    assert results[0]["chunk_ids"] == ["c1"]
    assert info["embedding"] == "local"
    assert remote.stats.fallbacks == 1
    assert remote.stats.embed_errors == 1


def test_a_remote_index_of_the_wrong_dimension_falls_back_instead_of_failing(tmp_path, caplog) -> None:
    write_index(tmp_path, "alt", ["x1"], np.asarray([[1.0, 0.0, 0.0]]))
    chunks = [make_chunk("c1", "https://example.test/a.html#A", "one")]
    store = RecordingStore(chunks, lexical=[], vector=["c1"])
    remote = RemoteRetrieval(
        embeddings=embeddings_client(lambda request: httpx.Response(200, json=embeddings_body([[1.0, 0.0]]))),
        index=VectorIndex(vectors_path_for("alt", tmp_path), vector_ids_path_for("alt", tmp_path)),
    )
    with caplog.at_level(logging.WARNING):
        results = hybrid_search(
            "question", policy="baseline", store=store, embedder=RecordingEmbedder(), remote=remote
        )
    assert results[0]["chunk_ids"] == ["c1"]
    assert remote.stats.fallbacks == 1 and remote.stats.embed_errors == 1
    assert "remote index unusable" in caplog.text


def test_translation_changes_only_the_lexical_query() -> None:
    chunks = [
        make_chunk("a1", "https://example.test/a.html#A", "partial index"),
    ]
    store = RecordingStore(chunks, lexical=["a1"], vector=[])
    embedder = RecordingEmbedder()
    caller: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        caller.append(json.loads(request.content))
        return httpx.Response(200, json=chat_body("How do I create a partial index?"))

    remote = RemoteRetrieval(translator=translator(handler))
    question = "¿Cómo creo un índice parcial?"
    results = hybrid_search(question, policy="baseline", store=store, embedder=embedder, remote=remote)

    assert store.lexical_queries == ["How do I create a partial index?"]  # lexical: translated
    assert embedder.queries == [question]  # vector: original
    assert caller[0]["messages"][-1]["content"] == question
    assert results[0]["page_url"] == "https://example.test/a.html"


def test_english_questions_are_not_translated() -> None:
    chunks = [make_chunk("a1", "https://example.test/a.html#A", "text")]
    store = RecordingStore(chunks, lexical=["a1"], vector=[])
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(200, json=chat_body("should not be used"))

    remote = RemoteRetrieval(translator=translator(handler))
    question = "How do I create a partial index?"
    hybrid_search(question, policy="baseline", store=store, embedder=RecordingEmbedder(), remote=remote)
    assert calls["n"] == 0
    assert store.lexical_queries == [question]


def test_a_failed_translation_keeps_the_local_lexical_behaviour() -> None:
    chunks = [make_chunk("a1", "https://example.test/a.html#A", "text")]
    store = RecordingStore(chunks, lexical=["a1"], vector=[])
    remote = RemoteRetrieval(translator=translator(lambda request: httpx.Response(502, json={})))
    hybrid_search(
        "¿Cómo creo un índice parcial?",
        policy="es-stopwords",
        store=store,
        embedder=RecordingEmbedder(),
        remote=remote,
    )
    assert store.lexical_queries == ["creo un índice parcial"]  # the stopword policy still applies
    assert remote.stats.fallbacks == 1


# ------------------------------------------------------------------ cost to app
def test_usage_add_cost_accumulates_in_the_month(tmp_path) -> None:
    from app.usage import UsageStore

    usage = UsageStore(tmp_path / "usage.sqlite", 0.5)
    assert usage.add_cost(0.0004) == pytest.approx(0.0004)
    assert usage.add_cost(0.0006) == pytest.approx(0.001)
    assert usage.add_cost(0.0) == pytest.approx(0.001)  # nothing to add: no row needed
    assert usage.tokens()[0] == 0  # a remote cost carries no token count


def test_the_api_charges_the_remote_cost_of_one_request(tmp_path) -> None:
    from app.main import create_app
    from app.usage import UsageStore
    from fastapi.testclient import TestClient

    from tests.helpers import FakeEmbedder, FakeStore, ScriptedLLM, make_settings

    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(200, json=chat_body("How do I create a partial index?", cost=0.00025))

    settings = make_settings(tmp_path, llm_price_input_per_m=1.0, llm_price_output_per_m=1.0)
    usage = UsageStore(settings.usage_db, settings.budget_usd_per_month, input_price_per_m=1.0, output_price_per_m=1.0)
    remote = RemoteRetrieval(translator=translator(handler))
    application = create_app(
        settings,
        store=FakeStore(),
        embedder=FakeEmbedder(),
        llm=ScriptedLLM(['{"abstain": true, "sentences": []}']),
        usage=usage,
        remote=remote,
        threshold={"signal": "vector_cosine", "value": 0.0},
    )
    client = TestClient(application)
    response = client.post("/api/ask", json={"question": "¿Cómo creo un índice parcial?"})
    assert response.status_code == 200
    assert calls["n"] == 1
    # The remote cost and the (mock, priced at 1.0/M) completion both land in the month.
    assert usage.spent() > 0.00025
    assert usage.status()["spent_usd"] == pytest.approx(usage.spent())


def test_the_api_reports_no_remote_parts_by_default(tmp_path) -> None:
    from fastapi.testclient import TestClient

    from tests.helpers import make_client

    client, _ = make_client(tmp_path)
    body = client.get("/api/health").json()
    assert body["retrieval_remote"] == {"embeddings": False, "rerank": False, "translate": False}


# ------------------------------------------------------------------ small units
def test_short_model_drops_the_provider_route() -> None:
    assert short_model("vendor/family/model-v1") == "model-v1"
    assert short_model("model-v1") == "model-v1"


def test_vector_index_reads_a_named_pair_and_ignores_a_missing_one(tmp_path) -> None:
    index = write_index(tmp_path, "alt", ["a", "b"], np.asarray([[1.0, 0.0], [0.0, 1.0]]))
    assert index.size == 2 and index.dim == 2
    assert index.search(np.asarray([0.9, 0.1], dtype=np.float32), 1) == [("a", pytest.approx(0.9938837))]

    empty = VectorIndex(tmp_path / "nope.npy", tmp_path / "nope.json")
    assert empty.exists() is False
    assert empty.search(np.asarray([1.0, 0.0], dtype=np.float32), 3) == []


def test_vector_index_rejects_a_query_of_the_wrong_dimension(tmp_path) -> None:
    index = write_index(tmp_path, "alt", ["a"], np.asarray([[1.0, 0.0]]))
    with pytest.raises(ValueError):
        index.search(np.asarray([1.0, 0.0, 0.0], dtype=np.float32), 1)


def test_stats_delta_between_two_requests() -> None:
    total = RemoteStats()
    before = RemoteStats()
    total.embed_calls += 2
    total.embed_tokens += 40
    total.fallbacks += 1
    total.add_cost(0.0002)
    change = total.delta(before)
    assert (change.embed_calls, change.embed_tokens, change.fallbacks) == (2, 40, 1)
    assert change.cost_usd == pytest.approx(0.0002)


def test_remote_error_is_a_runtime_error_without_state() -> None:
    error = RemoteError("rerank request failed with HTTP 500")
    assert isinstance(error, RuntimeError)
    assert str(error) == "rerank request failed with HTTP 500"


# --------------------------------------------------------------- configuration
def test_the_remote_variables_are_off_by_default_and_parse_from_the_environment(tmp_path) -> None:
    from generation.config import load_config

    config = load_config({})
    assert config.remote_active is False
    assert (config.embed_price_per_m, config.rerank_price_per_search) == (0.0, 0.0)

    configured = load_config(
        {
            "GROUNDED_RAG_LLM_BASE_URL": BASE_URL,
            "GROUNDED_RAG_RERANK_MODEL": "vendor/rerank-model",
            "GROUNDED_RAG_EMBED_MODEL": "vendor/embed-model",
            "GROUNDED_RAG_EMBED_INDEX": "alt",
            "GROUNDED_RAG_TRANSLATE_MODEL": "vendor/translate-model",
            "GROUNDED_RAG_EMBED_PRICE_PER_M": "2.5",
            "GROUNDED_RAG_RERANK_PRICE_PER_SEARCH": "0.002",
            "GROUNDED_RAG_TRANSLATE_PRICE_INPUT_PER_M": "1",
            "GROUNDED_RAG_TRANSLATE_PRICE_OUTPUT_PER_M": "10",
        }
    )
    assert configured.remote_active is True
    assert configured.embed_index_name == "alt"
    assert configured.embed_price_per_m == pytest.approx(2.5)
    assert configured.translate_price_output_per_m == pytest.approx(10.0)
    # The public report shape is unchanged: no remote model, no price, no route.
    assert set(configured.public()) == {
        "llm_model",
        "llm_price_input_per_m",
        "llm_price_output_per_m",
        "budget_usd_per_month",
        "generation",
    }


def test_a_missing_index_name_falls_back_to_a_slug_of_the_model() -> None:
    config = Config(llm_base_url=BASE_URL, llm_api_key=KEY, embed_model="vendor/Multilingual-E5 Large")
    assert config.embed_index_name == "multilingual-e5-large"
    assert Config(llm_base_url=BASE_URL).embed_index_name == ""


def test_an_unknown_remote_price_is_a_config_error() -> None:
    from generation.config import ConfigError, load_config

    with pytest.raises(ConfigError):
        load_config({"GROUNDED_RAG_EMBED_PRICE_PER_M": "-1"})
