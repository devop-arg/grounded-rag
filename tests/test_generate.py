"""Pipeline: retrieval -> threshold -> model -> validated answer (all offline)."""

from __future__ import annotations

from pathlib import Path

import httpx

from generation.generate import (
    Thresholds,
    answer_question,
    load_threshold,
    load_thresholds,
)
from generation.mock import MockLLM
from retrieval.remote import RemoteEmbeddings, RemoteRetrieval

from helpers import BRIN_TEXT, PARTIAL_INDEX_TEXT, FakeEmbedder, FakeStore, ScriptedLLM, make_settings

GOOD_QUOTE = "A partial index is an index built over a subset of a table"
OTHER_QUOTE = "BRIN indexes store summaries of the values in each block range"
REMOTE_BASE_URL = "https://llm.example/v1"


class FakeIndex:
    """A vector index with fixed hits: no file, no numpy, no dimension check."""

    def __init__(self, hits: list[tuple[str, float]]) -> None:
        self.hits = list(hits)

    def search(self, qvec, k: int) -> list[tuple[str, float]]:
        return self.hits[:k]


def embedding_body(vector: list[float], tokens: int = 8) -> dict:
    return {
        "data": [{"object": "embedding", "index": 0, "embedding": vector}],
        "usage": {"prompt_tokens": tokens},
    }


def remote_retrieval(hits: list[tuple[str, float]], *, fail: bool = False) -> RemoteRetrieval:
    """A remote query embedding (served by a mock transport) over a fake index."""

    def handler(request: httpx.Request) -> httpx.Response:
        if fail:
            return httpx.Response(500, json={})
        return httpx.Response(200, json=embedding_body([1.0, 0.0]))

    embeddings = RemoteEmbeddings(
        REMOTE_BASE_URL,
        "sk-test-value",
        "vendor/embed-model",
        transport=httpx.MockTransport(handler),
        sleep=lambda seconds: None,
    )
    return RemoteRetrieval(embeddings=embeddings, index=FakeIndex(hits))


def answer(llm, **kwargs):
    settings = kwargs.pop("config", None) or make_settings(Path("/tmp"))
    use_threshold = kwargs.pop("use_threshold", False)
    return answer_question(
        kwargs.pop("question", "How do partial indexes help?"),
        kwargs.pop("lang", "en"),
        config=settings,
        store=kwargs.pop("store", FakeStore()),
        embedder=kwargs.pop("embedder", FakeEmbedder()),
        llm=llm,
        use_threshold=use_threshold,
        **kwargs,
    )


def test_answered_question_returns_validated_sentences() -> None:
    llm = ScriptedLLM([{"abstain": False, "sentences": [{"text": "They restrict the rows.", "citations": [{"n": 1, "quote": GOOD_QUOTE}]}]}])
    result = answer(llm)
    assert result["abstain"] is False
    assert result["reason"] is None
    assert result["sentences"][0]["citations"] == [{"n": 1, "quote": GOOD_QUOTE}]
    assert result["sources"] == [
        {
            "n": 1,
            "url": "https://www.postgresql.org/docs/17/indexes-partial.html",
            "title": "Partial Indexes",
        }
    ]
    assert result["prompt_tokens"] == 100 and result["completion_tokens"] == 20
    assert result["llm_ms"] >= 0.0
    assert result["outcome"] == "answered"


def test_sources_are_only_the_cited_ones_and_keep_their_number() -> None:
    llm = ScriptedLLM(
        [
            {
                "abstain": False,
                "sentences": [
                    {"text": "BRIN is small.", "citations": [{"n": 2, "quote": OTHER_QUOTE}]},
                ],
            }
        ]
    )
    result = answer(llm)
    assert [source["n"] for source in result["sources"]] == [2]
    assert result["sources"][0]["url"].endswith("brin.html")


def test_abstention_below_the_threshold_does_not_call_the_model() -> None:
    llm = ScriptedLLM([{"abstain": False, "sentences": []}])
    result = answer(
        llm, thresholds=Thresholds(local={"signal": "vector_cosine", "value": 0.99}), use_threshold=True
    )
    assert result["abstain"] is True
    assert result["reason"] == "threshold"
    assert result["sentences"] == [] and result["sources"] == []
    assert result["prompt_tokens"] == 0
    assert llm.calls == 0
    assert result["signals"]["vector_cosine"] > 0.8


def test_signal_above_the_threshold_reaches_the_model() -> None:
    llm = ScriptedLLM([{"abstain": True, "sentences": []}])
    result = answer(
        llm, thresholds=Thresholds(local={"signal": "vector_cosine", "value": 0.5}), use_threshold=True
    )
    assert llm.calls == 1
    assert result["reason"] == "model"


def test_model_abstention_is_kept() -> None:
    llm = ScriptedLLM([{"abstain": True, "sentences": []}])
    result = answer(llm)
    assert result["abstain"] is True
    assert result["reason"] == "model"
    assert result["sources"] == []


def test_unparseable_json_is_an_abstention_by_validation() -> None:
    llm = ScriptedLLM(["I cannot answer that, sorry."])
    result = answer(llm)
    assert result["abstain"] is True
    assert result["reason"] == "validation"
    assert result["outcome"] == "invalid_payload"


def test_non_verbatim_citation_is_dropped_on_the_server() -> None:
    llm = ScriptedLLM(
        [
            {
                "abstain": False,
                "sentences": [
                    {"text": "Fabricated.", "citations": [{"n": 1, "quote": "partial indexes solve every performance problem"}]}
                ],
            }
        ]
    )
    result = answer(llm)
    assert result["abstain"] is True
    assert result["reason"] == "validation"
    assert result["stats"]["citations_valid"] == 0
    assert result["stats"]["citations_total"] == 1


def test_mock_model_answers_with_a_verbatim_quote() -> None:
    llm = MockLLM()
    result = answer(llm)
    assert result["abstain"] is False
    assert result["sentences"][0]["citations"][0]["n"] == 1
    assert result["stats"]["citations_valid"] == 1
    assert result["prompt_tokens"] > 0 and result["completion_tokens"] > 0


def test_mock_model_abstains_when_the_passage_is_too_short_to_quote() -> None:
    chunks = [{"chunk_id": "x", "url": "u", "page_title": "t", "section_path": [], "text": "tiny"}]
    result = answer(MockLLM(), store=FakeStore(chunks))
    assert result["abstain"] is True
    assert result["reason"] == "model"


def test_language_detection_is_used_when_not_given() -> None:
    llm = ScriptedLLM([{"abstain": True, "sentences": []}])
    result = answer_question(
        "¿cómo configuro un índice parcial?",
        config=make_settings(Path("/tmp")),
        store=FakeStore(),
        embedder=FakeEmbedder(),
        llm=llm,
        use_threshold=False,
    )
    assert result["lang"] == "es"
    assert "Spanish" in llm.messages[0][0]["content"]


def test_request_language_overrides_detection() -> None:
    llm = ScriptedLLM([{"abstain": True, "sentences": []}])
    result = answer(llm, lang="es")
    assert result["lang"] == "es"


def test_prompt_sent_to_the_model_contains_the_sources() -> None:
    llm = ScriptedLLM([{"abstain": True, "sentences": []}])
    answer(llm)
    user = llm.messages[0][1]["content"]
    assert "<<<SOURCE 1>>>" in user
    assert PARTIAL_INDEX_TEXT in user
    assert BRIN_TEXT in user


def test_threshold_file_is_valid(tmp_path) -> None:
    threshold = load_threshold()
    assert threshold is not None
    assert threshold["signal"] in {"vector_cosine", "rrf"}
    assert isinstance(threshold["value"], float)
    assert load_threshold(tmp_path / "missing.json") is None


def test_both_threshold_files_are_read_once_per_call(tmp_path) -> None:
    thresholds = load_thresholds()
    assert thresholds.active is True
    assert thresholds.local is not None and thresholds.remote is not None
    # The remote calibration lives on its own cosine scale.
    assert thresholds.remote["signal"] == "vector_cosine"
    assert thresholds.remote["value"] == 0.436503
    assert thresholds.remote["retrieval"] == "remote"
    assert thresholds.public()["remote"] == {"signal": "vector_cosine", "value": 0.436503}
    missing = load_thresholds(
        local_path=tmp_path / "none.json", remote_path=tmp_path / "none-remote.json"
    )
    assert missing.active is False
    assert missing.for_index("remote") is None and missing.for_index("local") is None


def test_only_the_remote_threshold_file_is_missing(tmp_path) -> None:
    thresholds = load_thresholds(remote_path=tmp_path / "none-remote.json")
    assert thresholds.local is not None
    assert thresholds.remote is None
    assert thresholds.public()["remote"] is None
    # The local file is not silently reused for the remote scale.
    assert thresholds.for_index("remote") is None


def test_the_threshold_follows_the_index_the_search_really_used() -> None:
    thresholds = Thresholds(
        local={"signal": "vector_cosine", "value": 0.80},
        remote={"signal": "vector_cosine", "value": 0.40},
    )
    store = FakeStore(vector_scores={"c1": 0.10, "c2": 0.05})

    remote_llm = ScriptedLLM([{"abstain": True, "sentences": []}])
    remote = answer_question(
        "How do partial indexes work?",
        "en",
        config=make_settings(Path("/tmp")),
        store=store,
        embedder=FakeEmbedder(),
        llm=remote_llm,
        thresholds=thresholds,
        use_threshold=True,
        remote=remote_retrieval([("c1", 0.50)]),
    )
    # 0.50 is below the local 0.80 but above the remote 0.40: the model decides.
    assert remote["embedding"] == "remote"
    assert remote["reason"] == "model"
    assert remote_llm.calls == 1

    local_llm = ScriptedLLM([{"abstain": True, "sentences": []}])
    local = answer_question(
        "How do partial indexes work?",
        "en",
        config=make_settings(Path("/tmp")),
        store=store,
        embedder=FakeEmbedder(),
        llm=local_llm,
        thresholds=thresholds,
        use_threshold=True,
    )
    assert local["embedding"] == "local"
    assert local["reason"] == "threshold"
    assert local_llm.calls == 0


def test_a_failed_remote_embedding_falls_back_to_the_local_threshold() -> None:
    thresholds = Thresholds(
        local={"signal": "vector_cosine", "value": 0.80},
        remote={"signal": "vector_cosine", "value": 0.40},
    )
    llm = ScriptedLLM([{"abstain": True, "sentences": []}])
    remote = remote_retrieval([("c1", 0.50)], fail=True)
    result = answer_question(
        "How do partial indexes work?",
        "en",
        config=make_settings(Path("/tmp")),
        store=FakeStore(vector_scores={"c1": 0.10, "c2": 0.05}),
        embedder=FakeEmbedder(),
        llm=llm,
        thresholds=thresholds,
        use_threshold=True,
        remote=remote,
    )
    assert remote.stats.embed_errors == 1 and remote.stats.fallbacks == 1
    assert result["embedding"] == "local"
    assert result["signals"]["vector_cosine"] == 0.10
    assert result["reason"] == "threshold"
    assert llm.calls == 0


def test_an_index_without_a_threshold_file_has_no_threshold() -> None:
    llm = ScriptedLLM([{"abstain": True, "sentences": []}])
    result = answer_question(
        "How do partial indexes work?",
        "en",
        config=make_settings(Path("/tmp")),
        store=FakeStore(vector_scores={"c1": 0.10, "c2": 0.05}),
        embedder=FakeEmbedder(),
        llm=llm,
        thresholds=Thresholds(local={"signal": "vector_cosine", "value": 0.80}),
        use_threshold=True,
        remote=remote_retrieval([("c1", 0.10)]),
    )
    # The remote index has no calibration of its own: the local one is not reused.
    assert result["embedding"] == "remote"
    assert result["reason"] == "model"
    assert llm.calls == 1
