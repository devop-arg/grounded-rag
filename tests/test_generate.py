"""Pipeline: retrieval -> threshold -> model -> validated answer (all offline)."""

from __future__ import annotations

from pathlib import Path

from generation.generate import answer_question, load_threshold
from generation.mock import MockLLM

from helpers import BRIN_TEXT, PARTIAL_INDEX_TEXT, FakeEmbedder, FakeStore, ScriptedLLM, make_settings

GOOD_QUOTE = "A partial index is an index built over a subset of a table"
OTHER_QUOTE = "BRIN indexes store summaries of the values in each block range"


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
    result = answer(llm, threshold={"signal": "vector_cosine", "value": 0.99}, use_threshold=True)
    assert result["abstain"] is True
    assert result["reason"] == "threshold"
    assert result["sentences"] == [] and result["sources"] == []
    assert result["prompt_tokens"] == 0
    assert llm.calls == 0
    assert result["signals"]["vector_cosine"] > 0.8


def test_signal_above_the_threshold_reaches_the_model() -> None:
    llm = ScriptedLLM([{"abstain": True, "sentences": []}])
    result = answer(llm, threshold={"signal": "vector_cosine", "value": 0.5}, use_threshold=True)
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
