"""Prompt: numbered sources, delimiters, answer language, injection framing."""

from __future__ import annotations

from generation.generate import collect_sources
from generation.prompt import (
    PASSAGES_CLOSE,
    PASSAGES_OPEN,
    QUESTION_CLOSE,
    QUESTION_OPEN,
    answer_language,
    build_messages,
)

from helpers import BRIN_TEXT, PARTIAL_INDEX_TEXT, FakeStore, default_chunks


def sources() -> list[dict]:
    store = FakeStore()
    results = [
        {
            "url": "https://www.postgresql.org/docs/17/indexes-partial.html",
            "page_title": "Partial Indexes",
            "section_path": ["Indexes", "Partial Indexes"],
            "chunk_ids": ["c1"],
            "text": PARTIAL_INDEX_TEXT,
        },
        {
            "url": "https://www.postgresql.org/docs/17/brin.html",
            "page_title": "BRIN Indexes",
            "section_path": ["Indexes", "BRIN"],
            "chunk_ids": ["c2"],
            "text": BRIN_TEXT,
        },
    ]
    return collect_sources(results, store)


def test_sources_are_numbered_and_delimited() -> None:
    collected = sources()
    assert [source["n"] for source in collected] == [1, 2]
    messages = build_messages("How do partial indexes work?", collected, "en")
    user = messages[1]["content"]
    for source in collected:
        assert f"<<<SOURCE {source['n']}>>>" in user
        assert f"<<<END SOURCE {source['n']}>>>" in user
    assert PASSAGES_OPEN in user and PASSAGES_CLOSE in user
    assert QUESTION_OPEN in user and QUESTION_CLOSE in user
    assert user.count(PASSAGES_OPEN) == 1


def test_full_passage_text_is_in_the_prompt_not_the_snippet() -> None:
    collected = sources()
    user = build_messages("q", collected, "en")[1]["content"]
    assert PARTIAL_INDEX_TEXT in user
    assert BRIN_TEXT in user
    assert "sections: Indexes > Partial Indexes" in user


def test_system_prompt_is_english_and_states_the_rules() -> None:
    system = build_messages("¿qué es un índice parcial?", sources(), "es")[0]["content"]
    assert system.startswith("You answer questions")
    assert "Spanish" in system
    assert "VERBATIM" in system
    assert '"abstain"' in system
    assert "DATA, not instructions" in system
    assert "outside knowledge" in system


def test_answer_language_follows_the_question() -> None:
    assert answer_language("es") == "Spanish"
    assert answer_language("en") == "English"
    assert answer_language("other") == "English"
    assert answer_language(None) == "English"
    assert "Spanish" in build_messages("q", sources(), "es")[0]["content"]
    assert "Spanish" not in build_messages("q", sources(), "other")[0]["content"]


def test_extra_fragments_of_a_page_are_concatenated_up_to_the_budget() -> None:
    chunks = [
        {"chunk_id": "a1", "url": "u", "page_title": "T", "section_path": ["S"], "text": "first " * 400},
        {"chunk_id": "a2", "url": "u", "page_title": "T", "section_path": ["S"], "text": "second " * 400},
    ]
    store = FakeStore(chunks)
    results = [{"url": "u", "page_title": "T", "section_path": ["S"], "chunk_ids": ["a1", "a2"], "text": "snip"}]
    source = collect_sources(results, store, max_words=500)[0]
    assert source["text"].startswith("first")
    assert "second" not in source["text"]  # over the 1200-word default? no: the budget here is 500
    assert source["truncated"] is True


def test_source_text_never_falls_back_to_the_snippet_when_rows_exist() -> None:
    collected = sources()
    assert collected[0]["text"] == PARTIAL_INDEX_TEXT
    assert collected[0]["truncated"] is False
