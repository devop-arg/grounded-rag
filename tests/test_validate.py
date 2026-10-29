"""Citation validation: range, verbatim text, length, normalization, bad JSON."""

from __future__ import annotations

import json

from generation.validate import (
    MIN_QUOTE_WORDS,
    citation_is_valid,
    normalize,
    parse_model_output,
    validate_answer,
)

SOURCE = (
    "A partial index is an index built over a subset of a table; the subset is defined by a "
    "conditional expression. It is useful for queries that only need to scan a few rows, and it "
    "cannot be used for queries that do not match the condition."
)
OTHER = "BRIN indexes store summaries of the values in each block range and stay very small."
SOURCES = [SOURCE, OTHER]


def payload(*sentences: dict) -> dict:
    return {"abstain": False, "sentences": list(sentences)}


def cite(n: int, quote: str) -> dict:
    return {"text": "sentence", "citations": [{"n": n, "quote": quote}]}


GOOD_QUOTE = "A partial index is an index built over a subset of a table"


def test_valid_citation_keeps_the_sentence() -> None:
    abstain, sentences, cited, stats, outcome = validate_answer(payload(cite(1, GOOD_QUOTE)), SOURCES)
    assert abstain is False
    assert outcome == "answered"
    assert cited == [1]
    assert sentences[0]["citations"] == [{"n": 1, "quote": GOOD_QUOTE}]
    assert stats["citations_total"] == 1 and stats["citations_valid"] == 1
    assert stats["sentences_total"] == 1 and stats["sentences_kept"] == 1


def test_out_of_range_number_is_dropped() -> None:
    abstain, sentences, cited, stats, outcome = validate_answer(payload(cite(3, GOOD_QUOTE)), SOURCES)
    assert abstain is True and sentences == [] and cited == []
    assert outcome == "no_valid_citation"
    assert stats["citations_total"] == 1 and stats["citations_valid"] == 0


def test_zero_and_negative_numbers_are_dropped() -> None:
    for number in (0, -1):
        abstain, _, _, stats, _ = validate_answer(payload(cite(number, GOOD_QUOTE)), SOURCES)
        assert abstain is True
        assert stats["citations_dropped"] == 1


def test_non_verbatim_quote_is_dropped() -> None:
    paraphrase = "A partial index only covers a part of the table rows"
    abstain, _, _, stats, _ = validate_answer(payload(cite(1, paraphrase)), SOURCES)
    assert abstain is True
    assert stats["citations_dropped"] == 1


def test_quote_from_the_wrong_source_is_dropped() -> None:
    abstain, _, _, _, _ = validate_answer(payload(cite(2, GOOD_QUOTE)), SOURCES)
    assert abstain is True


def test_short_quote_is_dropped() -> None:
    short = "permission denied must never be used as a quote"  # 9 words in one source, not this
    three_words = "A partial index"
    assert len(three_words.split()) < MIN_QUOTE_WORDS
    abstain, _, _, stats, _ = validate_answer(payload(cite(1, three_words)), SOURCES)
    assert abstain is True
    assert stats["citations_dropped"] == 1
    assert citation_is_valid({"n": 1, "quote": short}, [normalize(SOURCE), normalize(OTHER)]) is False


def test_normalization_accepts_case_whitespace_and_typographic_quotes() -> None:
    source = 'It is "useful" for queries   that only need\nto scan a few rows.'
    quote = "IT IS «USEFUL» FOR QUERIES THAT ONLY NEED TO SCAN A FEW ROWS"
    assert normalize('It is "useful" for queries that only need to scan a few rows.') == normalize(source)
    abstain, _, _, stats, _ = validate_answer(payload(cite(1, quote)), [source])
    assert abstain is False
    assert stats["citations_valid"] == 1


def test_nfkc_normalization_folds_ligatures_and_fullwidth() -> None:
    source = "The default is file based logging here and it works well."
    quote = "the default is ﬁle based logging here"  # U+FB01 ligature
    assert normalize("ﬁ") == "fi"
    abstain, _, _, _, _ = validate_answer(payload(cite(1, quote)), [source])
    assert abstain is False


def test_sentence_without_citations_is_dropped_but_the_other_survives() -> None:
    data = payload(
        {"text": "No citation here.", "citations": []},
        cite(1, GOOD_QUOTE),
    )
    abstain, sentences, _, stats, outcome = validate_answer(data, SOURCES)
    assert abstain is False and outcome == "answered"
    assert len(sentences) == 1
    assert stats["sentences_total"] == 2
    assert stats["sentences_with_valid_citation"] == 1
    assert stats["sentences_kept"] == 1


def test_all_sentences_dropped_means_abstain() -> None:
    data = payload(
        {"text": "Made up.", "citations": [{"n": 1, "quote": "this quote is not in the passage at all"}]},
    )
    abstain, sentences, cited, stats, outcome = validate_answer(data, SOURCES)
    assert abstain is True and sentences == [] and cited == []
    assert outcome == "no_valid_citation"
    assert stats["citations_total"] == 1 and stats["citations_valid"] == 0


def test_model_abstain_is_respected_even_with_sentences() -> None:
    data = {"abstain": True, "sentences": [cite(1, GOOD_QUOTE)]}
    abstain, sentences, cited, stats, outcome = validate_answer(data, SOURCES)
    assert abstain is True and sentences == [] and cited == []
    assert outcome == "model_abstain"
    assert stats["sentences_total"] == 0


def test_malformed_payload_is_an_abstention() -> None:
    abstain, sentences, cited, stats, outcome = validate_answer(None, SOURCES)
    assert abstain is True and sentences == [] and cited == []
    assert outcome == "invalid_payload"
    assert stats["sentences_total"] == 0


def test_well_formed_payload_without_sentences_is_an_abstention() -> None:
    for bad in ({}, {"abstain": False}, {"abstain": False, "sentences": "nope"}):
        abstain, sentences, cited, stats, outcome = validate_answer(bad, SOURCES)
        assert abstain is True and sentences == [] and cited == []
        assert outcome == "no_valid_citation"
        assert stats["sentences_total"] == 0


def test_bad_citation_shapes_are_dropped() -> None:
    data = {
        "abstain": False,
        "sentences": [
            {"text": "a", "citations": [{"n": True, "quote": GOOD_QUOTE}]},
            {"text": "b", "citations": [{"n": "1", "quote": GOOD_QUOTE}]},
            {"text": "c", "citations": [{"n": 1, "quote": None}]},
            {"text": "d", "citations": "nope"},
        ],
    }
    abstain, _, _, stats, _ = validate_answer(data, SOURCES)
    assert abstain is True
    assert stats["citations_total"] == 3
    assert stats["citations_valid"] == 0


def test_parse_plain_json() -> None:
    assert parse_model_output('{"abstain": false}') == {"abstain": False}


def test_parse_json_inside_a_code_fence() -> None:
    text = 'Here you go:\n```json\n{"abstain": true, "sentences": []}\n```\nthanks'
    assert parse_model_output(text) == {"abstain": True, "sentences": []}


def test_parse_json_with_extra_prose_around_it() -> None:
    text = 'Sure! {"abstain": false, "sentences": [{"text": "x", "citations": []}]} hope it helps'
    assert parse_model_output(text)["abstain"] is False


def test_parse_braces_inside_strings() -> None:
    text = 'prefix {"abstain": false, "sentences": [{"text": "a } b", "citations": []}]} suffix'
    parsed = parse_model_output(text)
    assert parsed is not None
    assert parsed["sentences"][0]["text"] == "a } b"


def test_broken_json_is_not_parsed() -> None:
    assert parse_model_output('{"abstain": tru') is None
    assert parse_model_output("") is None
    assert parse_model_output("[1, 2, 3]") is None


def test_end_to_end_with_fenced_json_counts_stats() -> None:
    data = payload(cite(1, GOOD_QUOTE))
    parsed = parse_model_output("```json\n" + json.dumps(data) + "\n```")
    abstain, sentences, cited, stats, outcome = validate_answer(parsed, SOURCES)
    assert abstain is False
    assert outcome == "answered"
    assert cited == [1]
    assert stats["citations_valid"] == 1
