"""Eval URL validation: expected URLs missing from the corpus are excluded."""

from __future__ import annotations

from eval.__main__ import url_exists, validate_urls

PAGES = {"https://d/x.html", "https://d/y.html"}
URLS = {"https://d/x.html#X", "https://d/y.html#Y"}

QUESTIONS = [
    {"id": "q1", "lang": "es", "question": "a", "answerable": True, "expected_urls": ["https://d/x.html#X"]},
    {"id": "q2", "lang": "es", "question": "b", "answerable": True, "expected_urls": ["https://d/x.html#NOPE"]},
    {
        "id": "q3",
        "lang": "es",
        "question": "c",
        "answerable": True,
        "expected_urls": ["https://d/y.html", "https://d/z.html"],  # page-level + a typo
    },
]


def test_page_url_exists_by_prefix() -> None:
    assert url_exists("https://d/y.html", PAGES, URLS)
    assert not url_exists("https://d/z.html", PAGES, URLS)


def test_anchor_url_must_exist_exactly() -> None:
    assert url_exists("https://d/x.html#X", PAGES, URLS)
    assert not url_exists("https://d/x.html#NOPE", PAGES, URLS)


def test_missing_urls_are_reported_and_dropped() -> None:
    cleaned, problems = validate_urls(QUESTIONS, PAGES, URLS)

    assert problems == [("q2", "https://d/x.html#NOPE"), ("q3", "https://d/z.html")]
    assert cleaned[0]["expected_urls"] == ["https://d/x.html#X"]
    assert cleaned[0]["excluded"] is False
    assert cleaned[1]["expected_urls"] == []
    assert cleaned[1]["excluded"] is True  # nothing left to score against
    assert cleaned[2]["expected_urls"] == ["https://d/y.html"]  # the valid one survives
    assert cleaned[2]["invalid_urls"] == ["https://d/z.html"]


def test_validation_does_not_mutate_the_input() -> None:
    validate_urls(QUESTIONS, PAGES, URLS)

    assert QUESTIONS[1]["expected_urls"] == ["https://d/x.html#NOPE"]
