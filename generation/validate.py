"""Server-side validation of the model's JSON answer.

The model is untrusted: a citation only survives when its passage number is in
range, its quote is **verbatim** in that passage (after NFKC, quote
normalization, whitespace collapsing and casefolding) and the quote has at
least :data:`MIN_QUOTE_WORDS` words.  Sentences left without a valid citation
are dropped; if nothing survives — or the model abstained — the answer is an
abstention.

This module is pure (no I/O, no model) so it is tested exhaustively.
"""

from __future__ import annotations

import json
import unicodedata
from typing import Any, Sequence

MIN_QUOTE_WORDS = 4

#: Typographic quotes/apostrophes -> plain ASCII, applied before comparing.
QUOTE_TRANSLATION = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201a": "'",
        "\u201b": "'",
        "\u2032": "'",
        "\u2035": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u201e": '"',
        "\u201f": '"',
        "\u2033": '"',
        "\u00ab": '"',
        "\u00bb": '"',
        "\u300c": '"',
        "\u300d": '"',
    }
)


def normalize(text: str) -> str:
    """NFKC, straight quotes, collapsed whitespace and casefold."""
    text = unicodedata.normalize("NFKC", str(text)).translate(QUOTE_TRANSLATION)
    return " ".join(text.split()).casefold()


def count_words(text: str) -> int:
    return len(str(text).split())


def _balanced_object(text: str) -> str | None:
    """First ``{...}`` span of ``text``, respecting string literals."""
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def parse_model_output(text: str) -> dict[str, Any] | None:
    """Return the JSON object in ``text``, tolerating fences and extra prose.

    Tries the whole string, then every ```` ``` ```` block (with or without a
    ``json`` tag) and finally the first balanced object.  ``None`` when nothing
    parses or when it parses to a non-object.
    """
    if not isinstance(text, str) or not text.strip():
        return None
    candidates: list[str] = [text.strip()]
    for block in text.split("```"):
        stripped = block.strip()
        if stripped.lower().startswith("json"):
            stripped = stripped[4:].strip()
        if stripped:
            candidates.append(stripped)
    balanced = _balanced_object(text)
    if balanced:
        candidates.append(balanced)

    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except (ValueError, TypeError):
            continue
        if isinstance(value, dict):
            return value
        if isinstance(value, str):
            nested = parse_model_output(value)
            if nested is not None:
                return nested
    return None


def citation_is_valid(citation: Any, sources_norm: Sequence[str]) -> bool:
    """True when ``citation`` is a real, long-enough verbatim quote of its source."""
    if not isinstance(citation, dict):
        return False
    number = citation.get("n")
    if isinstance(number, bool) or not isinstance(number, int):
        return False
    if not 1 <= number <= len(sources_norm):
        return False
    quote = citation.get("quote")
    if not isinstance(quote, str) or count_words(quote) < MIN_QUOTE_WORDS:
        return False
    normalized = normalize(quote)
    if not normalized:
        return False
    return normalized in sources_norm[number - 1]


def new_stats() -> dict[str, int]:
    return {
        "sentences_total": 0,
        "sentences_with_valid_citation": 0,
        "sentences_kept": 0,
        "citations_total": 0,
        "citations_valid": 0,
        "citations_dropped": 0,
    }


def validate_answer(
    payload: dict[str, Any] | None,
    sources: Sequence[str],
) -> tuple[bool, list[dict], list[int], dict[str, int], str]:
    """Validate the parsed model output against the passage texts.

    Returns ``(abstain, sentences, cited_source_numbers, stats, outcome)`` where
    ``outcome`` is ``"invalid_payload"``, ``"model_abstain"``,
    ``"no_valid_citation"`` or ``"answered"``.  ``sentences`` keeps only
    sentences with at least one valid citation, and citations are normalized to
    ``{"n": int, "quote": str}``.
    """
    stats = new_stats()
    sources_norm = [normalize(source) for source in sources]

    if not isinstance(payload, dict):
        return True, [], [], stats, "invalid_payload"
    if payload.get("abstain") is True:
        return True, [], [], stats, "model_abstain"

    raw_sentences = payload.get("sentences")
    if not isinstance(raw_sentences, list):
        raw_sentences = []

    kept: list[dict] = []
    cited: list[int] = []
    for raw in raw_sentences:
        if not isinstance(raw, dict):
            stats["sentences_total"] += 1
            continue
        stats["sentences_total"] += 1
        text = raw.get("text")
        text = text.strip() if isinstance(text, str) else ""
        raw_citations = raw.get("citations")
        if not isinstance(raw_citations, list):
            raw_citations = []
        stats["citations_total"] += len(raw_citations)
        valid: list[dict] = []
        for citation in raw_citations:
            if citation_is_valid(citation, sources_norm):
                valid.append({"n": int(citation["n"]), "quote": citation["quote"]})
                if int(citation["n"]) not in cited:
                    cited.append(int(citation["n"]))
            else:
                stats["citations_dropped"] += 1
        stats["citations_valid"] += len(valid)
        if valid and text:
            stats["sentences_with_valid_citation"] += 1
            stats["sentences_kept"] += 1
            kept.append({"text": text, "citations": valid})

    if not kept:
        return True, [], [], stats, "no_valid_citation"
    return False, kept, sorted(cited), stats, "answered"
