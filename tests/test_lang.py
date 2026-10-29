"""Query language detection tests."""

from __future__ import annotations

import pytest

from retrieval.lang import detect_lang, strip_spanish_stopwords

SPANISH = [
    "¿cómo configuro la replicación lógica?",
    "cómo hago una copia de seguridad con pg_dump",
    "¿qué niveles de aislamiento soporta PostgreSQL?",
    "índice parcial",
    "¿Cómo creo un trigger en una tabla?",
    "ñandú",
]

ENGLISH = [
    "how do I create a partial index",
    "what isolation levels does PostgreSQL support",
    "which autovacuum settings should I tune",
    "How do VACUUM and ANALYZE interact",
    "the WAL and its checkpoints",
]

#: SQL/CLI tokens carry no language signal on their own.
IDENTIFIERS = [
    "pg_dump",
    "VACUUM",
    "VACUUM FULL ANALYZE",
    "pg_dump --schema-only",
    "CREATE INDEX CONCURRENTLY",
    "MVCC",
    "jsonb_path_query_first",
]


@pytest.mark.parametrize("query", SPANISH)
def test_spanish_queries(query: str) -> None:
    assert detect_lang(query) == "es"


@pytest.mark.parametrize("query", ENGLISH)
def test_english_queries(query: str) -> None:
    assert detect_lang(query) == "en"


@pytest.mark.parametrize("query", IDENTIFIERS)
def test_identifiers_do_not_decide(query: str) -> None:
    assert detect_lang(query) == "other"


def test_accented_vowel_alone_is_spanish() -> None:
    assert detect_lang("replicación") == "es"


def test_inverted_question_mark_is_strong_evidence() -> None:
    assert detect_lang("¿vacuum?") == "es"


def test_mixed_query_goes_to_the_dominant_side() -> None:
    assert detect_lang("¿cómo creo un índice parcial? how do I create a partial index") == "es"
    assert detect_lang("how do I configure replication replicación lógica") == "en"


def test_balanced_mix_is_other() -> None:
    # one Spanish function word, one English function word, no strong characters
    assert detect_lang("la the") == "other"


def test_short_and_empty_queries_are_other() -> None:
    assert detect_lang("") == "other"
    assert detect_lang("   ") == "other"
    assert detect_lang("index") == "other"


def test_strip_drops_spanish_function_words() -> None:
    assert strip_spanish_stopwords("¿cómo configuro la replicación lógica?") == "configuro replicación lógica"


def test_strip_removes_question_signs_and_punctuation() -> None:
    assert strip_spanish_stopwords("¡cómo! de la tabla, con COPY;") == "tabla COPY"


def test_strip_returns_empty_when_every_term_is_a_function_word() -> None:
    assert strip_spanish_stopwords("¿qué es?") == ""
    assert strip_spanish_stopwords("¿y para qué?") == ""
    assert strip_spanish_stopwords("") == ""


def test_strip_keeps_identifiers_and_technical_terms() -> None:
    assert strip_spanish_stopwords("pg_dump VACUUM FREEZE") == "pg_dump VACUUM FREEZE"
    assert strip_spanish_stopwords("¿cómo uso pg_stat_activity?") == "uso pg_stat_activity"
    assert strip_spanish_stopwords("índice parcial con WHERE") == "índice parcial WHERE"


def test_strip_does_not_touch_english_function_words() -> None:
    # it is a Spanish filter: the English list is irrelevant here
    assert strip_spanish_stopwords("how do I create a partial index") == "how do I create a partial index"
