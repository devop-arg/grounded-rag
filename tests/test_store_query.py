"""Store tests: FTS5 query escaping, lexical ranking and vector search."""

from __future__ import annotations

import sqlite3

import numpy as np
import pytest

from retrieval.store import SqliteNumpyStore, fts_query

CHUNKS = [
    {
        "id": "a.html#A#0",
        "page_file": "a.html",
        "page_title": "Partial Indexes",
        "section_path": ["Partial Indexes"],
        "anchor": "A",
        "url": "https://www.postgresql.org/docs/17/a.html#A",
        "text": "A partial index is built over a subset of rows defined by a predicate.",
        "n_words": 13,
    },
    {
        "id": "b.html#B#0",
        "page_file": "b.html",
        "page_title": "Transaction Isolation",
        "section_path": ["Transaction Isolation"],
        "anchor": "B",
        "url": "https://www.postgresql.org/docs/17/b.html#B",
        "text": "PostgreSQL supports read committed, repeatable read and serializable isolation levels.",
        "n_words": 11,
    },
    {
        "id": "c.html#C#0",
        "page_file": "c.html",
        "page_title": "Logical Replication",
        "section_path": ["Logical Replication"],
        "anchor": "C",
        "url": "https://www.postgresql.org/docs/17/c.html#C",
        "text": "Logical replication sends row changes published by a publication to a subscription.",
        "n_words": 12,
    },
]


@pytest.fixture(scope="module")
def store(tmp_path_factory) -> SqliteNumpyStore:
    base = tmp_path_factory.mktemp("index")
    vectors = np.zeros((len(CHUNKS), 3), dtype=np.float32)
    vectors[0] = [1.0, 0.0, 0.0]
    vectors[1] = [0.3, 0.954, 0.0]
    vectors[2] = [0.0, 0.0, 1.0]
    return SqliteNumpyStore.build(
        CHUNKS,
        vectors=vectors,
        index_path=base / "index.sqlite",
        vectors_path=base / "vectors.npy",
        vector_ids_path=base / "vector_ids.json",
    )


def test_fts_query_quotes_every_token() -> None:
    assert fts_query("partial index") == '"partial" OR "index"'


def test_fts_query_escapes_special_characters() -> None:
    expression = fts_query('say "hello" AND * -drop NEAR/2')

    assert expression.startswith('"say" OR "hello"')
    assert '""' not in expression  # tokens are quoted once, quotes inside are doubled
    assert '*' not in expression and '-' not in expression and 'NEAR/2' not in expression
    assert 'AND' in expression  # plain token, quoted: cannot act as an operator


def test_fts_query_with_only_punctuation_is_empty() -> None:
    assert fts_query("   * - \" ") == ""


def test_escaping_survives_a_real_match(store: SqliteNumpyStore) -> None:
    # Danger: raw FTS5 syntax would be a syntax error / injection.
    assert store.lexical('" * - ( ) NEAR', 5) == []
    assert store.lexical("partial index", 5)[0][0] == "a.html#A#0"


def test_lexical_search_ranks_by_bm25(store: SqliteNumpyStore) -> None:
    results = store.lexical("isolation levels", 5)

    assert results
    assert results[0][0] == "b.html#B#0"
    scores = [score for _, score in results]
    assert scores == sorted(scores, reverse=True)


def test_empty_query_returns_nothing(store: SqliteNumpyStore) -> None:
    assert store.lexical("", 5) == []


def test_vector_search_by_dot_product(store: SqliteNumpyStore) -> None:
    query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    results = store.vector(query, 2)

    assert [chunk_id for chunk_id, _ in results] == ["a.html#A#0", "b.html#B#0"]
    assert results[0][1] == pytest.approx(1.0, rel=1e-5)
    assert results[1][1] == pytest.approx(0.3, rel=1e-4)


def test_vector_search_normalizes_the_query(store: SqliteNumpyStore) -> None:
    results = store.vector(np.array([10.0, 0.0, 0.0], dtype=np.float32), 1)

    assert results[0][1] == pytest.approx(1.0, rel=1e-5)


def test_get_preserves_requested_order(store: SqliteNumpyStore) -> None:
    rows = store.get(["c.html#C#0", "a.html#A#0", "missing"])

    assert [row["chunk_id"] for row in rows] == ["c.html#C#0", "a.html#A#0"]
    assert rows[0]["section_path"] == ["Logical Replication"]


def test_index_has_the_expected_fts_tokenizer(tmp_path) -> None:
    store = SqliteNumpyStore.build(
        CHUNKS,
        index_path=tmp_path / "index.sqlite",
        vectors_path=tmp_path / "vectors.npy",
        vector_ids_path=tmp_path / "vector_ids.json",
    )
    with sqlite3.connect(store.index_path) as conn:
        (sql,) = conn.execute("SELECT sql FROM sqlite_master WHERE name = 'chunks_fts'").fetchone()

    assert "porter unicode61 remove_diacritics 2" in sql


def test_index_has_a_stemless_fts_table(tmp_path) -> None:
    store = SqliteNumpyStore.build(
        CHUNKS,
        index_path=tmp_path / "index.sqlite",
        vectors_path=tmp_path / "vectors.npy",
        vector_ids_path=tmp_path / "vector_ids.json",
    )
    with sqlite3.connect(store.index_path) as conn:
        (sql,) = conn.execute("SELECT sql FROM sqlite_master WHERE name = 'chunks_fts_plain'").fetchone()

    assert "tokenize='unicode61 remove_diacritics 2'" in sql
    assert "porter" not in sql


def test_plain_mode_ranks_by_bm25_without_stemming(store: SqliteNumpyStore) -> None:
    results = store.lexical("partial index", 5, mode="plain")

    assert results
    assert results[0][0] == "a.html#A#0"
    scores = [score for _, score in results]
    assert scores == sorted(scores, reverse=True)


def test_stem_and_plain_modes_disagree_on_word_forms(store: SqliteNumpyStore) -> None:
    # "defining" only reaches "defined" through the porter stemmer ("defin").
    assert store.lexical("defining", 5, mode="stem")[0][0] == "a.html#A#0"
    assert store.lexical("defining", 5, mode="plain") == []


def test_unknown_lexical_mode_raises(store: SqliteNumpyStore) -> None:
    with pytest.raises(ValueError):
        store.lexical("partial index", 5, mode="trigram")


def test_missing_plain_table_raises_a_helpful_error(tmp_path) -> None:
    store = SqliteNumpyStore.build(
        CHUNKS,
        index_path=tmp_path / "index.sqlite",
        vectors_path=tmp_path / "vectors.npy",
        vector_ids_path=tmp_path / "vector_ids.json",
    )
    with sqlite3.connect(store.index_path) as conn:
        conn.execute("DROP TABLE chunks_fts_plain")

    assert store.lexical("partial index", 5)[0][0] == "a.html#A#0"  # stemmed index still works
    with pytest.raises(RuntimeError, match="reuse-vectors"):
        store.lexical("partial index", 5, mode="plain")



def test_build_without_vectors_still_serves_lexical(tmp_path) -> None:
    store = SqliteNumpyStore.build(
        CHUNKS,
        index_path=tmp_path / "index.sqlite",
        vectors_path=tmp_path / "vectors.npy",
        vector_ids_path=tmp_path / "vector_ids.json",
    )

    assert store.lexical("replication", 3)[0][0] == "c.html#C#0"
    assert store.vector(np.ones(3, dtype=np.float32), 3) == []
