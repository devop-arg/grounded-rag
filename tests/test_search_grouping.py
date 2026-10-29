"""Hybrid search tests with an in-memory store: page grouping and fusion policies."""

from __future__ import annotations

import numpy as np
import pytest

from retrieval.search import POLICIES, group_by_page, hybrid_search, resolve_policy

ES_QUERY = "¿cómo configuro la replicación lógica?"
EN_QUERY = "how do I create a partial index"


def make_row(chunk_id: str, url: str) -> dict:
    return {
        "chunk_id": chunk_id,
        "url": url,
        "page_title": url,
        "section_path": ["Section"],
        "text": f"text of {chunk_id}",
    }


DEFAULT_URLS = {
    "a1": "https://example.test/a.html#A",
    "a2": "https://example.test/a.html#A",  # same page: fragments must group
    "a3": "https://example.test/a.html#A",
    "b1": "https://example.test/b.html#B",
    "c1": "https://example.test/c.html#C",
    "lex1": "https://example.test/lex.html#L",
    "vec1": "https://example.test/vec.html#V",
    "vec2": "https://example.test/vec2.html#V",
}


class FakeStore:
    """Minimal :class:`retrieval.store.Store` with scripted rankings."""

    def __init__(self, lexical: list[str], vector: list[str], urls: dict[str, str] | None = None) -> None:
        self.lexical_hits = lexical
        self.vector_hits = vector
        self.modes: list[str] = []
        self.lexical_queries: list[str] = []
        urls = urls or DEFAULT_URLS
        self.rows = {chunk_id: make_row(chunk_id, url) for chunk_id, url in urls.items()}

    def lexical(self, query: str, k: int, mode: str = "stem") -> list[tuple[str, float]]:
        self.modes.append(mode)
        self.lexical_queries.append(query)
        if not query.strip():  # like the real store: an empty MATCH returns nothing
            return []
        return [(chunk_id, 1.0) for chunk_id in self.lexical_hits[:k]]

    def vector(self, qvec, k: int) -> list[tuple[str, float]]:
        return [(chunk_id, 1.0) for chunk_id in self.vector_hits[:k]]

    def get(self, ids: list[str]) -> list[dict]:
        return [self.rows[chunk_id] for chunk_id in ids if chunk_id in self.rows]


class FakeEmbedder:
    def encode_query(self, query: str) -> np.ndarray:
        return np.ones(3, dtype=np.float32)


def search(store: FakeStore, query: str = ES_QUERY, policy: str = "baseline", **kwargs) -> list[dict]:
    return hybrid_search(query, policy=policy, store=store, embedder=FakeEmbedder(), **kwargs)


def test_group_by_page_keeps_the_best_fragment_as_representative() -> None:
    rows = {chunk_id: make_row(chunk_id, url) for chunk_id, url in (("a1", "u/a"), ("a2", "u/a"), ("b1", "u/b"))}
    fused = [("b1", 1 / 61), ("a1", 1 / 62), ("a2", 1 / 63)]

    groups = group_by_page(fused, rows)

    assert [group["url"] for group in groups] == ["u/b", "u/a"]
    assert groups[1]["id"] == "a1"
    assert groups[1]["chunk_ids"] == ["a1", "a2"]
    assert groups[1]["n_chunks"] == 2


def test_two_anchors_of_one_page_become_one_result_with_the_best_anchor() -> None:
    """A page reached through two anchors is one result, not two."""
    rows = {
        "c1": make_row("c1", "https://example.test/createindex.html#id-1.9.3.69.7"),
        "c2": make_row("c2", "https://example.test/createindex.html#SQL-CREATEINDEX-CONCURRENTLY"),
        "d1": make_row("d1", "https://example.test/other.html#D"),
    }
    fused = [("c2", 1 / 61), ("c1", 1 / 62), ("d1", 1 / 63)]

    groups = group_by_page(fused, rows)

    assert len(groups) == 2
    page = groups[0]
    assert page["page_url"] == "https://example.test/createindex.html"
    assert page["url"] == "https://example.test/createindex.html#SQL-CREATEINDEX-CONCURRENTLY"
    assert page["id"] == "c2"
    assert page["chunk_ids"] == ["c2", "c1"]
    assert page["urls"] == [
        "https://example.test/createindex.html#SQL-CREATEINDEX-CONCURRENTLY",
        "https://example.test/createindex.html#id-1.9.3.69.7",
    ]
    assert page["rrf"] == pytest.approx(1 / 61)


def test_search_returns_eight_distinct_pages() -> None:
    urls = {f"c{index}": f"https://example.test/page{index // 2}.html#anchor{index}" for index in range(20)}
    store = FakeStore(lexical=[f"c{index}" for index in range(20)], vector=[], urls=urls)

    results = search(store, k=8)

    pages = [result["page_url"] for result in results]
    assert len(results) == 8
    assert len(set(pages)) == 8
    assert pages == [f"https://example.test/page{index}.html" for index in range(8)]
    assert all(len(result["chunk_ids"]) == 2 for result in results)  # both fragments kept


def test_group_score_is_the_best_fragment_not_the_sum() -> None:
    rows = {chunk_id: make_row(chunk_id, "u/a") for chunk_id in ["a1", "a2", "a3"]}
    fused = [("a1", 1 / 61), ("a2", 1 / 63), ("a3", 1 / 64)]

    (group,) = group_by_page(fused, rows)

    assert group["rrf"] == pytest.approx(1 / 61)
    assert group["rrf"] != pytest.approx(1 / 61 + 1 / 63 + 1 / 64)


def test_search_returns_distinct_urls() -> None:
    store = FakeStore(lexical=["a1", "b1", "a2", "a3"], vector=["c1"])

    results = search(store)

    urls = [result["url"] for result in results]
    assert urls == ["https://example.test/a.html#A", "https://example.test/c.html#C", "https://example.test/b.html#B"]
    assert len(urls) == len(set(urls))
    assert results[0]["chunk_ids"] == ["a1", "a2", "a3"]
    assert results[0]["n_chunks"] == 3
    assert results[0]["id"] == "a1"


def test_search_respects_k_groups() -> None:
    store = FakeStore(lexical=["a1", "b1", "a2", "a3"], vector=["c1"])

    assert len(search(store, k=2)) == 2
    assert len(search(store, k=1)) == 1


def test_search_keeps_the_rank_of_the_representative_fragment() -> None:
    store = FakeStore(lexical=["a1", "b1", "a2"], vector=[])

    results = search(store)

    assert results[0]["lexical_rank"] == 1
    assert results[1]["lexical_rank"] == 2
    assert results[0]["vector_rank"] is None


def test_baseline_and_plain_use_different_indexes_by_language() -> None:
    store = FakeStore(lexical=["a1"], vector=["c1"])

    search(store, query=EN_QUERY, policy="baseline")
    search(store, query=EN_QUERY, policy="plain")
    assert store.modes == ["stem", "stem"]  # English keeps the stemmed index

    store.modes.clear()
    search(store, query=ES_QUERY, policy="baseline")
    search(store, query=ES_QUERY, policy="plain")
    search(store, query=ES_QUERY, policy="downweight")
    search(store, query=ES_QUERY, policy="plain+downweight")
    assert store.modes == ["stem", "plain", "stem", "plain"]


def test_bare_identifiers_are_treated_like_english() -> None:
    """``"other"`` is not "non-English": no downweight, no plain index, no stopwords."""
    for query in ("pg_dump", "VACUUM FREEZE"):
        store = FakeStore(lexical=["lex1"], vector=["vec1"])
        config = resolve_policy(query, "downweight")

        assert config.lang == "other"
        assert config.lexical_mode == "stem"
        assert config.weights == [1.0, 1.0]
        assert config.drop_stopwords is False

        results = search(store, query=query, policy="downweight")
        assert results[0]["url"].endswith("lex.html#L")  # the lexical list still counts 1.0
        assert store.lexical_queries == [query]  # untouched


def test_stopword_policy_filters_the_lexical_query() -> None:
    store = FakeStore(lexical=["a1"], vector=["c1"])

    search(store, query=ES_QUERY, policy="es-stopwords")

    assert store.lexical_queries == ["configuro replicación lógica"]
    assert store.modes == ["stem"]


def test_stopword_policy_is_spanish_only() -> None:
    store = FakeStore(lexical=["a1"], vector=["c1"])

    search(store, query=EN_QUERY, policy="es-stopwords")
    search(store, query="pg_dump", policy="es-stopwords+downweight")

    assert store.lexical_queries == [EN_QUERY, "pg_dump"]


def test_stopword_policy_with_nothing_left_has_no_lexical_list() -> None:
    store = FakeStore(lexical=["lex1"], vector=["vec1"])

    results = search(store, query="¿cómo es?", policy="es-stopwords")

    assert store.lexical_queries == [""]
    assert [result["url"].rsplit("/", 1)[-1] for result in results] == ["vec.html#V"]


def test_stopword_policy_keeps_the_other_policy_knobs() -> None:
    assert tuple(resolve_policy(ES_QUERY, "es-stopwords")) == ("es", "stem", [1.0, 1.0], True)
    assert tuple(resolve_policy(ES_QUERY, "es-stopwords+downweight")) == ("es", "stem", [0.5, 1.0], True)


def test_downweight_lets_the_vector_list_outrank_the_lexical_one() -> None:
    store = FakeStore(lexical=["lex1"], vector=["vec1", "vec2"])

    baseline = search(store, policy="baseline")
    downweight = search(store, policy="downweight")

    assert baseline[0]["url"].endswith("lex.html#L")  # tie broken by id
    assert downweight[0]["url"].endswith("vec.html#V")
    assert downweight[0]["rrf"] == pytest.approx(1 / 61)


def test_downweight_is_not_applied_to_english_queries() -> None:
    store = FakeStore(lexical=["lex1"], vector=["vec1"])

    results = search(store, query=EN_QUERY, policy="downweight")

    assert results[0]["url"].endswith("lex.html#L")


def test_lexical_weight_override() -> None:
    store = FakeStore(lexical=["lex1"], vector=["vec1"])

    results = search(store, policy="downweight", lexical_weight=0.0)

    assert [result["url"].rsplit("/", 1)[-1] for result in results] == ["vec.html#V"]


@pytest.mark.parametrize(
    ("query", "policy", "expected"),
    [
        (EN_QUERY, "baseline", ("en", "stem", [1.0, 1.0], False)),
        (EN_QUERY, "plain+downweight", ("en", "stem", [1.0, 1.0], False)),
        (EN_QUERY, "es-stopwords+downweight", ("en", "stem", [1.0, 1.0], False)),
        (ES_QUERY, "baseline", ("es", "stem", [1.0, 1.0], False)),
        (ES_QUERY, "plain", ("es", "plain", [1.0, 1.0], False)),
        (ES_QUERY, "downweight", ("es", "stem", [0.5, 1.0], False)),
        (ES_QUERY, "plain+downweight", ("es", "plain", [0.5, 1.0], False)),
        (ES_QUERY, "es-stopwords", ("es", "stem", [1.0, 1.0], True)),
        (ES_QUERY, "es-stopwords+downweight", ("es", "stem", [0.5, 1.0], True)),
        # "other" behaves like English under every policy
        ("VACUUM FREEZE", "baseline", ("other", "stem", [1.0, 1.0], False)),
        ("VACUUM FREEZE", "plain", ("other", "stem", [1.0, 1.0], False)),
        ("VACUUM FREEZE", "downweight", ("other", "stem", [1.0, 1.0], False)),
        ("VACUUM FREEZE", "plain+downweight", ("other", "stem", [1.0, 1.0], False)),
        ("VACUUM FREEZE", "es-stopwords", ("other", "stem", [1.0, 1.0], False)),
        ("VACUUM FREEZE", "es-stopwords+downweight", ("other", "stem", [1.0, 1.0], False)),
    ],
)
def test_resolve_policy(query: str, policy: str, expected: tuple) -> None:
    assert tuple(resolve_policy(query, policy)) == expected


def test_resolve_policy_rejects_unknown_names() -> None:
    with pytest.raises(ValueError):
        resolve_policy(ES_QUERY, "vibes")


def test_all_policies_are_accepted() -> None:
    store = FakeStore(lexical=["a1"], vector=["c1"])

    for policy in POLICIES:
        assert search(store, policy=policy)
