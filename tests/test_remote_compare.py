"""``eval --compare-remote``: the report shape, the choice and the table, offline."""

from __future__ import annotations

import json

import numpy as np

from eval.remote_compare import (
    BASE_NAME,
    choose,
    format_table,
    make_remote,
    report_path,
    run,
    summarize,
    write_report,
)
from retrieval.remote import RemoteRetrieval, RemoteStats
from retrieval.search import hybrid_search

QUESTIONS = [
    {
        "id": "a01",
        "lang": "en",
        "question": "how do I create a partial index",
        "answerable": True,
        "expected_urls": ["https://example.test/a.html"],
        "excluded": False,
    },
    {
        "id": "a02",
        "lang": "es",
        "question": "¿cómo creo un índice parcial?",
        "answerable": True,
        "expected_urls": ["https://example.test/a.html"],
        "excluded": False,
    },
    {
        "id": "u01",
        "lang": "en",
        "question": "what is the airspeed of a swallow",
        "answerable": False,
        "expected_urls": [],
        "excluded": False,
    },
]

CHUNKS = {
    "a1": {
        "chunk_id": "a1",
        "url": "https://example.test/a.html#A",
        "page_title": "A",
        "section_path": ["S"],
        "text": "partial index",
    },
    "b1": {
        "chunk_id": "b1",
        "url": "https://example.test/b.html#B",
        "page_title": "B",
        "section_path": ["S"],
        "text": "brin index",
    },
}


class Store:
    def lexical(self, query: str, k: int, mode: str = "stem"):
        return [("a1", 1.0), ("b1", 0.9)] if query.strip() else []

    def vector(self, qvec, k: int):
        return [("a1", 0.9), ("b1", 0.8)]

    def get(self, ids):
        return [CHUNKS[chunk_id] for chunk_id in ids if chunk_id in CHUNKS]


class Embedder:
    def encode_query(self, query: str) -> np.ndarray:
        return np.ones(3, dtype=np.float32)


class FlakyReranker:
    """Puts the second document first (which would score worse)."""

    model = "vendor/rerank-model"

    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.calls = 0

    def rerank(self, query, documents, *, top_n=None, stats=None):
        self.calls += 1
        if stats is not None:
            stats.rerank_calls += 1
        if self.fail:
            if stats is not None:
                stats.rerank_errors += 1
            return None
        return [(1, 1.0), (0, 0.0)]


def url_matches(result_url: str, expected_url: str) -> bool:
    return result_url.split("#", 1)[0] == expected_url.split("#", 1)[0]


def test_summarize_counts_only_answerable_questions() -> None:
    rows = [
        {"answerable": True, "hit_rank": 1},
        {"answerable": True, "hit_rank": 6},
        {"answerable": False, "hit_rank": None},
    ]
    assert summarize(rows) == {"n": 2, "recall@5": 0.5, "mrr@8": 0.5833, "hits@5": 1}


def test_a_run_without_parts_has_one_combination_and_chooses_it() -> None:
    report = run(
        QUESTIONS,
        config=object(),
        policy="downweight",
        store=Store(),
        embedder=Embedder(),
        url_matches=url_matches,
        factory=lambda **kwargs: None,
    )
    assert [item["name"] for item in report["combinations"]] == [BASE_NAME]
    assert report["chosen"] == BASE_NAME
    assert report["cost_usd"] == 0.0
    assert report["questions"] == 3 and report["answerable"] == 2
    assert report["combinations"][0]["remote"]["errors"] == 0
    assert set(report["combinations"][0]["summary"]) == {"all", "en", "es"}


def test_a_part_that_does_not_improve_is_measured_and_left_out() -> None:
    """A rerank that demotes the right page is measured but not chosen."""
    report = run(
        QUESTIONS,
        config=object(),
        policy="downweight",
        store=Store(),
        embedder=Embedder(),
        url_matches=url_matches,
        rerank_model="vendor/rerank-model",
        factory=lambda **kwargs: RemoteRetrieval(reranker=FlakyReranker()) if kwargs.get("rerank_model") else None,
    )
    names = [item["name"] for item in report["combinations"]]
    assert names == [BASE_NAME, "+rerank"]
    assert report["chosen"] == BASE_NAME
    assert "raise" in report["reason"] or "raised" in report["reason"]
    assert report["combinations"][1]["models"]["rerank"] == "rerank-model"


def test_a_failing_part_is_marked_unavailable_and_the_baseline_wins() -> None:
    report = run(
        QUESTIONS,
        config=object(),
        policy="downweight",
        store=Store(),
        embedder=Embedder(),
        url_matches=url_matches,
        rerank_model="vendor/rerank-model",
        factory=lambda **kwargs: RemoteRetrieval(reranker=FlakyReranker(fail=True)),
    )
    rerank = report["combinations"][1]
    assert rerank["status"] == "unavailable (every remote call failed)"
    assert rerank["remote"]["errors"] == 3  # one call per question
    assert rerank["summary"] == report["combinations"][0]["summary"]  # the local order is kept
    assert report["chosen"] == BASE_NAME


def test_the_table_shows_every_language_latency_errors_and_cost() -> None:
    report = run(
        QUESTIONS,
        config=object(),
        policy="downweight",
        store=Store(),
        embedder=Embedder(),
        url_matches=url_matches,
        factory=lambda **kwargs: None,
    )
    table = format_table(report)
    assert "combination" in table
    assert "recall@5" in table and "mrr@8" in table
    assert "p50 ms" in table and "p95 ms" in table and "cost usd" in table
    for lang in ("all", "en", "es"):
        assert f"  {lang}  " in table


def test_the_report_is_written_without_paths_or_models_routes(tmp_path) -> None:
    report = run(
        QUESTIONS,
        config=object(),
        policy="downweight",
        store=Store(),
        embedder=Embedder(),
        url_matches=url_matches,
        factory=lambda **kwargs: None,
    )
    path = write_report(report, reports_dir=tmp_path)
    assert path == report_path(reports_dir=tmp_path)
    text = path.read_text()
    assert "/home/" not in text and "example.test" not in text
    saved = json.loads(text)
    assert saved["chosen"] == BASE_NAME
    assert saved["combinations"][0]["latency_ms"]["median"] >= 0


def test_make_remote_without_a_key_builds_nothing() -> None:
    from generation.config import Config

    assert make_remote(config=Config(llm_base_url="https://llm.example/v1"), translate_model="vendor/t") is None


def test_choose_prefers_the_best_mrr_that_keeps_the_recall() -> None:
    def combination(name: str, recall: float, mrr: float) -> dict:
        return {
            "name": name,
            "status": "ok",
            "summary": {"all": {"recall@5": recall, "mrr@8": mrr, "n": 10}},
        }

    base = combination(BASE_NAME, 0.90, 0.70)
    better = combination("+translation", 0.90, 0.72)
    worse_recall = combination("+rerank", 0.88, 0.99)
    chosen, reason = choose([base, better, worse_recall])
    assert chosen is better and "translation" in reason

    chosen, reason = choose([base, worse_recall])
    assert chosen is base and "stays off" in reason


def test_hybrid_search_is_deterministic_with_the_fake_store() -> None:
    """Sanity check of the fixture the comparison runs on."""
    results = hybrid_search("how do I create a partial index", store=Store(), embedder=Embedder())
    assert [result["page_url"] for result in results][0] == "https://example.test/a.html"
