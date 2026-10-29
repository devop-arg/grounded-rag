"""Hybrid search: lexical (FTS5/bm25) + dense vectors, fused with RRF."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .embed import Embedder
from .fusion import rank_map, rrf_fuse
from .store import SqliteNumpyStore, Store

CANDIDATES = 50
SNIPPET_CHARS = 400

_store: Store | None = None
_embedder: Embedder | None = None


def get_store(path: Path | str | None = None) -> Store:
    global _store
    if path is not None:
        return SqliteNumpyStore(index_path=path)
    if _store is None:
        _store = SqliteNumpyStore()
    return _store


def get_embedder() -> Embedder:
    global _embedder
    if _embedder is None:
        _embedder = Embedder()
    return _embedder


def snippet(text: str, limit: int = SNIPPET_CHARS) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit].rstrip() + "…"


def hybrid_search(
    query: str,
    k: int = 8,
    store: Store | None = None,
    embedder: Embedder | None = None,
    candidates: int = CANDIDATES,
) -> list[dict]:
    """Return the top ``k`` chunks for ``query`` (lexical top-50 + vector top-50, RRF)."""
    store = store or get_store()
    embedder = embedder or get_embedder()

    lexical = store.lexical(query, candidates)
    vector = store.vector(np.asarray(embedder.encode_query(query), dtype=np.float32), candidates)
    fused = rrf_fuse([lexical, vector], top_n=k)

    lexical_ranks = rank_map(lexical)
    vector_ranks = rank_map(vector)
    rows = {row["chunk_id"]: row for row in store.get([chunk_id for chunk_id, _ in fused])}

    results: list[dict] = []
    for chunk_id, score in fused:
        row = rows.get(chunk_id)
        if row is None:
            continue
        results.append(
            {
                "id": chunk_id,
                "rrf": score,
                "lexical_rank": lexical_ranks.get(chunk_id),
                "vector_rank": vector_ranks.get(chunk_id),
                "url": row["url"],
                "page_title": row["page_title"],
                "section_path": row["section_path"],
                "text": snippet(row["text"]),
            }
        )
    return results
