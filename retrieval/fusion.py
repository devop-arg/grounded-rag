"""Reciprocal Rank Fusion of ranked result lists."""

from __future__ import annotations

from typing import Iterable, Sequence

RRF_K = 60

Scored = tuple[str, float]


def rrf_fuse(ranked_lists: Sequence[Sequence[Scored | str]], k: int = RRF_K, top_n: int | None = None) -> list[Scored]:
    """Fuse ranked lists with ``score = sum(1 / (k + rank))`` (rank starts at 1).

    Each list must be ordered best first; items are ``(id, score)`` tuples or bare
    ids.  Ties are broken by id so the output is deterministic.  ``k`` is the RRF
    constant (60 by default), not the number of results.
    """
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, item in enumerate(ranked, start=1):
            doc_id = item[0] if isinstance(item, tuple) else item
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    fused = sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))
    return fused if top_n is None else fused[:top_n]


def rank_map(ranked: Iterable[Scored | str]) -> dict[str, int]:
    """Map id -> rank (1-based) for a single ranked list."""
    return {item[0] if isinstance(item, tuple) else item: rank for rank, item in enumerate(ranked, start=1)}
