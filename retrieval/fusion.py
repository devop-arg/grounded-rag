"""Reciprocal Rank Fusion of ranked result lists."""

from __future__ import annotations

from typing import Iterable, Sequence

RRF_K = 60

Scored = tuple[str, float]


def rrf_fuse(
    ranked_lists: Sequence[Sequence[Scored | str]],
    k: int = RRF_K,
    weights: Sequence[float] | None = None,
    top_n: int | None = None,
) -> list[Scored]:
    """Fuse ranked lists with ``score = sum(w_i / (k + rank))`` (rank starts at 1).

    Each list must be ordered best first; items are ``(id, score)`` tuples or bare
    ids.  Ties are broken by id so the output is deterministic.  ``k`` is the RRF
    constant (60 by default), not the number of results.  ``weights`` scales each
    list's contribution (``None`` means 1.0 for every list, i.e. the original
    unweighted fusion); a weight of ``0`` drops the list entirely.
    """
    if weights is None:
        weights = [1.0] * len(ranked_lists)
    if len(weights) != len(ranked_lists):
        raise ValueError(f"got {len(weights)} weights for {len(ranked_lists)} ranked lists")
    scores: dict[str, float] = {}
    for weight, ranked in zip(weights, ranked_lists):
        if weight == 0.0:  # nothing to add: the list is left out entirely
            continue
        for rank, item in enumerate(ranked, start=1):
            doc_id = item[0] if isinstance(item, tuple) else item
            scores[doc_id] = scores.get(doc_id, 0.0) + weight / (k + rank)
    fused = sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))
    return fused if top_n is None else fused[:top_n]


def rank_map(ranked: Iterable[Scored | str]) -> dict[str, int]:
    """Map id -> rank (1-based) for a single ranked list."""
    return {item[0] if isinstance(item, tuple) else item: rank for rank, item in enumerate(ranked, start=1)}
