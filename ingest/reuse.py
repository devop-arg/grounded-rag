"""Keep dense vectors when the chunk set only shrinks.

``python -m ingest --reuse-vectors`` re-chunks the cached HTML and rebuilds the
index without paying for the ~34 minutes of embeddings.  When the new chunk ids
are exactly the old ones nothing happens to ``vectors.npy``; when they are a
**subset** (a page was dropped from the corpus, e.g. the alphabetical index) the
rows are filtered by id and kept in ``chunks.jsonl`` order.  A chunk id with no
stored vector is a hard error: the run aborts before anything is written.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


class VectorReuseError(RuntimeError):
    """The stored vectors cannot be reused for the new chunk list."""


def reconcile_vectors(
    chunk_ids: Sequence[str],
    stored_ids: Sequence[str],
    vectors: np.ndarray,
) -> tuple[np.ndarray, list[str], str]:
    """Return ``(vectors, ids, mode)`` for ``chunk_ids`` with ``mode`` in
    ``{"identical", "filtered"}``.

    ``vectors`` rows follow ``stored_ids``; the result follows ``chunk_ids``.
    Raises :class:`VectorReuseError` when a new id has no vector or the matrix
    and the id list disagree, so the caller can abort without writing.
    """
    vectors = np.asarray(vectors)
    if len(stored_ids) != int(vectors.shape[0]):
        raise VectorReuseError(
            f"vector_ids.json has {len(stored_ids)} ids but vectors.npy has {vectors.shape[0]} rows"
        )
    if list(chunk_ids) == list(stored_ids):
        return vectors, list(stored_ids), "identical"
    if len(set(chunk_ids)) != len(chunk_ids):
        raise VectorReuseError("the new chunk list has duplicate ids")

    position: dict[str, int] = {}
    for index, chunk_id in enumerate(stored_ids):
        position.setdefault(chunk_id, index)

    missing = [chunk_id for chunk_id in chunk_ids if chunk_id not in position]
    if missing:
        raise VectorReuseError(
            f"{len(missing)} new chunk id(s) have no vector (first: {missing[0]}); re-embedding would be needed"
        )

    rows = np.asarray([position[chunk_id] for chunk_id in chunk_ids], dtype=np.intp)
    return vectors[rows], list(chunk_ids), "filtered"
