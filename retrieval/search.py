"""Hybrid search: lexical (FTS5/bm25) + dense vectors, fused with RRF.

Result units are **pages, not chunks**: candidate chunks (top 50 lexical + top 50
vector) are fused, then grouped by page (the URL without its ``#anchor``).  The
best ranked fragment of a page is the representative and the group score is
*that fragment's* RRF score (scores are not summed), so a page that wins on one
fragment is not inflated by the many fragments that share its page.  The group
keeps every fragment of that page that made it into the fused list (``chunk_ids``,
in rank order) and their full URLs (``urls``), so an expected ``#anchor`` matches
even when another fragment of the same page is the representative.

Every part also has an optional remote counterpart (see :mod:`retrieval.remote`)
that only runs when the caller passes one in: a remote embedding of the query
(the local model is then never loaded, unless that call fails and the local index
is the fallback), a rerank of the top :data:`RERANK_CANDIDATES` fragments *before*
grouping, and an English translation used **only** for the lexical query.  Any
remote failure keeps the local behaviour for that part, counts a fallback and
never fails the search.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import NamedTuple

from .embed import Embedder
from .fusion import rank_map, rrf_fuse
from .lang import detect_lang, strip_spanish_stopwords
from .remote import RERANK_DOCUMENT_CHARS, RemoteRetrieval, RemoteStats
from .store import SqliteNumpyStore, Store

CANDIDATES = 50
SNIPPET_CHARS = 400
#: Fragments sent to the reranker, before grouping by page.
RERANK_CANDIDATES = 20

logger = logging.getLogger("grounded_rag.search")


class PolicyConfig(NamedTuple):
    """Weights and lexical index a query gets, plus whether the stopword filter applies."""

    lang: str
    lexical_mode: str
    weights: list[float]
    drop_stopwords: bool


#: ``baseline``: stemmed index for every query, lexical/vector weights 1/1.
#: ``plain``: plain (unstemmed) index for Spanish queries.
#: ``downweight``: stemmed index, the lexical list counts half for Spanish queries.
#: ``plain+downweight``: both of the above.
#: ``es-stopwords``: stemmed index, Spanish function words dropped from the lexical query.
#: ``es-stopwords+downweight``: the stopword filter plus the half lexical weight.
#: Only ``lang == "es"`` gets any of this: ``"en"`` and ``"other"`` (bare
#: identifiers such as ``pg_dump`` or ``VACUUM FREEZE``, where the lexical signal
#: matters most) always use the stemmed index with weights 1/1.
POLICIES = (
    "baseline",
    "plain",
    "downweight",
    "plain+downweight",
    "es-stopwords",
    "es-stopwords+downweight",
)
#: Best policy measured on ``eval/questions.jsonl`` and ``eval/dev_es.jsonl``:
#: at lexical weight 0.5 ``downweight``, ``plain+downweight`` and
#: ``es-stopwords+downweight`` tie at MRR@8 0.8333 / 0.5579, so the simplest one
#: wins; ``es-stopwords`` alone is worse than ``baseline`` (0.7083 / 0.5437) and
#: ``plain`` too (0.8125 / 0.4552).
DEFAULT_POLICY = "downweight"
#: Lexical weight for the downweight policies.  ``downweight`` ties at 0.3 and 0.5
#: on both sets; 0.0 scores best on dev_es (MRR@8 0.5849) but much worse on the
#: sample set (0.6667 vs 0.8333), so 0.5 (the round, less aggressive value) stays.
LEXICAL_DOWNWEIGHT = 0.5

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


def _local_embedder(embedder: Embedder | None) -> Embedder:
    """The injected embedder, or the process-wide one (loaded on first use only)."""
    return embedder if embedder is not None else get_embedder()


def resolve_policy(
    query: str, policy: str | None = None, lexical_weight: float | None = None
) -> PolicyConfig:
    """Return the :class:`PolicyConfig` a query gets under ``policy``.

    Only Spanish queries get the non-English treatment; ``"en"`` and ``"other"``
    (bare identifiers such as ``pg_dump`` or ``VACUUM FREEZE``) keep the stemmed
    index and weights 1/1, because there the lexical list is the strongest signal.
    """
    policy = policy or DEFAULT_POLICY
    if policy not in POLICIES:
        raise ValueError(f"unknown policy {policy!r} (expected one of {POLICIES})")
    lang = detect_lang(query)
    mode = "stem"
    weights = [1.0, 1.0]
    drop_stopwords = False
    if lang == "es":
        if policy in ("plain", "plain+downweight"):
            mode = "plain"
        if policy in ("downweight", "plain+downweight", "es-stopwords+downweight"):
            weights[0] = LEXICAL_DOWNWEIGHT if lexical_weight is None else lexical_weight
        if policy in ("es-stopwords", "es-stopwords+downweight"):
            drop_stopwords = True
    return PolicyConfig(lang, mode, weights, drop_stopwords)


def group_by_page(
    fused: list[tuple[str, float]],
    rows: dict[str, dict],
    order: list[str] | None = None,
) -> list[dict]:
    """Collapse fused chunks into one entry per page, best fragment first.

    The grouping key is the page URL (``url`` without its ``#anchor``, i.e.
    ``page_file``), so two fragments of the same page never occupy two slots.
    ``fused`` is ordered best first, so the first chunk seen for a page is its
    representative: the group keeps *that* chunk's ``id``, score, URL (with its
    anchor), title, section and snippet.  Every chunk of the page that made it
    into ``fused`` is kept in ``chunk_ids`` (rank order) and its full URL in
    ``urls``.

    ``order`` (chunk ids, best first) overrides the sort: the reranker returns a
    new order for the head of the fused list while the groups keep their RRF
    score.  Groups it does not mention keep their RRF place, after it.
    """
    groups: dict[str, dict] = {}
    for chunk_id, score in fused:
        row = rows.get(chunk_id)
        if row is None:
            continue
        url = row["url"]
        page_url = url.split("#", 1)[0]
        group = groups.get(page_url)
        if group is None:
            groups[page_url] = {
                "id": chunk_id,
                "rrf": score,
                "url": url,
                "page_url": page_url,
                "page_title": row["page_title"],
                "section_path": row["section_path"],
                "text": snippet(row["text"]),
                "chunk_ids": [chunk_id],
                "urls": [url],
            }
        else:
            group["chunk_ids"].append(chunk_id)
            if url not in group["urls"]:
                group["urls"].append(url)
    if order is None:
        ordered = sorted(groups.values(), key=lambda group: (-group["rrf"], group["page_url"]))
    else:
        ordered = []
        taken: set[str] = set()
        for chunk_id in order:
            row = rows.get(chunk_id)
            if row is None:
                continue
            page_url = row["url"].split("#", 1)[0]
            if page_url in taken or page_url not in groups:
                continue
            taken.add(page_url)
            ordered.append(groups[page_url])
        rest = sorted(
            (group for page_url, group in groups.items() if page_url not in taken),
            key=lambda group: (-group["rrf"], group["page_url"]),
        )
        ordered.extend(rest)
    for group in ordered:
        group["n_chunks"] = len(group["chunk_ids"])
    return ordered


def _lexical_query(
    query: str, config: PolicyConfig, remote: RemoteRetrieval | None, stats: RemoteStats | None
) -> str:
    """The query the lexical index sees: the translation for non-English questions.

    Only the FTS5 query changes: the vector search and the prompt keep the
    original question.  A failed translation keeps the local behaviour (the
    stopword filter included) instead of failing the search.
    """
    if remote is not None and remote.translator is not None and config.lang != "en":
        translated = remote.translator.translate(query, stats)
        if translated:
            return translated
        if stats is not None:
            stats.fallbacks += 1
    return strip_spanish_stopwords(query) if config.drop_stopwords else query


def _vector_candidates(
    query: str,
    store: Store,
    embedder: Embedder | None,
    candidates: int,
    remote: RemoteRetrieval | None,
    stats: RemoteStats | None,
    info: dict | None,
) -> list[tuple[str, float]]:
    """Vector hits from the remote index, or from the local one (also as fallback)."""
    if remote is not None and remote.remote_vectors:
        assert remote.embeddings is not None and remote.index is not None
        qvec = remote.embeddings.encode_query(query, stats)
        if qvec is not None:
            try:
                hits = remote.index.search(qvec, candidates)
            except (ValueError, OSError) as error:
                # A damaged index or a dimension mismatch is a remote failure too.
                if stats is not None:
                    stats.embed_errors += 1
                    stats.fallbacks += 1
                logger.warning("embeddings: remote index unusable, using the local one (%s)", error)
            else:
                if info is not None:
                    info["embedding"] = "remote"
                    info["vector_cosine"] = float(hits[0][1]) if hits else 0.0
                return hits
        elif stats is not None:
            stats.fallbacks += 1
    hits = store.vector(_local_embedder(embedder).encode_query(query), candidates)
    if info is not None:
        info["embedding"] = "local"
        info["vector_cosine"] = float(hits[0][1]) if hits else 0.0
    return hits


def _rerank_order(
    query: str,
    fused: list[tuple[str, float]],
    rows: dict[str, dict],
    remote: RemoteRetrieval | None,
    stats: RemoteStats | None,
) -> tuple[list[str] | None, dict[str, float] | None]:
    """Rerank the first :data:`RERANK_CANDIDATES` fragments, before grouping."""
    if remote is None or remote.reranker is None or not fused:
        return None, None
    head = [chunk_id for chunk_id, _ in fused[:RERANK_CANDIDATES]]
    documents = []
    for chunk_id in head:
        row = rows.get(chunk_id) or {}
        documents.append(" ".join(str(row.get("text") or "").split())[:RERANK_DOCUMENT_CHARS])
    ranked = remote.reranker.rerank(query, documents, top_n=len(head), stats=stats)
    if not ranked:
        if stats is not None:
            stats.fallbacks += 1
        return None, None
    scores: dict[str, float] = {}
    order: list[str] = []
    for index, score in ranked:
        if 0 <= index < len(head):
            scores[head[index]] = score
            order.append(head[index])
    order.extend(chunk_id for chunk_id in head if chunk_id not in scores)
    return order, scores



def hybrid_search(
    query: str,
    k: int = 8,
    policy: str | None = None,
    store: Store | None = None,
    embedder: Embedder | None = None,
    candidates: int = CANDIDATES,
    lexical_weight: float | None = None,
    remote: RemoteRetrieval | None = None,
    stats: RemoteStats | None = None,
    info: dict | None = None,
) -> list[dict]:
    """Return the top ``k`` distinct pages for ``query``.

    Lexical top-``candidates`` + vector top-``candidates`` are fused with RRF and
    grouped by page (see :func:`group_by_page`).  ``policy`` picks the lexical
    index, the fusion weights and whether Spanish function words are dropped from
    the lexical query (see :data:`POLICIES`).

    ``remote`` turns the optional remote parts on (see :mod:`retrieval.remote`);
    ``stats`` is the counter they report to (``remote.stats`` by default) and
    ``info`` is filled with ``{"embedding": "remote"|"local", "vector_cosine"}``
    so a caller gets the signal without embedding the query twice.
    """
    store = store or get_store()
    tracker = stats if stats is not None else (remote.stats if remote is not None else None)
    config = resolve_policy(query, policy, lexical_weight)

    lexical_query = _lexical_query(query, config, remote, tracker)
    lexical = store.lexical(lexical_query, candidates, mode=config.lexical_mode)
    vector = _vector_candidates(query, store, embedder, candidates, remote, tracker, info)

    fused = rrf_fuse([lexical, vector], weights=config.weights)
    rows = {row["chunk_id"]: row for row in store.get([chunk_id for chunk_id, _ in fused])}
    order, rerank_scores = _rerank_order(query, fused, rows, remote, tracker)

    lexical_ranks = rank_map(lexical)
    vector_ranks = rank_map(vector)
    results = group_by_page(fused, rows, order=order)
    for result in results:
        result["lexical_rank"] = lexical_ranks.get(result["id"])
        result["vector_rank"] = vector_ranks.get(result["id"])
        if rerank_scores is not None:
            result["rerank_score"] = rerank_scores.get(result["id"])
    return results[:k]

