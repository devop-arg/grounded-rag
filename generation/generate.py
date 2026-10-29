"""Grounded answer pipeline: retrieval -> optional threshold -> model -> validation.

``answer_question`` is the single entry point used by the app and by
``eval --generate``.  It never logs or returns the API key.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from retrieval.lang import detect_lang
from retrieval.search import hybrid_search

from .config import Config, load_config
from .llm import ChatModel, build_llm
from .prompt import build_messages
from .validate import new_stats, parse_model_output, validate_answer

DEFAULT_K = 8
MAX_SOURCE_WORDS = 1200
SIGNAL_NAMES = ("vector_cosine", "rrf")
THRESHOLD_PATH = Path(__file__).with_name("threshold.json")

#: ``reason`` values of an abstention.
REASONS = ("threshold", "no_sources", "model", "validation", "generation_disabled")


def collect_sources(results: list[dict], store, max_words: int = MAX_SOURCE_WORDS) -> list[dict]:
    """Numbered sources: full text of the representative plus extra fragments.

    ``hybrid_search`` returns one group per URL with a short snippet and the
    representative first; here the whole text of every fragment of that page is
    concatenated (up to ``max_words``) so the model can quote it verbatim.
    """
    sources: list[dict] = []
    for n, result in enumerate(results, start=1):
        chunk_ids = list(result.get("chunk_ids") or [])
        rows = {row["chunk_id"]: row for row in store.get(chunk_ids)} if chunk_ids else {}
        texts: list[str] = []
        words = 0
        truncated = False
        for chunk_id in chunk_ids:
            row = rows.get(chunk_id)
            if row is None:
                continue
            count = len(row["text"].split())
            if texts and words + count > max_words:
                truncated = True
                break
            texts.append(row["text"])
            words += count
        sources.append(
            {
                "n": n,
                "url": result["url"],
                "title": result["page_title"],
                "section_path": list(result.get("section_path") or []),
                "text": "\n\n".join(texts) if texts else result.get("text", ""),
                "n_words": words,
                "truncated": truncated,
            }
        )
    return sources


def retrieval_signals(
    question: str, results: list[dict], store, embedder, info: dict | None = None
) -> dict[str, float]:
    """Candidate abstention signals: top-1 vector cosine and top-1 RRF.

    ``info`` is what :func:`retrieval.search.hybrid_search` filled in: the
    cosine of the vector index it actually used (the remote one when a remote
    embedding is on), so the query is not embedded twice.
    """
    cosine = None if info is None else info.get("vector_cosine")
    if cosine is None:
        query = np.asarray(embedder.encode_query(question), dtype=np.float32)
        top = store.vector(query, 1)
        cosine = float(top[0][1]) if top else 0.0
    return {
        "vector_cosine": float(cosine),
        "rrf": float(results[0]["rrf"]) if results else 0.0,
    }


def load_threshold(path: Path | str | None = THRESHOLD_PATH) -> dict | None:
    """Read ``threshold.json``; ``None`` when the file is absent or unusable."""
    path = Path(path) if path is not None else THRESHOLD_PATH
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except ValueError:
        return None
    signal = data.get("signal")
    value = data.get("value")
    if signal not in SIGNAL_NAMES or not isinstance(value, (int, float)):
        return None
    return data


def empty_answer(lang: str, reason: str, signals: dict | None = None) -> dict:
    return {
        "abstain": True,
        "sentences": [],
        "sources": [],
        "reason": reason,
        "lang": lang,
        "stats": new_stats(),
        "signals": signals or {},
        "retrieval_ms": 0.0,
        "llm_ms": 0.0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
    }


def answer_question(
    question: str,
    lang: str | None = None,
    *,
    config: Config | None = None,
    store=None,
    embedder=None,
    llm: ChatModel | None = None,
    k: int = DEFAULT_K,
    policy: str | None = None,
    threshold: dict | None = None,
    use_threshold: bool = True,
    remote=None,
    stats=None,
) -> dict:
    """Answer ``question`` with sentences whose citations were verified.

    The result is ``{abstain, sentences, sources, reason, lang, stats, signals,
    retrieval_ms, llm_ms, prompt_tokens, completion_tokens}``; ``sources`` holds
    only the passages cited by a valid citation, keeping their ``n``.

    ``remote``/``stats`` are the optional remote retrieval parts and their
    counters (see :mod:`retrieval.remote`); without them retrieval is local and
    the embedding model is loaded when the search needs it.
    """
    from retrieval.search import get_store, hybrid_search

    config = config or load_config()
    store = store or get_store()
    language = lang if lang in {"en", "es", "other"} else detect_lang(question)

    started = time.perf_counter()
    info: dict = {}
    results = hybrid_search(
        question,
        k=k,
        policy=policy,
        store=store,
        embedder=embedder,
        remote=remote,
        stats=stats,
        info=info,
    )
    retrieval_ms = (time.perf_counter() - started) * 1000

    signals = retrieval_signals(question, results, store, embedder, info=info)
    if use_threshold:
        threshold = threshold if threshold is not None else load_threshold()
        if threshold is not None and signals.get(threshold["signal"], 0.0) < float(threshold["value"]):
            answer = empty_answer(language, "threshold", signals)
            answer["retrieval_ms"] = round(retrieval_ms, 1)
            return answer
    if not results:
        answer = empty_answer(language, "no_sources", signals)
        answer["retrieval_ms"] = round(retrieval_ms, 1)
        return answer

    sources = collect_sources(results, store)
    llm = llm or build_llm(config)
    messages = build_messages(question, sources, language)

    started = time.perf_counter()
    result = llm.complete(messages)
    llm_ms = (time.perf_counter() - started) * 1000

    payload = parse_model_output(result.text)
    abstain, sentences, cited, stats, outcome = validate_answer(
        payload, [source["text"] for source in sources]
    )
    if not abstain:
        reason = None
    elif outcome == "model_abstain":
        reason = "model"
    else:
        reason = "validation"

    cited_sources = [
        {"n": source["n"], "url": source["url"], "title": source["title"]}
        for source in sources
        if source["n"] in cited
    ]
    return {
        "abstain": abstain,
        "sentences": sentences,
        "sources": cited_sources,
        "reason": reason,
        "lang": language,
        "outcome": outcome,
        "stats": stats,
        "signals": signals,
        "retrieval_ms": round(retrieval_ms, 1),
        "llm_ms": round(llm_ms, 1),
        "prompt_tokens": int(result.prompt_tokens),
        "completion_tokens": int(result.completion_tokens),
    }
