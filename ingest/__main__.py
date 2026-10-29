"""CLI: ``python -m ingest``.

Pipeline: fetch PostgreSQL 17 docs -> parse/chunk -> embeddings -> index.
Every stage is idempotent and can be skipped, and the run prints (and stores in
``data/ingest_stats.json``) timings and sizes.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

from .chunk import page_chunks
from .download import ensure_docs
from .parse import parse_html

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
HTML_DIR = DATA_DIR / "html"
CHUNKS_PATH = DATA_DIR / "chunks.jsonl"

MIN_AVAILABLE_MB = 900
PARSE_LOG_EVERY = 500


def embedding_model_name() -> str:
    """The local checkpoint's id, named only in :mod:`retrieval.embed`."""
    from retrieval.embed import MODEL_NAME

    return MODEL_NAME


def available_mb() -> int:
    with open("/proc/meminfo", encoding="ascii") as fh:
        for line in fh:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) // 1024
    return 0


def wait_for_memory(min_mb: int = MIN_AVAILABLE_MB, timeout_s: int = 180, poll_s: int = 5) -> int:
    """Wait (briefly) for enough free RAM; other tenants make MemAvailable noisy."""
    started = time.perf_counter()
    free = available_mb()
    while free < min_mb and time.perf_counter() - started < timeout_s:
        print(f"embed: {free} MB available, waiting for {min_mb} MB...", flush=True)
        time.sleep(poll_s)
        free = available_mb()
    return free


def write_chunks(chunks: list[dict], path: Path = CHUNKS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for chunk in chunks:
            fh.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    tmp.replace(path)


def build_chunks(html_dir: Path, limit: int | None = None) -> tuple[list[dict], dict]:
    files = sorted(html_dir.glob("*.html"))
    if limit is not None:
        files = files[:limit]
    chunks: list[dict] = []
    used = discarded = index_pages = 0
    logged = 0
    started = time.perf_counter()
    for page_file in files:
        page = parse_html(page_file.read_text("utf-8", "replace"), page_file.name)
        if page is None:
            discarded += 1
            continue
        if page.is_index:
            # Crawl seed, not documentation: see ingest.chunk.page_chunks.
            index_pages += 1
            continue
        used += 1
        chunks.extend(page_chunks(page))
        if len(chunks) // PARSE_LOG_EVERY > logged:
            logged = len(chunks) // PARSE_LOG_EVERY
            print(f"parse: {used} pages, {len(chunks)} chunks...", flush=True)
    seconds = time.perf_counter() - started
    print(
        f"parse: {len(files)} pages read, {used} kept, {discarded} discarded "
        f"(<40 words), {index_pages} alphabetical index page(s) excluded, {len(chunks)} chunks in {seconds:.1f}s",
        flush=True,
    )
    return chunks, {
        "pages_read": len(files),
        "pages_used": used,
        "pages_discarded": discarded,
        "pages_index_excluded": index_pages,
        "seconds": seconds,
    }


def embed_chunks(chunks: list[dict], batch: int = 64):
    from retrieval.embed import Embedder, chunk_documents
    documents = chunk_documents(chunks)
    embedder = Embedder()
    parts = []
    started = time.perf_counter()
    for start in range(0, len(documents), batch):
        parts.append(embedder.encode_passages(documents[start : start + batch]))
        done = min(start + batch, len(documents))
        elapsed = time.perf_counter() - started
        rate = done / elapsed if elapsed else 0.0
        print(f"embed: {done}/{len(documents)} chunks ({rate:.2f}/s, {elapsed:.0f}s)", flush=True)
    import numpy as np

    vectors = np.vstack(parts) if parts else np.zeros((0, 0), dtype=np.float32)
    seconds = time.perf_counter() - started
    print(f"embed: {len(vectors)} vectors of dim {vectors.shape[1] if len(vectors) else 0} in {seconds:.1f}s", flush=True)
    return vectors, seconds


def file_size(path: Path) -> int:
    return path.stat().st_size if path.exists() else 0


def reuse_vectors(limit: int | None = None) -> int:
    """Re-chunk the cached HTML and rebuild ``index.sqlite``, reusing embeddings.

    The embeddings cost ~34 minutes, so this path never re-embeds.  It re-reads
    ``data/html`` (the corpus rules may have changed), reconciles the new chunk
    ids with ``data/vector_ids.json``: identical ids keep ``vectors.npy`` as is,
    a strict subset filters its rows by id (see :func:`ingest.reuse.reconcile_vectors`)
    and anything else aborts **before any file is written**.
    """
    import numpy as np

    from retrieval.store import INDEX_PATH, VECTORS_PATH, VECTOR_IDS_PATH, SqliteNumpyStore

    from .reuse import VectorReuseError, reconcile_vectors

    if not HTML_DIR.exists() or not any(HTML_DIR.glob("*.html")):
        print(f"ERROR: {HTML_DIR} has no pages; --reuse-vectors needs the cached HTML", flush=True)
        return 2
    if not VECTOR_IDS_PATH.exists() or not VECTORS_PATH.exists():
        print(f"ERROR: {VECTORS_PATH.name} / {VECTOR_IDS_PATH.name} missing; nothing to reuse", flush=True)
        return 2

    chunks, parse_info = build_chunks(HTML_DIR, limit=limit)
    chunk_ids = [chunk["id"] for chunk in chunks]

    stored_ids = json.loads(VECTOR_IDS_PATH.read_text())
    stored_vectors = np.load(VECTORS_PATH)
    try:
        vectors, vector_ids, mode = reconcile_vectors(chunk_ids, stored_ids, stored_vectors)
    except VectorReuseError as error:
        print(f"ERROR: cannot reuse the stored vectors: {error}; aborting (nothing written)", flush=True)
        return 2
    if mode == "filtered":
        print(
            f"vectors: kept {len(vector_ids)}/{len(stored_ids)} rows by id (dropped {len(stored_ids) - len(vector_ids)})",
            flush=True,
        )
    else:
        print(f"vectors: {len(vector_ids)} ids unchanged", flush=True)

    # Everything is consistent: only now touch the files.
    write_chunks(chunks)
    if mode == "filtered":
        np.save(VECTORS_PATH, np.asarray(vectors, dtype=np.float32))
        VECTOR_IDS_PATH.write_text(json.dumps(vector_ids))

    started = time.perf_counter()
    store = SqliteNumpyStore.build(
        chunks,
        vectors=None,  # written above, in chunk order
        index_path=INDEX_PATH,
        vectors_path=VECTORS_PATH,
        vector_ids_path=VECTOR_IDS_PATH,
    )
    seconds = time.perf_counter() - started
    print(
        f"index: {store.size} chunks indexed in {seconds:.1f}s from {CHUNKS_PATH.name}"
        f" (vectors reused: {len(vector_ids)} x {np.load(VECTORS_PATH, mmap_mode='r').shape[1]}, {mode})",
        flush=True,
    )

    stats_path = DATA_DIR / "ingest_stats.json"
    stats = json.loads(stats_path.read_text()) if stats_path.exists() else {}
    stats.setdefault("seconds", {})["index"] = round(seconds, 2)
    stats["pages_read"] = parse_info["pages_read"]
    stats["pages_used"] = parse_info["pages_used"]
    stats["pages_discarded"] = parse_info["pages_discarded"]
    stats["pages_index_excluded"] = parse_info["pages_index_excluded"]
    stats["chunks"] = len(chunks)
    stats["index_bytes"] = {
        "chunks.jsonl": file_size(CHUNKS_PATH),
        "index.sqlite": file_size(INDEX_PATH),
        "vectors.npy": file_size(VECTORS_PATH),
        "vector_ids.json": file_size(VECTOR_IDS_PATH),
    }
    stats["vectors_reused"] = True
    stats["vectors_reuse_mode"] = mode
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(stats, indent=2, ensure_ascii=False), flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ingest", description="Build the PostgreSQL 17 documentation index")
    parser.add_argument("--skip-download", action="store_true", help="do not fetch docs (use data/html as is)")
    parser.add_argument("--skip-embed", action="store_true", help="skip embeddings and the vector index")
    parser.add_argument(
        "--reuse-vectors",
        action="store_true",
        help="re-chunk data/html and rebuild index.sqlite, reusing data/vectors.npy (no re-embedding)",
    )
    parser.add_argument("--limit", type=int, default=None, help="only process the first N pages (for tests)")
    parser.add_argument("--refresh", action="store_true", help="re-fetch cached docs pages from the site")
    args = parser.parse_args(argv)

    if args.reuse_vectors:
        if args.skip_embed:
            print("ERROR: --reuse-vectors and --skip-embed are incompatible (--skip-embed deletes the vectors)", flush=True)
            return 2
        return reuse_vectors(limit=args.limit)

    timings: dict[str, float] = {}
    total_started = time.perf_counter()

    started = time.perf_counter()
    if args.skip_download:
        marker = HTML_DIR / ".source.json"
        version = json.loads(marker.read_text()).get("version", "unknown") if marker.exists() else "unknown"
        docs_info = {
            "version": version,
            "source": "data/html (cached, --skip-download)",
            "pages": len(list(HTML_DIR.glob("*.html"))),
            "tarball": None,
        }
        print(f"docs: using cached HTML ({docs_info['pages']} pages, version {version})", flush=True)
    else:
        docs_info = ensure_docs(RAW_DIR, HTML_DIR, refresh=args.refresh, log=lambda msg: print(msg, flush=True))
    timings["download"] = time.perf_counter() - started

    chunks, parse_info = build_chunks(HTML_DIR, limit=args.limit)
    timings["parse_chunk"] = parse_info.pop("seconds")
    write_chunks(chunks)

    vectors = None
    timings["embed"] = 0.0
    available_at_embed = available_mb()
    if args.skip_embed:
        print("embed: skipped (--skip-embed)", flush=True)
        for stale in (DATA_DIR / "vectors.npy", DATA_DIR / "vector_ids.json"):
            if stale.exists():
                stale.unlink()
                print(f"embed: removed stale {stale.name}", flush=True)
    else:
        available_at_embed = wait_for_memory()
        print(f"embed: {available_at_embed} MB available", flush=True)
        if available_at_embed < MIN_AVAILABLE_MB:
            print(
                f"ERROR: only {available_at_embed} MB of RAM available (< {MIN_AVAILABLE_MB} MB) and there is no swap; "
                "refusing to load the embedding model",
                flush=True,
            )
            return 2
        vectors, timings["embed"] = embed_chunks(chunks)

    from retrieval.store import INDEX_PATH, VECTORS_PATH, VECTOR_IDS_PATH, SqliteNumpyStore

    started = time.perf_counter()
    store = SqliteNumpyStore.build(
        chunks,
        vectors=vectors,
        index_path=INDEX_PATH,
        vectors_path=VECTORS_PATH,
        vector_ids_path=VECTOR_IDS_PATH,
    )
    timings["index"] = time.perf_counter() - started
    print(f"index: {store.size} chunks indexed in {timings['index']:.1f}s", flush=True)

    words = [chunk["n_words"] for chunk in chunks] or [0]
    stats = {
        "version": docs_info["version"],
        "docs_source": docs_info["source"],
        "tarball": docs_info.get("tarball"),
        "pages": docs_info["pages"],
        "pages_read": parse_info["pages_read"],
        "pages_used": parse_info["pages_used"],
        "pages_discarded": parse_info["pages_discarded"],
        "pages_index_excluded": parse_info["pages_index_excluded"],
        "chunks": len(chunks),
        "words": {
            "min": min(words),
            "median": int(statistics.median(words)),
            "max": max(words),
            "mean": round(statistics.fmean(words), 1),
        },
        "seconds": {key: round(value, 2) for key, value in timings.items()},
        "seconds_total": round(time.perf_counter() - total_started, 2),
        "index_bytes": {
            "chunks.jsonl": file_size(CHUNKS_PATH),
            "index.sqlite": file_size(INDEX_PATH),
            "vectors.npy": file_size(VECTORS_PATH),
            "vector_ids.json": file_size(VECTOR_IDS_PATH),
        },
        "available_mb_at_embed": available_at_embed,
        "embedding_model": None if args.skip_embed else embedding_model_name(),
    }

    stats_path = DATA_DIR / "ingest_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(stats, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
