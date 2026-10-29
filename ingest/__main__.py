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
    used = discarded = 0
    logged = 0
    started = time.perf_counter()
    for page_file in files:
        page = parse_html(page_file.read_text("utf-8", "replace"), page_file.name)
        if page is None:
            discarded += 1
            continue
        used += 1
        chunks.extend(page_chunks(page))
        if len(chunks) // PARSE_LOG_EVERY > logged:
            logged = len(chunks) // PARSE_LOG_EVERY
            print(f"parse: {used} pages, {len(chunks)} chunks...", flush=True)
    seconds = time.perf_counter() - started
    print(
        f"parse: {len(files)} pages read, {used} kept, {discarded} discarded "
        f"(<40 words), {len(chunks)} chunks in {seconds:.1f}s",
        flush=True,
    )
    return chunks, {"pages_read": len(files), "pages_used": used, "pages_discarded": discarded, "seconds": seconds}


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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ingest", description="Build the PostgreSQL 17 documentation index")
    parser.add_argument("--skip-download", action="store_true", help="do not fetch docs (use data/html as is)")
    parser.add_argument("--skip-embed", action="store_true", help="skip embeddings and the vector index")
    parser.add_argument("--limit", type=int, default=None, help="only process the first N pages (for tests)")
    parser.add_argument("--refresh", action="store_true", help="re-fetch cached docs pages from the site")
    args = parser.parse_args(argv)

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
        "embedding_model": None if args.skip_embed else "intfloat/multilingual-e5-small",
    }

    stats_path = DATA_DIR / "ingest_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(stats, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
