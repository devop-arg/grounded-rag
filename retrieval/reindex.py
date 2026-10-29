"""``python -m retrieval.reindex``: build a remote-embedding vector index.

The vectors of a remote embedding model have their own dimensions, so they get
their own files — ``data/vectors-<name>.npy`` and ``data/vector_ids-<name>.json``
— and the local ``vectors.npy``/``vector_ids.json`` pair is never opened, never
written and never deleted.  The index is built in batches and is **resumable**:
every finished batch is flushed to ``data/vectors-<name>.part.npy`` and recorded
in ``data/vectors-<name>.progress.json``, so an interrupted run continues where
it stopped instead of paying for the same passages twice.  On completion the
partial file is renamed into place and both the partial file and its progress
file disappear.

Switching it on at query time: ``GROUNDED_RAG_EMBED_MODEL`` (the same model) and
``GROUNDED_RAG_EMBED_INDEX`` (this index's name).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Sequence

import numpy as np

from .embed import chunk_documents
from .remote import RemoteEmbeddings, RemoteStats
from .store import DATA_DIR, iter_chunks, vector_ids_path_for, vectors_path_for

DEFAULT_BATCH_SIZE = 16
MAX_ATTEMPTS = 3
MAX_BACKOFF_S = 10.0


class ReindexError(RuntimeError):
    """The index could not be finished; the partial files are kept for a resume."""


def _ids_digest(ids: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()


def _paths(name: str, data_dir: Path) -> tuple[Path, Path, Path, Path]:
    return (
        vectors_path_for(name, data_dir),
        vector_ids_path_for(name, data_dir),
        data_dir / f"vectors-{name}.part.npy",
        data_dir / f"vectors-{name}.progress.json",
    )


def _read_progress(path: Path) -> dict | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def build_index(
    chunks: Sequence[dict],
    embeddings: RemoteEmbeddings,
    *,
    name: str,
    data_dir: Path | str = DATA_DIR,
    batch_size: int = DEFAULT_BATCH_SIZE,
    sleep=time.sleep,
    max_attempts: int = MAX_ATTEMPTS,
    stats: RemoteStats | None = None,
    log=print,
) -> dict:
    """Embed every chunk and write ``vectors-<name>.npy`` plus its id list."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    vectors_path, ids_path, part_path, progress_path = _paths(name, data_dir)
    texts = chunk_documents(chunks)
    ids = [chunk["id"] for chunk in chunks]
    if not ids:
        raise ReindexError("there are no chunks to embed (run `python -m ingest` first)")
    digest = _ids_digest(ids)
    stats = stats if stats is not None else RemoteStats()

    start = 0
    matrix = None
    progress = _read_progress(progress_path)
    if (
        progress
        and progress.get("model") == embeddings.model
        and progress.get("ids") == digest
        and part_path.exists()
    ):
        start = int(progress.get("done") or 0)
        try:
            matrix = np.lib.format.open_memmap(part_path, mode="r+")
        except (OSError, ValueError):
            start, matrix = 0, None
    if start <= 0 or matrix is None:
        start = 0
        for leftover in (part_path, progress_path):
            if leftover.exists():
                leftover.unlink()
    else:
        log(f"resuming at {start}/{len(ids)} chunks (partial file kept)")

    def save_progress() -> None:
        progress_path.write_text(
            json.dumps(
                {
                    "model": embeddings.model,
                    "ids": digest,
                    "dim": int(matrix.shape[1]) if matrix is not None else 0,
                    "done": start,
                    "total": len(ids),
                }
            ),
            encoding="utf-8",
        )

    batches = 0
    started = time.perf_counter()
    while start < len(ids):
        batch = texts[start : start + batch_size]
        result = None
        attempt = 0
        while attempt < max_attempts:
            attempt += 1
            result = embeddings.encode_passages(batch, stats)
            if result is not None:
                break
            if attempt < max_attempts:
                pause = min(2.0**attempt, MAX_BACKOFF_S)
                log(f"batch at {start}: attempt {attempt}/{max_attempts} failed, retrying in {pause:g}s")
                sleep(pause)
        if result is None:
            save_progress()
            raise ReindexError(
                f"batch at {start} failed after {max_attempts} attempts;"
                f" the partial index is in {part_path.name} and a rerun will resume there"
            )
        if matrix is None:
            matrix = np.lib.format.open_memmap(
                part_path, mode="w+", dtype=np.float32, shape=(len(ids), result.shape[1])
            )
        if result.shape[1] != matrix.shape[1]:
            raise ReindexError(
                f"the model returned {result.shape[1]} dimensions, the partial index has {matrix.shape[1]}"
            )
        matrix[start : start + len(batch)] = result
        matrix.flush()
        start += len(batch)
        batches += 1
        save_progress()
        log(
            f"{start}/{len(ids)} chunks  {(time.perf_counter() - started):.1f}s"
            f"  tokens={stats.embed_tokens}  cost_usd={stats.cost_usd:.6f}"
        )

    assert matrix is not None
    dim = int(matrix.shape[1])
    del matrix

    final = np.load(part_path)
    tmp = vectors_path.with_suffix(vectors_path.suffix + ".tmp")
    # np.save appends ".npy" to a path: a file object keeps the temporary name.
    with tmp.open("wb") as handle:
        np.save(handle, final.astype(np.float32, copy=False))
    tmp.replace(vectors_path)
    ids_path.write_text(json.dumps(ids), encoding="utf-8")
    part_path.unlink(missing_ok=True)
    progress_path.unlink(missing_ok=True)

    elapsed = time.perf_counter() - started
    return {
        "index": name,
        "chunks": len(ids),
        "dim": dim,
        "batches": batches,
        "batch_size": batch_size,
        "seconds": round(elapsed, 1),
        "tokens": stats.embed_tokens,
        "cost_usd": round(stats.cost_usd, 6),
        "errors": stats.embed_errors,
        "vectors": str(vectors_path),
        "vector_ids": str(ids_path),
    }


def _remote_embeddings(config, dimensions=None):
    """The embeddings client alone (the index may not exist yet: that is the point)."""
    from generation.config import LLMNotConfigured, resolve_llm_api_key

    from .remote import TIMEOUT_S

    if not config.embed_model:
        raise SystemExit("ERROR: GROUNDED_RAG_EMBED_MODEL is empty: nothing to build")
    try:
        api_key = resolve_llm_api_key(config)
    except LLMNotConfigured as error:
        raise SystemExit(f"ERROR: {error}") from None
    return RemoteEmbeddings(
        base_url=config.llm_base_url,
        api_key=api_key,
        model=config.embed_model,
        price_per_m=config.embed_price_per_m,
        dimensions=dimensions if dimensions is not None else (config.embed_dims or None),
        timeout=TIMEOUT_S,
        referer=config.llm_referer,
        title=config.llm_title,
    )


def main(argv: list[str] | None = None) -> int:
    from dataclasses import replace

    from generation.config import load_config

    parser = argparse.ArgumentParser(
        prog="retrieval.reindex", description="Build a named vector index with a remote embedding model"
    )
    parser.add_argument(
        "--index",
        default=None,
        help="index name (default: GROUNDED_RAG_EMBED_INDEX, else a slug of the model)",
    )
    parser.add_argument(
        "--model", default=None, help="embedding model (default: GROUNDED_RAG_EMBED_MODEL)"
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="passages per request")
    parser.add_argument("--limit", type=int, default=None, help="first N chunks only (smoke test)")
    parser.add_argument(
        "--dims",
        type=int,
        default=None,
        help="output dimensions (default: GROUNDED_RAG_EMBED_DIMS, else the model's own)",
    )
    parser.add_argument("--data", default=str(DATA_DIR), help="directory of the chunk file and the indices")
    args = parser.parse_args(argv)

    config = load_config()
    if args.model:
        config = replace(config, embed_model=args.model)
    name = args.index or config.embed_index_name
    if not name:
        print("ERROR: no index name: pass --index or set GROUNDED_RAG_EMBED_INDEX", flush=True)
        return 2
    if args.batch_size <= 0:
        print("ERROR: --batch-size must be positive", flush=True)
        return 2

    data_dir = Path(args.data)
    chunks = list(iter_chunks(data_dir / "chunks.jsonl"))
    if args.limit:
        chunks = chunks[: args.limit]
    print(f"index: {name}")
    print(f"model: {config.embed_model.rsplit('/', 1)[-1]}")
    print(f"chunks: {len(chunks)}  batch size: {args.batch_size}  dims: {args.dims or config.embed_dims or 'model default'}")

    embeddings = _remote_embeddings(config, args.dims)
    stats = RemoteStats()
    try:
        summary = build_index(
            chunks,
            embeddings,
            name=name,
            data_dir=data_dir,
            batch_size=args.batch_size,
            stats=stats,
            log=lambda message: print(message, flush=True),
        )
    except ReindexError as error:
        print(f"ERROR: {error}", flush=True)
        return 1
    finally:
        embeddings.close()

    print()
    print("summary")
    print(f"  wrote: {Path(summary['vectors']).name} ({summary['chunks']} x {summary['dim']})")
    print(f"  wrote: {Path(summary['vector_ids']).name}")
    print(f"  batches: {summary['batches']}  seconds: {summary['seconds']}")
    print(f"  tokens: {summary['tokens']}  cost_usd: {summary['cost_usd']:.6f}  errors: {summary['errors']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
