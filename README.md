# grounded-rag

A grounded RAG assistant over the official **PostgreSQL 17** documentation.
Phase 1 (this repo state) covers **ingestion, local embeddings, hybrid retrieval
and retrieval evaluation only**: no LLM is called and no API keys are needed.

| Path | Purpose |
| --- | --- |
| `ingest/` | fetch the PG 17 docs, parse HTML into sections, chunk, embed, index |
| `retrieval/` | SQLite FTS5 + numpy vector store, RRF fusion, `hybrid_search` |
| `eval/` | recall@5 / MRR over a small labelled question set |
| `tests/` | unit tests (no network, no model) |
| `data/` | generated: raw tarball, HTML pages, `chunks.jsonl`, index, stats |

## Requirements

* Python 3.12
* ~1 GB of free RAM: no swap here, so embeddings run with 2 threads and 4-chunk
  batches on the int8 ONNX export (batch 32 peaked at ~1.5 GB of RSS)
* Network: `ftp.postgresql.org` (release tarball + `.sha256`),
  `www.postgresql.org` (rendered HTML; the source tarball ships DocBook/SGML
  only) and Hugging Face (fastembed model, cached in `data/models/`)

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Commands

```bash
.venv/bin/python -m ingest                                  # docs -> chunks -> vectors -> index
.venv/bin/python -m ingest --skip-download --skip-embed --limit 20
.venv/bin/python -m retrieval "how do I create a partial index" -k 8
.venv/bin/python -m eval
.venv/bin/python -m pytest -q
```

`ingest` flags: `--skip-download` (reuse `data/html`), `--skip-embed` (lexical
index only), `--limit N` (first N pages), `--refresh` (re-fetch cached pages).
Every run writes `data/ingest_stats.json` with page/chunk counts, word
statistics, per-stage seconds and index file sizes.

## Index design

* **Chunks** — a page is split at `h2`/`h3` section boundaries (`div.sect1`,
  `div.sect2`, `div.refsect1`...); deeper headings stay inside their `h3`.
  Consecutive small sections of a page are merged and large ones are cut on
  block boundaries (never inside a `pre`), targeting 300–600 words. Each chunk
  gets a stable id `<file>#<anchor>#<n>`, its section path and its docs URL.
* **Lexical** — SQLite FTS5 over page title, section path and text with
  `porter unicode61 remove_diacritics 2`, ranked by `bm25()`. User terms are
  tokenized, quoted and OR-ed, so FTS5 operators can never be injected.
* **Vector** — `intfloat/multilingual-e5-small` via fastembed (`passage: ` for
  chunks: title + breadcrumb + text, `query: ` for queries), from the int8 ONNX
  export; L2 normalized `float32` in `data/vectors.npy` (+ `data/vector_ids.json`),
  dot product search.
* **Fusion** — reciprocal rank fusion, `score = Σ 1/(60 + rank)`, over the top
  50 lexical and top 50 vector hits.

**Why SQLite + numpy?** pgvector needs a PostgreSQL extension (root on this
box) plus a server. SQLite is in the standard library, FTS5 gives real BM25
ranking, and even a few thousand 384-dim vectors are a handful of megabytes, so
the index is two plain files and the `Store` interface stays swappable for
pgvector later.

## Status

Phase 1 only: there is no generation step yet. On the 17.11 corpus (1143 pages,
2835 chunks) the sample set in `eval/questions.jsonl` scores recall@5 = 1.0 and
MRR@8 = 0.8; see `data/ingest_stats.json` for timings and sizes.

## License

MIT — Copyright (c) 2026 Cristian Catanzaro. See `LICENSE`.
