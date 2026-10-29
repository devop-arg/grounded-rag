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
.venv/bin/python -m ingest --reuse-vectors                  # rebuild index only, keep the vectors
.venv/bin/python -m retrieval "how do I create a partial index" -k 8
.venv/bin/python -m retrieval "¿cómo configuro la replicación lógica?" --policy baseline
.venv/bin/python -m eval
.venv/bin/python -m eval --file eval/dev_es.jsonl --policy plain
.venv/bin/python -m eval --file eval/dev_es.jsonl --compare
.venv/bin/python -m pytest -q
```

`ingest` flags: `--skip-download` (reuse `data/html`), `--skip-embed` (lexical
index only; it **deletes** `data/vectors.npy`), `--reuse-vectors` (rebuild
`data/index.sqlite` from `data/chunks.jsonl` and keep `data/vectors.npy` /
`data/vector_ids.json` untouched — it aborts if the two disagree, so a 34-minute
re-embedding is never accidental), `--limit N` (first N pages), `--refresh`
(re-fetch cached pages). Every run writes `data/ingest_stats.json` with
page/chunk counts, word statistics, per-stage seconds and index file sizes.

`eval` flags: `--file` (question set, default `eval/questions.jsonl`),
`--policy` (`baseline`, `plain`, `downweight`, `plain+downweight`,
`es-stopwords`, `es-stopwords+downweight`), `--compare`
(one table with recall@5 and MRR per policy, broken down by question language),
`--lexical-weight` (override the downweight value, for sweeps), `--json` and
`--strict` (exit 1 if an expected URL does not exist in the corpus; invalid
expected URLs are otherwise listed up front and dropped from the metrics).

## Index design

* **Chunks** — a page is split at `h2`/`h3` section boundaries (`div.sect1`,
  `div.sect2`, `div.refsect1`...); deeper headings stay inside their `h3`.
  Consecutive small sections of a page are merged and large ones are cut on
  block boundaries (never inside a `pre`), targeting 300–600 words. Each chunk
  gets a stable id `<file>#<anchor>#<n>`, its section path and its docs URL.
* **Lexical** — SQLite FTS5 over page title, section path and text, ranked by
  `bm25()`. Two tables share the same columns: `chunks_fts` with
  `porter unicode61 remove_diacritics 2` (English stemmer) and
  `chunks_fts_plain` with `unicode61 remove_diacritics 2` (no stemming, so a
  Spanish query is not folded into an unrelated English lexeme). User terms are
  tokenized, quoted and OR-ed, so FTS5 operators can never be injected.
* **Vector** — `intfloat/multilingual-e5-small` via fastembed (`passage: ` for
  chunks: title + breadcrumb + text, `query: ` for queries), from the int8 ONNX
  export; L2 normalized `float32` in `data/vectors.npy` (+ `data/vector_ids.json`),
  dot product search.
* **Fusion** — weighted reciprocal rank fusion,
  `score = Σ w_i / (60 + rank)`, over the top 50 lexical and top 50 vector hits.
  Without weights it is the original `Σ 1/(60 + rank)`.
* **Grouping** — candidates are grouped by URL before returning: the best ranked
  fragment of a URL is its representative, the score of a group is *that
  fragment's* score (not the sum, so a page is not inflated by having many
  fragments), and `chunk_ids` lists every candidate fragment of the page. The
  search returns `k` distinct URLs.

**Language-aware policies.** `detect_lang()` (`retrieval/lang.py`) is a pure
heuristic: `¿ ¡ ñ`, accented vowels and short function-word lists for Spanish and
English; SQL/CLI identifiers (`pg_dump`, `VACUUM`, anything with `_` or digits)
never vote, and a balanced mix is `"other"`. `hybrid_search(query, k=8,
policy=...)` uses it, and **only `lang == "es"` gets a non-English treatment** —
`"en"` and `"other"` (a bare `pg_dump` or `VACUUM FREEZE`, where the lexical
signal matters most) always keep the stemmed index with weights 1/1:

| Policy | Spanish queries |
| --- | --- |
| `baseline` | stemmed index, weights 1/1 (the old behaviour, still URL-grouped) |
| `plain` | plain index |
| `downweight` | lexical weight 0.5 |
| `plain+downweight` | plain index and lexical weight 0.5 |
| `es-stopwords` | `¿?¡!` and Spanish function words dropped from the lexical query |
| `es-stopwords+downweight` | the stopword filter plus lexical weight 0.5 |

The stopword filter (`strip_spanish_stopwords`) reuses `SPANISH_WORDS`: it
removes `la`, `de`, `cómo`, `qué`... but keeps identifiers and technical terms,
and a query made only of function words leaves the lexical list empty (the
vector list still answers). The default is `downweight`: at weight 0.5 it ties
with `plain+downweight` and `es-stopwords+downweight` on both sets (MRR@8
0.8333 and 0.5579) and is the simplest of the three. `es-stopwords` alone
(0.7083 / 0.5437) and `plain` (0.8125 / 0.4552) are worse than `baseline`
(0.8000 / 0.5250). See `patches/opencode/diag/es-compare.log`.

The lexical weight was also swept: `0.3` ties with `0.5` for `downweight` on both
sets, and `0.0` is best on the Spanish dev set alone (MRR@8 0.5849) but drops the
sample set to 0.6667, so 0.5 stays.

**Why SQLite + numpy?** pgvector needs a PostgreSQL extension (root on this
box) plus a server. SQLite is in the standard library, FTS5 gives real BM25
ranking, and even a few thousand 384-dim vectors are a handful of megabytes, so
the index is two plain files and the `Store` interface stays swappable for
pgvector later.

## Status

Phase 1 only: there is no generation step yet. On the 17.11 corpus (1143 pages,
2835 chunks), with URLs grouped, the default `downweight` policy scores:

| Set | recall@5 | MRR@8 |
| --- | --- | --- |
| `eval/questions.jsonl` (5 questions: 2 en, 2 es, 1 unanswerable) | 1.0 | 0.8333 |
| `eval/dev_es.jsonl` (16 questions: 12 es, 4 en) | 0.75 | 0.5579 |

The `baseline` policy (the previous behaviour, still URL-grouped) scores
1.0 / 0.8000 and 0.75 / 0.5250 on the same sets. The Spanish dev set is small
and noisy; treat the absolute numbers as a smoke test, not a benchmark. See
`data/ingest_stats.json` for timings and sizes.

## License

MIT — Copyright (c) 2026 Cristian Catanzaro. See `LICENSE`.
