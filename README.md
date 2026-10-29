# grounded-rag

A grounded retrieval-augmented assistant over the official **PostgreSQL 17**
documentation. It ingests and indexes the manual, answers questions from the
retrieved passages only, and verifies every citation on the server: a sentence
whose quote is not a verbatim phrase of the passage it cites is dropped. When
the documentation does not cover the question, it abstains instead of guessing.
Ingestion, embeddings, retrieval and retrieval evaluation run fully offline;
only the generation step talks to a model endpoint.

| Path | Purpose |
| --- | --- |
| `ingest/` | fetch the docs, parse HTML into sections, chunk, embed, index |
| `retrieval/` | SQLite FTS5 + numpy vector store, weighted RRF fusion, `hybrid_search` |
| `generation/` | config, prompt, OpenAI-compatible client, citation validation, threshold |
| `app/` | FastAPI: `POST /api/ask`, `GET /api/health`, `GET /api/metrics`, static `web/` |
| `web/` | the UI (HTML/CSS/vanilla JS), `recorded.json` and its recorder `web/record.py` |
| `eval/` | recall@5 / MRR, plus `--generate` answer/citation/abstention metrics |
| `tests/` | unit tests (no network, no model) |
| `data/` | generated: raw tarball, HTML pages, `chunks.jsonl`, index, stats, `usage.sqlite` |

## Requirements

* Python 3.12
* ~1 GB of free RAM: embeddings run with 2 threads and 4-chunk batches on the
  int8 ONNX export (batch 32 peaked at ~1.5 GB of RSS)
* Network for ingestion: `ftp.postgresql.org` (release tarball and `.sha256`),
  `www.postgresql.org` (rendered HTML; the source tarball ships DocBook/SGML
  only) and the model hub that serves the embedding model (cached in
  `data/models/`). Generation needs an OpenAI-compatible endpoint; everything
  else runs offline.

## Install

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Ingest

```bash
.venv/bin/python -m ingest                                  # docs -> chunks -> vectors -> index
.venv/bin/python -m ingest --skip-download --skip-embed --limit 20
.venv/bin/python -m ingest --skip-download --reuse-vectors  # re-chunk + reindex, keep the vectors
.venv/bin/python -m retrieval "how do I create a partial index" -k 8
```

`ingest` flags: `--skip-download` (reuse `data/html`), `--skip-embed` (lexical
index only; it deletes `data/vectors.npy`), `--reuse-vectors` (re-chunk
`data/html` and rebuild `data/index.sqlite` without re-embedding: identical
chunk ids keep `vectors.npy` untouched, a strict subset filters its rows by id,
and any new id without a vector aborts before writing anything), `--limit N`
(first N pages) and `--refresh` (re-fetch cached pages). Every run writes
`data/ingest_stats.json` with page/chunk counts, word statistics, per-stage
seconds and index file sizes.

Chunks are cut at `h2`/`h3` section boundaries, merged when small and split on
block boundaries (never inside a `pre`) when large, targeting 300–600 words.
The lexical index is SQLite FTS5 with `bm25()` ranking; the vector index is
`intfloat/multilingual-e5-small` through fastembed, L2-normalised `float32` in
`data/vectors.npy`. Both rankings are fused with weighted reciprocal rank
fusion and grouped by page, so one page never takes two of the eight slots.

## Evaluate

```bash
.venv/bin/python -m eval                                    # recall@5 / MRR on 80 questions
.venv/bin/python -m eval --file eval/dev_es.jsonl --compare  # one table per policy
.venv/bin/python -m generation.calibrate                     # abstention threshold (retrieval only)
.venv/bin/python -m eval --generate                          # answers, citations, abstention
.venv/bin/python -m eval --generate --mock --limit 5         # offline pipeline check
.venv/bin/python -m pytest -q
```

`eval` flags: `--file`, `--policy` (`baseline`, `plain`, `downweight`,
`plain+downweight`, `es-stopwords`, `es-stopwords+downweight`), `--compare`,
`--lexical-weight`, `--json`, `--strict` (exit 1 if an expected URL does not
exist in the corpus) and `--generate` / `--mock` / `--limit` for the generation
run. `eval/README.md` describes the question set.

### Evaluation

Retrieval on the 80-question set (`eval/reports/retrieval.json`, default
`downweight` policy): **recall@5 0.88 / MRR@8 0.7336**, with 50 answerable and 30
unanswerable questions. `plain+downweight` is the best policy on the same set at
0.90 / 0.7380. Grouping by page instead of by anchored URL left those numbers
untouched and removed the duplicate pages from the result lists.

The abstention threshold is calibrated on the same questions
(`generation/threshold.json`: top-1 vector cosine 0.820859, 10/30 unanswerable
abstained, 2/50 answerable lost), so both the threshold and the generation
metrics are optimistically biased: treat them as a smoke test, not a benchmark.
A `--generate --mock` run exercises the whole pipeline offline and is never
presented as a real evaluation: only a report without the `-mock` suffix is
summarised by the API.

## API

`python -m app` serves uvicorn on `127.0.0.1` only (a non-loopback `--host` or
`GROUNDED_RAG_HOST` is refused with exit 2). No systemd unit, no reverse proxy,
no CORS.

* `POST /api/ask` `{question, lang?}` → `{abstain, sentences, sources}` plus
  `reason` when abstaining (`threshold`, `model`, `validation`,
  `generation_disabled`). Empty/whitespace questions and questions over 500
  characters are 422; over the per-IP limit is 429; over the monthly USD budget
  is 503 with `{"detail": ..., "reason": "budget_exhausted"}` **without calling
  the model**; a failing model is 502 with no internal detail. The IP comes from
  `request.client.host` and is only ever used as `sha256(salt + ip)` in memory:
  neither the address nor the digest is stored or logged. `data/usage.sqlite`
  keeps one row per UTC month with a cost and two token counts, and nothing
  else.
* `GET /api/health` → `{status, generation, llm_configured, budget}`.
* `GET /api/metrics` → `{retrieval, threshold, generation, budget}` for the
  "How it is evaluated" panel.
* `GET /` serves the web UI: an EN/ES and light/dark page with the question
  form, the citations of every sentence (hover, focus or tap to read the verbatim
  quote) and the evaluation panel. If the model is unavailable, if generation is
  off, if the request fails or if the monthly budget is spent, the page switches
  to **recorded mode** and replays an answer recorded from `web/recorded.json`,
  behind a short banner. Regenerate that file with `python -m web.record` (no
  model, no network).

Every response carries `Content-Security-Policy` (no inline script or style, no
JS CDN), `X-Content-Type-Options: nosniff` and `Referrer-Policy: no-referrer`.

## Configuration

Every setting is a `GROUNDED_RAG_*` variable, read from
`~/.config/grounded-rag/env` (`CLAVE=valor`, optional, read with the standard
library) and then from the real environment, which wins. An empty value means
"not set", so the shipped `.env.example` (every value blank) changes nothing.
Nothing is hardcoded: the endpoint, the model and the key exist only if the
variables below are set. Secrets are never printed, logged, serialised into a
report or quoted in an error.

| Variable | Meaning |
| --- | --- |
| `GROUNDED_RAG_LLM_BASE_URL` | OpenAI-compatible base URL (`/chat/completions` is appended) |
| `GROUNDED_RAG_LLM_MODEL` | model name |
| `GROUNDED_RAG_LLM_API_KEY` | bearer token; without it (and without the key file) generation is unavailable and the API answers 503 |
| `GROUNDED_RAG_LLM_KEY_FILE` | JSON file holding the key; alternate to the token above, read only when that token is empty |
| `GROUNDED_RAG_LLM_KEY_PATH` | dotted path to the key inside that JSON (for example `llm.key`) |
| `GROUNDED_RAG_LLM_REFERER` | optional `HTTP-Referer` header, left out when empty |
| `GROUNDED_RAG_LLM_TITLE` | optional `X-Title` header, left out when empty |
| `GROUNDED_RAG_LLM_PRICE_INPUT_PER_M` | USD per million prompt tokens, used to price every answer (default `0`) |
| `GROUNDED_RAG_LLM_PRICE_OUTPUT_PER_M` | USD per million completion tokens (default `0`) |
| `GROUNDED_RAG_BUDGET_USD_PER_MONTH` | USD allowed per UTC month, persisted in `data/usage.sqlite` (default `0.50`) |
| `GROUNDED_RAG_GENERATION` | `on` (default) or `off`: retrieval only, `abstain: true` + `reason: "generation_disabled"` |
| `GROUNDED_RAG_RATE_PER_IP` | sliding-window limit per hashed IP (default `10/hour`) |
| `GROUNDED_RAG_IP_SALT` | salt mixed into `sha256(salt + ip)`; an empty salt means a random per-process one, never printed or logged |
| `GROUNDED_RAG_PORT` | API port (`127.0.0.1` only, default `8790`) |

The cost of an answer is `prompt_tokens × input / 1e6 + completion_tokens ×
output / 1e6`, accumulated per UTC month (`YYYY-MM`). Before calling the model
the server compares the month's spending with the budget and answers 503
`budget_exhausted` when it is spent; `/api/health` and `/api/metrics` expose the
month, the spending and the cap (never a key, a salt or a client IP).

`GROUNDED_RAG_LLM_FAKE=mock` (test only) swaps the client for the deterministic
offline model used by `eval --generate --mock` and by the app smoke test.

## License

MIT — see `LICENSE`.
