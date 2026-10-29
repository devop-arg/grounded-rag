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
The lexical index is SQLite FTS5 with `bm25()` ranking; the vector index is an
E5 *small* multilingual checkpoint (int8 ONNX, 384 dims, the id lives in
`retrieval/embed.py`) driven through fastembed, L2-normalised `float32` in
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
exist in the corpus) and `--generate` / `--mock` / `--limit` / `--out` for the
generation run. A `--generate` run with any remote retrieval part configured
writes `<date>-generate-remote.json` instead of `<date>-generate.json`, so it
never overwrites the local report of the same day (`--out PATH` overrides both
names). `eval/README.md` describes the question set.

### Evaluation

Retrieval on the 80-question set (`eval/reports/retrieval.json`, default
`downweight` policy): **recall@5 0.88 / MRR@8 0.7336**, with 50 answerable and 30
unanswerable questions. `plain+downweight` is the best policy on the same set at
0.90 / 0.7380. Grouping by page instead of by anchored URL left those numbers
untouched and removed the duplicate pages from the result lists.

Generation, the real model with one call per question, 50 answerable and 30 not,
run twice over the same 80 questions: once on the local index and once with the
remote embeddings plus translation.

Local index (`eval/reports/2026-10-29-generate.json`, threshold 0.820859):

| Metric | All (80) | EN (40) | ES (40) |
| --- | --- | --- | --- |
| Answerable questions answered | 44/50 (88%) | 24/25 (96%) | 20/25 (80%) |
| Abstention precision | 0.8235 | 0.9333 | 0.7368 |
| Abstention recall | 0.9333 | 0.9333 | 0.9333 |
| Abstained for `model` / `threshold` | 23 / 11 | 12 / 3 | 11 / 8 |
| Valid citations | 315/315 (100%) | 170/170 (100%) | 145/145 (100%) |
| Sentences carrying a valid citation | 241/241 (100%) | 124/124 (100%) | 117/117 (100%) |
| Retrieval latency, median / p95 | 111.0 / 196.8 ms | 116.5 / 197.6 ms | 99.7 / 174.4 ms |
| LLM latency, median / p95 | 5803.7 / 15094.6 ms | 5794.2 / 13073.0 ms | 5901.4 / 15071.9 ms |
| Mean tokens per question | 8084.7 | 8666.3 | 7503.1 |
| Cost | $0.063846 | $0.034627 | $0.029218 |

Remote index, embeddings plus translation
(`eval/reports/2026-10-29-generate-remote.json`, threshold 0.436503 of that
index):

| Metric | All (80) | EN (40) | ES (40) |
| --- | --- | --- | --- |
| Answerable questions answered | 46/50 (92%) | 25/25 (100%) | 21/25 (84%) |
| Abstention precision | 0.875 | 1.0 | 0.7778 |
| Abstention recall | 0.9333 | 0.9333 | 0.9333 |
| Abstained for `model` / `threshold` | 19 / 13 | 8 / 6 | 11 / 7 |
| Valid citations | 334/335 (99.7%) | 162/163 (99.4%) | 172/172 (100%) |
| Sentences carrying a valid citation | 252/253 (99.6%) | 128/129 (99.2%) | 124/124 (100%) |
| Retrieval latency, median / p95 | 1451.7 / 2957.2 ms | 628.6 / 965.1 ms | 2000.2 / 3288.3 ms |
| LLM latency, median / p95 | 5849.5 / 17393.4 ms | 5302.2 / 14392.4 ms | 6354.4 / 18477.3 ms |
| Mean tokens per question | 8293.2 | 8425.3 | 8161.1 |
| Cost | $0.070060 | $0.034553 | $0.035507 |

The local run costs 570,424 input plus 76,353 output tokens and the remote one
575,092 plus 88,361, at the configured $0.045 and $0.50 per million. The remote
run adds $0.000022 for 120 retrieval calls (80 query embeddings and 40
translations, no error and no fallback), so the whole remote evaluation cost
$0.070082; every one of its 80 rows used the remote index. What it buys is the
Spanish answer rate (20/25 → 21/25) and the abstention precision (0.8235 → 0.875);
what it costs is 1.3 s more retrieval per question and one sentence dropped by
citation validation. Two of the labelled-unanswerable questions (`u02`, `u07`)
were answered in both runs because they are in fact premise corrections:
PostgreSQL 17 has no `VIRTUAL` generated columns and no `CREATE MATERIALIZED TABLE`,
and the answers say so while quoting the documentation for the supported
construct. Of the four answerable questions that abstained in the remote run, two
are Spanish and fell to the threshold (`a02`, `a44`) and two abstained by the
model (`a04`, `a10`); the local run also lost `a43` and `a46`, which the remote
index recovers.

Both thresholds are calibrated on the same questions, so the threshold and the
generation metrics are optimistically biased: treat them as a smoke test, not a
benchmark. The local one (`generation/threshold.json`: top-1 vector cosine
0.820859, 10/30 unanswerable abstained, 2/50 answerable lost) and the remote one
(`generation/threshold-remote.json`: top-1 vector cosine 0.436503, 11/30
unanswerable abstained, 2/50 answerable lost) are separate calibrations of two
different cosine scales. A `--generate --mock` run exercises the whole pipeline
offline and is never presented as a real evaluation: only a report without the
`-mock` suffix is summarised by the API.

## Optional remote retrieval

Retrieval runs fully offline by default. Three parts can be switched on through
the same OpenAI-compatible endpoint and key as generation. Each one is off while
its variable is empty, and each one falls back to the local path on any error
(network, 4xx/5xx, a 10 s timeout, a malformed body): the search is not failed,
the error is counted and the local behaviour for that part is used.

| Variable | What it does |
| --- | --- |
| `GROUNDED_RAG_RERANK_MODEL` | reranks the 20 best fused fragments (`POST /rerank`) before grouping by page; a failure keeps the RRF order |
| `GROUNDED_RAG_EMBED_MODEL`, `GROUNDED_RAG_EMBED_INDEX`, `GROUNDED_RAG_EMBED_DIMS` | embeds the query remotely (`POST /embeddings`) against `data/vectors-<index>.npy`; the local model is not loaded at all, and only built if the remote call fails and the local vectors are the fallback |
| `GROUNDED_RAG_TRANSLATE_MODEL` | translates a non-English question into English for the lexical (FTS5) query only: the vector search and the prompt keep the original question |

Model names containing `e5` get the `query: `/`passage: ` prefixes those
checkpoints expect. The `*_PRICE_*` variables (`GROUNDED_RAG_EMBED_PRICE_PER_M`,
`GROUNDED_RAG_RERANK_PRICE_PER_SEARCH`, `GROUNDED_RAG_TRANSLATE_PRICE_INPUT_PER_M`
and `_OUTPUT_PER_M`) are only used when the endpoint reports no `usage.cost`;
either way the cost of a request is added to the same monthly budget as the
answers.

```bash
.venv/bin/python -m retrieval.reindex --index <name> --model <model>   # batched, resumable
.venv/bin/python -m eval --compare-remote --rerank-model <model> \
    --embed-model <model>[,<model>] --embed-index <name>[,<name>] \
    --embed-dims <n>[,<n>] --translate-model <model>
```

`reindex` writes `data/vectors-<name>.npy` and `data/vector_ids-<name>.json` batch
by batch — every finished batch is flushed next to a progress file, so an
interrupted run resumes where it stopped — and it never touches `data/vectors.npy`
or `data/vector_ids.json`. `--compare-remote` measures the local baseline, every
part on its own and the combination of the parts that help, then writes
`eval/reports/<date>-compare-remote.json` with recall@5 / MRR@8 per language, the
median and p95 latency per query, the remote errors/fallbacks and the USD cost of
each combination. The models come from the flags or from the environment of the
run, never from a default in the code.

Measured on the 80-question set (`eval/reports/2026-10-29-compare-remote.json`,
policy `downweight`; the two embedding rows are the same model at 1536 and 512
dimensions, because that is what the endpoint served; the rerank row is the
fallback path, every call to the configured rerank model failed):

| Combination | recall@5 | MRR@8 | MRR EN | MRR ES | median / p95 ms |
| --- | --- | --- | --- | --- | --- |
| local baseline | 0.8800 | 0.7336 | 0.8048 | 0.6624 | 80.5 / 115.0 |
| + rerank | 0.8800 | 0.7336 | 0.8048 | 0.6624 | 180.6 / 234.8 |
| + remote embeddings (1536) | 0.9600 | 0.8180 | 0.8613 | 0.7747 | 583.6 / 726.8 |
| + remote embeddings (512) | 0.9800 | 0.8093 | 0.8493 | 0.7693 | 546.2 / 660.8 |
| + translation | 0.9200 | 0.7982 | 0.8048 | 0.7917 | 570.8 / 1663.0 |
| embeddings (1536) + translation | **0.9600** | **0.8563** | 0.8613 | 0.8513 | 1602.3 / 2335.3 |

The last row is the combination kept: it is the best MRR that does not lower
recall@5, and it is the only one that gains on both languages at once (Spanish
MRR 0.6624 → 0.8513). Rerank stays off because no rerank model answered. The
remote index costs a network round trip per query (about +0.5 s) and the
translation another one per non-English query, so the offline baseline is still
the fastest configuration.

Each index has its own abstention threshold, because the cosine of a remote
embedding lives on a different scale than the local one:
`generation/threshold.json` for the local index and
`generation/threshold-remote.json` (calibrated with
`python -m generation.calibrate --remote`) for the remote one. Both are read once
at startup, and every answer is compared with the threshold of the index its
search **really** used: a question embedded remotely is checked against the
remote file, and one whose remote embedding failed (the local index is then the
fallback) against the local file. An index whose file is missing simply has no
threshold. `GET /api/metrics` reports the active one under the running
configuration, labelled `index: "remote"` or `"local"`.

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
  "How it is evaluated" panel; `threshold` is the one the configuration uses
  (local or remote index) and carries the index it belongs to.
* `GET /` serves the web UI: an EN/ES and light/dark page with the question
  form, the citations of every sentence (hover, focus or tap to read the verbatim
  quote) and the evaluation panel. If the model is unavailable, if generation is
  off, if the request fails or if the monthly budget is spent, the page switches
  to **recorded mode** and replays an answer recorded from `web/recorded.json`,
  behind a short banner. Regenerate that file with `python -m web.record` (no
  model, no network): it replays `--from-report <real *-generate.json>` when one
  exists (the remote run of the day wins over the local one), otherwise it
  records the retrieved sources only.

Every response carries `Content-Security-Policy` (no inline script or style, no
JS CDN), `X-Content-Type-Options: nosniff` and `Referrer-Policy: no-referrer`.

## Configuration

Every setting is a `GROUNDED_RAG_*` variable, read with the standard library
only from `~/.config/grounded-rag/env` (`CLAVE=valor`, optional), then from
`.env.local` at the root of the repository (same format, ignored by git: the
per-checkout settings) and finally from the real environment, which wins.
Precedence, highest first: the environment, `.env.local`, the user file and the
defaults of the code. A missing file is simply skipped and an empty value means
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
