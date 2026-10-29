# Evaluation set

`questions.jsonl` holds 80 questions about the PostgreSQL 17 documentation:

- 50 answerable (25 English, 25 Spanish). Each lists `expected_urls`: the doc pages that answer it. A hit counts when a retrieved chunk's URL, without its `#anchor`, matches one of them; an expected URL *with* an `#anchor` matches any fragment grouped into that result, not only its representative.
- 30 unanswerable (15 per language), with a `category`: `later_version`, `invented`, `nonexistent_feature`, `other_engine`, `outside_corpus`, `opinion_pricing`, `off_topic`, `prompt_injection`. The expected behaviour is to abstain.

The expected pages were written by hand from the structure of the PostgreSQL 17 docs. Before the first run, check every `expected_urls` entry against the crawled pages and fail loudly on any URL that is missing from the corpus. Fix the set rather than drop the question.

Metrics: recall@5 and MRR on the answerable questions (no API cost, runs in CI); abstention precision and recall, citation precision and faithfulness on the full set, with the generation step (`python -m eval --generate`, or `--generate --mock` offline).
