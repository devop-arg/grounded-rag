#!/usr/bin/env bash
# Parte 0: reindex sin re-embeddear (evidencia en patches/opencode/diag/f2-reindex.log)
set -euo pipefail
cd /home/capw/projects/grounded-rag
LOG=patches/opencode/diag/f2-reindex.log
{
  echo "== before: chunks.jsonl =="
  .venv/bin/python - <<'PY'
import json
rows=[json.loads(l) for l in open('data/chunks.jsonl', encoding='utf-8')]
from collections import Counter
c=Counter(r['page_file'] for r in rows)
print(f"chunks: {len(rows)}")
print(f"bookindex chunks: {c.get('bookindex.html',0)}")
print(f"bookindex urls: {sorted({r['url'].split('#')[0] for r in rows if r['page_file']=='bookindex.html'})}")
print(f"pages with chunks: {len(c)}")
PY
  echo
  echo "== vectors before =="
  .venv/bin/python - <<'PY'
import json, numpy as np
v=np.load('data/vectors.npy', mmap_mode='r')
ids=json.load(open('data/vector_ids.json'))
print(f"vectors.npy: {v.shape} {v.dtype}; vector_ids: {len(ids)}")
PY
  echo
  echo "== ingest --skip-download --reuse-vectors =="
  .venv/bin/python -m ingest --skip-download --reuse-vectors
  echo
  echo "== after =="
  .venv/bin/python - <<'PY'
import json, numpy as np
from collections import Counter
rows=[json.loads(l) for l in open('data/chunks.jsonl', encoding='utf-8')]
c=Counter(r['page_file'] for r in rows)
v=np.load('data/vectors.npy', mmap_mode='r')
ids=json.load(open('data/vector_ids.json'))
print(f"chunks: {len(rows)}")
print(f"bookindex chunks: {c.get('bookindex.html',0)}")
print(f"pages with chunks: {len(c)}")
print(f"vectors.npy: {v.shape} {v.dtype}; vector_ids: {len(ids)}; rows==ids: {v.shape[0]==len(ids)}")
print(f"vectors aligned with chunks: {ids==[r['id'] for r in rows]}")
PY
} 2>&1 | tee "$LOG"
