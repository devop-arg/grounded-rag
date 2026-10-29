#!/usr/bin/env bash
# Segunda corrida de --reuse-vectors con los ids ya filtrados: debe ser idempotente
# y no reescribir vectors.npy / vector_ids.json.
set -uo pipefail
cd /home/capw/projects/grounded-rag
LOG=patches/opencode/diag/f2-reindex.log
{
  echo
  echo "== segunda corrida: --skip-download --reuse-vectors (ids ya filtrados) =="
  .venv/bin/python - <<'PY'
import hashlib
for name in ("vectors.npy", "vector_ids.json"):
    print(f"sha256 {name}: {hashlib.sha256(open('data/'+name,'rb').read()).hexdigest()}")
PY
  .venv/bin/python -m ingest --skip-download --reuse-vectors
  echo "exit: $?"
  .venv/bin/python - <<'PY'
import hashlib
for name in ("vectors.npy", "vector_ids.json"):
    print(f"sha256 {name}: {hashlib.sha256(open('data/'+name,'rb').read()).hexdigest()}")
PY
} >> "$LOG" 2>&1
