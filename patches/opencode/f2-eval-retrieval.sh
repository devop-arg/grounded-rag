#!/usr/bin/env bash
# Eval de recuperación tras sacar bookindex.html (parte 0).
set -uo pipefail
cd /home/capw/projects/grounded-rag
LOG=patches/opencode/diag/f2-eval-retrieval.log
{
  echo "== python -m eval --strict (80 preguntas, policy por defecto) =="
  .venv/bin/python -m eval --strict
  echo "exit: $?"
  echo
  echo "== python -m eval --compare =="
  .venv/bin/python -m eval --compare
  echo "exit: $?"
} > "$LOG" 2>&1
