#!/usr/bin/env bash
# Parte 3: eval --generate (sin clave => código 2) y --generate --mock (pipeline completo offline).
set -uo pipefail
cd /home/capw/projects/grounded-rag
LOG=patches/opencode/diag/f2-generate-mock.log
{
  echo "== python -m eval --generate (sin API key configurada) =="
  .venv/bin/python -m eval --generate
  echo "exit: $?"
  echo
  echo "== python -m eval --generate --mock (80 preguntas, concurrencia 1) =="
  .venv/bin/python -m eval --generate --mock
  echo "exit: $?"
  echo
  echo "== reporte escrito =="
  ls -l eval/reports/ || true
} > "$LOG" 2>&1
