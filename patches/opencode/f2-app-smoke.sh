#!/usr/bin/env bash
# Parte 4: smoke de la API (generación off y LLM falso), 422/429, puerto liberado, sin IPs en disco.
set -uo pipefail
cd /home/capw/projects/grounded-rag
LOG=patches/opencode/diag/f2-app-smoke.log
OFF_PORT=8791
MOCK_PORT=8792
OFF="http://127.0.0.1:${OFF_PORT}"
MOCK="http://127.0.0.1:${MOCK_PORT}"

ask() {  # ask <base> <json>
  curl -sS -m 90 -H 'Content-Type: application/json' -d "$2" -w '\nHTTP %{http_code}\n' "$1/api/ask"
}

wait_health() {  # wait_health <base>
  for _ in $(seq 1 90); do
    if curl -sf -m 2 "$1/api/health" >/dev/null 2>&1; then return 0; fi
    sleep 1
  done
  return 1
}

{
  echo "== 1) GENERACIÓN OFF: GROUNDED_RAG_GENERATION=off, RATE_PER_IP=2/hour, puerto ${OFF_PORT} =="
  GROUNDED_RAG_GENERATION=off GROUNDED_RAG_RATE_PER_IP=2/hour \
    .venv/bin/python -m app --port "$OFF_PORT" > /tmp/opencode/f2-app-off.log 2>&1 &
  OFF_PID=$!
  wait_health "$OFF" || echo "ERROR: el servidor no respondió"
  echo "-- GET /api/health"
  curl -sS -m 5 -w '\nHTTP %{http_code}\n' "$OFF/api/health"
  echo "-- POST /api/ask (generación apagada: sólo fuentes recuperadas)"
  ask "$OFF" '{"question":"¿Cómo creo un índice parcial que sólo cubra filas no nulas?","lang":"es"}'
  echo "-- POST /api/ask (pregunta vacía => 422)"
  ask "$OFF" '{"question":"   "}'
  echo "-- POST /api/ask (501 caracteres => 422)"
  ask "$OFF" "$(python3 -c 'import json; print(json.dumps({"question": "a"*501}))')"
  echo "-- POST /api/ask (3ª petición con límite 2/hour => 429)"
  ask "$OFF" '{"question":"índice parcial"}'
  ask "$OFF" '{"question":"índice parcial otra vez"}'
  kill "$OFF_PID" 2>/dev/null
  wait "$OFF_PID" 2>/dev/null
  echo "-- puerto ${OFF_PORT} tras bajar el servidor:"
  ss -ltn 2>/dev/null | grep -F ":${OFF_PORT} " || echo "   libre"
  echo

  echo "== 2) LLM FALSO: GROUNDED_RAG_LLM_FAKE=mock, puerto ${MOCK_PORT} =="
  GROUNDED_RAG_LLM_FAKE=mock GROUNDED_RAG_RATE_PER_IP=10/hour \
    .venv/bin/python -m app --port "$MOCK_PORT" > /tmp/opencode/f2-app-mock.log 2>&1 &
  MOCK_PID=$!
  wait_health "$MOCK" || echo "ERROR: el servidor no respondió"
  echo "-- GET /api/health"
  curl -sS -m 5 -w '\nHTTP %{http_code}\n' "$MOCK/api/health"
  echo "-- POST /api/ask (respuesta con citas verificadas)"
  ask "$MOCK" '{"question":"How do I create a partial index?"}'
  echo "-- POST /api/ask (en español)"
  ask "$MOCK" '{"question":"¿Para qué sirve un índice BRIN?","lang":"es"}'
  kill "$MOCK_PID" 2>/dev/null
  wait "$MOCK_PID" 2>/dev/null
  echo "-- puerto ${MOCK_PORT} tras bajar el servidor:"
  ss -ltn 2>/dev/null | grep -F ":${MOCK_PORT} " || echo "   libre"
  echo

  echo "== 3) usage.sqlite: sólo fecha y tokens (sin preguntas, IPs ni cabeceras) =="
  .venv/bin/python - <<'PY'
import sqlite3
from pathlib import Path
path = Path("data/usage.sqlite")
rows = sqlite3.connect(path).execute("SELECT day, tokens FROM usage").fetchall()
print(f"filas: {rows}")
raw = path.read_bytes()
for needle in (b"127.0.0.1", b"testclient", b"Authorization", b"Bearer"):
    print(f"contiene {needle.decode()}: {needle in raw}")
PY
  echo
  echo "== 4) los logs del servidor no contienen cabeceras ni claves =="
  if grep -nEi 'authorization|bearer|sk-[a-z0-9]' /tmp/opencode/f2-app-off.log /tmp/opencode/f2-app-mock.log; then
    echo "   ¡ENCONTRADO!"
  else
    echo "   sin coincidencias"
  fi
  echo
  echo "== 5) limpieza: usage.sqlite era del smoke (tokens simulados), no consumo real =="
  rm -f data/usage.sqlite
  ss -ltn 2>/dev/null | grep -F -e ":${OFF_PORT} " -e ":${MOCK_PORT} " || echo "   ambos puertos libres"
} > "$LOG" 2>&1
