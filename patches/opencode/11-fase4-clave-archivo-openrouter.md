# Tarea: grounded-rag fase 4 — clave leída de archivo y proveedor OpenRouter

Repo `/home/capw/projects/grounded-rag`, rama `fase4-generacion-real` (ya creada; no cambies de rama).
Leé `README.md`, `.env.example`, `generation/config.py`, `generation/llm.py`, `app/main.py`,
`eval/__main__.py`/`eval/generate.py` y los tests de config y llm.

## Parte A — clave por archivo

- Variables nuevas: `GROUNDED_RAG_LLM_KEY_FILE` (ruta a un JSON) y `GROUNDED_RAG_LLM_KEY_PATH` (ruta con
  puntos dentro del JSON, por ejemplo `openrouter.key`). Precedencia: si `GROUNDED_RAG_LLM_API_KEY` no está
  vacía, gana. Si no, y están las dos de archivo, la clave se lee **en tiempo de ejecución** dentro de
  `build_llm` (no al importar ni en `load_config`), sin guardarla en `Config` ni escribirla en ningún lado.
- `llm_configured`/`require_llm` aceptan cualquiera de las dos vías (para el archivo, basta con que las dos
  variables estén y el archivo exista; no lo leas ahí).
- Errores (archivo inexistente, JSON inválido, ruta inexistente, valor vacío o no string) →
  `LLMNotConfigured` con un mensaje que nombra la variable y la ruta del archivo, **nunca** el valor ni el
  contenido del JSON. `repr`/`public()` muestran la ruta del archivo y la ruta de la clave, nunca la clave.
- **Prohibido** abrir, imprimir o leer de cualquier forma `~/.local/share/opencode/auth.json` y
  `~/.config/grounded-rag/env` vos mismo. Sólo los lee el programa. Los tests usan un JSON falso en `tmp_path`.

## Parte B — proveedor configurable, default OpenRouter

- `DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"` y `DEFAULT_MODEL = "deepseek/deepseek-v4.1-flash"`,
  los dos sobreescribibles por env como hoy.
- Sacar la cabecera `x-opencode-session` del cliente (y su mención en docstrings/README). No imitar
  cabeceras de otros clientes. Agregar las cabeceras opcionales de OpenRouter `HTTP-Referer:
  https://github.com/devop-arg/grounded-rag` y `X-Title: grounded-rag` (inofensivas en otros proveedores).
- Errores del proveedor: además del código HTTP, incluí en `LLMError` como máximo 200 caracteres de
  `error.message` del cuerpo JSON si existe (nunca headers), para diagnosticar.
- `.env.example` y README: proveedor por defecto, las variables nuevas, sin valores reales.

## Parte C — tests y humo

- Tests: precedencia, ruta con puntos, cada error sin filtrar el valor (asegurá que el valor falso no aparece
  en el mensaje), `repr`, cabeceras (sin `x-opencode-session`), defaults. `pytest` completo en verde →
  `patches/opencode/diag/f4-pytest.log`.
- Humo real (la config ya está en `~/.config/grounded-rag/env`): `free -m` (si "available" < 900 MB,
  esperá), luego `.venv/bin/python -m eval --generate --limit 3 > patches/opencode/diag/f4-generate-smoke.log 2>&1`.
  Si falla, copiá el error (sin clave) y terminá con BLOQUEADO. **No** corras las 80: eso lo hago yo.
- Comprobación de fuga: en Python, obtené la clave con la misma función del programa (sin imprimirla) e
  imprimí sólo `True/False` por archivo de `patches/opencode/diag/f4-*` y de `eval/reports/` nuevos.

## Recursos

Sin sudo, sin nginx/systemd, sin tocar `/home/capw/projects/dev0p/`. RAM escasa sin swap: un solo
proceso pesado a la vez. Comandos ≤ 90 s.

## Reglas fijas

- Este brief es tu contrato: no reclasifiques el riesgo ni amplíes el alcance con otras skills.
- No salgas del alcance: no refactorices ni "mejores" nada que no esté pedido.
- No hagas commit, push, deploy ni reinicies servicios.
- No inventes resultados: copiá salidas desde los logs. Si no corriste algo, decilo.
- Si no podés cumplir un criterio, terminá con BLOQUEADO + qué falta.

## Entregable

Archivos tocados y diff resumido, salida de pytest, resultado del humo con `--limit 3` (métricas y una
respuesta de ejemplo), comprobación de fuga, lo no verificado.
