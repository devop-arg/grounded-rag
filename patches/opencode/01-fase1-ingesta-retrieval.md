# Tarea: grounded-rag fase 1 — ingesta, índice híbrido, retrieval y eval

Trabajás en `/home/capw/projects/grounded-rag` (ya existe, vacío salvo `patches/`). Es un proyecto
nuevo: un asistente RAG sobre la documentación oficial de PostgreSQL 17. En esta fase NO se llama a
ninguna API de LLM: sólo ingesta, embeddings locales, búsqueda híbrida y evaluación de recuperación.

## Criterios de aceptación

1. `git init -b main` hecho (sin commits: los hago yo). `.gitignore` con `.venv/`, `data/`,
   `__pycache__/`, `.pytest_cache/`, `*.egg-info/`, `patches/opencode/diag/*.log` NO (los logs sí se versionan).
2. `LICENSE` MIT, `Copyright (c) 2026 Cristian Catanzaro`.
3. `.venv` creado con `python3 -m venv .venv`; dependencias en `requirements.txt` con versiones fijadas
   (`==`): `beautifulsoup4`, `lxml`, `fastembed`, `numpy`, `pytest`. Nada más salvo que sea imprescindible
   (justificarlo en el reporte). Instalar sólo con `.venv/bin/pip`.
4. `python -m ingest` (todo con `.venv/bin/python`) hace, de forma idempotente:
   - Detecta la última versión 17.x en `https://ftp.postgresql.org/pub/source/` (hoy es v17.11; verificá
     que exista `postgresql-17.N.tar.bz2`; si la última no tiene tarball, usá la anterior) y la descarga a
     `data/raw/` si no está ya. Verificá el `.sha256` publicado al lado del tarball.
   - Extrae SÓLO `postgresql-17.N/doc/src/sgml/html/*.html` a `data/html/` (usar `tarfile` con filtro de
     miembros; no extraer el tarball entero; rechazar rutas con `..` o absolutas).
   - Parsea cada página y la parte en fragmentos → `data/chunks.jsonl`.
   - Calcula embeddings y construye el índice (ver punto 5).
   - Imprime y guarda en `data/ingest_stats.json`: versión, páginas, fragmentos, palabras
     min/mediana/max por fragmento, segundos por etapa, tamaño en bytes de cada archivo del índice.
   - Flags: `--skip-download`, `--skip-embed` (para iterar rápido), `--limit N` (sólo N páginas, para pruebas).
5. Fragmentación (`ingest/chunk.py`, funciones puras testeables):
   - Las páginas HTML de PG usan `<div class="sect1|sect2|sect3|refsect1|refsect2|..." id="ANCLA">` con
     títulos en `h2`/`h3`/`h4` dentro de `div.titlepage`. Cada fragmento pertenece a la sección más
     interna con `id` que lo contiene. Cortar por secciones h2/h3; las h4 y más profundas quedan dentro de su h3.
   - Quitar navegación (`div.navheader`, `div.navfooter`), `div.toc` y el cuerpo de páginas que son sólo
     índice/TOC (si una página queda con < 40 palabras de texto propio, descartarla y contarla).
   - Mantener texto de `pre` (código) y tablas (celdas separadas por ` | `), normalizando espacios.
   - Tamaño objetivo 300–600 palabras: unir secciones consecutivas chicas de la misma página (el ancla del
     fragmento unido es la de la primera) y partir secciones grandes en límites de párrafo/bloque
     (nunca en medio de un `pre`; si un único bloque supera 600, se acepta solo). Una página entera
     menor a 300 palabras es un fragmento.
   - Campos por línea de `chunks.jsonl`: `id` (estable: `<archivo>#<ancla>#<n>`), `page_file`,
     `page_title`, `section_path` (lista de títulos desde la página hasta la sección), `anchor`,
     `url` = `https://www.postgresql.org/docs/17/<archivo>.html#<ANCLA>` (sin `#` si no hay ancla;
     respetar mayúsculas del id tal cual está en el HTML), `text`, `n_words`.
6. Índice (`retrieval/store.py`): no hay pgvector disponible sin sudo, así que se usa SQLite + numpy
   detrás de una interfaz común:
   ```python
   class Store(Protocol):
       def lexical(self, query: str, k: int) -> list[tuple[str, float]]: ...   # (chunk_id, score)
       def vector(self, qvec: np.ndarray, k: int) -> list[tuple[str, float]]: ...
       def get(self, ids: list[str]) -> list[dict]: ...
   ```
   Implementación `SqliteNumpyStore`: `data/index.sqlite` con tabla `chunks` y tabla FTS5
   (`tokenize='porter unicode61 remove_diacritics 2'`) sobre título de página, ruta de secciones y texto;
   ranking con `bm25()`. La consulta de usuario se tokeniza y cada término se pone entre comillas dobles
   unido con `OR` (escapar comillas; nunca pasar la cadena cruda a MATCH). Vectores en
   `data/vectors.npy` (float32, normalizados L2) + `data/vector_ids.json`; búsqueda por producto punto.
7. Embeddings (`retrieval/embed.py`): `fastembed.TextEmbedding("intfloat/multilingual-e5-small")`,
   prefijo `"passage: "` para fragmentos (embeber `page_title + " — " + " > ".join(section_path) + "\n" + text`)
   y `"query: "` para consultas. Caché del modelo en `data/models/` (`cache_dir`). Batch chico (32) y
   `threads=2`: la máquina tiene ~1,5 GB de RAM libre y NO tiene swap.
8. `retrieval/`: `hybrid_search(query, k=8)` toma top-50 léxico y top-50 vectorial y los fusiona con
   Reciprocal Rank Fusion `score = Σ 1/(60 + rank)` (rank desde 1), devuelve los 8 mejores con
   `rrf`, `lexical_rank`, `vector_rank` (o `None`), `url`, `page_title`, `section_path` y un recorte del texto.
   `rrf_fuse` en `retrieval/fusion.py`, función pura. CLI: `python -m retrieval "pregunta"` imprime la
   tabla y la latencia en ms (excluyendo la carga del modelo, que se imprime aparte); `--json` para salida JSON.
9. `eval/`: `python -m eval` lee `eval/questions.jsonl` (campos `id`, `lang`, `question`, `answerable`,
   `expected_urls`), corre `hybrid_search` y para las `answerable=true` calcula recall@5 (fracción de
   preguntas con alguna URL esperada en el top-5) y MRR (sobre top-8). Una URL esperada sin `#` acierta
   con cualquier fragmento de esa página; con `#` exige esa ancla. Imprime detalle por pregunta y
   resumen; `--json` opcional. Dejar 5 preguntas de ejemplo: 3 en inglés y 2 en español, 4 respondibles y
   1 no respondible (algo que la doc de PG no cubre, p. ej. sobre MySQL). Las `expected_urls` tienen que
   existir de verdad en `chunks.jsonl` (comprobalo con grep y dejalo en el log).
10. `tests/` con pytest, sin red ni modelo: parseo de un HTML fixture chico escrito a mano con la
    estructura de PG (sect1/sect2/sect3, navheader, toc, pre, table), fragmentación (unión de chicos,
    corte de grandes, URLs y anclas, ids estables), escape de la consulta FTS5 (comillas, `*`, `-`, `NEAR`),
    y `rrf_fuse` (orden, empates, listas disjuntas, k=60). Todos pasan.
11. `README.md` en inglés, corto (≤ 80 líneas): qué es, fase actual, requisitos, setup, comandos
    (`ingest`, `retrieval`, `eval`, `pytest`), diseño del índice y por qué SQLite+numpy, licencia.
    Sin números inventados: si citás métricas, copiarlas del log.

## Estructura esperada

```
ingest/__init__.py __main__.py download.py parse.py chunk.py
retrieval/__init__.py __main__.py store.py embed.py fusion.py search.py
eval/__init__.py __main__.py questions.jsonl
tests/fixtures/sample.html tests/test_parse_chunk.py tests/test_fusion.py tests/test_store_query.py
README.md LICENSE requirements.txt .gitignore
```
Ajustes menores de nombres están bien si se mantienen los comandos `python -m ingest|retrieval|eval`.

## Recursos: qué podés y qué no

- Sólo escribir dentro de `/home/capw/projects/grounded-rag`. No usar `sudo`, no `apt`, no instalar
  nada fuera del venv, no tocar Docker, PostgreSQL, nginx, servicios ni otros proyectos.
- No guardar claves ni tokens en ningún archivo. No hay llamadas a APIs de LLM en esta fase.
- Red permitida sólo para: ftp.postgresql.org, PyPI (pip) y Hugging Face (descarga del modelo de fastembed).

## Memoria y tiempos (importante)

- RAM libre ~1,5 GB y sin swap: antes del paso de embeddings corré `free -m`; si "available" < 900 MB,
  pará y reportalo. Nunca cargues dos instancias del modelo a la vez.
- Tu shell corta los comandos a 120 s. La ingesta completa tarda más: lanzala en segundo plano
  `nohup .venv/bin/python -m ingest > patches/opencode/diag/fase1-ingest.log 2>&1 &` y consultá el log
  cada tanto con `tail` (comandos cortos). Que la ingesta imprima progreso cada ~500 fragmentos.
- Probá primero con `--limit 20` antes de la corrida completa.

## Evidencia (cada verificación con tee a `patches/opencode/diag/`)

- `fase1-pytest.log`: `.venv/bin/python -m pytest -q`
- `fase1-ingest.log`: corrida completa (y `data/ingest_stats.json`)
- `fase1-search.log`: tres búsquedas: `"how do I create a partial index"`,
  `"what isolation levels does PostgreSQL support"`, `"¿cómo configuro la replicación lógica?"`
- `fase1-eval.log`: `python -m eval`
- `fase1-urls.log`: la comprobación de que las `expected_urls` existen en `chunks.jsonl`
- `du -sh data/*` en `fase1-sizes.log`

## Reglas fijas

- Este brief es tu contrato: no reclasifiques el riesgo ni amplíes el alcance con otras skills.
- No salgas del alcance: no refactorices ni "mejores" nada que no esté pedido.
- No hagas commit, push, deploy ni reinicies servicios salvo que este brief lo diga.
- No toques producción ni datos reales; si algo lo requiere, pará y reportalo.
- No inventes resultados: copiá números, hashes y salidas desde el log. Si no corriste algo, decilo.
- Si no podés cumplir un criterio, terminá con BLOQUEADO + qué falta, no con un éxito parcial disfrazado.

## Entregable (al final, en tu respuesta)

Lista de archivos creados, versión de PG usada, páginas/fragmentos/descartadas, tamaño del índice,
tiempo de ingesta por etapa, latencia de búsqueda, las 3 búsquedas (top-3 de cada una con URL),
resultado de eval, salida de pytest, rutas de logs y lo que no hayas verificado.
