# Tarea: búsqueda léxica sin ruido en español + resultados agrupados por URL

Repo `/home/capw/projects/grounded-rag`, rama `fase1-es` (ya creada; no cambies de rama). Leé primero
`README.md`, `retrieval/store.py`, `retrieval/search.py`, `retrieval/fusion.py`, `eval/__main__.py`,
`ingest/__main__.py` y `tests/`. No llames a ninguna API de LLM.

## Problema

El índice FTS5 usa `porter unicode61 remove_diacritics 2` (stemmer inglés). En consultas en español mete
ruido: `"¿cómo configuro la replicación lógica?"` devuelve primero `ddl-inherit.html` porque "lógica"
colisiona con "logical", y la página esperada `logical-replication.html#LOGICAL-REPLICATION` cae al puesto 5.
Además, de los 8 resultados varios suelen ser fragmentos de la misma URL, lo que desperdicia lugares.

## Criterios de aceptación

1. **Detección de idioma** (`retrieval/lang.py`, función pura `detect_lang(query) -> "en" | "es" | "other"`)
   sin dependencias nuevas: heurística con `¿ ¡ ñ`, vocales acentuadas y listas cortas de palabras
   funcionales en español e inglés. Tests con al menos 10 casos (incluye consultas cortas, identificadores
   SQL como `pg_dump`/`VACUUM` que no deben decidir el idioma, y mezclas).
2. **Segundo índice léxico sin stemming**: tabla FTS5 adicional `chunks_fts_plain` con
   `tokenize='unicode61 remove_diacritics 2'` sobre los mismos campos. `Store.lexical(query, k, mode=...)`
   con `mode` en `"stem"` (actual) o `"plain"`. Si se agrega un parámetro a la interfaz, actualizar el Protocol.
3. **Peso léxico en RRF**: `rrf_fuse(lists, k=60, weights=None, top_n=None)` con
   `score = Σ w_i / (k + rank)`; sin `weights` el comportamiento es idéntico al actual (tests de antes
   siguen pasando sin cambios, y tests nuevos para pesos).
4. **Agrupado por URL** en `hybrid_search`: fusionar sobre los candidatos (top-50 + top-50), agrupar por
   `url` conservando el fragmento mejor rankeado como representante, agregar `chunk_ids` (todos los de esa URL
   que estaban entre los candidatos fusionados) y devolver las 8 mejores URLs **distintas**. Puntaje del
   grupo = el del mejor fragmento (no sumar; documentarlo). Test con un store falso en memoria.
5. **Política configurable y elección por datos**: `hybrid_search(query, k=8, policy=...)` con políticas
   - `baseline`: stem para todo, pesos 1/1 (lo actual, pero agrupado por URL);
   - `plain`: si `detect_lang != "en"` usar `plain`;
   - `downweight`: si `detect_lang != "en"` peso léxico 0.5 (probar también 0.3 y 0.0);
   - `plain+downweight`: ambas.
   `python -m eval --policy <p>` y `python -m eval --compare` (tabla con recall@5 y MRR por política,
   desglosado por idioma `lang`). El default de `hybrid_search` y del CLI pasa a ser la mejor política
   medida; si empatan, la más simple.
6. **Set de desarrollo**: crear `eval/dev_es.jsonl` (mismo formato que `questions.jsonl`) con 12 preguntas
   en español respondibles sobre temas variados de la doc (índices, MVCC, roles y permisos, `COPY`, JSONB,
   particionado, `EXPLAIN`, WAL, autovacuum, extensiones, triggers, full text search) y 4 en inglés de
   control. Cada `expected_urls` tiene que existir en `data/chunks.jsonl` (comprobalo y dejalo en log).
   No toques `eval/questions.jsonl`. `python -m eval --file <ruta>` para elegir el set.
   La elección de política se hace con `questions.jsonl` + `dev_es.jsonl`; reportá ambas por separado.
7. **Validación de URLs en eval**: `python -m eval` debe listar al principio toda `expected_url` que no
   exista en el corpus (página o ancla) y excluirla del cálculo avisando; con `--strict` sale con código 1.
8. CLI `python -m retrieval`: mostrar idioma detectado, política, y por resultado cuántos fragmentos agrupa.
9. README: actualizar sólo las secciones afectadas (búsqueda, eval), con números copiados del log.

## Reconstrucción del índice (CUIDADO)

`python -m ingest --skip-embed` BORRA `data/vectors.npy`. No lo uses. Los embeddings tardan 34 min y no
deben recalcularse. Agregá `--reuse-vectors` a `ingest`: reconstruye `index.sqlite` (ambas tablas FTS) desde
`data/chunks.jsonl` y conserva `vectors.npy`/`vector_ids.json` tal cual, verificando que `vector_ids.json`
coincida con los ids de `chunks.jsonl` en el mismo orden (si no, abortar). Hay backup en
`data/backup-fase1/` con `SHA256SUMS`: no lo modifiques. Al final, `sha256sum data/vectors.npy` debe ser
igual al del backup (dejalo en el log).

## Recursos

Sólo dentro del repo. Sin `sudo`, sin dependencias nuevas, sin tocar Docker/PostgreSQL/nginx/otros
proyectos. Hay otra sesión de opencode corriendo en la máquina y la RAM es escasa (sin swap): antes de
cargar el modelo corré `free -m`; si "available" < 900 MB, esperá o pará y reportalo. Una sola instancia
del modelo a la vez. Comandos de ≤ 90 s; lo largo con `nohup` + log.

## Evidencia (tee a `patches/opencode/diag/`)

- `es-pytest.log`: `.venv/bin/python -m pytest -q`
- `es-reindex.log`: reconstrucción con `--reuse-vectors` + sha256 de `vectors.npy` antes/después
- `es-urls.log`: comprobación de URLs de `dev_es.jsonl`
- `es-compare.log`: `python -m eval --compare` sobre `questions.jsonl` y sobre `dev_es.jsonl`
- `es-search.log`: `python -m retrieval` con `"¿cómo configuro la replicación lógica?"`,
  `"how do I create a partial index"` y `"¿qué niveles de aislamiento soporta PostgreSQL?"`

## Reglas fijas

- Este brief es tu contrato: no reclasifiques el riesgo ni amplíes el alcance con otras skills.
- No salgas del alcance: no refactorices ni "mejores" nada que no esté pedido.
- No hagas commit, push, deploy ni reinicies servicios salvo que este brief lo diga.
- No toques producción ni datos reales; si algo lo requiere, pará y reportalo.
- No inventes resultados: copiá números, hashes y salidas desde el log. Si no corriste algo, decilo.
- Si no podés cumplir un criterio, terminá con BLOQUEADO + qué falta, no con un éxito parcial disfrazado.

## Entregable

Archivos tocados, tabla de `--compare` (ambos sets, por idioma), política elegida y por qué, las 3
búsquedas (top-3 con URL y fragmentos agrupados), salida de pytest, sha256 de vectores, logs y lo no verificado.
