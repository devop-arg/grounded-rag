# Brief 10: agrupar resultados por página (misma sesión, mismo contrato del brief 08)

Rama `fase3-interfaz`. Hoy `hybrid_search` agrupa por URL completa (con `#ancla`), así que una misma página
aparece varias veces (por ejemplo `sql-createindex.html#...CONCURRENTLY` y `sql-createindex.html#id-1.9.3.69.7`;
lo mismo con JSON Functions).

1. Agrupar por **página** (la URL sin `#ancla`, o sea `page_file`) en `retrieval/search.py`
   (`group_by_url` o una función nueva: decidí y actualizá docstring y callers). Representante = el fragmento mejor
   rankeado de esa página; el `url` devuelto conserva **su** ancla. `chunk_ids` = todos los fragmentos de esa página
   entre los candidatos fusionados, en orden de rank. Puntaje del grupo = el del mejor fragmento (como antes).
   Devolver 8 páginas distintas.
2. Eval: una URL esperada con `#ancla` debe acertar si **cualquiera** de los fragmentos agrupados en el resultado
   tiene esa ancla (no sólo el representante); sin `#`, si la página coincide. Hacé que `hybrid_search` exponga
   lo necesario (por ejemplo `anchors` o `urls` de los fragmentos del grupo) y ajustá `eval`. Documentalo en
   `eval/README.md` (una línea).
3. Tests: dos fragmentos de la misma página con anclas distintas → un solo resultado con el ancla del mejor y
   ambos `chunk_ids`; 8 páginas distintas; el acierto por ancla secundaria en eval.
4. Re-correr `python -m eval --strict` y `--compare` → `patches/opencode/diag/f3-group-page-eval.log`, con
   una tabla antes/después (antes: downweight all 0.88 / 0.7336, en 0.92/0.8048, es 0.84/0.6624) y la lista
   de preguntas cuyo rank cambió. Regenerá `eval/reports/retrieval.json` y `web/recorded.json`
   (`python -m web.record`), y verificá que en recorded.json ninguna entrada repite página en sus fuentes
   (dejalo en el log). Actualizá en el README los números que cambien.
5. `f3-pytest.log` en verde. Rehacé `f3-desktop.png` (mismo setup) y verificá que ya no se repite CREATE INDEX.

`free -m` antes de cargar el modelo de embeddings (si "available" < 900 MB, esperá); un solo proceso
pesado a la vez; bajá la app al terminar. Mismas reglas fijas: sin commit/push, sin tocar dev0p, sin leer
credenciales, sin inventar números.

Entregable: diff resumido, tabla antes/después, preguntas con rank cambiado, chequeo de recorded.json, pytest.
