# Devolución sobre el brief 03 (misma sesión, mismo contrato y reglas)

Buen trabajo y bien visto lo de `la`/`las`. Dos cambios acotados:

1. **"other" no es "no inglés"**: hoy `resolve_policy` trata `"other"` igual que `"es"`, así que una
   consulta sólo de identificadores (`pg_dump`, `VACUUM FREEZE`, `pg_stat_activity`) baja el peso
   léxico, y es justo el caso donde el léxico más sirve. Cambiar a: sólo `lang == "es"` recibe el
   tratamiento no inglés; `"other"` se trata como inglés. Tests para `pg_dump` y `VACUUM FREEZE`
   (stem, pesos 1/1).
2. **Stopwords en español para la consulta léxica**: agregar la política `es-stopwords`: si
   `lang == "es"`, antes de armar la expresión MATCH quitar las palabras funcionales en español
   (reusar `SPANISH_WORDS` de `retrieval/lang.py`, más signos `¿?¡!`), índice `stem`, pesos 1/1; y
   `es-stopwords+downweight` (además peso léxico 0.5). Si tras filtrar no queda ningún término, la lista
   léxica queda vacía. Tests de la función de filtrado.

Volvé a correr `python -m eval --compare` sobre `eval/questions.jsonl` y `eval/dev_es.jsonl` con las 6
políticas, elegí el default con el mismo criterio (mejor MRR en ambos sets; si empatan, la más simple),
actualizá el comentario de `DEFAULT_POLICY` y el README con los números del log, y regenerá
`es-compare.log`, `es-search.log` y `es-pytest.log`. No reconstruyas el índice (no hace falta).
Entregable: tabla nueva, default elegido y por qué, pytest.
