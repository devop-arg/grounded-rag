# Continuación de la fase 1 (misma tarea, cambio de modelo)

La sesión anterior se cortó por cupo agotado del modelo en `ingest/download.py`. Seguí con el brief
`patches/opencode/01-fase1-ingesta-retrieval.md`, que sigue siendo el contrato, con este único ajuste aprobado:

- Confirmado: el tarball de PostgreSQL 17.11 NO trae `doc/src/sgml/html/` (0 archivos; sólo fuentes SGML).
  Se acepta tu desvío: bajar el HTML ya generado desde `https://www.postgresql.org/docs/17/`.
  Condiciones:
  - Mantené la descarga y verificación sha256 del tarball sólo para registrar versión exacta (17.11)
    en `ingest_stats.json`; no hace falta extraer nada de él. Si preferís no bajarlo, registrá la versión
    leyendo el directorio de ftp y decilo.
  - Rastreo cortés: User-Agent identificable (`grounded-rag-ingest/0.1 (+https://github.com/devop-arg)`),
    una sola conexión, pausa ≥ 0,3 s entre pedidos, reintentos con backoff ante 429/5xx, timeout 30 s.
  - Partir de `https://www.postgresql.org/docs/17/index.html` y de `bookindex.html`, seguir sólo enlaces
    relativos a páginas `*.html` del mismo directorio `/docs/17/` (sin query ni fragmento). Nunca salir
    de ese prefijo.
  - Caché en `data/html/<archivo>.html`: si existe, no se vuelve a pedir (idempotente; `--refresh` fuerza).
  - El HTML del sitio envuelve la doc en la plantilla de postgresql.org: el parser debe quedarse sólo con
    el contenido de la doc (el `div#docContent` o equivalente; verificá con una página real) y descartar
    menú, buscador, pie, banner de versiones y "Submit correction".
  - La descarga completa tarda varios minutos: en segundo plano con `nohup` y log, como dice el brief.
- El fixture de tests debe imitar esa estructura real (contenedor de la plantilla + doc adentro).

Lo ya hecho (`.gitignore`, `LICENSE`, `requirements.txt`, `.venv`, `ingest/download.py`) revisalo y seguí;
no empieces de cero. Mismas reglas fijas y mismo entregable final del brief 01.
