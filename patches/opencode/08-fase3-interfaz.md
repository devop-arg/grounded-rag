# Tarea: grounded-rag fase 3: interfaz web bilingüe

Repo `/home/capw/projects/grounded-rag`, rama `fase3-interfaz` (ya creada desde `fase2-generacion`; no
cambies de rama). Leé primero `README.md`, `app/main.py`, `app/__main__.py`, `generation/generate.py`,
`eval/generate.py`, `eval/reports/` y `tests/test_app.py`.

## Estilo (sólo lectura)

Tomá los tokens de `/home/capw/projects/dev0p/index.html` (líneas ~20-35 y ~170-175). **No modifiques nada en
`/home/capw/projects/dev0p/`**. Tokens:
- colores `--bg --panel --panel-2 --line --fg --muted --signal --signal-soft --amber --amber-soft`, con
  modo oscuro por defecto y modo claro por `prefers-color-scheme: light`, `:root[data-theme="light"]` y
  `:root[data-theme="dark"]`;
- tipografías `--display` (Chakra Petch), `--body` (Hanken Grotesk) y `--mono` (Martian Mono), desde Google Fonts
  como allá;
- botones/píldoras/`.tag` en mono con bordes de 2–3 px de radio.
Estética "sala de control": sobria, densa, sin adornos. Sin frameworks, sin CDNs de JS, sin trackers ni
analytics. HTML + CSS + JS vanilla en `web/` (`index.html`, `app.css`, `app.js`, `i18n.json` o
diccionario en el JS).

## Contenido

- Encabezado: nombre "grounded-rag", subtítulo en una línea ("Answers from the PostgreSQL 17 docs, with
  verifiable citations" / "Respuestas desde la documentación de PostgreSQL 17, con citas verificables"),
  selector EN/ES (inglés por defecto; preferencia en `localStorage` dentro de try/catch; también respeta
  `?lang=es`), selector de tema claro/oscuro (también en localStorage con try/catch) y enlace al repo
  GitHub `https://github.com/devop-arg/grounded-rag`.
- Formulario: `textarea` con etiqueta, contador `n/500`, botón Preguntar/Ask (Enter envía, Shift+Enter
  salto), 4 ejemplos como botones (2 EN, 2 ES; preguntas respondibles reales de `eval/questions.jsonl`).
- Respuesta: cada frase con sus citas como `[n]` (botón accesible). Al pasar el mouse, enfocar o tocar
  aparece un popover con la frase citada textual, el título de la fuente y el enlace a la sección en
  postgresql.org (abre en pestaña nueva, `rel="noopener"`). Panel "Fuentes" con `n`, título y URL.
- Abstención: estado claro con borde ámbar: "No tengo información suficiente en la documentación para
  responder esto." / "The documentation does not contain enough information to answer this." Mostrar
  igual las fuentes recuperadas como "Lo más cercano que encontré".
- Estados: cargando (con `aria-busy`), error 422 (pregunta inválida), 429 (esperá un momento), 503/red
  (pasa a modo grabado), 502 (falló la generación).
- Panel "Cómo se evalúa / How it is evaluated": lee las métricas de un `GET /api/metrics` (nuevo
  endpoint que arma un resumen desde el último reporte no mock en `eval/reports/` y desde un
  `eval/reports/retrieval.json`, que tenés que generar con `python -m eval --strict --json` o equivalente y
  versionar). Mostrar recuperación (recall@5, MRR, por idioma, n preguntas), umbral de abstención
  (señal, valor, respondibles perdidas/no respondibles abstenidas, nota de sesgo) y generación. Si no
  hay reporte real de generación, la generación se muestra como **"pendiente: evaluada sólo con modelo
  simulado"** y **no** se muestran las métricas del mock como si fueran reales. Explicación en 3–4 líneas de
  cómo funciona (búsqueda híbrida → umbral → modelo → validación de cada cita contra el texto).

## Modo grabado

- `web/recorded.json`, generado por `python -m web.record` (script reproducible): para los 4 ejemplos
  más 4 preguntas extra (2 respondibles, 2 no respondibles, mitad EN y mitad ES) guarda la pregunta, el idioma y las
  fuentes recuperadas reales (`hybrid_search`, el embedder sólo se carga una vez). Si existe un reporte de
  generación **real** (no mock) que tenga esa pregunta con su respuesta, incluye `abstain`, `sentences` y
  `sources` de ese reporte (si el reporte no guarda las frases, extendé `eval --generate` para que las
  guarde por pregunta: texto, citas n/quote). Si no hay reporte real: `"answer": null` y
  `"status": "retrieval_only"`. Cada entrada lleva `"recorded_at"` y `"source": "eval-report:<archivo>"` o
  `"retrieval-only"`. **Nunca** usar respuestas del modelo simulado.
- Si `/api/ask` no responde (timeout 30 s, error de red) o devuelve 503: buscar la pregunta exacta
  (normalizada) en `recorded.json`; si está, mostrarla con una etiqueta visible "Respuesta grabada
  (sin modelo en vivo)" / "Recorded answer (no live model)"; con `retrieval_only`, mostrar sólo las
  fuentes con el texto "Generación en vivo no disponible: estas son las fuentes que se recuperarían".
  Si no está, ofrecer los ejemplos grabados. Banner discreto de "modo grabado".
- También al cargar la página: `GET /api/health`; si la generación no está disponible, mostrar el banner
  de entrada.

## Servidor

- `app/main.py` monta `web/` en `/` con `StaticFiles(html=True)` **después** de las rutas `/api/*`.
  Cabeceras en las respuestas estáticas y de API: `Content-Security-Policy` (`default-src 'self';
  style-src 'self' https://fonts.googleapis.com; font-src https://fonts.gstatic.com; img-src 'self' data:;
  connect-src 'self'; frame-ancestors 'none'; base-uri 'none'`), `X-Content-Type-Options: nosniff`,
  `Referrer-Policy: no-referrer`. Por la CSP, nada de JS ni CSS inline (atributos `style=` y `on*=` tampoco).
- Sigue escuchando sólo en 127.0.0.1.

## Calidad

- Accesible: navegable con teclado (foco visible con `--signal`), `aria-live="polite"` en la respuesta,
  popovers que se cierran con Esc, etiquetas, contraste AA en ambos temas (verificá con un cálculo
  simple los pares texto/fondo principales y dejalo en el log), `prefers-reduced-motion`.
- Usable a 360 px de ancho sin scroll horizontal (gutter 16 px).
- Textos profesionales; no mencionar clientes ni empresas. Sin emojis.

## Tests

pytest, sin red ni modelo: `GET /` sirve `index.html` (200, `text/html`, CSP presente); `GET /app.js` y
`/app.css`; `GET /api/metrics` con reportes falsos en `tmp_path` (real vs. mock: el mock nunca se
presenta como real); `recorded.json` válido (esquema, sin entradas de mock); las rutas `/api/*` siguen
ganando a los estáticos; 404 de estáticos. Todos en verde junto con los existentes.

## Capturas (evidencia)

Levantá la app con generación off (`GROUNDED_RAG_GENERATION=off`, puerto 8792), y con Playwright de
Python si está en el venv, si no con Node (`require('playwright')` funciona en esta máquina; chromium ya
está instalado, no corras `playwright install`; si hace falta usá `executablePath:
'/opt/pw-browsers/chromium'` o el chromium que encuentres) sacá:
- `patches/opencode/diag/f3-desktop.png` (1366×900, tema oscuro, tras preguntar un ejemplo en inglés);
- `patches/opencode/diag/f3-mobile.png` (360×780, tema claro, `?lang=es`, tras preguntar un ejemplo
  en español, página completa);
- `patches/opencode/diag/f3-abstain.png` (escritorio, una pregunta no respondible grabada).
Registrá en `f3-browser.log` los errores de consola (debe haber 0) y el ancho de scroll a 360 px. Bajá
la app al terminar y verificá que el puerto quedó libre. `free -m` antes de levantar la app (carga el modelo
de embeddings); si "available" < 900 MB, esperá.

## README

Sección corta "Web interface": cómo levantarla, el modo grabado, el endpoint de métricas, y la captura
de escritorio embebida (`patches/opencode/diag/f3-desktop.png`).

## Reglas fijas

- Este brief es tu contrato: no reclasifiques el riesgo ni amplíes el alcance con otras skills.
- No salgas del alcance: no refactorices ni "mejores" nada que no esté pedido.
- No hagas commit, push, deploy ni reinicies servicios salvo que este brief lo diga.
- No toques producción, nginx, systemd ni `/home/capw/projects/dev0p/`; si algo lo requiere, pará y reportalo.
- No leas archivos de credenciales; no hay clave de modelo disponible y no la necesitás.
- No inventes resultados: copiá números y salidas desde el log. Si no corriste algo, decilo.
- Si no podés cumplir un criterio, terminá con BLOQUEADO + qué falta, no con un éxito parcial disfrazado.

## Entregable

Archivos tocados, endpoints nuevos, cómo se genera `recorded.json` y qué contiene, tabla de contraste,
salida de pytest, rutas de las capturas, log del navegador y lo no verificado.
