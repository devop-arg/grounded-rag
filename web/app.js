/* grounded-rag web UI: vanilla JS, no CDN, no inline handlers (CSP-safe). */
(function () {
  'use strict';

  var STORAGE_LANG = 'grounded-rag.lang';
  var STORAGE_THEME = 'grounded-rag.theme';
  var MAX_CHARS = 500;
  var ASK_TIMEOUT_MS = 30000;

  var I18N = {
    en: {
      'skip': 'Skip to the question',
      'subtitle': 'Answers from the PostgreSQL 17 docs, with verifiable citations',
      'aria.lang': 'Language',
      'aria.theme': 'Toggle light or dark theme',
      'repo': 'GitHub',
      'theme.dark': 'Theme: dark',
      'theme.light': 'Theme: light',
      'form.heading': 'Question',
      'form.label': 'Question',
      'form.placeholder': 'Ask about PostgreSQL 17: indexes, vacuum, replication, JSON...',
      'form.hint': 'Enter sends · Shift+Enter adds a line',
      'form.ask': 'Ask',
      'form.examples': 'Examples',
      'counter.aria': 'Characters used',
      'status.searching': 'Searching the documentation',
      'status.offline': 'The live model did not answer',
      'answer.heading': 'Answer',
      'answer.sources': 'Sources',
      'answer.closest': 'Closest match',
      'answer.noSources': 'No source was retrieved for this question.',
      'answer.cite.aria': 'Citation {n}: {title}',
      'answer.source.unknown': 'source not listed',
      'answer.open': 'Open the section on postgresql.org',
      'abstain.text': 'The documentation does not contain enough information to answer this.',
      'error.empty': 'Write a question first.',
      'error.422': 'The question is not valid: it must be between 1 and 500 characters.',
      'error.429': 'Too many questions in a short time. Wait a moment and try again.',
      'error.502': 'The generation step failed. Try again.',
      'error.unknown': 'The request failed. Try again.',
      'banner.recorded': 'Recorded mode: the live model is not available. Answers come from the recorded set.',
      'banner.budget': 'Budget mode: the monthly spending budget of this demo is used up. Answers come from the recorded set.',
      'recorded.badge': 'Recorded answer (no live model)',
      'recorded.retrievalOnly': 'Live generation is unavailable: these are the sources that would be retrieved',
      'recorded.missing': 'There is no recorded answer for this question.',
      'recorded.examples': 'Recorded examples',
      'metrics.heading': 'How it is evaluated',
      'metrics.explain': 'Hybrid search fuses the lexical (BM25) and vector rankings and returns eight documentation pages. If the top similarity is below the calibrated threshold, the server abstains instead of guessing. Otherwise the model answers from those passages only. Every citation is checked against the passage text on the server, and a sentence without a valid citation is dropped.',
      'metrics.unavailable': 'Metrics are unavailable.',
      'metrics.retrieval': 'Retrieval',
      'metrics.recall': 'recall@5',
      'metrics.mrr': 'MRR@8',
      'metrics.questions': 'questions',
      'metrics.answerable': 'answerable',
      'metrics.english': 'English',
      'metrics.spanish': 'Spanish',
      'metrics.threshold': 'Abstention threshold',
      'metrics.signal': 'signal',
      'metrics.value': 'value',
      'metrics.lost': 'answerable lost',
      'metrics.abstained': 'unanswerable abstained',
      'metrics.bias': 'Calibrated on the same 80 questions used for reporting: optimistically biased.',
      'metrics.generation': 'Generation',
      'metrics.pending': 'pending: evaluated only with the simulated model',
      'metrics.pendingNote': 'No real generation report exists on this server, so no generation numbers are shown.',
      'metrics.measured': 'measured',
      'metrics.answered': 'answerable answered',
      'metrics.abstPrecision': 'abstention precision',
      'metrics.abstRecall': 'abstention recall',
      'metrics.citations': 'citations valid',
      'metrics.retrievalMs': 'retrieval median / p95',
      'metrics.lang.en': 'English',
      'metrics.lang.es': 'Spanish',
      'metrics.lang.other': 'other'
    },
    es: {
      'skip': 'Ir a la pregunta',
      'subtitle': 'Respuestas desde la documentación de PostgreSQL 17, con citas verificables',
      'aria.lang': 'Idioma',
      'aria.theme': 'Cambiar entre tema claro y oscuro',
      'repo': 'GitHub',
      'theme.dark': 'Tema: oscuro',
      'theme.light': 'Tema: claro',
      'form.heading': 'Pregunta',
      'form.label': 'Pregunta',
      'form.placeholder': 'Preguntá sobre PostgreSQL 17: índices, vacuum, replicación, JSON...',
      'form.hint': 'Enter envía · Shift+Enter agrega una línea',
      'form.ask': 'Preguntar',
      'form.examples': 'Ejemplos',
      'counter.aria': 'Caracteres usados',
      'status.searching': 'Buscando en la documentación',
      'status.offline': 'El modelo en vivo no respondió',
      'answer.heading': 'Respuesta',
      'answer.sources': 'Fuentes',
      'answer.closest': 'Lo más cercano que encontré',
      'answer.noSources': 'No se recuperó ninguna fuente para esta pregunta.',
      'answer.cite.aria': 'Cita {n}: {title}',
      'answer.source.unknown': 'fuente no listada',
      'answer.open': 'Abrir la sección en postgresql.org',
      'abstain.text': 'No tengo información suficiente en la documentación para responder esto.',
      'error.empty': 'Escribí una pregunta primero.',
      'error.422': 'La pregunta no es válida: debe tener entre 1 y 500 caracteres.',
      'error.429': 'Demasiadas preguntas en poco tiempo. Esperá un momento y volvé a intentar.',
      'error.502': 'Falló la generación. Volvé a intentar.',
      'error.unknown': 'Falló la solicitud. Volvé a intentar.',
      'banner.recorded': 'Modo grabado: el modelo en vivo no está disponible. Las respuestas salen del conjunto grabado.',
      'banner.budget': 'Modo presupuesto: el presupuesto mensual de esta demo está agotado. Las respuestas salen del conjunto grabado.',
      'recorded.badge': 'Respuesta grabada (sin modelo en vivo)',
      'recorded.retrievalOnly': 'Generación en vivo no disponible: estas son las fuentes que se recuperarían',
      'recorded.missing': 'No hay una respuesta grabada para esta pregunta.',
      'recorded.examples': 'Ejemplos grabados',
      'metrics.heading': 'Cómo se evalúa',
      'metrics.explain': 'La búsqueda híbrida fusiona el ranking léxico (BM25) y el vectorial y devuelve ocho páginas de la documentación. Si la similitud máxima queda por debajo del umbral calibrado, el servidor se abstiene en lugar de adivinar. Si no, el modelo responde sólo con esos pasajes. Cada cita se valida contra el texto del pasaje en el servidor, y la frase sin una cita válida se descarta.',
      'metrics.unavailable': 'Las métricas no están disponibles.',
      'metrics.retrieval': 'Recuperación',
      'metrics.recall': 'recall@5',
      'metrics.mrr': 'MRR@8',
      'metrics.questions': 'preguntas',
      'metrics.answerable': 'respondibles',
      'metrics.english': 'Inglés',
      'metrics.spanish': 'Español',
      'metrics.threshold': 'Umbral de abstención',
      'metrics.signal': 'señal',
      'metrics.value': 'valor',
      'metrics.lost': 'respondibles perdidas',
      'metrics.abstained': 'no respondibles abstenidas',
      'metrics.bias': 'Calibrado sobre las mismas 80 preguntas usadas para reportar: sesgo optimista.',
      'metrics.generation': 'Generación',
      'metrics.pending': 'pendiente: evaluada sólo con modelo simulado',
      'metrics.pendingNote': 'No hay un reporte real de generación en este servidor, así que no se muestran números de generación.',
      'metrics.measured': 'medida',
      'metrics.answered': 'respondibles respondidas',
      'metrics.abstPrecision': 'precisión de abstención',
      'metrics.abstRecall': 'cobertura de abstención',
      'metrics.citations': 'citas válidas',
      'metrics.retrievalMs': 'recuperación mediana / p95',
      'metrics.lang.en': 'Inglés',
      'metrics.lang.es': 'Español',
      'metrics.lang.other': 'otro'
    }
  };

  var state = {
    lang: 'en',
    theme: null,
    last: null,
    recorded: null,
    metrics: null,
    banner: null,
    budgetExhausted: false
  };
  var popSeq = 0;

  // ---------------------------------------------------------------- helpers
  function t(key, params) {
    var table = I18N[state.lang] || I18N.en;
    var text = table[key];
    if (typeof text !== 'string') text = I18N.en[key];
    if (typeof text !== 'string') text = key;
    if (params) {
      Object.keys(params).forEach(function (name) {
        text = text.split('{' + name + '}').join(String(params[name]));
      });
    }
    return text;
  }

  function el(tag, attrs) {
    var node = document.createElement(tag);
    var children = Array.prototype.slice.call(arguments, 2);
    if (attrs) {
      Object.keys(attrs).forEach(function (key) {
        var value = attrs[key];
        if (value === null || value === undefined || value === false) return;
        if (key === 'class') node.className = value;
        else node.setAttribute(key, value);
      });
    }
    children.forEach(function add(child) {
      if (child === null || child === undefined || child === false) return;
      if (Array.isArray(child)) { child.forEach(add); return; }
      node.appendChild(typeof child === 'string' ? document.createTextNode(child) : child);
    });
    return node;
  }

  function readStore(key) {
    try { return window.localStorage.getItem(key); } catch (error) { return null; }
  }
  function writeStore(key, value) {
    try { window.localStorage.setItem(key, value); } catch (error) { /* private mode */ }
  }

  function number(value) {
    if (value === null || value === undefined) return '-';
    if (typeof value !== 'number') return String(value);
    if (Number.isInteger(value)) return String(value);
    return String(Math.round(value * 10000) / 10000);
  }

  var QUOTES = {
    '\u2018': "'", '\u2019': "'", '\u201a': "'", '\u201b': "'", '\u2032': "'", '\u2035': "'",
    '\u201c': '"', '\u201d': '"', '\u201e': '"', '\u201f': '"', '\u2033': '"',
    '\u00ab': '"', '\u00bb': '"', '\u300c': '"', '\u300d': '"'
  };
  function normalizeQuestion(text) {
    var value = String(text).normalize('NFKC').replace(
      /[\u2018\u2019\u201a\u201b\u2032\u2035\u201c\u201d\u201e\u201f\u2033\u00ab\u00bb\u300c\u300d]/g,
      function (character) { return QUOTES[character] || character; }
    );
    return value.replace(/\s+/g, ' ').trim().toLowerCase();
  }

  // --------------------------------------------------------------- language
  function applyLanguage() {
    document.documentElement.lang = state.lang;
    Array.prototype.forEach.call(document.querySelectorAll('[data-i18n]'), function (node) {
      var key = node.getAttribute('data-i18n');
      var text = t(key);
      var attribute = node.getAttribute('data-i18n-attr');
      if (attribute) node.setAttribute(attribute, text);
      else node.textContent = text;
    });
    Array.prototype.forEach.call(document.querySelectorAll('[data-lang-set]'), function (button) {
      button.setAttribute('aria-pressed', String(button.getAttribute('data-lang-set') === state.lang));
    });
    updateThemeLabel();
    renderCounter();
    render();
    renderMetrics();
  }

  function setLanguage(next, persist) {
    state.lang = next === 'es' ? 'es' : 'en';
    if (persist) writeStore(STORAGE_LANG, state.lang);
    applyLanguage();
  }

  // ------------------------------------------------------------------ theme
  function prefersLight() {
    return !!(window.matchMedia && window.matchMedia('(prefers-color-scheme: light)').matches);
  }
  function effectiveTheme() {
    return state.theme || (prefersLight() ? 'light' : 'dark');
  }
  function applyTheme() {
    if (state.theme) document.documentElement.setAttribute('data-theme', state.theme);
    else document.documentElement.removeAttribute('data-theme');
    updateThemeLabel();
  }
  function updateThemeLabel() {
    var label = document.getElementById('theme-label');
    if (label) label.textContent = effectiveTheme() === 'light' ? t('theme.light') : t('theme.dark');
    var toggle = document.getElementById('theme-toggle');
    if (toggle) toggle.setAttribute('aria-label', t('aria.theme'));
  }

  // ------------------------------------------------------------------- form
  function textarea() { return document.getElementById('question'); }

  function renderCounter() {
    var field = textarea();
    var counter = document.getElementById('counter');
    if (!field || !counter) return;
    counter.textContent = String(field.value.length) + '/' + MAX_CHARS;
    counter.classList.toggle('over', field.value.length > MAX_CHARS);
    counter.setAttribute('aria-label', t('counter.aria'));
  }

  function setBusy(busy) {
    var result = document.getElementById('result');
    var button = document.getElementById('ask-button');
    if (result) result.setAttribute('aria-busy', busy ? 'true' : 'false');
    if (button) button.disabled = busy;
  }

  function submitQuestion(question) {
    var field = textarea();
    if (typeof question === 'string' && field) {
      field.value = question;
      renderCounter();
    }
    ask(field ? field.value : '');
  }

  // ------------------------------------------------------------------- ask
  function fetchWithTimeout(url, options, timeoutMs) {
    var controller = new AbortController();
    var timer = window.setTimeout(function () { controller.abort(); }, timeoutMs);
    var settings = Object.assign({}, options || {}, { signal: controller.signal });
    return window.fetch(url, settings).finally(function () { window.clearTimeout(timer); });
  }

  function ask(rawQuestion) {
    var question = String(rawQuestion || '').trim();
    if (!question) {
      state.last = { kind: 'error', message: t('error.empty') };
      render();
      return Promise.resolve();
    }
    if (question.length > MAX_CHARS) question = question.slice(0, MAX_CHARS);

    setBusy(true);
    state.last = { kind: 'loading' };
    render();

    if (state.budgetExhausted) {
      // The monthly budget is already known to be spent: no request is wasted.
      return recordedMode(question).then(function () { setBusy(false); });
    }

    return fetchWithTimeout(
      '/api/ask',
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: question })
      },
      ASK_TIMEOUT_MS
    ).then(function (response) {
      if (response.status === 503) return onUnavailable(response, question);
      if (response.status === 429) { state.last = { kind: 'error', message: t('error.429') }; render(); return; }
      if (response.status === 422) { state.last = { kind: 'error', message: t('error.422') }; render(); return; }
      if (response.status === 502) { state.last = { kind: 'error', message: t('error.502') }; render(); return; }
      if (!response.ok) throw new Error('HTTP ' + response.status);
      return response.json().then(function (answer) {
        if (answer && answer.reason === 'generation_disabled') return recordedMode(question);
        state.last = { kind: 'answer', answer: answer };
        render();
      });
    }).catch(function () {
      return recordedMode(question);
    }).then(function () {
      setBusy(false);
    });
  }

  // -------------------------------------------------------------- recorded
  function loadRecorded() {
    if (state.recorded) return Promise.resolve(state.recorded);
    return window.fetch('/recorded.json').then(function (response) {
      if (!response.ok) throw new Error('HTTP ' + response.status);
      return response.json();
    }).then(function (document_) {
      var entries = document_ && Array.isArray(document_.entries) ? document_.entries : [];
      state.recorded = entries.filter(function (entry) { return entry && typeof entry.question === 'string'; });
      return state.recorded;
    }).catch(function () {
      state.recorded = [];
      return state.recorded;
    });
  }

  function recordedMode(question) {
    showBanner(state.budgetExhausted ? t('banner.budget') : t('banner.recorded'));
    return loadRecorded().then(function (entries) {
      var key = normalizeQuestion(question);
      var match = entries.filter(function (entry) { return normalizeQuestion(entry.question) === key; })[0];
      if (match) {
        state.last = { kind: 'recorded', entry: match };
      } else {
        state.last = { kind: 'recorded-missing', question: question, examples: entries.slice(0, 4) };
      }
      render();
    });
  }

  // A 503 is "no live model": either the monthly budget is spent (the server
  // says so with reason=budget_exhausted) or generation is unavailable.
  function onUnavailable(response, question) {
    return response.json().then(function (body) {
      if (body && body.reason === 'budget_exhausted') state.budgetExhausted = true;
      return recordedMode(question);
    }).catch(function () {
      return recordedMode(question);
    });
  }

  function showBanner(text) {
    state.banner = text;
    var banner = document.getElementById('banner');
    var body = document.getElementById('banner-text');
    if (!banner || !body) return;
    body.textContent = text;
    banner.hidden = false;
  }

  // ---------------------------------------------------------------- render
  function result() { return document.getElementById('result'); }

  function clear(node) {
    while (node && node.firstChild) node.removeChild(node.firstChild);
  }

  function cardHead(title, badgeText) {
    var head = el('div', { class: 'card-head' }, el('h2', null, title));
    if (badgeText) head.appendChild(el('span', { class: 'badge' }, badgeText));
    return head;
  }

  function sourceItem(source) {
    var title = el('a', {
      class: 'source-title',
      href: source.url,
      target: '_blank',
      rel: 'noopener'
    }, source.title || source.url);
    var body = el('div', { class: 'source-body' }, title);
    if (source.title) body.appendChild(el('span', { class: 'source-url' }, source.url));
    return el('li', null, el('span', { class: 'source-n' }, String(source.n)), body);
  }

  function sourcesBlock(heading, sources) {
    if (!sources || !sources.length) return null;
    var list = el('ol');
    sources.forEach(function (source) { list.appendChild(sourceItem(source)); });
    return el('section', { class: 'sources', 'aria-label': heading },
      el('h3', null, heading), list);
  }

  function closePopovers(except) {
    Array.prototype.forEach.call(document.querySelectorAll('.cite-wrap.is-open'), function (wrap) {
      if (wrap === except) return;
      wrap.classList.remove('is-open');
      var button = wrap.querySelector('.cite');
      if (button) button.setAttribute('aria-expanded', 'false');
    });
  }

  function citationNode(citation, byNumber) {
    var source = byNumber[citation.n];
    var id = 'cite-pop-' + (++popSeq);
    var button = el('button', {
      type: 'button',
      class: 'cite',
      'aria-expanded': 'false',
      'aria-describedby': id,
      'aria-label': t('answer.cite.aria', {
        n: citation.n,
        title: source && source.title ? source.title : t('answer.source.unknown')
      })
    }, '[' + citation.n + ']');

    var popover = el('span', { class: 'popover', id: id, role: 'tooltip' },
      el('span', { class: 'quote' }, citation.quote || ''),
      el('span', { class: 'pop-title' }, source ? (source.title || source.url) : t('answer.source.unknown'))
    );
    if (source && source.url) {
      popover.appendChild(el('a', {
        class: 'pop-link',
        href: source.url,
        target: '_blank',
        rel: 'noopener',
        title: t('answer.open')
      }, source.url));
    }

    button.addEventListener('click', function () {
      var wrap = button.parentNode;
      var open = !wrap.classList.contains('is-open');
      closePopovers(wrap);
      wrap.classList.toggle('is-open', open);
      button.setAttribute('aria-expanded', String(open));
    });

    return el('span', { class: 'cite-wrap' }, button, popover);
  }

  function sentencesBlock(sentences, sources) {
    var byNumber = {};
    (sources || []).forEach(function (source) { byNumber[source.n] = source; });
    var wrapper = el('div', { class: 'sentences' });
    (sentences || []).forEach(function (sentence) {
      var paragraph = el('p', { class: 'sentence' }, sentence.text || '');
      var cites = el('span', { class: 'cites' });
      (sentence.citations || []).forEach(function (citation) {
        cites.appendChild(document.createTextNode(' '));
        cites.appendChild(citationNode(citation, byNumber));
      });
      paragraph.appendChild(cites);
      wrapper.appendChild(paragraph);
    });
    return wrapper;
  }

  function renderAnswer(answer, badgeText) {
    var resultNode = result();
    if (!resultNode) return;
    clear(resultNode);
    resultNode.hidden = false;
    resultNode.className = 'result' + (answer.abstain ? ' abstain' : '');

    var card = el('div', { class: 'card' }, cardHead(t('answer.heading'), badgeText));
    if (answer.abstain) {
      card.appendChild(el('p', { class: 'abstain-text' }, t('abstain.text')));
      var closest = sourcesBlock(t('answer.closest'), answer.sources);
      if (closest) card.appendChild(closest);
    } else {
      card.appendChild(sentencesBlock(answer.sentences, answer.sources));
      var list = sourcesBlock(t('answer.sources'), answer.sources);
      if (list) card.appendChild(list);
    }
    resultNode.appendChild(card);
  }

  function renderRecorded(entry) {
    if (entry.answer && Array.isArray(entry.answer.sentences)) {
      renderAnswer(
        { abstain: !!entry.answer.abstain, sentences: entry.answer.sentences, sources: entry.answer.sources || entry.sources || [] },
        t('recorded.badge')
      );
      return;
    }
    var resultNode = result();
    if (!resultNode) return;
    clear(resultNode);
    resultNode.hidden = false;
    resultNode.className = 'result abstain';
    var card = el('div', { class: 'card' }, cardHead(t('answer.heading'), t('recorded.badge')));
    card.appendChild(el('p', { class: 'abstain-text' }, t('recorded.retrievalOnly')));
    var list = sourcesBlock(t('answer.closest'), entry.sources);
    if (list) card.appendChild(list);
    else card.appendChild(el('p', { class: 'muted mono' }, t('answer.noSources')));
    resultNode.appendChild(card);
  }

  function exampleButton(question) {
    var button = el('button', { type: 'button', class: 'chip' }, question);
    button.addEventListener('click', function () { submitQuestion(question); });
    return button;
  }

  function renderRecordedMissing(question, examples) {
    var resultNode = result();
    if (!resultNode) return;
    clear(resultNode);
    resultNode.hidden = false;
    resultNode.className = 'result error';
    var card = el('div', { class: 'card' }, cardHead(t('answer.heading'), t('recorded.badge')));
    card.appendChild(el('p', { class: 'abstain-text' }, t('recorded.missing')));
    card.appendChild(el('p', { class: 'tag' }, t('recorded.examples')));
    var chips = el('div', { class: 'chips' });
    (examples || []).forEach(function (entry) { chips.appendChild(exampleButton(entry.question)); });
    card.appendChild(chips);
    resultNode.appendChild(card);
  }

  function renderError(message) {
    var resultNode = result();
    if (!resultNode) return;
    clear(resultNode);
    resultNode.hidden = false;
    resultNode.className = 'result error';
    resultNode.appendChild(el('div', { class: 'card' },
      el('p', { class: 'abstain-text' }, message)));
  }

  function renderLoading() {
    var resultNode = result();
    if (!resultNode) return;
    clear(resultNode);
    resultNode.hidden = false;
    resultNode.className = 'result';
    resultNode.appendChild(el('div', { class: 'card' },
      el('div', { class: 'status' },
        el('span', { class: 'dot', 'aria-hidden': 'true' }),
        el('span', null, t('status.searching')))));
  }

  function render() {
    var last = state.last;
    if (!last) { var empty = result(); if (empty) { clear(empty); empty.hidden = true; } return; }
    if (last.kind === 'loading') renderLoading();
    else if (last.kind === 'answer') renderAnswer(last.answer, null);
    else if (last.kind === 'recorded') renderRecorded(last.entry);
    else if (last.kind === 'recorded-missing') renderRecordedMissing(last.question, last.examples);
    else if (last.kind === 'error') renderError(last.message);
  }

  // --------------------------------------------------------------- metrics
  function metricRow(list, term, value) {
    list.appendChild(el('dt', null, term));
    list.appendChild(el('dd', null, value));
  }

  function metricBlock(title, rows, notes, pending) {
    var dl = el('dl');
    rows.forEach(function (row) { metricRow(dl, row[0], row[1]); });
    var block = el('div', { class: 'metric' + (pending ? ' pending' : '') }, el('h3', null, title), dl);
    (notes || []).forEach(function (note) {
      if (note) block.appendChild(el('p', { class: 'note' }, note));
    });
    return block;
  }

  function langLabel(lang) {
    var key = 'metrics.lang.' + lang;
    var label = t(key);
    return label === key ? String(lang).toUpperCase() : label;
  }

  function renderMetrics() {
    var body = document.getElementById('metrics-body');
    if (!body) return;
    clear(body);
    var data = state.metrics;

    if (!data || !data.retrieval) {
      body.appendChild(el('p', { class: 'mono muted' }, t('metrics.unavailable')));
      if (!data) return;
    }

    if (data && data.retrieval) {
      var retrieval = data.retrieval;
      var rows = [
        [t('metrics.recall'), number(retrieval['recall@5'])],
        [t('metrics.mrr'), number(retrieval['mrr@8'])],
        [t('metrics.questions'), number(retrieval.questions)],
        [t('metrics.answerable'), number(retrieval.answerable)]
      ];
      var byLang = retrieval.by_lang || {};
      Object.keys(byLang).sort().forEach(function (lang) {
        var block = byLang[lang] || {};
        rows.push([langLabel(lang) + ' · ' + t('metrics.recall'), number(block['recall@5'])]);
        rows.push([langLabel(lang) + ' · ' + t('metrics.mrr'), number(block['mrr@8'])]);
      });
      body.appendChild(metricBlock(t('metrics.retrieval'), rows, [], false));
    }

    if (data && data.threshold) {
      var threshold = data.threshold;
      var thresholdRows = [
        [t('metrics.signal'), String(threshold.signal || '-')],
        [t('metrics.value'), number(threshold.value)],
        [t('metrics.lost'), number(threshold.answerable_lost) + '/' + number(threshold.answerable_total)],
        [t('metrics.abstained'), number(threshold.unanswerable_abstained) + '/' + number(threshold.unanswerable_total)]
      ];
      body.appendChild(metricBlock(t('metrics.threshold'), thresholdRows, [threshold.note ? t('metrics.bias') : null], false));
    }

    if (data && data.generation) {
      var generation = data.generation;
      if (generation.status === 'measured') {
        var generationRows = [
          [t('metrics.answered'), number(generation.answered) + '/' + number(generation.answerable)],
          [t('metrics.abstPrecision'), number(generation.abstention && generation.abstention.precision)],
          [t('metrics.abstRecall'), number(generation.abstention && generation.abstention.recall)],
          [t('metrics.citations'), number(generation.citations && generation.citations.valid) + '/' + number(generation.citations && generation.citations.total)],
          [t('metrics.retrievalMs'), number(generation.retrieval_ms && generation.retrieval_ms.median) + ' / ' + number(generation.retrieval_ms && generation.retrieval_ms.p95) + ' ms']
        ];
        body.appendChild(metricBlock(t('metrics.generation'), generationRows, [generation.report], false));
      } else {
        body.appendChild(metricBlock(t('metrics.generation'), [], [t('metrics.pending'), t('metrics.pendingNote')], true));
      }
    }
  }

  function loadMetrics() {
    return window.fetch('/api/metrics').then(function (response) {
      if (!response.ok) throw new Error('HTTP ' + response.status);
      return response.json();
    }).then(function (data) {
      state.metrics = data;
      renderMetrics();
    }).catch(function () {
      state.metrics = null;
      renderMetrics();
    });
  }

  // ------------------------------------------------------------------ init
  function init() {
    var params = new URLSearchParams(window.location.search);
    var fromUrl = params.get('lang');
    var storedLang = readStore(STORAGE_LANG);
    var storedTheme = readStore(STORAGE_THEME);
    if (storedTheme === 'light' || storedTheme === 'dark') state.theme = storedTheme;

    var language = 'en';
    if (fromUrl === 'es' || fromUrl === 'en') language = fromUrl;
    else if (storedLang === 'es' || storedLang === 'en') language = storedLang;
    state.lang = language;
    if (fromUrl === 'es' || fromUrl === 'en') writeStore(STORAGE_LANG, fromUrl);

    applyTheme();
    applyLanguage();

    Array.prototype.forEach.call(document.querySelectorAll('[data-lang-set]'), function (button) {
      button.addEventListener('click', function () { setLanguage(button.getAttribute('data-lang-set'), true); });
    });

    var toggle = document.getElementById('theme-toggle');
    if (toggle) {
      toggle.addEventListener('click', function () {
        state.theme = effectiveTheme() === 'light' ? 'dark' : 'light';
        writeStore(STORAGE_THEME, state.theme);
        applyTheme();
      });
    }
    if (window.matchMedia) {
      var query = window.matchMedia('(prefers-color-scheme: light)');
      var onChange = function () { if (!state.theme) applyTheme(); };
      if (query.addEventListener) query.addEventListener('change', onChange);
      else if (query.addListener) query.addListener(onChange);
    }

    var field = textarea();
    if (field) {
      field.addEventListener('input', renderCounter);
      field.addEventListener('keydown', function (event) {
        if (event.key === 'Enter' && !event.shiftKey) {
          event.preventDefault();
          submitQuestion();
        }
      });
    }

    var form = document.getElementById('ask-form');
    if (form) {
      form.addEventListener('submit', function (event) {
        event.preventDefault();
        submitQuestion();
      });
    }

    var askButton = document.getElementById('ask-button');
    if (askButton) {
      askButton.addEventListener('click', function (event) { event.preventDefault(); submitQuestion(); });
    }

    Array.prototype.forEach.call(document.querySelectorAll('[data-example]'), function (button) {
      button.addEventListener('click', function () { submitQuestion(button.getAttribute('data-example')); });
    });

    document.addEventListener('keydown', function (event) {
      if (event.key !== 'Escape') return;
      var open = document.querySelector('.cite-wrap.is-open');
      closePopovers(null);
      if (open) {
        var button = open.querySelector('.cite');
        if (button) button.focus();
      }
    });
    document.addEventListener('click', function (event) {
      if (!event.target.closest || !event.target.closest('.cite-wrap')) closePopovers(null);
    });

    window.fetch('/api/health').then(function (response) {
      return response.ok ? response.json() : null;
    }).then(function (health) {
      if (!health) return;
      if (!health.generation || !health.llm_configured) {
        showBanner(t('banner.recorded'));
        return;
      }
      if (health.budget && health.budget.exhausted) {
        // Spent for the month: recorded mode from the first question on.
        state.budgetExhausted = true;
        showBanner(t('banner.budget'));
      }
    }).catch(function () { /* static page: no banner */ });

    loadRecorded();
    loadMetrics();
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
}());
