"""Web interface: static files behind the API, security headers, metrics, recording.

No network and no model: every report is a fake written to ``tmp_path``.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from app.metrics import build_metrics, generation_metrics, retrieval_metrics
from app.main import CONTENT_SECURITY_POLICY
from web.record import (
    build_entry,
    example_questions,
    latest_real_report,
    question_rows,
    record,
    report_answers,
)

from helpers import FakeEmbedder, FakeStore, PARTIAL_INDEX_TEXT, ScriptedLLM, make_client, make_settings

REPO_ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = REPO_ROOT / "web"
CSP_DIRECTIVES = ("default-src 'self'", "frame-ancestors 'none'", "base-uri 'none'", "connect-src 'self'")
EXAMPLE_IDS = (
    "a01",
    "x01-es",
    "a15",
    "x02-es",
    "x03-en",
    "a16",
    "x04-en",
    "a06",
    "a33",
    "a42",
    "u01",
    "u30",
)
GOOD_ANSWER = {
    "abstain": False,
    "sentences": [
        {"text": "It restricts the rows.", "citations": [{"n": 1, "quote": "A partial index is an index built over a subset of a table"}]}
    ],
}
ABSTAIN_ANSWER = {"abstain": True, "sentences": []}


# --------------------------------------------------------------------- fakes
def retrieval_report() -> dict:
    rows = [
        {"id": "a01", "lang": "en", "answerable": True, "excluded": False, "hit_rank": 1},
        {"id": "a02", "lang": "es", "answerable": True, "excluded": False, "hit_rank": 3},
        {"id": "u01", "lang": "en", "answerable": False, "excluded": False, "hit_rank": None},
    ]
    return {
        "summary": {
            "policy": "downweight",
            "questions": 3,
            "answerable": 2,
            "unanswerable": 1,
            "excluded": 0,
            "recall@5": 1.0,
            "mrr@8": 0.6667,
            "seconds": 0.1,
            "mean_ms_per_question": 1.0,
        },
        "rows": rows,
    }


def generation_report(
    answered: int, precision: float, *, valid: int = 60, total: int = 61, generated_at: str = "2026-10-29T00:00:00Z"
) -> dict:
    return {
        "generated_at": generated_at,
        "threshold": None,
        "config": {"llm_api_key": True},
        "summary": {
            "questions": 80,
            "answerable": 50,
            "answered": answered,
            "abstention": {"precision": precision, "recall": 0.5},
            "citations": {"valid": valid, "total": total, "validity_rate": round(valid / total, 4)},
            "latency_ms": {"retrieval": {"median": 84.2, "p95": 128.6}, "llm": {"median": 1.0, "p95": 2.0}},
        },
    }


def compare_report() -> dict:
    """A ``--compare-remote`` report where the chosen combination is not the best recall."""

    def block(n: int, recall: float, mrr: float) -> dict:
        return {"n": n, "recall@5": recall, "mrr@8": mrr, "hits@5": int(round(n * recall))}

    def combination(name: str, status: str, all_block: dict, en: dict, es: dict, models=None) -> dict:
        return {
            "name": name,
            "models": models or {"embeddings": "", "rerank": "", "translate": "", "index": ""},
            "status": status,
            "summary": {"all": all_block, "en": en, "es": es},
            "latency_ms": {"median": 80.5, "p95": 115.0},
            "remote": {"calls": 0, "errors": 0, "fallbacks": 0},
            "cost_usd": 0.0,
        }

    remote_models = {"embeddings": "vendor/embed", "rerank": "", "translate": "", "index": "vectors-vendor"}
    return {
        "generated_at": "2026-10-29T14:14:22Z",
        "questions": 80,
        "answerable": 50,
        "policy": "downweight",
        "k": 8,
        "models": {"rerank": "", "embeddings": ["vendor/embed"], "translate": []},
        "combinations": [
            combination(
                "local base",
                "ok",
                block(50, 0.88, 0.7336),
                block(25, 0.92, 0.8048),
                block(25, 0.84, 0.6624),
            ),
            # Better recall@5 than the chosen one: it must not be the row shown.
            combination(
                "+embeddings vendor/embed",
                "ok",
                block(50, 0.98, 0.8093),
                block(25, 0.99, 0.85),
                block(25, 0.97, 0.76),
                remote_models,
            ),
            # A failed combination is never read, whatever its numbers say.
            combination(
                "+rerank",
                "unavailable (every remote call failed)",
                block(50, 0.99, 0.9),
                block(25, 0.99, 0.9),
                block(25, 0.99, 0.9),
            ),
            combination(
                "best combination",
                "ok",
                block(50, 0.96, 0.8563),
                block(25, 0.96, 0.8613),
                block(25, 0.96, 0.8513),
                {**remote_models, "translate": "vendor/translate"},
            ),
        ],
        "chosen": "best combination",
        "chosen_models": {**remote_models, "translate": "vendor/translate"},
        "reason": "best combination has the best MRR@8 among the combinations that do not lower recall@5",
        "cost_usd": 0.0,
    }


THRESHOLD = {
    "signal": "vector_cosine",
    "value": 0.820859,
    "answerable_lost": 2,
    "answerable_total": 50,
    "unanswerable_abstained": 10,
    "unanswerable_total": 30,
    "note": "calibrated on the same eval set",
}


@pytest.fixture
def reports(tmp_path: Path) -> Path:
    directory = tmp_path / "reports"
    directory.mkdir()
    (directory / "retrieval.json").write_text(json.dumps(retrieval_report()), encoding="utf-8")
    (tmp_path / "threshold.json").write_text(json.dumps(THRESHOLD), encoding="utf-8")
    return directory


# ------------------------------------------------------------------- statics
def test_root_serves_the_page_with_the_security_headers(tmp_path) -> None:
    client, _ = make_client(tmp_path)
    with client:
        response = client.get("/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "grounded-rag" in response.text
    assert response.headers["content-security-policy"] == CONTENT_SECURITY_POLICY
    for directive in CSP_DIRECTIVES:
        assert directive in response.headers["content-security-policy"]
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["referrer-policy"] == "no-referrer"


def test_static_assets_are_served(tmp_path) -> None:
    client, _ = make_client(tmp_path)
    with client:
        script = client.get("/app.js")
        styles = client.get("/app.css")
        recorded = client.get("/recorded.json")
    assert script.status_code == 200 and "javascript" in script.headers["content-type"]
    assert styles.status_code == 200 and styles.headers["content-type"].startswith("text/css")
    assert recorded.status_code == 200 and recorded.headers["content-type"].startswith("application/json")


def test_api_routes_win_over_the_static_mount(tmp_path) -> None:
    client, _ = make_client(tmp_path)
    with client:
        health = client.get("/api/health")
        metrics = client.get("/api/metrics")
    assert health.status_code == 200
    assert health.headers["content-type"].startswith("application/json")
    assert health.headers["content-security-policy"] == CONTENT_SECURITY_POLICY
    assert metrics.status_code == 200
    assert metrics.headers["content-type"].startswith("application/json")


def test_missing_static_file_is_404(tmp_path) -> None:
    client, _ = make_client(tmp_path)
    with client:
        assert client.get("/nope.html").status_code == 404
        assert client.get("/nope/deeper.js").status_code == 404


def test_the_recorder_module_is_not_served(tmp_path) -> None:
    client, _ = make_client(tmp_path)
    with client:
        assert client.get("/record.py").status_code == 404
        assert client.get("/__pycache__/record.cpython-312.pyc").status_code == 404


def test_the_assets_are_csp_clean() -> None:
    """CSP forbids inline script, inline style and ``on*`` handlers."""
    pages = sorted(WEB_DIR.glob("*.html"))
    assert pages, "the page is served from web/"
    for page_path in pages:
        page = page_path.read_text(encoding="utf-8")
        assert "<style" not in page.lower()
        assert "<script>" not in page.lower()
        assert not re.search(r"\sstyle\s*=", page)
        assert not re.search(r"\son[a-z]+\s*=", page)
        assert not re.search(r"<svg[^>]*\sstyle\s*=", page)
    script = (WEB_DIR / "app.js").read_text(encoding="utf-8")
    assert ".style." not in script and "setAttribute('style'" not in script
    assert "unsafe-inline" not in CONTENT_SECURITY_POLICY
    assert "unsafe-eval" not in CONTENT_SECURITY_POLICY
    assert "script-src" not in CONTENT_SECURITY_POLICY  # default-src 'self' covers scripts


def _i18n_keys(script: str, lang: str) -> set[str]:
    start = script.index(f"    {lang}: {{")
    end = script.index("\n    }", start)
    return set(re.findall(r"'([A-Za-z0-9_.]+)':", script[start:end]))


def test_every_i18n_key_used_by_the_page_exists_in_both_languages() -> None:
    page = (WEB_DIR / "index.html").read_text(encoding="utf-8")
    script = (WEB_DIR / "app.js").read_text(encoding="utf-8")
    used = set(re.findall(r'data-i18n="([^"]+)"', page))
    english = _i18n_keys(script, "en")
    spanish = _i18n_keys(script, "es")
    assert used, "the page drives its texts from data-i18n"
    assert used <= english, sorted(used - english)
    assert used <= spanish, sorted(used - spanish)
    assert english == spanish, sorted(english ^ spanish)


# ------------------------------------------------------- initial language
def _i18n_value(script: str, lang: str, key: str) -> str | None:
    start = script.index(f"    {lang}: {{")
    end = script.index("\n    }", start)
    match = re.search(rf"'{re.escape(key)}': '((?:[^'\\]|\\.)*)'", script[start:end])
    return match.group(1) if match else None


def _js_language_block() -> str:
    """The pure picker of ``app.js`` with the real storage keys, ready for node."""
    script = (WEB_DIR / "app.js").read_text(encoding="utf-8")
    constants = re.findall(r"var STORAGE_LANG(?:_SITE)? = '[^']+';", script)
    assert constants == ["var STORAGE_LANG = 'grounded-rag.lang';", "var STORAGE_LANG_SITE = 'dev0p-lang';"]
    start = script.index("// --- initial language")
    end = script.index("// --- end initial language") + len("// --- end initial language")
    return "\n".join(constants) + "\n" + script[start:end] + "\n"


# The picker runs in node, without dependencies: the eight cases below are the
# whole rule, including storage that throws on every access.
NODE_PICKER_HARNESS = """
const fs = require('fs');
const cases = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
function storageFrom(spec) {
  if (spec === null) return null;
  if (spec === 'throws') return { getItem: function () { throw new Error('blocked'); } };
  return {
    getItem: function (key) {
      if (spec.throws && Object.prototype.hasOwnProperty.call(spec.throws, key)) throw new Error('blocked');
      return Object.prototype.hasOwnProperty.call(spec.items, key) ? spec.items[key] : null;
    }
  };
}
const picked = cases.map(function (item) { return pickInitialLanguage(item.search, storageFrom(item.storage)); });
console.log(JSON.stringify(picked));
"""


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_the_initial_language_is_the_url_then_our_key_then_the_site_default(tmp_path: Path) -> None:
    cases = [
        # (search, storage, expected)
        ("?lang=es", {"items": {"grounded-rag.lang": "en"}}, "es"),  # the URL wins
        ("?lang=en", {"items": {"grounded-rag.lang": "es", "dev0p-lang": "es"}}, "en"),
        ("?lang=fr", {"items": {"grounded-rag.lang": "es"}}, "es"),  # an unknown value is ignored
        ("?lang=ES", {}, "en"),  # the value is matched exactly
        ("?lang=", {"items": {"dev0p-lang": "es"}}, "es"),
        ("", {"items": {"grounded-rag.lang": "es", "dev0p-lang": "en"}}, "es"),  # our key wins
        ("?q=indices", {"items": {"dev0p-lang": "es"}}, "es"),  # inherited from dev0p.com
        ("", {"items": {"dev0p-lang": "fr"}}, "en"),  # an unknown inherited value is ignored
        ("", {"items": {}}, "en"),  # the default
        ("", None, "en"),  # no storage at all
        ("?lang=es", "throws", "es"),  # storage blocked, the URL still decides
        ("", "throws", "en"),  # storage blocked: the default
        ("", {"items": {"dev0p-lang": "es"}, "throws": {"grounded-rag.lang": True}}, "es"),
    ]
    harness = tmp_path / "harness.js"
    harness.write_text(_js_language_block() + NODE_PICKER_HARNESS, encoding="utf-8")
    payload = [{"search": search, "storage": storage} for search, storage, _ in cases]
    cases_path = tmp_path / "cases.json"
    cases_path.write_text(json.dumps(payload), encoding="utf-8")
    done = subprocess.run(
        [shutil.which("node"), str(harness), str(cases_path)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert done.returncode == 0, done.stderr
    assert json.loads(done.stdout) == [expected for _, _, expected in cases]


def test_the_selector_persists_only_our_key_and_the_page_carries_the_active_language() -> None:
    page = (WEB_DIR / "index.html").read_text(encoding="utf-8")
    script = (WEB_DIR / "app.js").read_text(encoding="utf-8")
    assert '<html lang="en">' in page  # the served default, before the script runs
    assert 'data-lang-set="en"' in page and 'data-lang-set="es"' in page
    assert "document.documentElement.lang = state.lang" in script
    assert "writeStore(STORAGE_LANG, state.lang)" in script  # the selector
    assert "writeStore(STORAGE_LANG, fromUrl)" in script  # an explicit ?lang= is remembered
    # The language of dev0p.com is only ever read, never written.
    assert "'dev0p-lang'" in script
    assert "writeStore(STORAGE_LANG_SITE" not in script
    assert "localStorage.setItem('dev0p-lang'" not in script


def test_the_page_title_the_description_the_footer_and_the_artifacts_follow_the_language() -> None:
    page = (WEB_DIR / "index.html").read_text(encoding="utf-8")
    script = (WEB_DIR / "app.js").read_text(encoding="utf-8")
    assert '<meta name="description"' in page
    assert '<title>grounded-rag</title>' in page
    assert "document.title = t('page.title')" in script
    assert 'meta[name="description"]' in script and "t('page.description')" in script
    assert 'data-i18n="foot.note"' in page
    assert 'data-i18n="how.s3.artifact"' in page and 'data-i18n="how.s4.artifact"' in page
    for key in ("page.title", "page.description", "foot.note", "how.s3.artifact", "how.s4.artifact"):
        english = _i18n_value(script, "en", key)
        spanish = _i18n_value(script, "es", key)
        assert english and spanish, key
        assert english != spanish, key
    # What belongs to the manual keeps its own language: the worked example and
    # the quotes of the animation are Spanish in both languages.
    assert page.count('lang="es"') >= 3 and 'lang="en"' in page


def test_an_error_message_keeps_its_key_so_a_language_switch_translates_it() -> None:
    script = (WEB_DIR / "app.js").read_text(encoding="utf-8")
    assert "kind: 'error', key:" in script
    assert not re.search(r"kind: 'error', message:", script)
    for key in ("error.empty", "error.422", "error.429", "error.502"):
        assert f"key: '{key}'" in script, key
        assert _i18n_value(script, "en", key) and _i18n_value(script, "es", key), key
    assert script.count("renderError(t(last.key))") == 1
    # Everything on screen is rebuilt when the language changes.
    apply_language = script.split("function applyLanguage()", 1)[1].split("function setLanguage", 1)[0]
    assert "render();" in apply_language


def test_the_example_chips_come_from_one_file_and_follow_the_active_language() -> None:
    data = json.loads((WEB_DIR / "examples.json").read_text(encoding="utf-8"))
    topics = data["topics"]
    assert len(topics) == 4
    texts = {lang: [topic[lang] for topic in topics] for lang in ("en", "es")}
    assert [len(texts[lang]) for lang in ("en", "es")] == [4, 4]
    assert len(set(texts["en"] + texts["es"])) == 8
    assert all(text.strip() for text in texts["en"] + texts["es"])
    # Two topics keep the eval text of each language, so a report can answer them.
    assert {topic["eval"].get("en") for topic in topics} - {None} == {"a01", "a15"}
    assert {topic["eval"].get("es") for topic in topics} - {None} == {"a16", "a06"}

    page = (WEB_DIR / "index.html").read_text(encoding="utf-8")
    script = (WEB_DIR / "app.js").read_text(encoding="utf-8")
    # No hardcoded examples in the page: the chips are rendered from the file.
    assert "examples-chips" in page
    assert not re.search(r'data-example="', page)
    assert "window.fetch('examples.json')" in script
    assert "function renderExamples()" in script
    assert "exampleQuestions(state.lang)" in script
    assert "lang: state.lang" in script  # every chip carries the language of its text
    assert script.count("renderExamples()") >= 3  # the load, applyLanguage and the render
    assert "recordedExamples()" in script  # the recorded mode lists the same examples


def test_the_how_it_works_animation_pauses_and_has_a_reduced_motion_version() -> None:
    page = (WEB_DIR / "index.html").read_text(encoding="utf-8")
    script = (WEB_DIR / "app.js").read_text(encoding="utf-8")
    styles = (WEB_DIR / "app.css").read_text(encoding="utf-8")

    # Seven steps of the same concrete example, all of them in the page.
    assert page.count('class="how-step"') == 7
    for step in range(1, 8):
        assert page.count(f'how.s{step}.title') == 1
        assert page.count(f'how.s{step}.text') == 1
    assert "VACUUM FULL" in page

    # Accessible pause/resume: a real button, aria-pressed and two i18n labels.
    assert 'id="how-pause"' in page and 'aria-pressed="false"' in page
    assert 'data-i18n="how.pause"' in page
    assert script.count("'how.pause':") == 2 and script.count("'how.resume':") == 2
    assert "function applyHowPaused()" in script
    assert "state.howPaused = !state.howPaused" in script
    assert "'aria-pressed'" in script and "applyHowPaused();" in script

    # Pausing is CSS-only: the class stops the animation, no timer in the page.
    assert ".how.is-paused .how-step{animation-play-state:paused}" in styles
    assert "@keyframes how-step" in styles
    assert "animation:how-step" in styles

    # Reduced motion: every step visible, no animation, the toggle is moot.
    blocks = styles.split("@media (prefers-reduced-motion:reduce)")
    assert len(blocks) >= 2
    reduced = blocks[1]
    assert ".how-step{animation:none" in reduced and "opacity:1" in reduced
    assert ".how-pause{display:none}" in reduced



def test_the_web_switches_to_recorded_mode_when_the_budget_is_spent() -> None:
    """No JS runner here: this pins the wiring the Playwright run exercises."""
    script = (WEB_DIR / "app.js").read_text(encoding="utf-8")
    assert script.count("'banner.budget':") == 2  # English and Spanish
    assert script.count("showBanner('banner.budget')") == 1  # the health path
    assert script.count("showBanner('banner.recorded')") == 1  # the health path
    # The 503 path picks the banner from the budget flag.
    assert "state.budgetExhausted ? 'banner.budget' : 'banner.recorded'" in script
    assert "'budget_exhausted'" in script
    assert "health.budget && health.budget.exhausted" in script
    assert "state.budgetExhausted = true" in script
    # The banner follows a language switch instead of freezing the text it got.
    assert "function refreshBanner()" in script and "state.banner = key;" in script
    assert "refreshBanner();" in script.split("function applyLanguage()", 1)[1].split("function setLanguage", 1)[0]


# ------------------------------------------------------------------- metrics
def test_retrieval_metrics_are_read_per_language(reports: Path) -> None:
    metrics = retrieval_metrics(reports / "retrieval.json")
    assert metrics["policy"] == "downweight"
    assert metrics["recall@5"] == 1.0
    assert metrics["mrr@8"] == 0.6667
    assert metrics["by_lang"]["en"] == {"questions": 1, "recall@5": 1.0, "mrr@8": 1.0}
    assert metrics["by_lang"]["es"] == {"questions": 1, "recall@5": 1.0, "mrr@8": 0.3333}


def test_missing_retrieval_report_is_null(tmp_path, reports: Path) -> None:
    metrics = build_metrics(reports_dir=tmp_path / "empty", threshold_path=tmp_path / "threshold.json")
    assert metrics["retrieval"] is None
    assert metrics["threshold"]["value"] == 0.820859


def test_the_threshold_block_follows_the_configured_index(reports: Path, tmp_path) -> None:
    remote_threshold = dict(THRESHOLD, value=0.436503, unanswerable_abstained=11)
    (tmp_path / "threshold-remote.json").write_text(json.dumps(remote_threshold), encoding="utf-8")

    def metrics(**kwargs):
        return build_metrics(
            reports_dir=reports,
            threshold_path=tmp_path / "threshold.json",
            remote_threshold_path=tmp_path / "threshold-remote.json",
            **kwargs,
        )

    remote = metrics(config=make_settings(tmp_path, embed_model="vendor/embed-model"))
    assert remote["threshold"]["index"] == "remote"
    assert remote["threshold"]["value"] == 0.436503
    assert "threshold.json" not in json.dumps(remote["threshold"])

    local = metrics(config=make_settings(tmp_path))
    assert local["threshold"]["index"] == "local"
    assert local["threshold"]["value"] == 0.820859
    # Without settings the historical behaviour is kept: the local index.
    assert metrics()["threshold"]["index"] == "local"

    missing = build_metrics(
        reports_dir=reports,
        threshold_path=tmp_path / "threshold.json",
        remote_threshold_path=tmp_path / "none.json",
        config=make_settings(tmp_path, embed_model="vendor/embed-model"),
    )
    assert missing["threshold"] is None


def test_a_mock_report_is_never_presented_as_real(reports: Path) -> None:
    (reports / "2026-10-29-generate-mock.json").write_text(
        json.dumps(generation_report(48, 0.8333)), encoding="utf-8"
    )
    generation = generation_metrics(reports)
    assert generation["status"] == "pending"
    assert generation["simulated_reports"] == 1
    assert generation["report"] is None
    assert "answered" not in generation and "citations" not in generation
    dumped = json.dumps(generation)
    assert "0.8333" not in dumped and "48" not in dumped


def test_a_real_report_wins_and_the_mock_is_only_counted(reports: Path) -> None:
    (reports / "2026-10-29-generate-mock.json").write_text(
        json.dumps(generation_report(48, 0.8333)), encoding="utf-8"
    )
    (reports / "2026-10-30-generate.json").write_text(
        json.dumps(generation_report(49, 0.9)), encoding="utf-8"
    )
    generation = generation_metrics(reports)
    assert generation["status"] == "measured"
    assert generation["report"] == "2026-10-30-generate.json"
    assert generation["answered"] == 49
    assert generation["abstention"]["precision"] == 0.9
    assert generation["simulated_reports"] == 1
    assert "2026-10-29-generate-mock.json" != generation["report"]


def test_build_metrics_has_the_three_blocks(reports: Path) -> None:
    (reports / "2026-10-30-generate.json").write_text(json.dumps(generation_report(49, 0.9)), encoding="utf-8")
    metrics = build_metrics(reports_dir=reports, threshold_path=reports.parent / "threshold.json")
    assert set(metrics) == {"retrieval", "threshold", "generation"}
    assert metrics["retrieval"]["questions"] == 3
    assert metrics["threshold"]["unanswerable_abstained"] == 10
    assert metrics["generation"]["status"] == "measured"


def test_metrics_endpoint_uses_the_versioned_reports(tmp_path) -> None:
    client, _ = make_client(tmp_path)
    with client:
        body = client.get("/api/metrics").json()
    assert set(body) == {"retrieval", "threshold", "generation", "budget"}
    assert body["retrieval"]["report"] == "retrieval.json"
    assert isinstance(body["retrieval"]["recall@5"], float)
    assert body["budget"]["budget_usd"] == 0.5
    assert body["generation"]["status"] in {"pending", "measured"}
    if body["generation"]["status"] == "pending":
        assert body["generation"]["report"] is None


# ------------------------------------------------- metrics follow the index
def _write_remote_reports(reports: Path, tmp_path: Path) -> None:
    """A remote comparison, both generation runs and a local report of the day."""
    older = compare_report()
    for combination in older["combinations"]:
        if combination["name"] == "best combination":
            combination["summary"]["all"]["recall@5"] = 0.5  # the newest run is the one shown
    (reports / "2026-10-28-compare-remote.json").write_text(
        json.dumps({**older, "generated_at": "2026-10-28T00:00:00Z"}), encoding="utf-8"
    )
    (reports / "2026-10-29-compare-remote.json").write_text(json.dumps(compare_report()), encoding="utf-8")
    (reports / "2026-10-29-generate.json").write_text(
        json.dumps(generation_report(44, 0.8235, valid=315, total=315)), encoding="utf-8"
    )
    (reports / "2026-10-29-generate-remote.json").write_text(
        json.dumps(generation_report(46, 0.875, valid=334, total=335)), encoding="utf-8"
    )
    (tmp_path / "threshold-remote.json").write_text(
        json.dumps(dict(THRESHOLD, value=0.436503, unanswerable_abstained=11)), encoding="utf-8"
    )


def _panel(reports: Path, tmp_path: Path, config) -> dict:
    return build_metrics(
        reports_dir=reports,
        threshold_path=tmp_path / "threshold.json",
        remote_threshold_path=tmp_path / "threshold-remote.json",
        config=config,
    )


def test_the_panel_reads_the_remote_reports_when_the_index_is_remote(reports: Path, tmp_path) -> None:
    _write_remote_reports(reports, tmp_path)
    remote = _panel(reports, tmp_path, make_settings(tmp_path, embed_model="vendor/embed-model"))

    retrieval = remote["retrieval"]
    assert retrieval["index"] == "remote"
    # The newest comparison, read from the combination that run chose: not the
    # one with the best recall@5 (0.98) and not the failed combination.
    assert retrieval["report"] == "2026-10-29-compare-remote.json"
    assert retrieval["combination"] == "best combination"
    assert retrieval["recall@5"] == 0.96
    assert retrieval["mrr@8"] == 0.8563
    assert retrieval["questions"] == 80 and retrieval["answerable"] == 50
    assert retrieval["by_lang"] == {
        "en": {"recall@5": 0.96, "mrr@8": 0.8613},
        "es": {"recall@5": 0.96, "mrr@8": 0.8513},
    }

    assert remote["threshold"]["index"] == "remote"
    generation = remote["generation"]
    assert generation["index"] == "remote"
    assert generation["report"] == "2026-10-29-generate-remote.json"
    assert generation["citations"] == {"valid": 334, "total": 335, "validity_rate": 0.997}
    assert generation["answered"] == 46
    # No path of this machine reaches the panel.
    dumped = json.dumps(remote)
    assert "/home/" not in dumped and "eval/reports" not in dumped


def test_the_panel_keeps_the_local_reports_when_the_index_is_local(reports: Path, tmp_path) -> None:
    _write_remote_reports(reports, tmp_path)
    local = _panel(reports, tmp_path, make_settings(tmp_path))

    retrieval = local["retrieval"]
    assert retrieval["index"] == "local"
    assert retrieval["report"] == "retrieval.json"
    assert retrieval["recall@5"] == 1.0  # the fixture of retrieval.json
    assert retrieval["questions"] == 3

    generation = local["generation"]
    assert generation["index"] == "local"
    assert generation["report"] == "2026-10-29-generate.json"
    assert generation["citations"]["valid"] == 315
    assert generation["answered"] == 44
    # The remote comparison and the remote run are never read here.
    dumped = json.dumps(local)
    assert "compare-remote" not in dumped and "generate-remote" not in dumped


def test_a_mock_report_is_never_shown_on_either_index(reports: Path, tmp_path) -> None:
    (reports / "2026-10-30-generate-mock.json").write_text(
        json.dumps(generation_report(48, 0.8333, valid=7, total=9)), encoding="utf-8"
    )
    for config in (make_settings(tmp_path), make_settings(tmp_path, embed_model="vendor/embed-model")):
        generation = _panel(reports, tmp_path, config)["generation"]
        assert generation["status"] == "pending"
        assert generation["report"] is None
        assert generation["simulated_reports"] == 1
        assert "citations" not in generation and "answered" not in generation
        dumped = json.dumps(generation)
        assert "0.8333" not in dumped and "48" not in dumped and "7/9" not in dumped


def test_the_remote_retrieval_block_is_null_without_a_comparison(reports: Path, tmp_path) -> None:
    # No comparison report: the panel shows nothing instead of the local numbers.
    remote = _panel(reports, tmp_path, make_settings(tmp_path, embed_model="vendor/embed-model"))
    assert remote["retrieval"] is None
    assert remote["generation"]["report"] is None


def test_a_comparison_without_a_usable_chosen_row_shows_the_best_measured_one(tmp_path) -> None:
    from app.metrics import compare_retrieval_metrics

    report = compare_report()
    report["chosen"] = "a name that is not in the file"
    path = tmp_path / "2026-10-30-compare-remote.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    metrics = compare_retrieval_metrics(path)
    # Best MRR@8 among the measured combinations; the failed one is skipped.
    assert metrics["combination"] == "best combination"
    assert metrics["recall@5"] == 0.96

    broken = {**report, "combinations": [{"name": "local base", "status": "unavailable", "summary": {}}]}
    path.write_text(json.dumps(broken), encoding="utf-8")
    assert compare_retrieval_metrics(path) is None


def _panel_number(value: float) -> str:
    """The panel's own formatting of a number (``metricNumber`` of ``app.js``)."""
    if isinstance(value, int) or (isinstance(value, float) and value.is_integer()):
        return str(int(value))
    return str(round(value, 4))


def _js_why_block() -> str:
    """The pure figures of ``app.js``, ready for node."""
    script = (WEB_DIR / "app.js").read_text(encoding="utf-8")
    start = script.index("// --- why numbers (pure")
    end = script.index("// --- end why numbers ---") + len("// --- end why numbers ---")
    return script[start:end] + "\n"


NODE_WHY_HARNESS = """
const fs = require('fs');
const payload = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
function formatter(lang) {
  return function (value) {
    if (typeof value !== 'number') return String(value);
    var text = Number.isInteger(value) ? String(value) : String(Math.round(value * 10000) / 10000);
    return lang === 'es' ? text.replace('.', ',') : text;
  };
}
function render(lang, params) {
  var text = payload.templates[lang]['why.p6'];
  Object.keys(params).forEach(function (name) {
    text = text.split('{' + name + '}').join(params[name]);
  });
  return text;
}
const out = {};
['remote', 'local'].forEach(function (name) {
  out[name] = {};
  ['en', 'es'].forEach(function (lang) {
    var params = whyNumbers(payload.metrics[name], formatter(lang));
    out[name][lang] = params ? render(lang, params) : null;
  });
});
out.missing = whyNumbers(null, formatter('en'));
out.unmeasured = whyNumbers(payload.metrics.unmeasured, formatter('en'));
console.log(JSON.stringify(out));
"""


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_why_rag_quotes_the_numbers_of_the_panel(reports: Path, tmp_path) -> None:
    _write_remote_reports(reports, tmp_path)
    metrics = {
        "remote": _panel(reports, tmp_path, make_settings(tmp_path, embed_model="vendor/embed-model")),
        "local": _panel(reports, tmp_path, make_settings(tmp_path)),
        # A panel without a measured generation report: no figures at all.
        "unmeasured": {
            "retrieval": {"recall@5": 0.96, "questions": 80, "answerable": 50},
            "generation": {"status": "pending"},
        },
    }
    script = (WEB_DIR / "app.js").read_text(encoding="utf-8")
    templates = {lang: {"why.p6": _i18n_value(script, lang, "why.p6")} for lang in ("en", "es")}
    payload = {"metrics": metrics, "templates": templates}
    cases_path = tmp_path / "why.json"
    cases_path.write_text(json.dumps(payload), encoding="utf-8")
    harness = tmp_path / "why.js"
    harness.write_text(_js_why_block() + NODE_WHY_HARNESS, encoding="utf-8")
    done = subprocess.run(
        [shutil.which("node"), str(harness), str(cases_path)], capture_output=True, text=True, timeout=60
    )
    assert done.returncode == 0, done.stderr
    rendered = json.loads(done.stdout)

    # The remote panel: the same figures the "Why RAG" line quotes.
    retrieval = metrics["remote"]["retrieval"]
    citations = metrics["remote"]["generation"]["citations"]
    english = rendered["remote"]["en"]
    spanish = rendered["remote"]["es"]
    assert f"recall@5 {retrieval['recall@5']}" in english
    assert f"{citations['valid']}/{citations['total']}" in english
    assert f"{retrieval['questions']} questions" in english
    assert "0,96" in spanish and f"{citations['valid']}/{citations['total']}" in spanish
    # 334/335 and 0.96 are exactly what the panel shows for the remote index.
    assert "334/335" in english and "recall@5 0.96" in english

    # The local index quotes the local reports, not the remote ones.
    local = metrics["local"]
    assert f"recall@5 {_panel_number(local['retrieval']['recall@5'])}" in rendered["local"]["en"]
    assert f"{local['generation']['citations']['valid']}/{local['generation']['citations']['total']}" in rendered["local"]["en"]
    assert "334/335" not in rendered["local"]["en"]

    # Without the numbers the sentence carries no figures (never an invention).
    # The names of the metrics are labels, not figures: they are removed first.
    assert rendered["missing"] is None and rendered["unmeasured"] is None
    for lang in ("en", "es"):
        text = re.sub(r"\{[a-z]+\}", "", templates[lang]["why.p6"])
        text = text.replace("recall@5", "").replace("MRR@8", "")
        assert not re.search(r"\d", text), lang


def test_the_why_line_is_rebuilt_from_the_metrics_on_every_language_switch() -> None:
    script = (WEB_DIR / "app.js").read_text(encoding="utf-8")
    assert "function applyWhy()" in script
    assert "whyNumbers(state.metrics, metricNumber)" in script
    apply_language = script.split("function applyLanguage()", 1)[1].split("function setLanguage", 1)[0]
    assert "applyWhy();" in apply_language
    load = script.split("function loadMetrics()", 1)[1].split("// ---", 1)[0]
    assert load.count("applyWhy();") == 2  # the loaded panel and the failed one
    assert "t('why.p6Fallback')" in script and "t('why.noteFallback')" in script


def test_the_panel_names_the_index_and_the_reports_it_quotes() -> None:
    script = (WEB_DIR / "app.js").read_text(encoding="utf-8")
    for key in ("metrics.source", "metrics.index.local", "metrics.index.remote"):
        assert _i18n_value(script, "en", key) and _i18n_value(script, "es", key), key
    # "Index" is only written once per line: "Index: remote", not "Index: remote index".
    assert _i18n_value(script, "en", "metrics.source") == "Index"
    assert _i18n_value(script, "en", "metrics.index.local") == "local"
    assert _i18n_value(script, "en", "metrics.index.remote") == "remote"
    assert _i18n_value(script, "es", "metrics.source") == "Índice"
    assert _i18n_value(script, "es", "metrics.index.local") == "local"
    assert _i18n_value(script, "es", "metrics.index.remote") == "remoto"
    assert "t('metrics.index.' + index)" in script
    assert "[retrieval.report]" in script  # the retrieval block names its report
    assert "[generation.report]" in script  # and so does the generation block


# ------------------------------------------------------ panel number formats
def _js_numbers_block() -> str:
    """The number formatting of ``app.js``, ready for node."""
    script = (WEB_DIR / "app.js").read_text(encoding="utf-8")
    start = script.index("// --- numbers (pure")
    end = script.index("// --- end numbers ---") + len("// --- end numbers ---")
    return script[start:end] + "\n"


NODE_NUMBERS_HARNESS = """
const fs = require('fs');
const payload = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
const out = {};
['en', 'es'].forEach(function (lang) {
  state.lang = lang;
  out[lang] = {
    numbers: payload.numbers.map(function (value) { return metricNumber(value); }),
    latency: latencyText(payload.latency, metricNumber),
    missing: latencyText(null, metricNumber)
  };
});
console.log(JSON.stringify(out));
"""


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_the_latency_row_writes_the_decimal_mark_of_the_language(tmp_path: Path) -> None:
    # The values of the remote generation report: 1451.7 / 2957.2 ms.
    payload = {"numbers": [0.96, 1451.7, 3, None], "latency": {"median": 1451.7, "p95": 2957.2}}
    harness = tmp_path / "numbers.js"
    harness.write_text(
        "var state = { lang: 'en' };\n" + _js_numbers_block() + NODE_NUMBERS_HARNESS, encoding="utf-8"
    )
    payload_path = tmp_path / "numbers.json"
    payload_path.write_text(json.dumps(payload), encoding="utf-8")
    done = subprocess.run(
        [shutil.which("node"), str(harness), str(payload_path)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert done.returncode == 0, done.stderr
    rendered = json.loads(done.stdout)
    assert rendered["en"]["latency"] == "1451.7 / 2957.2 ms"
    assert rendered["es"]["latency"] == "1451,7 / 2957,2 ms"
    assert rendered["es"]["numbers"] == ["0,96", "1451,7", "3", "-"]
    assert rendered["en"]["numbers"] == ["0.96", "1451.7", "3", "-"]
    assert rendered["es"]["missing"] == "- / - ms"

    # The row goes through the same formatter as the rest of the panel.
    script = (WEB_DIR / "app.js").read_text(encoding="utf-8")
    row = script.split("[t('metrics.retrievalMs')", 1)[1].split("]", 1)[0]
    assert "latencyText(generation.retrieval_ms, metricNumber)" in row
    assert "number(" not in row


# ------------------------------------------------------------------ recorded
def test_recorded_json_is_valid_and_mock_free() -> None:
    document = json.loads((WEB_DIR / "recorded.json").read_text(encoding="utf-8"))
    entries = document["entries"]
    assert document["generated_at"].endswith("Z")
    # The served document names the report, never a path of this machine.
    assert document["report"] is None or "/" not in document["report"]
    assert [entry["id"] for entry in entries] == list(EXAMPLE_IDS)
    assert sum(1 for entry in entries if entry["lang"] == "en") == 6
    assert sum(1 for entry in entries if entry["lang"] == "es") == 6
    assert {entry["id"] for entry in entries} & {"u01", "u30"} == {"u01", "u30"}
    for entry in entries:
        assert entry["question"].strip()
        assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", entry["recorded_at"])
        assert entry["source"] in {"retrieval-only", "live"} or entry["source"].startswith("eval-report:")
        assert "mock" not in entry["source"]
        assert entry["status"] in {"retrieval_only", "answered", "abstained"}
        assert [source["n"] for source in entry["sources"]] == list(range(1, len(entry["sources"]) + 1))
        assert all(source["url"].startswith("https://www.postgresql.org/docs/17/") for source in entry["sources"])
        assert all(source["title"] for source in entry["sources"])
        if "cost_usd" in entry:
            assert entry["cost_usd"] >= 0
        if entry["status"] == "retrieval_only":
            assert entry["answer"] is None
        else:  # pragma: no cover - only when a real report exists
            assert isinstance(entry["answer"]["sentences"], list)


def test_every_example_has_a_real_answer_in_recorded_json() -> None:
    document = json.loads((WEB_DIR / "recorded.json").read_text(encoding="utf-8"))
    by_question = {entry["question"]: entry for entry in document["entries"]}
    for lang in ("en", "es"):
        for question in [topic[lang] for topic in json.loads((WEB_DIR / "examples.json").read_text(encoding="utf-8"))["topics"]]:
            entry = by_question[question]
            assert entry["status"] in {"answered", "abstained"}, (lang, question, entry["status"])
            assert entry["answer"] is not None, (lang, question)
            if entry["source"] == "live":
                assert entry["cost_usd"] > 0


def test_a_mock_report_is_never_replayed(reports: Path) -> None:
    mock = reports / "2026-10-29-generate-mock.json"
    mock.write_text(json.dumps(generation_report(48, 0.8333)), encoding="utf-8")
    assert latest_real_report(reports) is None
    assert report_answers(None) == ({}, None)
    with pytest.raises(ValueError):
        from web.record import record

        record(report_path=mock, out_path=reports / "out.json")


def test_latest_real_report_ignores_mock_files(reports: Path) -> None:
    (reports / "2026-10-29-generate-mock.json").write_text("{}", encoding="utf-8")
    real = reports / "2026-10-28-generate.json"
    real.write_text("{}", encoding="utf-8")
    assert latest_real_report(reports) == real


def test_latest_real_report_prefers_the_remote_run_of_the_same_day(reports: Path) -> None:
    (reports / "2026-10-29-generate.json").write_text("{}", encoding="utf-8")
    remote = reports / "2026-10-29-generate-remote.json"
    remote.write_text("{}", encoding="utf-8")
    (reports / "2026-10-29-generate-mock.json").write_text("{}", encoding="utf-8")
    assert latest_real_report(reports) == remote
    later = reports / "2026-10-30-generate.json"
    later.write_text("{}", encoding="utf-8")
    assert latest_real_report(reports) == later


def test_build_entry_keeps_a_real_answer_and_otherwise_stays_retrieval_only() -> None:
    sources = [{"n": 1, "url": "https://www.postgresql.org/docs/17/x.html#s", "title": "X"}]
    question = {"id": "a01", "lang": "en", "question": "How do partial indexes work?", "answerable": True}

    retrieval_only = build_entry(question, sources, {}, None)
    assert retrieval_only["status"] == "retrieval_only"
    assert retrieval_only["source"] == "retrieval-only"
    assert retrieval_only["answer"] is None

    rows = {
        "how do partial indexes work?": {
            "abstain": False,
            "reason": None,
            "sentences": [{"text": "It restricts rows.", "citations": [{"n": 1, "quote": "a partial index"}]}],
            "sources": [{"n": 1, "url": "https://www.postgresql.org/docs/17/x.html#s", "title": "X"}],
        }
    }
    answered = build_entry(question, sources, rows, "eval-report:2026-10-30-generate.json")
    assert answered["status"] == "answered"
    assert answered["source"] == "eval-report:2026-10-30-generate.json"
    assert answered["answer"]["sentences"][0]["text"] == "It restricts rows."

    rows["how do partial indexes work?"]["abstain"] = True
    assert build_entry(question, sources, rows, "eval-report:x.json")["status"] == "abstained"


def test_the_web_uses_relative_urls_so_it_can_be_served_under_a_path_prefix() -> None:
    page = (WEB_DIR / "index.html").read_text(encoding="utf-8")
    script = (WEB_DIR / "app.js").read_text(encoding="utf-8")
    assert not re.search(r'(?:href|src)="/[^/]', page)
    assert not re.search(r"fetch\('/|'/api/|'/recorded\.json'", script)


# ------------------------------------------------------------------ --live
EXAMPLES_FIXTURE = {
    "version": 1,
    "topics": [
        {
            "key": "partial-index",
            "eval": {"en": "a01"},
            "en": "How do partial indexes work?",
            "es": "¿Cómo funcionan los índices parciales?",
        },
        {
            "key": "brin",
            "eval": {},
            "en": "What is a BRIN index?",
            "es": "¿Qué es un índice BRIN?",
        },
    ],
}


def search_results() -> list[dict]:
    return [
        {
            "url": "https://www.postgresql.org/docs/17/indexes-partial.html",
            "page_title": "Partial Indexes",
            "chunk_ids": ["c1"],
            "section_path": ["Partial Indexes"],
            "text": PARTIAL_INDEX_TEXT,
            "rrf": 0.0325,
        }
    ]


@pytest.fixture
def offline_record(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """A fake search, a fake embedder, a fake store and no threshold file."""
    import retrieval.search as search
    from generation.generate import Thresholds

    monkeypatch.setattr(search, "get_store", lambda: FakeStore())
    monkeypatch.setattr(search, "get_embedder", lambda: FakeEmbedder())
    monkeypatch.setattr(search, "hybrid_search", lambda question, **kwargs: search_results())
    monkeypatch.setattr(
        "generation.generate.load_thresholds",
        lambda *args, **kwargs: Thresholds(local={"signal": "vector_cosine", "value": 0.0}),
    )
    examples = tmp_path / "examples.json"
    examples.write_text(json.dumps(EXAMPLES_FIXTURE), encoding="utf-8")
    return examples


def test_record_without_live_never_calls_the_model(tmp_path, offline_record: Path) -> None:
    llm = ScriptedLLM([GOOD_ANSWER])
    document = record(
        live=False,
        llm=llm,
        report_path=None,
        out_path=tmp_path / "out.json",
        examples_path=offline_record,
    )
    assert llm.calls == 0
    assert "live" in document and document["live"]["used"] is False
    assert document["live"]["questions"] == 0 and document["live"]["cost_usd"] == 0.0
    # Four example texts (two topics, both languages) plus the four extras.
    assert len(document["entries"]) == 8
    assert {entry["status"] for entry in document["entries"]} == {"retrieval_only"}
    assert all(entry["answer"] is None and "cost_usd" not in entry for entry in document["entries"])


def test_record_live_calls_the_model_only_for_the_questions_the_report_misses(tmp_path, offline_record: Path) -> None:
    report = tmp_path / "2026-10-30-generate.json"
    report.write_text(
        json.dumps(
            {
                "questions": [
                    {
                        "id": "a01",
                        "question": "How do partial indexes work?",
                        "abstain": False,
                        "reason": None,
                        "sentences": GOOD_ANSWER["sentences"],
                        "sources": [
                            {"n": 1, "url": "https://www.postgresql.org/docs/17/indexes-partial.html", "title": "Partial Indexes"}
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    llm = ScriptedLLM([GOOD_ANSWER, ABSTAIN_ANSWER])
    config = make_settings(tmp_path, llm_price_input_per_m=1.0, llm_price_output_per_m=1.0)
    document = record(
        live=True,
        llm=llm,
        config=config,
        report_path=report,
        out_path=tmp_path / "out.json",
        examples_path=offline_record,
    )
    entries = document["entries"]
    assert llm.calls == 7  # the eight rows minus the one the report answers
    assert entries[0]["source"] == "eval-report:2026-10-30-generate.json"
    assert entries[0]["status"] == "answered" and "cost_usd" not in entries[0]
    live_entries = [entry for entry in entries if entry["source"] == "live"]
    assert len(live_entries) == 7
    assert all(entry["cost_usd"] > 0 and entry["answer"] is not None for entry in live_entries)
    # The abstention of the model is stored verbatim, not replaced by sources.
    assert {entry["status"] for entry in live_entries} == {"answered", "abstained"}
    abstained = [entry for entry in live_entries if entry["status"] == "abstained"]
    assert all(entry["answer"]["sources"] == [] and entry["answer"]["abstain"] for entry in abstained)
    assert document["live"]["used"] is True and document["live"]["questions"] == 7
    assert document["live"]["cost_usd"] == pytest.approx(7 * 0.00012)


def test_record_live_stops_at_the_cost_ceiling(tmp_path, offline_record: Path) -> None:
    llm = ScriptedLLM([GOOD_ANSWER])
    config = make_settings(tmp_path, llm_price_input_per_m=1.0, llm_price_output_per_m=1.0)
    document = record(
        live=True,
        llm=llm,
        config=config,
        report_path=None,
        out_path=tmp_path / "out.json",
        examples_path=offline_record,
        max_cost=0.0002,
    )
    # Every answer costs 120 tokens x 1 USD/M = 0.00012: one more call would pass the ceiling.
    assert llm.calls == 2
    assert document["live"]["skipped"] == 6
    assert sum(1 for entry in document["entries"] if entry["status"] == "retrieval_only") == 6
    assert document["live"]["cost_usd"] == pytest.approx(0.00024)


def test_record_live_refuses_without_a_configured_model(tmp_path, offline_record: Path) -> None:
    with pytest.raises(ValueError):
        record(
            live=True,
            config=make_settings(tmp_path),  # no base URL, no model, no key
            report_path=None,
            out_path=tmp_path / "out.json",
            examples_path=offline_record,
        )


def test_example_questions_cover_both_languages_of_every_topic() -> None:
    rows = example_questions()
    assert len(rows) == 8
    assert [row["lang"] for row in rows] == ["en", "es"] * 4
    # The four texts that already existed keep their eval id, the rest get a stable one.
    assert {row["id"] for row in rows} == {"a01", "a15", "a16", "a06", "x01-es", "x02-es", "x03-en", "x04-en"}
    assert all(row["question"].strip() for row in rows)
    assert len(question_rows()) == 12  # the eight examples plus the four extras

