"""HTTP API and web UI: ``/api/*`` plus the static files in ``web/`` (loopback only).

The embedding model is loaded once at startup (``lifespan``); a request never
loads it.  No CORS is enabled on purpose.  Nothing here logs the API key, the
IP salt or a client IP.  Every response (API and static) carries the same
security headers, and the ``/api/*`` routes are declared before the static
mount so they always win.
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, field_validator

from generation.config import Config, LLMNotConfigured, load_config
from generation.generate import answer_question, load_threshold
from generation.llm import LLMError, build_llm
from retrieval.search import get_embedder, get_store, hybrid_search

from .limits import RateLimiter
from .metrics import WEB_DIR, build_metrics
from .usage import UsageStore

MAX_QUESTION_CHARS = 500
DEFAULT_K = 8
#: No inline script/style, no CDN except the Google Fonts CSS/woff2 files.
CONTENT_SECURITY_POLICY = (
    "default-src 'self'; style-src 'self' https://fonts.googleapis.com; "
    "font-src https://fonts.gstatic.com; img-src 'self' data:; connect-src 'self'; "
    "frame-ancestors 'none'; base-uri 'none'"
)
logger = logging.getLogger("grounded_rag.app")


class AskRequest(BaseModel):
    """Body of ``POST /api/ask``."""

    question: str
    lang: Literal["en", "es", "other"] | None = None

    @field_validator("question")
    @classmethod
    def _validate_question(cls, value: str) -> str:
        text = value.strip()
        if not text:
            raise ValueError("question must not be empty")
        if len(text) > MAX_QUESTION_CHARS:
            raise ValueError(f"question must be at most {MAX_QUESTION_CHARS} characters")
        return text


def _retrieved_sources(results: list[dict]) -> list[dict]:
    return [
        {"n": n, "url": result["url"], "title": result["page_title"]}
        for n, result in enumerate(results, start=1)
    ]


def create_app(
    config: Config | None = None,
    *,
    store=None,
    embedder=None,
    llm=None,
    usage: UsageStore | None = None,
    limiter: RateLimiter | None = None,
    threshold: dict | None = None,
) -> FastAPI:
    """Build the app; every dependency is injectable (tests use fakes)."""
    settings = config or load_config()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if app.state.store is None:
            app.state.store = get_store()
        if app.state.embedder is None:
            app.state.embedder = get_embedder()
        yield

    app = FastAPI(title="grounded-rag", version="0.3.0", lifespan=lifespan)

    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        # ``web/`` also holds the recorder module and its bytecode: neither is served.
        path = request.url.path
        if path.endswith((".py", ".pyc")) or "__pycache__" in path:
            return Response(status_code=404)
        response = await call_next(request)
        response.headers["Content-Security-Policy"] = CONTENT_SECURITY_POLICY
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        return response

    app.state.config = settings
    app.state.store = store
    app.state.embedder = embedder
    app.state.llm = llm
    app.state.usage = usage or UsageStore(settings.usage_db, settings.daily_token_cap)
    limit, window = settings.rate_limit()
    app.state.limiter = limiter or RateLimiter(limit, window, salt=settings.ip_salt)
    app.state.threshold = load_threshold() if threshold is None else threshold

    def model_for(state) -> object:
        if state.llm is None:
            state.llm = build_llm(state.config)
        return state.llm

    @app.get("/api/health")
    def health(request: Request) -> dict:
        state = request.app.state
        return {
            "status": "ok",
            "generation": bool(state.config.generation),
            "llm_configured": state.config.llm_configured or state.config.llm_fake == "mock",
        }

    @app.get("/api/metrics")
    def metrics() -> dict:
        """Retrieval, abstention threshold and generation, read from the reports."""
        return build_metrics()

    @app.post("/api/ask")
    def ask(payload: AskRequest, request: Request) -> dict:
        state = request.app.state
        settings = state.config
        ip = request.client.host if request.client else ""

        if not state.limiter.allow(ip):
            raise HTTPException(status_code=429, detail="too many requests, slow down")

        if not settings.generation:
            started = time.perf_counter()
            results = hybrid_search(payload.question, k=DEFAULT_K, store=state.store, embedder=state.embedder)
            logger.info("ask (generation off): %d sources in %.0f ms", len(results), (time.perf_counter() - started) * 1000)
            return {
                "abstain": True,
                "sentences": [],
                "sources": _retrieved_sources(results),
                "reason": "generation_disabled",
            }

        if state.usage.exhausted():
            raise HTTPException(
                status_code=503,
                detail="daily token budget exhausted; the model is not called again until tomorrow (UTC)",
            )

        try:
            llm = model_for(state)
        except LLMNotConfigured:
            logger.warning("ask: generation model not configured")
            raise HTTPException(status_code=503, detail="generation is not configured on this server")

        try:
            answer = answer_question(
                payload.question,
                payload.lang,
                config=settings,
                store=state.store,
                embedder=state.embedder,
                llm=llm,
                k=DEFAULT_K,
                threshold=state.threshold,
                use_threshold=state.threshold is not None,
            )
        except LLMError as error:
            # Status codes only: no headers, no key, no upstream body.
            logger.warning("ask: LLM call failed (%s)", type(error).__name__)
            raise HTTPException(status_code=502, detail="generation failed")

        state.usage.add(answer["prompt_tokens"] + answer["completion_tokens"])
        body = {"abstain": answer["abstain"], "sentences": answer["sentences"], "sources": answer["sources"]}
        if answer["abstain"]:
            body["reason"] = answer["reason"]
        logger.info(
            "ask: abstain=%s reason=%s sentences=%d tokens=%d",
            answer["abstain"],
            answer["reason"],
            len(answer["sentences"]),
            answer["prompt_tokens"] + answer["completion_tokens"],
        )
        return body

    # Mounted last on purpose: every ``/api/*`` route above wins over ``/``.
    app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
    return app


app = create_app()
