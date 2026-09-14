"""Step 14.1 — the app factory, its lifespan, and the two endpoints that need
no auth: `/health` and `/v1/corpus`.

Everything the graph needs beyond a per-request connection is built exactly
once here, at startup: `resolve_serving()` first — plain SQL, so a vendor
outage cannot stop a boot (Step 6.6) — then chunks hydrated from the database
(TaxVerity is deploy-only; a production instance carries no `chunks.jsonl`),
then BM25, the citation shortcut, the term bridge, the reranker and the LLM
stack (`build_deps`, Step 13.2/Phase 14). Only `GraphDeps.conn` varies per
request.

A small `psycopg_pool.ConnectionPool` hands out one connection per request
for thread, fact-state and auth persistence, with prepared statements
disabled — the Supabase pooler runs in transaction-pooling mode, where a
server-side prepared statement from one checkout can be replayed against a
different backend on the next. The dense index keeps one dedicated
connection, opened once and never pooled: it only ever runs read-only vector
search, and psycopg3 connections are safe (if serialising) to share across
threads, so concurrent requests queue behind an in-flight search rather than
corrupt one another — an accepted simplification at this project's scale
(rule 01: the simplest production-sensible implementation), not a hidden
correctness gap.
"""

from __future__ import annotations

import dataclasses
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass

import psycopg
from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from psycopg_pool import ConnectionPool
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request as StarletteRequest
from starlette.responses import Response as StarletteResponse
from starlette.types import ASGIApp

from taxverity.api.errors import install_error_handlers
from taxverity.api.limits import MAX_BODY_BYTES
from taxverity.auth.tokens import AccessTokens
from taxverity.config import Settings
from taxverity.db.serving import ServedCorpus
from taxverity.graph.build import build_deps
from taxverity.graph.state import GraphDeps
from taxverity.mail.gmail import Mailer, mailer_from_settings
from taxverity.observability import configure_logging, get_logger

logger = get_logger(__name__)

POOL_MIN_SIZE = 1
POOL_MAX_SIZE = 10


class BodySizeLimitMiddleware(BaseHTTPMiddleware):
    """Step 14.6. Refuses on `Content-Length` alone — cheap, and every client
    this API expects (JSON POSTs from the frontend) sends one. A request with
    no `Content-Length` (chunked transfer) is not rejected here; it is not a
    request shape this project's own frontend produces."""

    def __init__(self, app: ASGIApp, max_bytes: int = MAX_BODY_BYTES) -> None:
        super().__init__(app)
        self._max_bytes = max_bytes

    async def dispatch(
        self, request: StarletteRequest, call_next
    ) -> StarletteResponse:  # type: ignore[no-untyped-def]
        content_length = request.headers.get("content-length")
        if content_length is not None and int(content_length) > self._max_bytes:
            return JSONResponse({"detail": "invalid_request"}, status_code=413)
        return await call_next(request)  # type: ignore[no-any-return]


@dataclass
class AppState:
    settings: Settings
    pool: ConnectionPool
    dense_conn: psycopg.Connection
    static_deps: GraphDeps
    served: ServedCorpus
    access_tokens: AccessTokens
    mailer: Mailer


def app_state(request: Request) -> AppState:
    return request.app.state.app_state  # type: ignore[no-any-return]


def _configure_connection(conn: psycopg.Connection) -> None:
    conn.autocommit = True
    conn.prepare_threshold = None


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        configure_logging()
        dsn = settings.require("database_url")
        pool = ConnectionPool(
            dsn,
            min_size=POOL_MIN_SIZE,
            max_size=POOL_MAX_SIZE,
            configure=_configure_connection,
            open=False,
        )
        pool.open(wait=True)
        dense_conn = psycopg.connect(dsn, autocommit=True)
        dense_conn.prepare_threshold = None
        try:
            static_deps, served = build_deps(settings, dense_conn)
        except Exception:
            pool.close()
            dense_conn.close()
            raise
        logger.info(
            "serving corpus %s…, embedding set %d, %d chunks",
            served.corpus_version[:16],
            served.embedding_set_id,
            served.chunk_count,
        )
        app.state.app_state = AppState(
            settings=settings,
            pool=pool,
            dense_conn=dense_conn,
            static_deps=static_deps,
            served=served,
            access_tokens=AccessTokens.from_settings(settings),
            mailer=mailer_from_settings(settings),
        )
        try:
            yield
        finally:
            pool.close()
            dense_conn.close()

    app = FastAPI(title="TaxVerity API", lifespan=lifespan)
    install_error_handlers(app)
    app.add_middleware(BodySizeLimitMiddleware)
    # Only the configured frontend origin (Vercel in prod, PLAN 14.6) — not a
    # wildcard, since credentials (the refresh cookie) are allowed.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[settings.app_base_url],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/corpus")
    def corpus(state: AppState = Depends(app_state)) -> dict[str, object]:  # noqa: B008
        served = state.served
        return {
            "corpus_version": served.corpus_version,
            "doc_id": served.doc_id,
            "chunk_count": served.chunk_count,
            "embedding_set_id": served.embedding_set_id,
            "vector_count": served.vector_count,
            "model": served.model.model_dump(),
        }

    from taxverity.api.auth_routes import router as auth_router
    from taxverity.api.guest_routes import router as guest_router
    from taxverity.api.threads_routes import router as threads_router
    from taxverity.api.turns_routes import router as turns_router

    app.include_router(auth_router)
    app.include_router(threads_router)
    app.include_router(turns_router)
    app.include_router(guest_router)
    return app


@contextmanager
def request_deps(state: AppState) -> Iterator[GraphDeps]:
    """One `GraphDeps` per request: every built-once component of
    `state.static_deps`, with `conn` swapped for a connection checked out from
    the pool for exactly this request's lifetime."""
    with state.pool.connection() as conn:
        yield dataclasses.replace(state.static_deps, conn=conn)
