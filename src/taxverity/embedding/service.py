"""Steps 4.1 and 4.2 — the `taxverity-embed` HTTP contract and its readiness gate.

A separate service because weights and load time have nothing to do with
request handling, and because it keeps torch out of the API image (plan §2.7).
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from enum import StrEnum
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from pydantic import BaseModel, ConfigDict, Field

from taxverity.embedding.backends import (
    MAX_BATCH,
    Embedder,
    EmbedKind,
    ModelInfo,
    StubEmbedder,
)
from taxverity.observability import configure_logging, get_logger

logger = get_logger(__name__)

WARMUP_TEXT = "warm-up"


class Readiness(StrEnum):
    PENDING = "pending"
    READY = "ready"
    FAILED = "failed"


class EmbedRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    texts: list[str] = Field(min_length=1, max_length=MAX_BATCH)
    # Required, with no default: a caller who forgets gets a 422 rather than a
    # plausible, well-formed, wrongly-encoded vector (ADR-069). One kind per
    # request, so a batch cannot be half-mislabelled.
    kind: EmbedKind


class EmbedResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    embeddings: list[list[float]]
    # Repeated on every response, not just /model-info: a client that checks
    # identity once at startup cannot see the service being replaced under it
    # by a rolling deploy, which is exactly when skew appears.
    model: ModelInfo
    # Echoed for the same reason, one level down: the client asserts the
    # service applied the encoding it asked for (ADR-069).
    kind: EmbedKind


class HealthResponse(BaseModel):
    status: str


class ReadyResponse(BaseModel):
    ready: bool
    # Carried so an operator can tell a task that is still loading from one
    # that will never load, without reading the logs. Both answer 503.
    state: Readiness


def get_embedder(request: Request) -> Embedder:
    return request.app.state.embedder


def get_readiness(request: Request) -> Readiness:
    return request.app.state.readiness


EmbedderDep = Annotated[Embedder, Depends(get_embedder)]
ReadinessDep = Annotated[Readiness, Depends(get_readiness)]


async def _warm_up(app: FastAPI) -> None:
    """The first forward pass, paid before the readiness gate opens (ADR-026).

    Runs off the event loop: a torch forward pass is 1-3 s of blocking CPU, and
    holding the loop through it would stall the very `/ready` probe this exists
    to answer.
    """
    embedder: Embedder = app.state.embedder
    info = embedder.info()
    started = time.perf_counter()
    try:
        # Both kinds, because asymmetric encoding means two code paths and
        # warming only one leaves a query-side recipe bug to surface at the
        # first user query instead of at /ready (ADR-069).
        for kind in EmbedKind:
            vectors = await asyncio.to_thread(embedder.embed, [WARMUP_TEXT], kind)
            # The warm-up doubles as the one place the backend is made to prove
            # it does what it declares. A width that disagrees with `dim` is the
            # same skew family as ADR-026's, caught at boot rather than at query
            # time.
            if len(vectors) != 1 or len(vectors[0]) != info.dim:
                raise ValueError(
                    f"backend declares dim={info.dim} but returned "
                    f"{len(vectors)} vector(s) of width "
                    f"{[len(v) for v in vectors]} for kind={kind.value}"
                )
    except Exception:
        # Fail closed and stay up: a task that answers /ready with 503 forever
        # is drained and replaced exactly like one that exited, and it can
        # still be asked /health and /model-info while being diagnosed.
        app.state.readiness = Readiness.FAILED
        logger.exception("embed warm-up failed; /ready stays closed")
        return
    app.state.readiness = Readiness.READY
    logger.info(
        "embed warm-up complete in %.3fs: model=%s revision=%s dim=%d "
        "runtime=%s encoding=%s",
        time.perf_counter() - started,
        info.model_id,
        info.revision,
        info.dim,
        info.runtime,
        info.encoding,
    )


def create_app(embedder: Embedder | None = None) -> FastAPI:
    embedder = embedder or StubEmbedder()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        configure_logging()
        info = app.state.embedder.info()
        logger.info(
            "embed service starting: model=%s revision=%s dim=%d "
            "runtime=%s encoding=%s",
            info.model_id,
            info.revision,
            info.dim,
            info.runtime,
            info.encoding,
        )
        # Started rather than awaited: blocking startup would leave the port
        # unbound, so a probe would see a refused connection instead of the
        # 503 the gate is specified to answer.
        task = asyncio.create_task(_warm_up(app))
        try:
            yield
        finally:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task

    app = FastAPI(title="taxverity-embed", lifespan=lifespan)
    app.state.embedder = embedder
    app.state.readiness = Readiness.PENDING

    @app.post("/embed", response_model=EmbedResponse)
    def embed(
        payload: EmbedRequest, backend: EmbedderDep, readiness: ReadinessDep
    ) -> EmbedResponse:
        # Request texts are never logged. This endpoint carries user query text
        # (salary figures, PAN, Aadhaar) and Rule 03 puts every such path
        # behind redact(); the cheapest way to hold that here is to emit none
        # of it. Batch size and timing belong to the client (Step 4.3).
        if readiness is Readiness.FAILED:
            # PENDING is deliberately still served: the caller merely pays the
            # first-pass cost the warm-up exists to move. FAILED would raise on
            # every call anyway, and a 503 says so more usefully than a 500.
            raise HTTPException(
                status_code=503, detail="embedding backend failed warm-up"
            )
        return EmbedResponse(
            embeddings=backend.embed(payload.texts, payload.kind),
            model=backend.info(),
            kind=payload.kind,
        )

    @app.get("/model-info", response_model=ModelInfo)
    def model_info(backend: EmbedderDep) -> ModelInfo:
        return backend.info()

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        # Liveness only, and independent of readiness: a task still warming up
        # or one that failed must stay diagnosable rather than look dead.
        return HealthResponse(status="ok")

    @app.get("/ready", response_model=ReadyResponse)
    def ready(response: Response, readiness: ReadinessDep) -> ReadyResponse:
        if readiness is not Readiness.READY:
            response.status_code = 503
        return ReadyResponse(ready=readiness is Readiness.READY, state=readiness)

    return app


app = create_app()
