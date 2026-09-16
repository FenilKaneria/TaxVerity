"""Closes an ADR-112 gap found live in Phase 17: `guests/quota.py` (Step
11.4e) has always existed, but nothing ever called it — no route, no
frontend. `POST /v1/guest/turns` is the unauthenticated counterpart to
`POST /v1/threads/{id}/turns`.

A guest turn is answered on its own — no thread, no message history, no fact
state (rule 04) — so this composes `deps.classifier` / `deps.retriever` /
`deps.packer` / `deps.generator` directly rather than through the compiled
LangGraph graph, which is built around a persisted thread and would need
every persistence node (`load_thread`, `merge_facts`, `finalize`) stubbed out
to fit a caller that has none of those. No fact extraction runs, so
`facts=None` and `computation=None` always — a guest question about a
specific figure gets a text-only, citation-backed answer, never a
computation; that is the concrete meaning of "no fact state" here, not an
oversight.

The guest id travels in an httpOnly cookie, minted on first use if absent.
`record_guest_turn` enforces `GUEST_TURN_LIMIT` per id and the looser
per-IP cap for a cleared cookie (Step 11.4e); `GuestLimitReached` becomes the
same 429 the daily-turn-cap path already uses.

Advisor pivot, Step 6 — this route duplicates the graph's classify/route
logic, so every routing change lands here too, not just in `graph/nodes.py`:
a refusal or conversational reply now streams its text on the final event
(it used to be dropped entirely on this path, unlike the authenticated graph,
which at least persisted it to the database); an in-scope turn now calls
`safety.evidence_gate.gate()` after generation, exactly like
`generate_verify` does, so a zero-grounded-claim guest answer gets the fixed
insufficient-evidence message instead of silence. Deliberately **not**
added: the graph's one-shot corrective retry (ADR-110 scopes it to the
authenticated graph) — re-running retrieval here would double the cost of
every guest turn on the free tier for a user this project does not know yet.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from typing import Any
from uuid import UUID, uuid4

import psycopg
from fastapi import APIRouter, Cookie, Depends, Request, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from taxverity.api.app import AppState, app_state, request_deps
from taxverity.api.deps import get_conn
from taxverity.api.errors import rate_limited
from taxverity.generation.claims import ClaimEvent
from taxverity.graph.state import FinalEvent, StageEvent, TraceEntry
from taxverity.guests.quota import (
    GUEST_TURN_LIMIT,
    GuestLimitReached,
    guest_turns_used,
    record_guest_turn,
)
from taxverity.observability import get_logger
from taxverity.safety.classifier import ScopeCategory
from taxverity.safety.evidence_gate import gate

logger = get_logger(__name__)

router = APIRouter(prefix="/v1/guest", tags=["guest"])

GUEST_COOKIE = "taxverity_guest_id"
GUEST_COOKIE_MAX_AGE = 60 * 60 * 24 * 30


class GuestTurnRequest(BaseModel):
    question: str = Field(min_length=1)


def _client_ip(request: Request) -> str:
    return request.client.host if request.client else "unknown"


def _guest_id(response: Response, raw: str | None) -> UUID:
    if raw is not None:
        try:
            return UUID(raw)
        except ValueError:
            pass
    new_id = uuid4()
    response.set_cookie(
        GUEST_COOKIE,
        str(new_id),
        httponly=True,
        secure=True,
        samesite="none",
        max_age=GUEST_COOKIE_MAX_AGE,
    )
    return new_id


def _event_name(chunk: dict) -> str:
    if "disclaimer" in chunk:
        return "final"
    if "reason" in chunk:
        return "withheld"
    if "verified" in chunk:
        return "claim"
    if "stage" in chunk:
        return "stage"
    raise ValueError(f"unrecognised stream event: {chunk!r}")


def _sse(chunk: dict) -> str:
    return f"event: {_event_name(chunk)}\ndata: {json.dumps(chunk)}\n\n"


def _served_citations(events: list) -> tuple[str, ...]:
    seen: list[str] = []
    for event in events:
        if not isinstance(event, ClaimEvent):
            continue
        for citation in event.citations:
            if citation.path not in seen:
                seen.append(citation.path)
    return tuple(seen)


@router.get("/status")
def guest_status_route(
    response: Response,
    conn: psycopg.Connection = Depends(get_conn),  # noqa: B008
    guest_id_raw: str | None = Cookie(default=None, alias=GUEST_COOKIE),
) -> dict[str, int]:
    """Lets the frontend show remaining turns before sending one — minted
    lazily here too, so a visitor who never sends a turn never gets an id."""
    guest_id = _guest_id(response, guest_id_raw)
    used = guest_turns_used(conn, guest_id)
    return {
        "used": used,
        "limit": GUEST_TURN_LIMIT,
        "remaining": max(0, GUEST_TURN_LIMIT - used),
    }


@router.post("/turns")
def create_guest_turn_route(
    body: GuestTurnRequest,
    request: Request,
    response: Response,
    state: AppState = Depends(app_state),  # noqa: B008
    conn: psycopg.Connection = Depends(get_conn),  # noqa: B008
    guest_id_raw: str | None = Cookie(default=None, alias=GUEST_COOKIE),
) -> StreamingResponse:
    guest_id = _guest_id(response, guest_id_raw)
    try:
        record_guest_turn(conn, guest_id, _client_ip(request))
    except GuestLimitReached:
        raise rate_limited() from None

    def event_stream() -> Iterator[str]:
        # R19: per-stage timings for the trace panel (always visible, per user
        # decision) — mirrors build.py's node wrapper for the authenticated
        # graph, since this route composes its own stages rather than going
        # through it (see module docstring).
        trace: list[TraceEntry] = []

        def timed(node: str, fn: Any, *args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            result = fn(*args, **kwargs)
            trace.append(TraceEntry(node=node, ms=round((time.perf_counter() - start) * 1000, 1)))
            return result

        try:
            with request_deps(state) as deps:
                yield _sse(StageEvent(stage="thinking").model_dump())
                result = timed("classify", deps.classifier.classify, body.question)
                if result.category is ScopeCategory.CONVERSATIONAL:
                    reply = timed(
                        "respond_conversational", deps.conversational.reply, body.question
                    )
                    final = FinalEvent(
                        route=result.category.value,
                        computation=None,
                        citations=(),
                        text=reply,
                        trace=tuple(trace),
                    )
                    yield _sse(final.model_dump())
                    return
                if result.category is not ScopeCategory.IN_SCOPE:
                    final = FinalEvent(
                        route=result.category.value,
                        computation=None,
                        citations=(),
                        text=result.response,
                        trace=tuple(trace),
                    )
                    yield _sse(final.model_dump())
                    return
                # getattr, not result.search_query: a test double's classifier
                # stub may predate R19 Phase B (ADR-120) and not set it.
                search_query = getattr(result, "search_query", None) or body.question
                results = timed("retrieve", deps.retriever.search, search_query, deps.pool_k)
                pack = timed("pack", deps.packer.pack, results, expand=False)
                yield _sse(
                    StageEvent(
                        stage="evidence",
                        chunks=tuple(unit.citation for unit in pack.units),
                    ).model_dump()
                )
                generate_start = time.perf_counter()
                events: list = []
                for event in deps.generator.generate(
                    body.question, pack, facts=None, computation=None
                ):
                    yield _sse(event.model_dump())
                    events.append(event)
                trace.append(
                    TraceEntry(
                        node="generate_verify",
                        ms=round((time.perf_counter() - generate_start) * 1000, 1),
                    )
                )
                answer_text = gate(pack, events)
                searched = (
                    tuple(unit.citation for unit in pack.units)
                    if answer_text is not None
                    else ()
                )
                final = FinalEvent(
                    route="guest",
                    computation=None,
                    citations=_served_citations(events),
                    text=answer_text,
                    searched=searched,
                    trace=tuple(trace),
                )
                yield _sse(final.model_dump())
        except Exception:
            logger.exception("guest turn stream failed")
            raise

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
