"""Step 14.5 — `POST /v1/threads/{thread_id}/turns`, the real caller of
`request_deps` and the compiled graph. This is the SSE contract rule 04
defines (`stage`/`clarify`/`claim`/`withheld`/`final`), served as a
`StreamingResponse` over `graph.stream(..., stream_mode=["custom"])`.

Ownership and the daily turn cap (Step 14.6) are checked with a short-lived
pooled connection *before* the stream starts, so a foreign or nonexistent
thread id and a capped user both get an ordinary HTTP error rather than a
stream that opens and then dies. The graph itself needs its own connection
open for the whole response body, past the point this route function
returns — `request_deps(state)` is called directly inside the generator
rather than through `Depends(get_conn)`, because FastAPI closes a `yield`
dependency's connection as soon as the endpoint function returns the
`StreamingResponse` object, before the body iterator (and this generator)
actually runs.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from uuid import UUID

import psycopg
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from taxverity.api.app import AppState, app_state, request_deps
from taxverity.api.deps import current_user, get_conn
from taxverity.api.limits import DailyLimitReached, check_daily_turn_limit
from taxverity.graph.build import build_graph
from taxverity.observability import get_logger
from taxverity.threads.store import ThreadNotFound, get_thread

logger = get_logger(__name__)

router = APIRouter(prefix="/v1/threads", tags=["turns"])


class TurnRequest(BaseModel):
    question: str = Field(min_length=1)


def _event_name(chunk: dict) -> str:
    if "disclaimer" in chunk:
        return "final"
    if "reason" in chunk:
        return "withheld"
    if "verified" in chunk:
        return "claim"
    if "questions" in chunk:
        return "clarify"
    if "stage" in chunk:
        return "stage"
    raise ValueError(f"unrecognised stream event: {chunk!r}")


def _sse(chunk: dict) -> str:
    return f"event: {_event_name(chunk)}\ndata: {json.dumps(chunk)}\n\n"


@router.post("/{thread_id}/turns")
def create_turn_route(
    thread_id: UUID,
    body: TurnRequest,
    user_id: UUID = Depends(current_user),  # noqa: B008
    state: AppState = Depends(app_state),  # noqa: B008
    conn: psycopg.Connection = Depends(get_conn),  # noqa: B008
) -> StreamingResponse:
    try:
        get_thread(conn, user_id, thread_id)
    except ThreadNotFound:
        raise HTTPException(status_code=404, detail="not_found") from None
    try:
        check_daily_turn_limit(conn, user_id)
    except DailyLimitReached:
        raise HTTPException(status_code=429, detail="rate_limited") from None

    def event_stream() -> Iterator[str]:
        try:
            with request_deps(state) as deps:
                graph = build_graph(deps)
                for _mode, chunk in graph.stream(
                    {"user_id": user_id, "thread_id": thread_id, "question": body.question},
                    stream_mode=["custom"],
                ):
                    yield _sse(chunk)
        except Exception:
            logger.exception("turn stream failed for thread %s", thread_id)
            raise

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
