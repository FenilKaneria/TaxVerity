"""`POST /v1/threads/{thread_id}/turns`, the real caller of `request_deps` and the
compiled graph."""

from __future__ import annotations

import json
from collections.abc import Iterator
from uuid import UUID

import psycopg
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from taxverity.api.app import AppState, app_state, request_deps
from taxverity.api.deps import current_user, get_conn
from taxverity.api.errors import not_found, rate_limited
from taxverity.api.limits import DailyLimitReached, check_daily_turn_limit
from taxverity.graph.build import build_graph
from taxverity.llm.tracing import flush_traces
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
        raise not_found() from None
    try:
        check_daily_turn_limit(conn, user_id)
    except DailyLimitReached:
        raise rate_limited() from None

    def event_stream() -> Iterator[str]:
        try:
            with request_deps(state) as deps:
                graph = build_graph(deps)
                for _mode, chunk in graph.stream(
                    {
                        "user_id": user_id,
                        "thread_id": thread_id,
                        "question": body.question,
                    },
                    stream_mode=["custom"],
                ):
                    yield _sse(chunk)
        except Exception:
            logger.exception("turn stream failed for thread %s", thread_id)
            raise
        finally:
            flush_traces()

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
