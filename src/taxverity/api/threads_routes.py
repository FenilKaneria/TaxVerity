"""Step 14.3 — thread CRUD and the facts-panel edit, both user-scoped.

Every `threads.store` and `memory.fact_state` call already filters by the
authenticated `user_id` (Step 11.4/11.5's IDOR discipline); this router adds
no isolation logic of its own; it only translates `ThreadNotFound` into a 404
so a foreign thread id reads exactly like a nonexistent one over HTTP too.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

import psycopg
from fastapi import APIRouter, Depends, status
from pydantic import BaseModel, ConfigDict

from taxverity.api.deps import current_user, get_conn
from taxverity.api.errors import invalid_request, not_found
from taxverity.facts import FactField
from taxverity.memory.fact_state import (
    apply_user_edit,
    load_fact_state,
    save_fact_state,
    to_json,
)
from taxverity.threads.store import (
    Role,
    ThreadNotFound,
    create_thread,
    delete_thread,
    get_thread,
    list_messages,
    list_threads,
    rename_thread,
)

router = APIRouter(prefix="/v1/threads", tags=["threads"])


class ThreadOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    thread_id: UUID
    title: str
    created_at: datetime
    updated_at: datetime


class CitationOut(BaseModel):
    path: str
    quote: str | None = None


class MessageOut(BaseModel):
    message_id: int
    role: Role
    content: str
    citations: list[CitationOut]
    created_at: datetime


class CreateThreadRequest(BaseModel):
    title: str


class RenameThreadRequest(BaseModel):
    title: str


class FactEditRequest(BaseModel):
    field: FactField
    raw_value: str


@router.post("", status_code=status.HTTP_201_CREATED)
def create_thread_route(
    body: CreateThreadRequest,
    user_id: UUID = Depends(current_user),  # noqa: B008
    conn: psycopg.Connection = Depends(get_conn),  # noqa: B008
) -> ThreadOut:
    try:
        thread = create_thread(conn, user_id, body.title)
    except ValueError as error:
        raise invalid_request(str(error)) from None
    return ThreadOut.model_validate(thread)


@router.get("")
def list_threads_route(
    user_id: UUID = Depends(current_user),  # noqa: B008
    conn: psycopg.Connection = Depends(get_conn),  # noqa: B008
) -> list[ThreadOut]:
    return [ThreadOut.model_validate(thread) for thread in list_threads(conn, user_id)]


@router.get("/{thread_id}")
def get_thread_route(
    thread_id: UUID,
    user_id: UUID = Depends(current_user),  # noqa: B008
    conn: psycopg.Connection = Depends(get_conn),  # noqa: B008
) -> ThreadOut:
    try:
        return ThreadOut.model_validate(get_thread(conn, user_id, thread_id))
    except ThreadNotFound:
        raise not_found() from None


@router.patch("/{thread_id}")
def rename_thread_route(
    thread_id: UUID,
    body: RenameThreadRequest,
    user_id: UUID = Depends(current_user),  # noqa: B008
    conn: psycopg.Connection = Depends(get_conn),  # noqa: B008
) -> ThreadOut:
    try:
        thread = rename_thread(conn, user_id, thread_id, body.title)
    except ThreadNotFound:
        raise not_found() from None
    except ValueError as error:
        raise invalid_request(str(error)) from None
    return ThreadOut.model_validate(thread)


@router.delete("/{thread_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_thread_route(
    thread_id: UUID,
    user_id: UUID = Depends(current_user),  # noqa: B008
    conn: psycopg.Connection = Depends(get_conn),  # noqa: B008
) -> None:
    try:
        delete_thread(conn, user_id, thread_id)
    except ThreadNotFound:
        raise not_found() from None


def _citations(payload: dict[str, Any]) -> list[CitationOut]:
    """A pre-quote message (`payload["citations"]` as bare path strings) still
    needs to render — as a path with no quote, not a crash."""
    return [
        CitationOut(path=entry, quote=None) if isinstance(entry, str) else CitationOut(**entry)
        for entry in payload.get("citations", [])
    ]


@router.get("/{thread_id}/messages")
def list_messages_route(
    thread_id: UUID,
    user_id: UUID = Depends(current_user),  # noqa: B008
    conn: psycopg.Connection = Depends(get_conn),  # noqa: B008
) -> list[MessageOut]:
    try:
        messages = list_messages(conn, user_id, thread_id)
    except ThreadNotFound:
        raise not_found() from None
    return [
        MessageOut(
            message_id=message.message_id,
            role=message.role,
            content=message.content,
            citations=_citations(dict(message.payload)),
            created_at=message.created_at,
        )
        for message in messages
    ]


@router.get("/{thread_id}/facts")
def get_facts_route(
    thread_id: UUID,
    user_id: UUID = Depends(current_user),  # noqa: B008
    conn: psycopg.Connection = Depends(get_conn),  # noqa: B008
) -> dict[str, Any]:
    try:
        return to_json(load_fact_state(conn, user_id, thread_id))
    except ThreadNotFound:
        raise not_found() from None


@router.patch("/{thread_id}/facts")
def edit_facts_route(
    thread_id: UUID,
    body: FactEditRequest,
    user_id: UUID = Depends(current_user),  # noqa: B008
    conn: psycopg.Connection = Depends(get_conn),  # noqa: B008
) -> dict[str, Any]:
    try:
        state = load_fact_state(conn, user_id, thread_id)
    except ThreadNotFound:
        raise not_found() from None
    try:
        state = apply_user_edit(state, body.field, body.raw_value)
    except ValueError as error:
        raise invalid_request(str(error)) from None
    save_fact_state(conn, user_id, thread_id, state)
    return to_json(state)
