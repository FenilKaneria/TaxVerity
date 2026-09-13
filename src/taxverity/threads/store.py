"""Step 11.4 — thread and message persistence, isolated per user.

Every function takes the acting user's id and every query filters by it. A
thread that does not exist and a thread owned by someone else raise the same
`ThreadNotFound`, so a guessed id tells the caller nothing. The schema backs
this up: a message points at its thread through (thread_id, user_id), so a
query that forgot its filter still could not attach a message to another
user's thread.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal
from uuid import UUID

import psycopg
from psycopg.types.json import Jsonb

MAX_TITLE_LENGTH = 200

Role = Literal["user", "assistant"]


class ThreadNotFound(LookupError):
    def __init__(self) -> None:
        super().__init__("thread not found")


@dataclass(frozen=True)
class Thread:
    thread_id: UUID
    user_id: UUID
    title: str
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True)
class Message:
    message_id: int
    thread_id: UUID
    role: Role
    content: str
    payload: Mapping[str, Any]
    created_at: datetime


_THREAD_COLUMNS = "thread_id, user_id, title, created_at, updated_at"
_MESSAGE_COLUMNS = "message_id, thread_id, role, content, payload, created_at"


def _title(title: str) -> str:
    cleaned = " ".join(title.split())[:MAX_TITLE_LENGTH]
    if not cleaned:
        raise ValueError("title must be non-empty")
    return cleaned


def create_thread(conn: psycopg.Connection, user_id: UUID, title: str) -> Thread:
    row = conn.execute(
        f"INSERT INTO threads (user_id, title) VALUES (%s, %s) RETURNING {_THREAD_COLUMNS}",
        (user_id, _title(title)),
    ).fetchone()
    return Thread(*row)


def list_threads(conn: psycopg.Connection, user_id: UUID) -> list[Thread]:
    rows = conn.execute(
        f"SELECT {_THREAD_COLUMNS} FROM threads WHERE user_id = %s "
        "ORDER BY updated_at DESC, thread_id",
        (user_id,),
    ).fetchall()
    return [Thread(*row) for row in rows]


def get_thread(conn: psycopg.Connection, user_id: UUID, thread_id: UUID) -> Thread:
    row = conn.execute(
        f"SELECT {_THREAD_COLUMNS} FROM threads WHERE thread_id = %s AND user_id = %s",
        (thread_id, user_id),
    ).fetchone()
    if row is None:
        raise ThreadNotFound()
    return Thread(*row)


def rename_thread(
    conn: psycopg.Connection, user_id: UUID, thread_id: UUID, title: str
) -> Thread:
    row = conn.execute(
        f"UPDATE threads SET title = %s, updated_at = now() "
        f"WHERE thread_id = %s AND user_id = %s RETURNING {_THREAD_COLUMNS}",
        (_title(title), thread_id, user_id),
    ).fetchone()
    if row is None:
        raise ThreadNotFound()
    return Thread(*row)


def delete_thread(conn: psycopg.Connection, user_id: UUID, thread_id: UUID) -> None:
    """Hard delete; messages and fact state cascade."""
    deleted = conn.execute(
        "DELETE FROM threads WHERE thread_id = %s AND user_id = %s",
        (thread_id, user_id),
    ).rowcount
    if deleted == 0:
        raise ThreadNotFound()


def append_message(
    conn: psycopg.Connection,
    user_id: UUID,
    thread_id: UUID,
    role: Role,
    content: str,
    payload: Mapping[str, Any] | None = None,
) -> Message:
    # The ownership check and the insert are one statement, so there is no
    # window between checking a thread and writing into it.
    row = conn.execute(
        f"WITH owned AS ("
        f"  UPDATE threads SET updated_at = now() "
        f"  WHERE thread_id = %s AND user_id = %s RETURNING thread_id, user_id"
        f") "
        f"INSERT INTO messages (thread_id, user_id, role, content, payload) "
        f"SELECT thread_id, user_id, %s, %s, %s FROM owned "
        f"RETURNING {_MESSAGE_COLUMNS}",
        (thread_id, user_id, role, content, Jsonb(dict(payload or {}))),
    ).fetchone()
    if row is None:
        raise ThreadNotFound()
    return Message(*row)


def list_messages(
    conn: psycopg.Connection,
    user_id: UUID,
    thread_id: UUID,
    *,
    last: int | None = None,
) -> list[Message]:
    """Oldest first. `last` keeps only the most recent n."""
    get_thread(conn, user_id, thread_id)
    if last is not None and last < 1:
        raise ValueError(f"last must be at least 1, not {last}")
    rows = conn.execute(
        f"SELECT {_MESSAGE_COLUMNS} FROM ("
        f"  SELECT {_MESSAGE_COLUMNS} FROM messages "
        f"  WHERE thread_id = %s AND user_id = %s "
        f"  ORDER BY message_id DESC LIMIT %s"
        f") recent ORDER BY message_id",
        (thread_id, user_id, last),
    ).fetchall()
    return [Message(*row) for row in rows]
