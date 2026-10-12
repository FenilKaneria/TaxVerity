"""The per-user daily turn cap and the request body size cap."""

from __future__ import annotations

from uuid import UUID

import psycopg

DAILY_TURN_LIMIT = 50
MAX_BODY_BYTES = 32_768


class DailyLimitReached(Exception):
    def __init__(self) -> None:
        super().__init__("daily turn limit reached")


def check_daily_turn_limit(conn: psycopg.Connection, user_id: UUID) -> None:
    count = conn.execute(
        "SELECT count(*) FROM messages WHERE user_id = %s AND role = 'user' "
        "AND created_at > now() - interval '24 hours'",
        (user_id,),
    ).fetchone()[0]
    if count >= DAILY_TURN_LIMIT:
        raise DailyLimitReached()
