"""Step 14.6 — the per-user daily turn cap and the request body size cap.

The turn cap lives in Postgres, not memory: Lambda runs many instances, so an
in-memory counter would not hold across them (rule 04). It counts recent
user-role `messages` rows rather than a separate counter table — that row is
already written by `finalize` for every served turn, so there is nothing new
to keep in sync. Both caps exist to protect the Groq and Jina free-tier quota
(Step 7.1 measured ~1.7 evidence-pack queries a minute) from one runaway
caller, not to meter a paid plan.
"""

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
