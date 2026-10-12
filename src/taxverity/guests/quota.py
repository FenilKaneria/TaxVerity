"""The guest chat trial."""

from __future__ import annotations

from datetime import timedelta
from uuid import UUID

import psycopg

GUEST_TURN_LIMIT = 5
GUEST_IP_WINDOW = timedelta(hours=24)
# Headroom for a shared network (an office, a college) where several
# genuine guests share one address.
GUEST_IP_LIMIT = 15


class GuestLimitReached(Exception):
    def __init__(self) -> None:
        super().__init__("guest turn limit reached; log in or sign up to continue")


def record_guest_turn(conn: psycopg.Connection, guest_id: UUID, ip: str) -> None:
    """Raises `GuestLimitReached` and records nothing if either cap is already met;
    otherwise records the turn."""
    if not conn.autocommit:
        raise ValueError("record_guest_turn needs an autocommit connection")
    with conn.transaction():
        conn.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (str(guest_id),))
        by_guest, by_ip = conn.execute(
            "SELECT "
            "count(*) FILTER (WHERE guest_id = %(guest_id)s), "
            "count(*) FILTER (WHERE ip = %(ip)s AND created_at > now() - %(window)s) "
            "FROM guest_turns WHERE guest_id = %(guest_id)s OR ip = %(ip)s",
            {"guest_id": guest_id, "ip": ip, "window": GUEST_IP_WINDOW},
        ).fetchone()
        if by_guest >= GUEST_TURN_LIMIT or by_ip >= GUEST_IP_LIMIT:
            raise GuestLimitReached()
        conn.execute(
            "INSERT INTO guest_turns (guest_id, ip) VALUES (%s, %s)",
            (guest_id, ip),
        )


def guest_turns_used(conn: psycopg.Connection, guest_id: UUID) -> int:
    return conn.execute(
        "SELECT count(*) FROM guest_turns WHERE guest_id = %s", (guest_id,)
    ).fetchone()[0]
