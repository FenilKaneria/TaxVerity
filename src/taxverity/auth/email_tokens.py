"""Step 11.4d — single-use tokens for email verification and password reset.

Same shape as `auth.tokens`'s refresh tokens: 256 random bits, only the sha256
stored, and consumed exactly once. A token that fails for any reason — wrong
purpose, expired, already used, or simply unknown — raises the same
`InvalidToken` a caller cannot use to tell those apart.
"""

from __future__ import annotations

import hashlib
import secrets
from datetime import timedelta
from typing import Literal
from uuid import UUID

import psycopg

from taxverity.auth.tokens import InvalidToken

Purpose = Literal["verify", "reset"]

VERIFY_TOKEN_TTL = timedelta(hours=24)
RESET_TOKEN_TTL = timedelta(minutes=30)

_TTL_BY_PURPOSE: dict[Purpose, timedelta] = {
    "verify": VERIFY_TOKEN_TTL,
    "reset": RESET_TOKEN_TTL,
}


def _digest(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def issue_email_token(
    conn: psycopg.Connection, user_id: UUID, purpose: Purpose
) -> str:
    token = secrets.token_urlsafe(32)
    conn.execute(
        "INSERT INTO email_tokens (user_id, purpose, token_sha256, expires_at) "
        "VALUES (%s, %s, %s, now() + %s)",
        (user_id, purpose, _digest(token), _TTL_BY_PURPOSE[purpose]),
    )
    return token


def consume_email_token(
    conn: psycopg.Connection, token: str, purpose: Purpose
) -> UUID:
    """Locks and marks the token used in one transaction, so two concurrent
    redemptions of one link cannot both succeed. Raises `InvalidToken` unless
    the token exists, matches `purpose`, is unused and unexpired."""
    if not conn.autocommit:
        raise ValueError("consume_email_token needs an autocommit connection")
    with conn.transaction():
        row = conn.execute(
            "SELECT token_id, user_id, purpose, used_at IS NOT NULL, "
            "expires_at <= now() "
            "FROM email_tokens WHERE token_sha256 = %s FOR UPDATE",
            (_digest(token),),
        ).fetchone()
        if row is None:
            raise InvalidToken()
        token_id, user_id, stored_purpose, used, expired = row
        if stored_purpose != purpose or used or expired:
            raise InvalidToken()
        conn.execute(
            "UPDATE email_tokens SET used_at = now() WHERE token_id = %s",
            (token_id,),
        )
    return user_id
