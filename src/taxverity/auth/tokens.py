"""Step 11.3 — access tokens, refresh rotation and reuse detection (ADR-029).

The access token is a short-lived HS256 JWT signed with `PyJWT`; nothing about
it is stored, so it cannot be revoked and is kept short instead. The refresh
token is 256 random bits, stored only as a sha256, and exchanged for a new one
on every use. Each exchange marks the old token used; presenting a used or
revoked token again means it was copied, so the whole family descended from
that login is revoked and both the thief and the user must log in again.

Expiry of refresh tokens is judged by the database clock, so every Lambda
instance agrees on it.
"""

from __future__ import annotations

import hashlib
import secrets
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

import jwt
import psycopg

from taxverity.config import Settings
from taxverity.observability import get_logger

logger = get_logger(__name__)

ACCESS_TOKEN_TTL = timedelta(minutes=15)
REFRESH_TOKEN_TTL = timedelta(days=30)
ISSUER = "taxverity"
ALGORITHM = "HS256"
# RFC 7518 §3.2: an HS256 key must be at least as long as the hash output.
MIN_SECRET_BYTES = 32


class InvalidToken(Exception):
    """Any token that does not authenticate. Deliberately says nothing more."""

    def __init__(self) -> None:
        super().__init__("invalid token")


class TokenReuse(InvalidToken):
    """A refresh token presented after it was used or revoked. Its family is revoked."""


class AccessTokens:
    def __init__(self, secret: str, *, ttl: timedelta = ACCESS_TOKEN_TTL) -> None:
        if len(secret.encode("utf-8")) < MIN_SECRET_BYTES:
            raise ValueError(
                f"the JWT secret must be at least {MIN_SECRET_BYTES} bytes"
            )
        self._secret = secret
        self._ttl = ttl

    @classmethod
    def from_settings(cls, settings: Settings) -> AccessTokens:
        return cls(settings.require("jwt_secret"))

    def issue(self, user_id: UUID, *, now: datetime | None = None) -> str:
        issued = now or datetime.now(UTC)
        claims = {
            "sub": str(user_id),
            "iss": ISSUER,
            "typ": "access",
            "iat": issued,
            "exp": issued + self._ttl,
        }
        return jwt.encode(claims, self._secret, algorithm=ALGORITHM)

    def verify(self, token: str) -> UUID:
        try:
            # `algorithms` is pinned: accepting the header's own choice is how
            # `alg: none` and key-confusion attacks get in.
            claims = jwt.decode(
                token,
                self._secret,
                algorithms=[ALGORITHM],
                issuer=ISSUER,
                options={"require": ["sub", "iss", "iat", "exp", "typ"]},
            )
            if claims["typ"] != "access":
                raise InvalidToken()
            return UUID(claims["sub"])
        except (jwt.PyJWTError, ValueError, TypeError):
            raise InvalidToken() from None


@dataclass(frozen=True)
class Rotation:
    user_id: UUID
    refresh_token: str


def _digest(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def issue_refresh_token(
    conn: psycopg.Connection,
    user_id: UUID,
    *,
    family_id: UUID | None = None,
    ttl: timedelta = REFRESH_TOKEN_TTL,
) -> str:
    """A new refresh token; a new family unless one is given (a fresh login)."""
    token = secrets.token_urlsafe(32)
    conn.execute(
        "INSERT INTO refresh_tokens (user_id, family_id, token_sha256, expires_at) "
        "VALUES (%s, %s, %s, now() + %s)",
        (user_id, family_id or uuid4(), _digest(token), ttl),
    )
    return token


def rotate_refresh_token(
    conn: psycopg.Connection, token: str, *, ttl: timedelta = REFRESH_TOKEN_TTL
) -> Rotation:
    if not conn.autocommit:
        # A reuse raises after revoking the family; inside a caller's
        # transaction that raise would roll the revocation back.
        raise ValueError("rotate_refresh_token needs an autocommit connection")
    reused: tuple[UUID, UUID] | None = None
    with conn.transaction():
        # FOR UPDATE serialises two exchanges of one token: the second sees
        # used_at set and is treated as the reuse it is.
        row = conn.execute(
            "SELECT user_id, family_id, used_at IS NOT NULL OR revoked_at IS NOT NULL, "
            "expires_at <= now() "
            "FROM refresh_tokens WHERE token_sha256 = %s FOR UPDATE",
            (_digest(token),),
        ).fetchone()
        if row is None:
            raise InvalidToken()
        user_id, family_id, spent, expired = row
        if spent:
            conn.execute(
                "UPDATE refresh_tokens SET revoked_at = now() "
                "WHERE family_id = %s AND revoked_at IS NULL",
                (family_id,),
            )
            reused = (user_id, family_id)
        elif expired:
            raise InvalidToken()
        else:
            conn.execute(
                "UPDATE refresh_tokens SET used_at = now() WHERE token_sha256 = %s",
                (_digest(token),),
            )
            successor = issue_refresh_token(conn, user_id, family_id=family_id, ttl=ttl)
    if reused is not None:
        logger.warning(
            "refresh token reuse: revoked family %s of user %s", reused[1], reused[0]
        )
        raise TokenReuse()
    return Rotation(user_id=user_id, refresh_token=successor)


def revoke_refresh_family(conn: psycopg.Connection, token: str) -> None:
    """Logout. Revokes every token of the presented token's family; an unknown
    token is a no-op, so logout never tells a caller whether a token existed."""
    conn.execute(
        "UPDATE refresh_tokens SET revoked_at = now() "
        "WHERE revoked_at IS NULL AND family_id = "
        "(SELECT family_id FROM refresh_tokens WHERE token_sha256 = %s)",
        (_digest(token),),
    )
