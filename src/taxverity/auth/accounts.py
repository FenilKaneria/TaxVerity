"""Step 11.2 — registration and login, with DB-backed rate limiting.

Login fails with one error whatever went wrong: no account, wrong password.
An unknown address is verified against a dummy hash so it takes as long as a
wrong password. Attempts are counted per address and per IP in
`login_attempts`, because Lambda runs many instances and a counter in memory
would reset with each one.

Registration is different: refusing a duplicate address necessarily tells the
caller the address is taken. Closing that needs an email-verification flow,
which this project does not build; registration is rate limited by the API
(Phase 14) instead. Recorded, not hidden.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import timedelta
from uuid import UUID

import psycopg
from psycopg import errors

from taxverity.auth.passwords import (
    dummy_hash,
    hash_password,
    needs_rehash,
    verify_password,
)
from taxverity.observability import get_logger

logger = get_logger(__name__)

LOGIN_WINDOW = timedelta(minutes=15)
# Failures inside the window before further attempts are refused. The IP limit
# is looser: many people can share one address behind a NAT.
MAX_FAILURES_PER_EMAIL = 5
MAX_FAILURES_PER_IP = 20

MAX_EMAIL_LENGTH = 254
EMAIL_SHAPE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


class InvalidEmail(ValueError):
    """Safe to show the user."""


class RegistrationRefused(Exception):
    """The address is already registered."""


class LoginFailed(Exception):
    """Wrong address or wrong password. Deliberately says nothing more."""

    def __init__(self) -> None:
        super().__init__("invalid email or password")


class LoginThrottled(Exception):
    def __init__(self) -> None:
        super().__init__("too many failed attempts; try again later")


@dataclass(frozen=True)
class Account:
    user_id: UUID
    email: str


def normalise_email(email: str) -> str:
    normalised = email.strip().lower()
    if len(normalised) > MAX_EMAIL_LENGTH or not EMAIL_SHAPE.match(normalised):
        raise InvalidEmail("not a valid email address")
    return normalised


def register(conn: psycopg.Connection, email: str, password: str) -> Account:
    _require_autocommit(conn)
    address = normalise_email(email)
    password_hash = hash_password(password)
    try:
        with conn.transaction():
            (user_id,) = conn.execute(
                "INSERT INTO users (email, password_hash) VALUES (%s, %s) "
                "RETURNING user_id",
                (address, password_hash),
            ).fetchone()
    except errors.UniqueViolation:
        logger.warning("registration refused: address already registered")
        raise RegistrationRefused("registration refused") from None
    logger.info("registered user %s", user_id)
    return Account(user_id=user_id, email=address)


def authenticate(
    conn: psycopg.Connection, email: str, password: str, *, ip: str
) -> Account:
    _require_autocommit(conn)
    try:
        address = normalise_email(email)
    except InvalidEmail:
        raise LoginFailed() from None

    if _throttled(conn, address, ip):
        logger.warning("login throttled")
        raise LoginThrottled()

    row = conn.execute(
        "SELECT user_id, password_hash FROM users WHERE email = %s", (address,)
    ).fetchone()
    if row is None:
        verify_password(dummy_hash(), password)
        ok = False
    else:
        ok = verify_password(row[1], password)

    conn.execute(
        "INSERT INTO login_attempts (email, ip, succeeded) VALUES (%s, %s, %s)",
        (address, ip, ok),
    )
    if not ok:
        logger.warning("login refused")
        raise LoginFailed()

    user_id, stored = row
    if needs_rehash(stored):
        conn.execute(
            "UPDATE users SET password_hash = %s WHERE user_id = %s",
            (hash_password(password), user_id),
        )
    return Account(user_id=user_id, email=address)


def _require_autocommit(conn: psycopg.Connection) -> None:
    # A failed login raises after recording its attempt. Inside a caller's
    # transaction that raise would roll the record back, and the rate limit
    # would count nothing.
    if not conn.autocommit:
        raise ValueError("accounts need an autocommit connection")


def _throttled(conn: psycopg.Connection, address: str, ip: str) -> bool:
    by_email, by_ip = conn.execute(
        "SELECT "
        "count(*) FILTER (WHERE email = %(email)s), "
        "count(*) FILTER (WHERE ip = %(ip)s) "
        "FROM login_attempts "
        "WHERE NOT succeeded AND attempted_at > now() - %(window)s "
        "AND (email = %(email)s OR ip = %(ip)s)",
        {"email": address, "ip": ip, "window": LOGIN_WINDOW},
    ).fetchone()
    return by_email >= MAX_FAILURES_PER_EMAIL or by_ip >= MAX_FAILURES_PER_IP
