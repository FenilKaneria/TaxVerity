"""Step 11.2/11.4 — registration, login and the email-verification and
password-reset flows, with DB-backed rate limiting.

Login fails with one error whatever went wrong: no account, wrong password, or
an account that has not yet verified its address. An unknown address is
verified against a dummy hash so it takes as long as a wrong password.
Attempts are counted per address and per IP in `login_attempts`, because
Lambda runs many instances and a counter in memory would reset with each one.

Registration must not tell a caller whether an address is already registered
(rule 03's account-enumeration concern) — `register()` always returns `None`
and always sends mail, so a careless caller cannot brand a response by
branching on a return value the function never gives it. A new address gets a
verification link; an address already in use gets a "you already have an
account, forgot your password?" message instead. Both paths do the same
password-hashing work, so timing does not distinguish them either.

Password reset and resend-verification follow the same shape: always silent,
always the same return, mail sent only when there is actually an account to
mail.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import timedelta
from uuid import UUID

import psycopg
from psycopg import errors

from taxverity.auth.email_tokens import consume_email_token, issue_email_token
from taxverity.auth.passwords import (
    dummy_hash,
    hash_password,
    needs_rehash,
    verify_password,
)
from taxverity.auth.tokens import revoke_all_refresh_tokens
from taxverity.mail.gmail import Mailer
from taxverity.mail.templates import (
    already_registered_body,
    reset_password_body,
    verify_email_body,
)
from taxverity.observability import get_logger

logger = get_logger(__name__)

LOGIN_WINDOW = timedelta(minutes=15)
# Failures inside the window before further attempts are refused. The IP limit
# is looser: many people can share one address behind a NAT.
MAX_FAILURES_PER_EMAIL = 5
MAX_FAILURES_PER_IP = 20

# How much mail one address or one IP can trigger. Registration, resend and
# reset all draw from the same budget — each is a way to make this project
# send mail to an address it does not control.
EMAIL_SEND_WINDOW = timedelta(hours=1)
MAX_SENDS_PER_ADDRESS = 3
MAX_SENDS_PER_IP = 10

MAX_EMAIL_LENGTH = 254
EMAIL_SHAPE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


class InvalidEmail(ValueError):
    """Safe to show the user."""


class LoginFailed(Exception):
    """Wrong address, wrong password, or an unverified account. Deliberately
    says nothing more."""

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


def register(
    conn: psycopg.Connection,
    email: str,
    password: str,
    *,
    ip: str,
    mailer: Mailer,
    base_url: str,
) -> None:
    """Always returns `None`. A new address is created unverified and sent a
    verification link; an address already in use is sent a reset-your-password
    message instead, and nothing new is created. Either way the caller learns
    nothing about which happened."""
    _require_autocommit(conn)
    address = normalise_email(email)
    # Hashed on both branches so a duplicate address costs the same time as a
    # new one.
    password_hash = hash_password(password)
    try:
        with conn.transaction():
            (user_id,) = conn.execute(
                "INSERT INTO users (email, password_hash) VALUES (%s, %s) "
                "RETURNING user_id",
                (address, password_hash),
            ).fetchone()
    except errors.UniqueViolation:
        logger.info("registration for an already-registered address")
        existing = conn.execute(
            "SELECT user_id FROM users WHERE email = %s", (address,)
        ).fetchone()
        if existing is not None and _send_allowed(conn, address, ip):
            token = issue_email_token(conn, existing[0], "reset")
            _record_send(conn, address, ip, "reset")
            mailer.send(
                address,
                "TaxVerity account already exists",
                already_registered_body(f"{base_url}/reset-password?token={token}"),
            )
        return None
    logger.info("registered user %s", user_id)
    if _send_allowed(conn, address, ip):
        token = issue_email_token(conn, user_id, "verify")
        _record_send(conn, address, ip, "verify")
        mailer.send(
            address,
            "Verify your TaxVerity account",
            verify_email_body(f"{base_url}/verify-email?token={token}"),
        )
    return None


def resend_verification(
    conn: psycopg.Connection, email: str, ip: str, mailer: Mailer, base_url: str
) -> None:
    """Always silent: no account, an already-verified account, and a genuinely
    unverified account all return the same nothing."""
    _require_autocommit(conn)
    try:
        address = normalise_email(email)
    except InvalidEmail:
        return None
    row = conn.execute(
        "SELECT user_id FROM users WHERE email = %s AND email_verified_at IS NULL",
        (address,),
    ).fetchone()
    if row is not None and _send_allowed(conn, address, ip):
        token = issue_email_token(conn, row[0], "verify")
        _record_send(conn, address, ip, "verify")
        mailer.send(
            address,
            "Verify your TaxVerity account",
            verify_email_body(f"{base_url}/verify-email?token={token}"),
        )
    return None


def verify_email(conn: psycopg.Connection, token: str) -> None:
    """Raises `InvalidToken` (expired, already used, or unknown) rather than
    saying which — the same generic failure as any other bad token."""
    _require_autocommit(conn)
    user_id = consume_email_token(conn, token, "verify")
    conn.execute(
        "UPDATE users SET email_verified_at = now() WHERE user_id = %s", (user_id,)
    )
    logger.info("verified user %s", user_id)


def request_password_reset(
    conn: psycopg.Connection, email: str, ip: str, mailer: Mailer, base_url: str
) -> None:
    """Always silent. Mail goes out only for a verified account that actually
    exists; an unknown or unverified address gets the same nothing."""
    _require_autocommit(conn)
    try:
        address = normalise_email(email)
    except InvalidEmail:
        return None
    row = conn.execute(
        "SELECT user_id FROM users "
        "WHERE email = %s AND email_verified_at IS NOT NULL",
        (address,),
    ).fetchone()
    if row is not None and _send_allowed(conn, address, ip):
        token = issue_email_token(conn, row[0], "reset")
        _record_send(conn, address, ip, "reset")
        mailer.send(
            address,
            "Reset your TaxVerity password",
            reset_password_body(f"{base_url}/reset-password?token={token}"),
        )
    return None


def reset_password(conn: psycopg.Connection, token: str, new_password: str) -> None:
    """Consumes the reset token, sets the new password, and revokes every
    refresh-token family the account holds — a reset ends every existing
    session, not only the one that requested it."""
    _require_autocommit(conn)
    password_hash = hash_password(new_password)
    user_id = consume_email_token(conn, token, "reset")
    conn.execute(
        "UPDATE users SET password_hash = %s WHERE user_id = %s",
        (password_hash, user_id),
    )
    revoke_all_refresh_tokens(conn, user_id)
    logger.info("password reset for user %s", user_id)


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
        "SELECT user_id, password_hash, email_verified_at IS NOT NULL "
        "FROM users WHERE email = %s",
        (address,),
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

    user_id, stored, verified = row
    if not verified:
        # The attempt above is still recorded as a success (the password was
        # right); refusing here, not earlier, is what keeps this branch from
        # being a second, faster way to test whether a password is correct.
        logger.warning("login refused: address not verified")
        raise LoginFailed()
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


def _send_allowed(conn: psycopg.Connection, address: str, ip: str) -> bool:
    by_email, by_ip = conn.execute(
        "SELECT "
        "count(*) FILTER (WHERE email = %(email)s), "
        "count(*) FILTER (WHERE ip = %(ip)s) "
        "FROM email_sends "
        "WHERE sent_at > now() - %(window)s "
        "AND (email = %(email)s OR ip = %(ip)s)",
        {"email": address, "ip": ip, "window": EMAIL_SEND_WINDOW},
    ).fetchone()
    return by_email < MAX_SENDS_PER_ADDRESS and by_ip < MAX_SENDS_PER_IP


def _record_send(conn: psycopg.Connection, address: str, ip: str, purpose: str) -> None:
    conn.execute(
        "INSERT INTO email_sends (email, ip, purpose) VALUES (%s, %s, %s)",
        (address, ip, purpose),
    )
