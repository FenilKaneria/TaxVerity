"""Step 11.3 — access JWTs, refresh rotation, reuse detection and logout."""

from __future__ import annotations

import hashlib
import logging
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import jwt
import psycopg
import pytest
from psycopg.conninfo import make_conninfo

from taxverity.auth.accounts import register
from taxverity.auth.tokens import (
    ACCESS_TOKEN_TTL,
    AccessTokens,
    InvalidToken,
    TokenReuse,
    issue_refresh_token,
    revoke_refresh_family,
    rotate_refresh_token,
)
from taxverity.config import MissingSettingError, Settings

SECRET = "s" * 48
PASSWORD = "correct horse battery"


@pytest.fixture
def tokens():
    return AccessTokens(SECRET)


# --- access tokens ---------------------------------------------------------


def test_an_access_token_round_trips_to_its_user(tokens):
    user = uuid4()
    assert tokens.verify(tokens.issue(user)) == user


def test_an_expired_access_token_is_refused(tokens):
    issued = datetime.now(UTC) - ACCESS_TOKEN_TTL - timedelta(seconds=1)
    with pytest.raises(InvalidToken):
        tokens.verify(tokens.issue(uuid4(), now=issued))


def test_an_access_token_just_inside_its_lifetime_is_accepted(tokens):
    issued = datetime.now(UTC) - ACCESS_TOKEN_TTL + timedelta(seconds=30)
    user = uuid4()
    assert tokens.verify(tokens.issue(user, now=issued)) == user


def test_a_token_signed_with_another_key_is_refused(tokens):
    forged = AccessTokens("x" * 48).issue(uuid4())
    with pytest.raises(InvalidToken):
        tokens.verify(forged)


def test_a_tampered_payload_is_refused(tokens):
    header, payload, signature = tokens.issue(uuid4()).split(".")
    other = tokens.issue(uuid4()).split(".")[1]
    with pytest.raises(InvalidToken):
        tokens.verify(f"{header}.{other[:-2]}AA.{signature}")
    with pytest.raises(InvalidToken):
        tokens.verify(f"{header}.{payload}.{signature[:-2]}AA")


def test_alg_none_is_refused(tokens):
    now = datetime.now(UTC)
    unsigned = jwt.encode(
        {"sub": str(uuid4()), "iss": "taxverity", "typ": "access", "iat": now,
         "exp": now + timedelta(minutes=5)},
        key=None,
        algorithm="none",
    )  # fmt: skip
    with pytest.raises(InvalidToken):
        tokens.verify(unsigned)


@pytest.mark.parametrize(
    "claims",
    [
        {"iss": "someone-else", "typ": "access"},
        {"iss": "taxverity", "typ": "refresh"},
        {"iss": "taxverity"},
        {"iss": "taxverity", "typ": "access", "sub": "not-a-uuid"},
    ],
)
def test_wrong_issuer_type_or_subject_is_refused(tokens, claims):
    now = datetime.now(UTC)
    payload = {"sub": str(uuid4()), "iat": now, "exp": now + timedelta(minutes=5)}
    token = jwt.encode({**payload, **claims}, SECRET, algorithm="HS256")
    with pytest.raises(InvalidToken):
        tokens.verify(token)


def test_garbage_is_refused_with_the_same_error(tokens):
    for token in ("", "abc", "a.b.c"):
        with pytest.raises(InvalidToken, match="^invalid token$"):
            tokens.verify(token)


def test_a_short_secret_is_refused():
    with pytest.raises(ValueError, match="32 bytes"):
        AccessTokens("too-short")


def test_from_settings_requires_the_secret(tmp_path):
    empty = tmp_path / ".env"
    empty.write_text("")
    with pytest.raises(MissingSettingError):
        AccessTokens.from_settings(Settings(_env_file=empty, jwt_secret=None))
    assert AccessTokens.from_settings(Settings(_env_file=empty, jwt_secret=SECRET))


# --- refresh tokens --------------------------------------------------------


@pytest.fixture
def user(schema):
    return register(schema, "alice@example.com", PASSWORD).user_id


def test_only_the_hash_of_a_refresh_token_is_stored(schema, user):
    token = issue_refresh_token(schema, user)
    dump = schema.execute(
        "SELECT row_to_json(r)::text FROM refresh_tokens r"
    ).fetchone()[0]
    assert token not in dump
    assert hashlib.sha256(token.encode()).hexdigest() in dump


def test_rotation_returns_a_new_token_for_the_same_user(schema, user):
    first = issue_refresh_token(schema, user)
    rotation = rotate_refresh_token(schema, first)
    assert rotation.user_id == user
    assert rotation.refresh_token != first
    assert rotate_refresh_token(schema, rotation.refresh_token).user_id == user


def test_rotation_keeps_the_family(schema, user):
    first = issue_refresh_token(schema, user)
    rotate_refresh_token(schema, first)
    families = schema.execute(
        "SELECT DISTINCT family_id FROM refresh_tokens"
    ).fetchall()
    assert len(families) == 1


def test_replaying_a_used_token_revokes_the_whole_family(schema, user):
    first = issue_refresh_token(schema, user)
    second = rotate_refresh_token(schema, first).refresh_token

    # The thief replays the token the user already exchanged.
    with pytest.raises(TokenReuse):
        rotate_refresh_token(schema, first)
    # The legitimate successor is dead too; everyone must log in again.
    with pytest.raises(InvalidToken):
        rotate_refresh_token(schema, second)
    live = schema.execute(
        "SELECT count(*) FROM refresh_tokens WHERE revoked_at IS NULL"
    ).fetchone()[0]
    assert live == 0


def test_the_thief_rotating_first_still_trips_reuse(schema, user):
    first = issue_refresh_token(schema, user)
    stolen = rotate_refresh_token(schema, first).refresh_token
    # The user, still holding `first`, tries to refresh.
    with pytest.raises(TokenReuse):
        rotate_refresh_token(schema, first)
    with pytest.raises(InvalidToken):
        rotate_refresh_token(schema, stolen)


def test_reuse_revokes_only_its_own_family(schema, user):
    laptop = issue_refresh_token(schema, user)
    phone = issue_refresh_token(schema, user)
    rotate_refresh_token(schema, laptop)
    with pytest.raises(TokenReuse):
        rotate_refresh_token(schema, laptop)
    assert rotate_refresh_token(schema, phone).user_id == user


def test_reuse_is_logged_without_the_token(schema, user):
    first = issue_refresh_token(schema, user)
    rotate_refresh_token(schema, first)
    records: list[logging.LogRecord] = []
    handler = logging.Handler()
    handler.emit = records.append
    logger = logging.getLogger("taxverity.auth.tokens")
    logger.addHandler(handler)
    try:
        with pytest.raises(TokenReuse):
            rotate_refresh_token(schema, first)
    finally:
        logger.removeHandler(handler)
    assert records and records[0].levelno == logging.WARNING
    assert all(first not in record.getMessage() for record in records)


def test_an_expired_refresh_token_is_refused_and_not_treated_as_reuse(schema, user):
    token = issue_refresh_token(schema, user)
    schema.execute(
        "UPDATE refresh_tokens SET created_at = now() - interval '31 days', "
        "expires_at = now() - interval '1 second'"
    )
    with pytest.raises(InvalidToken) as refused:
        rotate_refresh_token(schema, token)
    assert not isinstance(refused.value, TokenReuse)


def test_an_unknown_refresh_token_is_refused(schema, user):
    with pytest.raises(InvalidToken) as refused:
        rotate_refresh_token(schema, "never-issued")
    assert not isinstance(refused.value, TokenReuse)


def test_logout_revokes_the_family(schema, user):
    first = issue_refresh_token(schema, user)
    current = rotate_refresh_token(schema, first).refresh_token
    revoke_refresh_family(schema, current)
    with pytest.raises(InvalidToken):
        rotate_refresh_token(schema, current)


def test_logout_leaves_other_sessions_alone(schema, user):
    laptop = issue_refresh_token(schema, user)
    phone = issue_refresh_token(schema, user)
    revoke_refresh_family(schema, laptop)
    assert rotate_refresh_token(schema, phone).user_id == user


def test_logout_with_an_unknown_token_is_a_silent_no_op(schema, user):
    token = issue_refresh_token(schema, user)
    revoke_refresh_family(schema, "never-issued")
    assert rotate_refresh_token(schema, token).user_id == user


def test_deleting_a_user_deletes_their_tokens(schema, user):
    issue_refresh_token(schema, user)
    schema.execute("DELETE FROM users WHERE user_id = %s", (user,))
    assert schema.execute("SELECT count(*) FROM refresh_tokens").fetchone()[0] == 0


def test_rotation_needs_an_autocommit_connection(schema, user, admin_url):
    token = issue_refresh_token(schema, user)
    url = make_conninfo(admin_url, dbname=schema.info.dbname)
    with psycopg.connect(url) as conn, pytest.raises(ValueError, match="autocommit"):
        rotate_refresh_token(conn, token)
