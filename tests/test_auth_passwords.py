"""Step 11.2/11.4 — argon2id hashing, registration, and generic login errors.

Registration must not reveal whether an address is already taken (rule 03),
so both branches here are read off the `NullMailer`'s recorded sends rather
than off `register()`'s return value — it always returns `None`.
"""

from __future__ import annotations

import time

import psycopg
import pytest
from argon2 import PasswordHasher
from psycopg import errors
from psycopg.conninfo import make_conninfo

from conftest import register_account
from taxverity.auth.accounts import (
    InvalidEmail,
    LoginFailed,
    authenticate,
    normalise_email,
    register,
)
from taxverity.auth.passwords import (
    MAX_PASSWORD_LENGTH,
    WeakPassword,
    dummy_hash,
    hash_password,
    needs_rehash,
    verify_password,
)
from taxverity.mail.gmail import NullMailer

PASSWORD = "correct horse battery"
BASE_URL = "http://t"


def _register(conn, email, password, ip="1.1.1.1", mailer=None):
    mailer = mailer or NullMailer()
    register(conn, email, password, ip=ip, mailer=mailer, base_url=BASE_URL)
    return mailer


def test_hash_is_argon2id_and_salted():
    first, second = hash_password(PASSWORD), hash_password(PASSWORD)
    assert first.startswith("$argon2id$")
    assert first != second
    assert PASSWORD not in first


def test_verify_accepts_the_password_and_nothing_else():
    stored = hash_password(PASSWORD)
    assert verify_password(stored, PASSWORD)
    assert not verify_password(stored, PASSWORD + " ")
    assert not verify_password(stored, PASSWORD.upper())
    assert not verify_password(stored, "")


def test_verify_never_raises_on_a_malformed_hash():
    assert not verify_password("not-a-hash", PASSWORD)
    assert not verify_password("$argon2id$v=19$garbage", PASSWORD)


def test_policy_bounds_length_both_ways():
    with pytest.raises(WeakPassword):
        hash_password("short")
    with pytest.raises(WeakPassword):
        hash_password("x" * (MAX_PASSWORD_LENGTH + 1))
    too_long = "x" * (MAX_PASSWORD_LENGTH + 1)
    assert not verify_password(hash_password(PASSWORD), too_long)


def test_a_current_hash_needs_no_rehash():
    assert not needs_rehash(hash_password(PASSWORD))


def test_a_weaker_hash_is_flagged_for_rehash():
    weak = PasswordHasher(time_cost=1, memory_cost=8, parallelism=1).hash(PASSWORD)
    assert needs_rehash(weak)


def test_email_is_normalised_and_validated():
    assert normalise_email("  Alice@Example.COM ") == "alice@example.com"
    for bad in ("", "alice", "alice@", "@example.com", "a b@example.com", "a@b"):
        with pytest.raises(InvalidEmail):
            normalise_email(bad)


def test_register_sends_a_verification_link_and_creates_no_session(schema):
    mailer = _register(schema, "Alice@Example.com", PASSWORD)
    assert len(mailer.sent) == 1
    assert mailer.sent[0].to == "alice@example.com"
    assert "verify" in mailer.sent[0].body
    row = schema.execute(
        "SELECT email_verified_at FROM users WHERE email = 'alice@example.com'"
    ).fetchone()
    assert row[0] is None


def test_register_then_verify_then_login(schema):
    user_id = register_account(schema, "alice@example.com", PASSWORD)
    account = authenticate(schema, "alice@example.com ", PASSWORD, ip="1.1.1.1")
    assert account.user_id == user_id
    assert account.email == "alice@example.com"


def test_an_unverified_account_cannot_log_in(schema):
    _register(schema, "alice@example.com", PASSWORD)
    with pytest.raises(LoginFailed):
        authenticate(schema, "alice@example.com", PASSWORD, ip="1.1.1.1")


def test_a_second_spelling_of_an_address_gets_a_reset_link_not_a_second_account(
    schema,
):
    _register(schema, "alice@example.com", PASSWORD)
    mailer = _register(schema, "ALICE@example.com", "another password")
    assert schema.execute("SELECT count(*) FROM users").fetchone()[0] == 1
    assert len(mailer.sent) == 1
    assert "reset" in mailer.sent[0].body


def test_a_weak_password_is_refused_before_anything_is_stored(schema):
    with pytest.raises(WeakPassword):
        _register(schema, "alice@example.com", "short")
    assert schema.execute("SELECT count(*) FROM users").fetchone()[0] == 0


def test_the_schema_refuses_an_unnormalised_email_or_a_non_argon2_hash(schema):
    with pytest.raises(errors.CheckViolation):
        schema.execute(
            "INSERT INTO users (email, password_hash) VALUES (%s, %s)",
            ("Alice@example.com", dummy_hash()),
        )
    with pytest.raises(errors.CheckViolation):
        schema.execute(
            "INSERT INTO users (email, password_hash) VALUES (%s, %s)",
            ("bob@example.com", "plaintext"),
        )


def test_wrong_password_and_unknown_account_fail_identically(schema):
    register_account(schema, "alice@example.com", PASSWORD)
    with pytest.raises(LoginFailed) as wrong:
        authenticate(schema, "alice@example.com", "wrong password", ip="1.1.1.1")
    with pytest.raises(LoginFailed) as unknown:
        authenticate(schema, "nobody@example.com", PASSWORD, ip="1.1.1.1")
    with pytest.raises(LoginFailed) as malformed:
        authenticate(schema, "not-an-email", PASSWORD, ip="1.1.1.1")
    assert str(wrong.value) == str(unknown.value) == str(malformed.value)
    assert type(wrong.value) is type(unknown.value) is type(malformed.value)


def test_an_unverified_account_fails_identically_to_a_wrong_password(schema):
    _register(schema, "alice@example.com", PASSWORD)
    with pytest.raises(LoginFailed) as unverified:
        authenticate(schema, "alice@example.com", PASSWORD, ip="1.1.1.1")
    with pytest.raises(LoginFailed) as wrong:
        authenticate(schema, "alice@example.com", "wrong password", ip="1.1.1.1")
    assert str(unverified.value) == str(wrong.value)
    assert type(unverified.value) is type(wrong.value)


def test_an_unknown_account_still_pays_for_a_hash(schema):
    register_account(schema, "alice@example.com", PASSWORD)
    dummy_hash()

    def timed(email):
        started = time.perf_counter()
        with pytest.raises(LoginFailed):
            authenticate(schema, email, "wrong password", ip="2.2.2.2")
        return time.perf_counter() - started

    known = min(timed("alice@example.com") for _ in range(2))
    unknown = min(timed("nobody@example.com") for _ in range(2))
    # Without the dummy verify an unknown address returns in about a
    # millisecond, against tens of milliseconds for a real hash.
    assert unknown > known * 0.5


def test_accounts_need_an_autocommit_connection(schema, admin_url):
    url = make_conninfo(admin_url, dbname=schema.info.dbname)
    with psycopg.connect(url) as conn:
        with pytest.raises(ValueError, match="autocommit"):
            authenticate(conn, "alice@example.com", PASSWORD, ip="1.1.1.1")
        with pytest.raises(ValueError, match="autocommit"):
            register(
                conn,
                "alice@example.com",
                PASSWORD,
                ip="1.1.1.1",
                mailer=NullMailer(),
                base_url=BASE_URL,
            )
