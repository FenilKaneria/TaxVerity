"""Step 11.2 — login rate limiting in Postgres, per address and per IP."""

from __future__ import annotations

import pytest

from taxverity.auth.accounts import (
    LOGIN_WINDOW,
    MAX_FAILURES_PER_EMAIL,
    MAX_FAILURES_PER_IP,
    LoginFailed,
    LoginThrottled,
    authenticate,
    register,
)

PASSWORD = "correct horse battery"


def fail(conn, email="alice@example.com", ip="1.1.1.1", times=1):
    for _ in range(times):
        with pytest.raises(LoginFailed):
            authenticate(conn, email, "wrong password", ip=ip)


def test_the_address_is_throttled_after_its_failure_limit(schema):
    register(schema, "alice@example.com", PASSWORD)
    fail(schema, times=MAX_FAILURES_PER_EMAIL)
    # Even the right password from a fresh IP is refused once throttled, so
    # guessing cannot continue by rotating addresses.
    with pytest.raises(LoginThrottled):
        authenticate(schema, "alice@example.com", PASSWORD, ip="9.9.9.9")


def test_one_below_the_limit_still_logs_in(schema):
    register(schema, "alice@example.com", PASSWORD)
    fail(schema, times=MAX_FAILURES_PER_EMAIL - 1)
    authenticate(schema, "alice@example.com", PASSWORD, ip="1.1.1.1")


def test_an_address_with_no_account_is_throttled_the_same_way(schema):
    fail(schema, email="nobody@example.com", times=MAX_FAILURES_PER_EMAIL)
    with pytest.raises(LoginThrottled):
        authenticate(schema, "nobody@example.com", PASSWORD, ip="9.9.9.9")


def test_the_address_limit_spans_spellings(schema):
    register(schema, "alice@example.com", PASSWORD)
    fail(schema, email="ALICE@example.com", times=MAX_FAILURES_PER_EMAIL)
    with pytest.raises(LoginThrottled):
        authenticate(schema, "alice@example.com", PASSWORD, ip="9.9.9.9")


def test_one_throttled_address_does_not_throttle_another(schema):
    register(schema, "bob@example.com", PASSWORD)
    fail(schema, email="alice@example.com", ip="1.1.1.1", times=MAX_FAILURES_PER_EMAIL)
    authenticate(schema, "bob@example.com", PASSWORD, ip="2.2.2.2")


def test_the_ip_is_throttled_across_many_addresses(schema):
    register(schema, "victim@example.com", PASSWORD)
    for n in range(MAX_FAILURES_PER_IP):
        fail(schema, email=f"user{n}@example.com", ip="6.6.6.6")
    with pytest.raises(LoginThrottled):
        authenticate(schema, "victim@example.com", PASSWORD, ip="6.6.6.6")
    authenticate(schema, "victim@example.com", PASSWORD, ip="7.7.7.7")


def test_failures_outside_the_window_do_not_count(schema):
    register(schema, "alice@example.com", PASSWORD)
    fail(schema, times=MAX_FAILURES_PER_EMAIL)
    schema.execute(
        "UPDATE login_attempts SET attempted_at = now() - %s - interval '1 second'",
        (LOGIN_WINDOW,),
    )
    authenticate(schema, "alice@example.com", PASSWORD, ip="1.1.1.1")


def test_a_success_is_recorded_and_a_throttled_attempt_is_not(schema):
    register(schema, "alice@example.com", PASSWORD)
    authenticate(schema, "alice@example.com", PASSWORD, ip="1.1.1.1")
    fail(schema, times=MAX_FAILURES_PER_EMAIL)
    with pytest.raises(LoginThrottled):
        authenticate(schema, "alice@example.com", PASSWORD, ip="1.1.1.1")
    rows = schema.execute(
        "SELECT succeeded FROM login_attempts ORDER BY attempt_id"
    ).fetchall()
    assert rows == [(True,)] + [(False,)] * MAX_FAILURES_PER_EMAIL


def test_no_password_is_ever_stored_in_attempts(schema):
    fail(schema, times=2)
    dump = schema.execute(
        "SELECT row_to_json(a)::text FROM login_attempts a"
    ).fetchall()
    assert all("wrong password" not in row[0] for row in dump)
