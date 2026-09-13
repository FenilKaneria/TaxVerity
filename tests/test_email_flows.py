"""Step 11.4d — email verification, password reset, and the single-use
token store both flows share.

Registration must not tell a caller whether an address is already taken, so
several tests here read the `NullMailer`'s recorded sends rather than a
return value — `register()` always returns `None`.
"""

from __future__ import annotations

import logging

import psycopg
import pytest
from psycopg.conninfo import make_conninfo

from conftest import register_account
from taxverity.auth.accounts import (
    LoginFailed,
    authenticate,
    register,
    request_password_reset,
    resend_verification,
    reset_password,
    verify_email,
)
from taxverity.auth.email_tokens import consume_email_token, issue_email_token
from taxverity.auth.passwords import hash_password
from taxverity.auth.tokens import (
    InvalidToken,
    issue_refresh_token,
    rotate_refresh_token,
)
from taxverity.mail.gmail import NullMailer

PASSWORD = "correct horse battery"
NEW_PASSWORD = "another correct horse"
BASE_URL = "http://t"
LOGGER = "taxverity.auth.accounts"


def _register(conn, email, password=PASSWORD, ip="1.1.1.1", mailer=None):
    mailer = mailer or NullMailer()
    register(conn, email, password, ip=ip, mailer=mailer, base_url=BASE_URL)
    return mailer


def _user_id(conn, email):
    row = conn.execute(
        "SELECT user_id FROM users WHERE email = %s", (email,)
    ).fetchone()
    return row[0]


# --- email tokens ------------------------------------------------------------


def test_a_verify_token_verifies_the_right_user(schema):
    mailer = _register(schema, "alice@example.com")
    user_id = _user_id(schema, "alice@example.com")
    token = issue_email_token(schema, user_id, "verify")
    verify_email(schema, token)
    verified = schema.execute(
        "SELECT email_verified_at IS NOT NULL FROM users WHERE user_id = %s",
        (user_id,),
    ).fetchone()[0]
    assert verified
    assert mailer.sent


def test_a_token_is_single_use(schema):
    _register(schema, "alice@example.com")
    user_id = _user_id(schema, "alice@example.com")
    token = issue_email_token(schema, user_id, "verify")
    consume_email_token(schema, token, "verify")
    with pytest.raises(InvalidToken):
        consume_email_token(schema, token, "verify")


def test_an_expired_token_is_refused(schema):
    _register(schema, "alice@example.com")
    user_id = _user_id(schema, "alice@example.com")
    token = issue_email_token(schema, user_id, "verify")
    schema.execute(
        "UPDATE email_tokens SET created_at = now() - interval '25 hours', "
        "expires_at = now() - interval '1 second' WHERE user_id = %s",
        (user_id,),
    )
    with pytest.raises(InvalidToken):
        consume_email_token(schema, token, "verify")


def test_a_token_is_refused_for_the_wrong_purpose(schema):
    _register(schema, "alice@example.com")
    user_id = _user_id(schema, "alice@example.com")
    token = issue_email_token(schema, user_id, "verify")
    with pytest.raises(InvalidToken):
        consume_email_token(schema, token, "reset")


def test_an_unknown_token_is_refused(schema):
    with pytest.raises(InvalidToken):
        consume_email_token(schema, "not-a-real-token", "verify")


def test_a_users_token_does_not_verify_another_user(schema):
    _register(schema, "alice@example.com")
    _register(schema, "bob@example.com")
    alice_id = _user_id(schema, "alice@example.com")
    bob_id = _user_id(schema, "bob@example.com")
    token = issue_email_token(schema, alice_id, "verify")
    verify_email(schema, token)
    bob_verified = schema.execute(
        "SELECT email_verified_at IS NOT NULL FROM users WHERE user_id = %s",
        (bob_id,),
    ).fetchone()[0]
    assert not bob_verified


# --- registration -------------------------------------------------------------


def test_registering_a_new_address_creates_an_unverified_user_and_mails_a_link(
    schema,
):
    mailer = _register(schema, "alice@example.com")
    row = schema.execute(
        "SELECT email_verified_at FROM users WHERE email = 'alice@example.com'"
    ).fetchone()
    assert row[0] is None
    assert len(mailer.sent) == 1
    assert "verify" in mailer.sent[0].subject.lower()


def test_registering_a_taken_address_creates_nothing_and_mails_a_reset_link(schema):
    _register(schema, "alice@example.com")
    mailer = _register(schema, "alice@example.com", "a different password")
    assert schema.execute("SELECT count(*) FROM users").fetchone()[0] == 1
    assert len(mailer.sent) == 1
    assert "reset" in mailer.sent[0].body.lower()


# --- login gate -----------------------------------------------------------


def test_login_is_refused_until_verified_then_succeeds(schema):
    _register(schema, "alice@example.com")
    user_id = _user_id(schema, "alice@example.com")
    with pytest.raises(LoginFailed):
        authenticate(schema, "alice@example.com", PASSWORD, ip="1.1.1.1")
    token = issue_email_token(schema, user_id, "verify")
    verify_email(schema, token)
    account = authenticate(schema, "alice@example.com", PASSWORD, ip="1.1.1.1")
    assert account.user_id == user_id


# --- password reset ---------------------------------------------------------


def test_reset_request_for_an_unknown_address_is_silent(schema):
    mailer = NullMailer()
    request_password_reset(schema, "nobody@example.com", "1.1.1.1", mailer, BASE_URL)
    assert mailer.sent == []


def test_reset_request_for_an_unverified_account_is_silent(schema):
    _register(schema, "alice@example.com")
    mailer = NullMailer()
    request_password_reset(schema, "alice@example.com", "1.1.1.1", mailer, BASE_URL)
    assert mailer.sent == []


def test_reset_request_for_a_verified_account_mails_a_link(schema):
    register_account(schema, "alice@example.com", PASSWORD)
    mailer = NullMailer()
    request_password_reset(schema, "alice@example.com", "1.1.1.1", mailer, BASE_URL)
    assert len(mailer.sent) == 1
    assert "token=" in mailer.sent[0].body


def test_reset_password_changes_the_password_and_revokes_every_session(schema):
    user_id = register_account(schema, "alice@example.com", PASSWORD)
    old_refresh = issue_refresh_token(schema, user_id)
    token = issue_email_token(schema, user_id, "reset")
    reset_password(schema, token, NEW_PASSWORD)

    with pytest.raises(LoginFailed):
        authenticate(schema, "alice@example.com", PASSWORD, ip="1.1.1.1")
    account = authenticate(schema, "alice@example.com", NEW_PASSWORD, ip="1.1.1.1")
    assert account.user_id == user_id

    with pytest.raises(InvalidToken):
        rotate_refresh_token(schema, old_refresh)


def test_reset_token_is_single_use(schema):
    user_id = register_account(schema, "alice@example.com", PASSWORD)
    token = issue_email_token(schema, user_id, "reset")
    reset_password(schema, token, NEW_PASSWORD)
    with pytest.raises(InvalidToken):
        reset_password(schema, token, "yet another password")


# --- resend verification ----------------------------------------------------


def test_resend_verification_is_silent_for_an_unknown_address(schema):
    mailer = NullMailer()
    resend_verification(schema, "nobody@example.com", "1.1.1.1", mailer, BASE_URL)
    assert mailer.sent == []


def test_resend_verification_is_silent_for_an_already_verified_account(schema):
    register_account(schema, "alice@example.com", PASSWORD)
    mailer = NullMailer()
    resend_verification(schema, "alice@example.com", "1.1.1.1", mailer, BASE_URL)
    assert mailer.sent == []


def test_resend_verification_mails_a_fresh_link_for_an_unverified_account(schema):
    _register(schema, "alice@example.com")
    mailer = NullMailer()
    resend_verification(schema, "alice@example.com", "1.1.1.1", mailer, BASE_URL)
    assert len(mailer.sent) == 1


# --- send rate limiting -----------------------------------------------------


def test_repeated_registration_attempts_stop_sending_mail_past_the_limit(schema):
    # Seeded directly, so the address-level send budget starts empty: the
    # point of this test is what repeated *registration attempts* do to it,
    # not the setup account's own verification mail.
    schema.execute(
        "INSERT INTO users (email, password_hash) VALUES (%s, %s)",
        ("alice@example.com", hash_password(PASSWORD)),
    )
    mailer = NullMailer()
    for _ in range(5):
        register(
            schema,
            "alice@example.com",
            "another password",
            ip="1.1.1.1",
            mailer=mailer,
            base_url=BASE_URL,
        )
    # 3 sends per address per hour; further attempts send nothing more but
    # still raise no error and create no second account.
    assert len(mailer.sent) == 3
    assert schema.execute("SELECT count(*) FROM users").fetchone()[0] == 1


def test_repeated_resets_from_one_ip_stop_sending_past_the_ip_limit(schema):
    for n in range(12):
        register_account(schema, f"user{n}@example.com", PASSWORD)
    mailer = NullMailer()
    for n in range(12):
        request_password_reset(
            schema, f"user{n}@example.com", "9.9.9.9", mailer, BASE_URL
        )
    assert len(mailer.sent) == 10


# --- PII / token hygiene -----------------------------------------------------


def test_no_token_or_password_ever_reaches_a_log_record(schema, caplog):
    logger = logging.getLogger(LOGGER)
    logger.propagate = True
    with caplog.at_level(logging.INFO, logger=LOGGER):
        mailer = _register(schema, "alice@example.com")
        user_id = _user_id(schema, "alice@example.com")
        verify_token = issue_email_token(schema, user_id, "verify")
        verify_email(schema, verify_token)
        reset_token = issue_email_token(schema, user_id, "reset")
        reset_password(schema, reset_token, NEW_PASSWORD)
    text = "\n".join(record.getMessage() for record in caplog.records)
    assert verify_token not in text
    assert reset_token not in text
    assert PASSWORD not in text
    assert NEW_PASSWORD not in text
    for sent in mailer.sent:
        assert sent.body not in text


def test_accounts_need_an_autocommit_connection_for_every_email_flow(
    schema, admin_url
):
    url = make_conninfo(admin_url, dbname=schema.info.dbname)
    with psycopg.connect(url) as conn:
        with pytest.raises(ValueError, match="autocommit"):
            request_password_reset(
                conn, "alice@example.com", "1.1.1.1", NullMailer(), BASE_URL
            )
        with pytest.raises(ValueError, match="autocommit"):
            reset_password(conn, "some-token", NEW_PASSWORD)
        with pytest.raises(ValueError, match="autocommit"):
            verify_email(conn, "some-token")
        with pytest.raises(ValueError, match="autocommit"):
            resend_verification(
                conn, "alice@example.com", "1.1.1.1", NullMailer(), BASE_URL
            )
