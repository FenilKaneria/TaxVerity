"""Step 14.2 — auth endpoints over HTTP, against a throwaway database.

Routers are mounted directly (not through `create_app`'s full lifespan,
which needs a real corpus, embedding set and live vendor keys) with
`get_conn`/`app_state` overridden to a fake state backed by the `schema`
fixture and a `NullMailer` — the same seam `conftest.register_account` uses,
now exercised through the actual HTTP routes rather than called directly.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from taxverity.api.app import app_state
from taxverity.api.auth_routes import router as auth_router
from taxverity.api.deps import get_conn
from taxverity.auth.tokens import AccessTokens
from taxverity.mail.gmail import NullMailer

JWT_SECRET = "x" * 32


def _client(schema) -> tuple[TestClient, NullMailer]:
    mailer = NullMailer()
    fake_state = SimpleNamespace(
        mailer=mailer,
        settings=SimpleNamespace(app_base_url="http://t"),
        access_tokens=AccessTokens(JWT_SECRET),
    )
    def _get_conn_override():
        yield schema

    app = FastAPI()
    app.include_router(auth_router)
    app.dependency_overrides[app_state] = lambda: fake_state
    app.dependency_overrides[get_conn] = _get_conn_override
    return TestClient(app), mailer


def _token_from(body: str) -> str:
    return body.split("token=", 1)[1].split()[0]


@pytest.fixture
def client(schema):
    return _client(schema)


def test_register_is_silent_and_sends_a_verification_link(client):
    api, mailer = client
    response = api.post(
        "/v1/auth/register", json={"email": "a@example.com", "password": "correct-horse-1"}
    )
    assert response.status_code == 202
    assert len(mailer.sent) == 1
    assert "verify" in mailer.sent[0].subject.lower()


def test_a_weak_password_is_refused(client):
    api, _mailer = client
    response = api.post(
        "/v1/auth/register", json={"email": "b@example.com", "password": "short"}
    )
    assert response.status_code == 400


def test_login_before_verification_fails_with_the_generic_error(client):
    api, _mailer = client
    api.post("/v1/auth/register", json={"email": "c@example.com", "password": "correct-horse-1"})
    response = api.post(
        "/v1/auth/login", json={"email": "c@example.com", "password": "correct-horse-1"}
    )
    assert response.status_code == 401
    assert response.json()["detail"] == "auth_failed"


def test_verify_then_login_issues_an_access_token_and_a_refresh_cookie(client):
    api, mailer = client
    api.post("/v1/auth/register", json={"email": "d@example.com", "password": "correct-horse-1"})
    token = _token_from(mailer.sent[0].body)

    verify = api.post("/v1/auth/verify-email", json={"token": token})
    assert verify.status_code == 204

    login = api.post(
        "/v1/auth/login", json={"email": "d@example.com", "password": "correct-horse-1"}
    )
    assert login.status_code == 200
    assert login.json()["access_token"]
    assert login.cookies.get("refresh_token") is not None


def test_refresh_rotates_the_cookie_and_reuse_is_refused(client):
    api, mailer = client
    api.post("/v1/auth/register", json={"email": "e@example.com", "password": "correct-horse-1"})
    api.post("/v1/auth/verify-email", json={"token": _token_from(mailer.sent[0].body)})
    login = api.post(
        "/v1/auth/login", json={"email": "e@example.com", "password": "correct-horse-1"}
    )
    old_refresh = login.cookies["refresh_token"]

    refreshed = api.post("/v1/auth/refresh", cookies={"refresh_token": old_refresh})
    assert refreshed.status_code == 200
    assert refreshed.json()["access_token"]
    assert refreshed.cookies["refresh_token"] != old_refresh

    reused = api.post("/v1/auth/refresh", cookies={"refresh_token": old_refresh})
    assert reused.status_code == 401

    # Reuse revokes the whole family: even the token issued by the rotation
    # above is now dead.
    new_refresh = refreshed.cookies["refresh_token"]
    dead = api.post("/v1/auth/refresh", cookies={"refresh_token": new_refresh})
    assert dead.status_code == 401


def test_logout_revokes_the_refresh_token(client):
    api, mailer = client
    api.post("/v1/auth/register", json={"email": "f@example.com", "password": "correct-horse-1"})
    api.post("/v1/auth/verify-email", json={"token": _token_from(mailer.sent[0].body)})
    login = api.post(
        "/v1/auth/login", json={"email": "f@example.com", "password": "correct-horse-1"}
    )
    refresh = login.cookies["refresh_token"]

    logout = api.post("/v1/auth/logout", cookies={"refresh_token": refresh})
    assert logout.status_code == 204

    after = api.post("/v1/auth/refresh", cookies={"refresh_token": refresh})
    assert after.status_code == 401


def test_reset_password_revokes_every_existing_session(client):
    api, mailer = client
    api.post("/v1/auth/register", json={"email": "g@example.com", "password": "correct-horse-1"})
    api.post("/v1/auth/verify-email", json={"token": _token_from(mailer.sent[0].body)})
    login = api.post(
        "/v1/auth/login", json={"email": "g@example.com", "password": "correct-horse-1"}
    )
    refresh = login.cookies["refresh_token"]

    api.post("/v1/auth/forgot-password", json={"email": "g@example.com"})
    reset_token = _token_from(mailer.sent[-1].body)
    reset = api.post(
        "/v1/auth/reset-password",
        json={"token": reset_token, "new_password": "another-correct-1"},
    )
    assert reset.status_code == 204

    stale = api.post("/v1/auth/refresh", cookies={"refresh_token": refresh})
    assert stale.status_code == 401

    relogin = api.post(
        "/v1/auth/login", json={"email": "g@example.com", "password": "another-correct-1"}
    )
    assert relogin.status_code == 200
