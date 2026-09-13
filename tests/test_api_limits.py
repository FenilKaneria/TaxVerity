"""Step 14.6/14.8 — the per-user daily turn cap, the request body size cap,
and CORS restricted to the configured frontend origin.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.testclient import TestClient

from conftest import register_account
from taxverity.api.app import BodySizeLimitMiddleware, app_state, create_app
from taxverity.api.limits import (
    DAILY_TURN_LIMIT,
    MAX_BODY_BYTES,
    DailyLimitReached,
    check_daily_turn_limit,
)
from taxverity.api.turns_routes import router as turns_router
from taxverity.auth.tokens import AccessTokens
from taxverity.config import Settings
from taxverity.threads.store import append_message, create_thread
from test_api_turns_sse import PACK_RESULTS, FakePool, _auth
from test_graph_nodes import deps
from test_verifier import QUESTION

PASSWORD = "correct-horse-1"


class Boom:
    def __getattr__(self, name: str):
        def _fail(*_args: object, **_kwargs: object) -> None:
            raise AssertionError(f"{name} must not be called once the cap is hit")

        return _fail


# --- check_daily_turn_limit (pure DB, no HTTP) -------------------------------


@pytest.fixture
def alice(schema):
    return register_account(schema, "alice@example.com", PASSWORD)


@pytest.fixture
def thread_id(schema, alice):
    return create_thread(schema, alice, "t").thread_id


def test_under_the_cap_is_not_refused(schema, alice, thread_id):
    for _ in range(DAILY_TURN_LIMIT - 1):
        append_message(schema, alice, thread_id, "user", "hi")
    check_daily_turn_limit(schema, alice)  # does not raise


def test_at_the_cap_is_refused(schema, alice, thread_id):
    for _ in range(DAILY_TURN_LIMIT):
        append_message(schema, alice, thread_id, "user", "hi")
    with pytest.raises(DailyLimitReached):
        check_daily_turn_limit(schema, alice)


def test_the_cap_is_per_user_not_global(schema, alice, thread_id):
    bob = register_account(schema, "bob@example.com", PASSWORD)
    for _ in range(DAILY_TURN_LIMIT):
        append_message(schema, alice, thread_id, "user", "hi")
    check_daily_turn_limit(schema, bob)  # unaffected by alice's usage


# --- the cap enforced over HTTP, before any stream opens ---------------------


def test_the_capped_endpoint_refuses_with_429_and_never_calls_the_generator(
    schema, thread_id, alice
):
    for _ in range(DAILY_TURN_LIMIT):
        append_message(schema, alice, thread_id, "user", "hi")

    access_tokens = AccessTokens("y" * 32)
    app = FastAPI()
    app.include_router(turns_router)
    state = SimpleNamespace(
        pool=FakePool(schema),
        static_deps=deps(
            conn=None,
            retriever=SimpleNamespace(search=lambda query, k: PACK_RESULTS),
            generator=Boom(),
        ),
        access_tokens=access_tokens,
    )
    app.dependency_overrides[app_state] = lambda: state
    client = TestClient(app)

    response = client.post(
        f"/v1/threads/{thread_id}/turns",
        json={"question": QUESTION},
        headers=_auth(access_tokens, alice),
    )
    assert response.status_code == 429


# --- request body size cap ---------------------------------------------------


@pytest.fixture
def body_limit_client():
    app = FastAPI()
    app.add_middleware(BodySizeLimitMiddleware, max_bytes=64)

    @app.post("/echo")
    def echo(body: dict) -> dict:
        return body

    return TestClient(app)


def test_a_body_under_the_cap_is_accepted(body_limit_client):
    response = body_limit_client.post("/echo", json={"a": "x"})
    assert response.status_code == 200


def test_a_body_over_the_cap_is_refused_with_413(body_limit_client):
    response = body_limit_client.post("/echo", json={"a": "x" * 200})
    assert response.status_code == 413
    assert response.json()["detail"] == "invalid_request"


def test_max_body_bytes_is_a_real_cap_not_unlimited():
    assert 0 < MAX_BODY_BYTES < 10_000_000


# --- CORS: only the configured frontend origin, credentials allowed ---------


def test_cors_is_restricted_to_the_configured_frontend_origin():
    settings = Settings(app_base_url="https://taxverity.example")
    app = create_app(settings)
    cors = next(m for m in app.user_middleware if m.cls is CORSMiddleware)
    assert cors.kwargs["allow_origins"] == ["https://taxverity.example"]
    assert cors.kwargs["allow_credentials"] is True
