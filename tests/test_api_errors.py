"""Step 14.7 — the centralised error taxonomy: each helper returns the right
status code and `detail` string, and an unhandled exception never leaks its
own text into a response (`install_error_handlers`'s one job).
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from taxverity.api.errors import (
    AUTH_FAILED,
    INTERNAL_ERROR,
    INVALID_REQUEST,
    NOT_FOUND,
    RATE_LIMITED,
    UPSTREAM_UNAVAILABLE,
    auth_failed,
    install_error_handlers,
    invalid_request,
    not_found,
    rate_limited,
    upstream_unavailable,
)


def test_auth_failed_is_401_with_the_fixed_detail():
    error = auth_failed()
    assert error.status_code == 401
    assert error.detail == AUTH_FAILED


def test_not_found_is_404_with_the_fixed_detail():
    error = not_found()
    assert error.status_code == 404
    assert error.detail == NOT_FOUND


def test_rate_limited_is_429_with_the_fixed_detail():
    error = rate_limited()
    assert error.status_code == 429
    assert error.detail == RATE_LIMITED


def test_upstream_unavailable_is_503_with_the_fixed_detail():
    error = upstream_unavailable()
    assert error.status_code == 503
    assert error.detail == UPSTREAM_UNAVAILABLE


def test_invalid_request_defaults_to_the_fixed_detail():
    error = invalid_request()
    assert error.status_code == 400
    assert error.detail == INVALID_REQUEST


def test_invalid_request_carries_a_safe_caller_message_when_given_one():
    error = invalid_request("password must be at least 8 characters")
    assert error.status_code == 400
    assert error.detail == "password must be at least 8 characters"


def _app_with_a_route_that_blows_up() -> FastAPI:
    app = FastAPI()
    install_error_handlers(app)

    @app.get("/boom")
    def boom() -> None:
        raise ValueError("connection string: postgres://user:hunter2@db/prod")

    return app


def test_an_unhandled_exception_never_leaks_its_own_text():
    app = _app_with_a_route_that_blows_up()
    client = TestClient(app, raise_server_exceptions=False)
    response = client.get("/boom")

    assert response.status_code == 500
    assert response.json() == {"detail": INTERNAL_ERROR}
    assert "hunter2" not in response.text
    assert "postgres://" not in response.text


def test_an_ordinary_http_exception_is_unaffected_by_the_catch_all():
    """The catch-all handler must not shadow the routes' own `HTTPException`s
    — it is installed on `ServerErrorMiddleware`, one layer further out, and
    only ever fires for something no route anticipated."""
    app = FastAPI()
    install_error_handlers(app)

    @app.get("/refused")
    def refused() -> None:
        raise not_found()

    client = TestClient(app)
    response = client.get("/refused")
    assert response.status_code == 404
    assert response.json() == {"detail": NOT_FOUND}
