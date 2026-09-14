"""Step 15.1 — the uvicorn/Lambda entrypoint module.

`taxverity.api.main` calls `create_app()` at import time (no `Settings`
argument, so it reads the environment like the real deploy does). This must
not touch the database or any vendor: `create_app()` only wires the lifespan
closure and routes, it does not enter the lifespan. If that stopped being
true, importing this module in a plain test process would hang or raise.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient


def test_app_is_a_fastapi_instance_importable_with_no_live_dependencies() -> None:
    from taxverity.api.main import app

    assert isinstance(app, FastAPI)


def test_health_route_answers_with_no_lifespan_entered() -> None:
    from taxverity.api.main import app

    response = TestClient(app).get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
