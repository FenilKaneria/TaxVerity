"""Shared FastAPI dependencies: a request-scoped connection from the pool,
and the authenticated user id read from the access-token bearer header."""

from __future__ import annotations

from collections.abc import Iterator
from uuid import UUID

import psycopg
from fastapi import Depends, Header, HTTPException, status

from taxverity.api.app import AppState, app_state
from taxverity.auth.tokens import InvalidToken


def get_conn(state: AppState = Depends(app_state)) -> Iterator[psycopg.Connection]:  # noqa: B008
    with state.pool.connection() as conn:
        yield conn


def current_user(
    state: AppState = Depends(app_state),  # noqa: B008
    authorization: str | None = Header(default=None),
) -> UUID:
    if authorization is None or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="auth_failed")
    token = authorization.removeprefix("Bearer ").strip()
    try:
        return state.access_tokens.verify(token)
    except InvalidToken:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="auth_failed"
        ) from None
