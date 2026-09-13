"""Step 14.7 — the error taxonomy PLAN 14.7 names: a small, fixed vocabulary
of `detail` strings every route maps its failures onto, instead of each route
inventing its own string. A client branches on `detail`, not on guessing from
a status code plus free text.

`invalid_request()` takes an optional message because two of this project's
own exceptions (`InvalidEmail`, `WeakPassword`) are explicitly documented as
"safe to show the user" — that is a validation message, not an internal
leak, and the taxonomy is not asking routes to throw away information the
caller needs to fix their request. What must never happen is an *unhandled*
exception's own text reaching a response; `install_error_handlers()` is the
one place that boundary is enforced, not each route remembering to catch
`Exception`.
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse

from taxverity.observability import get_logger

logger = get_logger(__name__)

AUTH_FAILED = "auth_failed"
NOT_FOUND = "not_found"
RATE_LIMITED = "rate_limited"
UPSTREAM_UNAVAILABLE = "upstream_unavailable"
INVALID_REQUEST = "invalid_request"
INTERNAL_ERROR = "internal_error"


def auth_failed() -> HTTPException:
    return HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=AUTH_FAILED)


def not_found() -> HTTPException:
    return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=NOT_FOUND)


def rate_limited() -> HTTPException:
    return HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail=RATE_LIMITED)


def upstream_unavailable() -> HTTPException:
    """No caller yet (ADR-115): every vendor call on the request path today
    either degrades in-process (dense/rerank fallback) or runs inside the
    turns stream, past the point a status code can still be returned. Kept in
    the vocabulary for the route that does need it, rather than invented
    ahead of one."""
    return HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=UPSTREAM_UNAVAILABLE
    )


def invalid_request(message: str | None = None) -> HTTPException:
    return HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=message or INVALID_REQUEST)


def install_error_handlers(app: FastAPI) -> None:
    """The one place an unhandled exception is stopped from reaching a
    response body. Every route above raises `HTTPException` for anything it
    recognises; this handler only ever fires for something no route
    anticipated, which is exactly when leaking `str(error)` would be a bug,
    not a convenience."""

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("unhandled error on %s %s", request.method, request.url.path)
        return JSONResponse({"detail": INTERNAL_ERROR}, status_code=500)
