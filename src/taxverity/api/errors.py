"""A small, fixed vocabulary of `detail` strings every route maps its failures onto,
instead of each route inventing its own string."""

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
    return HTTPException(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail=RATE_LIMITED
    )


def upstream_unavailable() -> HTTPException:
    """No caller yet: every vendor call on the request path today either degrades in-
    process (dense/rerank fallback) or runs inside the turns stream, past the point a
    status code can still be returned."""
    return HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=UPSTREAM_UNAVAILABLE
    )


def invalid_request(message: str | None = None) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST, detail=message or INVALID_REQUEST
    )


def install_error_handlers(app: FastAPI) -> None:
    """The one place an unhandled exception is stopped from reaching a response body."""

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("unhandled error on %s %s", request.method, request.url.path)
        return JSONResponse({"detail": INTERNAL_ERROR}, status_code=500)
