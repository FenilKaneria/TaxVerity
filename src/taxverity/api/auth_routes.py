"""Step 14.2 — auth endpoints: register, login, refresh, logout, plus the
verify-email and password-reset flows Step 11.4 already built (a registered
account is otherwise never reachable, since it starts unverified).

The refresh token travels in an httpOnly, Secure, `SameSite=None` cookie
(rule 04); the access token goes in the response body only, never a cookie,
so the frontend holds it in memory. The cookie's `path` is `/v1/auth` rather
than the plan's literal "the refresh path" — scoping it to `/v1/auth/refresh`
alone would keep the browser from ever presenting it to `/v1/auth/logout`,
which needs to read it too.
"""

from __future__ import annotations

import psycopg
from fastapi import APIRouter, Cookie, Depends, HTTPException, Request, Response, status
from pydantic import BaseModel

from taxverity.api.app import AppState, app_state
from taxverity.api.deps import get_conn
from taxverity.auth.accounts import (
    InvalidEmail,
    LoginFailed,
    LoginThrottled,
    authenticate,
    register,
    request_password_reset,
    resend_verification,
    reset_password,
    verify_email,
)
from taxverity.auth.passwords import WeakPassword
from taxverity.auth.tokens import (
    InvalidToken,
    TokenReuse,
    issue_refresh_token,
    revoke_refresh_family,
    rotate_refresh_token,
)
from taxverity.observability import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/v1/auth", tags=["auth"])

REFRESH_COOKIE = "refresh_token"
REFRESH_COOKIE_PATH = "/v1/auth"


class RegisterRequest(BaseModel):
    email: str
    password: str


class LoginRequest(BaseModel):
    email: str
    password: str


class EmailRequest(BaseModel):
    email: str


class VerifyEmailRequest(BaseModel):
    token: str


class ResetPasswordRequest(BaseModel):
    token: str
    new_password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


def _client_ip(request: Request) -> str:
    return request.client.host if request.client else "unknown"


def _set_refresh_cookie(response: Response, token: str) -> None:
    response.set_cookie(
        REFRESH_COOKIE,
        token,
        httponly=True,
        secure=True,
        samesite="none",
        path=REFRESH_COOKIE_PATH,
    )


@router.post("/register", status_code=status.HTTP_202_ACCEPTED)
def register_route(
    body: RegisterRequest,
    request: Request,
    conn: psycopg.Connection = Depends(get_conn),  # noqa: B008
    state: AppState = Depends(app_state),  # noqa: B008
) -> dict[str, str]:
    try:
        register(
            conn,
            body.email,
            body.password,
            ip=_client_ip(request),
            mailer=state.mailer,
            base_url=state.settings.app_base_url,
        )
    except (InvalidEmail, WeakPassword) as error:
        raise HTTPException(status_code=400, detail=str(error)) from None
    # Deliberately the same response whatever happened inside `register()` —
    # rule 03's enumeration concern extends to this endpoint's own shape.
    return {"status": "if that address can register, a verification email was sent"}


@router.post("/login")
def login_route(
    body: LoginRequest,
    request: Request,
    response: Response,
    conn: psycopg.Connection = Depends(get_conn),  # noqa: B008
    state: AppState = Depends(app_state),  # noqa: B008
) -> TokenResponse:
    try:
        account = authenticate(conn, body.email, body.password, ip=_client_ip(request))
    except LoginThrottled:
        raise HTTPException(status_code=429, detail="rate_limited") from None
    except LoginFailed:
        raise HTTPException(status_code=401, detail="auth_failed") from None
    refresh = issue_refresh_token(conn, account.user_id)
    _set_refresh_cookie(response, refresh)
    return TokenResponse(access_token=state.access_tokens.issue(account.user_id))


@router.post("/refresh")
def refresh_route(
    response: Response,
    state: AppState = Depends(app_state),  # noqa: B008
    conn: psycopg.Connection = Depends(get_conn),  # noqa: B008
    refresh_token: str | None = Cookie(default=None, alias=REFRESH_COOKIE),
) -> TokenResponse:
    if refresh_token is None:
        raise HTTPException(status_code=401, detail="auth_failed")
    try:
        rotation = rotate_refresh_token(conn, refresh_token)
    except TokenReuse:
        response.delete_cookie(REFRESH_COOKIE, path=REFRESH_COOKIE_PATH)
        raise HTTPException(status_code=401, detail="auth_failed") from None
    except InvalidToken:
        raise HTTPException(status_code=401, detail="auth_failed") from None
    _set_refresh_cookie(response, rotation.refresh_token)
    return TokenResponse(access_token=state.access_tokens.issue(rotation.user_id))


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
def logout_route(
    response: Response,
    conn: psycopg.Connection = Depends(get_conn),  # noqa: B008
    refresh_token: str | None = Cookie(default=None, alias=REFRESH_COOKIE),
) -> None:
    if refresh_token is not None:
        revoke_refresh_family(conn, refresh_token)
    response.delete_cookie(REFRESH_COOKIE, path=REFRESH_COOKIE_PATH)


@router.post("/verify-email", status_code=status.HTTP_204_NO_CONTENT)
def verify_email_route(
    body: VerifyEmailRequest, conn: psycopg.Connection = Depends(get_conn)  # noqa: B008
) -> None:
    try:
        verify_email(conn, body.token)
    except InvalidToken:
        raise HTTPException(status_code=400, detail="invalid_request") from None


@router.post("/resend-verification", status_code=status.HTTP_202_ACCEPTED)
def resend_verification_route(
    body: EmailRequest,
    request: Request,
    conn: psycopg.Connection = Depends(get_conn),  # noqa: B008
    state: AppState = Depends(app_state),  # noqa: B008
) -> dict[str, str]:
    resend_verification(
        conn, body.email, _client_ip(request), state.mailer, state.settings.app_base_url
    )
    return {"status": "if that account needs verifying, an email was sent"}


@router.post("/forgot-password", status_code=status.HTTP_202_ACCEPTED)
def forgot_password_route(
    body: EmailRequest,
    request: Request,
    conn: psycopg.Connection = Depends(get_conn),  # noqa: B008
    state: AppState = Depends(app_state),  # noqa: B008
) -> dict[str, str]:
    request_password_reset(
        conn, body.email, _client_ip(request), state.mailer, state.settings.app_base_url
    )
    return {"status": "if that account exists, a reset email was sent"}


@router.post("/reset-password", status_code=status.HTTP_204_NO_CONTENT)
def reset_password_route(
    body: ResetPasswordRequest, conn: psycopg.Connection = Depends(get_conn)  # noqa: B008
) -> None:
    try:
        reset_password(conn, body.token, body.new_password)
    except InvalidToken:
        raise HTTPException(status_code=400, detail="invalid_request") from None
    except WeakPassword as error:
        raise HTTPException(status_code=400, detail=str(error)) from None
