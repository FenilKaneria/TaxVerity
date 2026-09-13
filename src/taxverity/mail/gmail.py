"""Step 11.4c — outbound mail for account verification and password reset.

Registration must not tell a caller whether an address is already registered
(rule 03), so both the new-account and already-registered paths send mail
instead of returning a distinguishing response. This is the delivery
mechanism for that: a personal Gmail sender, reached through Gmail's REST API
with an OAuth refresh token — there is no Workspace domain here to delegate a
service account from.

This is rule 03's egress path 6. The message body carries a one-time token
embedded in a link; it must never reach a log or a trace, so `send()` logs
only the purpose and recipient count, never the subject or body.
"""

from __future__ import annotations

import base64
import time
from dataclasses import dataclass
from email.message import EmailMessage
from typing import Protocol

import httpx2

from taxverity.config import Settings
from taxverity.observability import get_logger

logger = get_logger(__name__)

TOKEN_URL = "https://oauth2.googleapis.com/token"
SEND_URL = "https://gmail.googleapis.com/gmail/v1/users/me/messages/send"
DEFAULT_TIMEOUT = 15.0
# A fetched access token is short-lived; refreshed a little early so a send
# never races its own expiry.
TOKEN_REFRESH_MARGIN = 60.0


class MailError(RuntimeError):
    """The provider refused the send. Never retried automatically — a caller
    that must not fail silently (registration, reset) decides what to do."""


class Mailer(Protocol):
    def send(self, to: str, subject: str, body: str) -> None: ...


@dataclass
class SentMail:
    to: str
    subject: str
    body: str


class NullMailer:
    """No provider configured. Records what would have been sent so a test or
    a dev session can inspect it; never used in a real deployment."""

    def __init__(self) -> None:
        self.sent: list[SentMail] = []

    def send(self, to: str, subject: str, body: str) -> None:
        self.sent.append(SentMail(to=to, subject=subject, body=body))
        logger.warning("no mailer configured: mail not actually sent")


class GmailMailer:
    def __init__(
        self,
        client_id: str,
        client_secret: str,
        refresh_token: str,
        sender: str,
        *,
        http_client: httpx2.Client | None = None,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        if not sender or "@" not in sender:
            raise ValueError("sender must be an email address")
        self._client_id = client_id
        self._client_secret = client_secret
        self._refresh_token = refresh_token
        self._sender = sender
        self._owns_client = http_client is None
        self._client = http_client or httpx2.Client(timeout=timeout)
        self._access_token: str | None = None
        self._access_token_expires_at: float = 0.0

    @classmethod
    def from_settings(cls, settings: Settings, **kwargs: object) -> GmailMailer:
        return cls(
            settings.require("gmail_client_id"),
            settings.require("gmail_client_secret"),
            settings.require("gmail_refresh_token"),
            settings.require("gmail_sender"),
            **kwargs,  # type: ignore[arg-type]
        )

    def send(self, to: str, subject: str, body: str) -> None:
        raw = self._encode(to, subject, body)
        token = self._get_access_token()
        response = self._post_send(token, raw)
        if response.status_code == 401:
            # The cached token may have been revoked out from under us; refresh
            # once and retry, rather than failing on a token we know is stale.
            logger.warning("gmail send got 401; refreshing token and retrying once")
            token = self._get_access_token(force_refresh=True)
            response = self._post_send(token, raw)
        if response.status_code >= 400:
            logger.warning("gmail send failed: status %d", response.status_code)
            raise MailError(f"gmail send failed with status {response.status_code}")
        logger.info("mail sent")

    def _post_send(self, token: str, raw: str) -> httpx2.Response:
        return self._client.post(
            SEND_URL,
            headers={"Authorization": f"Bearer {token}"},
            json={"raw": raw},
        )

    def _encode(self, to: str, subject: str, body: str) -> str:
        message = EmailMessage()
        message["To"] = to
        message["From"] = self._sender
        message["Subject"] = subject
        message.set_content(body)
        return base64.urlsafe_b64encode(message.as_bytes()).decode("ascii")

    def _get_access_token(self, *, force_refresh: bool = False) -> str:
        if (
            not force_refresh
            and self._access_token is not None
            and time.monotonic() < self._access_token_expires_at
        ):
            return self._access_token
        response = self._client.post(
            TOKEN_URL,
            data={
                "client_id": self._client_id,
                "client_secret": self._client_secret,
                "refresh_token": self._refresh_token,
                "grant_type": "refresh_token",
            },
        )
        if response.status_code >= 400:
            logger.warning(
                "gmail token refresh failed: status %d", response.status_code
            )
            raise MailError(
                f"gmail token refresh failed with status {response.status_code}"
            )
        payload = response.json()
        self._access_token = payload["access_token"]
        expires_in = float(payload.get("expires_in", 3600))
        self._access_token_expires_at = (
            time.monotonic() + expires_in - TOKEN_REFRESH_MARGIN
        )
        return self._access_token

    def close(self) -> None:
        if self._owns_client:
            self._client.close()


def mailer_from_settings(settings: Settings) -> Mailer:
    """A `NullMailer` when Gmail is not configured, so a dev session or a test
    database run needs no real credentials — the same shape as
    `LLMClient.from_settings`'s optional fallback."""
    if (
        settings.gmail_client_id is None
        or settings.gmail_client_secret is None
        or settings.gmail_refresh_token is None
        or settings.gmail_sender is None
    ):
        logger.warning("no gmail sender configured: mail is recorded, not sent")
        return NullMailer()
    return GmailMailer.from_settings(settings)
