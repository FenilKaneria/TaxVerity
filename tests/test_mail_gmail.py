"""Step 11.4c — the Gmail mailer. All tests drive a scripted MockTransport;
none touch the network."""

from __future__ import annotations

import base64
import json

import httpx2
import pytest

from taxverity.mail.gmail import GmailMailer, MailError, NullMailer, SentMail

CLIENT_ID = "test-client-id"
CLIENT_SECRET = "test-client-secret"
REFRESH_TOKEN = "test-refresh-token"
SENDER = "sender@example.com"


class Recorder:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests: list[httpx2.Request] = []

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        if self.responses:
            scripted = self.responses.pop(0)
            if isinstance(scripted, Exception):
                raise scripted
            return scripted
        return token_response()

    @property
    def urls(self) -> list[str]:
        return [str(r.url) for r in self.requests]


def token_response(access_token: str = "test-access-token") -> httpx2.Response:
    return httpx2.Response(
        200, json={"access_token": access_token, "expires_in": 3600}
    )


def send_response(status: int = 200) -> httpx2.Response:
    return httpx2.Response(status, json={"id": "msg-1"})


def make(handler: Recorder) -> GmailMailer:
    http = httpx2.Client(transport=httpx2.MockTransport(handler))
    return GmailMailer(
        CLIENT_ID, CLIENT_SECRET, REFRESH_TOKEN, SENDER, http_client=http
    )


def test_a_send_first_fetches_a_token_then_posts_the_message():
    handler = Recorder(token_response(), send_response())
    mailer = make(handler)
    mailer.send("alice@example.com", "Subject", "Body text")
    assert handler.urls == [
        "https://oauth2.googleapis.com/token",
        "https://gmail.googleapis.com/gmail/v1/users/me/messages/send",
    ]
    send_body = json.loads(handler.requests[1].read())
    raw = base64.urlsafe_b64decode(send_body["raw"] + "==").decode("utf-8")
    assert "Subject" in raw
    assert "Body text" in raw
    assert "alice@example.com" in raw


def test_the_access_token_is_cached_across_sends():
    handler = Recorder(token_response(), send_response(), send_response())
    mailer = make(handler)
    mailer.send("alice@example.com", "s1", "b1")
    mailer.send("bob@example.com", "s2", "b2")
    assert handler.urls.count("https://oauth2.googleapis.com/token") == 1


def test_a_401_refreshes_the_token_once_and_retries():
    handler = Recorder(
        token_response("stale-token"),
        send_response(401),
        token_response("fresh-token"),
        send_response(200),
    )
    mailer = make(handler)
    mailer.send("alice@example.com", "s", "b")
    assert handler.urls.count("https://oauth2.googleapis.com/token") == 2


def test_a_failed_token_refresh_raises():
    handler = Recorder(httpx2.Response(400, json={"error": "invalid_grant"}))
    mailer = make(handler)
    with pytest.raises(MailError):
        mailer.send("alice@example.com", "s", "b")


def test_a_failed_send_raises():
    handler = Recorder(token_response(), send_response(500))
    mailer = make(handler)
    with pytest.raises(MailError):
        mailer.send("alice@example.com", "s", "b")


def test_a_blank_sender_is_refused():
    with pytest.raises(ValueError):
        GmailMailer(CLIENT_ID, CLIENT_SECRET, REFRESH_TOKEN, "not-an-address")


def test_null_mailer_records_what_it_would_send():
    mailer = NullMailer()
    mailer.send("alice@example.com", "s", "b")
    assert mailer.sent == [SentMail(to="alice@example.com", subject="s", body="b")]
