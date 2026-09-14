"""Step 14.3 — thread CRUD and facts edit over HTTP, including the IDOR case:
a foreign thread id must read exactly like a nonexistent one (rule 04/11.4's
isolation discipline, now checked at the HTTP layer too, not just the store).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from conftest import register_account
from taxverity.api.app import app_state
from taxverity.api.deps import get_conn
from taxverity.api.threads_routes import router as threads_router
from taxverity.auth.tokens import AccessTokens
from taxverity.threads.store import append_message

JWT_SECRET = "y" * 32
PASSWORD = "correct-horse-1"


@pytest.fixture
def access_tokens():
    return AccessTokens(JWT_SECRET)


@pytest.fixture
def client(schema, access_tokens):
    def _get_conn_override():
        yield schema

    app = FastAPI()
    app.include_router(threads_router)
    app.dependency_overrides[app_state] = lambda: SimpleNamespace(access_tokens=access_tokens)
    app.dependency_overrides[get_conn] = _get_conn_override
    return TestClient(app)


def _auth(access_tokens, user_id) -> dict[str, str]:
    return {"Authorization": f"Bearer {access_tokens.issue(user_id)}"}


@pytest.fixture
def alice(schema):
    return register_account(schema, "alice@example.com", PASSWORD)


@pytest.fixture
def bob(schema):
    return register_account(schema, "bob@example.com", PASSWORD)


def test_no_auth_header_is_refused(client):
    response = client.get("/v1/threads")
    assert response.status_code == 401


def test_create_list_get_rename_delete(client, access_tokens, alice):
    headers = _auth(access_tokens, alice)

    created = client.post("/v1/threads", json={"title": "House property"}, headers=headers)
    assert created.status_code == 201
    thread_id = created.json()["thread_id"]

    listed = client.get("/v1/threads", headers=headers)
    assert [t["thread_id"] for t in listed.json()] == [thread_id]

    got = client.get(f"/v1/threads/{thread_id}", headers=headers)
    assert got.json()["title"] == "House property"

    renamed = client.patch(
        f"/v1/threads/{thread_id}", json={"title": "Salary"}, headers=headers
    )
    assert renamed.json()["title"] == "Salary"

    deleted = client.delete(f"/v1/threads/{thread_id}", headers=headers)
    assert deleted.status_code == 204

    gone = client.get(f"/v1/threads/{thread_id}", headers=headers)
    assert gone.status_code == 404


def test_a_foreign_thread_id_reads_as_not_found(client, access_tokens, alice, bob):
    created = client.post(
        "/v1/threads", json={"title": "Alice only"}, headers=_auth(access_tokens, alice)
    )
    thread_id = created.json()["thread_id"]

    bob_headers = _auth(access_tokens, bob)
    assert client.get(f"/v1/threads/{thread_id}", headers=bob_headers).status_code == 404
    assert (
        client.patch(
            f"/v1/threads/{thread_id}", json={"title": "hijacked"}, headers=bob_headers
        ).status_code
        == 404
    )
    assert client.delete(f"/v1/threads/{thread_id}", headers=bob_headers).status_code == 404
    assert (
        client.get(f"/v1/threads/{thread_id}/messages", headers=bob_headers).status_code
        == 404
    )
    assert client.get(f"/v1/threads/{thread_id}/facts", headers=bob_headers).status_code == 404
    assert (
        client.patch(
            f"/v1/threads/{thread_id}/facts",
            json={"field": "salary_income", "raw_value": "1500000"},
            headers=bob_headers,
        ).status_code
        == 404
    )

    # Untouched from Alice's own side.
    still_hers = client.get(f"/v1/threads/{thread_id}", headers=_auth(access_tokens, alice))
    assert still_hers.json()["title"] == "Alice only"


def test_facts_edit_round_trips_as_a_stated_override(client, access_tokens, alice):
    headers = _auth(access_tokens, alice)
    thread_id = client.post(
        "/v1/threads", json={"title": "t"}, headers=headers
    ).json()["thread_id"]

    empty = client.get(f"/v1/threads/{thread_id}/facts", headers=headers)
    assert empty.json()["facts"] == {}

    edited = client.patch(
        f"/v1/threads/{thread_id}/facts",
        json={"field": "salary_income", "raw_value": "1500000"},
        headers=headers,
    )
    assert edited.status_code == 200
    assert edited.json()["facts"]["salary_income"]["status"] == "stated"
    assert edited.json()["facts"]["salary_income"]["raw_value"] == "1500000"

    reread = client.get(f"/v1/threads/{thread_id}/facts", headers=headers)
    assert reread.json()["facts"]["salary_income"]["raw_value"] == "1500000"

    overridden = client.patch(
        f"/v1/threads/{thread_id}/facts",
        json={"field": "salary_income", "raw_value": "1600000"},
        headers=headers,
    )
    assert len(overridden.json()["overrides"]) == 1


def test_messages_requires_auth(client):
    assert client.get("/v1/threads/00000000-0000-0000-0000-000000000000/messages").status_code == 401


def test_an_empty_thread_has_no_messages(client, access_tokens, alice):
    headers = _auth(access_tokens, alice)
    thread_id = client.post(
        "/v1/threads", json={"title": "t"}, headers=headers
    ).json()["thread_id"]

    empty = client.get(f"/v1/threads/{thread_id}/messages", headers=headers)
    assert empty.status_code == 200
    assert empty.json() == []


def test_messages_round_trip_oldest_first_with_citations(
    client, schema, access_tokens, alice
):
    headers = _auth(access_tokens, alice)
    thread_id = client.post(
        "/v1/threads", json={"title": "t"}, headers=headers
    ).json()["thread_id"]

    append_message(schema, alice, thread_id, "user", "What is section 19(1)?")
    append_message(
        schema,
        alice,
        thread_id,
        "assistant",
        "A standard deduction of fifty thousand rupees applies.",
        payload={"citations": ["19(1)"]},
    )

    response = client.get(f"/v1/threads/{thread_id}/messages", headers=headers)
    assert response.status_code == 200
    messages = response.json()
    assert [m["role"] for m in messages] == ["user", "assistant"]
    assert messages[0]["content"] == "What is section 19(1)?"
    assert messages[0]["citations"] == []
    assert messages[1]["citations"] == ["19(1)"]
    assert messages[1]["message_id"] > messages[0]["message_id"]
