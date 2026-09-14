"""Guest chat trial, closing the ADR-112 gap found live in Phase 17:
`guests/quota.py` (Step 11.4e) existed but no route ever called it. Mirrors
`test_api_turns_sse.py`'s pattern — a bare FastAPI with `app_state`
overridden, `schema` as the real (throwaway) Postgres connection `guests`'
own SQL needs.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from types import SimpleNamespace
from uuid import uuid4

from fastapi import FastAPI
from fastapi.testclient import TestClient

from taxverity.api.app import app_state
from taxverity.api.guest_routes import GUEST_COOKIE
from taxverity.api.guest_routes import router as guest_router
from taxverity.generation.generate import AnswerGenerator
from taxverity.guests.quota import GUEST_TURN_LIMIT
from taxverity.retrieval.base import ScoredChunk
from taxverity.safety.classifier import FIXED_RESPONSES, ScopeCategory
from taxverity.safety.evidence_gate import INSUFFICIENT_EVIDENCE_MESSAGE
from test_generation import GOOD, FakeLLM, ndjson
from test_graph_nodes import deps as build_deps
from test_verifier import CHUNKS, QUESTION

PACK_RESULTS = [ScoredChunk(chunk=CHUNKS["22(1)"], score=2.0), ScoredChunk(chunk=CHUNKS["24"], score=1.0)]


class FakePool:
    def __init__(self, conn):
        self._conn = conn

    @contextmanager
    def connection(self):
        yield self._conn


def _static_deps(category=ScopeCategory.IN_SCOPE, generator=None, conversational=None):
    if generator is None:
        generator = AnswerGenerator(FakeLLM(ndjson(GOOD)), CHUNKS)
    response = FIXED_RESPONSES.get(category)
    kwargs = dict(
        conn=None,
        classifier=SimpleNamespace(
            classify=lambda q: SimpleNamespace(category=category, response=response)
        ),
        retriever=SimpleNamespace(search=lambda query, k: PACK_RESULTS),
        generator=generator,
    )
    if conversational is not None:
        kwargs["conversational"] = conversational
    return build_deps(**kwargs)


def _client(schema, category=ScopeCategory.IN_SCOPE, generator=None, conversational=None):
    app = FastAPI()
    app.include_router(guest_router)
    state = SimpleNamespace(
        pool=FakePool(schema), static_deps=_static_deps(category, generator, conversational)
    )
    app.dependency_overrides[app_state] = lambda: state
    return TestClient(app)


def _parse_sse(text: str) -> list[tuple[str, dict]]:
    events = []
    for block in text.strip().split("\n\n"):
        if not block:
            continue
        lines = block.split("\n")
        name = lines[0].removeprefix("event: ")
        data = json.loads(lines[1].removeprefix("data: "))
        events.append((name, data))
    return events


def test_a_fresh_visitor_gets_a_guest_cookie_minted(schema):
    client = _client(schema)
    response = client.get("/v1/guest/status")
    assert response.status_code == 200
    assert response.json() == {"used": 0, "limit": GUEST_TURN_LIMIT, "remaining": GUEST_TURN_LIMIT}
    assert GUEST_COOKIE in response.cookies


def test_an_in_scope_turn_streams_stage_then_claim_then_final_no_persistence(schema):
    client = _client(schema)
    response = client.post("/v1/guest/turns", json={"question": QUESTION})
    assert response.status_code == 200
    events = _parse_sse(response.text)
    names = [name for name, _ in events]
    assert names[0] == "stage"
    assert "claim" in names
    assert names[-1] == "final"
    final_data = events[-1][1]
    assert final_data["route"] == "guest"
    assert "disclaimer" in final_data


def test_an_adjacent_question_gets_the_fixed_response_and_no_claims(schema):
    client = _client(schema, category=ScopeCategory.ADJACENT)
    response = client.post("/v1/guest/turns", json={"question": "what is the GST rate?"})
    events = _parse_sse(response.text)
    names = [name for name, _ in events]
    assert "claim" not in names
    assert names[-1] == "final"
    assert events[-1][1]["route"] == "adjacent"
    # advisor pivot, Step 6: this used to be dropped entirely on the guest
    # path (unlike the authenticated graph, which at least persisted it).
    assert events[-1][1]["text"] == FIXED_RESPONSES[ScopeCategory.ADJACENT]


def test_a_conversational_question_gets_the_guarded_reply(schema):
    conversational = SimpleNamespace(reply=lambda q: "Hi! Ask me about the Act.")
    client = _client(schema, category=ScopeCategory.CONVERSATIONAL, conversational=conversational)
    response = client.post("/v1/guest/turns", json={"question": "hi there"})
    events = _parse_sse(response.text)
    names = [name for name, _ in events]
    assert "claim" not in names
    assert names[-1] == "final"
    final_data = events[-1][1]
    assert final_data["route"] == "conversational"
    assert final_data["text"] == "Hi! Ask me about the Act."


def test_an_in_scope_turn_with_zero_grounded_claims_streams_the_gated_message(schema):
    generator = AnswerGenerator(FakeLLM(""), CHUNKS)  # the model emits nothing
    client = _client(schema, generator=generator)
    response = client.post("/v1/guest/turns", json={"question": QUESTION})
    events = _parse_sse(response.text)
    names = [name for name, _ in events]
    assert "claim" not in names
    assert names[-1] == "final"
    final_data = events[-1][1]
    assert final_data["text"] == INSUFFICIENT_EVIDENCE_MESSAGE
    assert final_data["searched"] == ["22(1)", "24"]


def test_the_sixth_turn_is_refused_with_429(schema):
    client = _client(schema)
    guest_id = str(uuid4())
    client.cookies.set(GUEST_COOKIE, guest_id)
    for _ in range(GUEST_TURN_LIMIT):
        response = client.post("/v1/guest/turns", json={"question": QUESTION})
        assert response.status_code == 200
    response = client.post("/v1/guest/turns", json={"question": QUESTION})
    assert response.status_code == 429


def test_status_reflects_turns_already_used(schema):
    client = _client(schema)
    guest_id = str(uuid4())
    client.cookies.set(GUEST_COOKIE, guest_id)
    client.post("/v1/guest/turns", json={"question": QUESTION})
    client.post("/v1/guest/turns", json={"question": QUESTION})
    response = client.get("/v1/guest/status")
    body = response.json()
    assert body["used"] == 2
    assert body["remaining"] == GUEST_TURN_LIMIT - 2


def test_two_different_guest_cookies_have_independent_limits(schema):
    client = _client(schema)
    for cookie_num in range(2):
        client.cookies.set(GUEST_COOKIE, str(uuid4()))
        for _ in range(GUEST_TURN_LIMIT):
            response = client.post("/v1/guest/turns", json={"question": QUESTION})
            assert response.status_code == 200, f"guest {cookie_num} turn should succeed"
