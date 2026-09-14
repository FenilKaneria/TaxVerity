"""Step 14.5/14.8 — the SSE turns endpoint over real HTTP: event order, the
no-`verified:false` invariant, fault injection, and the refusal paths (no
auth, a foreign thread id). Reuses `test_graph_stream.py`'s fixtures for the
graph-level scenarios and drives them through `create_turn_route` instead of
`build_graph` directly, so the HTTP framing (SSE `event:`/`data:` lines, the
404/401 mapping) is what is actually under test here.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from conftest import register_account
from taxverity.api.app import app_state
from taxverity.api.threads_routes import router as threads_router
from taxverity.api.turns_routes import router as turns_router
from taxverity.auth.tokens import AccessTokens
from taxverity.facts import UserFacts
from taxverity.generation.generate import AnswerGenerator
from taxverity.llm.client import LLMUnavailable
from taxverity.llm.extract import ExtractionResult
from taxverity.retrieval.base import ScoredChunk
from taxverity.safety.classifier import ScopeCategory
from taxverity.safety.evidence_gate import INSUFFICIENT_EVIDENCE_MESSAGE
from taxverity.threads.store import create_thread
from test_generation import FABRICATED, GOOD, FakeLLM, ndjson
from test_graph_nodes import deps
from test_verifier import CHUNKS, QUESTION

JWT_SECRET = "y" * 32
PASSWORD = "correct-horse-1"

PACK_RESULTS = [
    ScoredChunk(chunk=CHUNKS["22(1)"], score=2.0),
    ScoredChunk(chunk=CHUNKS["24"], score=1.0),
]


class FakePool:
    """Duck-types `psycopg_pool.ConnectionPool`'s one used method: a single
    connection handed out for the caller's lifetime, real enough for
    `get_conn` and `request_deps` to both work against the `schema` fixture."""

    def __init__(self, conn):
        self._conn = conn

    @contextmanager
    def connection(self):
        yield self._conn


def _static_deps(generator):
    return deps(
        conn=None,
        classifier=SimpleNamespace(
            classify=lambda q: SimpleNamespace(category=ScopeCategory.IN_SCOPE, response=None)
        ),
        contextualizer=SimpleNamespace(
            contextualize=lambda q, prior: SimpleNamespace(
                query=q, rewritten=False, completion=None
            )
        ),
        extractor=SimpleNamespace(
            extract=lambda turn: ExtractionResult(
                facts=UserFacts(facts=()), rejections=(), repairable=(), repaired=False, completions=()
            )
        ),
        retriever=SimpleNamespace(search=lambda query, k: PACK_RESULTS),
        generator=generator,
    )


@pytest.fixture
def access_tokens():
    return AccessTokens(JWT_SECRET)


@pytest.fixture
def alice(schema):
    return register_account(schema, "alice@example.com", PASSWORD)


@pytest.fixture
def thread_id(schema, alice):
    return create_thread(schema, alice, "House property").thread_id


def _client(schema, access_tokens, generator):
    app = FastAPI()
    app.include_router(threads_router)
    app.include_router(turns_router)
    state = SimpleNamespace(
        pool=FakePool(schema), static_deps=_static_deps(generator), access_tokens=access_tokens
    )
    app.dependency_overrides[app_state] = lambda: state
    return TestClient(app)


def _auth(access_tokens, user_id) -> dict[str, str]:
    return {"Authorization": f"Bearer {access_tokens.issue(user_id)}"}


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


def test_no_auth_header_is_refused(schema, access_tokens, thread_id):
    client = _client(schema, access_tokens, AnswerGenerator(FakeLLM(ndjson(GOOD)), CHUNKS))
    response = client.post(f"/v1/threads/{thread_id}/turns", json={"question": QUESTION})
    assert response.status_code == 401


def test_a_foreign_thread_id_is_refused_before_any_stream_opens(schema, access_tokens, alice):
    bob = register_account(schema, "bob@example.com", PASSWORD)
    alice_thread = create_thread(schema, alice, "Alice only").thread_id
    client = _client(schema, access_tokens, AnswerGenerator(FakeLLM(ndjson(GOOD)), CHUNKS))
    response = client.post(
        f"/v1/threads/{alice_thread}/turns",
        json={"question": QUESTION},
        headers=_auth(access_tokens, bob),
    )
    assert response.status_code == 404


def test_event_order_is_stage_then_claims_then_final(schema, access_tokens, alice, thread_id):
    client = _client(schema, access_tokens, AnswerGenerator(FakeLLM(ndjson(GOOD)), CHUNKS))
    response = client.post(
        f"/v1/threads/{thread_id}/turns",
        json={"question": QUESTION},
        headers=_auth(access_tokens, alice),
    )
    assert response.status_code == 200
    events = _parse_sse(response.text)

    names = [name for name, _ in events]
    assert names[:3] == ["stage", "stage", "stage"]
    assert "claim" in names
    assert names[-1] == "final"
    assert names.index("stage") < names.index("claim")
    final_data = events[-1][1]
    assert "disclaimer" in final_data and "route" in final_data


def test_no_claim_event_ever_carries_verified_false(schema, access_tokens, alice, thread_id):
    generator = AnswerGenerator(
        FakeLLM(ndjson(GOOD, FABRICATED), LLMUnavailable("no repair")), CHUNKS
    )
    client = _client(schema, access_tokens, generator)
    response = client.post(
        f"/v1/threads/{thread_id}/turns",
        json={"question": QUESTION},
        headers=_auth(access_tokens, alice),
    )
    events = _parse_sse(response.text)
    claims = [data for name, data in events if name == "claim"]
    assert claims
    assert all(claim["verified"] is True for claim in claims)


def test_fault_injection_withholds_the_bad_claim_and_keeps_the_good_one_intact(
    schema, access_tokens, alice, thread_id
):
    generator = AnswerGenerator(
        FakeLLM(ndjson(GOOD, FABRICATED), LLMUnavailable("no repair")), CHUNKS
    )
    client = _client(schema, access_tokens, generator)
    response = client.post(
        f"/v1/threads/{thread_id}/turns",
        json={"question": QUESTION},
        headers=_auth(access_tokens, alice),
    )
    events = _parse_sse(response.text)
    claim_events = [data for name, data in events if name == "claim"]
    withheld_events = [data for name, data in events if name == "withheld"]

    assert len(claim_events) == 1
    assert claim_events[0]["id"] == 1
    assert claim_events[0]["verified"] is True
    assert len(withheld_events) == 1
    assert withheld_events[0]["id"] == 2
    assert withheld_events[0]["reason"] == "citation_not_in_evidence"


def test_a_gated_turn_still_names_its_frame_final_and_carries_the_message(
    schema, access_tokens, alice, thread_id
):
    """Advisor pivot, Step 4: `_event_name` sniffs `"disclaimer"` before
    `"verified"`/`"reason"`, so adding `text`/`searched` to `FinalEvent` must
    not make a gated turn's frame collide with `claim`/`withheld`."""
    generator = AnswerGenerator(FakeLLM(""), CHUNKS)  # the model emits nothing
    client = _client(schema, access_tokens, generator)
    response = client.post(
        f"/v1/threads/{thread_id}/turns",
        json={"question": QUESTION},
        headers=_auth(access_tokens, alice),
    )
    events = _parse_sse(response.text)
    assert events[-1][0] == "final"
    final_data = events[-1][1]
    assert final_data["text"] == INSUFFICIENT_EVIDENCE_MESSAGE
    assert final_data["searched"] == ["22(1)", "24"]
    assert not any(name in ("claim", "withheld") for name, _ in events)
