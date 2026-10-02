"""Step 13.7 — the streaming contract through the compiled graph (rule 04):
event order, the no-`verified:false` invariant, and fault injection. A claim
citing evidence not in the pack must come back `withheld` while an earlier,
already-released claim in the same stream stays exactly as it was released —
nothing shown is ever retracted.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from conftest import register_account
from taxverity.facts import UserFacts
from taxverity.generation.generate import AnswerGenerator
from taxverity.graph.build import build_graph
from taxverity.graph.nodes import RETRY_POOL
from taxverity.llm.extract import ExtractionResult
from taxverity.retrieval.base import ScoredChunk
from taxverity.safety.classifier import ScopeCategory
from taxverity.safety.evidence_gate import INSUFFICIENT_EVIDENCE_MESSAGE
from taxverity.threads.store import create_thread
from test_generation import FABRICATED, GOOD, answer
from test_graph_nodes import PASSWORD, FakeLLM, deps
from test_graph_paths import RESCUABLE
from test_verifier import CHUNKS, QUESTION

# Retrieved: 22(1) and 24, matching test_verifier.PACK (not 23).
PACK_RESULTS = [
    ScoredChunk(chunk=CHUNKS["22(1)"], score=2.0),
    ScoredChunk(chunk=CHUNKS["24"], score=1.0),
]


@pytest.fixture
def alice(schema):
    return register_account(schema, "alice@example.com", PASSWORD)


@pytest.fixture
def thread_id(schema, alice):
    return create_thread(schema, alice, "House property").thread_id


def _streaming_deps(schema, generator):
    return deps(
        conn=schema,
        classifier=SimpleNamespace(
            classify=lambda q, **_: SimpleNamespace(
                category=ScopeCategory.IN_SCOPE, response=None, search_query=q
            )
        ),
        extractor=SimpleNamespace(
            extract=lambda turn: ExtractionResult(
                facts=UserFacts(facts=()),
                rejections=(),
                repairable=(),
                repaired=False,
                completions=(),
            )
        ),
        retriever=SimpleNamespace(search=lambda query, k: PACK_RESULTS),
        generator=generator,
    )


def _stream(deps_obj, alice, thread_id):
    graph = build_graph(deps_obj)
    return [
        chunk
        for _mode, chunk in graph.stream(
            {"user_id": alice, "thread_id": thread_id, "question": QUESTION},
            stream_mode=["custom"],
        )
    ]


def test_event_order_is_stage_then_claims_then_final(schema, alice, thread_id):
    d = _streaming_deps(schema, AnswerGenerator(FakeLLM(answer(GOOD)), CHUNKS))
    emitted = _stream(d, alice, thread_id)

    stages = [e["stage"] for e in emitted if "stage" in e]
    # R19 Phase C: `extract_facts`/`merge_facts` (-> "facts") and `retrieve`
    # (-> "evidence") run in parallel branches off `classify`, so their
    # relative order is no longer guaranteed — only that "thinking" leads.
    assert stages[0] == "thinking"
    assert set(stages[1:-1]) == {"facts", "evidence"}
    # R22 Part A: generation announces itself once both branches are in.
    assert stages[-1] == "writing"
    claim_positions = [i for i, e in enumerate(emitted) if e.get("type") == "content"]
    stage_positions = [i for i, e in enumerate(emitted) if "stage" in e]
    assert claim_positions and max(stage_positions) < min(claim_positions)
    # The final event is last and carries the disclaimer (rule 03).
    assert "disclaimer" in emitted[-1] and "route" in emitted[-1]
    assert all("disclaimer" not in e for e in emitted[:-1])


def test_no_claim_event_ever_carries_verified_false(schema, alice, thread_id):
    """Structural per rule 04 (`ClaimEvent.verified: Literal[True]`), checked
    here against what actually crosses the wire, including a withheld claim
    alongside a served one."""
    d = _streaming_deps(
        schema,
        AnswerGenerator(FakeLLM(answer(GOOD, FABRICATED), answer(FABRICATED)), CHUNKS),
    )
    emitted = _stream(d, alice, thread_id)
    claims = [e for e in emitted if "verified" in e]
    assert claims  # the fixture must actually exercise a claim event
    assert all(claim["verified"] is True for claim in claims)


def test_fault_injection_withholds_the_bad_claim_and_keeps_the_good_one_intact(
    schema, alice, thread_id
):
    """GOOD cites marker [1] (22(1)), in the pack, and releases clean.
    FABRICATED cites marker [99] — nothing is packed at that position (only
    2 units are packed here) — so it is withheld, and GOOD's already-released
    event is untouched by the failure that comes after it."""
    d = _streaming_deps(
        schema,
        AnswerGenerator(FakeLLM(answer(GOOD, FABRICATED), answer(FABRICATED)), CHUNKS),
    )
    emitted = _stream(d, alice, thread_id)

    claim_events = [e for e in emitted if "verified" in e]
    withheld_events = [e for e in emitted if "reason" in e]
    assert len(claim_events) == 1
    assert claim_events[0]["id"] == 1
    assert claim_events[0]["text"] == GOOD
    assert claim_events[0]["verified"] is True
    assert len(withheld_events) == 1
    assert withheld_events[0]["id"] == 2
    assert withheld_events[0]["reason"] == "marker_not_in_evidence"


# --- R22 Part B: release only what will be served ------------------------------


def test_a_corrective_retry_streams_only_its_own_pass(schema, alice, thread_id):
    """First pass: nothing retrieved, so RESCUABLE's [1] resolves to nothing
    and the pass serves no grounded claim. None of its events reach the
    stream; the retry's pass (pack = `23`) is the only answer shown."""
    retriever = SimpleNamespace(
        search=lambda query, k: (
            [ScoredChunk(chunk=CHUNKS["23"], score=1.0)] if k == RETRY_POOL else []
        )
    )
    d = _streaming_deps(
        schema,
        AnswerGenerator(
            FakeLLM(answer(RESCUABLE), answer(RESCUABLE), answer(RESCUABLE)), CHUNKS
        ),
    )
    d.retriever = retriever
    emitted = _stream(d, alice, thread_id)

    claims = [e for e in emitted if "verified" in e]
    assert [claim["text"] for claim in claims] == [RESCUABLE]
    assert all(claim["verified"] is True for claim in claims)
    assert not [e for e in emitted if "reason" in e]  # first pass's withheld rows
    stages = [e["stage"] for e in emitted if "stage" in e]
    refining = stages.index("refining search")
    first_claim = emitted.index(claims[0])
    assert (
        emitted.index({"stage": "refining search", "facts": None, "chunks": None})
        < first_claim
    )
    assert refining > 0
    assert emitted[-1]["text"] is None


def test_a_turn_gated_after_the_retry_streams_no_claim_or_withheld_row(
    schema, alice, thread_id
):
    d = _streaming_deps(
        schema, AnswerGenerator(FakeLLM(*([answer(RESCUABLE)] * 4)), CHUNKS)
    )
    d.retriever = SimpleNamespace(search=lambda query, k: [])
    emitted = _stream(d, alice, thread_id)

    assert not [e for e in emitted if "verified" in e or "reason" in e]
    assert emitted[-1]["text"] == INSUFFICIENT_EVIDENCE_MESSAGE
    assert emitted[-1]["citations"] == ()
