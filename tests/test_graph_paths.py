"""Step 13.6 — path coverage through the compiled graph (PLAN's ~7 scenarios):
compute, clarify, text-only, prohibited, adjacent, corrective rescue,
corrective still withheld. Reuses `test_graph_nodes.py`'s `deps` helper,
`test_scope.py`'s `BASE` facts and `test_verifier.py`'s `CHUNKS` fixture
rather than building a second fixture set.
"""

from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace

import pytest

from conftest import register_account
from taxverity.calculator.scope import Route
from taxverity.facts import FactField, UserFacts
from taxverity.generation.claims import ClaimEvent
from taxverity.generation.generate import AnswerGenerator
from taxverity.graph.build import build_graph
from taxverity.graph.nodes import RETRY_POOL
from taxverity.llm.extract import ExtractionResult
from taxverity.retrieval.base import ScoredChunk
from taxverity.safety.classifier import FIXED_RESPONSES, ScopeCategory
from taxverity.safety.evidence_gate import INSUFFICIENT_EVIDENCE_MESSAGE
from taxverity.threads.store import create_thread, list_messages
from test_generation import GOOD, answer
from test_graph_nodes import PASSWORD, FakeLLM, deps
from test_scope import BASE
from test_scope import fact as make_fact
from test_verifier import CHUNKS, QUESTION

# Cites marker [1] — unresolvable on a first pass over an empty pack (nothing
# retrieved yet), resolvable once the corrective retry's wider pool packs
# `23` as the pack's only (and therefore first) unit.
RESCUABLE = "- Arrears received are taxed under this provision [1]."

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


def _in_scope_deps(schema, **kwargs) -> object:
    base = dict(
        conn=schema,
        classifier=SimpleNamespace(
            classify=lambda q: SimpleNamespace(category=ScopeCategory.IN_SCOPE, response=None, search_query=q)
        ),
        contextualizer=SimpleNamespace(
            contextualize=lambda q, prior: SimpleNamespace(query=q, rewritten=False, completion=None)
        ),
    )
    base.update(kwargs)
    return deps(**base)


def _facts(missing: tuple[FactField, ...] = (), overrides: dict[FactField, object] | None = None):
    values = dict(BASE)
    for field_ in missing:
        del values[field_]
    values.update(overrides or {})
    return UserFacts(facts=tuple(make_fact(name, value) for name, value in values.items()))


def _extractor_for(
    missing: tuple[FactField, ...] = (), overrides: dict[FactField, object] | None = None
):
    return SimpleNamespace(
        extract=lambda turn: ExtractionResult(
            facts=_facts(missing, overrides),
            rejections=(),
            repairable=(),
            repaired=False,
            completions=(),
        )
    )


def _fixed_retriever(results):
    return SimpleNamespace(search=lambda query, k: results)


# --- compute -------------------------------------------------------------


def test_compute_path_serves_a_claim_and_a_computation(schema, alice, thread_id):
    d = _in_scope_deps(
        schema,
        extractor=_extractor_for(),
        retriever=_fixed_retriever(PACK_RESULTS),
        generator=AnswerGenerator(FakeLLM(answer(GOOD)), CHUNKS),
    )
    result = build_graph(d).invoke({"user_id": alice, "thread_id": thread_id, "question": QUESTION})
    assert result["scope_decision"].route is Route.COMPUTE
    assert result["computation"] is not None
    assert result["final"].route == "compute"
    assert [type(e) for e in result["events"]] == [ClaimEvent]


# --- clarify ---------------------------------------------------------------


def test_clarify_path_asks_and_answers_text_only(schema, alice, thread_id):
    d = _in_scope_deps(
        schema,
        extractor=_extractor_for(missing=(FactField.SALARY_INCOME,)),
        retriever=_fixed_retriever(PACK_RESULTS),
        generator=AnswerGenerator(FakeLLM(answer(GOOD)), CHUNKS),
    )
    result = build_graph(d).invoke({"user_id": alice, "thread_id": thread_id, "question": QUESTION})
    assert result["scope_decision"].route is Route.INCOMPLETE
    assert result["computation"] is None
    assert result["clarify_questions"] != ()
    assert result["final"].route == "incomplete"
    assert result["final"].computation is None


# --- text-only ---------------------------------------------------------------


def test_text_only_path_answers_with_no_computation(schema, alice, thread_id):
    d = _in_scope_deps(
        schema,
        extractor=_extractor_for(overrides={FactField.HOUSE_PROPERTY_INCOME: Decimal("-50000")}),
        retriever=_fixed_retriever(PACK_RESULTS),
        generator=AnswerGenerator(FakeLLM(answer(GOOD)), CHUNKS),
    )
    result = build_graph(d).invoke({"user_id": alice, "thread_id": thread_id, "question": QUESTION})
    assert result["scope_decision"].route is Route.TEXT_ONLY
    assert result["computation"] is None
    assert result["final"].route == "text_only"
    assert result["final"].computation is None


# --- prohibited / adjacent (respond_fixed, no retrieval or LLM) --------------


@pytest.mark.parametrize("category", [ScopeCategory.PROHIBITED, ScopeCategory.ADJACENT])
def test_refused_categories_short_circuit_to_the_fixed_template(schema, alice, thread_id, category):
    d = deps(
        conn=schema,
        classifier=SimpleNamespace(
            classify=lambda q: SimpleNamespace(
                category=category, response=FIXED_RESPONSES[category], search_query=q
            )
        ),
        contextualizer=SimpleNamespace(
            contextualize=lambda q, prior: SimpleNamespace(query=q, rewritten=False, completion=None)
        ),
    )
    result = build_graph(d).invoke(
        {"user_id": alice, "thread_id": thread_id, "question": "how do I hide freelance income?"}
    )
    assert result["final"].route == category.value
    assert result["final"].text == FIXED_RESPONSES[category]
    assert result["events"] == []
    messages = list_messages(schema, alice, thread_id)
    assert messages[-1].content == FIXED_RESPONSES[category]


# --- conversational (respond_conversational, no retrieval or verifier) -------


def test_conversational_category_short_circuits_to_a_guarded_reply(schema, alice, thread_id):
    d = deps(
        conn=schema,
        classifier=SimpleNamespace(
            classify=lambda q: SimpleNamespace(
                category=ScopeCategory.CONVERSATIONAL, response=None, search_query=q
            )
        ),
        contextualizer=SimpleNamespace(
            contextualize=lambda q, prior: SimpleNamespace(query=q, rewritten=False, completion=None)
        ),
        conversational=SimpleNamespace(reply=lambda q: "Hello! Ask me about the Act."),
    )
    result = build_graph(d).invoke(
        {"user_id": alice, "thread_id": thread_id, "question": "hi there"}
    )
    assert result["final"].route == "conversational"
    assert result["final"].text == "Hello! Ask me about the Act."
    assert result["final"].citations == ()
    assert result["events"] == []
    messages = list_messages(schema, alice, thread_id)
    assert messages[-1].content == "Hello! Ask me about the Act."


# --- corrective loop ---------------------------------------------------------


def test_corrective_loop_rescues_a_first_pass_with_no_evidence(schema, alice, thread_id):
    """First pass: retriever returns nothing, pack is empty, marker [1] in
    RESCUABLE resolves to nothing, the gate withholds. Retry (k =
    RETRY_POOL): retriever finds `23`, which becomes the pack's own unit
    [1], so the same line now resolves and the retried pass serves a real
    claim. The LLM's streamed text can be identical both times — what
    changes between passes is the evidence pack, not the model output."""
    retriever = SimpleNamespace(
        search=lambda query, k: (
            [ScoredChunk(chunk=CHUNKS["23"], score=1.0)] if k == RETRY_POOL else []
        )
    )
    d = _in_scope_deps(
        schema,
        extractor=_extractor_for(),
        retriever=retriever,
        generator=AnswerGenerator(FakeLLM(answer(RESCUABLE)), CHUNKS),
    )
    result = build_graph(d).invoke({"user_id": alice, "thread_id": thread_id, "question": QUESTION})
    assert result["retried"] is True
    assert [type(e) for e in result["events"]] == [ClaimEvent]
    assert result["final"].citations == ("23",)


def test_corrective_loop_still_withholds_when_the_retry_finds_nothing(schema, alice, thread_id):
    """Both passes retrieve nothing: the retry runs once (bounded), and the
    turn is still withheld rather than looping or fabricating."""
    d = _in_scope_deps(
        schema,
        extractor=_extractor_for(),
        retriever=_fixed_retriever([]),
        generator=AnswerGenerator(FakeLLM(answer(RESCUABLE)), CHUNKS),
    )
    result = build_graph(d).invoke({"user_id": alice, "thread_id": thread_id, "question": QUESTION})
    assert result["retried"] is True
    assert not any(isinstance(e, ClaimEvent) for e in result["events"])  # nothing was ever grounded
    assert result["final"].text == INSUFFICIENT_EVIDENCE_MESSAGE
    assert result["final"].searched == ()
    messages = list_messages(schema, alice, thread_id)
    assert messages[-1].content == INSUFFICIENT_EVIDENCE_MESSAGE


# --- trace panel (R19) ------------------------------------------------------


def test_trace_accumulates_one_entry_per_node_run(schema, alice, thread_id):
    """The retry cycle runs `retrieve_retry` and `generate_verify` twice, and
    the trace shows it — a duplicate node name in the trace *is* the evidence
    the corrective loop fired, not a bug to dedupe."""
    d = _in_scope_deps(
        schema,
        extractor=_extractor_for(),
        retriever=_fixed_retriever([]),
        generator=AnswerGenerator(FakeLLM(answer(RESCUABLE)), CHUNKS),
    )
    result = build_graph(d).invoke({"user_id": alice, "thread_id": thread_id, "question": QUESTION})
    names = [entry["node"] for entry in result["trace"]]
    assert names.count("generate_verify") == 2
    assert names.count("retrieve_retry") == 1
    assert all(isinstance(entry["ms"], float) for entry in result["trace"])
    # `final.trace` is built inside `finalize` from the trace accumulated so
    # far, so it is everything but `finalize`'s own (not-yet-measured) entry.
    assert [t.node for t in result["final"].trace] == names[:-1]
    assert names[-1] == "finalize"


def test_finalize_persists_trace_and_withheld_reasons_on_the_message(schema, alice, thread_id):
    d = _in_scope_deps(
        schema,
        extractor=_extractor_for(missing=(FactField.SALARY_INCOME,)),
        retriever=_fixed_retriever(PACK_RESULTS),
        generator=AnswerGenerator(FakeLLM(answer(GOOD)), CHUNKS),
    )
    build_graph(d).invoke({"user_id": alice, "thread_id": thread_id, "question": QUESTION})
    message = list_messages(schema, alice, thread_id)[-1]
    payload = dict(message.payload)
    assert payload["trace"]
    assert payload["trace"][0].keys() == {"node", "ms"}
    assert payload["clarify_questions"]  # this path asked a clarifying question
    assert "withheld" in payload
