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
from taxverity.generation.claims import ClaimEvent, ClaimType, WithheldEvent
from taxverity.generation.generate import AnswerGenerator
from taxverity.graph.build import build_graph
from taxverity.graph.nodes import RETRY_POOL
from taxverity.llm.extract import ExtractionResult
from taxverity.reasoning.models import (
    AnswerPlan,
    CheckStatus,
    ConclusionKind,
    Condition,
    ConditionCheck,
    LegalRule,
    MissingFact,
    ReasoningAnalysis,
)
from taxverity.retrieval.base import ScoredChunk
from taxverity.safety.classifier import FIXED_RESPONSES, Intent, ScopeCategory
from taxverity.safety.evidence_gate import INSUFFICIENT_EVIDENCE_MESSAGE
from taxverity.threads.store import create_thread, list_messages
from test_generation import APPLICATION_LINE, GOOD, UNKNOWN_LINE, answer
from test_graph_nodes import PASSWORD, FakeLLM, StubReasoner, deps, reason_result
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


def _in_scope_deps_with_intent(schema, intent: Intent, **kwargs) -> object:
    """R20 Step 20.9: like `_in_scope_deps`, but the classifier stub also
    sets `intent`, so `reason` (20.5) actually runs on the path being
    tested rather than skipping to `_NO_ANALYSIS` for the default
    `Intent.EXPLANATION`."""
    kwargs.setdefault(
        "classifier",
        SimpleNamespace(
            classify=lambda q: SimpleNamespace(
                category=ScopeCategory.IN_SCOPE, response=None, search_query=q, intent=intent
            )
        ),
    )
    return _in_scope_deps(schema, **kwargs)


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
        # First pass: generate + its one repair call, both against an empty
        # pack, both still fail. Retry pass: generate against the wider
        # pool grounds cleanly, no repair needed. 3 `complete()` calls.
        generator=AnswerGenerator(
            FakeLLM(answer(RESCUABLE), answer(RESCUABLE), answer(RESCUABLE)), CHUNKS
        ),
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
        # Both passes run against an empty pack: generate + repair each
        # time, all 4 still fail.
        generator=AnswerGenerator(
            FakeLLM(*([answer(RESCUABLE)] * 4)), CHUNKS
        ),
    )
    result = build_graph(d).invoke({"user_id": alice, "thread_id": thread_id, "question": QUESTION})
    assert result["retried"] is True
    assert not any(isinstance(e, ClaimEvent) for e in result["events"])  # nothing was ever grounded
    assert result["final"].text == INSUFFICIENT_EVIDENCE_MESSAGE
    assert result["final"].searched == ()
    messages = list_messages(schema, alice, thread_id)
    assert messages[-1].content == INSUFFICIENT_EVIDENCE_MESSAGE


# --- reasoning path (R20 Step 20.9: reason -> decide -> generate_verify) ----


def test_reasoning_path_serves_a_grounded_application_claim(schema, alice, thread_id):
    """A reasoning intent runs `reason`, a satisfied condition survives
    validation, and `generate_verify` gets `analysis=` — an APPLICATION line
    citing the same marker the condition is grounded on is served, not
    withheld, proving `analysis` actually reached the verifier through the
    graph (not just in the unit tests)."""
    analysis = ReasoningAnalysis(
        legal_rules=(
            LegalRule(
                id="r1",
                markers=(1,),
                rule="Thirty per cent of the annual value is deductible.",
                conditions=(Condition(id="c1", text="x", markers=(1,)),),
            ),
        ),
        applicability=(
            ConditionCheck(
                condition_id="c1", status=CheckStatus.SATISFIED, fact_refs=("salary_income",)
            ),
        ),
        answer_plan=AnswerPlan(conclusion_kind=ConclusionKind.CONDITIONAL),
    )
    stub = StubReasoner(reason_result(analysis))
    d = _in_scope_deps_with_intent(
        schema,
        Intent.ELIGIBILITY,
        extractor=_extractor_for(),
        retriever=_fixed_retriever(PACK_RESULTS),
        reasoner=stub,
        generator=AnswerGenerator(FakeLLM(answer(APPLICATION_LINE)), CHUNKS),
    )
    result = build_graph(d).invoke({"user_id": alice, "thread_id": thread_id, "question": QUESTION})
    assert stub.calls  # `reason` actually ran, not skipped
    assert [type(e) for e in result["events"]] == [ClaimEvent]
    assert result["events"][0].type is ClaimType.APPLICATION


def test_reasoning_path_withholds_an_application_claim_the_analysis_contradicts(schema, alice, thread_id):
    """The mirror case: a condition the analysis marks unsatisfied gates an
    APPLICATION claim that affirms it anyway (`UNSUPPORTED_APPLICATION`),
    proving the gate reaches all the way through the compiled graph, not
    only `Verifier` in isolation."""
    analysis = ReasoningAnalysis(
        legal_rules=(
            LegalRule(
                id="r1",
                markers=(1,),
                rule="Thirty per cent of the annual value is deductible.",
                conditions=(Condition(id="c1", text="x", markers=(1,)),),
            ),
        ),
        applicability=(
            ConditionCheck(
                condition_id="c1", status=CheckStatus.NOT_SATISFIED, fact_refs=("salary_income",)
            ),
        ),
        answer_plan=AnswerPlan(conclusion_kind=ConclusionKind.CONDITIONAL),
    )
    stub = StubReasoner(reason_result(analysis))
    d = _in_scope_deps_with_intent(
        schema,
        Intent.ELIGIBILITY,
        extractor=_extractor_for(),
        retriever=_fixed_retriever(PACK_RESULTS),
        reasoner=stub,
        # A withheld-only pass grounds nothing (`served_grounded_claims`
        # counts only a *served* content/application claim), so the
        # corrective retry fires once — every one of both passes' generate
        # + one repair call affirms the same unsupported conclusion, so all
        # 4 still fail.
        generator=AnswerGenerator(FakeLLM(*([answer(APPLICATION_LINE)] * 4)), CHUNKS),
    )
    result = build_graph(d).invoke({"user_id": alice, "thread_id": thread_id, "question": QUESTION})
    assert result["retried"] is True
    assert len(stub.calls) == 1  # `reason` ran once, before the retry loop
    assert [type(e) for e in result["events"]] == [WithheldEvent]
    assert result["events"][0].reason == "unsupported_application"


def test_reasoning_path_asks_a_material_missing_fact_and_serves_an_unknown_claim(
    schema, alice, thread_id
):
    """`decide` (20.6) turns a material `MissingFact` into its own clarify
    question, on top of whatever `route_calc`'s deterministic templates
    already asked, and `generate_verify` serves the matching UNKNOWN claim
    — end to end through the compiled graph, not just the two nodes in
    isolation. An UNKNOWN claim never counts as grounded (same footing as
    `no_basis`/`computation`, `evidence_gate.GROUNDED_CLAIM_TYPES`), so an
    unknown-only pass still triggers the corrective retry and the final
    text is still the fixed insufficient-evidence message — consistent
    with how a computation-only pass already behaves (evidence_gate.py's
    own docstring)."""
    question = "Have you already used part of this cap earlier in the tax year?"
    analysis = ReasoningAnalysis(
        legal_rules=(
            LegalRule(
                id="r1",
                markers=(2,),
                rule="The deduction is capped at Rs. 2,00,000.",
                conditions=(Condition(id="c1", text="x", markers=(2,)),),
            ),
        ),
        applicability=(ConditionCheck(condition_id="c1", status=CheckStatus.UNKNOWN),),
        missing_facts=(MissingFact(condition_id="c1", question=question, material=True),),
        answer_plan=AnswerPlan(conclusion_kind=ConclusionKind.CONDITIONAL),
    )
    stub = StubReasoner(reason_result(analysis))
    d = _in_scope_deps_with_intent(
        schema,
        Intent.ELIGIBILITY,
        extractor=_extractor_for(),
        retriever=_fixed_retriever(PACK_RESULTS),
        reasoner=stub,
        # An UNKNOWN claim passes verification cleanly both times (no
        # repair needed), but each pass still grounds nothing, so the
        # retry fires once: 2 calls total.
        generator=AnswerGenerator(FakeLLM(answer(UNKNOWN_LINE), answer(UNKNOWN_LINE)), CHUNKS),
    )
    result = build_graph(d).invoke({"user_id": alice, "thread_id": thread_id, "question": QUESTION})
    assert question in result["clarify_questions"]
    assert result["retried"] is True
    assert len(stub.calls) == 1  # `reason` ran once, before the retry loop
    assert [type(e) for e in result["events"]] == [ClaimEvent]
    assert result["events"][0].type is ClaimType.UNKNOWN
    assert result["final"].text == INSUFFICIENT_EVIDENCE_MESSAGE


def test_a_non_reasoning_intent_never_touches_the_reasoner(schema, alice, thread_id):
    """The default `Intent.EXPLANATION` (every classifier stub elsewhere in
    this file leaves `intent` unset) must reach `generate_verify` with no
    analysis at all — `reason` skips the reasoner outright, proving the
    graph wiring, not just the node in isolation, respects the intent gate."""
    d = _in_scope_deps(
        schema,
        extractor=_extractor_for(),
        retriever=_fixed_retriever(PACK_RESULTS),
        reasoner=SimpleNamespace(reason=_boom_reason),
        generator=AnswerGenerator(FakeLLM(answer(GOOD)), CHUNKS),
    )
    result = build_graph(d).invoke({"user_id": alice, "thread_id": thread_id, "question": QUESTION})
    assert result["legal_rules"] == ()
    assert [type(e) for e in result["events"]] == [ClaimEvent]


def _boom_reason(*_args: object, **_kwargs: object) -> None:
    raise AssertionError("the reasoner must not be called for a non-reasoning intent")


def test_the_corrective_retry_reuses_the_first_passs_analysis_without_re_reasoning(
    schema, alice, thread_id
):
    """Step 13.5's retry only widens the evidence pack (ADR-033/PLAN 13.5);
    it re-enters at `retrieve_retry -> generate_verify`, never back through
    `reason`. An empty first-pass pack grounds no rule at all, so
    `analysis` stays `None` through both passes — and the reasoner is
    called exactly once, not twice, even though `generate_verify` runs
    twice."""
    analysis = ReasoningAnalysis(
        legal_rules=(
            LegalRule(id="r1", markers=(1,), rule="x", conditions=(Condition(id="c1", text="x"),)),
        ),
        answer_plan=AnswerPlan(conclusion_kind=ConclusionKind.CONDITIONAL),
    )
    stub = StubReasoner(reason_result(analysis))
    retriever = SimpleNamespace(
        search=lambda query, k: (
            [ScoredChunk(chunk=CHUNKS["23"], score=1.0)] if k == RETRY_POOL else []
        )
    )
    d = _in_scope_deps_with_intent(
        schema,
        Intent.ELIGIBILITY,
        extractor=_extractor_for(),
        retriever=retriever,
        reasoner=stub,
        generator=AnswerGenerator(
            FakeLLM(answer(RESCUABLE), answer(RESCUABLE), answer(RESCUABLE)), CHUNKS
        ),
    )
    result = build_graph(d).invoke({"user_id": alice, "thread_id": thread_id, "question": QUESTION})
    assert len(stub.calls) == 1  # `reason` ran once, off the first (empty) pack
    assert result["legal_rules"] == ()  # marker 1 didn't exist in that empty pack
    assert result["retried"] is True
    assert [type(e) for e in result["events"]] == [ClaimEvent]


# --- trace panel (R19) ------------------------------------------------------


def test_trace_accumulates_one_entry_per_node_run(schema, alice, thread_id):
    """The retry cycle runs `retrieve_retry` and `generate_verify` twice, and
    the trace shows it — a duplicate node name in the trace *is* the evidence
    the corrective loop fired, not a bug to dedupe."""
    d = _in_scope_deps(
        schema,
        extractor=_extractor_for(),
        retriever=_fixed_retriever([]),
        generator=AnswerGenerator(FakeLLM(*([answer(RESCUABLE)] * 4)), CHUNKS),
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
