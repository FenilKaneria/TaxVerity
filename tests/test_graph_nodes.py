"""Steps 13.1-13.3 — graph state, node wrappers and the compiled graph's edges.

Each node is tested directly first (PLAN 13.2: thin wrappers, no logic moved
into the graph), with a fake `writer` so no graph context is needed. The
DB-touching nodes (`load_thread`, `merge_facts`, `finalize`) run against a
real throwaway database, the same discipline `test_threads.py` and
`test_fact_state.py` already use. A handful of end-to-end runs through
`build_graph` (13.1/13.3) close the loop: the dedicated ~7-scenario path
suite PLAN reserves for Step 13.6 is not duplicated here.
"""

from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace

import pytest

from conftest import register_account
from taxverity.calculator.scope import Route
from taxverity.facts import FactField, FactStatus, UserFacts
from taxverity.generation.claims import ClaimEvent, ClaimType
from taxverity.generation.generate import AnswerGenerator
from taxverity.graph.build import build_graph
from taxverity.graph.nodes import (
    classify,
    decide,
    extract_facts,
    finalize,
    generate_verify,
    load_thread,
    may_state_facts,
    merge_facts,
    reason,
    respond_conversational,
    respond_fixed,
    retrieve,
    route_calc,
)
from taxverity.graph.state import (
    CALCULATION_CLARIFY,
    CLARIFY_TEMPLATES,
    OTHER_INCOME_CLARIFY,
    ClarifyEvent,
    FinalEvent,
    GraphDeps,
    StageEvent,
)
from taxverity.llm.client import Completion, Usage
from taxverity.llm.extract import ExtractionResult
from taxverity.memory.fact_state import ThreadFactState, load_fact_state
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
from taxverity.reasoning.reason import ReasonResult
from taxverity.retrieval.base import ScoredChunk
from taxverity.retrieval.evidence import EvidencePacker
from taxverity.safety.classifier import FIXED_RESPONSES, Intent, ScopeCategory
from taxverity.safety.evidence_gate import INSUFFICIENT_EVIDENCE_MESSAGE
from taxverity.threads.store import append_message, create_thread, list_messages
from test_generation import ALSO_GOOD, GOOD, QUESTION, UNSUPPORTED, FakeLLM, answer
from test_scope import BASE, fact
from test_verifier import CHUNKS, PACK

PASSWORD = "correct horse battery"


class Recorder:
    """A fake `writer`, and a fake for anything a short-circuit path must
    never call — any attribute access returns a callable that fails."""

    def __init__(self) -> None:
        self.events: list[object] = []

    def __call__(self, event: object) -> None:
        self.events.append(event)


class Boom:
    def __getattr__(self, name: str):
        def _fail(*_args: object, **_kwargs: object) -> None:
            raise AssertionError(f"{name} must not be called on this path")

        return _fail


def facts_from(overrides: dict[FactField, object] | None = None) -> UserFacts:
    values = {**BASE, **(overrides or {})}
    return UserFacts(facts=tuple(fact(name, value) for name, value in values.items()))


def thread_state(overrides: dict[FactField, object] | None = None) -> ThreadFactState:
    facts = facts_from(overrides).facts
    return ThreadFactState(
        facts={f.field: f for f in facts},
        provenance={f.field: "stated in turn 1" for f in facts},
    )


def deps(**kwargs: object) -> GraphDeps:
    base = dict(
        conn=None,
        chunks=CHUNKS,
        retriever=Boom(),
        packer=EvidencePacker(CHUNKS.values()),
        classifier=Boom(),
        extractor=Boom(),
        generator=AnswerGenerator(FakeLLM(""), CHUNKS),
        conversational=Boom(),
        reasoner=Boom(),
    )
    base.update(kwargs)
    return GraphDeps(**base)  # type: ignore[arg-type]


# --- extract_facts: R22 Part A's deterministic skip (no DB, no LLM) ---------


@pytest.mark.parametrize(
    "turn",
    [
        "What does section 22(1)(b) allow?",
        "What does Schedule XV(1) of the Income-tax Act, 2025 cover?",
        "I didn't understand that. Can you explain it in simple terms?",
        "Can you give me an example?",
        "How do I e-verify a return?",
    ],
)
def test_a_turn_with_no_figure_and_no_fact_cue_states_no_fact(turn):
    assert not may_state_facts(turn)


@pytest.mark.parametrize(
    "turn",
    [
        "Salary 15 lakh, no other income.",
        "I earn fifteen lakh a year.",
        "I'm 65 and retired.",
        "I sold my flat last year.",
        "We sold the plot.",
        "Our family pays rent.",
    ],
)
def test_a_figure_or_a_fact_cue_may_state_a_fact(turn):
    assert may_state_facts(turn)


def test_extract_facts_skips_the_call_for_a_turn_that_states_nothing():
    state = {"question": "What does section 22 allow?", "intent": Intent.EXPLANATION}
    result = extract_facts(state, deps(extractor=Boom()), writer=Recorder())
    extraction = result["extraction"]
    assert extraction.completions == ()
    assert all(f.status is FactStatus.MISSING for f in extraction.facts.facts)
    assert {f.field for f in extraction.facts.facts} == set(FactField)


def test_extract_facts_still_calls_for_a_turn_stating_a_figure():
    called: list[str] = []
    extractor = SimpleNamespace(extract=lambda turn: called.append(turn) or "x")
    state = {"question": "Salary 15 lakh, how much tax?", "intent": Intent.EXPLANATION}
    extract_facts(state, deps(extractor=extractor), writer=Recorder())
    assert called == ["Salary 15 lakh, how much tax?"]


def test_extract_facts_always_calls_for_a_reasoning_intent():
    # A reasoning answer applies the person's facts, so it never skips —
    # even on a turn with no figure and no cue.
    called: list[str] = []
    extractor = SimpleNamespace(extract=lambda turn: called.append(turn) or "x")
    state = {
        "question": "Can a senior citizen claim this?",
        "intent": Intent.ELIGIBILITY,
    }
    extract_facts(state, deps(extractor=extractor), writer=Recorder())
    assert called == ["Can a senior citizen claim this?"]


# --- route_calc (no DB, no LLM) ----------------------------------------------


def test_route_calc_computes_when_every_input_is_known():
    recorder = Recorder()
    result = route_calc({"fact_state": thread_state()}, deps(), writer=recorder)
    assert result["scope_decision"].route is Route.COMPUTE
    assert result["computation"] is not None
    assert result["clarify_questions"] == ()
    assert recorder.events == []  # no clarify event when nothing is asked


def test_route_calc_routes_text_only_and_carries_no_computation():
    state = thread_state({FactField.HOUSE_PROPERTY_INCOME: Decimal("-50000")})
    result = route_calc({"fact_state": state}, deps(), writer=Recorder())
    assert result["scope_decision"].route is Route.TEXT_ONLY
    assert result["computation"] is None


def test_route_calc_asks_via_deterministic_templates_never_the_llm():
    values = dict(BASE)
    del values[FactField.SALARY_INCOME]
    facts = UserFacts(facts=tuple(fact(name, value) for name, value in values.items()))
    state = ThreadFactState(
        facts={f.field: f for f in facts.facts},
        provenance={f.field: "stated in turn 1" for f in facts.facts},
    )
    recorder = Recorder()
    result = route_calc({"fact_state": state}, deps(), writer=recorder)
    assert result["scope_decision"].route is Route.INCOMPLETE
    assert result["computation"] is None
    assert result["clarify_questions"] == (CLARIFY_TEMPLATES[FactField.SALARY_INCOME],)
    assert recorder.events == [
        ClarifyEvent(questions=result["clarify_questions"]).model_dump()
    ]


def test_route_calc_skips_the_probe_when_the_thread_has_stated_no_amount_yet():
    """R19 Phase C: a fresh thread with no amount fact at all must not be
    interrogated with 6-7 clarify questions on its first turn."""
    recorder = Recorder()
    result = route_calc({"fact_state": ThreadFactState()}, deps(), writer=recorder)
    assert result["scope_decision"].route is Route.INCOMPLETE
    assert result["computation"] is None
    assert result["clarify_questions"] == ()
    assert recorder.events == []  # no clarify event either


def salary_only_state(**extra):
    stated = {FactField.SALARY_INCOME: Decimal("1000000"), **extra}
    return ThreadFactState(
        facts={name: fact(name, value) for name, value in stated.items()},
        provenance={name: "stated in turn 1" for name in stated},
    )


def test_a_stated_salary_alone_asks_residency_and_other_income_not_six_heads():
    """The 10-lakh case: residency moves the tax (only a resident gets the
    rebate), so it is asked and the figure waits; the four heads and other
    sources are one combined question, not five chips."""
    recorder = Recorder()
    result = route_calc({"fact_state": salary_only_state()}, deps(), writer=recorder)
    assert result["scope_decision"].route is Route.INCOMPLETE
    assert result["computation"] is None
    assert result["assumed_nil"] == ()
    assert result["clarify_questions"] == (
        CLARIFY_TEMPLATES[FactField.RESIDENTIAL_STATUS],
        OTHER_INCOME_CLARIFY,
    )
    assert recorder.events == [
        ClarifyEvent(questions=result["clarify_questions"]).model_dump()
    ]


def test_a_resident_salary_gets_a_provisional_figure_and_one_question():
    state = salary_only_state(**{FactField.RESIDENTIAL_STATUS: "resident"})
    result = route_calc({"fact_state": state}, deps(), writer=Recorder())
    assert result["computation"].comparison.under_202_1.payable.amount == 0
    assert FactField.OTHER_SOURCES_INCOME in result["assumed_nil"]
    assert FactField.RESIDENTIAL_STATUS not in result["assumed_nil"]
    assert result["clarify_questions"] == (OTHER_INCOME_CLARIFY,)


def test_no_question_once_everything_that_matters_is_known():
    zero = Decimal(0)
    state = salary_only_state(
        **{
            FactField.RESIDENTIAL_STATUS: "resident",
            FactField.HOUSE_PROPERTY_INCOME: zero,
            FactField.BUSINESS_INCOME: zero,
            FactField.CAPITAL_GAINS_SHORT_TERM: zero,
            FactField.CAPITAL_GAINS_LONG_TERM: zero,
            FactField.OTHER_SOURCES_INCOME: zero,
        }
    )
    recorder = Recorder()
    result = route_calc({"fact_state": state}, deps(), writer=recorder)
    assert result["computation"] is not None
    assert result["clarify_questions"] == ()
    assert recorder.events == []


def test_route_calc_computes_when_incomplete_resolves_with_nothing_left_to_ask():
    values = dict(BASE)
    del values[FactField.TDS_PAID]
    del values[FactField.ADVANCE_TAX_PAID]
    facts = UserFacts(facts=tuple(fact(name, value) for name, value in values.items()))
    state = ThreadFactState(
        facts={f.field: f for f in facts.facts},
        provenance={f.field: "stated in turn 1" for f in facts.facts},
    )
    recorder = Recorder()
    result = route_calc({"fact_state": state}, deps(), writer=recorder)
    assert result["scope_decision"].route is Route.INCOMPLETE
    assert result["computation"] is not None
    assert result["clarify_questions"] == ()
    assert recorder.events == []


# --- respond_fixed and classify (no DB, no LLM stream) -----------------------


def test_respond_fixed_carries_the_classifiers_template_and_touches_nothing_else():
    state = {"fixed_response": FIXED_RESPONSES[ScopeCategory.PROHIBITED]}
    result = respond_fixed(state, deps())
    assert result == {
        "answer_text": FIXED_RESPONSES[ScopeCategory.PROHIBITED],
        "events": [],
        "computation": None,
        "clarify_questions": (),
    }


def test_respond_conversational_carries_the_guarded_reply_and_touches_nothing_else():
    conversational = SimpleNamespace(reply=lambda q: "Hi! Ask me about the Act.")
    state = {"query": "hi there"}
    result = respond_conversational(state, deps(conversational=conversational))
    assert result == {
        "answer_text": "Hi! Ask me about the Act.",
        "events": [],
        "computation": None,
        "clarify_questions": (),
    }


def test_classify_reads_the_category_and_response_off_the_classifier():
    classifier = SimpleNamespace(
        classify=lambda q, **_: SimpleNamespace(
            category=ScopeCategory.PROHIBITED,
            response=FIXED_RESPONSES[ScopeCategory.PROHIBITED],
        )
    )
    result = classify(
        {"question": "how do I hide income?"}, deps(classifier=classifier)
    )
    assert result == {
        "query": "how do I hide income?",
        "category": ScopeCategory.PROHIBITED,
        "fixed_response": FIXED_RESPONSES[ScopeCategory.PROHIBITED],
        "search_query": "how do I hide income?",
        "sub_queries": (),
        "intent": Intent.EXPLANATION,
        "reopened": False,
    }


def test_classify_uses_the_classifiers_search_query_when_it_sets_one():
    classifier = SimpleNamespace(
        classify=lambda q, **_: SimpleNamespace(
            category=ScopeCategory.IN_SCOPE,
            response=None,
            search_query="interest on borrowed capital; house property",
        )
    )
    result = classify(
        {"question": "home loan tax benefit"}, deps(classifier=classifier)
    )
    assert result["search_query"] == "interest on borrowed capital; house property"


def test_classify_carries_the_classifiers_sub_queries():
    classifier = SimpleNamespace(
        classify=lambda q, **_: SimpleNamespace(
            category=ScopeCategory.IN_SCOPE,
            response=None,
            search_query=q,
            sub_queries=("the slab rates", "the standard deduction"),
        )
    )
    result = classify({"question": "what tax do I pay?"}, deps(classifier=classifier))
    assert result["sub_queries"] == ("the slab rates", "the standard deduction")


def test_classify_carries_the_classifiers_intent():
    classifier = SimpleNamespace(
        classify=lambda q, **_: SimpleNamespace(
            category=ScopeCategory.IN_SCOPE,
            response=None,
            search_query=q,
            sub_queries=(),
            intent=Intent.CALCULATION,
        )
    )
    result = classify(
        {"question": "what tax do I pay on 18L salary?"}, deps(classifier=classifier)
    )
    assert result["intent"] is Intent.CALCULATION


def test_classify_intent_defaults_to_explanation_for_a_stub_predating_it():
    classifier = SimpleNamespace(
        classify=lambda q, **_: SimpleNamespace(
            category=ScopeCategory.IN_SCOPE, response=None
        )
    )
    result = classify(
        {"question": "what does section 19 say?"}, deps(classifier=classifier)
    )
    assert result["intent"] is Intent.EXPLANATION


# --- retrieve (R20 Step 20.2: single search vs. per-sub-query merge) --------


def test_retrieve_searches_once_on_search_query_when_there_are_no_sub_queries():
    calls: list[str] = []

    class FakeRetriever:
        def search(self, query: str, k: int) -> list[ScoredChunk]:
            calls.append(query)
            return [ScoredChunk(chunk=CHUNKS["22"], score=1.0)]

    result = retrieve(
        {"query": "raw", "search_query": "rewritten", "sub_queries": ()},
        deps(retriever=FakeRetriever(), packer=EvidencePacker(CHUNKS.values())),
        writer=Recorder(),
    )
    assert calls == ["rewritten"]
    assert result["trace"] == []
    assert [unit.citation for unit in result["pack"].units] == ["22"]


def test_retrieve_merges_and_dedupes_across_sub_queries():
    """Round-robin by rank, first occurrence wins: issue a's and issue b's
    own top hits both survive; issue b's second hit is issue a's own first
    hit's sibling `23`, already carried, and is dropped as a duplicate."""
    per_query = {
        "issue a": [
            ScoredChunk(chunk=CHUNKS["22(1)(a)"], score=2.0),
            ScoredChunk(chunk=CHUNKS["23"], score=1.0),
        ],
        "issue b": [
            ScoredChunk(chunk=CHUNKS["24"], score=2.0),
            ScoredChunk(chunk=CHUNKS["23"], score=1.0),
        ],
    }

    class FakeRetriever:
        def search(self, query: str, k: int) -> list[ScoredChunk]:
            return per_query[query]

    result = retrieve(
        {"query": "q", "search_query": "q", "sub_queries": ("issue a", "issue b")},
        deps(retriever=FakeRetriever(), packer=EvidencePacker(CHUNKS.values())),
        writer=Recorder(),
    )
    assert [unit.citation for unit in result["pack"].units] == ["22(1)(a)", "24", "23"]
    assert [entry["node"] for entry in result["trace"]] == [
        "retrieve.subquery.1",
        "retrieve.subquery.2",
    ]


def test_sub_query_searches_run_two_at_a_time_and_merge_in_sub_query_order():
    """R21 Part B: concurrency never exceeds Jina's limit of 2, and a
    sub-query finishing first does not reorder the merge."""
    import threading
    import time as clock

    delays = {"issue a": 0.3, "issue b": 0.05, "issue c": 0.05}
    per_query = {
        "issue a": [ScoredChunk(chunk=CHUNKS["22(1)(a)"], score=1.0)],
        "issue b": [ScoredChunk(chunk=CHUNKS["24"], score=1.0)],
        "issue c": [ScoredChunk(chunk=CHUNKS["23"], score=1.0)],
    }
    lock = threading.Lock()
    active = peak = 0

    class SlowRetriever:
        def search(self, query: str, k: int) -> list[ScoredChunk]:
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            clock.sleep(delays[query])
            with lock:
                active -= 1
            return per_query[query]

    started = clock.perf_counter()
    result = retrieve(
        {"query": "q", "search_query": "q", "sub_queries": tuple(delays)},
        deps(retriever=SlowRetriever(), packer=EvidencePacker(CHUNKS.values())),
        writer=Recorder(),
    )
    elapsed = clock.perf_counter() - started
    assert peak == 2
    assert elapsed < sum(delays.values())
    assert [unit.citation for unit in result["pack"].units] == ["22(1)(a)", "24", "23"]
    assert [entry["node"] for entry in result["trace"]] == [
        "retrieve.subquery.1",
        "retrieve.subquery.2",
        "retrieve.subquery.3",
    ]


def test_an_error_in_a_worker_thread_surfaces_rather_than_being_swallowed():
    """Vendor degradation (FallbackRetriever, rerank fallback) lives inside each
    search, so threads change nothing about it; anything else raising is a bug
    and must reach the caller, not vanish in the pool."""

    class Broken:
        def search(self, query: str, k: int) -> list[ScoredChunk]:
            raise RuntimeError("bug")

    with pytest.raises(RuntimeError, match="bug"):
        retrieve(
            {"query": "q", "search_query": "q", "sub_queries": ("a", "b")},
            deps(retriever=Broken(), packer=EvidencePacker(CHUNKS.values())),
            writer=Recorder(),
        )


# --- reason (R20 Step 20.5, no DB, not yet wired into the compiled graph) ----


class StubReasoner:
    def __init__(self, result: ReasonResult) -> None:
        self.result = result
        self.calls: list[tuple] = []

    def reason(self, question, pack, fact_state, computation):
        self.calls.append((question, pack, fact_state, computation))
        return self.result


def reason_result(analysis: ReasoningAnalysis | None) -> ReasonResult:
    return ReasonResult(
        analysis=analysis,
        completion=Completion(
            text="",
            provider="test",
            model="test",
            finish_reason="stop",
            usage=Usage(),
            degraded=False,
        ),
    )


GOVERNING_RULE = LegalRule(
    id="r1",
    markers=(2,),
    rule="The deduction is capped at Rs. 2,00,000.",
    conditions=(Condition(id="c1", text="x", markers=(2,)),),
)


def test_reason_skips_explanation_questions_without_touching_the_reasoner():
    state = {
        "intent": Intent.EXPLANATION,
        "query": "x",
        "pack": PACK,
        "fact_state": ThreadFactState(),
    }
    result = reason(state, deps(reasoner=Boom()))
    assert result == {
        "legal_rules": (),
        "applicability": (),
        "missing_facts": (),
        "answer_plan": None,
    }


def test_reason_calls_the_reasoner_for_a_reasoning_intent():
    stub = StubReasoner(reason_result(None))
    state = {
        "intent": Intent.ELIGIBILITY,
        "query": QUESTION,
        "pack": PACK,
        "fact_state": ThreadFactState(),
        "computation": None,
    }
    recorder = Recorder()
    reason(state, deps(reasoner=stub), writer=recorder)
    assert stub.calls == [(QUESTION, PACK, ThreadFactState(), None)]
    assert recorder.events == [StageEvent(stage="analysing").model_dump()]


def test_reason_falls_back_when_the_completion_does_not_parse():
    stub = StubReasoner(reason_result(None))
    state = {
        "intent": Intent.CALCULATION,
        "query": "x",
        "pack": PACK,
        "fact_state": ThreadFactState(),
    }
    result = reason(state, deps(reasoner=stub), writer=Recorder())
    assert result["answer_plan"] is None
    assert result["legal_rules"] == ()


def test_reason_falls_back_when_nothing_survives_validation():
    bad_analysis = ReasoningAnalysis(
        legal_rules=(),
        answer_plan=AnswerPlan(conclusion_kind=ConclusionKind.NO_BASIS),
    )
    stub = StubReasoner(reason_result(bad_analysis))
    state = {
        "intent": Intent.COMPARISON,
        "query": "x",
        "pack": PACK,
        "fact_state": ThreadFactState(),
    }
    result = reason(state, deps(reasoner=stub), writer=Recorder())
    assert result["legal_rules"] == ()
    assert result["answer_plan"] is None


def test_reason_returns_the_validated_analysis_when_a_rule_survives():
    analysis = ReasoningAnalysis(
        legal_rules=(GOVERNING_RULE,),
        applicability=(ConditionCheck(condition_id="c1", status=CheckStatus.UNKNOWN),),
        answer_plan=AnswerPlan(conclusion_kind=ConclusionKind.CONDITIONAL),
    )
    stub = StubReasoner(reason_result(analysis))
    state = {
        "intent": Intent.MULTI_ISSUE,
        "query": "x",
        "pack": PACK,
        "fact_state": ThreadFactState(),
    }
    result = reason(state, deps(reasoner=stub), writer=Recorder())
    assert [r.id for r in result["legal_rules"]] == ["r1"]
    assert result["applicability"][0].status is CheckStatus.UNKNOWN
    assert result["answer_plan"].conclusion_kind is ConclusionKind.CONDITIONAL


# --- decide (R20 Step 20.6, no DB, not yet wired into the compiled graph) ----


def test_decide_does_nothing_when_no_missing_facts_and_no_prior_clarify():
    result = decide({}, deps(), writer=Recorder())
    assert result == {"clarify_questions": ()}


def test_decide_stays_silent_on_a_non_material_missing_fact():
    state = {
        "missing_facts": (
            MissingFact(
                condition_id="c1", question="Are you a resident?", material=False
            ),
        )
    }
    recorder = Recorder()
    result = decide(state, deps(), writer=recorder)
    assert result["clarify_questions"] == ()
    assert recorder.events == []


def test_decide_emits_one_question_per_material_missing_fact():
    state = {
        "missing_facts": (
            MissingFact(
                condition_id="c1", question="Are you a resident?", material=True
            ),
            MissingFact(condition_id="c2", question="Ignore me", material=False),
            MissingFact(
                condition_id="c3", question="Do you own the house?", material=True
            ),
        )
    }
    recorder = Recorder()
    result = decide(state, deps(), writer=recorder)
    assert result["clarify_questions"] == (
        "Are you a resident?",
        "Do you own the house?",
    )
    assert recorder.events == [
        ClarifyEvent(
            questions=("Are you a resident?", "Do you own the house?")
        ).model_dump()
    ]


def test_decide_leaves_calculator_clarify_questions_untouched_when_nothing_new():
    """route_calc already emitted its own ClarifyEvent for these — decide
    must not re-announce them."""
    state = {"clarify_questions": ("What is your salary income for the tax year?",)}
    recorder = Recorder()
    result = decide(state, deps(), writer=recorder)
    assert result["clarify_questions"] == (
        "What is your salary income for the tax year?",
    )
    assert recorder.events == []


def test_decide_appends_reasoning_questions_to_calculator_ones():
    state = {
        "clarify_questions": ("What is your salary income for the tax year?",),
        "missing_facts": (
            MissingFact(
                condition_id="c1", question="Do you own the house?", material=True
            ),
        ),
    }
    recorder = Recorder()
    result = decide(state, deps(), writer=recorder)
    assert result["clarify_questions"] == (
        "What is your salary income for the tax year?",
        "Do you own the house?",
    )
    assert recorder.events == [
        ClarifyEvent(questions=("Do you own the house?",)).model_dump()
    ]


def test_decide_deduplicates_a_question_already_present():
    state = {
        "clarify_questions": ("Do you own the house?",),
        "missing_facts": (
            MissingFact(
                condition_id="c1", question="Do you own the house?", material=True
            ),
        ),
    }
    result = decide(state, deps(), writer=Recorder())
    assert result["clarify_questions"] == ("Do you own the house?",)


# --- generate_verify + the evidence gate (no DB) ------------------------------


def test_generate_verify_serves_a_grounded_claim_and_the_gate_stays_silent():
    llm = FakeLLM(answer(GOOD))
    d = deps(generator=AnswerGenerator(llm, CHUNKS))
    state = {
        "query": QUESTION,
        "pack": PACK,
        "fact_state": thread_state(),
        "computation": None,
    }
    recorder = Recorder()
    result = generate_verify(state, d, writer=recorder)
    assert [type(e) for e in result["events"]] == [ClaimEvent]
    assert result["answer_text"] is None
    assert recorder.events == [
        StageEvent(stage="writing").model_dump(),
        result["events"][0].model_dump(),
    ]


def test_generate_verify_announces_the_repair_pass_only_when_one_runs():
    llm = FakeLLM(answer(GOOD, UNSUPPORTED), answer(ALSO_GOOD))
    d = deps(generator=AnswerGenerator(llm, CHUNKS))
    state = {
        "query": QUESTION,
        "pack": PACK,
        "fact_state": thread_state(),
        "computation": None,
    }
    recorder = Recorder()
    generate_verify(state, d, writer=recorder)
    stages = [e["stage"] for e in recorder.events if "stage" in e]
    assert stages == ["writing", "checking"]
    assert len(llm.calls) == 2
    # Released only after the repair: every claim follows the "checking" stage.
    assert all("stage" in e for e in recorder.events[:2])


def test_generate_verify_gates_to_insufficient_evidence_on_zero_grounded_claims():
    llm = FakeLLM("")  # the model emits nothing
    d = deps(generator=AnswerGenerator(llm, CHUNKS))
    state = {
        "query": QUESTION,
        "pack": PACK,
        "fact_state": thread_state(),
        "computation": None,
    }
    result = generate_verify(state, d, writer=Recorder())
    assert result["events"] == []
    assert result["answer_text"] == INSUFFICIENT_EVIDENCE_MESSAGE


# --- DB-backed nodes: load_thread, merge_facts, finalize ----------------------


@pytest.fixture
def alice(schema):
    return register_account(schema, "alice@example.com", PASSWORD)


@pytest.fixture
def thread_id(schema, alice):
    return create_thread(schema, alice, "House property").thread_id


def test_load_thread_reads_the_recent_window_and_a_fresh_fact_state(
    schema, alice, thread_id
):
    append_message(schema, alice, thread_id, "user", "first turn")
    append_message(schema, alice, thread_id, "assistant", "first answer")
    append_message(schema, alice, thread_id, "user", "second turn")
    state = {"user_id": alice, "thread_id": thread_id}
    recorder = Recorder()
    result = load_thread(state, deps(conn=schema), writer=recorder)
    assert result["prior_turns"] == ["first turn", "second turn"]
    assert result["turn"] == 3
    assert result["fact_state"] == ThreadFactState()
    assert recorder.events == [StageEvent(stage="thinking").model_dump()]
    assert result["previous_answer"] == "first answer"


def test_load_thread_strips_markers_from_the_previous_answer(schema, alice, thread_id):
    # R21: the previous answer's [n] numbers named that turn's pack, not this
    # one's — carried forward as plain prose only.
    append_message(schema, alice, thread_id, "user", "q")
    append_message(
        schema,
        alice,
        thread_id,
        "assistant",
        "## Topic\n- Loss is capped [2].\n- Suppose x [1][eg].",
    )
    state = {"user_id": alice, "thread_id": thread_id}
    result = load_thread(state, deps(conn=schema), writer=Recorder())
    assert result["previous_answer"] == "## Topic\n- Loss is capped.\n- Suppose x."


def test_merge_facts_persists_to_the_database_and_emits_the_facts_stage(
    schema, alice, thread_id
):
    extraction = ExtractionResult(
        facts=facts_from({FactField.SALARY_INCOME: Decimal("1500000")}),
        rejections=(),
        repairable=(),
        repaired=False,
        completions=(),
    )
    state = {
        "user_id": alice,
        "thread_id": thread_id,
        "turn": 1,
        "fact_state": ThreadFactState(),
        "extraction": extraction,
    }
    recorder = Recorder()
    result = merge_facts(state, deps(conn=schema), writer=recorder)
    assert result["fact_state"].get(FactField.SALARY_INCOME).value == Decimal("1500000")
    assert load_fact_state(schema, alice, thread_id) == result["fact_state"]
    (event,) = recorder.events
    assert event["stage"] == "facts"
    assert event["facts"][FactField.SALARY_INCOME.value] == "1500000"


def test_finalize_persists_both_messages_and_composes_the_final_event(
    schema, alice, thread_id
):
    claim = ClaimEvent(id=1, type=ClaimType.CONTENT, text=GOOD, citations=())
    state = {
        "user_id": alice,
        "thread_id": thread_id,
        "question": "What can I deduct from house property income?",
        "category": ScopeCategory.IN_SCOPE,
        "scope_decision": SimpleNamespace(route=Route.TEXT_ONLY),
        "computation": None,
        "events": [claim],
    }
    recorder = Recorder()
    result = finalize(state, deps(conn=schema), writer=recorder)
    assert isinstance(result["final"], FinalEvent)
    assert result["final"].route == "text_only"
    assert result["final"].computation is None
    messages = list_messages(schema, alice, thread_id)
    assert [m.role for m in messages] == ["user", "assistant"]
    assert messages[0].content == state["question"]
    assert messages[1].content == GOOD
    assert recorder.events == [result["final"].model_dump()]
    assert result["final"].text is None
    assert result["final"].searched == ()


def test_finalize_streams_the_gated_text_and_the_provisions_searched(
    schema, alice, thread_id
):
    # advisor pivot, Step 4: the insufficient-evidence message actually
    # reaches the final event, along with the pack it was gated against —
    # not just the persisted database message.
    state = {
        "user_id": alice,
        "thread_id": thread_id,
        "question": QUESTION,
        "category": ScopeCategory.IN_SCOPE,
        "scope_decision": SimpleNamespace(route=Route.TEXT_ONLY),
        "computation": None,
        "events": [],
        "answer_text": INSUFFICIENT_EVIDENCE_MESSAGE,
        "pack": PACK,
    }
    result = finalize(state, deps(conn=schema), writer=Recorder())
    assert result["final"].text == INSUFFICIENT_EVIDENCE_MESSAGE
    assert result["final"].searched == ("22(1)", "24")
    messages = list_messages(schema, alice, thread_id)
    assert messages[-1].content == INSUFFICIENT_EVIDENCE_MESSAGE


def test_finalize_never_reports_searched_provisions_for_a_fixed_refusal(
    schema, alice, thread_id
):
    # respond_fixed never retrieves, so there is no pack to report.
    state = {
        "user_id": alice,
        "thread_id": thread_id,
        "question": "how do I hide income?",
        "category": ScopeCategory.PROHIBITED,
        "computation": None,
        "events": [],
        "answer_text": FIXED_RESPONSES[ScopeCategory.PROHIBITED],
    }
    result = finalize(state, deps(conn=schema), writer=Recorder())
    assert result["final"].text == FIXED_RESPONSES[ScopeCategory.PROHIBITED]
    assert result["final"].searched == ()


# --- end to end through the compiled graph ------------------------------------


def _end_to_end_deps(schema: object) -> GraphDeps:
    return deps(
        conn=schema,
        classifier=SimpleNamespace(
            classify=lambda q, **_: SimpleNamespace(
                category=ScopeCategory.IN_SCOPE, response=None, search_query=q
            )
        ),
        extractor=SimpleNamespace(
            extract=lambda turn: ExtractionResult(
                facts=facts_from(),
                rejections=(),
                repairable=(),
                repaired=False,
                completions=(),
            )
        ),
        retriever=SimpleNamespace(
            search=lambda query, k: [
                ScoredChunk(chunk=CHUNKS["22(1)"], score=2.0),
                ScoredChunk(chunk=CHUNKS["24"], score=1.0),
            ]
        ),
        generator=AnswerGenerator(FakeLLM(answer(GOOD)), CHUNKS),
    )


# The scripted extractor reports a full fact set, so the turn must be one that
# could state facts — R22 Part A skips extraction for a turn that cannot.
FACT_TURN = f"My salary is 12 lakh. {QUESTION}"


def test_the_graph_answers_an_in_scope_question_end_to_end(schema, alice, thread_id):
    graph = build_graph(_end_to_end_deps(schema))
    result = graph.invoke(
        {"user_id": alice, "thread_id": thread_id, "question": FACT_TURN}
    )
    assert result["category"] is ScopeCategory.IN_SCOPE
    assert [type(e) for e in result["events"]] == [ClaimEvent]
    assert result["final"].route == "compute"
    assert result["final"].citations == ("22(1)",)
    messages = list_messages(schema, alice, thread_id)
    assert [m.role for m in messages] == ["user", "assistant"]
    assert (
        load_fact_state(schema, alice, thread_id).get(FactField.SALARY_INCOME)
        is not None
    )


def test_the_graph_short_circuits_a_prohibited_question_with_no_llm_or_retrieval(
    schema, alice, thread_id
):
    prohibited = deps(
        conn=schema,
        classifier=SimpleNamespace(
            classify=lambda q, **_: SimpleNamespace(
                category=ScopeCategory.PROHIBITED,
                response=FIXED_RESPONSES[ScopeCategory.PROHIBITED],
            )
        ),
    )
    graph = build_graph(prohibited)
    result = graph.invoke(
        {
            "user_id": alice,
            "thread_id": thread_id,
            "question": "How do I hide freelance income?",
        }
    )
    assert result["final"].route == "prohibited"
    assert result["events"] == []
    messages = list_messages(schema, alice, thread_id)
    assert messages[-1].content == FIXED_RESPONSES[ScopeCategory.PROHIBITED]
    # extract_facts/retrieve/route_calc/generate_verify never ran: the fact
    # state stays exactly what load_thread produced (a fresh, empty thread).
    assert load_fact_state(schema, alice, thread_id) == ThreadFactState()


def test_streaming_the_graph_emits_stage_then_claim_then_final(
    schema, alice, thread_id
):
    graph = build_graph(_end_to_end_deps(schema))
    emitted = [
        chunk
        for _mode, chunk in graph.stream(
            {"user_id": alice, "thread_id": thread_id, "question": QUESTION},
            stream_mode=["custom"],
        )
    ]
    stages = [e["stage"] for e in emitted if "stage" in e]
    # R19 Phase C: `extract_facts`/`merge_facts` (-> "facts") and `retrieve`
    # (-> "evidence") run in parallel branches off `classify`, so their
    # relative order is no longer guaranteed — only that "thinking" leads.
    assert stages[0] == "thinking"
    assert set(stages[1:-1]) == {"facts", "evidence"}
    assert stages[-1] == "writing"
    assert any(e.get("type") == "content" for e in emitted)
    assert emitted[-1]["disclaimer"]


# --- calculation questions: pinned provisions and the income question ---------


def pinning_retrieve(intent: Intent) -> list[str]:
    class FakeRetriever:
        def search(self, query: str, k: int) -> list[ScoredChunk]:
            return [ScoredChunk(chunk=CHUNKS["22"], score=1.0)]

    pins = (
        ScoredChunk(chunk=CHUNKS["24"], score=1.0),
        ScoredChunk(chunk=CHUNKS["23"], score=1.0),
    )
    result = retrieve(
        {"query": "calculate my tax", "sub_queries": (), "intent": intent},
        deps(retriever=FakeRetriever(), calc_pins=pins),
        writer=Recorder(),
    )
    return [unit.citation for unit in result["pack"].units]


def test_a_calculation_question_packs_the_calculator_provisions_first():
    assert pinning_retrieve(Intent.CALCULATION) == ["24", "23", "22"]


@pytest.mark.parametrize(
    "intent", [Intent.EXPLANATION, Intent.ELIGIBILITY, Intent.PROCEDURE]
)
def test_no_other_intent_is_pinned(intent):
    assert pinning_retrieve(intent) == ["22"]


def test_a_pinned_provision_the_ranking_also_found_appears_once():
    class FakeRetriever:
        def search(self, query: str, k: int) -> list[ScoredChunk]:
            return [ScoredChunk(chunk=CHUNKS["24"], score=1.0)]

    result = retrieve(
        {"query": "q", "sub_queries": (), "intent": Intent.CALCULATION},
        deps(
            retriever=FakeRetriever(),
            calc_pins=(ScoredChunk(chunk=CHUNKS["24"], score=1.0),),
        ),
        writer=Recorder(),
    )
    assert [unit.citation for unit in result["pack"].units] == ["24"]


def test_a_calculation_with_no_amount_asks_for_income_with_the_fixed_question():
    recorder = Recorder()
    result = route_calc(
        {"fact_state": ThreadFactState(), "intent": Intent.CALCULATION},
        deps(),
        writer=recorder,
    )
    assert result["computation"] is None
    assert result["clarify_questions"] == (CALCULATION_CLARIFY,)
    assert recorder.events == [
        ClarifyEvent(questions=(CALCULATION_CLARIFY,)).model_dump()
    ]


@pytest.mark.parametrize(
    "intent", [Intent.EXPLANATION, Intent.ELIGIBILITY, Intent.COMPARISON]
)
def test_other_intents_keep_the_no_first_turn_question_rule(intent):
    recorder = Recorder()
    result = route_calc(
        {"fact_state": ThreadFactState(), "intent": intent}, deps(), writer=recorder
    )
    assert result["clarify_questions"] == ()
    assert recorder.events == []


def test_a_calculation_with_every_input_known_still_computes():
    result = route_calc(
        {"fact_state": thread_state(), "intent": Intent.CALCULATION},
        deps(),
        writer=Recorder(),
    )
    assert result["computation"] is not None
    assert result["clarify_questions"] == ()


def test_generate_verify_tells_the_generator_only_when_income_was_asked_for():
    seen: list[bool] = []

    class SpyGenerator:
        def generate(self, question, pack, **kwargs):
            seen.append(kwargs["calculation_pending"])
            return []

    base = {
        "query": "q",
        "pack": PACK,
        "fact_state": ThreadFactState(),
        "computation": None,
    }
    generate_verify(
        {**base, "clarify_questions": (CALCULATION_CLARIFY,)},
        deps(generator=SpyGenerator()),
        writer=Recorder(),
    )
    generate_verify(
        {
            **base,
            "clarify_questions": ("What is your salary income for the tax year?",),
        },
        deps(generator=SpyGenerator()),
        writer=Recorder(),
    )
    assert seen == [True, False]


def test_decide_adds_no_reasoning_questions_while_the_income_question_is_pending():
    recorder = Recorder()
    missing = SimpleNamespace(question="What is your total income?", material=True)
    result = decide(
        {"clarify_questions": (CALCULATION_CLARIFY,), "missing_facts": (missing,)},
        deps(),
        writer=recorder,
    )
    assert result["clarify_questions"] == (CALCULATION_CLARIFY,)
    assert recorder.events == []


# --- R23: classify resolves follow-ups (the contextualize call is gone) -------


class _RecordingClassifier:
    def __init__(self, tax_request: str = "") -> None:
        self.tax_request = tax_request
        self.calls: list[dict] = []

    def classify(self, question, **context):
        self.calls.append({"question": question, **context})
        return SimpleNamespace(
            category=ScopeCategory.IN_SCOPE,
            response=None,
            search_query="interest on borrowed capital",
            tax_request=self.tax_request,
        )


def test_a_first_turn_is_classified_and_answered_as_written():
    classifier = _RecordingClassifier(tax_request="restated")
    result = classify(
        {"question": "Can I claim home loan interest?"}, deps(classifier=classifier)
    )
    assert classifier.calls == [
        {
            "question": "Can I claim home loan interest?",
            "prior_turns": (),
            "previous_answer": "",
        }
    ]
    assert result["query"] == "Can I claim home loan interest?"


def test_a_follow_up_sends_history_and_answers_the_resolved_question():
    classifier = _RecordingClassifier(
        tax_request="explain the home loan interest deduction in simple terms"
    )
    state = {
        "question": "explain that more simply",
        "prior_turns": ["Can I claim home loan interest?"],
        "previous_answer": "You can deduct the interest.",
    }
    result = classify(state, deps(classifier=classifier))
    assert classifier.calls[0]["prior_turns"] == ["Can I claim home loan interest?"]
    assert classifier.calls[0]["previous_answer"] == "You can deduct the interest."
    # The raw message is what gets classified...
    assert classifier.calls[0]["question"] == "explain that more simply"
    # ...and the resolved restatement is what gets answered.
    assert result["query"] == "explain the home loan interest deduction in simple terms"
    assert result["search_query"] == "interest on borrowed capital"


def test_a_standalone_question_after_earlier_turns_gets_no_history():
    """The deterministic check (ADR-113): no follow-up marker and a question
    mark, so the classifier sees the message alone."""
    classifier = _RecordingClassifier(tax_request="ignored")
    result = classify(
        {
            "question": "What is the standard deduction for salaried people?",
            "prior_turns": ["Can I claim home loan interest?"],
        },
        deps(classifier=classifier),
    )
    assert classifier.calls[0]["prior_turns"] == ()
    assert result["query"] == "What is the standard deduction for salaried people?"


def test_a_follow_up_with_no_restatement_falls_back_to_the_raw_message():
    classifier = _RecordingClassifier(tax_request="")
    result = classify(
        {"question": "thanks, that helps", "prior_turns": ["Can I claim HRA?"]},
        deps(classifier=classifier),
    )
    assert result["query"] == "thanks, that helps"


def test_a_statement_answering_the_last_questions_reopens_the_last_request():
    """R24's a09: "I'm resident and my salary is my only income" left as it
    was searches for nothing; it is the last request with these facts."""
    classifier = _RecordingClassifier(tax_request="")
    state = {
        "question": "I am a resident of India and my salary is my only income.",
        "prior_turns": ["My income is 10 lakh. What income tax do I have to pay?"],
    }
    result = classify(state, deps(classifier=classifier))
    assert result["query"] == (
        "My income is 10 lakh. What income tax do I have to pay?\n"
        "I am a resident of India and my salary is my only income."
    )
    # The joined request is classified again, on its own, before it is used.
    assert classifier.calls[1] == {
        "question": result["query"],
        "prior_turns": (),
        "previous_answer": "",
    }
    assert result["search_query"] == "interest on borrowed capital"
    assert result["reopened"] is True


def test_a_refused_request_cannot_ride_back_in_on_a_fact_statement():
    class RefusesTheJoinedRequest(_RecordingClassifier):
        def classify(self, question, **context):
            self.calls.append({"question": question, **context})
            refused = "hide" in question
            return SimpleNamespace(
                category=ScopeCategory.PROHIBITED
                if refused
                else ScopeCategory.IN_SCOPE,
                response="I can't help with that." if refused else None,
                search_query=question,
                tax_request="",
            )

    state = {
        "question": "My salary is 10 lakh.",
        "prior_turns": ["How do I hide my cash income from the tax department?"],
    }
    result = classify(state, deps(classifier=RefusesTheJoinedRequest()))
    assert result["category"] is ScopeCategory.PROHIBITED
    assert result["fixed_response"] == "I can't help with that."


def test_a_thank_you_is_never_turned_into_the_last_request():
    classifier = _RecordingClassifier(tax_request="")
    result = classify(
        {"question": "great, thank you", "prior_turns": ["Can I claim HRA?"]},
        deps(classifier=classifier),
    )
    assert result["query"] == "great, thank you"


def test_decide_drops_a_reasoning_question_the_calculator_already_asked():
    residency = CLARIFY_TEMPLATES[FactField.RESIDENTIAL_STATUS]
    state = {
        "clarify_questions": (residency,),
        "missing_facts": (
            MissingFact(
                condition_id="c1",
                question="Are you a resident individual in India this year?",
                material=True,
            ),
            MissingFact(
                condition_id="c2",
                question="Do you meet the definition of a resident individual?",
                material=True,
            ),
            MissingFact(
                condition_id="c3", question="Do you own the house?", material=True
            ),
        ),
    }
    result = decide(state, deps(), writer=Recorder())
    assert result["clarify_questions"] == (residency, "Do you own the house?")


def test_decide_keeps_only_the_first_of_two_reasoning_questions_on_one_fact():
    state = {
        "clarify_questions": (),
        "missing_facts": (
            MissingFact(
                condition_id="c1", question="Are you a resident?", material=True
            ),
            MissingFact(
                condition_id="c2",
                question="What is your residency status?",
                material=True,
            ),
        ),
    }
    result = decide(state, deps(), writer=Recorder())
    assert result["clarify_questions"] == ("Are you a resident?",)


def test_decide_never_asks_for_a_figure_the_calculator_computed():
    state = {
        "clarify_questions": (),
        "computation": object(),
        "missing_facts": (
            MissingFact(
                condition_id="c1",
                question="What is the income-tax payable on your income before the rebate?",
                material=True,
            ),
            MissingFact(
                condition_id="c2", question="Do you own the house?", material=True
            ),
        ),
    }
    result = decide(state, deps(), writer=Recorder())
    assert result["clarify_questions"] == ("Do you own the house?",)


def test_a_fact_statement_with_a_new_figure_is_answered_afresh_not_as_a_follow_up():
    from taxverity.graph.nodes import _answers_the_last_questions

    statement = "I am a resident of India and my salary is my only income."
    assert _answers_the_last_questions({"question": statement, "computation": object()})
    assert _answers_the_last_questions({"question": statement, "reopened": True})
    # A real follow-up question keeps the follow-up layout.
    assert not _answers_the_last_questions(
        {"question": "How can I reduce it?", "computation": object()}
    )
    assert not _answers_the_last_questions({"question": statement})
