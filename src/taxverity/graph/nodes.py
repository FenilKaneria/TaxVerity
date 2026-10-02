"""Step 13.2 — thin node wrappers over Phases 7-12, plus Step 13.5's minimal
corrective retry. No logic moves into the graph: every node is a few lines
composing an already-built, already-tested function or class. `route_calc`'s
branch on `Route` and `finalize`'s framing of a served answer are the only
real decisions made here, and both are mechanical reflections of what
`calculator.scope` / `safety.evidence_gate` already decided, not new policy.
`retrieve_retry` is the one exception with a genuine decision of its own — a
fixed wider pool and `expand=True` — but the *whether to retry* decision
lives in `build.py`'s conditional edge, not here.

Every node takes `(state, deps, writer=None)`. `writer` defaults to
`get_stream_writer()`, which only resolves inside a running graph (rule 04's
`stream_mode="custom"`); tests pass a fake writer explicitly instead, so a
node is directly callable with no graph context required.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from langgraph.config import get_stream_writer

from taxverity.calculator.materiality import Outcome, probe
from taxverity.calculator.scope import Computation, Route, compute, route
from taxverity.calculator.scope import run as run_calculator
from taxverity.facts import FIELDS, FactField, FactStatus, UserFacts, ValueKind
from taxverity.generation.claims import ClaimEvent, WithheldEvent, strip_all_markers
from taxverity.generation.generate import render_computation
from taxverity.generation.verifier import numbers_in
from taxverity.graph.state import (
    CALCULATION_CLARIFY,
    CLARIFY_TEMPLATES,
    RECENT_TURNS_WINDOW,
    ClarifyEvent,
    FinalEvent,
    GraphDeps,
    GraphState,
    StageEvent,
    TraceEntry,
)
from taxverity.llm.extract import no_facts
from taxverity.memory.contextualize import (
    is_style_followup,
    needs_contextualization,
)
from taxverity.memory.fact_state import (
    ThreadFactState,
    load_fact_state,
    merge_turn,
    save_fact_state,
)
from taxverity.observability import get_logger
from taxverity.reasoning.validate import ValidatedAnalysis, validate
from taxverity.retrieval.base import ScoredChunk
from taxverity.safety.classifier import Intent
from taxverity.safety.evidence_gate import guidance_lines, release
from taxverity.threads.store import append_message, list_messages

logger = get_logger(__name__)

Writer = Callable[[Any], None]

# Step 13.5's corrective retry: a fixed wider pool, not `deps.pool_k * n` —
# registered before any run, per PLAN 13.5.
RETRY_POOL = 40

# R21 Part B: sub-query searches run concurrently, two at a time. Each search
# makes its Jina calls (embed, then rerank) one after another, so two workers
# never exceed Jina's free-tier concurrency limit of 2 (ADR-123).
SUBQUERY_WORKERS = 2

# R19 Phase C: fields the materiality probe can sweep or ask about. A thread
# with none of these stated or inferred yet has said nothing quantitative at
# all — asking 6-7 clarify questions on that first turn is an interrogation,
# not a conversation. `_has_amount_fact` gates the probe on there being at
# least one such fact already in the thread before it runs.
_AMOUNT_FIELDS = tuple(f for f in FactField if FIELDS[f].kind is ValueKind.MONEY)


# R22 Part A: a possessive, or a first-person statement of the kind that
# carries a fact ("I earn", "I'm 65", "we sold"). Deliberately not a bare "I"
# or "me": "I didn't understand, give me an example" states nothing.
_FACT_CUE = re.compile(
    r"\b(?:my|mine|our|ours|i'?m|i am|i was|i have|i've|i had|we are|we're|we "
    r"have|we've|(?:i|we) (?:earn|earned|pay|paid|live|lived|own|owned|sold|bought|"
    r"work|worked|get|got|receive|received|invest|invested|rent|rented|retired|"
    r"turned|spend|spent|hold|held|inherited|gifted))\b",
    re.IGNORECASE,
)


# "section 22(1)", "Schedule XV(1)", "the Act, 2025": numbers that name law,
# not the person. Masked before the figure check; anything this misses (a bare
# "80C") only makes the gate call extraction when it need not.
_PROVISION_REF = re.compile(
    r"\b(?:sections?|sec\.|s\.|u/s|sub-sections?|clauses?|schedules?|chapters?)"
    r"\s+[\w()]+|\bAct,?\s+\d{4}\b",
    re.IGNORECASE,
)


def may_state_facts(turn: str) -> bool:
    """Whether a turn could carry a fact worth an extraction call: a figure
    (digits or number words) or a fact-bearing first-person cue. Fail-safe by
    construction — skipping can only miss a fact (which surfaces later as a
    clarify question), never invent one."""
    figures = numbers_in(_PROVISION_REF.sub(" ", turn))
    return bool(figures) or _FACT_CUE.search(turn) is not None


def _has_amount_fact(facts: UserFacts) -> bool:
    return any(
        (known := facts.get(field_)) is not None
        and known.status in (FactStatus.STATED, FactStatus.INFERRED)
        for field_ in _AMOUNT_FIELDS
    )


def load_thread(
    state: GraphState, deps: GraphDeps, writer: Writer | None = None
) -> dict:
    emit = writer or get_stream_writer()
    emit(StageEvent(stage="thinking").model_dump())
    messages = list_messages(deps.conn, state["user_id"], state["thread_id"])
    user_turns = [message.content for message in messages if message.role == "user"]
    fact_state = load_fact_state(deps.conn, state["user_id"], state["thread_id"])
    previous_answer = _previous_answer(messages)
    return {
        "prior_turns": user_turns[-RECENT_TURNS_WINDOW:],
        "previous_answer": previous_answer,
        "previous_citations": _previous_citations(messages),
        "style_followup": is_style_followup(state.get("question", ""), previous_answer),
        "turn": len(user_turns) + 1,
        "fact_state": fact_state,
    }


def _previous_answer(messages: Sequence[Any]) -> str:
    for message in reversed(messages):
        if message.role != "assistant":
            continue
        lines = (strip_all_markers(line) for line in message.content.splitlines())
        return "\n".join(line for line in lines if line)
    return ""


def _previous_citations(messages: Sequence[Any]) -> tuple[str, ...]:
    for message in reversed(messages):
        if message.role != "assistant":
            continue
        records = (message.payload or {}).get("citations") or ()
        paths = (record.get("path") for record in records if isinstance(record, dict))
        return tuple(dict.fromkeys(path for path in paths if path))
    return ()


def classify(state: GraphState, deps: GraphDeps, writer: Writer | None = None) -> dict:
    """R23: also resolves a follow-up, replacing the separate contextualize
    call. The deterministic check (ADR-113) decides whether the classifier
    sees the recent turns; a standalone question is classified and answered
    exactly as written."""
    question = state["question"]
    prior_turns = state.get("prior_turns") or []
    follow_up = needs_contextualization(question, prior_turns)
    result = deps.classifier.classify(
        question,
        prior_turns=prior_turns if follow_up else (),
        previous_answer=state.get("previous_answer", "") if follow_up else "",
    )
    # A follow-up is answered as the classifier's resolved restatement of it,
    # a standalone question as the person wrote it.
    query = (getattr(result, "tax_request", "") or question) if follow_up else question
    # getattr, not result.search_query: a test double's classifier stub may
    # predate R19 Phase B (ADR-120) and not set it.
    search_query = getattr(result, "search_query", None) or query
    # R20 Step 20.2: same getattr guard for a stub predating sub_queries.
    sub_queries = tuple(getattr(result, "sub_queries", None) or ())
    # R20 Step 20.3: same getattr guard for a stub predating intent.
    intent = getattr(result, "intent", None) or Intent.EXPLANATION
    return {
        "query": query,
        "category": result.category,
        "fixed_response": result.response,
        "search_query": search_query,
        "sub_queries": sub_queries,
        "intent": intent,
    }


def respond_fixed(
    state: GraphState, deps: GraphDeps, writer: Writer | None = None
) -> dict:
    """Reached only for `adjacent | out_of_scope | prohibited` (rule 03). No
    LLM call, no retrieval — the classifier already picked the fixed template."""
    return {
        "answer_text": state["fixed_response"],
        "events": [],
        "computation": None,
        "clarify_questions": (),
    }


def respond_conversational(
    state: GraphState, deps: GraphDeps, writer: Writer | None = None
) -> dict:
    """Reached only for `conversational` (advisor pivot, Step 5). No
    retrieval, no facts, no claims — `deps.conversational` is a guarded LLM
    call that cannot say anything about the Act (see llm/conversational.py);
    its reply rides to the browser as `FinalEvent.text`, the same field
    `respond_fixed`'s template uses, so no new event type is needed."""
    return {
        "answer_text": deps.conversational.reply(state["query"]),
        "events": [],
        "computation": None,
        "clarify_questions": (),
    }


def extract_facts(
    state: GraphState, deps: GraphDeps, writer: Writer | None = None
) -> dict:
    # The raw turn, never the contextualized query (rule 04): a follow-up
    # rewrite resolves references for retrieval, it is never fact truth.
    question = state["question"]
    # R22 Part A: rule 01's deterministic pre-check, the same shape as the
    # contextualizer's skip. A reasoning intent always extracts, since its
    # answer applies the person's facts; any other intent extracts only when
    # the turn itself could state one.
    intent = state.get("intent", Intent.EXPLANATION)
    if intent not in _REASONING_INTENTS and not may_state_facts(question):
        return {"extraction": no_facts()}
    return {"extraction": deps.extractor.extract(question)}


def merge_facts(
    state: GraphState, deps: GraphDeps, writer: Writer | None = None
) -> dict:
    emit = writer or get_stream_writer()
    extraction = state["extraction"]
    # getattr, not extraction.situation_facts: a test double's ExtractionResult
    # stub may predate R20 Step 20.4, same guard style as sub_queries/intent.
    situation_facts = getattr(extraction, "situation_facts", None) or ()
    merged = merge_turn(
        state["fact_state"],
        extraction.facts,
        turn=state["turn"],
        situation_facts=situation_facts,
    )
    save_fact_state(deps.conn, state["user_id"], state["thread_id"], merged)
    emit(StageEvent(stage="facts", facts=_facts_payload(merged)).model_dump())
    return {"fact_state": merged}


def retrieve(state: GraphState, deps: GraphDeps, writer: Writer | None = None) -> dict:
    return _retrieve(state, deps, writer, k=deps.pool_k, expand=False)


def retrieve_retry(
    state: GraphState, deps: GraphDeps, writer: Writer | None = None
) -> dict:
    """Step 13.5's one corrective retry (ADR-033 as amended by ADR-110, PLAN
    13.5): fires only when `generate_verify`'s first pass served zero statute
    claims, over a wider pool with `pack(expand=True)`. Bounded to one retry by
    `build.py`'s `_retry_branch` checking `state["retried"]`, not by anything
    here — this node has no memory of whether it already ran."""
    emit = writer or get_stream_writer()
    emit(StageEvent(stage="refining search").model_dump())
    result = _retrieve(state, deps, writer, k=RETRY_POOL, expand=True)
    result["retried"] = True
    return result


def _merge_subquery_results(
    per_query: Sequence[Sequence[ScoredChunk]], limit: int
) -> list[ScoredChunk]:
    """Round-robin interleave each sub-query's own ranking, deduplicated by
    chunk id (first occurrence wins), so no single sub-query's pool
    dominates the merged order the packer then walks (rule 01: the packer
    itself is untouched — this only changes what ranking it is handed)."""
    seen: set[str] = set()
    merged: list[ScoredChunk] = []
    depth = max((len(results) for results in per_query), default=0)
    for i in range(depth):
        for results in per_query:
            if len(merged) >= limit:
                return merged
            if i >= len(results):
                continue
            result = results[i]
            if result.chunk.chunk_id in seen:
                continue
            seen.add(result.chunk.chunk_id)
            merged.append(result)
    return merged


def pinned_first(
    pins: Sequence[ScoredChunk], results: Sequence[ScoredChunk]
) -> list[ScoredChunk]:
    """The calculator's own provisions ahead of the ranking for a calculation
    question: a first "can you calculate my tax?" names no section, so the
    citation shortcut never fires, and the ranking alone surfaced ss.190/405
    rather than the slab rates on a live turn."""
    pinned = {pin.chunk.chunk_id for pin in pins}
    return [
        *pins,
        *(result for result in results if result.chunk.chunk_id not in pinned),
    ]


def _style_pins(state: GraphState, deps: GraphDeps) -> list[ScoredChunk]:
    """R22 Part B: the passages the last answer cited, for a style follow-up
    ("explain simply"). The rewrite then works from the same law that answer
    stood on, instead of a fresh search re-reading the statute."""
    if not state.get("style_followup") or deps.citation_lookup is None:
        return []
    found = (deps.citation_lookup(path) for path in state.get("previous_citations", ()))
    # Two cited paths can resolve to the same chunk (an ancestor fallback).
    unique = {hit.chunk.chunk_id: hit for hit in found if hit is not None}
    return list(unique.values())


def _retrieve(
    state: GraphState, deps: GraphDeps, writer: Writer | None, *, k: int, expand: bool
) -> dict:
    emit = writer or get_stream_writer()
    # The first pass only: a corrective retry searches afresh, which is also
    # what rescues a message misread as a style follow-up.
    style_pins = [] if expand else _style_pins(state, deps)
    if style_pins:
        pack = deps.packer.pack(pinned_first(style_pins, []), expand=False)
        emit(
            StageEvent(
                stage="evidence", chunks=tuple(unit.citation for unit in pack.units)
            ).model_dump()
        )
        return {"pack": pack, "trace": []}
    # R19 Phase B (ADR-120): retrieval runs on the classifier's Act-vocabulary
    # rewrite when one exists, falling back to the raw query for callers
    # (tests, an older classifier stub) that don't set it.
    base_query = state.get("search_query") or state["query"]
    sub_queries = state.get("sub_queries") or ()
    sub_trace: list[dict[str, str | float]] = []
    if sub_queries:
        # R20 Step 20.2: one retrieval pass per legal sub-question for a
        # multi-issue question, reranked independently (each pass already
        # goes through `deps.retriever`'s own reranker) and merged before
        # packing — a plain question still takes the single-search path
        # below (rule 01: no extra pass where one suffices).
        #
        # R21 Part B: run concurrently; `map` returns results in sub-query
        # order, so the round-robin merge is exactly as deterministic as the
        # sequential loop it replaced.
        def timed_search(sub_query: str) -> tuple[Sequence[ScoredChunk], float]:
            started = time.perf_counter()
            found = deps.retriever.search(sub_query, k)
            return found, (time.perf_counter() - started) * 1000

        with ThreadPoolExecutor(max_workers=SUBQUERY_WORKERS) as pool:
            timed = list(pool.map(timed_search, sub_queries))
        per_query = [found for found, _ in timed]
        sub_trace = [
            {"node": f"retrieve.subquery.{i}", "ms": ms}
            for i, (_, ms) in enumerate(timed, start=1)
        ]
        results = _merge_subquery_results(per_query, k)
    else:
        results = list(deps.retriever.search(base_query, k))
    if state.get("intent") is Intent.CALCULATION:
        results = pinned_first(deps.calc_pins, results)
    pack = deps.packer.pack(results, expand=expand)
    emit(
        StageEvent(
            stage="evidence", chunks=tuple(unit.citation for unit in pack.units)
        ).model_dump()
    )
    return {"pack": pack, "trace": sub_trace}


def route_calc(
    state: GraphState, deps: GraphDeps, writer: Writer | None = None
) -> dict:
    """`scope.route`, then `materiality.probe` or `compute` (PLAN 13.2).

    `TEXT_ONLY` and `INCOMPLETE` are not graph branches — only what this
    node hands `generate_verify` varies. An `INCOMPLETE` decision with
    nothing left to ask (every unknown field is `ASSUME`/`NOT_COMPUTED`) is
    computed here exactly like a `COMPUTE` decision; one still waiting on an
    `ASK`/`DEFERRED` field emits a `clarify` event and answers text-only
    (PLAN 13.3).

    R19 Phase C: an `INCOMPLETE` decision on a thread that has stated or
    inferred no amount fact at all skips the probe entirely rather than
    interrogating a first turn with 6-7 questions — it answers text-only,
    same as an `INCOMPLETE` decision the probe itself resolves with nothing
    to ask."""
    emit = writer or get_stream_writer()
    facts = state["fact_state"].as_user_facts()
    decision = route(facts)
    computation: Computation | None = None
    clarify_questions: tuple[str, ...] = ()
    if decision.route is Route.COMPUTE:
        computation = compute(decision)
    elif decision.route is Route.INCOMPLETE and _has_amount_fact(facts):
        found = probe(facts, decision)
        if found.inputs is not None:
            computation = run_calculator(found.inputs)
        else:
            clarify_questions = tuple(
                CLARIFY_TEMPLATES[finding.field]
                for finding in found.by_outcome(Outcome.ASK)
            )
    elif (
        decision.route is Route.INCOMPLETE and state.get("intent") is Intent.CALCULATION
    ):
        # The one exception to R19 Phase C's "no first-turn questions": the
        # person asked for a calculation, so their income is the question.
        clarify_questions = (CALCULATION_CLARIFY,)
    if clarify_questions:
        emit(ClarifyEvent(questions=clarify_questions).model_dump())
    return {
        "scope_decision": decision,
        "computation": computation,
        "clarify_questions": clarify_questions,
    }


# R20 Step 20.5: `reason` runs only for these intents (standing decision 3,
# `i-want-you-to-misty-pudding.md`) — a plain "what does section X say"
# explanation question, and a procedure or single-deduction-limit question,
# already answer directly from the pack with no condition-checking to do,
# so a second LLM call for them is exactly the "no LLM call for a job code
# already does" rule 01 forbids.
_REASONING_INTENTS = frozenset(
    {
        Intent.ELIGIBILITY,
        Intent.CALCULATION,
        Intent.COMPARISON,
        Intent.APPLICABILITY,
        Intent.MULTI_ISSUE,
    }
)

_NO_ANALYSIS: dict[str, Any] = {
    "legal_rules": (),
    "applicability": (),
    "missing_facts": (),
    "answer_plan": None,
}


def reason(state: GraphState, deps: GraphDeps, writer: Writer | None = None) -> dict:
    """Not yet wired into the compiled graph (that is Step 20.8's "graph
    rewire") — directly callable and directly testable, the same way
    `retrieve_retry` existed for two steps before its edge was added.

    Skipped for an intent that needs no condition-checking, and skipped or
    falling back whenever the model's own output does not survive
    `reasoning/validate.py` — a reasoning failure must never stop a turn
    from answering; `decide`/`generate` (20.6-20.7) treat `answer_plan is
    None` as "reason over the pack directly", the existing path unchanged.
    """
    intent = state.get("intent", Intent.EXPLANATION)
    if intent not in _REASONING_INTENTS:
        return dict(_NO_ANALYSIS)

    emit = writer or get_stream_writer()
    emit(StageEvent(stage="analysing").model_dump())
    result = deps.reasoner.reason(
        state["query"], state["pack"], state["fact_state"], state.get("computation")
    )
    if result.analysis is None:
        return dict(_NO_ANALYSIS)

    validated = validate(result.analysis, state["pack"], fact_state=state["fact_state"])
    if not validated.has_governing_rule:
        logger.warning("no legal rule survived reasoning validation for this turn")
        return dict(_NO_ANALYSIS)

    return {
        "legal_rules": validated.legal_rules,
        "applicability": validated.applicability,
        "missing_facts": validated.missing_facts,
        "answer_plan": validated.answer_plan,
    }


def decide(state: GraphState, deps: GraphDeps, writer: Writer | None = None) -> dict:
    """R20 Step 20.6: deterministic clarify | answer gate, downstream of
    `reason` (20.5). Not yet wired into the compiled graph (Step 20.8's
    "graph rewire") — directly callable and directly testable, the same way
    `reason` existed for one step before its edge was added.

    Calculator-sourced clarify questions (`route_calc`, Step 13.2, fixed
    `CLARIFY_TEMPLATES`) are untouched and already emitted their own event.
    This node adds reasoning-sourced questions: one per `missing_facts`
    entry the model itself marked `material` (PLAN's decision 2 — "missing
    facts that change the conclusion"). Every such question already passed
    `reasoning/validate.py`'s gate before reaching here — grounded in the
    linked condition's own citations, no invented legal number, pointed at
    a condition that genuinely survived as `unknown` — so trusting
    `material` here is exactly as safe as trusting a validated `LegalRule`;
    the untrustworthy part of the model's output was already stripped out
    upstream, not here."""
    emit = writer or get_stream_writer()
    existing = state.get("clarify_questions", ())
    if CALCULATION_CLARIFY in existing:
        # The calculator's income question comes first: reasoning questions
        # asked before any figure exists re-ask it in other words, and any
        # still material once the income is known are asked on that turn.
        return {"clarify_questions": existing}
    material_questions = tuple(
        missing.question
        for missing in state.get("missing_facts", ())
        if missing.material
    )
    new_questions = tuple(q for q in material_questions if q not in existing)
    if new_questions:
        emit(ClarifyEvent(questions=new_questions).model_dump())
    return {"clarify_questions": existing + new_questions}


def _analysis_from_state(state: GraphState) -> ValidatedAnalysis | None:
    """Step 20.8: rebuilds R20's validated reasoning output from the state
    fields `reason` (20.5) wrote, or `None` when `reason` was skipped or
    nothing survived validation — `AnswerGenerator.generate()` treats `None`
    exactly like the pre-R20 path, no fallback logic needed here."""
    legal_rules = state.get("legal_rules") or ()
    if not legal_rules:
        return None
    return ValidatedAnalysis(
        legal_rules=legal_rules,
        applicability=state.get("applicability") or (),
        missing_facts=state.get("missing_facts") or (),
        answer_plan=state["answer_plan"],
    )


def generate_verify(
    state: GraphState, deps: GraphDeps, writer: Writer | None = None
) -> dict:
    """R22 Part B: releases only what will be served. The events are held
    until the evidence gate has judged them; an answer the gate replaces (an
    empty pack, or zero grounded claims, whether or not the corrective retry
    still follows) emits none of them. A retry therefore shows only its own
    pass, never a first pass stacked above it.

    R22 Part C: once no retry is left, a gated answer that still has
    guidance lines releases those alone (`evidence_gate.release`)."""
    emit = writer or get_stream_writer()
    facts = state["fact_state"].as_user_facts()
    emit(StageEvent(stage="writing").model_dump())
    events = deps.generator.generate(
        state["query"],
        state["pack"],
        facts=facts,
        computation=state["computation"],
        analysis=_analysis_from_state(state),
        request=state.get("question"),
        previous_answer=state.get("previous_answer") or None,
        calculation_pending=CALCULATION_CLARIFY in state.get("clarify_questions", ()),
        style_request=bool(state.get("style_followup")),
        on_repair=lambda: emit(StageEvent(stage="checking").model_dump()),
    )
    answer_text, released = release(state["pack"], events)
    if answer_text is not None and not state.get("retried"):
        # Gated with the retry still to come: the retry's pass replaces this.
        released = []
    for event in released:
        emit(event.model_dump())
    return {"events": list(events), "answer_text": answer_text}


def finalize(state: GraphState, deps: GraphDeps, writer: Writer | None = None) -> dict:
    """Persists the turn — this node, not `merge_facts`, owns messages, so a
    crash mid-generation leaves no assistant message for a user turn nobody
    saw answered. Facts are persisted as soon as they are merged (rule 04:
    "persist with per-turn provenance"), independent of how the turn ends."""
    emit = writer or get_stream_writer()
    decision = state.get("scope_decision")
    route_label = (
        decision.route.value if decision is not None else state["category"].value
    )
    events = state.get("events", [])
    answer_text = state.get("answer_text")
    text = answer_text if answer_text is not None else _served_text(events)
    computation = state.get("computation")
    if answer_text is not None:
        # A gated turn keeps only its guidance lines (R22 Part C); a refusal
        # or conversational reply has no events at all.
        events = guidance_lines(events)
        if events:
            text = answer_text + "\n" + _served_text(events)
    citations = _served_citations(events)
    pack = state.get("pack")
    searched = (
        tuple(unit.citation for unit in pack.units)
        if answer_text is not None and pack is not None
        else ()
    )
    trace = tuple(TraceEntry(**entry) for entry in state.get("trace", []))
    final_event = FinalEvent(
        route=route_label,
        computation=_computation_summary(computation)
        if computation is not None
        else None,
        citations=citations,
        text=answer_text,
        searched=searched,
        trace=trace,
    )
    emit(final_event.model_dump())
    append_message(
        deps.conn, state["user_id"], state["thread_id"], "user", state["question"]
    )
    append_message(
        deps.conn,
        state["user_id"],
        state["thread_id"],
        "assistant",
        text,
        payload={
            "citations": _served_citation_records(events),
            "withheld": _withheld_records(events),
            "clarify_questions": list(state.get("clarify_questions", ())),
            "trace": [entry.model_dump() for entry in trace],
            # R19 Phase B (ADR-120): True when `text` is the generator's own
            # markdown claim lines (frontend renders it with MarkdownAnswer);
            # False when it is a fixed/gated plain-prose string (a refusal
            # template, the conversational reply, or the insufficient-
            # evidence message) — those are never run through line
            # classification, and never bulleted.
            "structured": answer_text is None or bool(events),
        },
    )
    return {"final": final_event}


def _facts_payload(state: ThreadFactState) -> dict[str, str]:
    return {fact.field.value: str(fact.value) for fact in state.facts.values()}


def _served_text(events: list[ClaimEvent | WithheldEvent]) -> str:
    # R19 Phase B (ADR-120): a newline, not a space — each claim is a whole
    # markdown line (a heading, a bullet, a no_basis sentence), and joining
    # with a space would run them onto one line and destroy that structure
    # for the frontend's markdown renderer.
    return "\n".join(event.text for event in events if isinstance(event, ClaimEvent))


def _served_citations(events: list[ClaimEvent | WithheldEvent]) -> tuple[str, ...]:
    seen: list[str] = []
    for event in events:
        if not isinstance(event, ClaimEvent):
            continue
        for citation in event.citations:
            if citation.path not in seen:
                seen.append(citation.path)
    return tuple(seen)


def _served_citation_records(
    events: list[ClaimEvent | WithheldEvent],
) -> list[dict[str, str | int]]:
    """First-seen (marker, path, quote) triples, persisted on the message so
    history can both reopen the citation dialog and resolve the `[n]`
    markers still embedded in the persisted text (R19 Phase B, ADR-120) —
    without a live turn's own `cited` state. `marker` numbers a single
    evidence pack (this turn's), so they never collide within one message
    even across a corrective retry: `events` is replaced, not accumulated,
    by whichever `generate_verify` pass actually produced what is served."""
    seen: set[int] = set()
    records: list[dict[str, str | int]] = []
    for event in events:
        if not isinstance(event, ClaimEvent):
            continue
        for citation in event.citations:
            if citation.marker in seen:
                continue
            seen.add(citation.marker)
            records.append(
                {
                    "marker": citation.marker,
                    "path": citation.path,
                    "quote": citation.quote,
                }
            )
    return records


def _withheld_records(events: list[ClaimEvent | WithheldEvent]) -> list[dict[str, str]]:
    return [
        {"id": str(event.id), "reason": event.reason}
        for event in events
        if isinstance(event, WithheldEvent)
    ]


def _computation_summary(computation: Computation) -> dict[str, str]:
    return {
        "tax_year": computation.comparison.tax_year,
        "payable": str(computation.comparison.under_202_1.payable.amount),
        "trace": render_computation(computation),
    }
