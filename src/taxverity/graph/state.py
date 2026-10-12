"""Graph state, the SSE event vocabulary this graph assembles, and the injected
dependencies every node runs against."""

from __future__ import annotations

import operator
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Annotated, Literal, TypedDict
from uuid import UUID

import psycopg
from pydantic import BaseModel, ConfigDict

from taxverity.calculator.scope import Computation, ScopeDecision
from taxverity.chunking.models import Chunk
from taxverity.facts import FactField
from taxverity.generation.claims import DISCLAIMER, ClaimEvent, WithheldEvent
from taxverity.generation.generate import AnswerGenerator
from taxverity.llm.conversational import Conversationalist
from taxverity.llm.extract import ExtractionResult, FactExtractor
from taxverity.memory.fact_state import ThreadFactState
from taxverity.reasoning.models import (
    AnswerPlan,
    ConditionCheck,
    LegalRule,
    MissingFact,
)
from taxverity.reasoning.reason import Reasoner
from taxverity.retrieval.base import Retriever, ScoredChunk
from taxverity.retrieval.evidence import EVIDENCE_POOL, EvidencePack, EvidencePacker
from taxverity.safety.classifier import Intent, IntentClassifier, ScopeCategory

GRAPH_STAGE_VERSION = 12

# "a short recent-turns window (2-3 turns of text)".
RECENT_TURNS_WINDOW = 3


class StageEvent(BaseModel):
    model_config = ConfigDict(frozen=True)

    # "analysing" (the reason call), "writing" (generation) and
    # "checking" (the repair pass) name the long LLM waits a turn spends
    # after "evidence", so the screen says what is happening rather than
    # sitting on "Checking the Act…" for the whole of generation.
    stage: Literal[
        "thinking",
        "facts",
        "evidence",
        "analysing",
        "writing",
        "checking",
        "refining search",
    ]
    facts: dict[str, str] | None = None
    chunks: tuple[str, ...] | None = None


class ClarifyEvent(BaseModel):
    model_config = ConfigDict(frozen=True)

    questions: tuple[str, ...]


class TraceEntry(BaseModel):
    """One node's wall-clock cost, collected by `build.py`'s node wrapper and carried
    through to the trace panel (always visible, per user decision)."""

    model_config = ConfigDict(frozen=True)

    node: str
    ms: float


class FinalEvent(BaseModel):
    model_config = ConfigDict(frozen=True)

    route: str
    computation: dict[str, str] | None
    citations: tuple[str, ...]
    disclaimer: str = DISCLAIMER
    # Per-node timings for the always-visible trace panel.
    trace: tuple[TraceEntry, ...] = ()
    # The fixed/gated answer text — a refusal template, the conversational
    # reply, or the insufficient-evidence message. None means the turn served
    # claim events and the browser already rendered the answer from those; a
    # non-None value is never a duplicate of already-streamed claim text.
    text: str | None = None
    # Provision paths the evidence pack actually held when `text` names an
    # insufficient-evidence refusal — makes that refusal auditable rather than
    # a dead end. Empty for every other route (no pack, or the turn served).
    searched: tuple[str, ...] = ()


CLARIFY_TEMPLATES: dict[FactField, str] = {
    FactField.SALARY_INCOME: "What is your salary income for the tax year?",
    FactField.OTHER_SOURCES_INCOME: (
        "What is your income from other sources, such as interest, for the tax year?"
    ),
    FactField.HOUSE_PROPERTY_INCOME: (
        "Do you have any income or loss from house property this tax year? "
        "If so, how much?"
    ),
    FactField.BUSINESS_INCOME: (
        "Do you have any income or loss from business or profession this tax "
        "year? If so, how much?"
    ),
    FactField.CAPITAL_GAINS_SHORT_TERM: (
        "Do you have any short-term capital gains or losses this tax year? "
        "If so, how much?"
    ),
    FactField.CAPITAL_GAINS_LONG_TERM: (
        "Do you have any long-term capital gains or losses this tax year? "
        "If so, how much?"
    ),
    FactField.DEDUCTION_OTHER: (
        "Are you claiming any other deduction not already covered? If so, "
        "which one and how much?"
    ),
    FactField.RESIDENTIAL_STATUS: (
        "What is your residential status for the tax year — resident, "
        "resident but not ordinarily resident, or non-resident?"
    ),
}


# Asked when the person wants their tax calculated but has stated no amount
# yet. Fixed text, and salary first because it is the calculator's
# main input; the extractor picks up any other income named in the reply.
CALCULATION_CLARIFY = (
    "What is your salary income for the tax year, and do you have any other "
    "income, such as interest, rent, business income or capital gains?"
)

# Asked alongside a figure computed on the assumption that a salaried person
# has no other income (`materiality.ASSUMED_NIL`). One question for every
# head, not one per head: a salaried person usually answers all of them at
# once, and six chips for one "no" read as an interrogation.
OTHER_INCOME_CLARIFY = (
    "Is your salary your only income? If you also have rent, business income, "
    "interest or capital gains, how much?"
)

# Provisions pinned ahead of the ranking for a calculation question: the
# new-regime slab rates and the rebate, the two the calculator's figures come
# from. 19(1) (standard deduction) is left to retrieval: at about 1,100 of the
# 2,500-token production budget it would crowd out everything else.
CALC_PIN_CITATIONS = ("202(1)", "156")


class GraphState(TypedDict, total=False):
    user_id: UUID
    thread_id: UUID
    question: str
    prior_turns: list[str]
    # The last served answer's plain text (markers stripped), so a
    # follow-up ("explain simply", "give examples") builds on it rather than
    # re-answering blind. Context only — never evidence, never fact truth.
    previous_answer: str
    # Provision paths the last served answer cited, in the order
    # it first cited them (its persisted `payload["citations"]`).
    previous_citations: tuple[str, ...]
    # `memory.contextualize.is_style_followup` on this turn. When
    # set, `retrieve` packs `previous_citations` instead of searching, and
    # generation gets the fixed rewrite note.
    style_followup: bool
    turn: int
    query: str
    search_query: str
    sub_queries: tuple[str, ...]
    intent: Intent
    legal_rules: tuple[LegalRule, ...]
    applicability: tuple[ConditionCheck, ...]
    missing_facts: tuple[MissingFact, ...]
    answer_plan: AnswerPlan | None
    category: ScopeCategory
    fixed_response: str | None
    fact_state: ThreadFactState
    extraction: ExtractionResult
    scope_decision: ScopeDecision
    computation: Computation | None
    # Set by `classify` when a statement answering the last turn's questions
    # reopened that turn's request; generation then answers it afresh.
    reopened: bool
    # Fields the computation took as nil because the person never mentioned
    # them; the answer says so (`generate`'s fixed assumptions note).
    assumed_nil: tuple[FactField, ...]
    clarify_questions: tuple[str, ...]
    pack: EvidencePack
    events: list[ClaimEvent | WithheldEvent]
    answer_text: str | None
    retried: bool  # Set by `retrieve_retry`, bounds the corrective loop to one cycle.
    trace: Annotated[list[dict[str, str | float]], operator.add]
    final: FinalEvent


@dataclass
class GraphDeps:
    """Everything a node needs beyond the state itself."""

    conn: psycopg.Connection
    chunks: dict[str, Chunk]
    retriever: Retriever
    packer: EvidencePacker
    classifier: IntentClassifier
    extractor: FactExtractor
    generator: AnswerGenerator
    conversational: Conversationalist
    reasoner: Reasoner
    pool_k: int = field(default=EVIDENCE_POOL)
    # CALC_PIN_CITATIONS resolved to chunks once at startup (`build_deps`).
    calc_pins: tuple[ScoredChunk, ...] = ()
    # `CitationRetriever.lookup`, so a style follow-up can rebuild
    # the last answer's passages. None leaves every turn on a fresh search.
    citation_lookup: Callable[[str], ScoredChunk | None] | None = None
