"""Step 13.1 — graph state, the SSE event vocabulary this graph assembles, and
the injected dependencies every node runs against.

Rule 04 defines the streaming contract's `stage`, `clarify` and `final` event
shapes; `claim`/`withheld` already exist in `generation/claims.py` and are
reused unchanged. `GraphDeps` is the seam node functions are tested against —
production wires a real one in `build.py`, tests hand nodes a fake with
scripted retrievers, classifiers and LLM clients, the same discipline
`test_generation.py` and `test_injection.py` already use.

`GraphDeps` lives here rather than in `nodes.py` so `build.py` can import both
from one place without `nodes.py` importing `build.py` back.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, TypedDict
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
from taxverity.memory.contextualize import QueryContextualizer
from taxverity.memory.fact_state import ThreadFactState
from taxverity.retrieval.base import Retriever
from taxverity.retrieval.evidence import EVIDENCE_POOL, EvidencePack, EvidencePacker
from taxverity.safety.classifier import IntentClassifier, ScopeCategory

GRAPH_STAGE_VERSION = 5

# rule 04: "a short recent-turns window (2-3 turns of text)".
RECENT_TURNS_WINDOW = 3


class StageEvent(BaseModel):
    model_config = ConfigDict(frozen=True)

    stage: Literal["thinking", "facts", "evidence", "refining search"]
    facts: dict[str, str] | None = None
    chunks: tuple[str, ...] | None = None


class ClarifyEvent(BaseModel):
    model_config = ConfigDict(frozen=True)

    questions: tuple[str, ...]


class TraceEntry(BaseModel):
    """R19 — one node's wall-clock cost, collected by `build.py`'s node
    wrapper and carried through to the trace panel (always visible, per user
    decision). Timings only: per-call token counts would need instrumenting
    every LLM-calling class individually, which is deferred as quality-additive,
    not correctness-critical (rule 01)."""

    model_config = ConfigDict(frozen=True)

    node: str
    ms: float


class FinalEvent(BaseModel):
    model_config = ConfigDict(frozen=True)

    route: str
    computation: dict[str, str] | None
    citations: tuple[str, ...]
    disclaimer: str = DISCLAIMER
    # R19: per-node timings for the always-visible trace panel.
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


# Deterministic clarifying-question templates (rule 04: "never from the LLM"),
# covering exactly the fields `materiality.probe` can return with outcome ASK.
# Not every FactField appears here: the rest are either not calculator inputs
# (regime, age) or the probe never asks about them (tax_year is assumed,
# deduction claims are assumed nil under 202(1)).
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


class GraphState(TypedDict, total=False):
    user_id: UUID
    thread_id: UUID
    question: str
    prior_turns: list[str]
    turn: int
    query: str
    # R19 Phase B (ADR-120): the classifier's Act-vocabulary rewrite of
    # `query`, used for retrieval only — `query` itself still goes to
    # generation, so the model answers what the person actually asked.
    search_query: str
    category: ScopeCategory
    fixed_response: str | None
    fact_state: ThreadFactState
    extraction: ExtractionResult
    scope_decision: ScopeDecision
    computation: Computation | None
    clarify_questions: tuple[str, ...]
    pack: EvidencePack
    events: list[ClaimEvent | WithheldEvent]
    answer_text: str | None
    retried: bool  # Step 13.5: set by `retrieve_retry`, bounds the corrective loop to one cycle.
    trace: list[dict[str, str | float]]  # R19: accumulated by build.py's per-node timing wrapper.
    final: FinalEvent


@dataclass
class GraphDeps:
    """Everything a node needs beyond the state itself. One instance per
    process (Phase 14 builds it once at API startup); `conn` is the one field
    a future connection-pooled deployment must swap per request, not per
    process — Phase 14's problem, not this one's."""

    conn: psycopg.Connection
    chunks: dict[str, Chunk]
    retriever: Retriever
    packer: EvidencePacker
    classifier: IntentClassifier
    contextualizer: QueryContextualizer
    extractor: FactExtractor
    generator: AnswerGenerator
    conversational: Conversationalist
    pool_k: int = field(default=EVIDENCE_POOL)
