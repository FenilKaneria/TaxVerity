"""Step 3.6 — the baseline measurement: one retriever, the Step 3.1 gold set,
the Step 3.2 ruler, and the per-query detail needed to say *why* a query fails.

Every later retrieval component — dense (Phase 4), fusion and reranking
(Phase 5) — is judged against the numbers this module produces.
"""

from __future__ import annotations

import time
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict

from taxverity.evals.gold import GoldQuery, QuerySlice
from taxverity.evals.metrics import (
    CitationIndex,
    CreditMode,
    RunReport,
    credits,
    normalise_citation,
    score_run,
)
from taxverity.observability import get_logger
from taxverity.retrieval.base import Retriever, as_ranked_citations

logger = get_logger(__name__)

BASELINE_STAGE_VERSION = 1

# A curve, not a single number: Phase 4 will compare a dense retriever against
# this one, and recall@1 against recall@20 says whether a change moved the top
# of the ranking or only its tail. PRIMARY_K is what the prose quotes.
K_VALUES = (1, 5, 10, 20)
PRIMARY_K = 10


class QueryOutcome(BaseModel):
    """One answerable query at PRIMARY_K, with enough detail to diagnose it."""

    model_config = ConfigDict(frozen=True)

    query_id: str
    slice: QuerySlice
    question: str
    required: tuple[str, ...]
    retrieved: tuple[str, ...]
    top_score: float | None
    strict_recall: float
    lenient_recall: float
    # Rank of the first result crediting any label under LENIENT, 1-based.
    first_hit_rank: int | None
    # Labels no retrieved citation credited, even leniently.
    missed: tuple[str, ...]


class NegativeOutcome(BaseModel):
    """A negative query carries no label, so it has no rank to score. What it
    does have is a score, and the separation between these and the answerable
    ones is the first evidence a Phase 12 threshold could be argued from."""

    model_config = ConfigDict(frozen=True)

    query_id: str
    question: str
    top_score: float | None
    top_citation: str | None


class RetrieverBaseline(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    # A composed retriever (Step 3.5's ShortcutRetriever) emits positions, not
    # a similarity: its scores rank its own results and say nothing else, so
    # the negative-separation table is meaningless for it and is suppressed.
    ordinal_scores: bool
    reports: dict[int, RunReport]
    outcomes: tuple[QueryOutcome, ...]
    negatives: tuple[NegativeOutcome, ...]
    elapsed_seconds: float

    @property
    def primary(self) -> RunReport:
        return self.reports[PRIMARY_K]

    @property
    def failures(self) -> tuple[QueryOutcome, ...]:
        """Anything short of a complete lenient answer at PRIMARY_K. Strict
        imprecision is not a failure — ADR-060 keeps the two numbers apart."""
        return tuple(o for o in self.outcomes if o.lenient_recall < 1.0)


class BaselineReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    corpus_version: str
    chunk_count: int
    gold_count: int
    k_values: tuple[int, ...]
    primary_k: int
    retrievers: tuple[RetrieverBaseline, ...]


def _first_hit_rank(retrieved: Sequence[str], required: Sequence[str]) -> int | None:
    for rank, citation in enumerate(retrieved, start=1):
        if any(credits(citation, want, CreditMode.LENIENT) for want in required):
            return rank
    return None


def _missed(retrieved: Sequence[str], required: Sequence[str]) -> tuple[str, ...]:
    return tuple(
        want
        for want in required
        if not any(credits(got, want, CreditMode.LENIENT) for got in retrieved)
    )


def measure(
    name: str,
    retriever: Retriever,
    gold: Sequence[GoldQuery],
    index: CitationIndex,
    k_values: Sequence[int] = K_VALUES,
    *,
    ordinal_scores: bool = False,
) -> RetrieverBaseline:
    """Run one retriever over the whole gold set once, at max(k_values), and
    score the same ranking at every k — a retriever's top 10 is the prefix of
    its top 20, so re-querying would measure the same thing more slowly."""
    started = time.perf_counter()
    widest = max(k_values)
    answerable = [q for q in gold if q.slice is not QuerySlice.NEGATIVE]

    ranked: dict[str, list[str]] = {}
    scores: dict[str, float | None] = {}
    for query in gold:
        results = retriever.search(query.question, widest)
        ranked[query.query_id] = as_ranked_citations(results)
        scores[query.query_id] = results[0].score if results else None

    reports = {k: score_run(gold, ranked, k, index) for k in k_values}

    outcomes = []
    for query in answerable:
        retrieved = [normalise_citation(c) for c in ranked[query.query_id][:PRIMARY_K]]
        scored = next(
            s for s in reports[PRIMARY_K].scored if s.query_id == query.query_id
        )
        outcomes.append(
            QueryOutcome(
                query_id=query.query_id,
                slice=query.slice,
                question=query.question,
                required=query.required,
                retrieved=tuple(retrieved),
                top_score=scores[query.query_id],
                strict_recall=scored.strict.recall,
                lenient_recall=scored.lenient.recall,
                first_hit_rank=_first_hit_rank(retrieved, query.required),
                missed=_missed(retrieved, query.required),
            )
        )

    negatives = tuple(
        NegativeOutcome(
            query_id=query.query_id,
            question=query.question,
            top_score=scores[query.query_id],
            top_citation=(
                ranked[query.query_id][0] if ranked[query.query_id] else None
            ),
        )
        for query in gold
        if query.slice is QuerySlice.NEGATIVE
    )

    elapsed = time.perf_counter() - started
    logger.info(
        "%s: %d queries at k=%d, strict recall %.3f, lenient recall %.3f, in %.2fs",
        name,
        len(answerable),
        PRIMARY_K,
        reports[PRIMARY_K].overall[CreditMode.STRICT].recall,
        reports[PRIMARY_K].overall[CreditMode.LENIENT].recall,
        elapsed,
    )
    return RetrieverBaseline(
        name=name,
        ordinal_scores=ordinal_scores,
        reports=reports,
        outcomes=tuple(outcomes),
        negatives=negatives,
        elapsed_seconds=elapsed,
    )
