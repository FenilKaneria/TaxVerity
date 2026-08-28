"""Step 3.2 — the ruler: recall@k, MRR and nDCG@k over the Step 3.1 gold set,
scored strictly and leniently, broken down per slice."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from enum import StrEnum

from pydantic import BaseModel, ConfigDict

from taxverity.chunking.models import Chunk
from taxverity.corpus.nodes import NodePath
from taxverity.evals.gold import GoldQuery, QuerySlice
from taxverity.observability import get_logger

logger = get_logger(__name__)

METRICS_STAGE_VERSION = 1


class CreditMode(StrEnum):
    """STRICT credits only the labelled unit itself. LENIENT also credits an
    ancestor, because ADR-055 gives a chunk its whole subtree: a retriever
    returning ``22`` for a ``22(2)`` question is imprecise, not wrong."""

    STRICT = "strict"
    LENIENT = "lenient"


class UnresolvedCitationError(LookupError):
    """A gold label naming no chunk. It would score as a retrieval miss while
    actually being a broken label — the one failure this module must not hide."""


def normalise_citation(citation: str) -> str:
    return NodePath.parse(citation).render()


def credits(retrieved: str, required: str, mode: CreditMode) -> bool:
    """A retrieved citation earns credit for a required one. Never a descendant:
    ``22(2)``'s text does not contain ``22``'s."""
    got = NodePath.parse(retrieved).components
    want = NodePath.parse(required).components
    if got == want:
        return True
    if mode is CreditMode.STRICT:
        return False
    return len(got) < len(want) and want[: len(got)] == got


class CitationIndex:
    """Citation -> chunk id, built at measurement time. ADR-059 stores citations
    in the gold set precisely so the ids may change under it."""

    def __init__(self, chunks: Iterable[Chunk]) -> None:
        self._by_path = {
            normalise_citation(chunk.node_path): chunk.chunk_id for chunk in chunks
        }

    def __len__(self) -> int:
        return len(self._by_path)

    def __contains__(self, citation: str) -> bool:
        return normalise_citation(citation) in self._by_path

    def resolve(self, citation: str) -> str:
        try:
            return self._by_path[normalise_citation(citation)]
        except KeyError:
            raise UnresolvedCitationError(f"{citation} names no chunk") from None


class Scores(BaseModel):
    model_config = ConfigDict(frozen=True)

    recall: float
    mrr: float
    ndcg: float


class QueryScore(BaseModel):
    model_config = ConfigDict(frozen=True)

    query_id: str
    slice: QuerySlice
    k: int
    retrieved: int
    strict: Scores
    lenient: Scores


class RunReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    k: int
    scored: tuple[QueryScore, ...]
    overall: dict[CreditMode, Scores]
    per_slice: dict[QuerySlice, dict[CreditMode, Scores]]
    # Negatives carry no label, so recall, MRR and nDCG are all undefined for
    # them. Measuring them needs a score threshold, which no component owns
    # until Phase 12; Step 3.6 reports their score separation instead.
    negatives: int


def _dedupe(retrieved: Sequence[str], k: int) -> list[str]:
    seen: list[str] = []
    for citation in retrieved:
        rendered = normalise_citation(citation)
        if rendered not in seen:
            seen.append(rendered)
        if len(seen) == k:
            break
    return seen


def _score(ranked: Sequence[str], required: Sequence[str], mode: CreditMode) -> Scores:
    outstanding = [normalise_citation(citation) for citation in required]
    first_hit = 0
    gains: list[int] = []
    for rank, citation in enumerate(ranked, start=1):
        # Gain is coverage, not relevance: a root chunk and its own children all
        # match the same label under LENIENT, and counting each would let one
        # retrieved item score a two-label query 1.0.
        newly = [want for want in outstanding if credits(citation, want, mode)]
        gains.append(1 if newly else 0)
        if newly:
            first_hit = first_hit or rank
            outstanding = [want for want in outstanding if want not in newly]

    found = len(required) - len(outstanding)
    dcg = sum(gain / math.log2(rank + 1) for rank, gain in enumerate(gains, start=1))
    ideal = sum(1 / math.log2(rank + 1) for rank in range(1, len(required) + 1))
    return Scores(
        recall=found / len(required),
        mrr=1 / first_hit if first_hit else 0.0,
        ndcg=dcg / ideal if ideal else 0.0,
    )


def score_query(
    query: GoldQuery, retrieved: Sequence[str], k: int, index: CitationIndex
) -> QueryScore:
    if k < 1:
        raise ValueError(f"k must be at least 1, not {k}")
    if query.slice is QuerySlice.NEGATIVE:
        raise ValueError(f"{query.query_id}: a negative query has nothing to score")
    for citation in query.required:
        index.resolve(citation)
    ranked = _dedupe(retrieved, k)
    return QueryScore(
        query_id=query.query_id,
        slice=query.slice,
        k=k,
        retrieved=len(ranked),
        strict=_score(ranked, query.required, CreditMode.STRICT),
        lenient=_score(ranked, query.required, CreditMode.LENIENT),
    )


def _mean(scores: Sequence[Scores]) -> Scores:
    if not scores:
        return Scores(recall=0.0, mrr=0.0, ndcg=0.0)
    n = len(scores)
    return Scores(
        recall=sum(s.recall for s in scores) / n,
        mrr=sum(s.mrr for s in scores) / n,
        ndcg=sum(s.ndcg for s in scores) / n,
    )


def _aggregate(scored: Sequence[QueryScore]) -> dict[CreditMode, Scores]:
    return {
        CreditMode.STRICT: _mean([s.strict for s in scored]),
        CreditMode.LENIENT: _mean([s.lenient for s in scored]),
    }


def score_run(
    queries: Sequence[GoldQuery],
    runs: Mapping[str, Sequence[str]],
    k: int,
    index: CitationIndex,
) -> RunReport:
    """Score one retriever over the whole gold set. `runs` maps query_id to the
    citations it returned, best first."""
    answerable = [q for q in queries if q.slice is not QuerySlice.NEGATIVE]
    missing = {q.query_id for q in answerable} - set(runs)
    if missing:
        raise KeyError(f"no run recorded for {sorted(missing)}")

    scored = tuple(score_query(q, runs[q.query_id], k, index) for q in answerable)
    per_slice = {
        member: _aggregate([s for s in scored if s.slice is member])
        for member in QuerySlice
        if member is not QuerySlice.NEGATIVE
    }
    report = RunReport(
        k=k,
        scored=scored,
        overall=_aggregate(scored),
        per_slice=per_slice,
        negatives=len(queries) - len(answerable),
    )
    logger.info(
        "scored %d queries at k=%d: strict recall %.3f, lenient recall %.3f",
        len(scored),
        k,
        report.overall[CreditMode.STRICT].recall,
        report.overall[CreditMode.LENIENT].recall,
    )
    return report
