"""Step 6.5 — does the Postgres index rank the same way the NumPy one does, and
does this corpus need an ANN index at all.

Two separate questions, and only the first is parity. The Postgres index is an
exact search like NumPy's, so any disagreement is float noise or a broken
tie-break, and the gold numbers must be identical. HNSW is approximate by
construction, so it can only ever cost recall; the rule below is registered here
before the measurement so the latency number cannot argue for itself.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from pydantic import BaseModel, ConfigDict

from taxverity.evals.ladder import TOLERANCE, Verdict
from taxverity.evals.metrics import CreditMode, RunReport
from taxverity.retrieval.base import ScoredChunk

# The query path this search sits inside is dominated by two vendor round trips
# measured earlier: query embedding at p50 324 ms (Step 4.6) and reranking at
# p95 1,126 ms (Step 5.6). An exact scan under this budget is not the binding
# constraint, and an approximate index bought against a constraint that does not
# bind is a recall risk taken for nothing.
EXACT_P95_BUDGET_MS = 100.0

HNSW_M = 16
HNSW_EF_CONSTRUCTION = 64
HNSW_EF_SEARCH = 40


class Agreement(BaseModel):
    """How two rankings of the same queries differ, chunk by chunk."""

    model_config = ConfigDict(frozen=True)

    k: int
    queries: int
    identical: tuple[str, ...]
    reordered: tuple[str, ...]
    different: tuple[str, ...]
    mean_overlap: float
    max_score_delta: float
    worst_query: str | None

    @property
    def identical_share(self) -> float:
        return len(self.identical) / self.queries if self.queries else 0.0


def agreement(
    a: Mapping[str, Sequence[ScoredChunk]],
    b: Mapping[str, Sequence[ScoredChunk]],
    k: int,
) -> Agreement:
    """Compared by chunk id, not citation: two chunks can share neither, and the
    id is what the index actually returned."""
    if a.keys() != b.keys():
        raise ValueError("the two runs do not cover the same queries")
    identical, reordered, different = [], [], []
    overlaps: list[float] = []
    max_delta, worst = 0.0, None
    for query_id in a:
        mine = list(a[query_id])[:k]
        theirs = list(b[query_id])[:k]
        mine_ids = [hit.chunk.chunk_id for hit in mine]
        their_ids = [hit.chunk.chunk_id for hit in theirs]
        if mine_ids == their_ids:
            identical.append(query_id)
        elif set(mine_ids) == set(their_ids):
            reordered.append(query_id)
        else:
            different.append(query_id)
        shared = set(mine_ids) & set(their_ids)
        overlaps.append(len(shared) / len(mine_ids) if mine_ids else 1.0)
        their_scores = {hit.chunk.chunk_id: hit.score for hit in theirs}
        for hit in mine:
            if hit.chunk.chunk_id not in their_scores:
                continue
            delta = abs(hit.score - their_scores[hit.chunk.chunk_id])
            if delta > max_delta:
                max_delta, worst = delta, query_id
    return Agreement(
        k=k,
        queries=len(a),
        identical=tuple(identical),
        reordered=tuple(reordered),
        different=tuple(different),
        mean_overlap=sum(overlaps) / len(overlaps) if overlaps else 0.0,
        max_score_delta=max_delta,
        worst_query=worst,
    )


def judge_parity(found: Agreement, *, score_tolerance: float) -> Verdict:
    """Two exact searches of the same vectors must return the same ranking. A
    score difference is float noise up to the tolerance; a different chunk, or
    the same chunks in a different order, is a broken tie-break."""
    reasons = []
    if found.different:
        reasons.append(
            f"{len(found.different)} quer{'y' if len(found.different) == 1 else 'ies'} "
            f"returned a different set, first {found.different[0]}"
        )
    if found.reordered:
        reasons.append(
            f"{len(found.reordered)} quer{'y' if len(found.reordered) == 1 else 'ies'} "
            f"returned the same chunks in a different order, first {found.reordered[0]}"
        )
    if found.max_score_delta > score_tolerance:
        reasons.append(
            f"max score difference {found.max_score_delta:.2e} exceeds "
            f"{score_tolerance:.0e} ({found.worst_query})"
        )
    return Verdict(adopted=not reasons, reasons=tuple(reasons))


def judge_hnsw(
    exact_p95_ms: float,
    candidate: RunReport,
    incumbent: RunReport,
    *,
    budget_ms: float = EXACT_P95_BUDGET_MS,
) -> Verdict:
    """Necessity then safety. An approximate index is adopted only if exact
    search is actually over budget, and only if it costs nothing measurable on
    the gold set."""
    if candidate.k != incumbent.k:
        raise ValueError(f"cannot compare k={candidate.k} against k={incumbent.k}")
    new = candidate.overall[CreditMode.LENIENT]
    old = incumbent.overall[CreditMode.LENIENT]
    reasons = []
    if exact_p95_ms <= budget_ms:
        reasons.append(
            f"exact search p95 {exact_p95_ms:.1f} ms is within the {budget_ms:.0f} ms "
            f"budget, so there is no latency constraint to relieve"
        )
    if new.recall < old.recall - TOLERANCE:
        reasons.append(f"lenient recall fell {old.recall:.3f} -> {new.recall:.3f}")
    if new.ndcg < old.ndcg - TOLERANCE:
        reasons.append(f"lenient nDCG fell {old.ndcg:.3f} -> {new.ndcg:.3f}")
    return Verdict(adopted=not reasons, reasons=tuple(reasons))
