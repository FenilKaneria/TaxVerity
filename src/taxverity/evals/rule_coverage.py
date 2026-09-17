"""Step 20.1 — Legal Rule Coverage@k and friends, scored over the
`gold_r20.R20GoldQuery` benchmark. Deterministic, no model, same discipline
as `evals.metrics` (which this module deliberately does not modify — the
Step 3.2 gold-v2 floors stay pinned to what they already measure).

Two things are measured that `evals.metrics` does not:

- **Precision@k**, meaningful here because a query names several
  `rule_units`, so "how much of the top k is actually relevant" is not the
  same question as recall.
- **Legal Rule Coverage@k**: the fraction of a question's `rule_units`
  (governing provision + its conditions/limits/exceptions/cross-references)
  credited within the top k, and `coverage_complete@k`, true only when every
  one is. A retriever can score perfect *recall* on `required` (find the
  governing section) while covering none of a rule's actual conditions —
  that gap is exactly what R20's `reason`/`applicability` stage depends on
  not having.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from pydantic import BaseModel, ConfigDict

from taxverity.evals.gold_r20 import R20GoldQuery, R20Slice
from taxverity.evals.metrics import CitationIndex, CreditMode, _dedupe, credits


class CoverageScore(BaseModel):
    model_config = ConfigDict(frozen=True)

    query_id: str
    slice: R20Slice
    k: int
    retrieved: int
    # Against `required` (the governing provision alone) — same meaning as
    # evals.metrics.Scores.recall/mrr, kept local so this module has no
    # runtime dependency on gold-v2's GoldQuery type.
    governing_recall: float
    governing_mrr: float
    precision: float
    rule_coverage: float
    coverage_complete: bool
    missing_units: tuple[str, ...]


def score_coverage(
    query: R20GoldQuery,
    retrieved: Sequence[str],
    k: int,
    index: CitationIndex,
    mode: CreditMode = CreditMode.LENIENT,
) -> CoverageScore:
    if k < 1:
        raise ValueError(f"k must be at least 1, not {k}")
    if query.slice is R20Slice.NEGATIVE:
        raise ValueError(f"{query.query_id}: a negative query has nothing to score")
    for citation in query.rule_units:
        index.resolve(citation)

    ranked = _dedupe(retrieved, k)

    governing_hit = 0
    for rank, citation in enumerate(ranked, start=1):
        if any(credits(citation, want, mode) for want in query.required):
            governing_hit = rank
            break

    hit_units = {
        want
        for want in query.rule_units
        if any(credits(citation, want, mode) for citation in ranked)
    }
    relevant_in_top_k = sum(
        1
        for citation in ranked
        if any(credits(citation, want, mode) for want in query.rule_units)
    )

    return CoverageScore(
        query_id=query.query_id,
        slice=query.slice,
        k=k,
        retrieved=len(ranked),
        governing_recall=1.0 if governing_hit else 0.0,
        governing_mrr=1 / governing_hit if governing_hit else 0.0,
        precision=relevant_in_top_k / len(ranked) if ranked else 0.0,
        rule_coverage=len(hit_units) / len(query.rule_units),
        coverage_complete=len(hit_units) == len(query.rule_units),
        missing_units=tuple(sorted(set(query.rule_units) - hit_units)),
    )


class CoverageReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    k: int
    scored: tuple[CoverageScore, ...]
    mean_governing_recall: float
    mean_precision: float
    mean_rule_coverage: float
    coverage_complete_rate: float


def score_coverage_run(
    queries: Sequence[R20GoldQuery],
    runs: Mapping[str, Sequence[str]],
    k: int,
    index: CitationIndex,
) -> CoverageReport:
    answerable = [q for q in queries if q.slice is not R20Slice.NEGATIVE]
    missing = {q.query_id for q in answerable} - set(runs)
    if missing:
        raise KeyError(f"no run recorded for {sorted(missing)}")
    scored = tuple(score_coverage(q, runs[q.query_id], k, index) for q in answerable)
    n = len(scored) or 1
    return CoverageReport(
        k=k,
        scored=scored,
        mean_governing_recall=sum(s.governing_recall for s in scored) / n,
        mean_precision=sum(s.precision for s in scored) / n,
        mean_rule_coverage=sum(s.rule_coverage for s in scored) / n,
        coverage_complete_rate=sum(1 for s in scored if s.coverage_complete) / n,
    )
