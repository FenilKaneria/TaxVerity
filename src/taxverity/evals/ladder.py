"""Step 4.4b — the comparison ladder's pure parts: a cross-validation split,
parameter selection, and the adoption rule pre-registered before any run."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence

from pydantic import BaseModel, ConfigDict

from taxverity.evals.gold import GoldQuery, QuerySlice
from taxverity.evals.metrics import CreditMode, RunReport

# Float means over 64 queries: equal runs can differ in the last bits.
TOLERANCE = 1e-9

Params = tuple[float, float]


class Verdict(BaseModel):
    model_config = ConfigDict(frozen=True)

    adopted: bool
    reasons: tuple[str, ...]


def two_fold_split(
    gold: Sequence[GoldQuery],
) -> tuple[tuple[GoldQuery, ...], tuple[GoldQuery, ...]]:
    """Answerable queries by id parity. The ids were assigned in authoring
    order, not by slice, so each fold carries every slice."""
    answerable = [q for q in gold if q.slice is not QuerySlice.NEGATIVE]
    even = tuple(q for q in answerable if int(q.query_id[1:]) % 2 == 0)
    odd = tuple(q for q in answerable if int(q.query_id[1:]) % 2 == 1)
    return even, odd


def select_params(scores: Mapping[Params, float], *, prefer: Params) -> Params:
    """The best-scoring setting. A tie keeps `prefer` (the defaults), so tuning
    moves off them only on a strict gain; any other tie goes to the smallest
    setting, so the pick never depends on dict order."""
    if not scores:
        raise ValueError("no settings to select from")
    best = max(scores.values())
    if prefer in scores and scores[prefer] >= best - TOLERANCE:
        return prefer
    return min(params for params, score in scores.items() if score >= best - TOLERANCE)


def judge(
    candidate: RunReport, incumbent: RunReport, *, extra: Sequence[str] = ()
) -> Verdict:
    """Rules 1 and 2 of Step 4.4b. `extra` carries the rung-specific failures
    (cross-validation, latency) that only the caller can evaluate."""
    if candidate.k != incumbent.k:
        raise ValueError(f"cannot compare k={candidate.k} against k={incumbent.k}")
    reasons = list(extra)
    new = candidate.overall[CreditMode.LENIENT]
    old = incumbent.overall[CreditMode.LENIENT]
    if new.recall < old.recall - TOLERANCE:
        reasons.append(f"lenient recall fell {old.recall:.3f} -> {new.recall:.3f}")
    if new.ndcg < old.ndcg - TOLERANCE:
        reasons.append(f"lenient nDCG fell {old.ndcg:.3f} -> {new.ndcg:.3f}")
    if new.recall <= old.recall + TOLERANCE and new.ndcg <= old.ndcg + TOLERANCE:
        reasons.append("neither lenient recall nor lenient nDCG rose")

    sizes = Counter(score.slice for score in candidate.scored)
    for member, size in sorted(sizes.items()):
        drop = (
            incumbent.per_slice[member][CreditMode.LENIENT].recall
            - candidate.per_slice[member][CreditMode.LENIENT].recall
        )
        if drop * size > 1 + TOLERANCE:
            reasons.append(
                f"{member.value} slice lost {drop * size:.2f} queries of lenient recall"
            )
    return Verdict(adopted=not reasons, reasons=tuple(reasons))
