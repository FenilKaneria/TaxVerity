"""Step 5.2 — the adoption rule for hybrid retrieval, registered in code before
the measurement ran (ADR-010, ADR-081).

ADR-010 asks for a win against both single retrievers, per slice. A strict rise
on every slice is unreachable, because both incumbents already score 1.000 on
the citation slice. So a win is an overall rise, with no slice falling.
"""

from __future__ import annotations

from collections.abc import Mapping

from taxverity.evals.gold import QuerySlice
from taxverity.evals.ladder import TOLERANCE, Verdict
from taxverity.evals.metrics import CreditMode, RunReport


def judge_hybrid(candidate: RunReport, incumbents: Mapping[str, RunReport]) -> Verdict:
    """Adopt only if, against every incumbent, lenient recall and lenient nDCG
    both hold, at least one rises, and no slice loses lenient recall."""
    if not incumbents:
        raise ValueError("a hybrid is judged against at least one incumbent")
    reasons: list[str] = []
    for name, incumbent in incumbents.items():
        if candidate.k != incumbent.k:
            raise ValueError(f"cannot compare k={candidate.k} against k={incumbent.k}")
        new = candidate.overall[CreditMode.LENIENT]
        old = incumbent.overall[CreditMode.LENIENT]
        if new.recall < old.recall - TOLERANCE:
            reasons.append(f"vs {name}: lenient recall fell {old.recall:.3f} -> {new.recall:.3f}")
        if new.ndcg < old.ndcg - TOLERANCE:
            reasons.append(f"vs {name}: lenient nDCG fell {old.ndcg:.3f} -> {new.ndcg:.3f}")
        if new.recall <= old.recall + TOLERANCE and new.ndcg <= old.ndcg + TOLERANCE:
            reasons.append(f"vs {name}: neither lenient recall nor lenient nDCG rose")
        for member, scores in sorted(incumbent.per_slice.items()):
            if member is QuerySlice.NEGATIVE:
                continue
            before = scores[CreditMode.LENIENT].recall
            after = candidate.per_slice[member][CreditMode.LENIENT].recall
            if after < before - TOLERANCE:
                reasons.append(
                    f"vs {name}: {member.value} slice lenient recall fell "
                    f"{before:.3f} -> {after:.3f}"
                )
    return Verdict(adopted=not reasons, reasons=tuple(reasons))
