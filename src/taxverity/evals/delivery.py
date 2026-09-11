"""Steps 5.3 and 5.4 — the rules evidence delivery is held to, each written
before its measurement script first ran (ADR-082, ADR-083)."""

from __future__ import annotations

from taxverity.evals.gold import QuerySlice
from taxverity.evals.ladder import TOLERANCE, Verdict
from taxverity.evals.metrics import CreditMode, RunReport


def judge_delivery(delivered: RunReport, ranked: RunReport) -> Verdict:
    """Delivery is adopted only if its evidence credits every label the adopted
    ranking's top k credited: lenient recall must not fall, overall or on any
    slice. No rise is required. Delivery exists to fit a budget without losing
    evidence, not to find more.

    The two reports are scored at different k on purpose. The ranking is scored
    at the k it was adopted at. The pack is scored at its pool depth, because
    the budget bounds it, not k.
    """
    if {s.query_id for s in delivered.scored} != {s.query_id for s in ranked.scored}:
        raise ValueError("the two runs do not cover the same queries")
    reasons = []
    after = delivered.overall[CreditMode.LENIENT].recall
    before = ranked.overall[CreditMode.LENIENT].recall
    if after < before - TOLERANCE:
        reasons.append(f"lenient recall fell {before:.3f} -> {after:.3f}")
    for member, scores in ranked.per_slice.items():
        before = scores[CreditMode.LENIENT].recall
        after = delivered.per_slice[member][CreditMode.LENIENT].recall
        if after < before - TOLERANCE:
            reasons.append(
                f"{member.value} slice lenient recall fell {before:.3f} -> {after:.3f}"
            )
    return Verdict(adopted=not reasons, reasons=tuple(reasons))


def judge_expansion(expanded: RunReport, packed: RunReport) -> Verdict:
    """Cross-reference expansion is adopted only if the crossref slice's lenient
    recall rises, and neither the overall figure nor any slice falls, against
    the Step 5.3 pack built from the same ranking at the same budget.

    The rise is the point of the step (plan 5.5). The no-fall clause is the
    "acceptable noise" guard: referenced text shares the budget, so the risk is
    that it displaces a retrieved label. Tokens handed to negatives are
    reported, not gated, because scope is Phase 12's.
    """
    if {s.query_id for s in expanded.scored} != {s.query_id for s in packed.scored}:
        raise ValueError("the two runs do not cover the same queries")
    reasons = list(judge_delivery(expanded, packed).reasons)
    before = packed.per_slice[QuerySlice.CROSSREF][CreditMode.LENIENT].recall
    after = expanded.per_slice[QuerySlice.CROSSREF][CreditMode.LENIENT].recall
    if after <= before + TOLERANCE:
        reasons.append(f"crossref slice lenient recall did not rise ({before:.3f} -> {after:.3f})")
    return Verdict(adopted=not reasons, reasons=tuple(reasons))
