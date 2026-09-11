"""Step 5.3 — the rule evidence delivery is held to, written before
`scripts/measure_evidence.py` first ran (ADR-082)."""

from __future__ import annotations

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
