"""The loss-sign fix's adoption rule, registered before the fix was written or
the held-out turns were measured (ADR-099).

Step 7.7 found the node dropping the sign of a stated loss (t013, t017). A loss
read as income inverts the tax, so the rule is all-or-nothing on sign, not a
rate:

1. every held-out turn is clean, the 8 loss turns and the 4 controls alike: a
   control that comes back negative, or drops its fact, is the fix
   over-correcting;
2. t013 and t017 of the 7.7 gold set now carry the right value;
3. no 7.7 turn that was clean before the fix is unclean after it.
"""

from __future__ import annotations

from collections.abc import Sequence

from taxverity.evals.extraction import ExtractionScore, TurnJudgement
from taxverity.evals.ladder import Verdict

SIGN_FAILURES_BEFORE = ("t013", "t017")


def unclean(judgement: TurnJudgement) -> bool:
    return bool(judgement.missed or judgement.spurious or judgement.value_wrong)


def judge_loss_fix(
    holdout: ExtractionScore,
    gold_before: ExtractionScore,
    gold_after: ExtractionScore,
    *,
    fixed: Sequence[str] = SIGN_FAILURES_BEFORE,
) -> Verdict:
    reasons = [f"held-out {t.turn_id} is not clean" for t in holdout.turns if unclean(t)]
    after = {t.turn_id: t for t in gold_after.turns}
    if set(after) != {t.turn_id for t in gold_before.turns}:
        raise ValueError("the two gold runs do not cover the same turns")
    reasons += [f"{turn_id} still has a wrong value" for turn_id in fixed if after[turn_id].value_wrong]
    reasons += [
        f"{t.turn_id} was clean and is not now"
        for t in gold_before.turns
        if not unclean(t) and unclean(after[t.turn_id])
    ]
    return Verdict(adopted=not reasons, reasons=tuple(reasons))
