"""Step 4.6 — the pure parts of the dense measurement: how two retrievers compare
query by query, and the long-chunk rule pre-registered at R15 before any dense
result existed (PLAN 4.6, ADR-075)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from pydantic import BaseModel, ConfigDict

from taxverity.evals.baseline import QueryOutcome
from taxverity.evals.gold import GoldQuery
from taxverity.evals.ladder import TOLERANCE, Verdict
from taxverity.evals.metrics import CreditMode, RunReport, credits, normalise_citation

# The ten chunks over 4,096 real tokens, named in PLAN 4.6 at R15. Fixed as
# registered: re-deriving the set after the dense run exists would let the
# result choose its own watch list.
WATCH_SET = (
    "2",
    "393",
    "Schedule III",
    "206",
    "Schedule XI",
    "206(1)",
    "70",
    "9",
    "Schedule XV",
    "402",
)
INTRUSION_THRESHOLD = 8


def intrusions(
    runs: Mapping[str, Sequence[str]],
    gold: Sequence[GoldQuery],
    watch: Sequence[str] = WATCH_SET,
) -> dict[str, int]:
    """Per watched chunk, the queries whose run contains it while it is neither
    a label nor an ancestor of one. The caller truncates each run to k.

    A negative carries no label, so any appearance there counts: a giant root
    surfacing for an unanswerable question is the dilution the rule watches for.
    """
    counts = dict.fromkeys(watch, 0)
    for query in gold:
        retrieved = {normalise_citation(c) for c in runs[query.query_id]}
        for citation in watch:
            if normalise_citation(citation) not in retrieved:
                continue
            if not any(
                credits(citation, label, CreditMode.LENIENT) for label in query.required
            ):
                counts[citation] += 1
    return counts


def rule_triggered(counts: Mapping[str, int]) -> bool:
    return any(count >= INTRUSION_THRESHOLD for count in counts.values())


def judge_exclusion(candidate: RunReport, incumbent: RunReport) -> Verdict:
    """Adopt the reduced index only if lenient recall and lenient nDCG both hold.
    Unlike Step 4.4b no rise is required: the intrusion count is the case for
    the exclusion, and the exclusion has only to cost nothing."""
    if candidate.k != incumbent.k:
        raise ValueError(f"cannot compare k={candidate.k} against k={incumbent.k}")
    new = candidate.overall[CreditMode.LENIENT]
    old = incumbent.overall[CreditMode.LENIENT]
    reasons = []
    if new.recall < old.recall - TOLERANCE:
        reasons.append(f"lenient recall fell {old.recall:.3f} -> {new.recall:.3f}")
    if new.ndcg < old.ndcg - TOLERANCE:
        reasons.append(f"lenient nDCG fell {old.ndcg:.3f} -> {new.ndcg:.3f}")
    return Verdict(adopted=not reasons, reasons=tuple(reasons))


class HeadToHead(BaseModel):
    """Retriever `a` against retriever `b`, by lenient recall per query."""

    model_config = ConfigDict(frozen=True)

    a_wins: tuple[str, ...]
    b_wins: tuple[str, ...]
    both_complete: tuple[str, ...]
    tied_incomplete: tuple[str, ...]
    # Per query, the share of labels credited by either run. An upper bound on
    # what fusing the two could reach, and over up to 2k results, not k.
    union: dict[str, float]

    @property
    def union_recall(self) -> float:
        return sum(self.union.values()) / len(self.union) if self.union else 0.0


def head_to_head(a: Sequence[QueryOutcome], b: Sequence[QueryOutcome]) -> HeadToHead:
    theirs = {outcome.query_id: outcome for outcome in b}
    if theirs.keys() != {outcome.query_id for outcome in a}:
        raise ValueError("the two runs do not cover the same queries")
    a_wins, b_wins, complete, tied = [], [], [], []
    union: dict[str, float] = {}
    for mine in a:
        other = theirs[mine.query_id]
        if mine.lenient_recall > other.lenient_recall + TOLERANCE:
            a_wins.append(mine.query_id)
        elif other.lenient_recall > mine.lenient_recall + TOLERANCE:
            b_wins.append(mine.query_id)
        elif mine.lenient_recall >= 1.0 - TOLERANCE:
            complete.append(mine.query_id)
        else:
            tied.append(mine.query_id)
        missed_by_both = set(mine.missed) & set(other.missed)
        union[mine.query_id] = 1.0 - len(missed_by_both) / len(mine.required)
    return HeadToHead(
        a_wins=tuple(a_wins),
        b_wins=tuple(b_wins),
        both_complete=tuple(complete),
        tied_incomplete=tuple(tied),
        union=union,
    )
