"""Step 5.8 — the rules the term bridge and the definitions pull are held to,
written before `scripts/measure_bridge.py` first ran (ADR-086)."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

from taxverity.evals.gold import GoldQuery, QuerySlice
from taxverity.evals.ladder import Verdict
from taxverity.evals.metrics import CreditMode, credits

# The bridged questions' vectors and rerank scores, kept apart from the gold
# files so those stay exactly what Steps 5.2 and 5.6 measured. Keyed by the
# rewritten text, so a changed map misses rather than reusing old scores.
BRIDGE_VECTORS_FILENAME = "bridge_query_vectors.json"
BRIDGE_SCORES_FILENAME = "bridge_rerank_scores.json"

Delivered = Mapping[str, frozenset[str]]


def delivered(gold: Iterable[GoldQuery], packs: Mapping[str, Sequence[str]]) -> dict[str, frozenset[str]]:
    """Per answerable question, the labels some delivered unit credits
    leniently (ADR-060)."""
    return {
        query.query_id: frozenset(
            label
            for label in query.required
            if any(credits(unit, label, CreditMode.LENIENT) for unit in packs[query.query_id])
        )
        for query in gold
        if query.slice is not QuerySlice.NEGATIVE
    }


def lost(before: Delivered, after: Delivered) -> tuple[tuple[str, str], ...]:
    if before.keys() != after.keys():
        raise ValueError("the two runs do not cover the same queries")
    return tuple(sorted((q, label) for q in before for label in before[q] - after[q]))


def gained(before: Delivered, after: Delivered) -> tuple[tuple[str, str], ...]:
    return lost(after, before)


def judge_bridge(
    before: Delivered,
    after: Delivered,
    cohort: Iterable[tuple[str, str]],
    expanded_citation_queries: Sequence[str],
) -> Verdict:
    """The bridge is adopted only if it recovers at least one label of the
    frozen vocabulary cohort (ADR-085), no answerable question loses a label
    it was delivered, and no citation-slice question is rewritten.

    The no-loss clause is per question, stricter than Step 5.4's per-slice
    rule: the bridge fires on few questions, so one loss would hide in a slice
    average. The citation clause is plan 5.8's "no false expansion on
    already-statutory queries".
    """
    cohort = tuple(cohort)
    reasons = []
    recovered = [pair for pair in cohort if pair[1] in after[pair[0]]]
    if not recovered:
        reasons.append(f"recovered none of the {len(cohort)} vocabulary labels")
    reasons += [f"{q} lost {label}" for q, label in lost(before, after)]
    if expanded_citation_queries:
        reasons.append(f"rewrote citation-slice questions: {', '.join(expanded_citation_queries)}")
    return Verdict(adopted=not reasons, reasons=tuple(reasons))


def judge_definitions(before: Delivered, after: Delivered) -> Verdict:
    """The definitions pull is adopted only if it delivers a label the bridged
    pack did not, and loses none. Judged against the bridged pack, so the two
    decisions cannot mask each other."""
    reasons = []
    if not gained(before, after):
        reasons.append("delivered no label the bridged pack missed")
    reasons += [f"{q} lost {label}" for q, label in lost(before, after)]
    return Verdict(adopted=not reasons, reasons=tuple(reasons))
