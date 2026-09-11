"""Step 4.4b — the comparison ladder: split, parameter selection, adoption rule."""

from __future__ import annotations

import pytest

from taxverity.evals.gold import GoldQuery, QuerySlice
from taxverity.evals.ladder import judge, select_params, two_fold_split
from taxverity.evals.metrics import CreditMode, QueryScore, RunReport, Scores

DEFAULTS = (1.5, 0.75)


def query(number, slice_=QuerySlice.PARAPHRASE):
    return GoldQuery(
        query_id=f"q{number:03d}",
        slice=slice_,
        question="What is taxed?" if slice_ is not QuerySlice.CITATION else "What does section 22 say?",
        required=() if slice_ is QuerySlice.NEGATIVE else ("22",),
        notes="synthetic",
    )


def report(recalls, ndcgs=None, k=10):
    """`recalls` maps query id to (slice, lenient recall); nDCG defaults to recall."""
    ndcgs = ndcgs or {qid: recall for qid, (_, recall) in recalls.items()}
    scored = tuple(
        QueryScore(
            query_id=qid,
            slice=slice_,
            k=k,
            retrieved=k,
            strict=Scores(recall=recall, mrr=0.0, ndcg=ndcgs[qid]),
            lenient=Scores(recall=recall, mrr=0.0, ndcg=ndcgs[qid]),
        )
        for qid, (slice_, recall) in recalls.items()
    )

    def mean(scores):
        n = len(scores)
        return Scores(
            recall=sum(s.lenient.recall for s in scores) / n,
            mrr=0.0,
            ndcg=sum(s.lenient.ndcg for s in scores) / n,
        )

    slices = {s.slice for s in scored}
    return RunReport(
        k=k,
        scored=scored,
        overall={mode: mean(scored) for mode in CreditMode},
        per_slice={
            member: {mode: mean([s for s in scored if s.slice is member]) for mode in CreditMode}
            for member in slices
        },
        negatives=0,
    )


P, C = QuerySlice.PARAPHRASE, QuerySlice.CITATION


def test_the_split_is_disjoint_complete_and_drops_negatives():
    gold = [query(n) for n in range(1, 9)] + [query(9, QuerySlice.NEGATIVE)]
    even, odd = two_fold_split(gold)
    ids = [q.query_id for q in even + odd]
    assert len(ids) == len(set(ids)) == 8
    assert all(q.slice is not QuerySlice.NEGATIVE for q in even + odd)
    assert [q.query_id for q in even] == ["q002", "q004", "q006", "q008"]


def test_the_split_depends_on_ids_not_on_order():
    gold = [query(n) for n in range(1, 21)]
    forward = two_fold_split(gold)
    backward = two_fold_split(list(reversed(gold)))
    for fold, other in zip(forward, backward, strict=True):
        assert {q.query_id for q in fold} == {q.query_id for q in other}


def test_select_picks_the_best_setting():
    assert select_params({DEFAULTS: 0.5, (1.2, 0.3): 0.6}, prefer=DEFAULTS) == (1.2, 0.3)


def test_a_tie_keeps_the_defaults():
    assert select_params({(0.9, 0.3): 0.6, DEFAULTS: 0.6}, prefer=DEFAULTS) == DEFAULTS


def test_a_tie_without_the_defaults_goes_to_the_smallest_setting():
    scores = {(2.1, 0.9): 0.6, (0.9, 0.5): 0.6, DEFAULTS: 0.4}
    assert select_params(scores, prefer=DEFAULTS) == (0.9, 0.5)


def test_select_refuses_an_empty_grid():
    with pytest.raises(ValueError):
        select_params({}, prefer=DEFAULTS)


def test_a_strict_gain_with_nothing_falling_is_adopted():
    old = report({"q001": (P, 0.0), "q002": (P, 1.0)})
    new = report({"q001": (P, 1.0), "q002": (P, 1.0)})
    assert judge(new, old).adopted


def test_a_flat_rung_is_rejected():
    old = report({"q001": (P, 1.0)})
    verdict = judge(report({"q001": (P, 1.0)}), old)
    assert not verdict.adopted
    assert "neither" in verdict.reasons[0]


def test_a_fall_in_ndcg_rejects_even_when_recall_rises():
    old = report({"q001": (P, 0.5), "q002": (P, 1.0)}, {"q001": 0.5, "q002": 1.0})
    new = report({"q001": (P, 1.0), "q002": (P, 1.0)}, {"q001": 0.5, "q002": 0.6})
    verdict = judge(new, old)
    assert not verdict.adopted
    assert any("nDCG fell" in reason for reason in verdict.reasons)


def test_a_slice_losing_more_than_one_query_rejects_despite_an_overall_gain():
    old = report(
        {"q001": (C, 1.0), "q002": (C, 1.0), **{f"q1{n:02d}": (P, 0.0) for n in range(6)}}
    )
    new = report(
        {"q001": (C, 0.0), "q002": (C, 0.0), **{f"q1{n:02d}": (P, 1.0) for n in range(6)}}
    )
    verdict = judge(new, old)
    assert not verdict.adopted
    assert any("citation slice lost 2.00" in reason for reason in verdict.reasons)


def test_a_slice_losing_exactly_one_query_is_tolerated():
    old = report({"q001": (C, 1.0), **{f"q1{n:02d}": (P, 0.0) for n in range(3)}})
    new = report({"q001": (C, 0.0), **{f"q1{n:02d}": (P, 1.0) for n in range(3)}})
    assert judge(new, old).adopted


def test_a_caller_supplied_failure_rejects():
    old = report({"q001": (P, 0.0)})
    new = report({"q001": (P, 1.0)})
    verdict = judge(new, old, extra=["p95 180.0 ms exceeds 100 ms"])
    assert not verdict.adopted
    assert verdict.reasons == ("p95 180.0 ms exceeds 100 ms",)


def test_reports_at_different_k_cannot_be_compared():
    with pytest.raises(ValueError, match="k=5"):
        judge(report({"q001": (P, 1.0)}, k=5), report({"q001": (P, 1.0)}))
