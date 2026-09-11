"""Step 4.6 — query-by-query comparison and the pre-registered long-chunk rule."""

from __future__ import annotations

import pytest

from taxverity.evals.baseline import QueryOutcome
from taxverity.evals.comparison import (
    INTRUSION_THRESHOLD,
    WATCH_SET,
    head_to_head,
    intrusions,
    judge_exclusion,
    rule_triggered,
)
from taxverity.evals.gold import GoldQuery, QuerySlice
from taxverity.evals.metrics import (
    CreditMode,
    QueryScore,
    RunReport,
    Scores,
    normalise_citation,
)

P = QuerySlice.PARAPHRASE


def query(number: int, *required: str) -> GoldQuery:
    return GoldQuery(
        query_id=f"q{number:03d}",
        slice=P if required else QuerySlice.NEGATIVE,
        question="What is taxed?",
        required=required,
        notes="synthetic",
    )


def report(recall: float, ndcg: float, k: int = 10) -> RunReport:
    scores = Scores(recall=recall, mrr=0.0, ndcg=ndcg)
    scored = (
        QueryScore(query_id="q001", slice=P, k=k, retrieved=k, strict=scores, lenient=scores),
    )
    return RunReport(
        k=k,
        scored=scored,
        overall=dict.fromkeys(CreditMode, scores),
        per_slice={P: dict.fromkeys(CreditMode, scores)},
        negatives=0,
    )


def outcome(query_id: str, recall: float, missed=(), required=("22", "23")) -> QueryOutcome:
    return QueryOutcome(
        query_id=query_id,
        slice=P,
        question="What is taxed?",
        required=required,
        retrieved=(),
        top_score=None,
        strict_recall=recall,
        lenient_recall=recall,
        first_hit_rank=None,
        missed=missed,
    )


# --- the watch set -------------------------------------------------------------


def test_the_watch_set_is_the_ten_registered_citations_in_canonical_form():
    assert len(WATCH_SET) == len(set(WATCH_SET)) == 10
    assert all(normalise_citation(c) == c for c in WATCH_SET)


# --- intrusions ------------------------------------------------------------------


def test_a_watched_chunk_counts_only_where_it_is_neither_label_nor_ancestor():
    gold = [query(1, "2(5)"), query(2, "393"), query(3, "22"), query(4)]
    runs = {
        "q001": ["2", "22"],  # 2 is an ancestor of the label: fair
        "q002": ["393", "206(1)"],  # 393 is the label: fair; 206(1) intrudes
        "q003": ["2", "206"],  # both intrude
        "q004": ["2"],  # a negative has no label, so any appearance intrudes
    }
    counts = intrusions(runs, gold)
    assert {c: n for c, n in counts.items() if n} == {"2": 2, "206": 1, "206(1)": 1}
    assert counts.keys() == set(WATCH_SET)


def test_a_watched_descendant_of_the_label_still_intrudes():
    """206(1) under a `206` label is neither the label nor its ancestor; ADR-060
    credits no descendant, and neither does the rule."""
    counts = intrusions({"q001": ["206(1)"]}, [query(1, "206")])
    assert counts["206(1)"] == 1


def test_only_the_run_given_is_read():
    counts = intrusions({"q001": ["22", "23"]}, [query(1, "22")])
    assert not any(counts.values())


def test_a_query_with_no_run_is_an_error_not_a_zero():
    with pytest.raises(KeyError):
        intrusions({}, [query(1, "22")])


@pytest.mark.parametrize(
    ("counts", "fires"),
    [
        ({"2": INTRUSION_THRESHOLD - 1}, False),
        ({"2": INTRUSION_THRESHOLD}, True),
        ({"2": 0, "393": INTRUSION_THRESHOLD + 3}, True),
        ({}, False),
    ],
)
def test_the_rule_fires_at_the_registered_threshold(counts, fires):
    assert INTRUSION_THRESHOLD == 8
    assert rule_triggered(counts) is fires


# --- the exclusion verdict -------------------------------------------------------


def test_an_exclusion_that_costs_nothing_is_adopted_without_needing_a_rise():
    assert judge_exclusion(report(0.5, 0.4), report(0.5, 0.4)).adopted


def test_float_noise_is_not_a_fall():
    assert judge_exclusion(report(0.5 - 1e-12, 0.4), report(0.5, 0.4)).adopted


@pytest.mark.parametrize(
    ("candidate", "reason"),
    [((0.49, 0.4), "recall fell"), ((0.5, 0.39), "nDCG fell")],
)
def test_an_exclusion_that_costs_either_measure_is_rejected(candidate, reason):
    verdict = judge_exclusion(report(*candidate), report(0.5, 0.4))
    assert not verdict.adopted
    assert any(reason in r for r in verdict.reasons)


def test_reports_at_different_k_cannot_be_compared():
    with pytest.raises(ValueError, match="k="):
        judge_exclusion(report(0.5, 0.4, k=10), report(0.5, 0.4, k=20))


# --- head to head --------------------------------------------------------------


def test_each_query_lands_in_exactly_one_bucket():
    a = [
        outcome("q001", 1.0),
        outcome("q002", 0.0, missed=("22", "23")),
        outcome("q003", 1.0),
        outcome("q004", 0.5, missed=("23",)),
    ]
    b = [
        outcome("q001", 0.5, missed=("22",)),
        outcome("q002", 1.0),
        outcome("q003", 1.0),
        outcome("q004", 0.5, missed=("23",)),
    ]
    duel = head_to_head(a, b)
    assert duel.a_wins == ("q001",)
    assert duel.b_wins == ("q002",)
    assert duel.both_complete == ("q003",)
    assert duel.tied_incomplete == ("q004",)


def test_union_credits_a_label_found_by_either_run():
    a = [outcome("q001", 0.5, missed=("22",)), outcome("q002", 0.5, missed=("23",))]
    b = [outcome("q001", 0.5, missed=("23",)), outcome("q002", 0.5, missed=("23",))]
    duel = head_to_head(a, b)
    # q001: each finds the label the other missed. q002: both miss 23.
    assert duel.union == {"q001": 1.0, "q002": 0.5}
    assert duel.union_recall == pytest.approx(0.75)


def test_runs_over_different_queries_cannot_be_compared():
    with pytest.raises(ValueError, match="same queries"):
        head_to_head([outcome("q001", 1.0)], [outcome("q002", 1.0)])
