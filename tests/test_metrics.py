"""Step 3.2 — testing the ruler."""

from __future__ import annotations

import math
import random

import pytest

from taxverity.evals.gold import GoldQuery, QuerySlice, by_slice
from taxverity.evals.metrics import (
    CitationIndex,
    CreditMode,
    QueryScore,
    Scores,
    UnresolvedCitationError,
    credits,
    normalise_citation,
    score_query,
    score_run,
)

STRICT = CreditMode.STRICT
LENIENT = CreditMode.LENIENT


def query(query_id="q001", slice=QuerySlice.PARAPHRASE, required=("22(2)",)):
    return GoldQuery(
        query_id=query_id,
        slice=slice,
        question="what does it say",
        required=required,
        notes="fixture",
    )


class FakeIndex(CitationIndex):
    """Every citation resolves. Resolution itself is tested against the real
    chunk set below; the arithmetic tests should not need 7,561 chunks."""

    def __init__(self, known=None):
        self._known = known

    def resolve(self, citation):
        if self._known is not None and normalise_citation(citation) not in self._known:
            raise UnresolvedCitationError(citation)
        return "0" * 16


def test_a_citation_credits_itself():
    for mode in CreditMode:
        assert credits("22(2)", "22(2)", mode)


def test_an_ancestor_is_credited_only_leniently():
    assert not credits("22", "22(2)", STRICT)
    assert credits("22", "22(2)", LENIENT)


def test_a_descendant_is_never_credited():
    """ADR-055 puts 22(2)'s text inside 22, not the other way round."""
    for mode in CreditMode:
        assert not credits("22(2)", "22", mode)


def test_a_sibling_is_never_credited():
    for mode in CreditMode:
        assert not credits("22(3)", "22(2)", mode)
        assert not credits("23(2)", "22(2)", mode)


def test_a_schedule_and_a_section_do_not_credit_each_other():
    for mode in CreditMode:
        assert not credits("Schedule II(2)", "2(2)", mode)


def test_credit_is_by_path_not_by_spelling():
    assert credits(" 22(2) ", "22(2)", STRICT)


def test_a_perfect_retriever_scores_one():
    scored = score_query(
        query(required=("22(2)", "22(5)")), ["22(2)", "22(5)"], 5, FakeIndex()
    )
    for mode_scores in (scored.strict, scored.lenient):
        assert mode_scores == Scores(recall=1.0, mrr=1.0, ndcg=1.0)


def test_a_retriever_returning_nothing_scores_zero():
    scored = score_query(query(), [], 5, FakeIndex())
    assert scored.strict == Scores(recall=0.0, mrr=0.0, ndcg=0.0)
    assert scored.retrieved == 0


def test_an_ancestor_only_retriever_separates_the_two_modes():
    """The number ADR-059 note 3 exists for: strict says missed, lenient says
    imprecise, and reporting only one of them hides which happened."""
    scored = score_query(query(required=("22(2)",)), ["22"], 5, FakeIndex())
    assert scored.strict.recall == 0.0
    assert scored.lenient.recall == 1.0


def test_the_metrics_are_hand_computable():
    scored = score_query(
        query(required=("22(2)", "22(5)")), ["1", "22(2)", "22(5)"], 5, FakeIndex()
    )
    dcg = 1 / math.log2(3) + 1 / math.log2(4)
    ideal = 1 + 1 / math.log2(3)
    assert scored.strict.recall == 1.0
    assert scored.strict.mrr == 0.5
    assert scored.strict.ndcg == pytest.approx(dcg / ideal)


def test_one_chunk_covering_two_labels_does_not_score_a_perfect_ndcg():
    """Gain is coverage, not relevance. Counting a root as relevant to each of
    its own labels would let a single result score a two-label query 1.0."""
    scored = score_query(query(required=("22(2)", "22(5)")), ["22"], 5, FakeIndex())
    assert scored.lenient.recall == 1.0
    assert scored.lenient.ndcg == pytest.approx(1 / (1 + 1 / math.log2(3)))


def test_a_duplicate_result_does_not_earn_credit_twice():
    scored = score_query(
        query(required=("22(2)", "22(5)")), ["22(2)", "22(2)", "22(5)"], 5, FakeIndex()
    )
    assert scored.retrieved == 2
    assert scored.strict.ndcg == 1.0


def test_k_truncates_the_ranking():
    hit_at_four = ["1", "2", "3", "22(2)"]
    assert score_query(query(), hit_at_four, 3, FakeIndex()).strict.recall == 0.0
    assert score_query(query(), hit_at_four, 4, FakeIndex()).strict.recall == 1.0


def test_k_must_be_positive():
    with pytest.raises(ValueError, match="at least 1"):
        score_query(query(), ["22(2)"], 0, FakeIndex())


def test_a_negative_query_is_refused_by_the_scorer():
    negative = GoldQuery(
        query_id="q002",
        slice=QuerySlice.NEGATIVE,
        question="what is the GST rate",
        required=(),
        notes="fixture",
    )
    with pytest.raises(ValueError, match="nothing to score"):
        score_query(negative, ["22(2)"], 5, FakeIndex())


def test_a_label_naming_no_chunk_is_an_error_not_a_miss():
    with pytest.raises(UnresolvedCitationError):
        score_query(query(required=("999(9)",)), ["999(9)"], 5, FakeIndex(known=set()))


def test_a_missing_run_is_an_error_not_a_zero():
    with pytest.raises(KeyError, match="q001"):
        score_run([query()], {}, 5, FakeIndex())


def test_a_run_reports_per_slice_and_counts_negatives():
    queries = [
        query("q001", QuerySlice.CITATION, ("22(2)",)),
        query("q002", QuerySlice.PARAPHRASE, ("23(1)",)),
        GoldQuery(
            query_id="q003",
            slice=QuerySlice.NEGATIVE,
            question="gst rate",
            required=(),
            notes="fixture",
        ),
    ]
    runs = {"q001": ["22(2)"], "q002": ["9"]}
    report = score_run(queries, runs, 5, FakeIndex())
    assert report.negatives == 1
    assert len(report.scored) == 2
    assert report.per_slice[QuerySlice.CITATION][STRICT].recall == 1.0
    assert report.per_slice[QuerySlice.PARAPHRASE][STRICT].recall == 0.0
    assert report.overall[STRICT].recall == 0.5
    assert QuerySlice.NEGATIVE not in report.per_slice


def test_an_empty_slice_averages_to_zero_rather_than_dividing_by_zero():
    report = score_run(
        [query("q001", QuerySlice.CITATION)], {"q001": ["22(2)"]}, 5, FakeIndex()
    )
    assert report.per_slice[QuerySlice.CROSSREF][STRICT] == Scores(
        recall=0.0, mrr=0.0, ndcg=0.0
    )


def test_query_scores_are_frozen():
    scored = score_query(query(), ["22(2)"], 5, FakeIndex())
    with pytest.raises(ValueError):
        scored.k = 9  # type: ignore[misc]
    assert isinstance(scored, QueryScore)


def test_the_index_resolves_every_gold_label_to_a_chunk(gold, chunks):
    """The resolution ADR-059 moved to measurement time. A citation that stops
    resolving is a broken label, and must not look like a retrieval failure."""
    index = CitationIndex(chunks)
    assert len(index) == len(chunks)
    for q in gold:
        for citation in q.required:
            assert len(index.resolve(citation)) == 16


def test_a_perfect_retriever_scores_one_on_the_real_gold_set(gold, chunks):
    index = CitationIndex(chunks)
    answerable = [q for q in gold if q.slice is not QuerySlice.NEGATIVE]
    runs = {q.query_id: list(q.required) for q in answerable}
    report = score_run(gold, runs, 10, index)
    assert report.overall[STRICT] == Scores(recall=1.0, mrr=1.0, ndcg=1.0)
    assert report.negatives == len(by_slice(gold)[QuerySlice.NEGATIVE])


def test_a_random_retriever_scores_near_chance_on_the_real_gold_set(gold, chunks):
    """The other half of testing a ruler: a ruler that scores everything 1.0
    measures nothing. 10 draws from 7,561 chunks land on a label occasionally
    — seed 0 does, once — so the bound is near-chance, not exactly zero."""
    index = CitationIndex(chunks)
    paths = sorted(chunk.node_path for chunk in chunks)
    rng = random.Random(0)
    answerable = [q for q in gold if q.slice is not QuerySlice.NEGATIVE]
    runs = {q.query_id: rng.sample(paths, 10) for q in answerable}
    report = score_run(gold, runs, 10, index)
    assert report.overall[STRICT].recall < 0.05
    assert report.overall[LENIENT].recall < 0.10
    assert report.overall[LENIENT].mrr < 0.05
