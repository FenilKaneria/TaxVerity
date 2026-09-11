"""Step 3.6 — the baseline measurement: per-query diagnosis, the k curve, and
the floors every later retrieval component must clear."""

from __future__ import annotations

import pytest

from taxverity.chunking.models import Chunk
from taxverity.corpus.nodes import NodeType
from taxverity.evals.baseline import PRIMARY_K, BaselineReport, measure
from taxverity.evals.gold import GoldQuery, QuerySlice
from taxverity.evals.metrics import CitationIndex, CreditMode
from taxverity.retrieval.base import ScoredChunk
from taxverity.retrieval.bm25 import BM25Retriever
from taxverity.retrieval.citations import CitationRetriever, ShortcutRetriever

CORPUS_VERSION = "v" * 64


def chunk(node_path, text):
    return Chunk.create(
        CORPUS_VERSION,
        node_path,
        text,
        parent_id=None if "(" not in node_path else "0" * 16,
        doc_id="income-tax-act-2025",
        node_type=NodeType.SUBSECTION if "(" in node_path else NodeType.SECTION,
        section_number=node_path.split("(")[0],
        page_start=1,
        page_end=1,
    )


CORPUS = [
    chunk("22", "Deductions from income from house property."),
    chunk("22(2)", "Interest on borrowed capital."),
    chunk("23", "Annual value of a let-out property."),
    chunk("24", "Arrears of rent received."),
]
INDEX = CitationIndex(CORPUS)

GOLD = (
    GoldQuery(
        query_id="q001",
        slice=QuerySlice.CITATION,
        question="What does section 22(2) allow?",
        required=("22(2)",),
        notes="Answered by the citation shortcut, missed by pure lexical scoring.",
    ),
    GoldQuery(
        query_id="q002",
        slice=QuerySlice.PARAPHRASE,
        question="arrears of rent",
        required=("24",),
        notes="Verbatim lexical overlap.",
    ),
    GoldQuery(
        query_id="q003",
        slice=QuerySlice.NEGATIVE,
        question="What is the GST rate on restaurant services?",
        required=(),
        notes="Nothing in this corpus answers it.",
    ),
)


class Fixed:
    """A retriever with a scripted ranking, so an outcome can be asserted
    exactly rather than inferred from a scoring function."""

    def __init__(self, ranking):
        self._ranking = [(next(c for c in CORPUS if c.node_path == p), s) for p, s in ranking]

    def search(self, query: str, k: int):
        return [ScoredChunk(chunk=c, score=s) for c, s in self._ranking][:k]


@pytest.fixture(scope="module")
def bm25():
    return measure("bm25", BM25Retriever(CORPUS), GOLD, INDEX)


def test_only_answerable_queries_become_outcomes(bm25):
    assert [o.query_id for o in bm25.outcomes] == ["q001", "q002"]


def test_a_negative_query_becomes_a_negative_outcome(bm25):
    assert [n.query_id for n in bm25.negatives] == ["q003"]
    assert bm25.negatives[0].top_citation is not None


def test_every_k_is_reported(bm25):
    assert sorted(bm25.reports) == [1, 5, 10, 20]
    assert bm25.primary is bm25.reports[PRIMARY_K]


def test_recall_is_non_decreasing_in_k(bm25):
    recalls = [
        bm25.reports[k].overall[CreditMode.LENIENT].recall for k in sorted(bm25.reports)
    ]
    assert recalls == sorted(recalls)


def test_one_search_per_query_is_scored_at_every_k():
    """The top 10 is a prefix of the top 20, so re-querying per k would measure
    the same ranking twice — and let a non-deterministic retriever disagree
    with itself between two rows of one table."""
    calls: list[int] = []

    class Counting(Fixed):
        def search(self, query, k):
            calls.append(k)
            return super().search(query, k)

    measure("counting", Counting([("22", 3.0), ("24", 2.0)]), GOLD, INDEX)
    assert calls == [20, 20, 20]


def test_a_hit_at_rank_two_is_recorded_as_such():
    baseline = measure("fixed", Fixed([("23", 9.0), ("24", 4.0)]), GOLD, INDEX)
    outcome = next(o for o in baseline.outcomes if o.query_id == "q002")
    assert outcome.first_hit_rank == 2
    assert outcome.missed == ()
    assert outcome.top_score == 9.0


def test_a_missed_label_is_named():
    baseline = measure("fixed", Fixed([("23", 9.0)]), GOLD, INDEX)
    outcome = next(o for o in baseline.outcomes if o.query_id == "q002")
    assert outcome.first_hit_rank is None
    assert outcome.missed == ("24",)
    assert outcome.lenient_recall == 0.0


def test_an_ancestor_hit_is_lenient_only_and_is_not_a_failure():
    """ADR-060's whole point: `22` for a `22(2)` question is imprecise, not
    wrong, so it belongs in the imprecise table and not the failure one."""
    baseline = measure("fixed", Fixed([("22", 9.0)]), GOLD, INDEX)
    outcome = next(o for o in baseline.outcomes if o.query_id == "q001")
    assert outcome.strict_recall == 0.0
    assert outcome.lenient_recall == 1.0
    assert outcome.missed == ()
    assert outcome.query_id not in [f.query_id for f in baseline.failures]


def test_failures_are_the_lenient_misses(bm25):
    assert [f.query_id for f in bm25.failures] == [
        o.query_id for o in bm25.outcomes if o.lenient_recall < 1.0
    ]


def test_a_retriever_returning_nothing_scores_zero_without_raising():
    baseline = measure("empty", Fixed([]), GOLD, INDEX)
    assert baseline.primary.overall[CreditMode.LENIENT].recall == 0.0
    assert all(o.top_score is None for o in baseline.outcomes)
    assert baseline.negatives[0].top_score is None


def test_the_report_round_trips_through_json(bm25):
    report = BaselineReport(
        corpus_version=CORPUS_VERSION,
        chunk_count=len(CORPUS),
        gold_count=len(GOLD),
        k_values=(1, 5, 10, 20),
        primary_k=PRIMARY_K,
        retrievers=(bm25,),
    )
    assert BaselineReport.model_validate_json(report.model_dump_json()) == report


# --- the real corpus: the floors Phase 4 and Phase 5 are judged against ---


@pytest.fixture(scope="module")
def corpus_baselines(gold, chunks):
    index = CitationIndex(chunks)
    lexical = BM25Retriever(chunks)
    return (
        measure("bm25", lexical, gold, index),
        measure(
            "bm25+citation_shortcut",
            ShortcutRetriever(CitationRetriever(chunks), lexical),
            gold,
            index,
            ordinal_scores=True,
        ),
    )


def test_the_recorded_baseline_does_not_regress(corpus_baselines):
    """A floor, not a target: the numbers in reports/retrieval_baseline.md are
    0.453 strict and 0.602 lenient over the 64 answerable queries of the Step
    3.7 gold set. A retrieval change that drops below these should have to say
    so out loud.

    Raised at Step 4.4b (ADR-077), when b was tuned 0.75 -> 0.3 under two-fold
    cross-validation: bm25 alone went 0.438 / 0.461 -> 0.453 / 0.602.

    Re-derived at Step 1.10, not defended, exactly as they were at Step 3.7
    (where the v1 floors of 0.55 / 0.58 were dropped rather than argued with).
    Narrowing the trust rule split 19 roots into their real sub-structure, so
    the chunk set went 7,561 -> 8,351 and the ruler moved under the retriever:
    bm25 alone went 0.445 / 0.469 -> 0.438 / 0.461.

    The whole of that fall is one query, q064, losing half credit (0.5 / 64 is
    the entire delta). Its labels are `102(1)` / `195(1)`; the newly reachable
    definition `2(70)` "maximum marginal rate" -- the rate an unexplained cash
    credit is actually charged at -- now takes rank 1 and pushes `102(1)` to
    rank 11. The retriever surfaced an on-point unlabelled provision and was
    scored down for it, which is a gold-set limitation rather than a retrieval
    one. Re-labelling is deliberately its own later pass: editing the gold set
    here would move the ruler and the measured thing in the same change."""
    lexical = corpus_baselines[0].primary.overall
    assert lexical[CreditMode.STRICT].recall >= 0.43
    assert lexical[CreditMode.LENIENT].recall >= 0.58


def test_the_shortcut_never_makes_the_baseline_worse(corpus_baselines):
    lexical, composed = corpus_baselines
    for mode in CreditMode:
        assert (
            composed.primary.overall[mode].recall >= lexical.primary.overall[mode].recall
        )
        assert composed.primary.overall[mode].mrr >= lexical.primary.overall[mode].mrr


def test_the_shortcut_answers_the_citation_slice(corpus_baselines):
    """The slice it exists for: every citation query is answered leniently.
    This held over v1's 8 citation queries and still holds over v2's 20, none
    of which existed when the shortcut was built -- so it generalises rather
    than fitting the set it was measured on."""
    composed = corpus_baselines[1]
    per_slice = composed.primary.per_slice[QuerySlice.CITATION]
    assert per_slice[CreditMode.LENIENT].recall == 1.0


def test_the_lexical_score_does_not_separate_negatives(corpus_baselines):
    """Measured, not assumed: the best-scoring negative outscores the worst
    answerable query, so no single BM25 threshold can gate scope. Phase 12
    needs more than a lexical score, and this is the evidence for it."""
    lexical = corpus_baselines[0]
    answerable = [o.top_score for o in lexical.outcomes if o.top_score is not None]
    negatives = [n.top_score for n in lexical.negatives if n.top_score is not None]
    assert max(negatives) > min(answerable)
