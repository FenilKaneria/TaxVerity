"""Step 5.2 — the dense leg's degradation, the hybrid adoption rule, and the
cached gold-question vectors (ADR-081)."""

from __future__ import annotations

import logging

import pytest

from taxverity.embedding.backends import StubEmbedder
from taxverity.embedding.store import StaleVectorStoreError
from taxverity.evals.baseline import measure
from taxverity.evals.gold import QuerySlice
from taxverity.evals.hybrid import judge_hybrid
from taxverity.evals.metrics import CreditMode, RunReport, Scores
from taxverity.evals.query_vectors import (
    CachedQueryRetriever,
    QueryVectors,
    load_query_vectors,
    write_query_vectors,
)
from taxverity.retrieval.base import Retriever, ScoredChunk
from taxverity.retrieval.bm25 import BM25Retriever
from taxverity.retrieval.citations import CitationRetriever, ShortcutRetriever
from taxverity.retrieval.dense import DenseRetrievalError
from taxverity.retrieval.fallback import FallbackRetriever
from taxverity.retrieval.fusion import FusionRetriever
from test_dense import CORPUS, VECTORS, Counting, Failing
from test_dense import retriever as dense_retriever

LOGGER = "taxverity.retrieval.fallback"
PAN = "ABCDE1234F"


class CaptureHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


@pytest.fixture
def captured():
    # On the module's own logger, below the `propagate = False` root and below
    # RedactingFilter: the assertion is about what the call site passed.
    handler = CaptureHandler()
    logger = logging.getLogger(LOGGER)
    previous = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    yield handler
    logger.removeHandler(handler)
    logger.setLevel(previous)


class Scripted:
    def __init__(self, *chunks) -> None:
        self._results = [
            ScoredChunk(chunk=c, score=float(len(chunks) - i)) for i, c in enumerate(chunks)
        ]

    def search(self, query: str, k: int):
        return self._results[:k]


class Raising:
    def __init__(self, error: Exception) -> None:
        self._error = error

    def search(self, query: str, k: int):
        raise self._error


def paths(results):
    return [r.chunk.node_path for r in results]


# --- fallback --------------------------------------------------------------------


def test_the_fallback_retriever_satisfies_the_protocol():
    assert isinstance(FallbackRetriever(Scripted(), Scripted()), Retriever)


def test_a_working_primary_answers_and_nothing_is_logged(captured):
    results = FallbackRetriever(Scripted(CORPUS[0]), Scripted(CORPUS[1])).search("q", 5)
    assert paths(results) == ["21"]
    assert captured.records == []


def test_a_dense_failure_degrades_to_the_fallback_with_a_warning(captured):
    retriever = FallbackRetriever(
        Raising(DenseRetrievalError("query embedding failed: 503")), Scripted(CORPUS[1])
    )
    assert paths(retriever.search(f"my PAN is {PAN}", 5)) == ["22"]
    (record,) = captured.records
    assert record.levelno == logging.WARNING
    assert "503" in record.getMessage()
    assert PAN not in record.getMessage(), "the query is user input and is never logged"


@pytest.mark.parametrize("error", [RuntimeError("bug"), ValueError("bug"), KeyError("bug")])
def test_any_other_error_propagates_rather_than_degrading(error):
    """Degrading on a bug would hide it behind a working-looking answer."""
    with pytest.raises(type(error)):
        FallbackRetriever(Raising(error), Scripted(CORPUS[0])).search("q", 5)


def test_k_below_one_is_refused():
    with pytest.raises(ValueError, match="k must be"):
        FallbackRetriever(Scripted(), Scripted()).search("q", 0)


def test_a_forced_api_failure_yields_the_bm25_and_shortcut_ordering(captured):
    """The required test (ADR-075): a real DenseRetriever whose vendor calls all
    fail, inside the hybrid, answers exactly as BM25 + the shortcut would, and
    the query does not fail."""
    dense = dense_retriever(Failing(failures=100))
    bm25 = BM25Retriever(CORPUS)
    shortcut = CitationRetriever(CORPUS)
    composed = ShortcutRetriever(
        shortcut, FallbackRetriever(FusionRetriever([dense, bm25]), bm25)
    )
    expected = ShortcutRetriever(shortcut, bm25)
    for query in ("what does section 23 say about b?", "a d"):
        assert paths(composed.search(query, 4)) == paths(expected.search(query, 4))
    assert [r.levelno for r in captured.records] == [logging.WARNING] * 2


# --- the adoption rule -------------------------------------------------------------


def scores(recall: float, ndcg: float = 0.0) -> dict[CreditMode, Scores]:
    return {mode: Scores(recall=recall, mrr=0.0, ndcg=ndcg) for mode in CreditMode}


def report(recall, ndcg, *, paraphrase=None, crossref=None, k=10) -> RunReport:
    return RunReport(
        k=k,
        scored=(),
        overall=scores(recall, ndcg),
        per_slice={
            QuerySlice.CITATION: scores(1.0),
            QuerySlice.PARAPHRASE: scores(recall if paraphrase is None else paraphrase),
            QuerySlice.CROSSREF: scores(recall if crossref is None else crossref),
        },
        negatives=0,
    )


BM25 = report(0.633, 0.547)
DENSE = report(0.758, 0.673)


def test_a_hybrid_rising_overall_and_holding_every_slice_is_adopted():
    verdict = judge_hybrid(report(0.80, 0.70), {"bm25": BM25, "dense": DENSE})
    assert verdict.adopted and verdict.reasons == ()


def test_the_citation_ceiling_does_not_block_adoption():
    """Both incumbents already score 1.000 there; a tie must not read as a loss."""
    verdict = judge_hybrid(report(0.80, 0.70), {"dense": DENSE})
    assert verdict.adopted


def test_beating_one_incumbent_is_not_enough():
    verdict = judge_hybrid(report(0.70, 0.60), {"bm25": BM25, "dense": DENSE})
    assert not verdict.adopted
    assert all(reason.startswith("vs dense:") for reason in verdict.reasons)


def test_any_slice_falling_rejects_even_with_an_overall_rise():
    """ADR-010's per-slice clause, with no one-query tolerance: an aggregate that
    hides "hybrid is worse on crossref" is the lie the rule exists to catch."""
    verdict = judge_hybrid(
        report(0.80, 0.70, crossref=0.75), {"dense": report(0.758, 0.673, crossref=0.76)}
    )
    assert not verdict.adopted
    assert verdict.reasons == ("vs dense: crossref slice lenient recall fell 0.760 -> 0.750",)


def test_a_recall_rise_bought_with_an_ndcg_fall_is_rejected():
    verdict = judge_hybrid(report(0.80, 0.60), {"dense": DENSE})
    assert verdict.reasons == ("vs dense: lenient nDCG fell 0.673 -> 0.600",)


def test_matching_an_incumbent_exactly_is_not_a_win():
    verdict = judge_hybrid(report(0.758, 0.673), {"dense": DENSE})
    assert verdict.reasons == ("vs dense: neither lenient recall nor lenient nDCG rose",)


def test_no_incumbent_or_a_different_k_is_refused():
    with pytest.raises(ValueError, match="at least one"):
        judge_hybrid(DENSE, {})
    with pytest.raises(ValueError, match="k=20"):
        judge_hybrid(report(0.8, 0.7, k=20), {"dense": DENSE})


# --- cached question vectors -------------------------------------------------------

MODEL = StubEmbedder().info()


def cache(**vectors) -> QueryVectors:
    return QueryVectors(model=MODEL, fingerprint_cosine=0.99996, vectors=vectors)


def test_cached_vectors_round_trip_byte_identically(tmp_path):
    original = cache(rent=(0.6, 0.8), salary=(1.0, 0.0))
    first, second = tmp_path / "a.json", tmp_path / "b.json"
    write_query_vectors(first, original)
    write_query_vectors(second, load_query_vectors(first, model=MODEL, questions=["rent"]))
    assert load_query_vectors(first, model=MODEL, questions=["rent", "salary"]) == original
    assert first.read_bytes() == second.read_bytes()


def test_vectors_from_another_model_are_refused(tmp_path):
    path = tmp_path / "q.json"
    write_query_vectors(path, cache(rent=(1.0,)))
    other = MODEL.model_copy(update={"encoding": "another-recipe"})
    with pytest.raises(StaleVectorStoreError, match="index was built with"):
        load_query_vectors(path, model=other, questions=["rent"])


def test_a_question_without_a_vector_is_refused(tmp_path):
    """An edited gold question lands here, rather than reusing its old wording's
    vector."""
    path = tmp_path / "q.json"
    write_query_vectors(path, cache(rent=(1.0,)))
    with pytest.raises(StaleVectorStoreError, match="no vector for 1"):
        load_query_vectors(path, model=MODEL, questions=["rent", "rent paid to my mother"])


def test_the_cached_retriever_searches_the_stored_vector_and_never_embeds():
    embedder = Counting()
    retriever = CachedQueryRetriever(
        dense_retriever(embedder), {"rent": tuple(float(v) for v in VECTORS[0])}
    )
    assert paths(retriever.search("rent", 2)) == ["21", "24"]
    assert embedder.calls == []
    with pytest.raises(KeyError):
        retriever.search("a question never embedded", 2)


# --- corpus: the real store and the cached question vectors, no network ----------
# `retrieval_legs` lives in conftest.py, shared with the Step 5.3 suite.


@pytest.fixture(scope="module")
def measured(gold, retrieval_legs):
    index, dense, bm25, shortcut = retrieval_legs
    legs = (("bm25", bm25), ("dense", dense), ("hybrid", FusionRetriever([dense, bm25])))
    return {
        name: measure(
            name, ShortcutRetriever(shortcut, leg), gold, index, ordinal_scores=True
        ).primary
        for name, leg in legs
    }


def test_the_dense_and_hybrid_floors_hold(measured):
    """Floors, not targets. reports/hybrid_measurement.md measured lenient
    recall@10 of 0.758 for dense + shortcut and 0.797 / nDCG 0.700 for hybrid +
    shortcut over the 64 answerable gold v2 queries. The first dense floor in
    the suite, which ADR-079 deferred until the question vectors were cached."""
    assert measured["dense"].overall[CreditMode.LENIENT].recall >= 0.73
    assert measured["hybrid"].overall[CreditMode.LENIENT].recall >= 0.77
    assert measured["hybrid"].overall[CreditMode.LENIENT].ndcg >= 0.67


def test_the_adoption_verdict_still_holds(measured):
    """A retrieval change that erodes the case for hybrid must say so out loud,
    not leave ADR-081 standing on numbers that no longer hold."""
    verdict = judge_hybrid(
        measured["hybrid"],
        {"bm25+shortcut": measured["bm25"], "dense+shortcut": measured["dense"]},
    )
    assert verdict.adopted, verdict.reasons


def test_a_dead_dense_leg_reproduces_bm25_and_shortcut_on_every_gold_question(
    gold, retrieval_legs, captured
):
    _, _, bm25, shortcut = retrieval_legs
    down = Raising(DenseRetrievalError("query embedding failed: 503"))
    composed = ShortcutRetriever(
        shortcut, FallbackRetriever(FusionRetriever([down, bm25]), bm25)
    )
    expected = ShortcutRetriever(shortcut, bm25)
    for query in gold:
        assert paths(composed.search(query.question, 10)) == paths(
            expected.search(query.question, 10)
        )
    assert len(captured.records) == len(gold)
