"""Step 3.3 — the `Retriever` Protocol and the bridge into the Step 3.2 ruler."""

from __future__ import annotations

import math

import pytest

from taxverity.chunking.models import Chunk
from taxverity.corpus.nodes import NodeType
from taxverity.evals.gold import GoldQuery, QuerySlice
from taxverity.evals.metrics import CitationIndex, score_query
from taxverity.retrieval.base import Retriever, ScoredChunk, as_ranked_citations

CORPUS_VERSION = "v" * 64


def chunk(node_path="22(2)", text="whatever the clause says"):
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


class FakeRetriever:
    def __init__(self, results):
        self._results = results

    def search(self, query: str, k: int):
        return self._results[:k]


def test_a_plain_object_with_search_satisfies_the_protocol():
    """The point of the Protocol: every later component substitutes in without
    inheriting from anything."""
    assert isinstance(FakeRetriever([]), Retriever)


def test_an_object_without_search_does_not():
    assert not isinstance(object(), Retriever)


def test_a_score_must_be_finite():
    for bad in (math.nan, math.inf, -math.inf):
        with pytest.raises(ValueError, match="finite"):
            ScoredChunk(chunk=chunk(), score=bad)


def test_a_negative_score_is_allowed():
    """Cosine similarity is signed. Refusing a negative score here would rule
    out the Phase 4 dense retriever before it is written."""
    assert ScoredChunk(chunk=chunk(), score=-0.4).score == -0.4


def test_a_scored_chunk_is_frozen():
    result = ScoredChunk(chunk=chunk(), score=1.0)
    with pytest.raises(ValueError):
        result.score = 2.0  # type: ignore[misc]


def test_ranked_citations_are_the_node_paths_in_order():
    results = [
        ScoredChunk(chunk=chunk("22(2)"), score=9.0),
        ScoredChunk(chunk=chunk("23"), score=4.5),
        ScoredChunk(chunk=chunk("24(1)"), score=4.5),
    ]
    assert as_ranked_citations(results) == ["22(2)", "23", "24(1)"]


def test_an_empty_result_set_is_allowed():
    assert as_ranked_citations([]) == []


def test_results_out_of_score_order_are_refused():
    """The metrics rank by list position, so an unsorted result set would be
    scored on an order the retriever never returned — silently."""
    results = [
        ScoredChunk(chunk=chunk("22(2)"), score=1.0),
        ScoredChunk(chunk=chunk("23"), score=7.0),
    ]
    with pytest.raises(ValueError, match="not ranked"):
        as_ranked_citations(results)


def test_the_same_chunk_returned_twice_is_refused():
    duplicate = chunk("22(2)")
    results = [
        ScoredChunk(chunk=duplicate, score=9.0),
        ScoredChunk(chunk=duplicate, score=8.0),
    ]
    with pytest.raises(ValueError, match="twice"):
        as_ranked_citations(results)


def test_search_results_flow_into_the_metrics():
    """The whole reason this step exists between 3.2 and 3.4: one call chain
    from a retriever to a number."""
    retriever = FakeRetriever(
        [
            ScoredChunk(chunk=chunk("23"), score=9.0),
            ScoredChunk(chunk=chunk("22(2)"), score=8.0),
        ]
    )
    index = CitationIndex([chunk("22(2)"), chunk("23")])
    query = GoldQuery(
        query_id="q001",
        slice=QuerySlice.PARAPHRASE,
        question="what does it say",
        required=("22(2)",),
        notes="fixture",
    )
    citations = as_ranked_citations(retriever.search(query.question, 10))
    scored = score_query(query, citations, 10, index)
    assert scored.strict.recall == 1.0
    assert scored.strict.mrr == 0.5
    assert scored.lenient.mrr == 0.5


def test_k_bounds_what_search_returns():
    retriever = FakeRetriever(
        [ScoredChunk(chunk=chunk(f"{n}"), score=10.0 - n) for n in range(1, 6)]
    )
    assert len(retriever.search("anything", 3)) == 3


def test_the_index_resolves_a_citation_a_retriever_returned():
    index = CitationIndex([chunk("22(2)")])
    assert index.resolve("22(2)") == chunk("22(2)").chunk_id
