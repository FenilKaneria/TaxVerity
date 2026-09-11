"""Step 5.1 — RRF arithmetic on synthetic rankings (ADR-010, ADR-080)."""

from __future__ import annotations

import pytest

from taxverity.chunking.models import Chunk
from taxverity.corpus.nodes import NodeType
from taxverity.retrieval.base import Retriever, ScoredChunk, as_ranked_citations
from taxverity.retrieval.citations import CitationRetriever, ShortcutRetriever
from taxverity.retrieval.dense import DenseRetrievalError
from taxverity.retrieval.fusion import (
    RRF_K,
    FusionRetriever,
    reciprocal_rank_fusion,
)

CORPUS_VERSION = "v" * 64


def chunk(node_path):
    return Chunk.create(
        CORPUS_VERSION,
        node_path,
        f"text of {node_path}",
        parent_id=None,
        doc_id="income-tax-act-2025",
        node_type=NodeType.SECTION,
        section_number=node_path,
        page_start=1,
        page_end=1,
    )


A, B, C, D, E = (chunk(str(n)) for n in (21, 22, 23, 24, 25))


def ranking(*chunks, scores=None):
    scores = scores or [float(len(chunks) - i) for i in range(len(chunks))]
    return [ScoredChunk(chunk=c, score=s) for c, s in zip(chunks, scores, strict=True)]


def paths(results):
    return [result.chunk.node_path for result in results]


class FakeRetriever:
    def __init__(self, results):
        self._results = results
        self.asked: list[int] = []

    def search(self, query: str, k: int):
        self.asked.append(k)
        return self._results[:k]


class FailingRetriever:
    def search(self, query: str, k: int):
        raise DenseRetrievalError("vendor down")


# --- arithmetic ----------------------------------------------------------


def test_each_chunk_scores_the_sum_of_its_reciprocal_ranks():
    fused = reciprocal_rank_fusion([ranking(A, B, C), ranking(B, D)])
    scores = {r.chunk.node_path: r.score for r in fused}
    assert scores == {
        "21": 1 / 61,
        "22": pytest.approx(1 / 62 + 1 / 61),
        "23": 1 / 63,
        "24": 1 / 62,
    }
    assert paths(fused) == ["22", "21", "24", "23"]


def test_agreement_beats_a_single_top_rank():
    """What fusion is for: second in both outranks first in one."""
    fused = reciprocal_rank_fusion([ranking(A, C), ranking(B, C)])
    assert paths(fused)[0] == "23"


def test_the_constant_is_the_one_adr_010_fixed():
    assert RRF_K == 60


def test_scores_are_never_read_only_positions():
    """The reason for RRF over score fusion: a BM25 score and a cosine live on
    different scales. Rescaling one input, even into negatives, changes nothing."""
    plain = reciprocal_rank_fusion([ranking(A, B, C), ranking(C, D)])
    rescaled = reciprocal_rank_fusion(
        [ranking(A, B, C, scores=[900.0, 5.0, 0.1]), ranking(C, D, scores=[-0.2, -0.9])]
    )
    assert [(r.chunk.chunk_id, r.score) for r in plain] == [
        (r.chunk.chunk_id, r.score) for r in rescaled
    ]


def test_the_output_is_a_valid_ranking():
    fused = reciprocal_rank_fusion([ranking(A, B, C, D), ranking(D, E, A)])
    assert as_ranked_citations(fused) == paths(fused)


# --- ties ----------------------------------------------------------------


def test_an_exact_tie_goes_to_the_earlier_ranking():
    assert paths(reciprocal_rank_fusion([ranking(A), ranking(B)])) == ["21", "22"]
    assert paths(reciprocal_rank_fusion([ranking(B), ranking(A)])) == ["22", "21"]


def test_a_tie_on_score_goes_to_the_better_single_rank_first():
    """With rrf_k=0, C at ranks 2 and 2 scores 1/2 + 1/2 = 1, exactly what A and
    B score at rank 1 alone. The two rank-1 chunks lead; C, never better than
    second, comes last."""
    fused = reciprocal_rank_fusion([ranking(A, C), ranking(B, C)], rrf_k=0)
    assert [r.score for r in fused] == [1.0, 1.0, 1.0]
    assert paths(fused) == ["21", "22", "23"]


def test_equal_rank_sets_tie_exactly_whatever_the_summation_order():
    """A holds ranks 1, 5, 9 and B holds 9, 1, 5: the same set, summed in a
    different order. Plain float addition is not associative; fsum is exactly
    rounded, so the two tie exactly and the earlier ranking breaks it."""
    filler = iter([chunk(str(n)) for n in range(100, 121)])

    def place(at):
        return ranking(*(at.get(rank) or next(filler) for rank in range(1, 10)))

    fused = reciprocal_rank_fusion(
        [place({1: A, 9: B}), place({1: B, 5: A}), place({5: B, 9: A})]
    )
    scores = {r.chunk.node_path: r.score for r in fused}
    assert scores["21"] == scores["22"]
    assert paths(fused)[:2] == ["21", "22"]


# --- edge cases ----------------------------------------------------------


def test_no_rankings_fuse_to_nothing():
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []


def test_an_empty_ranking_leaves_the_other_order_intact():
    assert paths(reciprocal_rank_fusion([[], ranking(C, A, B)])) == ["23", "21", "22"]


def test_a_chunk_twice_in_one_ranking_is_refused():
    with pytest.raises(ValueError, match="twice"):
        reciprocal_rank_fusion([ranking(A, B, A)])


def test_a_negative_constant_is_refused():
    with pytest.raises(ValueError, match="rrf_k"):
        reciprocal_rank_fusion([ranking(A)], rrf_k=-1)


# --- the retriever -------------------------------------------------------


def test_the_fusion_retriever_satisfies_the_protocol():
    assert isinstance(FusionRetriever([FakeRetriever([])]), Retriever)


def test_each_input_is_read_to_depth_not_to_k():
    first, second = FakeRetriever(ranking(A, B)), FakeRetriever(ranking(C))
    FusionRetriever([first, second], depth=50).search("q", 10)
    assert first.asked == second.asked == [50]


def test_depth_never_falls_below_k():
    inner = FakeRetriever(ranking(A, B, C))
    FusionRetriever([inner], depth=2).search("q", 3)
    assert inner.asked == [3]


def test_a_chunk_outside_both_top_k_can_win():
    """Why depth exists: C is in neither input's top 1, yet leads the fusion."""
    fused = FusionRetriever(
        [FakeRetriever(ranking(A, C)), FakeRetriever(ranking(B, C))]
    ).search("q", 1)
    assert paths(fused) == ["23"]


def test_k_bounds_the_fused_result():
    fused = FusionRetriever(
        [FakeRetriever(ranking(A, B, C)), FakeRetriever(ranking(D, E))]
    ).search("q", 2)
    assert len(fused) == 2


@pytest.mark.parametrize("bad", [0, -1])
def test_k_and_depth_below_one_are_refused(bad):
    with pytest.raises(ValueError, match="k must be"):
        FusionRetriever([FakeRetriever([])]).search("q", bad)
    with pytest.raises(ValueError, match="depth"):
        FusionRetriever([FakeRetriever([])], depth=bad)


def test_a_failing_leg_propagates_rather_than_fusing_one_leg():
    """Fallback is Step 5.2's decision; fusion must not hide the failure."""
    retriever = FusionRetriever([FailingRetriever(), FakeRetriever(ranking(A))])
    with pytest.raises(DenseRetrievalError):
        retriever.search("q", 5)


def test_the_citation_shortcut_stays_ahead_of_the_fusion():
    """ADR-080: the shortcut is positional, outside the fusion, so a typed
    citation keeps rank 1 even when neither fused leg ranks it at all."""
    fused = FusionRetriever(
        [FakeRetriever(ranking(A, B)), FakeRetriever(ranking(B, C))]
    )
    composed = ShortcutRetriever(CitationRetriever([A, B, C, E]), fused)
    assert paths(composed.search("what does section 25 say?", 3)) == ["25", "22", "21"]
