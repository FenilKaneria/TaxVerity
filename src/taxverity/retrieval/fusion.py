"""Reciprocal Rank Fusion: several rankings combined by position alone, so a BM25 score
and a cosine are never compared."""

from __future__ import annotations

import math
from collections.abc import Sequence

from taxverity.chunking.models import Chunk
from taxverity.retrieval.base import Retriever, ScoredChunk

FUSION_STAGE_VERSION = 1

RRF_K = 60
FUSION_DEPTH = 100


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[ScoredChunk]], rrf_k: int = RRF_K
) -> list[ScoredChunk]:
    """Each chunk scores the sum of 1 / (rrf_k + rank) over the rankings that hold it."""
    if rrf_k < 0:
        raise ValueError(f"rrf_k must be non-negative, not {rrf_k}")
    parts: dict[str, list[float]] = {}
    best: dict[str, tuple[int, int]] = {}
    chunks: dict[str, Chunk] = {}
    for source, ranking in enumerate(rankings):
        seen: set[str] = set()
        for rank, result in enumerate(ranking, start=1):
            chunk_id = result.chunk.chunk_id
            if chunk_id in seen:
                raise ValueError(f"ranking {source} holds chunk {chunk_id} twice")
            seen.add(chunk_id)
            parts.setdefault(chunk_id, []).append(1.0 / (rrf_k + rank))
            best[chunk_id] = min(best.get(chunk_id, (rank, source)), (rank, source))
            chunks.setdefault(chunk_id, result.chunk)
    # fsum is exactly rounded, so equal rank sets tie exactly whatever order the
    # rankings were summed in. A running `+` differs in the last bit; the builtin
    # sum is compensated since 3.12 but not guaranteed exact.
    scores = {chunk_id: math.fsum(values) for chunk_id, values in parts.items()}
    order = sorted(scores, key=lambda chunk_id: (-scores[chunk_id], best[chunk_id]))
    return [ScoredChunk(chunk=chunks[i], score=scores[i]) for i in order]


class FusionRetriever:
    def __init__(
        self,
        retrievers: Sequence[Retriever],
        *,
        depth: int = FUSION_DEPTH,
        rrf_k: int = RRF_K,
    ) -> None:
        if depth < 1:
            raise ValueError(f"depth must be at least 1, not {depth}")
        self._retrievers = tuple(retrievers)
        self._depth = depth
        self._rrf_k = rrf_k

    def search(self, query: str, k: int) -> Sequence[ScoredChunk]:
        if k < 1:
            raise ValueError(f"k must be at least 1, not {k}")
        depth = max(self._depth, k)
        rankings = [retriever.search(query, depth) for retriever in self._retrievers]
        return reciprocal_rank_fusion(rankings, self._rrf_k)[:k]
