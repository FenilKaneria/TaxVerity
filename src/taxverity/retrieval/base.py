"""Step 3.3 — the one abstraction introduced this early: what a retriever is,
and what it hands to the Step 3.2 metrics."""

from __future__ import annotations

import math
from collections.abc import Sequence
from itertools import pairwise
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, field_validator

from taxverity.chunking.models import Chunk


class ScoredChunk(BaseModel):
    model_config = ConfigDict(frozen=True)

    chunk: Chunk
    # Comparable only within one retriever's own results: BM25 is unbounded and
    # non-negative, cosine similarity is bounded and may be negative. Nothing
    # downstream may compare a score across retrievers or against a constant.
    score: float

    @field_validator("score")
    @classmethod
    def score_is_finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError(f"score must be finite, not {value}")
        return value


@runtime_checkable
class Retriever(Protocol):
    def search(self, query: str, k: int) -> Sequence[ScoredChunk]:
        """At most `k` results, best first, no chunk twice."""
        ...


def as_ranked_citations(results: Sequence[ScoredChunk]) -> list[str]:
    """The bridge to `score_run`, which takes citations rather than chunks.

    The ordering and duplicate checks live here because this is where a broken
    ranking stops being visible: the metrics rank by list position, so results
    out of score order would score a retriever on an order it did not return.
    """
    seen: set[str] = set()
    for previous, current in pairwise(results):
        if current.score > previous.score:
            raise ValueError(
                f"results are not ranked: {current.score} follows {previous.score}"
            )
    for result in results:
        if result.chunk.chunk_id in seen:
            raise ValueError(f"chunk {result.chunk.chunk_id} returned twice")
        seen.add(result.chunk.chunk_id)
    return [result.chunk.node_path for result in results]
