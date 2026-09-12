"""Step 5.2 — gold-question vectors, embedded once through the API and kept, so
dense and hybrid runs can be re-measured and floor-tested without a network
call. Deferred here from Step 4.6 (ADR-079), where it had no second consumer."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, ConfigDict

from taxverity.embedding.backends import ModelInfo, describe
from taxverity.embedding.store import StaleVectorStoreError
from taxverity.retrieval.base import ScoredChunk

QUERY_VECTORS_FILENAME = "gold_query_vectors.json"


class QueryVectors(BaseModel):
    model_config = ConfigDict(frozen=True)

    model: ModelInfo
    # The lowest probe cosine when these were embedded: the evidence that they
    # came from the model the corpus index was built with.
    fingerprint_cosine: float
    # Keyed by the exact question text, so an edited gold question misses
    # rather than silently reusing the vector of its old wording.
    vectors: dict[str, tuple[float, ...]]


def write_query_vectors(path: Path, vectors: QueryVectors) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(
        vectors.model_dump(mode="json"),
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(text + "\n")


def load_query_vectors(
    path: Path, *, model: ModelInfo, questions: Sequence[str]
) -> QueryVectors:
    """Refuses vectors from another model, or missing any question asked for."""
    loaded = QueryVectors.model_validate_json(path.read_text(encoding="utf-8"))
    if loaded.model != model:
        raise StaleVectorStoreError(
            f"{path} holds vectors from {describe(loaded.model)}, the index was "
            f"built with {describe(model)}"
        )
    missing = [q for q in questions if q not in loaded.vectors]
    if missing:
        raise StaleVectorStoreError(
            f"{path} has no vector for {len(missing)} question(s), first {missing[0]!r}"
        )
    return loaded


class VectorIndex(Protocol):
    """Both dense indexes, the NumPy one and Step 6.4's Postgres one, so a
    cached-vector run can be pointed at either (Step 6.5)."""

    def search_vector(self, vector: Sequence[float], k: int) -> Sequence[ScoredChunk]: ...


class CachedQueryRetriever:
    """A dense index searched with stored question vectors. A question with no
    vector raises KeyError; it never reaches the network."""

    def __init__(self, dense: VectorIndex, vectors: Mapping[str, Sequence[float]]) -> None:
        self._dense = dense
        self._vectors = vectors

    def search(self, query: str, k: int) -> Sequence[ScoredChunk]:
        return self._dense.search_vector(self._vectors[query], k)
