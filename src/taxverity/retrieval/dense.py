"""Step 4.5 — exact dense retrieval: brute-force cosine over the Step 4.4 vector
store, with the query embedded by the same hosted model that built it (ADR-075).

Two kinds of failure, deliberately handled differently (ADR-078). A store or an
embedder that disagrees with itself is our own misconfiguration, so it refuses
loudly at construction. The vendor going away, or drifting, is what the BM25 +
citation-shortcut fallback exists for, so it surfaces from `search()` as one
typed `DenseRetrievalError` that Step 5.2 catches.
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np

from taxverity.chunking.models import Chunk
from taxverity.embedding.backends import (
    Embedder,
    EmbedKind,
    ModelIdentityError,
    describe,
)
from taxverity.embedding.jina_api import EmbeddingAPIError
from taxverity.embedding.store import (
    PROBE_SET_VERSION,
    StaleVectorStoreError,
    VectorManifest,
    load_vector_store,
    verify_fingerprint,
)
from taxverity.observability import get_logger
from taxverity.retrieval.base import ScoredChunk

logger = get_logger(__name__)

DENSE_STAGE_VERSION = 1

# Rows are stored as float32, so a unit vector's norm lands within ~1e-7 of 1.
# The check exists because a dot product is the cosine only for unit rows; a
# store that is not normalised ranks by length, silently.
UNIT_NORM_TOLERANCE = 1e-4


class DenseRetrievalError(RuntimeError):
    """The dense path cannot answer this query. The caller falls back to BM25 +
    the citation shortcut; it is never told "no results" instead."""


class DenseRetriever:
    """Satisfies the Step 3.3 `Retriever` Protocol."""

    def __init__(
        self,
        chunks: Iterable[Chunk],
        vectors: np.ndarray,
        chunk_ids: Sequence[str],
        manifest: VectorManifest,
        embedder: Embedder,
    ) -> None:
        started = time.perf_counter()
        # A vector carries no evidence of how it was encoded (ADR-069).
        if manifest.kind != EmbedKind.DOCUMENT.value:
            raise StaleVectorStoreError(
                f"index was encoded as {manifest.kind!r}, not 'document' — "
                f"searching it with query vectors compares two different spaces."
            )
        if manifest.probe_set_version != PROBE_SET_VERSION:
            raise StaleVectorStoreError(
                f"index fingerprinted with probe set {manifest.probe_set_version}, "
                f"this code carries {PROBE_SET_VERSION} — rebuild the store."
            )
        served = embedder.info()
        if served != manifest.model:
            raise ModelIdentityError(
                f"embedder is {describe(served)}, index was built with "
                f"{describe(manifest.model)}"
            )

        matrix = np.asarray(vectors)
        if matrix.dtype != np.float32 or matrix.shape != (len(chunk_ids), manifest.dim):
            raise StaleVectorStoreError(
                f"vectors are {matrix.dtype} {matrix.shape}, expected float32 "
                f"({len(chunk_ids)}, {manifest.dim})"
            )
        norms = np.linalg.norm(matrix, axis=1)
        if not np.allclose(norms, 1.0, atol=UNIT_NORM_TOLERANCE):
            worst = int(np.argmax(np.abs(norms - 1.0)))
            raise StaleVectorStoreError(
                f"vector row {worst} has norm {norms[worst]:.6f}; the store must "
                f"hold unit vectors for a dot product to be the cosine."
            )

        # The join is by chunk id in both directions: a chunk without a vector is
        # unreachable by this retriever, and a vector without a chunk is a
        # result nothing can be delivered for. Either one is a stale store.
        self._chunks = tuple(chunks)
        row = {chunk_id: i for i, chunk_id in enumerate(chunk_ids)}
        if len(row) != len(chunk_ids):
            raise StaleVectorStoreError("the vector store lists a chunk id twice")
        chunk_ids_here = {chunk.chunk_id for chunk in self._chunks}
        unvectored = chunk_ids_here - row.keys()
        orphaned = row.keys() - chunk_ids_here
        if unvectored or orphaned or len(chunk_ids_here) != len(self._chunks):
            raise StaleVectorStoreError(
                f"chunk set and vector store disagree: {len(unvectored)} chunks have "
                f"no vector, {len(orphaned)} vectors have no chunk — rebuild the store."
            )
        # Rows are put in chunk order, so a tie breaks by corpus order exactly as
        # it does in BM25, whatever order the store happened to be written in.
        self._matrix = np.ascontiguousarray(
            matrix[[row[chunk.chunk_id] for chunk in self._chunks]]
        )
        self._manifest = manifest
        self._embedder = embedder
        self._verified = False
        self._refusal: str | None = None
        self.fingerprint_cosine: float | None = None
        logger.info(
            "dense index: %d vectors, dim %d, %s, in %.2fs",
            len(self._chunks),
            manifest.dim,
            describe(manifest.model),
            time.perf_counter() - started,
        )

    @classmethod
    def from_store(
        cls,
        directory: Path,
        chunks: Iterable[Chunk],
        embedder: Embedder,
        *,
        corpus_version: str,
    ) -> DenseRetriever:
        vectors, chunk_ids, manifest = load_vector_store(
            directory, corpus_version=corpus_version
        )
        return cls(chunks, vectors, chunk_ids, manifest, embedder)

    def __len__(self) -> int:
        return len(self._chunks)

    def search(self, query: str, k: int) -> Sequence[ScoredChunk]:
        if k < 1:
            raise ValueError(f"k must be at least 1, not {k}")
        self._ensure_verified()
        try:
            # QUERY is a literal here and the only query-side call site in the
            # system (ADR-069). It is never taken from an argument.
            (vector,) = self._embedder.embed([query], EmbedKind.QUERY)
        except EmbeddingAPIError as error:
            raise DenseRetrievalError(f"query embedding failed: {error}") from error
        return self.search_vector(vector, k)

    def search_vector(self, vector: Sequence[float], k: int) -> Sequence[ScoredChunk]:
        if k < 1:
            raise ValueError(f"k must be at least 1, not {k}")
        query = np.asarray(vector, dtype=np.float32)
        if query.shape != (self._manifest.dim,):
            raise ValueError(
                f"query vector has shape {query.shape}, index dim is {self._manifest.dim}"
            )
        scores = self._matrix @ query
        # A stable sort, so equal scores keep corpus order and a run reproduces.
        order = np.argsort(-scores, kind="stable")[:k]
        return [
            ScoredChunk(chunk=self._chunks[i], score=float(scores[i])) for i in order
        ]

    def _ensure_verified(self) -> None:
        # Deferred to the first search rather than done at construction: the
        # check is a vendor call, and a vendor outage at startup must degrade
        # like one at query time, not stop the process.
        if self._refusal is not None:
            raise DenseRetrievalError(self._refusal)
        if self._verified:
            return
        try:
            self.fingerprint_cosine = verify_fingerprint(self._embedder, self._manifest)
        except ModelIdentityError as error:
            # Sticky: the same upstream model will fail the same probes on every
            # query, and re-probing would spend tokens to learn nothing.
            self._refusal = f"dense index refused, the embedder has drifted: {error}"
            logger.warning("%s", self._refusal)
            raise DenseRetrievalError(self._refusal) from error
        except EmbeddingAPIError as error:
            # Not sticky: an outage ends, and the next query tries again.
            raise DenseRetrievalError(f"fingerprint check could not run: {error}") from error
        self._verified = True
