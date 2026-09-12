"""Step 6.4 — exact dense search served from Postgres (ADR-090).

Two pure tests cover the row mapping; the rest run against a throwaway database
loaded by the Step 6.3 ingest, because what is being tested is the SQL — the
ordering, the tie-break and the join — and a fake cannot fail the way Postgres
can.
"""

from __future__ import annotations

import hashlib
import math
import struct

import numpy as np
import pytest

from conftest import MICRO_DIM as DIM
from conftest import MICRO_MODEL as MODEL
from conftest import MICRO_V1 as V1
from conftest import micro_chunk_manifest, micro_chunks
from taxverity.chunking.models import Chunk
from taxverity.db.ingest import ingest_chunks, ingest_vectors
from taxverity.embedding.backends import (
    EmbedKind,
    ModelIdentityError,
    ModelInfo,
)
from taxverity.embedding.jina_api import EmbeddingAPIError
from taxverity.embedding.store import (
    PROBE_SET_VERSION,
    StaleVectorStoreError,
    VectorManifest,
    embed_probes,
)
from taxverity.retrieval.base import Retriever
from taxverity.retrieval.dense import DenseRetrievalError, DenseRetriever
from taxverity.retrieval.pgvector import (
    PgVectorIndex,
    chunk_from_row,
    manifest_from_row,
)


class FakeEmbedder:
    """Deterministic 1024-wide unit vectors from a text hash. Carries no
    semantics: it exists so the fingerprint and identity checks have something
    to agree with, not to rank anything."""

    def __init__(self, model: ModelInfo = MODEL) -> None:
        self._model = model
        self.calls: list[tuple[EmbedKind, int]] = []
        self.fail: Exception | None = None
        self.drift = False

    def info(self) -> ModelInfo:
        return self._model

    def embed(self, texts, kind: EmbedKind) -> list[list[float]]:
        self.calls.append((kind, len(texts)))
        if self.fail is not None:
            raise self.fail
        if self.drift:
            return [basis(1).tolist() for _ in texts]
        return [_hash_vector(f"{kind.value}\x00{text}") for text in texts]


def _hash_vector(seed: str) -> list[float]:
    raw = b""
    counter = 0
    while len(raw) < DIM * 4:
        raw += hashlib.sha256(seed.encode() + counter.to_bytes(4, "big")).digest()
        counter += 1
    values = [
        struct.unpack(">I", raw[i * 4 : i * 4 + 4])[0] / 2**31 - 1.0 for i in range(DIM)
    ]
    norm = math.sqrt(sum(v * v for v in values))
    return [v / norm for v in values]


def basis(*weights: float) -> np.ndarray:
    row = np.zeros(DIM, dtype=np.float64)
    row[: len(weights)] = weights
    return (row / np.linalg.norm(row)).astype(np.float32)


# Along e0 the order is known by construction: the root at cosine 1, the
# schedule at cos 45°, and the two subsections tied at 0 — which is what the
# ordinal tie-break has to resolve.
GEOMETRY = np.stack([basis(1), basis(0, 1), basis(0, 0, 1), basis(1, 1)])


def vector_manifest(chunks, embedder=None, **overrides) -> VectorManifest:
    embedder = embedder or FakeEmbedder()
    fields = {
        "store_version": 2,
        "doc_id": "test-act",
        "corpus_version": V1,
        "model": embedder.info(),
        "kind": "document",
        "device": "test",
        "dim": DIM,
        "chunk_count": len(chunks),
        "vectors_sha256": "2" * 64,
        "ids_sha256": "3" * 64,
        "probe_set_version": PROBE_SET_VERSION,
        "probes": embed_probes(embedder),
    }
    return VectorManifest(**(fields | overrides))


def loaded(conn, vectors=GEOMETRY, embedder=None, **manifest_overrides):
    """Ingest the micro corpus and one embedding set, and return both."""
    chunks = micro_chunks(V1)
    ingest_chunks(conn, chunks, micro_chunk_manifest(chunks))
    manifest = vector_manifest(chunks, embedder=embedder, **manifest_overrides)
    result = ingest_vectors(
        conn, vectors, [chunk.chunk_id for chunk in chunks], manifest
    )
    return chunks, result.embedding_set_id


def row_of(chunk: Chunk) -> dict:
    return {
        "chunk_id": chunk.chunk_id,
        "parent_id": chunk.parent_id,
        "doc_id": chunk.doc_id,
        "corpus_version": chunk.corpus_version,
        "node_type": chunk.node_type.value,
        "node_path": chunk.node_path,
        "section_number": chunk.section_number,
        "schedule_number": chunk.schedule_number,
        "root_title": chunk.root_title,
        "chapter_numeral": chunk.chapter_numeral,
        "chapter_title": chunk.chapter_title,
        "text": chunk.text,
        "page_start": chunk.page_start,
        "page_end": chunk.page_end,
        "char_start": chunk.char_start,
        "char_end": chunk.char_end,
        "defined_terms": list(chunk.defined_terms),
        "outgoing_refs": list(chunk.outgoing_refs),
        "token_count": chunk.token_count,
    }


def test_a_row_rebuilds_the_chunk_it_was_ingested_from():
    for chunk in micro_chunks(V1):
        assert chunk_from_row(row_of(chunk)) == chunk


def test_an_edited_row_refuses_itself():
    row = row_of(micro_chunks(V1)[0])
    row["text"] = row["text"].replace("Test Act", "Other Act")
    with pytest.raises(ValueError):
        chunk_from_row(row)


def test_a_set_row_rebuilds_the_vector_manifest():
    manifest = vector_manifest(micro_chunks(V1))
    row = {
        "corpus_version": V1,
        "doc_id": "test-act",
        "model_id": MODEL.model_id,
        "dim": DIM,
        "revision": MODEL.revision,
        "runtime": MODEL.runtime,
        "encoding": MODEL.encoding,
        "kind": "document",
        "device": "test",
        "chunk_count": 4,
        "vectors_sha256": "2" * 64,
        "ids_sha256": "3" * 64,
        "probe_set_version": PROBE_SET_VERSION,
        "probes": [probe.model_dump(mode="json") for probe in manifest.probes],
    }
    rebuilt = manifest_from_row(row)
    assert rebuilt.model == MODEL
    assert rebuilt.corpus_version == V1
    assert rebuilt.probes == manifest.probes


def test_it_satisfies_the_retriever_protocol(schema):
    _, set_id = loaded(schema)
    assert isinstance(PgVectorIndex(schema, set_id, FakeEmbedder()), Retriever)


def test_search_ranks_by_cosine_and_breaks_ties_by_corpus_order(schema):
    chunks, set_id = loaded(schema)
    index = PgVectorIndex(schema, set_id, FakeEmbedder())

    results = index.search_vector(basis(1), 4)

    assert [result.chunk.node_path for result in results] == [
        "1",
        "Schedule I",
        "1(1)",
        "1(2)",
    ]
    assert [result.score for result in results] == pytest.approx(
        [1.0, math.sqrt(0.5), 0.0, 0.0], abs=1e-6
    )
    assert len(index) == len(chunks)


def test_it_agrees_with_the_numpy_index(schema):
    chunks, set_id = loaded(schema)
    manifest = vector_manifest(chunks)
    dense = DenseRetriever(
        chunks, GEOMETRY, [c.chunk_id for c in chunks], manifest, FakeEmbedder()
    )
    index = PgVectorIndex(schema, set_id, FakeEmbedder())

    query = basis(0.9, 0.3, 0.1)
    here = index.search_vector(query, 4)
    there = dense.search_vector(query, 4)

    assert [r.chunk.node_path for r in here] == [r.chunk.node_path for r in there]
    assert [r.score for r in here] == pytest.approx([r.score for r in there], abs=1e-6)
    assert [r.chunk for r in here] == [r.chunk for r in there]


def test_a_hit_carries_its_edges_and_defined_terms(schema):
    chunks, set_id = loaded(schema)
    index = PgVectorIndex(schema, set_id, FakeEmbedder())

    (result,) = index.search_vector(basis(1), 1)

    assert result.chunk == chunks[0]
    assert result.chunk.outgoing_refs == ("2(1)", "Schedule I")
    assert result.chunk.defined_terms == ("tax", "person")


def test_k_larger_than_the_corpus_returns_every_chunk(schema):
    chunks, set_id = loaded(schema)
    index = PgVectorIndex(schema, set_id, FakeEmbedder())
    assert len(index.search_vector(basis(1), 50)) == len(chunks)


def test_k_below_one_is_refused(schema):
    _, set_id = loaded(schema)
    index = PgVectorIndex(schema, set_id, FakeEmbedder())
    with pytest.raises(ValueError, match="k must be at least 1"):
        index.search_vector(basis(1), 0)
    with pytest.raises(ValueError, match="k must be at least 1"):
        index.search("anything", 0)


def test_a_query_vector_of_the_wrong_width_is_refused(schema):
    _, set_id = loaded(schema)
    index = PgVectorIndex(schema, set_id, FakeEmbedder())
    with pytest.raises(ValueError, match="width 3"):
        index.search_vector([0.1, 0.2, 0.3], 2)


def test_search_embeds_the_query_as_a_query(schema):
    _, set_id = loaded(schema)
    embedder = FakeEmbedder()
    index = PgVectorIndex(schema, set_id, embedder)

    results = index.search("what is the short title?", 2)

    assert len(results) == 2
    # The probe set first, then the query itself — never as a document.
    assert embedder.calls[-1] == (EmbedKind.QUERY, 1)
    assert index.fingerprint_cosine == pytest.approx(1.0)


def test_an_unknown_embedding_set_is_refused(schema):
    loaded(schema)
    with pytest.raises(StaleVectorStoreError, match="no embedding set 999"):
        PgVectorIndex(schema, 999, FakeEmbedder())


def test_a_different_embedder_is_refused(schema):
    _, set_id = loaded(schema)
    other = FakeEmbedder(MODEL.model_copy(update={"revision": "r2"}))
    with pytest.raises(ModelIdentityError, match="was built with"):
        PgVectorIndex(schema, set_id, other)


def test_a_partially_deleted_set_is_refused(schema):
    chunks, set_id = loaded(schema)
    schema.execute(
        "DELETE FROM chunk_embeddings WHERE embedding_set_id = %s AND chunk_id = %s",
        (set_id, chunks[3].chunk_id),
    )
    with pytest.raises(StaleVectorStoreError, match="claims 4 vectors but holds 3"):
        PgVectorIndex(schema, set_id, FakeEmbedder())


def test_a_set_that_does_not_cover_its_corpus_is_refused(schema):
    _, set_id = loaded(schema)
    # No ingest can produce this; a hand-edited database can, and a short index
    # answers every query rather than failing.
    schema.execute(
        "UPDATE corpus_versions SET chunk_count = 5 WHERE corpus_version = %s", (V1,)
    )
    with pytest.raises(StaleVectorStoreError, match="partial index"):
        PgVectorIndex(schema, set_id, FakeEmbedder())


def test_only_the_named_set_is_searched(schema):
    chunks, first = loaded(schema)
    # A second set over the same corpus, with the geometry reversed, so the two
    # disagree about rank 1 and the id being passed is the only thing deciding.
    second = ingest_vectors(
        schema,
        GEOMETRY[::-1],
        [chunk.chunk_id for chunk in chunks],
        vector_manifest(chunks, vectors_sha256="4" * 64),
    ).embedding_set_id
    assert second != first

    embedder = FakeEmbedder()

    def top(set_id: int) -> str:
        index = PgVectorIndex(schema, set_id, embedder)
        return index.search_vector(basis(1), 1)[0].chunk.node_path

    assert top(first) == "1"
    assert top(second) == "Schedule I"


def test_a_drifted_embedder_refuses_stickily(schema):
    _, set_id = loaded(schema)
    drifted = FakeEmbedder()
    index = PgVectorIndex(schema, set_id, drifted)
    # The identity fields still match: a hosted API exposes no weights revision,
    # so only the probes can catch this (ADR-075).
    drifted.drift = True

    with pytest.raises(DenseRetrievalError, match="drifted"):
        index.search("anything", 2)
    calls_before = len(drifted.calls)
    with pytest.raises(DenseRetrievalError, match="drifted"):
        index.search("anything", 2)
    assert len(drifted.calls) == calls_before


def test_a_vendor_outage_surfaces_as_the_fallback_type(schema):
    _, set_id = loaded(schema)
    embedder = FakeEmbedder()
    index = PgVectorIndex(schema, set_id, embedder)
    embedder.fail = EmbeddingAPIError("503 from the vendor")

    with pytest.raises(DenseRetrievalError):
        index.search("anything", 2)

    # Not sticky: the outage ends and the next query tries again.
    embedder.fail = None
    assert len(index.search("anything", 2)) == 2
