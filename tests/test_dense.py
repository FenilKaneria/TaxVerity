"""Step 4.5 — exact dense retrieval over the vector store (ADR-078).

Pure tests use a hand-built geometry and the stub embedder: no network, no key.
The corpus tests read the real Step 4.4 store and skip when it is absent; the
live test spends a few hundred tokens and skips without a key.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from conftest import Offline
from taxverity.chunking.models import Chunk
from taxverity.chunking.pipeline import read_corpus_version
from taxverity.chunking.store import load_chunks
from taxverity.config import Settings
from taxverity.corpus.nodes import NodeType
from taxverity.embedding.backends import (
    EmbedKind,
    ModelIdentityError,
    StubEmbedder,
)
from taxverity.embedding.jina_api import EmbeddingAPIError, JinaAPIEmbedder
from taxverity.embedding.store import (
    PROBE_SET_VERSION,
    StaleVectorStoreError,
    VectorManifest,
    embed_probes,
    load_vector_store,
)
from taxverity.retrieval.base import Retriever, as_ranked_citations
from taxverity.retrieval.dense import DenseRetrievalError, DenseRetriever

CORPUS_VERSION = "v" * 64
DIM = StubEmbedder().info().dim


def chunk(node_path: str, text: str) -> Chunk:
    return Chunk.create(
        CORPUS_VERSION,
        node_path,
        text,
        parent_id=None,
        doc_id="income-tax-act-2025",
        node_type=NodeType.SECTION,
        section_number=node_path,
        root_title=None,
        page_start=1,
        page_end=1,
    )


def basis(*weights: float) -> np.ndarray:
    row = np.zeros(DIM, dtype=np.float64)
    row[: len(weights)] = weights
    return (row / np.linalg.norm(row)).astype(np.float32)


# A geometry whose neighbours are known by construction: along e0 the order is
# 21 (1.0), then 24 (cos 45°), then 22 and 23 tied at 0.
CORPUS = [chunk("21", "a"), chunk("22", "b"), chunk("23", "c"), chunk("24", "d")]
VECTORS = np.stack([basis(1), basis(0, 1), basis(0, 0, 1), basis(1, 1)])
IDS = tuple(c.chunk_id for c in CORPUS)


def manifest(embedder=None, **overrides) -> VectorManifest:
    embedder = embedder or StubEmbedder()
    base = dict(
        store_version=2,
        doc_id="income-tax-act-2025",
        corpus_version=CORPUS_VERSION,
        model=embedder.info().model_dump(mode="json"),
        kind="document",
        device="test",
        dim=DIM,
        chunk_count=len(IDS),
        vectors_sha256="0" * 64,
        ids_sha256="0" * 64,
        probe_set_version=PROBE_SET_VERSION,
        probes=[p.model_dump(mode="json") for p in embed_probes(StubEmbedder())],
    )
    base.update(overrides)
    return VectorManifest.model_validate(base)


class Counting(StubEmbedder):
    """The stub, recording every call's kind and size."""

    def __init__(self) -> None:
        self.calls: list[tuple[EmbedKind, int]] = []

    def embed(self, texts, kind):
        self.calls.append((kind, len(texts)))
        return super().embed(texts, kind)


class Drifted(Counting):
    """Reports the stub's identity but returns other vectors: a silent upstream
    model swap."""

    def embed(self, texts, kind):
        self.calls.append((kind, len(texts)))
        return StubEmbedder.embed(self, [f"drifted {t}" for t in texts], kind)


class Failing(Counting):
    """Fails the first `failures` calls the way the Jina client does when its
    retries are exhausted."""

    def __init__(self, failures: int) -> None:
        super().__init__()
        self.failures = failures

    def embed(self, texts, kind):
        if self.failures:
            self.failures -= 1
            self.calls.append((kind, len(texts)))
            raise EmbeddingAPIError("embedding API failed after 6 attempts: 503")
        return super().embed(texts, kind)


def retriever(embedder=None, *, chunks=CORPUS, vectors=VECTORS, ids=IDS, **overrides):
    embedder = embedder or StubEmbedder()
    return DenseRetriever(chunks, vectors, ids, manifest(embedder, **overrides), embedder)


def paths(results):
    return [r.chunk.node_path for r in results]


# --- ranking ---------------------------------------------------------------------


def test_satisfies_the_retriever_protocol():
    assert isinstance(retriever(), Retriever)


def test_known_neighbours_come_back_in_cosine_order():
    results = retriever().search_vector(basis(1), 4)
    assert paths(results) == ["21", "24", "22", "23"]
    assert [r.score for r in results] == pytest.approx([1.0, math.sqrt(0.5), 0.0, 0.0])


def test_the_nearest_neighbour_of_each_row_is_itself():
    dense = retriever()
    for c, row in zip(CORPUS, VECTORS, strict=True):
        assert paths(dense.search_vector(row, 1)) == [c.node_path]


def test_ties_break_by_corpus_order_whatever_the_store_order():
    """The store is written in chunk order today, but nothing guarantees it; a
    reversed store must rank exactly as the forward one does."""
    reversed_dense = retriever(vectors=VECTORS[::-1].copy(), ids=IDS[::-1])
    assert paths(reversed_dense.search_vector(basis(1), 4)) == ["21", "24", "22", "23"]


def test_k_bounds_the_results_and_a_large_k_returns_everything():
    dense = retriever()
    assert len(dense.search_vector(basis(1), 2)) == 2
    assert len(dense.search_vector(basis(1), 50)) == len(CORPUS)


@pytest.mark.parametrize("k", [0, -1])
def test_k_below_one_is_refused_before_any_embedding(k):
    embedder = Counting()
    dense = retriever(embedder)
    with pytest.raises(ValueError, match="k must be"):
        dense.search("rent", k)
    with pytest.raises(ValueError, match="k must be"):
        dense.search_vector(basis(1), k)
    assert embedder.calls == []


def test_results_cross_the_bridge_into_the_metrics():
    results = retriever().search("annual value of property", 4)
    assert len(as_ranked_citations(results)) == 4


def test_a_query_vector_of_the_wrong_width_is_refused():
    with pytest.raises(ValueError, match="dim"):
        retriever().search_vector([1.0, 0.0], 1)


def test_search_is_search_vector_of_the_query_encoding():
    stub = StubEmbedder()
    query = "can I pay rent to my mother"
    (vector,) = stub.embed([query], EmbedKind.QUERY)
    dense = retriever(stub)
    assert dense.search(query, 4) == dense.search_vector(vector, 4)


def test_embed_query_is_the_query_encoding_from_one_call():
    embedder = Counting()
    dense = retriever(embedder)
    dense.search("warm", 1)
    before = len(embedder.calls)
    vector = dense.embed_query("rent paid to my mother")
    assert vector == StubEmbedder().embed(["rent paid to my mother"], EmbedKind.QUERY)[0]
    assert embedder.calls[before:] == [(EmbedKind.QUERY, 1)]


# --- construction refusals: our own misconfiguration, loud ---------------------


def test_an_index_not_encoded_as_documents_is_refused():
    with pytest.raises(StaleVectorStoreError, match="not 'document'"):
        retriever(kind="query")


def test_an_embedder_other_than_the_one_that_built_the_index_is_refused():
    class Renamed(StubEmbedder):
        def info(self):
            return super().info().model_copy(update={"encoding": "other"})

    stub = StubEmbedder()
    with pytest.raises(ModelIdentityError, match="index was built with"):
        DenseRetriever(CORPUS, VECTORS, IDS, manifest(stub), Renamed())


def test_a_stale_probe_set_is_refused_at_construction():
    with pytest.raises(StaleVectorStoreError, match="probe set"):
        retriever(probe_set_version=PROBE_SET_VERSION + 1)


def test_a_chunk_without_a_vector_is_refused():
    with pytest.raises(StaleVectorStoreError, match="1 chunks have no vector"):
        retriever(chunks=[*CORPUS, chunk("25", "e")])


def test_a_vector_without_a_chunk_is_refused():
    with pytest.raises(StaleVectorStoreError, match="1 vectors have no chunk"):
        retriever(chunks=CORPUS[:3])


def test_a_chunk_id_listed_twice_is_refused():
    ids = (IDS[0], IDS[0], IDS[2], IDS[3])
    with pytest.raises(StaleVectorStoreError, match="twice"):
        retriever(ids=ids)


def test_vectors_that_are_not_unit_length_are_refused():
    with pytest.raises(StaleVectorStoreError, match="norm"):
        retriever(vectors=VECTORS * 2)


@pytest.mark.parametrize(
    "vectors",
    [VECTORS.astype(np.float64), VECTORS[:3]],
    ids=["float64", "short"],
)
def test_vectors_of_the_wrong_dtype_or_shape_are_refused(vectors):
    with pytest.raises(StaleVectorStoreError, match="float32"):
        retriever(vectors=vectors)


# --- vendor failure: typed, so Step 5.2 can fall back --------------------------


def test_the_fingerprint_is_checked_once_before_the_first_search():
    embedder = Counting()
    dense = retriever(embedder)
    assert embedder.calls == []
    dense.search("rent", 2)
    # Probes go one request per kind, then the query itself.
    assert embedder.calls == [
        (EmbedKind.QUERY, 2),
        (EmbedKind.DOCUMENT, 2),
        (EmbedKind.QUERY, 1),
    ]
    dense.search("rent", 2)
    assert embedder.calls[3:] == [(EmbedKind.QUERY, 1)]
    assert dense.fingerprint_cosine == pytest.approx(1.0)


def test_a_drifted_embedder_raises_and_stays_refused():
    embedder = Drifted()
    dense = retriever(embedder)
    with pytest.raises(DenseRetrievalError, match="drifted"):
        dense.search("rent", 2)
    probes_spent = len(embedder.calls)
    with pytest.raises(DenseRetrievalError, match="drifted"):
        dense.search("rent", 2)
    assert len(embedder.calls) == probes_spent, "a refused index re-probed"


def test_a_query_embedding_failure_raises_rather_than_returning_nothing():
    # 2 probe calls succeed, the query call fails.
    embedder = Failing(failures=0)
    dense = retriever(embedder)
    dense.search("warm", 1)
    embedder.failures = 1
    with pytest.raises(DenseRetrievalError, match="query embedding failed") as info:
        dense.search("rent", 2)
    assert isinstance(info.value.__cause__, EmbeddingAPIError)
    assert len(dense.search("rent", 2)) == 2, "an outage must not be sticky"


def test_an_outage_during_the_fingerprint_check_is_not_sticky():
    embedder = Failing(failures=1)
    dense = retriever(embedder)
    with pytest.raises(DenseRetrievalError, match="fingerprint check could not run"):
        dense.search("rent", 2)
    assert len(dense.search("rent", 2)) == 2
    assert dense.fingerprint_cosine == pytest.approx(1.0)


# --- corpus: the real Step 4.4 store, no network --------------------------------

SETTINGS = Settings()
STORE = SETTINGS.vectors_dir / "jina-api"


@pytest.fixture(scope="module")
def store():
    if not (STORE / "vector_manifest.json").exists():
        pytest.skip("run scripts/embed_corpus.py to build the vector store")
    corpus_version = read_corpus_version(SETTINGS.interim_dir / "corpus_manifest.json")
    chunks, _ = load_chunks(SETTINGS.interim_dir, corpus_version=corpus_version)
    vectors, ids, loaded = load_vector_store(STORE, corpus_version=corpus_version)
    dense = DenseRetriever(chunks, vectors, ids, loaded, Offline(loaded.model))
    return dense, chunks, vectors, ids


def test_the_real_store_joins_the_real_chunk_set(store):
    dense, chunks, _, _ = store
    assert len(dense) == len(chunks) == 8351


def test_every_sampled_chunk_is_its_own_nearest_neighbour(store):
    """Self-retrieval over the real index: a chunk's stored vector must rank that
    chunk first. A join off by one row, or an unnormalised store, fails here."""
    dense, _, vectors, ids = store
    for row in range(0, len(ids), 97):
        (top,) = dense.search_vector(vectors[row], 1)
        assert top.chunk.chunk_id == ids[row]
        assert top.score == pytest.approx(1.0, abs=1e-5)


# --- live: a few hundred tokens, skipped without a key ---------------------------

needs_key = pytest.mark.skipif(
    SETTINGS.jina_api_key is None, reason="TAXVERITY_JINA_API_KEY not configured"
)


@needs_key
def test_the_live_api_verifies_the_fingerprint_and_finds_a_quoted_provision(store):
    _, chunks, _, _ = store
    target = next(c for c in chunks if c.node_path == "21(1)")
    with JinaAPIEmbedder.from_settings(SETTINGS) as embedder:
        dense = DenseRetriever.from_store(
            STORE,
            chunks,
            embedder,
            corpus_version=read_corpus_version(
                SETTINGS.interim_dir / "corpus_manifest.json"
            ),
        )
        results = dense.search(target.text, 10)
    assert dense.fingerprint_cosine is not None and dense.fingerprint_cosine > 0.999
    assert target.chunk_id in [r.chunk.chunk_id for r in results]
