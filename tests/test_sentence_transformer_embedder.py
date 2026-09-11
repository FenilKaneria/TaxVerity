"""Step 4.4 — the local reference spec and the sentence-transformers backend.

Since R15 this backend is the offline fidelity reference for the Jina hosted
API, not a serving path (ADR-075). The integration tests load ~1.2 GB of
weights and are skipped unless the snapshot is already in the Hugging Face
cache (scripts/download_models.py puts it there). CI, which has neither the
`embed` extra nor the cache, runs only the pure tests below.
"""

from __future__ import annotations

import dataclasses
import math

import pytest

from taxverity.embedding.backends import Embedder, EmbedKind
from taxverity.embedding.candidates import (
    CANDIDATES,
    JINA_V5,
    ST_ENCODING,
    EmbedderSpec,
)


def _cached(spec: EmbedderSpec) -> bool:
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return False
    hit = try_to_load_from_cache(
        spec.model_id, "model.safetensors", revision=spec.revision
    )
    return isinstance(hit, str)


needs_jina = pytest.mark.skipif(
    not _cached(JINA_V5), reason="jina-v5 snapshot not in the HF cache"
)


# --- pure: specs -----------------------------------------------------------


def test_only_the_reference_model_is_registered():
    """Qwen3 was dropped with the Step 4.7 bake-off at R15 (ADR-075)."""
    assert CANDIDATES == (JINA_V5,)


def test_specs_are_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        JINA_V5.dim = 512  # type: ignore[misc]


def test_the_reference_matches_the_served_width():
    from taxverity.embedding.jina_api import DIM

    assert JINA_V5.dim == DIM == 1024


def test_the_reference_is_pinned_to_a_commit_sha():
    assert len(JINA_V5.revision) == 40
    assert set(JINA_V5.revision) <= set("0123456789abcdef")


def test_the_reference_carries_the_versioned_encoding_recipe():
    assert JINA_V5.encoding == ST_ENCODING


def test_a_bad_revision_is_refused():
    with pytest.raises(ValueError, match="commit sha"):
        EmbedderSpec(key="x", model_id="x/y", revision="main", dim=1024)


# --- integration: the real backend --------------------------------------


@pytest.fixture(scope="module")
def jina():
    from taxverity.embedding.sentence_transformer import SentenceTransformerEmbedder

    return SentenceTransformerEmbedder(JINA_V5, device="cpu")


@needs_jina
def test_backend_satisfies_the_embedder_protocol(jina):
    assert isinstance(jina, Embedder)


@needs_jina
def test_info_reports_the_spec_identity(jina):
    info = jina.info()
    assert info.model_id == JINA_V5.model_id
    assert info.revision == JINA_V5.revision
    assert info.dim == 1024
    assert info.runtime == "torch"
    assert info.encoding == ST_ENCODING


@needs_jina
def test_document_vectors_are_unit_length_and_the_right_width(jina):
    vectors = jina.embed(["income from house property"], EmbedKind.DOCUMENT)
    assert len(vectors) == 1
    assert len(vectors[0]) == 1024
    assert math.isclose(math.sqrt(sum(v * v for v in vectors[0])), 1.0, abs_tol=1e-9)


@needs_jina
def test_the_two_kinds_produce_different_vectors_for_one_text(jina):
    text = "deduction on rental income"
    doc = jina.embed([text], EmbedKind.DOCUMENT)[0]
    query = jina.embed([text], EmbedKind.QUERY)[0]
    assert doc != query


@needs_jina
def test_encoding_is_deterministic(jina):
    a = jina.embed(["section 80C"], EmbedKind.DOCUMENT)
    b = jina.embed(["section 80C"], EmbedKind.DOCUMENT)
    assert a == b


@needs_jina
def test_a_batch_returns_one_vector_per_text_in_order(jina):
    texts = ["house property", "capital gains", "salary"]
    vectors = jina.embed(texts, EmbedKind.DOCUMENT)
    assert len(vectors) == 3
    assert vectors[0] == jina.embed(["house property"], EmbedKind.DOCUMENT)[0]


@needs_jina
def test_a_dim_disagreement_is_refused():
    from taxverity.embedding.sentence_transformer import SentenceTransformerEmbedder

    wrong = dataclasses.replace(JINA_V5, dim=512)
    with pytest.raises(ValueError, match="embeds at dim=1024"):
        SentenceTransformerEmbedder(wrong, device="cpu")
