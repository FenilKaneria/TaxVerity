"""Step 4.4 — candidate specs and the real sentence-transformers backend.

The integration tests load ~1.2 GB of weights and are skipped unless the
snapshot is already in the Hugging Face cache (scripts/download_models.py puts
it there). CI, which has neither the `embed` extra nor the cache, runs only the
pure tests below.
"""

from __future__ import annotations

import dataclasses
import math

import pytest

from taxverity.embedding.backends import Embedder, EmbedKind
from taxverity.embedding.candidates import (
    CANDIDATES,
    JINA_V5,
    QWEN3,
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
needs_qwen = pytest.mark.skipif(
    not _cached(QWEN3), reason="qwen3 snapshot not in the HF cache"
)


# --- pure: specs -----------------------------------------------------------


def test_both_finalists_are_registered():
    assert {s.key for s in CANDIDATES} == {"jina-v5", "qwen3-0.6b"}


def test_specs_are_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        JINA_V5.dim = 512  # type: ignore[misc]


def test_specs_declare_1024_dim():
    assert JINA_V5.dim == 1024
    assert QWEN3.dim == 1024


def test_specs_carry_distinct_models_at_pinned_shas():
    assert JINA_V5.model_id != QWEN3.model_id
    for spec in CANDIDATES:
        assert len(spec.revision) == 40
        assert set(spec.revision) <= set("0123456789abcdef")


def test_specs_share_the_versioned_encoding_recipe():
    assert JINA_V5.encoding == ST_ENCODING == QWEN3.encoding


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


@needs_qwen
def test_qwen_loads_and_reports_1024():
    from taxverity.embedding.sentence_transformer import SentenceTransformerEmbedder

    embedder = SentenceTransformerEmbedder(QWEN3, device="cpu")
    assert embedder.info().dim == 1024
    assert len(embedder.embed(["advance tax"], EmbedKind.DOCUMENT)[0]) == 1024
