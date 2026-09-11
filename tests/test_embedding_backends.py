"""Step 4.1 — the Embedder contract's pure half: ModelInfo, the identity
helpers, and the stub every offline test embeds with.

Moved here at R15 from the deleted service suite (ADR-075)."""

from __future__ import annotations

import math

from taxverity.embedding.backends import (
    STUB_DIM,
    STUB_ENCODING,
    STUB_MODEL_ID,
    STUB_REVISION,
    STUB_RUNTIME,
    Embedder,
    EmbedKind,
    ModelIdentityError,
    ModelInfo,
    StubEmbedder,
    describe,
)


def test_stub_satisfies_the_embedder_protocol():
    assert isinstance(StubEmbedder(), Embedder)


def test_stub_info_matches_its_constants():
    info = StubEmbedder().info()
    assert info.model_id == STUB_MODEL_ID
    assert info.dim == STUB_DIM
    assert info.revision == STUB_REVISION
    assert info.runtime == STUB_RUNTIME
    assert info.encoding == STUB_ENCODING


def test_stub_returns_one_vector_per_text_in_order():
    vectors = StubEmbedder().embed(["alpha", "beta", "alpha"], EmbedKind.DOCUMENT)
    assert len(vectors) == 3
    assert all(len(v) == STUB_DIM for v in vectors)
    assert vectors[0] == vectors[2]
    assert vectors[0] != vectors[1]


def test_stub_vectors_are_unit_length():
    texts = ["alpha", "", "section 80C"]
    for vector in StubEmbedder().embed(texts, EmbedKind.QUERY):
        assert math.isclose(math.sqrt(sum(v * v for v in vector)), 1.0, rel_tol=1e-12)


def test_stub_is_deterministic_across_instances():
    assert StubEmbedder().embed(["section 22(2)"], EmbedKind.QUERY) == (
        StubEmbedder().embed(["section 22(2)"], EmbedKind.QUERY)
    )


def test_stub_normalises_before_hashing():
    # Step 1.1: the corpus carries soft hyphens and NBSPs that read identically.
    assert StubEmbedder().embed(["sub­section"], EmbedKind.DOCUMENT) == (
        StubEmbedder().embed(["subsection"], EmbedKind.DOCUMENT)
    )


def test_stub_folds_kind_into_the_vector():
    text = ["deduction on rental income"]
    assert StubEmbedder().embed(text, EmbedKind.QUERY) != (
        StubEmbedder().embed(text, EmbedKind.DOCUMENT)
    )


def test_stub_embeds_an_empty_batch_to_an_empty_list():
    assert StubEmbedder().embed([], EmbedKind.DOCUMENT) == []


def test_describe_names_all_five_identity_fields():
    info = ModelInfo(model_id="m", dim=4, revision="r", runtime="x", encoding="e")
    assert describe(info) == "m@r dim=4 runtime=x encoding=e"


def test_model_identity_error_is_a_runtime_error():
    assert issubclass(ModelIdentityError, RuntimeError)
