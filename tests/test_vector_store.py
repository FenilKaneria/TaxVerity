"""Step 4.4 — the vector store manifest, writer and integrity-checked loader."""

from __future__ import annotations

import json

import pytest

from taxverity.embedding.backends import EmbedKind, ModelInfo
from taxverity.embedding.store import (
    IDS_FILENAME,
    VECTOR_MANIFEST_FILENAME,
    VECTOR_STORE_VERSION,
    VECTORS_FILENAME,
    StaleVectorStoreError,
    VectorManifest,
    load_vector_store,
    write_vector_store,
)

MODEL = ModelInfo(model_id="m", dim=4, revision="rev", runtime="torch", encoding="v1")
_SHA = "0" * 64


def _manifest(**overrides) -> dict:
    base = dict(
        store_version=VECTOR_STORE_VERSION,
        doc_id="income-tax-act-2025",
        corpus_version="c" * 64,
        model=MODEL.model_dump(mode="json"),
        kind="document",
        device="cuda",
        dim=4,
        chunk_count=3,
        vectors_sha256=_SHA,
        ids_sha256=_SHA,
    )
    base.update(overrides)
    return base


# --- pure: manifest validation ------------------------------------------


def test_manifest_round_trips():
    m = VectorManifest.model_validate(_manifest())
    assert m.kind == "document"
    assert m.model.dim == 4


def test_manifest_rejects_dim_disagreeing_with_model():
    with pytest.raises(ValueError, match="dim"):
        VectorManifest.model_validate(_manifest(dim=8))


def test_manifest_rejects_a_non_embedkind_kind():
    with pytest.raises(ValueError, match="EmbedKind"):
        VectorManifest.model_validate(_manifest(kind="passage"))


def test_manifest_rejects_a_short_sha():
    with pytest.raises(ValueError):
        VectorManifest.model_validate(_manifest(vectors_sha256="abc"))


def test_manifest_rejects_store_version_below_one():
    with pytest.raises(ValueError):
        VectorManifest.model_validate(_manifest(store_version=0))


# --- integration: writer + loader (needs numpy) -------------------------

np = pytest.importorskip("numpy")


def _unit_rows(n: int, dim: int):
    rng = np.random.default_rng(0)
    arr = rng.standard_normal((n, dim)).astype(np.float32)
    arr /= np.linalg.norm(arr, axis=1, keepdims=True)
    return arr


def _write(tmp_path, *, ids=("a" * 16, "b" * 16, "c" * 16), dim=4):
    vectors = _unit_rows(len(ids), dim)
    manifest = write_vector_store(
        tmp_path,
        vectors=vectors,
        chunk_ids=ids,
        corpus_version="c" * 64,
        doc_id="income-tax-act-2025",
        model=ModelInfo(
            model_id="m", dim=dim, revision="rev", runtime="torch", encoding="v1"
        ),
        kind=EmbedKind.DOCUMENT,
        device="cuda",
    )
    return vectors, manifest


def test_write_then_load_returns_the_same_data(tmp_path):
    written, manifest = _write(tmp_path)
    vectors, chunk_ids, loaded = load_vector_store(tmp_path, corpus_version="c" * 64)
    assert np.array_equal(vectors, written)
    assert chunk_ids == ("a" * 16, "b" * 16, "c" * 16)
    assert loaded == manifest
    assert loaded.kind == "document"
    assert loaded.device == "cuda"


def test_load_refuses_a_different_corpus_version(tmp_path):
    _write(tmp_path)
    with pytest.raises(StaleVectorStoreError, match="corpus_version"):
        load_vector_store(tmp_path, corpus_version="d" * 64)


def test_load_detects_a_tampered_vectors_file(tmp_path):
    _write(tmp_path)
    (tmp_path / VECTORS_FILENAME).write_bytes(b"\x00" * 512)
    with pytest.raises(StaleVectorStoreError, match="manifest hash"):
        load_vector_store(tmp_path)


def test_load_detects_a_tampered_ids_file(tmp_path):
    _write(tmp_path)
    (tmp_path / IDS_FILENAME).write_text(json.dumps(["x" * 16, "y" * 16, "z" * 16]))
    with pytest.raises(StaleVectorStoreError, match="manifest hash"):
        load_vector_store(tmp_path)


def test_load_refuses_a_bumped_store_version(tmp_path):
    _write(tmp_path)
    path = tmp_path / VECTOR_MANIFEST_FILENAME
    data = json.loads(path.read_text())
    data["store_version"] = VECTOR_STORE_VERSION + 1
    path.write_text(json.dumps(data))
    with pytest.raises(StaleVectorStoreError, match="store version"):
        load_vector_store(tmp_path)


def test_load_raises_when_a_file_is_missing(tmp_path):
    _write(tmp_path)
    (tmp_path / IDS_FILENAME).unlink()
    with pytest.raises(StaleVectorStoreError, match="missing"):
        load_vector_store(tmp_path)


def test_write_refuses_non_float32_vectors(tmp_path):
    with pytest.raises(ValueError, match="float32"):
        write_vector_store(
            tmp_path,
            vectors=np.zeros((3, 4), dtype=np.float64),
            chunk_ids=("a" * 16, "b" * 16, "c" * 16),
            corpus_version="c" * 64,
            doc_id="d",
            model=MODEL,
            kind=EmbedKind.DOCUMENT,
            device="cpu",
        )


def test_write_refuses_a_shape_that_disagrees_with_the_ids(tmp_path):
    with pytest.raises(ValueError, match="shape"):
        write_vector_store(
            tmp_path,
            vectors=np.zeros((2, 4), dtype=np.float32),
            chunk_ids=("a" * 16, "b" * 16, "c" * 16),
            corpus_version="c" * 64,
            doc_id="d",
            model=MODEL,
            kind=EmbedKind.DOCUMENT,
            device="cpu",
        )
