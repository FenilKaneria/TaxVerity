"""Step 4.4 — the vector store manifest, writer, integrity-checked loader, and
the probe fingerprint that stands in for a weights revision (ADR-075)."""

from __future__ import annotations

import json

import pytest

from taxverity.embedding.backends import (
    EmbedKind,
    ModelIdentityError,
    ModelInfo,
    StubEmbedder,
)
from taxverity.embedding.store import (
    IDS_FILENAME,
    PROBE_SET_VERSION,
    PROBES,
    VECTOR_MANIFEST_FILENAME,
    VECTOR_STORE_VERSION,
    VECTORS_FILENAME,
    ProbeVector,
    StaleVectorStoreError,
    VectorManifest,
    embed_probes,
    load_vector_store,
    verify_fingerprint,
    write_vector_store,
)

MODEL = ModelInfo(model_id="m", dim=4, revision="rev", runtime="torch", encoding="v1")
_SHA = "0" * 64
PROBE = {"kind": "document", "text": "t", "vector": [1.0, 0.0, 0.0, 0.0]}


def _manifest(**overrides) -> dict:
    base = dict(
        store_version=VECTOR_STORE_VERSION,
        doc_id="income-tax-act-2025",
        corpus_version="c" * 64,
        model=MODEL.model_dump(mode="json"),
        kind="document",
        device="jina-api",
        dim=4,
        chunk_count=3,
        vectors_sha256=_SHA,
        ids_sha256=_SHA,
        probe_set_version=PROBE_SET_VERSION,
        probes=[PROBE],
    )
    base.update(overrides)
    return base


# --- pure: manifest validation ------------------------------------------


def test_manifest_round_trips():
    m = VectorManifest.model_validate(_manifest())
    assert m.kind == "document"
    assert m.model.dim == 4
    assert m.probes[0].kind is EmbedKind.DOCUMENT


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


def test_manifest_requires_a_fingerprint():
    with pytest.raises(ValueError):
        VectorManifest.model_validate(_manifest(probes=[]))


def test_manifest_rejects_a_probe_of_the_wrong_width():
    bad = {**PROBE, "vector": [1.0, 0.0]}
    with pytest.raises(ValueError, match="width"):
        VectorManifest.model_validate(_manifest(probes=[bad]))


# --- pure: the probe fingerprint ------------------------------------------


class Drifted(StubEmbedder):
    """Reports the stub's identity but returns other vectors — a silent
    upstream model swap, which is exactly what the fingerprint exists for."""

    def embed(self, texts, kind):
        return super().embed([f"drifted {text}" for text in texts], kind)


class Renamed(StubEmbedder):
    def info(self):
        return super().info().model_copy(update={"revision": "other"})


def _stub_manifest() -> VectorManifest:
    stub = StubEmbedder()
    return VectorManifest.model_validate(
        _manifest(
            model=stub.info().model_dump(mode="json"),
            dim=stub.info().dim,
            probes=[p.model_dump(mode="json") for p in embed_probes(stub)],
        )
    )


def test_probes_cover_both_kinds_and_every_probe_text():
    probes = embed_probes(StubEmbedder())
    assert {p.kind for p in probes} == set(EmbedKind)
    assert sorted(p.text for p in probes) == sorted(text for _, text in PROBES)


def test_probe_vectors_match_what_the_embedder_returns():
    stub = StubEmbedder()
    for probe in embed_probes(stub):
        fresh = stub.embed([probe.text], probe.kind)[0]
        assert probe.vector == pytest.approx(fresh, abs=1e-7)


def test_the_fingerprint_passes_for_the_embedder_that_built_it():
    assert verify_fingerprint(StubEmbedder(), _stub_manifest()) == pytest.approx(1.0)


def test_the_fingerprint_refuses_a_different_identity():
    with pytest.raises(ModelIdentityError, match="index was built with"):
        verify_fingerprint(Renamed(), _stub_manifest())


def test_the_fingerprint_refuses_same_identity_different_vectors():
    with pytest.raises(ModelIdentityError, match="same identity"):
        verify_fingerprint(Drifted(), _stub_manifest())


def test_the_fingerprint_refuses_a_stale_probe_set():
    stale = _stub_manifest().model_copy(update={"probe_set_version": PROBE_SET_VERSION + 1})
    with pytest.raises(StaleVectorStoreError, match="probe set"):
        verify_fingerprint(StubEmbedder(), stale)


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
        device="jina-api",
        probes=(ProbeVector(kind=EmbedKind.DOCUMENT, text="t", vector=(1.0,) + (0.0,) * (dim - 1)),),
    )
    return vectors, manifest


def test_write_then_load_returns_the_same_data(tmp_path):
    written, manifest = _write(tmp_path)
    vectors, chunk_ids, loaded = load_vector_store(tmp_path, corpus_version="c" * 64)
    assert np.array_equal(vectors, written)
    assert chunk_ids == ("a" * 16, "b" * 16, "c" * 16)
    assert loaded == manifest
    assert loaded.kind == "document"
    assert loaded.device == "jina-api"
    assert loaded.probe_set_version == PROBE_SET_VERSION


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


@pytest.mark.parametrize("version", [1, VECTOR_STORE_VERSION + 1])
def test_load_refuses_any_other_store_version(tmp_path, version):
    """Version 1 is the pre-R15 manifest with no fingerprint: it must read as
    stale, not fail validation as if it were corrupt."""
    _write(tmp_path)
    path = tmp_path / VECTOR_MANIFEST_FILENAME
    data = json.loads(path.read_text())
    data["store_version"] = version
    if version == 1:
        del data["probes"], data["probe_set_version"]
    path.write_text(json.dumps(data))
    with pytest.raises(StaleVectorStoreError, match="store version"):
        load_vector_store(tmp_path)


def test_load_raises_when_a_file_is_missing(tmp_path):
    _write(tmp_path)
    (tmp_path / IDS_FILENAME).unlink()
    with pytest.raises(StaleVectorStoreError, match="missing"):
        load_vector_store(tmp_path)


def _write_bad(tmp_path, vectors):
    write_vector_store(
        tmp_path,
        vectors=vectors,
        chunk_ids=("a" * 16, "b" * 16, "c" * 16),
        corpus_version="c" * 64,
        doc_id="d",
        model=MODEL,
        kind=EmbedKind.DOCUMENT,
        device="cpu",
        probes=(ProbeVector(**PROBE),),
    )


def test_write_refuses_non_float32_vectors(tmp_path):
    with pytest.raises(ValueError, match="float32"):
        _write_bad(tmp_path, np.zeros((3, 4), dtype=np.float64))


def test_write_refuses_a_shape_that_disagrees_with_the_ids(tmp_path):
    with pytest.raises(ValueError, match="shape"):
        _write_bad(tmp_path, np.zeros((2, 4), dtype=np.float32))
