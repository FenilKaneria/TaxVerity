"""Step 4.4 — the vector store: the offline-built corpus embeddings and their
manifest.

Three files, written by `scripts/embed_corpus.py` and read by Step 4.5's vector
search:

- `corpus_vectors.npy`   float32 matrix, shape [N, dim], row i is chunk i
- `corpus_vectors.ids.json`  the N chunk ids, row order — the join key
- `vector_manifest.json`  identity and integrity metadata

Unlike every other artifact in this project the `.npy` is *not* byte-reproducible
across runs — CUDA reduces in a nondeterministic order, so a re-encode drifts at
~1e-6 (ADR-073). `vectors_sha256` is therefore an integrity seal for one built
artifact, not a reproducibility claim; the thing that says two indexes are
comparable is `corpus_version` plus the `ModelInfo` tuple.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

from taxverity.corpus.loader import hash_file
from taxverity.embedding.backends import EmbedKind, ModelInfo
from taxverity.observability import get_logger

logger = get_logger(__name__)

VECTOR_STORE_VERSION = 1

VECTORS_FILENAME = "corpus_vectors.npy"
IDS_FILENAME = "corpus_vectors.ids.json"
VECTOR_MANIFEST_FILENAME = "vector_manifest.json"

_SHA_HEX_LEN = 64


class StaleVectorStoreError(RuntimeError):
    pass


class VectorManifest(BaseModel):
    model_config = ConfigDict(frozen=True)

    store_version: int = Field(ge=1)
    doc_id: str = Field(min_length=1)
    # The corpus these vectors describe. A chunk id is derived from
    # corpus_version, so vectors built against one corpus are mislabelled, not
    # merely old, against another — same rule the chunk store enforces.
    corpus_version: str = Field(min_length=1)
    model: ModelInfo
    # EmbedKind.value. A vector carries no evidence of how it was encoded, so
    # Step 4.5 refuses an index that is not "document" (ADR-069).
    kind: str = Field(min_length=1)
    # Provenance only — the device the batch job ran on. Never compared as skew
    # (ADR-072).
    device: str = Field(min_length=1)
    dim: int = Field(gt=0)
    chunk_count: int = Field(ge=1)
    vectors_sha256: str = Field(min_length=_SHA_HEX_LEN, max_length=_SHA_HEX_LEN)
    ids_sha256: str = Field(min_length=_SHA_HEX_LEN, max_length=_SHA_HEX_LEN)

    @model_validator(mode="after")
    def _fields_agree(self) -> VectorManifest:
        if self.dim != self.model.dim:
            raise ValueError(f"manifest dim {self.dim} != model dim {self.model.dim}")
        if self.kind not in tuple(k.value for k in EmbedKind):
            raise ValueError(f"kind {self.kind!r} is not an EmbedKind")
        return self


def _write_json(payload: object, destination: Path) -> None:
    text = json.dumps(payload, sort_keys=True, ensure_ascii=False, indent=2)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        handle.write(text + "\n")


def write_vector_store(
    directory: Path,
    *,
    vectors: object,
    chunk_ids: tuple[str, ...],
    corpus_version: str,
    doc_id: str,
    model: ModelInfo,
    kind: EmbedKind,
    device: str,
) -> VectorManifest:
    """Write the three files and return the manifest. `vectors` is a float32
    ndarray of shape (len(chunk_ids), model.dim)."""
    import numpy as np

    arr = np.asarray(vectors)
    if arr.dtype != np.float32:
        raise ValueError(f"vectors must be float32, got {arr.dtype}")
    if arr.shape != (len(chunk_ids), model.dim):
        raise ValueError(
            f"vectors shape {arr.shape} != ({len(chunk_ids)}, {model.dim})"
        )

    directory.mkdir(parents=True, exist_ok=True)
    vectors_path = directory / VECTORS_FILENAME
    ids_path = directory / IDS_FILENAME

    with vectors_path.open("wb") as handle:
        np.save(handle, arr, allow_pickle=False)
    _write_json(list(chunk_ids), ids_path)

    manifest = VectorManifest(
        store_version=VECTOR_STORE_VERSION,
        doc_id=doc_id,
        corpus_version=corpus_version,
        model=model,
        kind=kind.value,
        device=device,
        dim=model.dim,
        chunk_count=len(chunk_ids),
        vectors_sha256=hash_file(vectors_path),
        ids_sha256=hash_file(ids_path),
    )
    _write_json(manifest.model_dump(mode="json"), directory / VECTOR_MANIFEST_FILENAME)
    logger.info("wrote %d vectors (dim %d) to %s", len(chunk_ids), model.dim, directory)
    return manifest


def load_vector_store(
    directory: Path, *, corpus_version: str | None = None
) -> tuple[object, tuple[str, ...], VectorManifest]:
    """Read the store, refusing anything that does not describe this corpus or
    that fails its own integrity seals. Returns (vectors ndarray, chunk_ids,
    manifest)."""
    import numpy as np

    vectors_path = directory / VECTORS_FILENAME
    ids_path = directory / IDS_FILENAME
    manifest_path = directory / VECTOR_MANIFEST_FILENAME
    for path in (vectors_path, ids_path, manifest_path):
        if not path.exists():
            raise StaleVectorStoreError(f"missing {path} — run scripts/embed_corpus.py")

    manifest = VectorManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    )
    if manifest.store_version != VECTOR_STORE_VERSION:
        raise StaleVectorStoreError(
            f"{directory} is store version {manifest.store_version}, "
            f"not {VECTOR_STORE_VERSION} — rebuild it."
        )
    if corpus_version is not None and manifest.corpus_version != corpus_version:
        raise StaleVectorStoreError(
            f"{directory} was built for corpus_version {manifest.corpus_version}, "
            f"not {corpus_version}. Vectors are keyed to the corpus by chunk id — "
            f"rebuild the store."
        )
    if hash_file(vectors_path) != manifest.vectors_sha256:
        raise StaleVectorStoreError(f"{vectors_path} does not match its manifest hash.")
    if hash_file(ids_path) != manifest.ids_sha256:
        raise StaleVectorStoreError(f"{ids_path} does not match its manifest hash.")

    chunk_ids = tuple(json.loads(ids_path.read_text(encoding="utf-8")))
    with vectors_path.open("rb") as handle:
        vectors = np.load(handle, allow_pickle=False)
    if vectors.shape != (manifest.chunk_count, manifest.dim):
        raise StaleVectorStoreError(
            f"{vectors_path} has shape {vectors.shape}, manifest says "
            f"({manifest.chunk_count}, {manifest.dim})."
        )
    if len(chunk_ids) != manifest.chunk_count:
        raise StaleVectorStoreError(
            f"{ids_path} holds {len(chunk_ids)} ids, manifest says "
            f"{manifest.chunk_count}."
        )
    return vectors, chunk_ids, manifest
