"""Step 4.4 — the vector store: the offline-built corpus embeddings and their
manifest.

Three files, written by `scripts/embed_corpus.py` and read by Step 4.5's vector
search:

- `corpus_vectors.npy`   float32 matrix, shape [N, dim], row i is chunk i
- `corpus_vectors.ids.json`  the N chunk ids, row order — the join key
- `vector_manifest.json`  identity, integrity and fingerprint metadata

Unlike every other artifact in this project the `.npy` is *not* byte-reproducible
across runs — a re-encode drifts at ~1e-6 (ADR-073). `vectors_sha256` is
therefore an integrity seal for one built artifact, not a reproducibility claim;
the thing that says two indexes are comparable is `corpus_version` plus the
`ModelInfo` tuple plus the probe fingerprint.

The fingerprint exists because the served embedder is a hosted API (ADR-075),
which exposes no weights revision. A fixed set of probe texts is embedded when
the index is built; before the index is searched they are embedded again, and
an embedder that no longer reproduces them is refused.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

from taxverity.corpus.loader import hash_file
from taxverity.embedding.backends import (
    Embedder,
    EmbedKind,
    ModelIdentityError,
    ModelInfo,
    describe,
)
from taxverity.observability import get_logger

logger = get_logger(__name__)

VECTOR_STORE_VERSION = 2

VECTORS_FILENAME = "corpus_vectors.npy"
IDS_FILENAME = "corpus_vectors.ids.json"
VECTOR_MANIFEST_FILENAME = "vector_manifest.json"

_SHA_HEX_LEN = 64

# Fixed and versioned: changing a probe changes every stored fingerprint, so
# bump PROBE_SET_VERSION with it. Both kinds are probed because the API applies
# a different task adapter to each, and a change to either moves one side of
# the index.
PROBE_SET_VERSION = 1
PROBES: tuple[tuple[EmbedKind, str], ...] = (
    (
        EmbedKind.DOCUMENT,
        "The annual value of any property shall be deemed to be the sum for "
        "which the property might reasonably be expected to let from year to year.",
    ),
    (
        EmbedKind.DOCUMENT,
        "Any person responsible for paying any income chargeable under the head "
        "Salaries shall deduct income-tax at the time of payment.",
    ),
    (EmbedKind.QUERY, "Can I claim a deduction for rent paid to my mother?"),
    (EmbedKind.QUERY, "How is long-term capital gain on listed shares taxed?"),
)
# Measured 2026-09-11: re-embedding the probes through the Jina API drifts to a
# lowest cosine of 0.999958 (~4e-5), and the API agrees with the pinned local
# weights at >= 0.99991. 0.999 sits ~24x above that noise and far below the gap
# any model change opens.
FINGERPRINT_MIN_COSINE = 0.999
_PROBE_DECIMALS = 7


class StaleVectorStoreError(RuntimeError):
    pass


class ProbeVector(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: EmbedKind
    text: str = Field(min_length=1)
    vector: tuple[float, ...] = Field(min_length=1)


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
    # Provenance only — where the batch job ran. Never compared as skew
    # (ADR-072).
    device: str = Field(min_length=1)
    dim: int = Field(gt=0)
    chunk_count: int = Field(ge=1)
    vectors_sha256: str = Field(min_length=_SHA_HEX_LEN, max_length=_SHA_HEX_LEN)
    ids_sha256: str = Field(min_length=_SHA_HEX_LEN, max_length=_SHA_HEX_LEN)
    probe_set_version: int = Field(ge=1)
    probes: tuple[ProbeVector, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _fields_agree(self) -> VectorManifest:
        if self.dim != self.model.dim:
            raise ValueError(f"manifest dim {self.dim} != model dim {self.model.dim}")
        if self.kind not in tuple(k.value for k in EmbedKind):
            raise ValueError(f"kind {self.kind!r} is not an EmbedKind")
        for probe in self.probes:
            if len(probe.vector) != self.dim:
                raise ValueError(
                    f"probe {probe.text!r} has width {len(probe.vector)}, "
                    f"manifest dim is {self.dim}"
                )
        return self


def embed_probes(embedder: Embedder) -> tuple[ProbeVector, ...]:
    """Embed the fixed probe set, one request per kind."""
    probes: list[ProbeVector] = []
    for kind in EmbedKind:
        texts = [text for probe_kind, text in PROBES if probe_kind is kind]
        for text, vector in zip(texts, embedder.embed(texts, kind), strict=True):
            probes.append(
                ProbeVector(
                    kind=kind,
                    text=text,
                    vector=tuple(round(v, _PROBE_DECIMALS) for v in vector),
                )
            )
    return tuple(probes)


def verify_fingerprint(
    embedder: Embedder,
    manifest: VectorManifest,
    *,
    min_cosine: float = FINGERPRINT_MIN_COSINE,
) -> float:
    """Refuse an embedder that no longer reproduces the index's probe vectors.

    Returns the lowest probe cosine, so a caller can record the margin.
    """
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
    worst, worst_text = 1.0, ""
    for kind in EmbedKind:
        stored = [probe for probe in manifest.probes if probe.kind is kind]
        if not stored:
            continue
        fresh = embedder.embed([probe.text for probe in stored], kind)
        for probe, vector in zip(stored, fresh, strict=True):
            cosine = _cosine(probe.vector, vector)
            if cosine < worst:
                worst, worst_text = cosine, probe.text
    if worst < min_cosine:
        raise ModelIdentityError(
            f"probe {worst_text!r} re-embeds at cosine {worst:.6f} < {min_cosine}: "
            f"the embedder no longer produces the vectors this index was built "
            f"with, though it reports the same identity."
        )
    logger.info("fingerprint verified: lowest probe cosine %.6f", worst)
    return worst


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = math.fsum(x * y for x, y in zip(a, b, strict=True))
    norm = math.sqrt(math.fsum(x * x for x in a)) * math.sqrt(
        math.fsum(y * y for y in b)
    )
    return dot / norm


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
    probes: tuple[ProbeVector, ...],
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
        probe_set_version=PROBE_SET_VERSION,
        probes=probes,
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

    # The version is checked before validation: an older manifest lacks fields
    # this one requires, and should read as stale rather than as malformed.
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    if raw.get("store_version") != VECTOR_STORE_VERSION:
        raise StaleVectorStoreError(
            f"{directory} is store version {raw.get('store_version')}, "
            f"not {VECTOR_STORE_VERSION} — rebuild it."
        )
    manifest = VectorManifest.model_validate(raw)
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
