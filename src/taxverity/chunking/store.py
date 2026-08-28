"""Step 2.4 — the chunk store: a flat JSONL file and its manifest.

No database yet. The store's one non-obvious job is refusing to serve a stale
file: a chunk id is derived from corpus_version, so chunks built against an
older corpus are not merely out of date, they are mislabelled.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from taxverity.chunking.chunker import CHUNKER_STAGE_VERSION, DOC_ID
from taxverity.chunking.models import CHUNK_STAGE_VERSION, Chunk
from taxverity.corpus.loader import hash_file
from taxverity.observability import get_logger

logger = get_logger(__name__)

CHUNKS_FILENAME = "chunks.jsonl"
CHUNK_MANIFEST_FILENAME = "chunk_manifest.json"

# Deliberately not folded into corpus_version: corpus_version is an *input* to
# every chunk id, so a chunker bump feeding back into it would make the id
# depend on the code that produced it rather than on the corpus it describes.
CHUNK_STAGE_VERSIONS = {
    "chunk_model": CHUNK_STAGE_VERSION,
    "chunker": CHUNKER_STAGE_VERSION,
}


class StaleChunkStoreError(RuntimeError):
    pass


class ChunkManifest(BaseModel):
    model_config = ConfigDict(frozen=True)

    doc_id: str
    corpus_version: str
    chunk_count: int
    root_count: int
    stage_versions: dict[str, int]
    artifact_sha256: str


def to_json_line(chunk: Chunk) -> str:
    return json.dumps(
        chunk.model_dump(mode="json"),
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def write_chunks_jsonl(
    chunks: Iterable[Chunk], destination: Path
) -> tuple[int, str]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    count = 0
    # newline="" so LF is not translated to CRLF on Windows, which would make
    # the artifact and its hash platform-dependent.
    with destination.open("w", encoding="utf-8", newline="") as handle:
        for chunk in chunks:
            line = to_json_line(chunk) + "\n"
            handle.write(line)
            digest.update(line.encode("utf-8"))
            count += 1
    artifact_sha256 = digest.hexdigest()
    logger.info("wrote %d chunks to %s (sha256 %s)", count, destination, artifact_sha256)
    return count, artifact_sha256


def read_chunks_jsonl(source: Path) -> Iterator[Chunk]:
    logger.info("reading chunks from %s", source)
    with source.open(encoding="utf-8", newline="") as handle:
        for line in handle:
            if line.strip():
                yield Chunk.model_validate_json(line)


def build_chunk_manifest(
    chunks: Sequence[Chunk], corpus_version: str, artifact_sha256: str
) -> ChunkManifest:
    return ChunkManifest(
        doc_id=DOC_ID,
        corpus_version=corpus_version,
        chunk_count=len(chunks),
        root_count=sum(1 for chunk in chunks if chunk.is_root),
        stage_versions=dict(CHUNK_STAGE_VERSIONS),
        artifact_sha256=artifact_sha256,
    )


def write_chunk_manifest(manifest: ChunkManifest, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        manifest.model_dump(mode="json"), sort_keys=True, ensure_ascii=False, indent=2
    )
    with destination.open("w", encoding="utf-8", newline="") as handle:
        handle.write(payload + "\n")


def read_chunk_manifest(source: Path) -> ChunkManifest:
    return ChunkManifest.model_validate_json(source.read_text(encoding="utf-8"))


def load_chunks(
    directory: Path, *, corpus_version: str | None = None
) -> tuple[tuple[Chunk, ...], ChunkManifest]:
    """Read the store, refusing anything that does not describe this corpus."""
    chunks_path = directory / CHUNKS_FILENAME
    manifest_path = directory / CHUNK_MANIFEST_FILENAME
    for path in (chunks_path, manifest_path):
        if not path.exists():
            raise StaleChunkStoreError(
                f"missing {path} — run scripts/build_chunks.py"
            )

    manifest = read_chunk_manifest(manifest_path)
    if corpus_version is not None and manifest.corpus_version != corpus_version:
        raise StaleChunkStoreError(
            f"{chunks_path} was built for corpus_version {manifest.corpus_version}, "
            f"not {corpus_version}. Chunk ids are derived from corpus_version, so "
            f"these chunks are mislabelled, not merely old — rebuild the store."
        )
    if manifest.stage_versions != CHUNK_STAGE_VERSIONS:
        raise StaleChunkStoreError(
            f"{chunks_path} was built by chunking stages {manifest.stage_versions}, "
            f"not {CHUNK_STAGE_VERSIONS} — rebuild the store."
        )

    digest = hash_file(chunks_path)
    if digest != manifest.artifact_sha256:
        raise StaleChunkStoreError(
            f"{chunks_path} hashes to {digest}, manifest records "
            f"{manifest.artifact_sha256} — the store was truncated or edited."
        )

    chunks = tuple(read_chunks_jsonl(chunks_path))
    if len(chunks) != manifest.chunk_count:
        raise StaleChunkStoreError(
            f"{chunks_path} holds {len(chunks)} chunks, manifest claims "
            f"{manifest.chunk_count}."
        )
    return chunks, manifest
