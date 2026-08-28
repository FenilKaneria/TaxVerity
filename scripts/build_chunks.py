"""Step 2.4 — build the chunk store from the extracted corpus."""

from __future__ import annotations

import sys
import time
from pathlib import Path

from taxverity.chunking.pipeline import build_from_corpus, read_corpus_version
from taxverity.chunking.store import (
    CHUNK_MANIFEST_FILENAME,
    CHUNKS_FILENAME,
    build_chunk_manifest,
    write_chunk_manifest,
    write_chunks_jsonl,
)
from taxverity.config import Settings
from taxverity.observability import configure_logging, get_logger

logger = get_logger(__name__)


def main() -> int:
    configure_logging()
    settings = Settings()
    pages = settings.interim_dir / "pages.jsonl"
    if not pages.exists():
        logger.error("missing %s — run scripts/extract_corpus.py first", pages)
        return 1

    started = time.perf_counter()
    corpus_version = read_corpus_version(settings.interim_dir / "corpus_manifest.json")
    build = build_from_corpus(pages, settings.resolve_corpus_pdf(), corpus_version)

    destination = settings.interim_dir / CHUNKS_FILENAME
    count, artifact_sha256 = write_chunks_jsonl(build.chunks, destination)
    manifest = build_chunk_manifest(build.chunks, corpus_version, artifact_sha256)
    write_chunk_manifest(manifest, settings.interim_dir / CHUNK_MANIFEST_FILENAME)
    elapsed = time.perf_counter() - started

    size = Path(destination).stat().st_size
    print(f"chunks          {count} ({manifest.root_count} roots)")
    print(f"corpus_version  {corpus_version}")
    print(f"artifact        {destination} ({size / 1024 / 1024:.1f} MiB)")
    print(f"sha256          {artifact_sha256}")
    print(f"elapsed         {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
