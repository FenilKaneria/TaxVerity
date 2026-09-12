"""Step 6.3 — load the chunk store and the vector store into Postgres."""

from __future__ import annotations

import sys
import time

import psycopg

from taxverity.chunking.pipeline import read_corpus_version
from taxverity.chunking.store import StaleChunkStoreError, load_chunks
from taxverity.config import MissingSettingError, Settings
from taxverity.db.ingest import IngestError, ingest_chunks, ingest_vectors
from taxverity.embedding.store import StaleVectorStoreError, load_vector_store
from taxverity.observability import configure_logging, get_logger

logger = get_logger(__name__)

VECTORS_KEY = "jina-api"


def _state(already_present: bool) -> str:
    return "already present" if already_present else "inserted"


def main() -> int:
    configure_logging()
    settings = Settings()
    started = time.perf_counter()

    try:
        url = settings.require("database_url")
        corpus_version = read_corpus_version(
            settings.interim_dir / "corpus_manifest.json"
        )
        chunks, chunk_manifest = load_chunks(
            settings.interim_dir, corpus_version=corpus_version
        )
        vectors, chunk_ids, vector_manifest = load_vector_store(
            settings.vectors_dir / VECTORS_KEY, corpus_version=corpus_version
        )
    except (MissingSettingError, StaleChunkStoreError, StaleVectorStoreError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    with psycopg.connect(url, autocommit=True) as conn:
        if conn.execute("SELECT to_regclass('corpus_versions')").fetchone() == (None,):
            print(
                "error: this database has no schema — run scripts/migrate.py first",
                file=sys.stderr,
            )
            return 1
        try:
            chunk_result = ingest_chunks(conn, chunks, chunk_manifest)
            vector_result = ingest_vectors(conn, vectors, chunk_ids, vector_manifest)
        except IngestError as error:
            print(f"error: {error}", file=sys.stderr)
            return 1

    elapsed = time.perf_counter() - started
    print(f"corpus_version  {corpus_version}")
    print(
        f"chunks          {chunk_result.chunk_count} {_state(chunk_result.already_present)}"
    )
    print(f"xref_edges      {chunk_result.edge_count}")
    print(
        f"embedding_set   {vector_result.embedding_set_id} "
        f"{_state(vector_result.already_present)}"
    )
    print(f"vectors         {vector_result.chunk_count} (dim {vector_manifest.dim})")
    print(f"vectors_sha256  {vector_manifest.vectors_sha256}")
    print(f"elapsed         {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
