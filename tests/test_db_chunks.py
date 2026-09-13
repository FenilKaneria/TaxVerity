"""Phase 14 — `load_chunks_from_db` against a throwaway database, reusing the
Step 6.3/6.4 micro-corpus fixtures rather than a real corpus."""

from __future__ import annotations

from conftest import MICRO_V1, micro_chunk_manifest, micro_chunks
from taxverity.db.chunks import load_chunks_from_db
from taxverity.db.ingest import ingest_chunks


def test_every_micro_chunk_comes_back_in_corpus_order(schema):
    chunks = micro_chunks(MICRO_V1)
    ingest_chunks(schema, chunks, micro_chunk_manifest(chunks))

    loaded = load_chunks_from_db(schema, MICRO_V1)

    assert [chunk.node_path for chunk in loaded] == [chunk.node_path for chunk in chunks]
    assert [chunk.text for chunk in loaded] == [chunk.text for chunk in chunks]


def test_outgoing_refs_round_trip(schema):
    chunks = micro_chunks(MICRO_V1)
    ingest_chunks(schema, chunks, micro_chunk_manifest(chunks))

    loaded = {chunk.node_path: chunk for chunk in load_chunks_from_db(schema, MICRO_V1)}

    for chunk in chunks:
        assert loaded[chunk.node_path].outgoing_refs == chunk.outgoing_refs


def test_an_uningested_corpus_version_returns_nothing(schema):
    assert load_chunks_from_db(schema, "z" * 64) == []
