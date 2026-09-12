import numpy as np
import psycopg
import pytest
from psycopg import errors

from conftest import (
    MICRO_DIM as DIM,
)
from conftest import (
    MICRO_MODEL as MODEL,
)
from conftest import (
    MICRO_V1 as V1,
)
from conftest import (
    MICRO_V2 as V2,
)
from conftest import micro_chunk_manifest as chunk_manifest
from conftest import micro_chunks
from conftest import micro_unit_vectors as unit_vectors
from taxverity.chunking.models import Chunk
from taxverity.chunking.store import ChunkManifest
from taxverity.corpus.nodes import NodeType
from taxverity.db.ingest import (
    CHUNK_COLUMNS,
    ChunkIngest,
    IngestError,
    _check_existing_corpus,
    _chunk_row,
    _xref_rows,
    ingest_chunks,
    ingest_vectors,
    vector_literal,
)
from taxverity.embedding.store import ProbeVector, VectorManifest


def vector_manifest(chunks, version=V1, **overrides) -> VectorManifest:
    fields = {
        "store_version": 2,
        "doc_id": "test-act",
        "corpus_version": version,
        "model": MODEL,
        "kind": "document",
        "device": "test",
        "dim": DIM,
        "chunk_count": len(chunks),
        "vectors_sha256": "2" * 64,
        "ids_sha256": "3" * 64,
        "probe_set_version": 1,
        "probes": (
            ProbeVector(kind="document", text="probe", vector=tuple([0.0] * DIM)),
        ),
    }
    return VectorManifest(**(fields | overrides))


def stored_corpus_row(manifest: ChunkManifest) -> dict:
    return {
        "doc_id": manifest.doc_id,
        "chunk_count": manifest.chunk_count,
        "root_count": manifest.root_count,
        "chunks_sha256": manifest.artifact_sha256,
        "chunk_stage_versions": dict(manifest.stage_versions),
    }


def test_chunk_row_matches_the_column_list():
    chunks = micro_chunks(V1)
    row = _chunk_row(chunks[1], 7)
    assert len(row) == len(CHUNK_COLUMNS)
    columns = dict(zip(CHUNK_COLUMNS, row, strict=True))
    assert columns["chunk_id"] == chunks[1].chunk_id
    assert columns["ordinal"] == 7
    assert columns["parent_id"] == chunks[0].chunk_id
    assert columns["node_type"] == "subsection"
    assert columns["defined_terms"] == []
    assert columns["schedule_number"] is None
    assert columns["token_count"] is None
    assert columns["char_start"] == 17


def test_xref_rows_preserve_outgoing_ref_order():
    root = micro_chunks(V1)[0]
    assert list(_xref_rows(root)) == [
        (root.chunk_id, 0, "2(1)"),
        (root.chunk_id, 1, "Schedule I"),
    ]


def test_a_vector_literal_round_trips_float32_exactly():
    vector = unit_vectors(1, seed=3)[0]
    text = vector_literal(vector)
    assert np.array_equal(
        np.array([float(x) for x in text.strip("[]").split(",")], np.float32), vector
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("doc_id", "other-act"),
        ("chunk_count", 3),
        ("root_count", 1),
        ("chunks_sha256", "9" * 64),
        ("chunk_stage_versions", {"chunk_model": 1, "chunker": 2}),
    ],
)
def test_a_changed_manifest_field_is_refused_by_name(field, value):
    chunks = micro_chunks(V1)
    manifest = chunk_manifest(chunks)
    stored = stored_corpus_row(manifest) | {field: value}
    with pytest.raises(IngestError, match=field):
        _check_existing_corpus(stored, manifest)


def test_an_unchanged_manifest_passes_the_check():
    manifest = chunk_manifest(micro_chunks(V1))
    _check_existing_corpus(stored_corpus_row(manifest), manifest)


def test_ingest_refuses_a_connection_in_a_transaction(admin_url):
    with psycopg.connect(admin_url) as conn:
        with pytest.raises(ValueError, match="autocommit"):
            ingest_chunks(conn, (), chunk_manifest(micro_chunks(V1)))


def test_ingest_writes_chunks_and_edges(schema):
    chunks = micro_chunks(V1)
    result = ingest_chunks(schema, chunks, chunk_manifest(chunks))
    assert result == ChunkIngest(V1, 4, 3, already_present=False)

    rows = schema.execute(
        "SELECT ordinal, node_path FROM chunks WHERE corpus_version = %s "
        "ORDER BY ordinal",
        (V1,),
    ).fetchall()
    assert rows == [(0, "1"), (1, "1(1)"), (2, "1(2)"), (3, "Schedule I")]

    stored = schema.execute(
        "SELECT parent_id, node_type, root_title, chapter_numeral, text, page_start, "
        "page_end, char_start, char_end, defined_terms, token_count, section_number, "
        "schedule_number FROM chunks WHERE chunk_id = %s",
        (chunks[0].chunk_id,),
    ).fetchone()
    assert stored == (
        None,
        "section",
        "Short title",
        "I",
        chunks[0].text,
        0,
        1,
        None,
        None,
        ["tax", "person"],
        None,
        "1",
        None,
    )

    edges = schema.execute(
        "SELECT position, target_path FROM xref_edges WHERE chunk_id = %s "
        "ORDER BY position",
        (chunks[0].chunk_id,),
    ).fetchall()
    assert edges == [(0, "2(1)"), (1, "Schedule I")]


def test_a_child_may_be_ingested_before_its_parent(schema):
    chunks = micro_chunks(V1)
    reordered = (chunks[1], chunks[0], chunks[2], chunks[3])
    ingest_chunks(schema, reordered, chunk_manifest(reordered))
    assert schema.execute(
        "SELECT ordinal FROM chunks WHERE chunk_id = %s", (chunks[1].chunk_id,)
    ).fetchone() == (0,)


def test_re_ingesting_the_same_corpus_is_a_no_op(schema):
    chunks = micro_chunks(V1)
    manifest = chunk_manifest(chunks)
    ingest_chunks(schema, chunks, manifest)
    ingested_at = schema.execute("SELECT ingested_at FROM corpus_versions").fetchone()

    result = ingest_chunks(schema, chunks, manifest)

    assert result == ChunkIngest(V1, 4, 3, already_present=True)
    assert schema.execute("SELECT count(*) FROM chunks").fetchone() == (4,)
    assert schema.execute("SELECT count(*) FROM xref_edges").fetchone() == (3,)
    assert schema.execute("SELECT ingested_at FROM corpus_versions").fetchone() == (
        ingested_at
    )


def test_a_changed_store_under_the_same_version_is_refused(schema):
    chunks = micro_chunks(V1)
    ingest_chunks(schema, chunks, chunk_manifest(chunks))
    with pytest.raises(IngestError, match="chunks_sha256"):
        ingest_chunks(schema, chunks, chunk_manifest(chunks, artifact_sha256="9" * 64))
    assert schema.execute("SELECT count(*) FROM chunks").fetchone() == (4,)


def test_a_partially_deleted_corpus_is_refused(schema):
    chunks = micro_chunks(V1)
    manifest = chunk_manifest(chunks)
    ingest_chunks(schema, chunks, manifest)
    schema.execute("DELETE FROM chunks WHERE node_path = 'Schedule I'")
    with pytest.raises(IngestError, match="partially deleted"):
        ingest_chunks(schema, chunks, manifest)


def test_two_corpus_versions_coexist(schema):
    for version in (V1, V2):
        chunks = micro_chunks(version)
        ingest_chunks(schema, chunks, chunk_manifest(chunks, version=version))
    counts = schema.execute(
        "SELECT corpus_version, count(*) FROM chunks GROUP BY corpus_version "
        "ORDER BY corpus_version"
    ).fetchall()
    assert counts == [(V1, 4), (V2, 4)]


def test_a_failed_ingest_leaves_no_rows(schema):
    chunks = micro_chunks(V1)
    orphan = Chunk.create(
        V1,
        "1(3)",
        "(3) Orphaned.",
        parent_id="f" * 16,
        doc_id="test-act",
        node_type=NodeType.SUBSECTION,
        section_number="1",
        page_start=1,
        page_end=1,
    )
    broken = (*chunks, orphan)
    # The parent foreign key is deferred, so this fails at COMMIT, not at COPY.
    with pytest.raises(errors.ForeignKeyViolation):
        ingest_chunks(schema, broken, chunk_manifest(broken))
    assert schema.execute("SELECT count(*) FROM chunks").fetchone() == (0,)
    assert schema.execute("SELECT count(*) FROM corpus_versions").fetchone() == (0,)


def test_vectors_without_chunks_are_refused(schema):
    chunks = micro_chunks(V1)
    with pytest.raises(IngestError, match="no chunks ingested"):
        ingest_vectors(
            schema,
            unit_vectors(len(chunks)),
            [chunk.chunk_id for chunk in chunks],
            vector_manifest(chunks),
        )
    assert schema.execute("SELECT count(*) FROM embedding_sets").fetchone() == (0,)


def test_a_vector_store_shorter_than_the_corpus_is_refused(schema):
    chunks = micro_chunks(V1)
    ingest_chunks(schema, chunks, chunk_manifest(chunks))
    partial = chunks[:-1]
    with pytest.raises(IngestError, match="partial index"):
        ingest_vectors(
            schema,
            unit_vectors(len(partial)),
            [chunk.chunk_id for chunk in partial],
            vector_manifest(partial),
        )
    assert schema.execute("SELECT count(*) FROM embedding_sets").fetchone() == (0,)


def test_a_query_encoded_store_is_refused(schema):
    chunks = micro_chunks(V1)
    ingest_chunks(schema, chunks, chunk_manifest(chunks))
    with pytest.raises(IngestError, match="document index"):
        ingest_vectors(
            schema,
            unit_vectors(len(chunks)),
            [chunk.chunk_id for chunk in chunks],
            vector_manifest(chunks, kind="query"),
        )


def test_ingested_vectors_read_back_exactly(schema):
    chunks = micro_chunks(V1)
    ingest_chunks(schema, chunks, chunk_manifest(chunks))
    vectors = unit_vectors(len(chunks), seed=5)
    chunk_ids = [chunk.chunk_id for chunk in chunks]

    result = ingest_vectors(schema, vectors, chunk_ids, vector_manifest(chunks))

    assert result.already_present is False
    assert result.chunk_count == 4
    for chunk_id, vector in zip(chunk_ids, vectors, strict=True):
        text = schema.execute(
            "SELECT embedding::text FROM chunk_embeddings WHERE chunk_id = %s",
            (chunk_id,),
        ).fetchone()[0]
        stored = np.array([float(x) for x in text.strip("[]").split(",")], np.float32)
        assert np.array_equal(stored, vector)


def test_re_ingesting_the_same_vector_store_is_a_no_op(schema):
    chunks = micro_chunks(V1)
    ingest_chunks(schema, chunks, chunk_manifest(chunks))
    vectors = unit_vectors(len(chunks))
    chunk_ids = [chunk.chunk_id for chunk in chunks]
    manifest = vector_manifest(chunks)
    first = ingest_vectors(schema, vectors, chunk_ids, manifest)

    second = ingest_vectors(schema, vectors, chunk_ids, manifest)

    assert second.embedding_set_id == first.embedding_set_id
    assert second.already_present is True
    assert schema.execute("SELECT count(*) FROM embedding_sets").fetchone() == (1,)
    assert schema.execute("SELECT count(*) FROM chunk_embeddings").fetchone() == (4,)


def test_a_second_embedding_set_coexists_with_the_first(schema):
    chunks = micro_chunks(V1)
    ingest_chunks(schema, chunks, chunk_manifest(chunks))
    chunk_ids = [chunk.chunk_id for chunk in chunks]
    first = ingest_vectors(
        schema, unit_vectors(len(chunks), seed=1), chunk_ids, vector_manifest(chunks)
    )
    second = ingest_vectors(
        schema,
        unit_vectors(len(chunks), seed=2),
        chunk_ids,
        vector_manifest(chunks, vectors_sha256="4" * 64),
    )

    assert second.embedding_set_id != first.embedding_set_id
    assert schema.execute("SELECT count(*) FROM embedding_sets").fetchone() == (2,)
    assert schema.execute(
        "SELECT count(*) FROM chunk_embeddings WHERE chunk_id = %s", (chunk_ids[0],)
    ).fetchone() == (2,)


def test_a_changed_identity_under_the_same_vectors_is_refused(schema):
    chunks = micro_chunks(V1)
    ingest_chunks(schema, chunks, chunk_manifest(chunks))
    chunk_ids = [chunk.chunk_id for chunk in chunks]
    vectors = unit_vectors(len(chunks))
    ingest_vectors(schema, vectors, chunk_ids, vector_manifest(chunks))
    with pytest.raises(IngestError, match="ids_sha256"):
        ingest_vectors(
            schema, vectors, chunk_ids, vector_manifest(chunks, ids_sha256="5" * 64)
        )
