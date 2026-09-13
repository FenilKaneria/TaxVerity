"""Step 6.6 — the startup guard.

Against a throwaway database, like the Step 6.3 and 6.4 suites: what is being
tested is whether the database's own state is read correctly, and a fake cannot
be wrong in the ways a database is.
"""

from __future__ import annotations

import pytest

from conftest import MICRO_V1 as V1
from conftest import MICRO_V2 as V2
from conftest import micro_chunk_manifest, micro_chunks
from taxverity.chunking.store import CHUNK_STAGE_VERSIONS
from taxverity.config import MissingSettingError, Settings
from taxverity.db.ingest import ingest_chunks, ingest_vectors
from taxverity.db.serving import (
    ServingRefusal,
    resolve_serving,
    serving_target,
)
from test_pgvector import GEOMETRY, vector_manifest


def load(conn, version=V1):
    chunks = micro_chunks(version)
    ingest_chunks(conn, chunks, micro_chunk_manifest(chunks, version=version))
    manifest = vector_manifest(chunks, corpus_version=version)
    result = ingest_vectors(
        conn, GEOMETRY, [chunk.chunk_id for chunk in chunks], manifest
    )
    return chunks, result.embedding_set_id


# --- configuration ------------------------------------------------------------


def test_serving_target_names_the_env_var_it_is_missing():
    settings = Settings(serving_corpus_version=None, serving_embedding_set_id=1)
    with pytest.raises(MissingSettingError, match="TAXVERITY_SERVING_CORPUS_VERSION"):
        serving_target(settings)


def test_serving_target_returns_both_when_configured():
    settings = Settings(serving_corpus_version=V1, serving_embedding_set_id=7)
    assert serving_target(settings) == (V1, 7)


# --- against a throwaway database ---------------------------------------------


def test_a_loaded_database_resolves_to_what_it_holds(schema):
    chunks, set_id = load(schema)
    served = resolve_serving(schema, corpus_version=V1, embedding_set_id=set_id)
    assert served.corpus_version == V1
    assert served.doc_id == "test-act"
    assert served.chunk_count == len(chunks)
    assert served.vector_count == len(chunks)
    assert served.embedding_set_id == set_id
    assert served.model.dim == 1024


def test_an_unmigrated_database_is_refused_before_any_table_is_read(db):
    # The tables do not exist yet, so this also pins that the schema check runs
    # first: anything else would raise UndefinedTable instead of refusing.
    with pytest.raises(ServingRefusal) as refusal:
        resolve_serving(db, corpus_version=V1, embedding_set_id=1)
    assert refusal.value.reasons == (
        "schema is behind this code: migration(s) 0001, 0002 unapplied — "
        "run scripts/migrate.py",
    )


def test_a_migration_edited_after_it_was_applied_is_refused(schema):
    schema.execute("UPDATE schema_migrations SET sha256 = %s", ("f" * 64,))
    with pytest.raises(ServingRefusal, match="changed after it was applied"):
        resolve_serving(schema, corpus_version=V1, embedding_set_id=1)


def test_an_absent_corpus_is_refused_and_the_message_says_what_is_held(schema):
    _, set_id = load(schema, version=V2)
    with pytest.raises(ServingRefusal) as refusal:
        resolve_serving(schema, corpus_version=V1, embedding_set_id=set_id)
    assert "is not in this database" in refusal.value.reasons[0]
    assert V2[:16] in refusal.value.reasons[0]


def test_an_embedding_set_built_for_another_corpus_is_refused(schema):
    load(schema, version=V1)
    _, other = load(schema, version=V2)
    with pytest.raises(ServingRefusal) as refusal:
        resolve_serving(schema, corpus_version=V1, embedding_set_id=other)
    assert refusal.value.reasons == (
        f"embedding set {other} was built for corpus_version {V2[:16]}…, "
        f"not the configured {V1[:16]}…",
    )


def test_an_absent_embedding_set_is_refused(schema):
    load(schema)
    with pytest.raises(ServingRefusal, match="holds no embedding set 999"):
        resolve_serving(schema, corpus_version=V1, embedding_set_id=999)


def test_chunks_ingested_by_other_chunking_code_are_refused(schema):
    chunks = micro_chunks(V1)
    stale = dict(CHUNK_STAGE_VERSIONS) | {"chunker": 99}
    ingest_chunks(schema, chunks, micro_chunk_manifest(chunks, stage_versions=stale))
    with pytest.raises(ServingRefusal, match="chunk stage versions"):
        resolve_serving(schema, corpus_version=V1, embedding_set_id=1)


def test_every_refusal_is_reported_at_once_not_one_per_deploy(schema):
    with pytest.raises(ServingRefusal) as refusal:
        resolve_serving(schema, corpus_version=V1, embedding_set_id=4)
    assert len(refusal.value.reasons) == 2
    assert "is not in this database" in refusal.value.reasons[0]
    assert "holds no embedding set 4" in refusal.value.reasons[1]
