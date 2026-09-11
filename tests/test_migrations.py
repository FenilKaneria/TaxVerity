import uuid

import numpy as np
import psycopg
import pytest
from psycopg import errors, sql
from psycopg.conninfo import make_conninfo

from taxverity.chunking.models import CHUNKABLE_TYPES
from taxverity.config import Settings
from taxverity.corpus.nodes import NodeType
from taxverity.db.migrate import (
    MIGRATIONS_DIR,
    MigrationError,
    discover_migrations,
    migrate,
)

V1 = "a" * 64
V2 = "b" * 64
DIM = 1024


def _write(directory, name, text):
    (directory / name).write_text(text, encoding="utf-8", newline="")


def test_shipped_migrations_run_from_one_without_gaps():
    versions = [m.version for m in discover_migrations()]
    assert versions == list(range(1, len(versions) + 1))
    assert versions


def test_first_migration_creates_the_vector_extension():
    assert "CREATE EXTENSION IF NOT EXISTS vector" in discover_migrations()[0].sql


def test_migrations_ship_inside_the_package():
    assert MIGRATIONS_DIR.parent.name == "db"
    assert MIGRATIONS_DIR.parent.parent.name == "taxverity"


def test_discovery_refuses_a_gap(tmp_path):
    _write(tmp_path, "0001_a.sql", "SELECT 1;")
    _write(tmp_path, "0003_c.sql", "SELECT 1;")
    with pytest.raises(MigrationError, match="no gap"):
        discover_migrations(tmp_path)


def test_discovery_refuses_a_duplicate_version(tmp_path):
    _write(tmp_path, "0001_a.sql", "SELECT 1;")
    _write(tmp_path, "0001_b.sql", "SELECT 1;")
    with pytest.raises(MigrationError, match="duplicate"):
        discover_migrations(tmp_path)


def test_discovery_refuses_a_misnamed_file(tmp_path):
    _write(tmp_path, "1_init.sql", "SELECT 1;")
    with pytest.raises(MigrationError, match="NNNN_name"):
        discover_migrations(tmp_path)


def test_hash_ignores_line_endings(tmp_path):
    lf, crlf = tmp_path / "lf", tmp_path / "crlf"
    lf.mkdir()
    crlf.mkdir()
    _write(lf, "0001_a.sql", "SELECT 1;\nSELECT 2;\n")
    _write(crlf, "0001_a.sql", "SELECT 1;\r\nSELECT 2;\r\n")
    assert discover_migrations(lf)[0].sha256 == discover_migrations(crlf)[0].sha256


# Every database test gets its own throwaway database, so the dev database is
# never touched and tests cannot see each other's rows.
@pytest.fixture(scope="module")
def admin_url():
    url = Settings().database_url
    if url is None:
        pytest.skip("TAXVERITY_DATABASE_URL not set; run `docker compose up -d`")
    return url.get_secret_value()


@pytest.fixture
def db(admin_url):
    name = f"taxverity_test_{uuid.uuid4().hex[:12]}"
    with psycopg.connect(admin_url, autocommit=True) as admin:
        admin.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
    try:
        url = make_conninfo(admin_url, dbname=name)
        with psycopg.connect(url, autocommit=True) as conn:
            yield conn
    finally:
        with psycopg.connect(admin_url, autocommit=True) as admin:
            admin.execute(
                sql.SQL("DROP DATABASE {} WITH (FORCE)").format(sql.Identifier(name))
            )


@pytest.fixture
def schema(db):
    migrate(db)
    return db


def test_migrate_applies_everything_once(db):
    shipped = discover_migrations()
    assert migrate(db) == tuple(m.version for m in shipped)
    assert migrate(db) == ()
    recorded = db.execute(
        "SELECT version, sha256 FROM schema_migrations ORDER BY version"
    ).fetchall()
    assert recorded == [(m.version, m.sha256) for m in shipped]


def test_a_later_migration_applies_on_top(db, tmp_path):
    _write(tmp_path, "0001_a.sql", "CREATE TABLE a (x int);")
    assert migrate(db, tmp_path) == (1,)
    _write(tmp_path, "0002_b.sql", "CREATE TABLE b (x int);")
    assert migrate(db, tmp_path) == (2,)


def test_an_edited_migration_is_refused(db, tmp_path):
    _write(tmp_path, "0001_a.sql", "CREATE TABLE a (x int);")
    migrate(db, tmp_path)
    _write(tmp_path, "0001_a.sql", "CREATE TABLE a (x bigint);")
    with pytest.raises(MigrationError, match="changed after it was applied"):
        migrate(db, tmp_path)


def test_a_database_ahead_of_the_code_is_refused(db, tmp_path):
    _write(tmp_path, "0001_a.sql", "CREATE TABLE a (x int);")
    _write(tmp_path, "0002_b.sql", "CREATE TABLE b (x int);")
    migrate(db, tmp_path)
    (tmp_path / "0002_b.sql").unlink()
    with pytest.raises(MigrationError, match="only knows up to 0001"):
        migrate(db, tmp_path)


def test_a_failed_migration_leaves_no_trace(db, tmp_path):
    _write(tmp_path, "0001_a.sql", "CREATE TABLE a (x int);\nSELECT 1 / 0;\n")
    with pytest.raises(errors.DivisionByZero):
        migrate(db, tmp_path)
    assert db.execute("SELECT to_regclass('a')").fetchone() == (None,)
    assert db.execute("SELECT count(*) FROM schema_migrations").fetchone() == (0,)


def test_migrate_refuses_a_connection_in_a_transaction(admin_url):
    with psycopg.connect(admin_url) as conn:
        with pytest.raises(ValueError, match="autocommit"):
            migrate(conn)


def _corpus(conn, version):
    conn.execute(
        "INSERT INTO corpus_versions (corpus_version, doc_id, chunk_count, "
        "root_count, chunk_stage_versions, chunks_sha256) "
        "VALUES (%s, 'doc', 1, 1, '{}', %s)",
        (version, "0" * 64),
    )


def _chunk(
    conn,
    version,
    number,
    node_path,
    *,
    parent=None,
    node_type="section",
    section="1",
    schedule=None,
):
    conn.execute(
        "INSERT INTO chunks (chunk_id, corpus_version, ordinal, parent_id, doc_id, "
        "node_type, node_path, section_number, schedule_number, text, page_start, "
        "page_end) VALUES (%s, %s, %s, %s, 'doc', %s, %s, %s, %s, 'text', 0, 0)",
        (
            f"{number:016x}",
            version,
            number,
            None if parent is None else f"{parent:016x}",
            node_type,
            node_path,
            section,
            schedule,
        ),
    )


def _embedding_set(conn, version, kind="document"):
    return conn.execute(
        "INSERT INTO embedding_sets (corpus_version, model_id, dim, revision, "
        "runtime, encoding, kind, device, chunk_count, vectors_sha256, ids_sha256, "
        "probe_set_version, probes) VALUES (%s, 'm', 1024, 'r', 'rt', 'e', %s, "
        "'d', 1, %s, %s, 1, '[]') RETURNING embedding_set_id",
        (version, kind, "1" * 64, "2" * 64),
    ).fetchone()[0]


def _literal(vector):
    # float() of a float32 is exact, and pgvector parses to the nearest float32.
    return "[" + ",".join(repr(float(x)) for x in vector) + "]"


def _embedding(conn, set_id, version, number, vector):
    conn.execute(
        "INSERT INTO chunk_embeddings (embedding_set_id, corpus_version, chunk_id, "
        "embedding) VALUES (%s, %s, %s, %s::vector)",
        (set_id, version, f"{number:016x}", _literal(vector)),
    )


def test_a_citation_names_one_chunk_per_corpus(schema):
    _corpus(schema, V1)
    _corpus(schema, V2)
    _chunk(schema, V1, 1, "1")
    _chunk(schema, V2, 2, "1")
    with pytest.raises(errors.UniqueViolation):
        _chunk(schema, V1, 3, "1")


def test_a_parent_must_belong_to_the_same_corpus(schema):
    _corpus(schema, V1)
    _corpus(schema, V2)
    _chunk(schema, V1, 1, "1")
    with pytest.raises(errors.ForeignKeyViolation):
        with schema.transaction():
            _chunk(schema, V2, 2, "1(1)", parent=1, node_type="subsection")


def test_a_child_may_be_inserted_before_its_parent(schema):
    _corpus(schema, V1)
    with schema.transaction():
        _chunk(schema, V1, 2, "1(1)", parent=1, node_type="subsection")
        _chunk(schema, V1, 1, "1")
    assert schema.execute("SELECT count(*) FROM chunks").fetchone() == (2,)


def test_only_chunkable_node_types_are_storable(schema):
    _corpus(schema, V1)
    for number, node_type in enumerate(NodeType, start=1):
        try:
            _chunk(schema, V1, number, str(number), node_type=node_type.value)
        except errors.CheckViolation:
            assert node_type not in CHUNKABLE_TYPES
        else:
            assert node_type in CHUNKABLE_TYPES


def test_a_chunk_has_exactly_one_root(schema):
    _corpus(schema, V1)
    with pytest.raises(errors.CheckViolation):
        _chunk(schema, V1, 1, "1", schedule="I")


def test_only_a_document_index_is_storable(schema):
    _corpus(schema, V1)
    with pytest.raises(errors.CheckViolation):
        _embedding_set(schema, V1, kind="query")


def test_a_vector_cannot_attach_to_another_corpus(schema):
    _corpus(schema, V1)
    _corpus(schema, V2)
    _chunk(schema, V2, 1, "1")
    set_id = _embedding_set(schema, V1)
    with pytest.raises(errors.ForeignKeyViolation):
        _embedding(schema, set_id, V2, 1, np.zeros(DIM, dtype=np.float32))


def test_the_vector_width_is_fixed(schema):
    _corpus(schema, V1)
    _chunk(schema, V1, 1, "1")
    set_id = _embedding_set(schema, V1)
    with pytest.raises(errors.DataException):
        _embedding(schema, set_id, V1, 1, np.zeros(3, dtype=np.float32))


# 6.5's parity check compares against the NumPy float32 index, so the column
# must hold float32 exactly, which vector does and halfvec would not.
def test_a_float32_vector_round_trips_exactly(schema):
    _corpus(schema, V1)
    _chunk(schema, V1, 1, "1")
    set_id = _embedding_set(schema, V1)
    vector = np.random.default_rng(0).standard_normal(DIM).astype(np.float32)
    vector /= np.linalg.norm(vector)
    _embedding(schema, set_id, V1, 1, vector)
    text = schema.execute("SELECT embedding::text FROM chunk_embeddings").fetchone()[0]
    stored = np.array([float(x) for x in text.strip("[]").split(",")], np.float32)
    assert np.array_equal(stored, vector)


def test_deleting_a_corpus_version_removes_everything_under_it(schema):
    _corpus(schema, V1)
    _chunk(schema, V1, 1, "1")
    schema.execute(
        "INSERT INTO xref_edges (chunk_id, position, target_path) VALUES (%s, 0, '2')",
        (f"{1:016x}",),
    )
    _embedding(schema, _embedding_set(schema, V1), V1, 1, np.ones(DIM, np.float32))
    schema.execute("DELETE FROM corpus_versions WHERE corpus_version = %s", (V1,))
    for table in ("chunks", "xref_edges", "embedding_sets", "chunk_embeddings"):
        count = schema.execute(
            sql.SQL("SELECT count(*) FROM {}").format(sql.Identifier(table))
        ).fetchone()
        assert count == (0,), table
