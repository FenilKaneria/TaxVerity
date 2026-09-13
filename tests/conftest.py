import os
import uuid
from pathlib import Path

import numpy as np
import psycopg
import pytest
from psycopg import sql
from psycopg.conninfo import make_conninfo

from taxverity.chunking.chunker import build_chunks
from taxverity.chunking.models import Chunk
from taxverity.chunking.pipeline import read_corpus_version
from taxverity.chunking.store import ChunkManifest, load_chunks
from taxverity.config import ENV_PREFIX, MissingSettingError, Settings
from taxverity.corpus.crossrefs import extract_crossrefs
from taxverity.corpus.loader import read_pages_jsonl
from taxverity.corpus.nodes import NodeType
from taxverity.corpus.schedules import FIRST_SCHEDULE_PAGE, parse_schedules
from taxverity.corpus.sections import parse
from taxverity.corpus.substructure import candidate_table_pages, parse_substructure
from taxverity.corpus.tables import find_table_regions
from taxverity.db.migrate import migrate
from taxverity.embedding.backends import ModelInfo
from taxverity.embedding.store import load_vector_store
from taxverity.evals.gold import GOLD_V2_FILENAME, load_gold_set
from taxverity.evals.metrics import CitationIndex
from taxverity.evals.query_vectors import (
    QUERY_VECTORS_FILENAME,
    CachedQueryRetriever,
    load_query_vectors,
)
from taxverity.retrieval.bm25 import BM25Retriever
from taxverity.retrieval.citations import CitationRetriever
from taxverity.retrieval.dense import DenseRetriever


# Session-scoped, and autouse so it is ordered ahead of every other fixture in
# this scope: the session-scoped corpus extraction resolves its path during
# setup, which is too early for a function-scoped patch to have cleaned up.
@pytest.fixture(scope="session", autouse=True)
def isolated_environment():
    with pytest.MonkeyPatch.context() as patcher:
        for name in [key for key in os.environ if key.startswith(ENV_PREFIX)]:
            patcher.delenv(name, raising=False)
        yield


# --- shared corpus fixtures ---------------------------------------------------
# Session scope is per-file in pytest only for the *value*, not the definition:
# a fixture defined in a test module is re-evaluated for every module that
# defines its own copy. Four modules were each running their own pdfplumber
# geometry scan of the same pages; defined here, one scan serves all of them.

INTERIM = Path(__file__).resolve().parents[1] / "data" / "interim" / "pages.jsonl"


@pytest.fixture(scope="session")
def act_pages():
    if not INTERIM.exists():
        pytest.skip("run scripts/extract_corpus.py to build data/interim/pages.jsonl")
    return list(read_pages_jsonl(INTERIM))


@pytest.fixture(scope="session")
def act(act_pages):
    return parse(act_pages)


@pytest.fixture(scope="session")
def pdf_path():
    try:
        return Settings().resolve_corpus_pdf()
    except MissingSettingError:
        pytest.skip("corpus PDF not found — table geometry needs the real PDF")


@pytest.fixture(scope="session")
def section_table_regions(act, pdf_path):
    return find_table_regions(pdf_path, candidate_table_pages(act))


@pytest.fixture(scope="session")
def schedule_table_regions(act_pages, pdf_path):
    return find_table_regions(pdf_path, range(FIRST_SCHEDULE_PAGE, len(act_pages)))


@pytest.fixture(scope="session")
def sub(act, section_table_regions):
    return parse_substructure(act, table_regions=section_table_regions)


@pytest.fixture(scope="session")
def parsed_schedules(act_pages, schedule_table_regions):
    return parse_schedules(act_pages, table_regions=schedule_table_regions)


@pytest.fixture(scope="session")
def crossrefs(sub, parsed_schedules):
    return extract_crossrefs(sub.sections, parsed_schedules.schedules)


# Step 2.2's chunk set, shared by the chunker and store corpus suites so the
# whole parse pipeline runs once per session rather than once per module.
CHUNK_TEST_VERSION = "c" * 64


@pytest.fixture(scope="session")
def untrusted(sub, parsed_schedules):
    return set(sub.unreliable) | set(parsed_schedules.unreliable)


@pytest.fixture(scope="session")
def chunks(act, sub, parsed_schedules, crossrefs, untrusted):
    return build_chunks(
        CHUNK_TEST_VERSION,
        [*sub.sections, *parsed_schedules.schedules],
        chapters=act.chapters,
        crossrefs=crossrefs,
        untrusted=untrusted,
    )


# The Step 3.7 gold set, shared by its own suite and the Step 3.2 metrics suite.
@pytest.fixture(scope="session")
def gold():
    return load_gold_set(Settings().evals_dir / "datasets" / GOLD_V2_FILENAME)


# --- the real retrieval legs, no network ------------------------------------------
# Built from the stored chunk set, the Step 4.4 vector store and the Step 5.2
# cached question vectors. Shared by the hybrid and evidence-delivery suites so
# BM25 is indexed once.

VECTOR_STORE = Settings().vectors_dir / "jina-api"


class Offline:
    """The served identity with no way to embed: the corpus tests search by
    stored vector and must never reach the network."""

    def __init__(self, info) -> None:
        self._info = info

    def info(self):
        return self._info

    def embed(self, texts, kind):
        raise AssertionError("the corpus tests must not embed")


@pytest.fixture(scope="session")
def stored_chunks():
    """The Step 2.4 store as built, not the test-versioned `chunks` parse: the
    vector store's ids were derived from the real corpus_version."""
    interim = Settings().interim_dir
    if not (interim / "chunks.jsonl").exists():
        pytest.skip("run scripts/build_chunks.py to build the chunk store")
    corpus_version = read_corpus_version(interim / "corpus_manifest.json")
    stored, _ = load_chunks(interim, corpus_version=corpus_version)
    return corpus_version, stored


@pytest.fixture(scope="session")
def retrieval_legs(gold, stored_chunks):
    cache_path = VECTOR_STORE / QUERY_VECTORS_FILENAME
    if not (VECTOR_STORE / "vector_manifest.json").exists() or not cache_path.exists():
        pytest.skip("run scripts/embed_corpus.py, then scripts/measure_hybrid.py")
    corpus_version, stored = stored_chunks
    vectors, ids, manifest = load_vector_store(
        VECTOR_STORE, corpus_version=corpus_version
    )
    cached = load_query_vectors(
        cache_path, model=manifest.model, questions=[q.question for q in gold]
    )
    dense = DenseRetriever(stored, vectors, ids, manifest, Offline(manifest.model))
    return (
        CitationIndex(stored),
        CachedQueryRetriever(dense, cached.vectors),
        BM25Retriever(stored),
        CitationRetriever(stored),
    )


# --- database fixtures --------------------------------------------------------
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


def register_account(conn, email: str, password: str) -> uuid.UUID:
    """Test helper: drives the real `register()` flow (so its side effects,
    like a rate-limited email send, still happen) and then verifies the
    address directly in the database, the way clicking the emailed link
    would. Tests that only need a usable, logged-in-able account use this
    instead of asserting on the verification flow themselves."""
    from taxverity.auth.accounts import normalise_email, register
    from taxverity.mail.gmail import NullMailer

    register(
        conn, email, password, ip="127.0.0.1", mailer=NullMailer(), base_url="http://t"
    )
    address = normalise_email(email)
    (user_id,) = conn.execute(
        "SELECT user_id FROM users WHERE email = %s", (address,)
    ).fetchone()
    conn.execute(
        "UPDATE users SET email_verified_at = now() WHERE user_id = %s", (user_id,)
    )
    return user_id


# A hand-built four-chunk corpus with matching manifests: the Step 6.3 ingest
# and the Step 6.4 index both need rows in a database, and neither is testing
# the real corpus.
MICRO_V1 = "a" * 64
MICRO_V2 = "b" * 64
MICRO_DIM = 1024
MICRO_STAGE_VERSIONS = {"chunk_model": 1, "chunker": 1}
MICRO_MODEL = ModelInfo(
    model_id="test-model",
    dim=MICRO_DIM,
    revision="r1",
    runtime="test",
    encoding="test/v1",
)


def micro_chunks(version: str) -> tuple[Chunk, ...]:
    root = Chunk.create(
        version,
        "1",
        "1. Short title.\n(1) This Act may be called the Test Act.\n(2) It extends.",
        parent_id=None,
        doc_id="test-act",
        node_type=NodeType.SECTION,
        section_number="1",
        root_title="Short title",
        chapter_numeral="I",
        chapter_title="Preliminary",
        page_start=0,
        page_end=1,
        defined_terms=("tax", "person"),
        outgoing_refs=("2(1)", "Schedule I"),
    )
    first = Chunk.create(
        version,
        "1(1)",
        "(1) This Act may be called the Test Act.",
        parent_id=root.chunk_id,
        doc_id="test-act",
        node_type=NodeType.SUBSECTION,
        section_number="1",
        root_title="Short title",
        chapter_numeral="I",
        chapter_title="Preliminary",
        page_start=0,
        page_end=0,
        char_start=17,
        char_end=56,
        outgoing_refs=("2(1)",),
    )
    second = Chunk.create(
        version,
        "1(2)",
        "(2) It extends.",
        parent_id=root.chunk_id,
        doc_id="test-act",
        node_type=NodeType.SUBSECTION,
        section_number="1",
        root_title="Short title",
        page_start=1,
        page_end=1,
    )
    schedule = Chunk.create(
        version,
        "Schedule I",
        "SCHEDULE I\n1. Gold.",
        parent_id=None,
        doc_id="test-act",
        node_type=NodeType.SCHEDULE,
        schedule_number="I",
        page_start=2,
        page_end=2,
    )
    return root, first, second, schedule


def micro_chunk_manifest(chunks, version=MICRO_V1, **overrides) -> ChunkManifest:
    fields = {
        "doc_id": "test-act",
        "corpus_version": version,
        "chunk_count": len(chunks),
        "root_count": sum(1 for chunk in chunks if chunk.is_root),
        "stage_versions": dict(MICRO_STAGE_VERSIONS),
        "artifact_sha256": "1" * 64,
    }
    return ChunkManifest(**(fields | overrides))


def micro_unit_vectors(count: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vectors = rng.standard_normal((count, MICRO_DIM)).astype(np.float32)
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
