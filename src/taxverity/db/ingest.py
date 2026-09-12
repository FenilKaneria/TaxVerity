"""Step 6.3 — load the chunk store and the vector store into Postgres (ADR-089).

Idempotent, keyed on `corpus_version`. A chunk id is derived from
`corpus_version`, so two different chunk sets claiming one version are
mislabelled rather than stale: this module inserts, no-ops, or refuses, and
never overwrites. Recovery from a refusal is an operator's explicit
`DELETE FROM corpus_versions`, which cascades.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Json

from taxverity.chunking.models import Chunk
from taxverity.chunking.store import ChunkManifest
from taxverity.embedding.backends import EmbedKind
from taxverity.embedding.store import VectorManifest
from taxverity.observability import get_logger

logger = get_logger(__name__)

INGEST_STAGE_VERSION = 1

CHUNK_COLUMNS = (
    "chunk_id",
    "corpus_version",
    "ordinal",
    "parent_id",
    "doc_id",
    "node_type",
    "node_path",
    "section_number",
    "schedule_number",
    "root_title",
    "chapter_numeral",
    "chapter_title",
    "text",
    "page_start",
    "page_end",
    "char_start",
    "char_end",
    "defined_terms",
    "token_count",
)
XREF_COLUMNS = ("chunk_id", "position", "target_path")
EMBEDDING_COLUMNS = ("embedding_set_id", "corpus_version", "chunk_id", "embedding")

# The fields a stored row must still agree with before an ingest is a no-op.
CORPUS_IDENTITY = ("doc_id", "chunk_count", "root_count", "chunks_sha256")
SET_IDENTITY = (
    "ids_sha256",
    "model_id",
    "revision",
    "runtime",
    "encoding",
    "device",
    "dim",
    "chunk_count",
    "probe_set_version",
)


class IngestError(RuntimeError):
    pass


@dataclass(frozen=True)
class ChunkIngest:
    corpus_version: str
    chunk_count: int
    edge_count: int
    already_present: bool


@dataclass(frozen=True)
class VectorIngest:
    embedding_set_id: int
    corpus_version: str
    chunk_count: int
    already_present: bool


def _require_autocommit(conn: psycopg.Connection) -> None:
    # On a connection already in a transaction, conn.transaction() opens a
    # savepoint instead — and the deferred parent foreign key is checked at
    # COMMIT, which releasing a savepoint is not.
    if not conn.autocommit:
        raise ValueError(
            "ingest needs an autocommit connection: it commits the chunk load "
            "and the vector load in separate transactions"
        )


def _chunk_row(chunk: Chunk, ordinal: int) -> tuple:
    return (
        chunk.chunk_id,
        chunk.corpus_version,
        ordinal,
        chunk.parent_id,
        chunk.doc_id,
        chunk.node_type.value,
        chunk.node_path,
        chunk.section_number,
        chunk.schedule_number,
        chunk.root_title,
        chunk.chapter_numeral,
        chunk.chapter_title,
        chunk.text,
        chunk.page_start,
        chunk.page_end,
        chunk.char_start,
        chunk.char_end,
        list(chunk.defined_terms),
        chunk.token_count,
    )


def _xref_rows(chunk: Chunk) -> Iterator[tuple[str, int, str]]:
    for position, target in enumerate(chunk.outgoing_refs):
        yield chunk.chunk_id, position, target


def vector_literal(vector: Sequence[float]) -> str:
    # 9 significant digits is FLT_DECIMAL_DIG: every float32 round-trips exactly,
    # which is what Step 6.5 compares against the NumPy index. Do not shorten it.
    return "[" + ",".join(f"{float(value):.9g}" for value in vector) + "]"


def _refuse_difference(label: str, field: str, stored: object, given: object) -> None:
    raise IngestError(
        f"{label} differs on {field}: the database holds {stored!r}, the store "
        f"being ingested says {given!r}. Nothing was written. To replace it, "
        f"delete the corpus version and re-run."
    )


def _check_existing_corpus(stored: dict, manifest: ChunkManifest) -> None:
    given = {
        "doc_id": manifest.doc_id,
        "chunk_count": manifest.chunk_count,
        "root_count": manifest.root_count,
        "chunks_sha256": manifest.artifact_sha256,
    }
    label = f"corpus_version {manifest.corpus_version}"
    for field in CORPUS_IDENTITY:
        if stored[field] != given[field]:
            _refuse_difference(label, field, stored[field], given[field])
    # jsonb normalises key order, so this is a dict comparison, never a text one.
    if stored["chunk_stage_versions"] != manifest.stage_versions:
        _refuse_difference(
            label,
            "chunk_stage_versions",
            stored["chunk_stage_versions"],
            manifest.stage_versions,
        )


def _check_existing_set(stored: dict, manifest: VectorManifest) -> None:
    given = {
        "ids_sha256": manifest.ids_sha256,
        "model_id": manifest.model.model_id,
        "revision": manifest.model.revision,
        "runtime": manifest.model.runtime,
        "encoding": manifest.model.encoding,
        "device": manifest.device,
        "dim": manifest.dim,
        "chunk_count": manifest.chunk_count,
        "probe_set_version": manifest.probe_set_version,
    }
    label = f"embedding set {manifest.vectors_sha256[:16]}"
    for field in SET_IDENTITY:
        if stored[field] != given[field]:
            _refuse_difference(label, field, stored[field], given[field])


def ingest_chunks(
    conn: psycopg.Connection, chunks: Iterable[Chunk], manifest: ChunkManifest
) -> ChunkIngest:
    """Load chunks and their cross-reference edges. A second call is a no-op."""
    _require_autocommit(conn)
    chunks = tuple(chunks)
    version = manifest.corpus_version

    with conn.transaction():
        stored = (
            conn.cursor(row_factory=dict_row)
            .execute(
                "SELECT doc_id, chunk_count, root_count, chunk_stage_versions, "
                "chunks_sha256 FROM corpus_versions WHERE corpus_version = %s",
                (version,),
            )
            .fetchone()
        )
        if stored is not None:
            _check_existing_corpus(stored, manifest)
            present = conn.execute(
                "SELECT count(*) FROM chunks WHERE corpus_version = %s", (version,)
            ).fetchone()[0]
            if present != manifest.chunk_count:
                raise IngestError(
                    f"corpus_version {version} claims {manifest.chunk_count} chunks "
                    f"but the table holds {present}: it is partially deleted. "
                    f"DELETE FROM corpus_versions WHERE corpus_version = "
                    f"'{version}' and re-run."
                )
            edges = conn.execute(
                "SELECT count(*) FROM xref_edges e JOIN chunks c USING (chunk_id) "
                "WHERE c.corpus_version = %s",
                (version,),
            ).fetchone()[0]
            logger.info(
                "corpus_version %s is already ingested (%d chunks)", version, present
            )
            return ChunkIngest(version, present, edges, already_present=True)

        conn.execute(
            "INSERT INTO corpus_versions (corpus_version, doc_id, chunk_count, "
            "root_count, chunk_stage_versions, chunks_sha256) "
            "VALUES (%s, %s, %s, %s, %s, %s)",
            (
                version,
                manifest.doc_id,
                manifest.chunk_count,
                manifest.root_count,
                Json(manifest.stage_versions),
                manifest.artifact_sha256,
            ),
        )
        # Store order is ordinal order, and the parent foreign key is deferred to
        # COMMIT, so parents and children stream interleaved with no sort.
        with conn.cursor().copy(
            f"COPY chunks ({', '.join(CHUNK_COLUMNS)}) FROM STDIN"
        ) as copy:
            for ordinal, chunk in enumerate(chunks):
                copy.write_row(_chunk_row(chunk, ordinal))

        edge_count = 0
        with conn.cursor().copy(
            f"COPY xref_edges ({', '.join(XREF_COLUMNS)}) FROM STDIN"
        ) as copy:
            for chunk in chunks:
                for row in _xref_rows(chunk):
                    copy.write_row(row)
                    edge_count += 1

    logger.info(
        "ingested %d chunks and %d edges for corpus_version %s",
        len(chunks),
        edge_count,
        version,
    )
    return ChunkIngest(version, len(chunks), edge_count, already_present=False)


def ingest_vectors(
    conn: psycopg.Connection,
    vectors,
    chunk_ids: Sequence[str],
    manifest: VectorManifest,
) -> VectorIngest:
    """Load one built vector store as an embedding set. A second call is a no-op."""
    _require_autocommit(conn)
    version = manifest.corpus_version

    with conn.transaction():
        corpus = (
            conn.cursor(row_factory=dict_row)
            .execute(
                "SELECT chunk_count FROM corpus_versions WHERE corpus_version = %s",
                (version,),
            )
            .fetchone()
        )
        if corpus is None:
            raise IngestError(
                f"the vectors are for corpus_version {version}, which has no "
                f"chunks ingested — load the chunks first."
            )
        # No constraint can catch this: a store covering fewer chunks than the
        # corpus is a silently partial index, and every foreign key accepts it.
        if len(chunk_ids) != corpus["chunk_count"]:
            raise IngestError(
                f"the vector store holds {len(chunk_ids)} vectors but "
                f"corpus_version {version} has {corpus['chunk_count']} chunks: "
                f"a partial index would be served as a complete one."
            )
        if manifest.kind != EmbedKind.DOCUMENT.value:
            raise IngestError(
                f"the vector store is encoded as {manifest.kind!r}; only a "
                f"document index is searchable (ADR-069)."
            )

        stored = (
            conn.cursor(row_factory=dict_row)
            .execute(
                f"SELECT embedding_set_id, {', '.join(SET_IDENTITY)} "
                f"FROM embedding_sets "
                f"WHERE corpus_version = %s AND vectors_sha256 = %s",
                (version, manifest.vectors_sha256),
            )
            .fetchone()
        )
        if stored is not None:
            _check_existing_set(stored, manifest)
            set_id = stored["embedding_set_id"]
            present = conn.execute(
                "SELECT count(*) FROM chunk_embeddings WHERE embedding_set_id = %s",
                (set_id,),
            ).fetchone()[0]
            if present != manifest.chunk_count:
                raise IngestError(
                    f"embedding set {set_id} claims {manifest.chunk_count} vectors "
                    f"but holds {present}: it is partially deleted. Delete the set "
                    f"and re-run."
                )
            logger.info(
                "embedding set %d is already ingested (%d vectors)", set_id, present
            )
            return VectorIngest(set_id, version, present, already_present=True)

        set_id = conn.execute(
            "INSERT INTO embedding_sets (corpus_version, model_id, dim, revision, "
            "runtime, encoding, kind, device, chunk_count, vectors_sha256, "
            "ids_sha256, probe_set_version, probes) VALUES "
            "(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) "
            "RETURNING embedding_set_id",
            (
                version,
                manifest.model.model_id,
                manifest.dim,
                manifest.model.revision,
                manifest.model.runtime,
                manifest.model.encoding,
                manifest.kind,
                manifest.device,
                manifest.chunk_count,
                manifest.vectors_sha256,
                manifest.ids_sha256,
                manifest.probe_set_version,
                Json([probe.model_dump(mode="json") for probe in manifest.probes]),
            ),
        ).fetchone()[0]

        with conn.cursor().copy(
            f"COPY chunk_embeddings ({', '.join(EMBEDDING_COLUMNS)}) FROM STDIN"
        ) as copy:
            for chunk_id, vector in zip(chunk_ids, vectors, strict=True):
                copy.write_row((set_id, version, chunk_id, vector_literal(vector)))

        sets = conn.execute(
            "SELECT count(*) FROM embedding_sets WHERE corpus_version = %s", (version,)
        ).fetchone()[0]

    if sets > 1:
        logger.info("corpus_version %s now carries %d embedding sets", version, sets)
    logger.info(
        "ingested %d vectors as embedding set %d for corpus_version %s",
        len(chunk_ids),
        set_id,
        version,
    )
    return VectorIngest(set_id, version, len(chunk_ids), already_present=False)
