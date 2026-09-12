"""Step 6.4 — the same exact dense search, served from Postgres (ADR-090).

Behaviourally a drop-in for `DenseRetriever`: the same `Retriever` Protocol, the
same score (cosine, since both sides are unit vectors), the same tie-break by
corpus order, and the same `DenseRetrievalError` on vendor failure, so Step 5.2's
fallback catches one type whichever index is wired in. Step 6.5 measures whether
the two actually agree on the gold set.

Two things differ, both deliberate. The set that is served is named explicitly —
a corpus may carry several embedding sets and the schema has no way to say which
one is current, so guessing would silently serve a stale index. And the chunks
come back from the database with the hit, so serving needs no chunk file.
"""

from __future__ import annotations

import time
from collections.abc import Sequence

import psycopg
from psycopg.rows import dict_row

from taxverity.chunking.models import Chunk
from taxverity.corpus.nodes import NodeType
from taxverity.db.ingest import vector_literal
from taxverity.embedding.backends import (
    Embedder,
    EmbedKind,
    ModelIdentityError,
    ModelInfo,
    describe,
)
from taxverity.embedding.store import (
    PROBE_SET_VERSION,
    VECTOR_STORE_VERSION,
    StaleVectorStoreError,
    VectorManifest,
)
from taxverity.observability import get_logger
from taxverity.retrieval.base import ScoredChunk
from taxverity.retrieval.dense import FingerprintGate

logger = get_logger(__name__)

PGVECTOR_STAGE_VERSION = 1

_SET_SQL = """
SELECT s.embedding_set_id, s.corpus_version, s.model_id, s.dim, s.revision,
       s.runtime, s.encoding, s.kind, s.device, s.chunk_count, s.vectors_sha256,
       s.ids_sha256, s.probe_set_version, s.probes,
       v.doc_id AS doc_id,
       v.chunk_count AS corpus_chunk_count,
       (SELECT count(*) FROM chunk_embeddings e
         WHERE e.embedding_set_id = s.embedding_set_id) AS stored_vectors
  FROM embedding_sets s
  JOIN corpus_versions v USING (corpus_version)
 WHERE s.embedding_set_id = %s
"""

# One round trip per query: the neighbours and the chunks they name. `<=>` is
# cosine distance, so 1 - distance is the cosine the NumPy index returns as a dot
# product. The ordinal tie-break is why chunk_embeddings carries no ordinal of
# its own — the corpus order lives in one place (ADR-089).
_SEARCH_SQL = """
SELECT c.chunk_id, c.parent_id, c.doc_id, c.corpus_version, c.node_type,
       c.node_path, c.section_number, c.schedule_number, c.root_title,
       c.chapter_numeral, c.chapter_title, c.text, c.page_start, c.page_end,
       c.char_start, c.char_end, c.defined_terms, c.token_count,
       COALESCE((SELECT array_agg(x.target_path ORDER BY x.position)
                   FROM xref_edges x WHERE x.chunk_id = c.chunk_id), '{}')
           AS outgoing_refs,
       1 - (e.embedding <=> %s::vector) AS score
  FROM chunk_embeddings e
  JOIN chunks c ON c.corpus_version = e.corpus_version AND c.chunk_id = e.chunk_id
 WHERE e.embedding_set_id = %s
 ORDER BY e.embedding <=> %s::vector, c.ordinal
 LIMIT %s
"""


def chunk_from_row(row: dict) -> Chunk:
    """The inverse of `db.ingest._chunk_row`, plus the edges.

    `Chunk` re-derives its own id from the row's own text, so a row that has
    been edited in the database refuses itself rather than being served.
    """
    return Chunk(
        chunk_id=row["chunk_id"],
        parent_id=row["parent_id"],
        doc_id=row["doc_id"],
        corpus_version=row["corpus_version"],
        node_type=NodeType(row["node_type"]),
        node_path=row["node_path"],
        section_number=row["section_number"],
        schedule_number=row["schedule_number"],
        root_title=row["root_title"],
        chapter_numeral=row["chapter_numeral"],
        chapter_title=row["chapter_title"],
        text=row["text"],
        page_start=row["page_start"],
        page_end=row["page_end"],
        char_start=row["char_start"],
        char_end=row["char_end"],
        outgoing_refs=tuple(row["outgoing_refs"]),
        defined_terms=tuple(row["defined_terms"]),
        token_count=row["token_count"],
    )


def manifest_from_row(row: dict) -> VectorManifest:
    """Rebuild the vector manifest an embedding set was ingested from, so the
    fingerprint check is the same code the file-backed index runs."""
    return VectorManifest(
        # Not persisted: the row is the ingested form of a store that Step 6.3
        # already validated, so this records the reader's version rather than
        # claiming anything about the writer's.
        store_version=VECTOR_STORE_VERSION,
        doc_id=row["doc_id"],
        corpus_version=row["corpus_version"],
        model=ModelInfo(
            model_id=row["model_id"],
            dim=row["dim"],
            revision=row["revision"],
            runtime=row["runtime"],
            encoding=row["encoding"],
        ),
        kind=row["kind"],
        device=row["device"],
        dim=row["dim"],
        chunk_count=row["chunk_count"],
        vectors_sha256=row["vectors_sha256"],
        ids_sha256=row["ids_sha256"],
        probe_set_version=row["probe_set_version"],
        probes=tuple(row["probes"]),
    )


class PgVectorIndex:
    """Satisfies the Step 3.3 `Retriever` Protocol, over a named embedding set."""

    def __init__(
        self,
        conn: psycopg.Connection,
        embedding_set_id: int,
        embedder: Embedder,
    ) -> None:
        started = time.perf_counter()
        row = (
            conn.cursor(row_factory=dict_row)
            .execute(_SET_SQL, (embedding_set_id,))
            .fetchone()
        )
        if row is None:
            raise StaleVectorStoreError(
                f"this database holds no embedding set {embedding_set_id}"
            )
        manifest = manifest_from_row(row)
        # The schema refuses a non-document set on insert; checked again because
        # an index searched in the wrong space is silently wrong (ADR-069), and
        # a constraint can be dropped by a migration this code never sees.
        if manifest.kind != EmbedKind.DOCUMENT.value:
            raise StaleVectorStoreError(
                f"embedding set {embedding_set_id} was encoded as {manifest.kind!r}, "
                f"not 'document' — searching it with query vectors compares two "
                f"different spaces."
            )
        if manifest.probe_set_version != PROBE_SET_VERSION:
            raise StaleVectorStoreError(
                f"embedding set {embedding_set_id} was fingerprinted with probe set "
                f"{manifest.probe_set_version}, this code carries {PROBE_SET_VERSION} "
                f"— rebuild the store and re-ingest."
            )
        served = embedder.info()
        if served != manifest.model:
            raise ModelIdentityError(
                f"embedder is {describe(served)}, embedding set {embedding_set_id} "
                f"was built with {describe(manifest.model)}"
            )
        # Ingest makes both of these true; either can stop being true afterwards,
        # by a delete. A short index answers every query, just worse.
        if row["stored_vectors"] != manifest.chunk_count:
            raise StaleVectorStoreError(
                f"embedding set {embedding_set_id} claims {manifest.chunk_count} "
                f"vectors but holds {row['stored_vectors']}"
            )
        if row["corpus_chunk_count"] != manifest.chunk_count:
            raise StaleVectorStoreError(
                f"embedding set {embedding_set_id} covers {manifest.chunk_count} of "
                f"corpus_version {manifest.corpus_version}'s "
                f"{row['corpus_chunk_count']} chunks: a partial index would be "
                f"served as a complete one."
            )

        self._conn = conn
        self._set_id = embedding_set_id
        self._manifest = manifest
        self._gate = FingerprintGate(embedder, manifest)
        logger.info(
            "pgvector index: embedding set %d, %d vectors, dim %d, %s, in %.2fs",
            embedding_set_id,
            manifest.chunk_count,
            manifest.dim,
            describe(manifest.model),
            time.perf_counter() - started,
        )

    def __len__(self) -> int:
        return self._manifest.chunk_count

    @property
    def embedding_set_id(self) -> int:
        return self._set_id

    @property
    def corpus_version(self) -> str:
        return self._manifest.corpus_version

    @property
    def fingerprint_cosine(self) -> float | None:
        return self._gate.fingerprint_cosine

    def search(self, query: str, k: int) -> Sequence[ScoredChunk]:
        if k < 1:
            raise ValueError(f"k must be at least 1, not {k}")
        return self.search_vector(self.embed_query(query), k)

    def embed_query(self, query: str) -> list[float]:
        return self._gate.embed_query(query)

    def search_vector(self, vector: Sequence[float], k: int) -> Sequence[ScoredChunk]:
        if k < 1:
            raise ValueError(f"k must be at least 1, not {k}")
        vector = tuple(vector)
        if len(vector) != self._manifest.dim:
            raise ValueError(
                f"query vector has width {len(vector)}, index dim is "
                f"{self._manifest.dim}"
            )
        literal = vector_literal(vector)
        rows = (
            self._conn.cursor(row_factory=dict_row)
            .execute(_SEARCH_SQL, (literal, self._set_id, literal, k))
            .fetchall()
        )
        return [
            ScoredChunk(chunk=chunk_from_row(row), score=float(row["score"]))
            for row in rows
        ]
