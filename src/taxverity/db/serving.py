"""Step 6.6 — the startup guard: what this deployment is configured to serve
must be what the database actually holds.

Every check here compares configuration against the database and nothing else.
Model identity is deliberately absent: `PgVectorIndex` already refuses an
embedding set whose model is not the embedder's (ADR-090), and a second copy of
a correctness check is a second place it can quietly stop being made. Startup is
`resolve_serving()` first — plain SQL, no vendor call, so an outage cannot stop a
boot — and the index second.

Both the corpus version and the embedding set are named in configuration, and
each is checked against the other. An embedding set id is a global serial: a
database reloaded from a different corpus hands out the same small integers, so
an id alone names nothing durable.
"""

from __future__ import annotations

import psycopg
from psycopg.rows import dict_row
from pydantic import BaseModel, ConfigDict

from taxverity.chunking.store import CHUNK_STAGE_VERSIONS
from taxverity.config import Settings
from taxverity.db.migrate import (
    applied_migrations,
    discover_migrations,
    schema_conflicts,
)
from taxverity.embedding.backends import ModelInfo, describe
from taxverity.observability import get_logger

logger = get_logger(__name__)

SERVING_STAGE_VERSION = 1


class ServingRefusal(RuntimeError):
    """The database cannot serve what this deployment is configured to serve."""

    def __init__(self, reasons: tuple[str, ...]) -> None:
        self.reasons = reasons
        joined = "".join(f"\n  - {reason}" for reason in reasons)
        super().__init__(f"refusing to serve:{joined}")


class ServedCorpus(BaseModel):
    """What a passing guard establishes: the identity of the index about to be
    served, for the startup log and for a later /health style report."""

    model_config = ConfigDict(frozen=True)

    corpus_version: str
    doc_id: str
    chunk_count: int
    embedding_set_id: int
    vector_count: int
    model: ModelInfo


def _refuse(reasons: tuple[str, ...]) -> None:
    for reason in reasons:
        logger.error("startup guard: %s", reason)
    raise ServingRefusal(reasons)


_CORPUS_SQL = """
SELECT doc_id, chunk_count, chunk_stage_versions
  FROM corpus_versions
 WHERE corpus_version = %s
"""

_SET_SQL = """
SELECT corpus_version, model_id, dim, revision, runtime, encoding, chunk_count
  FROM embedding_sets
 WHERE embedding_set_id = %s
"""


def serving_target(settings: Settings) -> tuple[str, int]:
    """The configured corpus and embedding set. Raises `MissingSettingError`
    naming the env var when either is unset — serving without one is not a
    default this code is willing to pick."""
    return settings.require("serving_corpus_version"), settings.require(
        "serving_embedding_set_id"
    )


def resolve_serving(
    conn: psycopg.Connection, *, corpus_version: str, embedding_set_id: int
) -> ServedCorpus:
    """Refuse unless this database holds that corpus and that embedding set, at
    the schema and chunk stage versions this code carries.

    Every reason is collected rather than raised at the first one: a deployer
    who fixes one refusal per deploy learns the next one a deploy too late.
    """
    reasons: list[str] = []

    migrations = discover_migrations()
    applied = applied_migrations(conn)
    reasons.extend(schema_conflicts(migrations, applied))
    pending = [m.version for m in migrations if m.version not in applied]
    if pending:
        listed = ", ".join(f"{version:04d}" for version in pending)
        reasons.append(
            f"schema is behind this code: migration(s) {listed} unapplied — "
            f"run scripts/migrate.py"
        )
    if reasons:
        # The only short circuit: every check below reads a table a pending
        # migration may not have created yet.
        _refuse(tuple(reasons))

    cursor = conn.cursor(row_factory=dict_row)
    corpus = cursor.execute(_CORPUS_SQL, (corpus_version,)).fetchone()
    if corpus is None:
        held = [
            row[0]
            for row in conn.execute(
                "SELECT corpus_version FROM corpus_versions ORDER BY ingested_at"
            ).fetchall()
        ]
        reasons.append(
            f"corpus_version {corpus_version[:16]}… is not in this database "
            f"(it holds {[version[:16] + '…' for version in held] or 'none'}) — "
            f"run scripts/ingest_corpus.py"
        )
    elif corpus["chunk_stage_versions"] != CHUNK_STAGE_VERSIONS:
        # A chunk id is derived from corpus_version alone, so chunks built by
        # different chunking code carry ids that are mislabelled, not merely old
        # (ADR-058). The corpus_version cannot see that difference; this can.
        reasons.append(
            f"corpus_version {corpus_version[:16]}… was ingested with chunk stage "
            f"versions {corpus['chunk_stage_versions']}, this code carries "
            f"{CHUNK_STAGE_VERSIONS} — re-ingest"
        )

    served = cursor.execute(_SET_SQL, (embedding_set_id,)).fetchone()
    if served is None:
        reasons.append(f"this database holds no embedding set {embedding_set_id}")
    elif served["corpus_version"] != corpus_version:
        reasons.append(
            f"embedding set {embedding_set_id} was built for corpus_version "
            f"{served['corpus_version'][:16]}…, not the configured "
            f"{corpus_version[:16]}…"
        )

    if reasons:
        _refuse(tuple(reasons))

    assert corpus is not None and served is not None
    resolved = ServedCorpus(
        corpus_version=corpus_version,
        doc_id=corpus["doc_id"],
        chunk_count=corpus["chunk_count"],
        embedding_set_id=embedding_set_id,
        vector_count=served["chunk_count"],
        model=ModelInfo(
            model_id=served["model_id"],
            dim=served["dim"],
            revision=served["revision"],
            runtime=served["runtime"],
            encoding=served["encoding"],
        ),
    )
    logger.info(
        "serving corpus %s (%d chunks) from embedding set %d, %s",
        resolved.corpus_version[:16],
        resolved.chunk_count,
        resolved.embedding_set_id,
        describe(resolved.model),
    )
    return resolved
