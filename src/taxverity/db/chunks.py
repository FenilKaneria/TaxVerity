"""Phase 14 — bulk chunk hydration from Postgres.

Step 6.4's `PgVectorIndex` already hydrates a chunk from its row on every hit
(ADR-090), so serving needs no `chunks.jsonl`. BM25 and the citation shortcut
need every chunk up front, not just the ones a vector search happens to
return, so this is the same hydration widened to a whole corpus. TaxVerity is
deploy-only (the corpus PDF and `data/interim/` are never shipped to a
production instance) — the API's startup composition must read the corpus it
serves from the database, not from a local file.
"""

from __future__ import annotations

import psycopg
from psycopg.rows import dict_row

from taxverity.chunking.models import Chunk
from taxverity.retrieval.pgvector import chunk_from_row

# Mirrors `pgvector.SEARCH_SQL`'s chunk columns and its `xref_edges` aggregate,
# ordered by `ordinal` so BM25's tie-break (corpus order) matches every other
# retriever's.
_ALL_CHUNKS_SQL = """
SELECT c.chunk_id, c.parent_id, c.doc_id, c.corpus_version, c.node_type,
       c.node_path, c.section_number, c.schedule_number, c.root_title,
       c.chapter_numeral, c.chapter_title, c.text, c.page_start, c.page_end,
       c.char_start, c.char_end, c.defined_terms, c.token_count,
       COALESCE((SELECT array_agg(x.target_path ORDER BY x.position)
                   FROM xref_edges x WHERE x.chunk_id = c.chunk_id), '{}')
           AS outgoing_refs
  FROM chunks c
 WHERE c.corpus_version = %s
 ORDER BY c.ordinal
"""


def load_chunks_from_db(conn: psycopg.Connection, corpus_version: str) -> list[Chunk]:
    """Every chunk of `corpus_version`, in corpus order. Empty for a version
    not ingested — a caller comparing the count against `resolve_serving()`'s
    `ServedCorpus.chunk_count` catches that rather than silently building an
    empty index."""
    rows = conn.cursor(row_factory=dict_row).execute(_ALL_CHUNKS_SQL, (corpus_version,))
    return [chunk_from_row(row) for row in rows]
