-- Step 6.2 — chunks, vectors and cross-reference edges (ADR-088).
--
-- Several corpus versions live side by side: a corpus update is a data deploy
-- that can be rolled back, so nothing here is keyed on "the" corpus. Every
-- child row carries its corpus_version through a composite foreign key, so the
-- database itself refuses to attach a vector or a parent from one corpus to a
-- chunk of another. A chunk id is derived from corpus_version, so such a row
-- would be mislabelled, not merely old (ADR-058).

-- Created here rather than by a Docker init script: RDS and Supabase never run
-- one, and this is the path every environment shares (ADR-087).
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE corpus_versions (
    corpus_version       text PRIMARY KEY CHECK (corpus_version ~ '^[0-9a-f]{64}$'),
    doc_id               text NOT NULL CHECK (doc_id <> ''),
    chunk_count          integer NOT NULL CHECK (chunk_count > 0),
    root_count           integer NOT NULL CHECK (root_count > 0),
    -- The chunk manifest's stage versions and artifact hash: what 6.3 checks
    -- before it will treat an existing corpus_version as already ingested.
    chunk_stage_versions jsonb NOT NULL,
    chunks_sha256        text NOT NULL CHECK (chunks_sha256 ~ '^[0-9a-f]{64}$'),
    ingested_at          timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE chunks (
    chunk_id        text PRIMARY KEY CHECK (chunk_id ~ '^[0-9a-f]{16}$'),
    corpus_version  text NOT NULL REFERENCES corpus_versions ON DELETE CASCADE,
    -- Position in the chunk store. BM25 and dense search break ties by corpus
    -- order, so pgvector search needs it to reproduce their rankings (6.5).
    ordinal         integer NOT NULL CHECK (ordinal >= 0),
    parent_id       text,
    doc_id          text NOT NULL,
    node_type       text NOT NULL CHECK (node_type IN (
                        'section', 'subsection', 'clause', 'subclause', 'item',
                        'subitem', 'schedule', 'schedule_paragraph')),
    node_path       text NOT NULL,
    section_number  text,
    schedule_number text,
    root_title      text,
    chapter_numeral text,
    chapter_title   text,
    text            text NOT NULL CHECK (text <> ''),
    page_start      integer NOT NULL CHECK (page_start >= 0),
    page_end        integer NOT NULL,
    char_start      integer,
    char_end        integer,
    defined_terms   text[] NOT NULL DEFAULT '{}',
    token_count     integer CHECK (token_count >= 0),

    -- Citation -> chunk is injective within a corpus (ADR-056): the verifier
    -- resolves a citation to exactly one text.
    UNIQUE (corpus_version, node_path),
    UNIQUE (corpus_version, ordinal),
    -- Target of the composite foreign keys below.
    UNIQUE (corpus_version, chunk_id),
    -- Deferred so 6.3 need not insert parents before children.
    FOREIGN KEY (corpus_version, parent_id)
        REFERENCES chunks (corpus_version, chunk_id)
        ON DELETE CASCADE DEFERRABLE INITIALLY DEFERRED,
    CHECK ((section_number IS NULL) <> (schedule_number IS NULL)),
    CHECK (page_end >= page_start),
    CHECK ((char_start IS NULL) = (char_end IS NULL)),
    CHECK (char_start >= 0 AND char_end >= char_start)
);

-- Mirrors Chunk.outgoing_refs, so a chunk reads back exactly. The target is a
-- citation, not a foreign key: some targets name no chunk and resolve to their
-- nearest ancestor at query time (ADR-062). Refs are aggregated over the
-- chunk's subtree, as ADR-055 makes the chunk's text.
CREATE TABLE xref_edges (
    chunk_id    text NOT NULL REFERENCES chunks ON DELETE CASCADE,
    position    integer NOT NULL CHECK (position >= 0),
    target_path text NOT NULL CHECK (target_path <> ''),
    PRIMARY KEY (chunk_id, position)
);

-- One row per built vector store: the vector manifest's identity, integrity and
-- probe fingerprint, which the 6.6 startup guard checks before serving.
CREATE TABLE embedding_sets (
    embedding_set_id  bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    corpus_version    text NOT NULL REFERENCES corpus_versions ON DELETE CASCADE,
    model_id          text NOT NULL,
    dim               integer NOT NULL CHECK (dim = 1024),
    revision          text NOT NULL,
    runtime           text NOT NULL,
    encoding          text NOT NULL,
    -- Only a document index is ever searched; a query-encoded index is
    -- silently wrong, so it cannot be stored at all (ADR-069).
    kind              text NOT NULL CHECK (kind = 'document'),
    device            text NOT NULL,
    chunk_count       integer NOT NULL CHECK (chunk_count > 0),
    vectors_sha256    text NOT NULL CHECK (vectors_sha256 ~ '^[0-9a-f]{64}$'),
    ids_sha256        text NOT NULL CHECK (ids_sha256 ~ '^[0-9a-f]{64}$'),
    probe_set_version integer NOT NULL CHECK (probe_set_version >= 1),
    probes            jsonb NOT NULL,
    created_at        timestamptz NOT NULL DEFAULT now(),
    -- 6.3's idempotency key: the same built store is never loaded twice.
    UNIQUE (corpus_version, vectors_sha256),
    UNIQUE (embedding_set_id, corpus_version)
);

-- vector(1024) fixes the width: a model with another width is a new migration,
-- made deliberately. No ANN index: exact search over ~8k rows is cheap, and
-- 6.5 decides on measurement whether HNSW earns one.
CREATE TABLE chunk_embeddings (
    embedding_set_id bigint NOT NULL,
    corpus_version   text NOT NULL,
    chunk_id         text NOT NULL,
    embedding        vector(1024) NOT NULL,
    PRIMARY KEY (embedding_set_id, chunk_id),
    FOREIGN KEY (embedding_set_id, corpus_version)
        REFERENCES embedding_sets (embedding_set_id, corpus_version)
        ON DELETE CASCADE,
    FOREIGN KEY (corpus_version, chunk_id)
        REFERENCES chunks (corpus_version, chunk_id)
        ON DELETE CASCADE
);
