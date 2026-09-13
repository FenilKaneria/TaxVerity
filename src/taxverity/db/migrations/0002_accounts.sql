-- Step 11.1 — accounts, refresh tokens, login attempts, threads and messages.
--
-- Isolation is enforced by the schema as well as by the queries. Every row a
-- user owns carries user_id, and a message or a fact state points at its thread
-- through (thread_id, user_id), so the database refuses to attach one user's
-- message to another user's thread even if a query forgets its filter.
--
-- No user_profiles table: the cross-thread profile is deferred (ADR-110).

CREATE TABLE users (
    user_id       uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    -- Stored normalised (trimmed, lower-cased) by the application, and checked
    -- here so a second spelling of one address cannot become a second account.
    email         text NOT NULL UNIQUE
                  CHECK (email = lower(btrim(email)) AND email LIKE '%_@_%'),
    password_hash text NOT NULL CHECK (password_hash LIKE '$argon2id$%'),
    created_at    timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE refresh_tokens (
    token_id     uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id      uuid NOT NULL REFERENCES users ON DELETE CASCADE,
    -- Every token descended from one login shares a family. A replayed token
    -- revokes the whole family (rule 04).
    family_id    uuid NOT NULL,
    -- Only the sha256 of the token is stored; the token itself is never
    -- written anywhere.
    token_sha256 text NOT NULL UNIQUE CHECK (token_sha256 ~ '^[0-9a-f]{64}$'),
    created_at   timestamptz NOT NULL DEFAULT now(),
    expires_at   timestamptz NOT NULL,
    -- Set when the token is exchanged for its successor. Presenting a token
    -- with used_at set is the reuse signal.
    used_at      timestamptz,
    revoked_at   timestamptz,
    CHECK (expires_at > created_at)
);

CREATE INDEX refresh_tokens_family ON refresh_tokens (family_id);
CREATE INDEX refresh_tokens_user ON refresh_tokens (user_id);

-- Login rate limiting lives here, not in memory: Lambda runs many instances.
-- Not keyed to users, so an attempt against an address with no account is
-- counted exactly like one against a real account.
CREATE TABLE login_attempts (
    attempt_id   bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    email        text NOT NULL,
    ip           text NOT NULL,
    succeeded    boolean NOT NULL,
    attempted_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX login_attempts_email ON login_attempts (email, attempted_at);
CREATE INDEX login_attempts_ip ON login_attempts (ip, attempted_at);

CREATE TABLE threads (
    thread_id  uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id    uuid NOT NULL REFERENCES users ON DELETE CASCADE,
    title      text NOT NULL CHECK (title <> ''),
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    -- Target of the composite foreign keys below.
    UNIQUE (thread_id, user_id)
);

CREATE INDEX threads_user ON threads (user_id, updated_at DESC);

CREATE TABLE messages (
    message_id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    thread_id  uuid NOT NULL,
    user_id    uuid NOT NULL,
    role       text NOT NULL CHECK (role IN ('user', 'assistant')),
    content    text NOT NULL,
    -- The assistant turn's released events (claims, withheld, final); empty for
    -- a user turn.
    payload    jsonb NOT NULL DEFAULT '{}',
    created_at timestamptz NOT NULL DEFAULT now(),
    FOREIGN KEY (thread_id, user_id)
        REFERENCES threads (thread_id, user_id) ON DELETE CASCADE
);

CREATE INDEX messages_thread ON messages (thread_id, message_id);

-- One fact state per thread; Step 11.5 owns what goes in it.
CREATE TABLE thread_facts (
    thread_id  uuid PRIMARY KEY,
    user_id    uuid NOT NULL,
    facts      jsonb NOT NULL DEFAULT '{}',
    updated_at timestamptz NOT NULL DEFAULT now(),
    FOREIGN KEY (thread_id, user_id)
        REFERENCES threads (thread_id, user_id) ON DELETE CASCADE
);
