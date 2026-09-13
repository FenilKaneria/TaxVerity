-- Step 11.4b — email verification, password reset, guest trial.
--
-- Registration must not reveal whether an address is taken (rule 03's
-- account-enumeration concern), so a new account starts unverified and login
-- refuses until it is verified. Both verification and password-reset links are
-- single-use tokens stored only as a sha256, the same pattern as
-- refresh_tokens. email_sends bounds how much mail one address or IP can
-- trigger. guest_turns counts an unauthenticated visitor's chat turns so the
-- API can cut them off after the free trial.

ALTER TABLE users
    ADD COLUMN email_verified_at timestamptz;

CREATE TABLE email_tokens (
    token_id     uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id      uuid NOT NULL REFERENCES users ON DELETE CASCADE,
    purpose      text NOT NULL CHECK (purpose IN ('verify', 'reset')),
    token_sha256 text NOT NULL UNIQUE CHECK (token_sha256 ~ '^[0-9a-f]{64}$'),
    created_at   timestamptz NOT NULL DEFAULT now(),
    expires_at   timestamptz NOT NULL,
    used_at      timestamptz,
    CHECK (expires_at > created_at)
);

CREATE INDEX email_tokens_user ON email_tokens (user_id, purpose);

-- Outgoing mail is rate limited here, not in memory, for the same reason as
-- login_attempts: Lambda runs many instances.
CREATE TABLE email_sends (
    send_id   bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    email     text NOT NULL,
    ip        text NOT NULL,
    purpose   text NOT NULL CHECK (purpose IN ('verify', 'reset')),
    sent_at   timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX email_sends_email ON email_sends (email, purpose, sent_at);
CREATE INDEX email_sends_ip ON email_sends (ip, sent_at);

-- No user_id: a guest has no account yet. guest_id is a random id carried in
-- an httpOnly cookie; ip is the second, coarser counter for a cleared cookie.
CREATE TABLE guest_turns (
    turn_id    bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    guest_id   uuid NOT NULL,
    ip         text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX guest_turns_guest ON guest_turns (guest_id, created_at);
CREATE INDEX guest_turns_ip ON guest_turns (ip, created_at);
