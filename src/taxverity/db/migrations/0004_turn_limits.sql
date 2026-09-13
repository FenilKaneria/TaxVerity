-- Step 14.6 — index backing the per-user daily turn cap. The cap counts
-- recent user-role messages for one user; without this the count() is a
-- sequential scan of the whole table.

CREATE INDEX messages_user_recent ON messages (user_id, created_at) WHERE role = 'user';
