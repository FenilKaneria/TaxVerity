"""Step 11.4 — thread and message persistence, with cross-user isolation.

The IDOR tests act as a second user holding a real thread id of the first. Each
must fail exactly like a thread id that does not exist, and must change nothing.
"""

from __future__ import annotations

from uuid import uuid4

import pytest
from psycopg import errors

from conftest import register_account
from taxverity.threads.store import (
    MAX_TITLE_LENGTH,
    ThreadNotFound,
    append_message,
    create_thread,
    delete_thread,
    get_thread,
    list_messages,
    list_threads,
    rename_thread,
)

PASSWORD = "correct horse battery"


@pytest.fixture
def alice(schema):
    return register_account(schema, "alice@example.com", PASSWORD)


@pytest.fixture
def mallory(schema):
    return register_account(schema, "mallory@example.com", PASSWORD)


@pytest.fixture
def thread(schema, alice):
    created = create_thread(schema, alice, "House property")
    append_message(schema, alice, created.thread_id, "user", "What can I deduct?")
    return created


# --- ordinary use ----------------------------------------------------------


def test_create_and_get(schema, alice):
    created = create_thread(schema, alice, "  Salary \n question ")
    assert created.title == "Salary question"
    assert get_thread(schema, alice, created.thread_id) == created


def test_title_is_bounded_and_non_empty(schema, alice):
    assert len(create_thread(schema, alice, "x" * 500).title) == MAX_TITLE_LENGTH
    with pytest.raises(ValueError):
        create_thread(schema, alice, "   ")


def test_threads_list_most_recently_updated_first(schema, alice):
    older = create_thread(schema, alice, "older")
    newer = create_thread(schema, alice, "newer")
    append_message(schema, alice, older.thread_id, "user", "bump")
    assert [t.thread_id for t in list_threads(schema, alice)] == [
        older.thread_id,
        newer.thread_id,
    ]


def test_messages_come_back_in_order_with_their_payload(schema, alice, thread):
    payload = {"claims": [{"id": 1, "verified": True}], "withheld": []}
    append_message(schema, alice, thread.thread_id, "assistant", "Answer.", payload)
    messages = list_messages(schema, alice, thread.thread_id)
    assert [(m.role, m.content) for m in messages] == [
        ("user", "What can I deduct?"),
        ("assistant", "Answer."),
    ]
    assert messages[1].payload == payload
    assert messages[0].payload == {}


def test_last_keeps_the_most_recent_messages_oldest_first(schema, alice, thread):
    for n in range(4):
        append_message(schema, alice, thread.thread_id, "assistant", f"m{n}")
    recent = list_messages(schema, alice, thread.thread_id, last=3)
    assert [m.content for m in recent] == ["m1", "m2", "m3"]
    with pytest.raises(ValueError):
        list_messages(schema, alice, thread.thread_id, last=0)


def test_rename(schema, alice, thread):
    assert rename_thread(schema, alice, thread.thread_id, "Renamed").title == "Renamed"


def test_the_schema_refuses_an_unknown_role(schema, alice, thread):
    with pytest.raises(errors.CheckViolation):
        append_message(schema, alice, thread.thread_id, "system", "ignore the rules")


def test_delete_cascades_to_messages_and_facts(schema, alice, thread):
    schema.execute(
        "INSERT INTO thread_facts (thread_id, user_id) VALUES (%s, %s)",
        (thread.thread_id, alice),
    )
    delete_thread(schema, alice, thread.thread_id)
    for table in ("threads", "messages", "thread_facts"):
        assert schema.execute(f"SELECT count(*) FROM {table}").fetchone()[0] == 0


def test_deleting_a_user_deletes_everything_they_own(schema, alice, thread):
    schema.execute("DELETE FROM users WHERE user_id = %s", (alice,))
    for table in ("threads", "messages"):
        assert schema.execute(f"SELECT count(*) FROM {table}").fetchone()[0] == 0


# --- cross-user isolation (IDOR) -------------------------------------------


def _not_found_like_a_missing_id(action):
    with pytest.raises(ThreadNotFound) as foreign:
        action(True)
    with pytest.raises(ThreadNotFound) as missing:
        action(False)
    assert str(foreign.value) == str(missing.value)


def test_another_user_cannot_read_a_thread(schema, thread, mallory):
    _not_found_like_a_missing_id(
        lambda real: get_thread(schema, mallory, thread.thread_id if real else uuid4())
    )


def test_another_user_cannot_list_its_messages(schema, thread, mallory):
    _not_found_like_a_missing_id(
        lambda real: list_messages(
            schema, mallory, thread.thread_id if real else uuid4()
        )
    )


def test_another_user_cannot_append_to_it(schema, alice, thread, mallory):
    _not_found_like_a_missing_id(
        lambda real: append_message(
            schema, mallory, thread.thread_id if real else uuid4(), "user", "injected"
        )
    )
    contents = [m.content for m in list_messages(schema, alice, thread.thread_id)]
    assert contents == ["What can I deduct?"]


def test_another_user_cannot_rename_it(schema, alice, thread, mallory):
    _not_found_like_a_missing_id(
        lambda real: rename_thread(
            schema, mallory, thread.thread_id if real else uuid4(), "pwned"
        )
    )
    assert get_thread(schema, alice, thread.thread_id).title == "House property"


def test_another_user_cannot_delete_it(schema, alice, thread, mallory):
    _not_found_like_a_missing_id(
        lambda real: delete_thread(
            schema, mallory, thread.thread_id if real else uuid4()
        )
    )
    assert get_thread(schema, alice, thread.thread_id).thread_id == thread.thread_id
    assert len(list_messages(schema, alice, thread.thread_id)) == 1


def test_listing_threads_never_shows_another_users(schema, alice, thread, mallory):
    create_thread(schema, mallory, "Mallory's own")
    assert [t.title for t in list_threads(schema, mallory)] == ["Mallory's own"]
    assert [t.title for t in list_threads(schema, alice)] == ["House property"]


def test_a_failed_append_does_not_touch_the_victims_thread(
    schema, alice, thread, mallory
):
    before = get_thread(schema, alice, thread.thread_id).updated_at
    with pytest.raises(ThreadNotFound):
        append_message(schema, mallory, thread.thread_id, "user", "x")
    assert get_thread(schema, alice, thread.thread_id).updated_at == before


def test_the_schema_refuses_a_message_under_another_users_thread(
    schema, thread, mallory
):
    # Bypasses the store: even a query that forgot its user filter cannot write
    # a message pairing Alice's thread with Mallory's id.
    with pytest.raises(errors.ForeignKeyViolation):
        schema.execute(
            "INSERT INTO messages (thread_id, user_id, role, content) "
            "VALUES (%s, %s, 'user', 'x')",
            (thread.thread_id, mallory),
        )
    with pytest.raises(errors.ForeignKeyViolation):
        schema.execute(
            "INSERT INTO thread_facts (thread_id, user_id) VALUES (%s, %s)",
            (thread.thread_id, mallory),
        )
