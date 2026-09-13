"""Step 11.4e — the guest chat trial: 5 turns per guest cookie, with a
looser per-IP cap for a cleared cookie."""

from __future__ import annotations

from uuid import uuid4

import psycopg
import pytest
from psycopg.conninfo import make_conninfo

from taxverity.guests.quota import (
    GUEST_IP_LIMIT,
    GUEST_TURN_LIMIT,
    GuestLimitReached,
    guest_turns_used,
    record_guest_turn,
)


def test_a_guest_may_send_up_to_the_limit(schema):
    guest_id = uuid4()
    for _ in range(GUEST_TURN_LIMIT):
        record_guest_turn(schema, guest_id, "1.1.1.1")
    assert guest_turns_used(schema, guest_id) == GUEST_TURN_LIMIT


def test_the_turn_after_the_limit_is_refused_and_not_recorded(schema):
    guest_id = uuid4()
    for _ in range(GUEST_TURN_LIMIT):
        record_guest_turn(schema, guest_id, "1.1.1.1")
    with pytest.raises(GuestLimitReached):
        record_guest_turn(schema, guest_id, "1.1.1.1")
    assert guest_turns_used(schema, guest_id) == GUEST_TURN_LIMIT


def test_one_guests_turns_do_not_count_against_another(schema):
    first, second = uuid4(), uuid4()
    for _ in range(GUEST_TURN_LIMIT):
        record_guest_turn(schema, first, "1.1.1.1")
    record_guest_turn(schema, second, "1.1.1.1")
    assert guest_turns_used(schema, second) == 1


def test_a_new_cookie_from_the_same_ip_hits_the_ip_cap(schema):
    ip = "2.2.2.2"
    for _ in range(GUEST_IP_LIMIT):
        record_guest_turn(schema, uuid4(), ip)
    with pytest.raises(GuestLimitReached):
        record_guest_turn(schema, uuid4(), ip)


def test_needs_an_autocommit_connection(schema, admin_url):
    url = make_conninfo(admin_url, dbname=schema.info.dbname)
    with psycopg.connect(url) as conn:
        with pytest.raises(ValueError, match="autocommit"):
            record_guest_turn(conn, uuid4(), "1.1.1.1")
