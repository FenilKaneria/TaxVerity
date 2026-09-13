"""Step 11.5 — thread-scoped fact-state merge, provenance and persistence.

Memory is the accumulated `UserFacts`, never raw turn history (rule 04): a
newer stated fact overrides an older one and the override is recorded; an
inferred fact never overrides a stated one; a user edit is stated and always
wins; `profile_default` can never enter at all.
"""

from __future__ import annotations

from decimal import Decimal
from uuid import uuid4

import pytest

from conftest import register_account
from taxverity.facts import Fact, FactField, FactStatus, UserFacts
from taxverity.memory.fact_state import (
    ThreadFactState,
    apply_user_edit,
    from_json,
    load_fact_state,
    merge_turn,
    save_fact_state,
    to_json,
)
from taxverity.threads.store import ThreadNotFound, create_thread

PASSWORD = "correct horse battery"


def stated(field: FactField, raw: str, value, span: str) -> Fact:
    return Fact(
        field=field, status=FactStatus.STATED, raw_value=raw, value=value, source_span=span
    )


def inferred(field: FactField, raw: str, value, span: str) -> Fact:
    return Fact(
        field=field, status=FactStatus.INFERRED, raw_value=raw, value=value, source_span=span
    )


def missing(field: FactField) -> Fact:
    return Fact(field=field, status=FactStatus.MISSING, raw_value="", value=None, source_span="")


# --- merge semantics ---------------------------------------------------------


def test_a_fresh_field_is_recorded_with_no_override():
    salary = stated(FactField.SALARY_INCOME, "1400000", Decimal("1400000"), "my salary is 1400000")
    state = merge_turn(ThreadFactState(), UserFacts(facts=(salary,)), turn=1)
    assert state.get(FactField.SALARY_INCOME) == salary
    assert state.provenance_for(FactField.SALARY_INCOME) == "stated in turn 1"
    assert state.overrides == ()


def test_a_newer_stated_fact_overrides_an_older_one_and_it_is_recorded():
    first = stated(FactField.SALARY_INCOME, "1400000", Decimal("1400000"), "my salary is 1400000")
    second = stated(FactField.SALARY_INCOME, "1600000", Decimal("1600000"), "actually it is 1600000")
    state = merge_turn(ThreadFactState(), UserFacts(facts=(first,)), turn=1)
    state = merge_turn(state, UserFacts(facts=(second,)), turn=2)
    assert state.get(FactField.SALARY_INCOME) == second
    assert state.provenance_for(FactField.SALARY_INCOME) == "stated in turn 2"
    assert len(state.overrides) == 1
    override = state.overrides[0]
    assert override.field is FactField.SALARY_INCOME
    assert override.previous == first
    assert override.previous_provenance == "stated in turn 1"
    assert override.new == second
    assert override.new_provenance == "stated in turn 2"


def test_an_inferred_fact_never_overrides_a_stated_one():
    stated_fact = stated(FactField.REGIME, "new", "new", "I want the new regime")
    guessed = inferred(FactField.REGIME, "old", "old", "salaried employees usually pick old")
    state = merge_turn(ThreadFactState(), UserFacts(facts=(stated_fact,)), turn=1)
    state = merge_turn(state, UserFacts(facts=(guessed,)), turn=2)
    assert state.get(FactField.REGIME) == stated_fact
    assert state.provenance_for(FactField.REGIME) == "stated in turn 1"
    assert state.overrides == ()


def test_a_stated_fact_overrides_an_earlier_inferred_one_and_it_is_recorded():
    guessed = inferred(FactField.REGIME, "old", "old", "salaried employees usually pick old")
    stated_fact = stated(FactField.REGIME, "new", "new", "I want the new regime")
    state = merge_turn(ThreadFactState(), UserFacts(facts=(guessed,)), turn=1)
    state = merge_turn(state, UserFacts(facts=(stated_fact,)), turn=2)
    assert state.get(FactField.REGIME) == stated_fact
    assert len(state.overrides) == 1
    assert state.overrides[0].previous == guessed
    assert state.overrides[0].new == stated_fact


def test_a_missing_fact_is_dropped_and_changes_nothing():
    salary = stated(FactField.SALARY_INCOME, "1400000", Decimal("1400000"), "my salary is 1400000")
    state = merge_turn(ThreadFactState(), UserFacts(facts=(salary,)), turn=1)
    state = merge_turn(state, UserFacts(facts=(missing(FactField.SALARY_INCOME),)), turn=2)
    assert state.get(FactField.SALARY_INCOME) == salary
    assert state.provenance_for(FactField.SALARY_INCOME) == "stated in turn 1"
    assert state.overrides == ()


def test_a_profile_default_fact_is_refused_not_merged():
    profile = Fact(
        field=FactField.RESIDENTIAL_STATUS,
        status=FactStatus.PROFILE_DEFAULT,
        raw_value="",
        value="resident",
        source_span="",
    )
    with pytest.raises(ValueError, match="profile-default"):
        merge_turn(ThreadFactState(), UserFacts(facts=(profile,)), turn=1)


# --- user edits --------------------------------------------------------------


def test_a_user_edit_is_stated_and_always_overrides():
    guessed = inferred(FactField.REGIME, "old", "old", "salaried employees usually pick old")
    state = merge_turn(ThreadFactState(), UserFacts(facts=(guessed,)), turn=1)
    state = apply_user_edit(state, FactField.REGIME, "new")
    fact = state.get(FactField.REGIME)
    assert fact.status is FactStatus.STATED
    assert fact.value == "new"
    assert state.provenance_for(FactField.REGIME) == "edited by user"
    assert len(state.overrides) == 1
    assert state.overrides[0].new_provenance == "edited by user"


def test_a_user_edit_overrides_an_earlier_stated_fact_too():
    original = stated(FactField.SALARY_INCOME, "1400000", Decimal("1400000"), "my salary is 1400000")
    state = merge_turn(ThreadFactState(), UserFacts(facts=(original,)), turn=1)
    state = apply_user_edit(state, FactField.SALARY_INCOME, "1500000")
    assert state.get(FactField.SALARY_INCOME).value == Decimal("1500000")
    assert state.overrides[-1].previous == original


def test_an_invalid_user_edit_raises_and_changes_nothing():
    state = ThreadFactState()
    with pytest.raises(ValueError):
        apply_user_edit(state, FactField.SALARY_INCOME, "not a number")


# --- JSON round-trip -----------------------------------------------------------


def test_state_round_trips_through_json_including_decimal_values():
    salary = stated(FactField.SALARY_INCOME, "1400000", Decimal("1400000"), "my salary is 1400000")
    guessed = inferred(FactField.REGIME, "old", "old", "salaried employees usually pick old")
    stated_regime = stated(FactField.REGIME, "new", "new", "I want the new regime")
    state = merge_turn(ThreadFactState(), UserFacts(facts=(salary, guessed)), turn=1)
    state = merge_turn(state, UserFacts(facts=(stated_regime,)), turn=2)

    restored = from_json(to_json(state))
    assert restored.get(FactField.SALARY_INCOME) == salary
    assert restored.get(FactField.REGIME) == stated_regime
    assert restored.provenance_for(FactField.SALARY_INCOME) == "stated in turn 1"
    assert restored.provenance_for(FactField.REGIME) == "stated in turn 2"
    assert len(restored.overrides) == 1
    assert restored.overrides[0].previous == guessed
    assert restored.overrides[0].new == stated_regime


# --- persistence + isolation -------------------------------------------------


@pytest.fixture
def alice(schema):
    return register_account(schema, "alice@example.com", PASSWORD)


@pytest.fixture
def mallory(schema):
    return register_account(schema, "mallory@example.com", PASSWORD)


def test_save_then_load_round_trips_for_the_owning_user(schema, alice):
    thread = create_thread(schema, alice, "House property")
    salary = stated(FactField.SALARY_INCOME, "1400000", Decimal("1400000"), "my salary is 1400000")
    state = merge_turn(ThreadFactState(), UserFacts(facts=(salary,)), turn=1)
    save_fact_state(schema, alice, thread.thread_id, state)
    loaded = load_fact_state(schema, alice, thread.thread_id)
    assert loaded.get(FactField.SALARY_INCOME) == salary


def test_loading_an_unwritten_thread_returns_empty_state(schema, alice):
    thread = create_thread(schema, alice, "House property")
    state = load_fact_state(schema, alice, thread.thread_id)
    assert state.facts == {}
    assert state.overrides == ()


def test_saving_twice_updates_in_place(schema, alice):
    thread = create_thread(schema, alice, "House property")
    salary = stated(FactField.SALARY_INCOME, "1400000", Decimal("1400000"), "my salary is 1400000")
    save_fact_state(schema, alice, thread.thread_id, merge_turn(ThreadFactState(), UserFacts(facts=(salary,)), turn=1))
    revised = stated(FactField.SALARY_INCOME, "1600000", Decimal("1600000"), "actually 1600000")
    state = merge_turn(load_fact_state(schema, alice, thread.thread_id), UserFacts(facts=(revised,)), turn=2)
    save_fact_state(schema, alice, thread.thread_id, state)
    loaded = load_fact_state(schema, alice, thread.thread_id)
    assert loaded.get(FactField.SALARY_INCOME) == revised
    assert len(loaded.overrides) == 1
    count = schema.execute("SELECT count(*) FROM thread_facts").fetchone()[0]
    assert count == 1


def test_a_second_user_cannot_load_another_users_thread_facts(schema, alice, mallory):
    thread = create_thread(schema, alice, "House property")
    salary = stated(FactField.SALARY_INCOME, "1400000", Decimal("1400000"), "my salary is 1400000")
    save_fact_state(schema, alice, thread.thread_id, merge_turn(ThreadFactState(), UserFacts(facts=(salary,)), turn=1))
    with pytest.raises(ThreadNotFound):
        load_fact_state(schema, mallory, thread.thread_id)


def test_a_second_user_cannot_save_over_another_users_thread_facts(schema, alice, mallory):
    thread = create_thread(schema, alice, "House property")
    salary = stated(FactField.SALARY_INCOME, "1400000", Decimal("1400000"), "my salary is 1400000")
    with pytest.raises(ThreadNotFound):
        save_fact_state(
            schema, mallory, thread.thread_id, merge_turn(ThreadFactState(), UserFacts(facts=(salary,)), turn=1)
        )
    assert schema.execute("SELECT count(*) FROM thread_facts").fetchone()[0] == 0


def test_a_nonexistent_thread_is_refused_the_same_way(schema, alice):
    with pytest.raises(ThreadNotFound):
        load_fact_state(schema, alice, uuid4())
