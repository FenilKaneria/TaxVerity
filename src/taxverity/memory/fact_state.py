"""Step 11.5 — thread-scoped fact-state merge and persistence.

Memory is the accumulated `UserFacts`, not a transcript (rule 04): each turn's
extraction is merged into what the thread already knows, with an explicit,
recorded conflict resolution rather than a silent overwrite. A newer *stated*
fact overrides an older one; an *inferred* fact never overrides a stated one,
because the person's own words outrank a guess made from them. A user editing
the facts panel is the most authoritative signal there is, so it always wins
and is recorded the same way any other override is — never applied silently.

`FactStatus.PROFILE_DEFAULT` cannot enter here at all: the cross-thread
profile is deferred (ADR-110), and rule 04 makes reusing a stale profile fact
as `stated` a correctness bug, not a convenience. Refusing it here, at the one
place facts are merged, is what keeps that unforgeable rather than a
convention later code has to remember.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

import psycopg
from psycopg.types.json import Jsonb

from taxverity.facts import Fact, FactField, FactStatus, UserFacts, fact_value
from taxverity.threads.store import get_thread

USER_EDIT_PROVENANCE = "edited by user"


@dataclass(frozen=True)
class Override:
    """One conflict resolution: `new` replaced `previous` for `field`."""

    field: FactField
    previous: Fact
    previous_provenance: str
    new: Fact
    new_provenance: str


@dataclass(frozen=True)
class ThreadFactState:
    facts: dict[FactField, Fact] = field(default_factory=dict)
    provenance: dict[FactField, str] = field(default_factory=dict)
    overrides: tuple[Override, ...] = ()

    def get(self, field_: FactField) -> Fact | None:
        return self.facts.get(field_)

    def provenance_for(self, field_: FactField) -> str | None:
        return self.provenance.get(field_)

    def as_user_facts(self) -> UserFacts:
        """What Phase 9's calculator computes from — the thread's current
        belief, independent of which turn or edit produced each field."""
        return UserFacts(facts=tuple(self.facts.values()))


def merge_turn(
    state: ThreadFactState, extracted: UserFacts, *, turn: int
) -> ThreadFactState:
    """Merges one turn's extraction into `state`. `MISSING` facts are dropped —
    silence about a field says nothing about what was known before."""
    facts = dict(state.facts)
    provenance = dict(state.provenance)
    overrides = list(state.overrides)

    for fact in extracted.facts:
        if fact.status is FactStatus.MISSING:
            continue
        if fact.status is FactStatus.PROFILE_DEFAULT:
            raise ValueError(
                "a profile-default fact may not be merged into thread fact state"
            )
        existing = facts.get(fact.field)
        if (
            existing is not None
            and existing.status is FactStatus.STATED
            and fact.status is FactStatus.INFERRED
        ):
            # A guess from this turn does not outrank what the person already
            # told this thread.
            continue
        new_provenance = f"{fact.status.value} in turn {turn}"
        if existing is not None:
            overrides.append(
                Override(
                    field=fact.field,
                    previous=existing,
                    previous_provenance=provenance[fact.field],
                    new=fact,
                    new_provenance=new_provenance,
                )
            )
        facts[fact.field] = fact
        provenance[fact.field] = new_provenance

    return ThreadFactState(facts=facts, provenance=provenance, overrides=tuple(overrides))


def apply_user_edit(
    state: ThreadFactState, field_: FactField, raw_value: str
) -> ThreadFactState:
    """A user edit is always a `stated` fact and always wins, recorded with
    provenance `"edited by user"` like any other override."""
    value = fact_value(field_, raw_value, USER_EDIT_PROVENANCE)
    if value is None:
        raise ValueError(f"{raw_value!r} is not a valid value for {field_}")
    new_fact = Fact(
        field=field_,
        status=FactStatus.STATED,
        raw_value=raw_value,
        value=value,
        source_span=USER_EDIT_PROVENANCE,
    )
    facts = dict(state.facts)
    provenance = dict(state.provenance)
    overrides = list(state.overrides)
    existing = facts.get(field_)
    if existing is not None:
        overrides.append(
            Override(
                field=field_,
                previous=existing,
                previous_provenance=provenance[field_],
                new=new_fact,
                new_provenance=USER_EDIT_PROVENANCE,
            )
        )
    facts[field_] = new_fact
    provenance[field_] = USER_EDIT_PROVENANCE
    return ThreadFactState(facts=facts, provenance=provenance, overrides=tuple(overrides))


def _fact_to_json(fact: Fact) -> dict[str, Any]:
    # `value` is never stored: a Decimal round-tripped through JSON comes back
    # as a string, and the model's `Decimal | int | str` union then keeps it a
    # string instead of a Decimal (Step 7.3's same hazard). Re-derived instead,
    # by the same `fact_value()` normalisation that produced it the first time.
    return {
        "field": fact.field.value,
        "status": fact.status.value,
        "raw_value": fact.raw_value,
        "source_span": fact.source_span,
    }


def _fact_from_json(data: dict[str, Any]) -> Fact:
    field_ = FactField(data["field"])
    status = FactStatus(data["status"])
    raw_value = data["raw_value"]
    source_span = data["source_span"]
    value = (
        None
        if status is FactStatus.MISSING
        else fact_value(field_, raw_value, source_span)
    )
    return Fact(
        field=field_,
        status=status,
        raw_value=raw_value,
        value=value,
        source_span=source_span,
    )


def to_json(state: ThreadFactState) -> dict[str, Any]:
    return {
        "facts": {
            field_.value: _fact_to_json(fact) for field_, fact in state.facts.items()
        },
        "provenance": {
            field_.value: note for field_, note in state.provenance.items()
        },
        "overrides": [
            {
                "field": override.field.value,
                "previous": _fact_to_json(override.previous),
                "previous_provenance": override.previous_provenance,
                "new": _fact_to_json(override.new),
                "new_provenance": override.new_provenance,
            }
            for override in state.overrides
        ],
    }


def from_json(data: dict[str, Any]) -> ThreadFactState:
    facts = {
        FactField(name): _fact_from_json(payload)
        for name, payload in data.get("facts", {}).items()
    }
    provenance = {
        FactField(name): note for name, note in data.get("provenance", {}).items()
    }
    overrides = tuple(
        Override(
            field=FactField(entry["field"]),
            previous=_fact_from_json(entry["previous"]),
            previous_provenance=entry["previous_provenance"],
            new=_fact_from_json(entry["new"]),
            new_provenance=entry["new_provenance"],
        )
        for entry in data.get("overrides", [])
    )
    return ThreadFactState(facts=facts, provenance=provenance, overrides=overrides)


def load_fact_state(
    conn: psycopg.Connection, user_id: UUID, thread_id: UUID
) -> ThreadFactState:
    """Raises `ThreadNotFound` for a thread that does not exist or is not the
    caller's — the same isolation `threads.store` enforces everywhere else."""
    get_thread(conn, user_id, thread_id)
    row = conn.execute(
        "SELECT facts FROM thread_facts WHERE thread_id = %s AND user_id = %s",
        (thread_id, user_id),
    ).fetchone()
    if row is None:
        return ThreadFactState()
    return from_json(row[0])


def save_fact_state(
    conn: psycopg.Connection, user_id: UUID, thread_id: UUID, state: ThreadFactState
) -> None:
    get_thread(conn, user_id, thread_id)
    conn.execute(
        "INSERT INTO thread_facts (thread_id, user_id, facts) VALUES (%s, %s, %s) "
        "ON CONFLICT (thread_id) DO UPDATE SET "
        "facts = EXCLUDED.facts, updated_at = now()",
        (thread_id, user_id, Jsonb(to_json(state))),
    )
