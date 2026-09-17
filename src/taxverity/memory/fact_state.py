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

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

import psycopg
from psycopg.types.json import Jsonb

from taxverity.facts import (
    Fact,
    FactField,
    FactStatus,
    SituationFact,
    UserFacts,
    fact_value,
)
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
class SituationOverride:
    """The same conflict resolution as `Override`, keyed by name rather than
    a closed `FactField` — R20 Step 20.4's open-vocabulary facts have no enum
    to key on, so the normalised (stripped, casefolded) name stands in."""

    key: str
    previous: SituationFact
    previous_provenance: str
    new: SituationFact
    new_provenance: str


@dataclass(frozen=True)
class ThreadFactState:
    facts: dict[FactField, Fact] = field(default_factory=dict)
    provenance: dict[FactField, str] = field(default_factory=dict)
    overrides: tuple[Override, ...] = ()
    # R20 Step 20.4. Never read by `as_user_facts()` or Phase 9's calculator —
    # these are the open facts a future `reason` node (20.5) will read
    # alongside the closed ones, not a calculator input.
    situation: dict[str, SituationFact] = field(default_factory=dict)
    situation_provenance: dict[str, str] = field(default_factory=dict)
    situation_overrides: tuple[SituationOverride, ...] = ()

    def get(self, field_: FactField) -> Fact | None:
        return self.facts.get(field_)

    def provenance_for(self, field_: FactField) -> str | None:
        return self.provenance.get(field_)

    def as_user_facts(self) -> UserFacts:
        """What Phase 9's calculator computes from — the thread's current
        belief, independent of which turn or edit produced each field."""
        return UserFacts(facts=tuple(self.facts.values()))

    def situation_facts(self) -> tuple[SituationFact, ...]:
        """The thread's current open-vocabulary belief, one per name."""
        return tuple(self.situation.values())


def _situation_key(fact: SituationFact) -> str:
    return fact.name.strip().casefold()


def merge_turn(
    state: ThreadFactState,
    extracted: UserFacts,
    *,
    turn: int,
    situation_facts: Sequence[SituationFact] = (),
) -> ThreadFactState:
    """Merges one turn's extraction into `state`. `MISSING` facts are dropped —
    silence about a field says nothing about what was known before.

    `situation_facts` (R20 Step 20.4) merges by the same override rule as the
    closed fields — a newer stated fact wins, an inferred one never outranks
    an existing stated one, and every replacement is recorded rather than
    applied silently (rule 04) — keyed by name instead of a `FactField`, since
    the vocabulary here is open.
    """
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

    situation = dict(state.situation)
    situation_provenance = dict(state.situation_provenance)
    situation_overrides = list(state.situation_overrides)

    for situation_fact in situation_facts:
        key = _situation_key(situation_fact)
        existing_situation = situation.get(key)
        if (
            existing_situation is not None
            and existing_situation.status is FactStatus.STATED
            and situation_fact.status is FactStatus.INFERRED
        ):
            continue
        new_provenance = f"{situation_fact.status.value} in turn {turn}"
        if existing_situation is not None:
            situation_overrides.append(
                SituationOverride(
                    key=key,
                    previous=existing_situation,
                    previous_provenance=situation_provenance[key],
                    new=situation_fact,
                    new_provenance=new_provenance,
                )
            )
        situation[key] = situation_fact
        situation_provenance[key] = new_provenance

    return ThreadFactState(
        facts=facts,
        provenance=provenance,
        overrides=tuple(overrides),
        situation=situation,
        situation_provenance=situation_provenance,
        situation_overrides=tuple(situation_overrides),
    )


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


def _situation_to_json(fact: SituationFact) -> dict[str, Any]:
    return {
        "name": fact.name,
        "status": fact.status.value,
        "raw_value": fact.raw_value,
        "source_span": fact.source_span,
    }


def _situation_from_json(data: dict[str, Any]) -> SituationFact:
    return SituationFact(
        name=data["name"],
        status=FactStatus(data["status"]),
        raw_value=data["raw_value"],
        source_span=data["source_span"],
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
        "situation": {
            key: _situation_to_json(fact) for key, fact in state.situation.items()
        },
        "situation_provenance": dict(state.situation_provenance),
        "situation_overrides": [
            {
                "key": override.key,
                "previous": _situation_to_json(override.previous),
                "previous_provenance": override.previous_provenance,
                "new": _situation_to_json(override.new),
                "new_provenance": override.new_provenance,
            }
            for override in state.situation_overrides
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
    # `.get(..., {})`/`.get(..., [])`: a row saved before R20 Step 20.4 has
    # none of these keys, and must load as "no situation facts yet", not raise.
    situation = {
        key: _situation_from_json(payload)
        for key, payload in data.get("situation", {}).items()
    }
    situation_provenance = dict(data.get("situation_provenance", {}))
    situation_overrides = tuple(
        SituationOverride(
            key=entry["key"],
            previous=_situation_from_json(entry["previous"]),
            previous_provenance=entry["previous_provenance"],
            new=_situation_from_json(entry["new"]),
            new_provenance=entry["new_provenance"],
        )
        for entry in data.get("situation_overrides", [])
    )
    return ThreadFactState(
        facts=facts,
        provenance=provenance,
        overrides=overrides,
        situation=situation,
        situation_provenance=situation_provenance,
        situation_overrides=situation_overrides,
    )


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
