"""Steps 10.2 and 10.4 — what the generator emits, and how a stream becomes it.

The model writes NDJSON, one claim per line (ADR-022), so each claim can be
verified and released the moment its line is complete. The id is assigned here
in stream order rather than read from the model: an id is how a `withheld`
event names the claim it replaces, so it must never repeat.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

CLAIMS_STAGE_VERSION = 2


class ClaimType(StrEnum):
    # A statement of what the Act says. It must cite evidence.
    STATUTE = "statute"
    # Applied advice for the person's own situation. Cited exactly like STATUTE
    # (citations, verbatim quotes) — the difference is voice, not evidence.
    ADVICE = "advice"
    # A restatement of the calculator's trace. Its numbers must come from it.
    COMPUTATION = "computation"
    # "The Act does not deal with X." The one claim type that carries no
    # citation — gated harder for exactly that reason (see verifier.py).
    NO_BASIS = "no_basis"


class Citation(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    path: str = Field(min_length=1)
    # Copied verbatim from the cited provision; the verifier checks that.
    quote: str = Field(min_length=1)


class Claim(BaseModel):
    """One line as the model wrote it, before verification."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    type: ClaimType
    text: str = Field(min_length=1)
    citations: tuple[Citation, ...] = ()


class MalformedClaim(ValueError):
    """A line that is not a claim. Verification treats it as a failed claim."""


class ClaimEvent(BaseModel):
    """A released claim. `verified` can only be True: rule 04's invariant is a type."""

    model_config = ConfigDict(frozen=True)

    id: int
    type: ClaimType
    text: str
    citations: tuple[Citation, ...]
    verified: Literal[True] = True


class WithheldEvent(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: int
    reason: str


# Step 12.6 (rule 03): a constant, never model-generated, carried in every
# `final` event the streaming contract (rule 04) defines - Phase 13's graph is
# what assembles that event, but the text lives here so it has one source of
# truth from the moment any caller needs it, verbatim from
# `docs/SAFETY_POLICY.md`'s own Disclaimer section. The frontend (Phase 16)
# renders it non-dismissible.
DISCLAIMER = (
    "This is general information about the Income-tax Act, 2025, not "
    "professional tax advice. Confirm anything material with a qualified "
    "professional before acting on it."
)


def parse_claim(line: str) -> Claim:
    try:
        payload = json.loads(line)
    except ValueError as error:
        raise MalformedClaim(f"not JSON: {error}") from error
    if not isinstance(payload, dict):
        raise MalformedClaim("not a JSON object")
    try:
        return Claim.model_validate(payload)
    except ValidationError as error:
        raise MalformedClaim(f"not a claim: {error.error_count()} schema errors") from error


class LineBuffer:
    """Complete lines out of arbitrarily split text deltas."""

    def __init__(self) -> None:
        self._partial = ""

    def feed(self, delta: str) -> list[str]:
        text = self._partial + delta
        *complete, self._partial = text.split("\n")
        return [line for line in (_clean(raw) for raw in complete) if line]

    def flush(self) -> list[str]:
        # The last line of a stream has no trailing newline (Step 7.1).
        line, self._partial = _clean(self._partial), ""
        return [line] if line else []


def iter_lines(deltas: Iterable[str]) -> Iterator[str]:
    """Closing this closes the source too, so a stopped answer stops its stream."""
    source = iter(deltas)
    buffer = LineBuffer()
    try:
        for delta in source:
            yield from buffer.feed(delta)
        yield from buffer.flush()
    finally:
        close = getattr(source, "close", None)
        if close is not None:
            close()


def _clean(raw: str) -> str:
    line = raw.strip()
    # A model sometimes fences its output despite the prompt. A fence carries no
    # claim, so it is dropped rather than withheld as a malformed one.
    if line.startswith("```"):
        return ""
    return line
