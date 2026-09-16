"""R19 Phase B (ADR-120) — what the generator emits, and how a stream becomes
it.

The model writes plain markdown, one releasable unit per line (a "## "
heading, a "- " bullet, or a plain sentence), so each line can be verified
and released the moment it is complete — same streaming contract as before
(one claim = roughly one line), just a different line grammar. `iter_lines()`
and `LineBuffer` are unchanged from the NDJSON design; only what a line
*means* changed.

A citation is no longer a model-supplied path-and-quote pair. The evidence
pack numbers its units `[1]..[n]` (see `generate.render_context`), and the
model cites by writing that number inline, e.g. "...deductible [2]." — a
small integer is far harder for a model to get subtly wrong than a section
path plus a verbatim quote, and it is what actually lets a sentence be
released as ordinary prose instead of one JSON object per claim. The
verifier (see `verifier.py`) resolves `[n]` to the pack unit at that
position; `Claim.citations` is filled in by the verifier from what it
resolved, never trusted from the model.

A line restating a figure from the calculator's trace (never from a
passage) carries the literal marker `[calc]` instead of a number, so the
verifier can tell "cite the evidence" and "restate the computation" apart
without a JSON `type` field.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

CLAIMS_STAGE_VERSION = 3

# "The Act does not ...", "The Act is silent on ...", "Nothing in the Act
# ...": the one claim type carrying no citation, recognisable only by its own
# opening words (rule 03) — kept here, not in verifier.py, since classifying
# a line by its text is this module's job.
NO_BASIS_OPENERS = ("The Act does not", "The Act is silent on", "Nothing in the Act")

_HEADING = re.compile(r"^#{1,6}\s+\S")
CALC_MARKER = "[calc]"
MARKER = re.compile(r"\[(\d+)\]")


class ClaimType(StrEnum):
    # A "## " heading naming what follows. Structural only — no citation, no
    # figure, never restates the Act.
    HEADING = "heading"
    # A sentence or bullet applying or restating the Act — the difference
    # between "you may deduct X" and "X is deductible" is voice, not
    # evidence, so both are this one kind. Must cite at least one of the
    # evidence pack's `[n]` markers (verifier.py, `NO_CITATION`) — no
    # allowance for an uncited "connective" line, found to be a real
    # injection-resistance gap while this phase was built (ADR-120).
    CONTENT = "content"
    # A restatement of the calculator's trace, marked `[calc]` instead of a
    # numeric citation. Its numbers must come from the computation, not a
    # passage.
    COMPUTATION = "computation"
    # "The Act does not deal with X." Carries no citation at all — gated
    # harder for exactly that reason (see verifier.py).
    NO_BASIS = "no_basis"


class Citation(BaseModel):
    """Resolved by the verifier from a `[n]` marker — never trusted from the
    model, which supplies only the bare integer inline in the claim's text."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    marker: int = Field(ge=1)
    path: str = Field(min_length=1)
    # An excerpt of the cited passage, for the frontend's citation popup —
    # not a model-chosen quote (there is none to choose anymore).
    quote: str = Field(min_length=1)


class Claim(BaseModel):
    """One released line, its citations filled in by the verifier."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    type: ClaimType
    text: str = Field(min_length=1)
    citations: tuple[Citation, ...] = ()


class MalformedClaim(ValueError):
    """A line with no usable content once markers are stripped, or one that
    is nothing but a marker. Verification treats it as a failed claim."""


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


def classify_line(text: str) -> ClaimType:
    if _HEADING.match(text):
        return ClaimType.HEADING
    if text.startswith(NO_BASIS_OPENERS):
        return ClaimType.NO_BASIS
    if CALC_MARKER in text:
        return ClaimType.COMPUTATION
    return ClaimType.CONTENT


def parse_claim(line: str) -> Claim:
    text = line.strip()
    without_markers = MARKER.sub("", text.replace(CALC_MARKER, ""))
    if not without_markers.lstrip("#-* ").strip():
        raise MalformedClaim("no content once markers and markdown punctuation are stripped")
    return Claim(type=classify_line(text), text=text)


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
