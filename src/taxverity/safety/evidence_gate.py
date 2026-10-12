"""The evidence gate, simplified."""

from __future__ import annotations

from collections.abc import Sequence

from taxverity.generation.claims import ClaimEvent, ClaimType, WithheldEvent
from taxverity.retrieval.evidence import EvidencePack

EVIDENCE_GATE_STAGE_VERSION = 5

INSUFFICIENT_EVIDENCE_MESSAGE = (
    "I couldn't find anything in the Income-tax Act, 2025 that answers this — "
    "I don't have a grounded basis for an answer here."
)


# Fixed, never generated. Leads a gated answer whose
# only served lines are general guidance.
GUIDANCE_ONLY_MESSAGE = (
    "The Act doesn't directly cover this. General guidance below is not from the Act."
)


GROUNDED_CLAIM_TYPES = (ClaimType.CONTENT, ClaimType.APPLICATION)


def served_grounded_claims(events: Sequence[ClaimEvent | WithheldEvent]) -> int:
    """A `content` or `application` claim counts only once it actually carries a
    resolved citation — the merged statute/advice voice also covers a connective line
    with none, which restates nothing about the Act."""
    return sum(
        1
        for event in events
        if isinstance(event, ClaimEvent)
        and event.type in GROUNDED_CLAIM_TYPES
        and event.citations
    )


def gate(
    pack: EvidencePack, events: Sequence[ClaimEvent | WithheldEvent]
) -> str | None:
    """The fixed insufficient-evidence response, or None to serve the answer."""
    if not pack.units:
        return INSUFFICIENT_EVIDENCE_MESSAGE
    if served_grounded_claims(events) == 0:
        return INSUFFICIENT_EVIDENCE_MESSAGE
    return None


def guidance_lines(events: Sequence[ClaimEvent | WithheldEvent]) -> list[ClaimEvent]:
    return [
        event
        for event in events
        if isinstance(event, ClaimEvent) and event.type is ClaimType.GUIDANCE
    ]


def release(
    pack: EvidencePack, events: Sequence[ClaimEvent | WithheldEvent]
) -> tuple[str | None, list[ClaimEvent | WithheldEvent]]:
    """The fixed text that leads the answer (None for a served answer) and the events
    that may reach the person."""
    text = gate(pack, events)
    if text is None:
        return None, list(events)
    guidance = guidance_lines(events)
    if guidance:
        return GUIDANCE_ONLY_MESSAGE, list(guidance)
    return text, []
