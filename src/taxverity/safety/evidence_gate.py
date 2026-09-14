"""Step 12.3 — the evidence gate, simplified (ADR-110).

The plan originally called for a score threshold calibrated on the negative
slice. Steps 3.6, 4.6 and 5.6 already measured that no score — lexical,
cosine, or reranked — separates a negative question from an answerable one, so
no calibration is run; that path was not built by decision, not left undone.

The gate instead is the verifier outcome itself: an empty evidence pack, or a
generated answer that served zero grounded (`statute` or `advice`) claims,
means nothing traceable to the Act was found, and the fixed "insufficient
evidence" response is served instead of whatever text (if any) was generated.
A `computation` claim never substitutes for this — it restates the
calculator's trace, not a statutory answer — and neither does a `no_basis`
claim, which by construction cites nothing (advisor pivot). A question with a
computation but no statute/advice claim still gates.

Phase 13's graph runs the corrective retry (rule 02, ADR-110's minimal loop)
before calling this: it re-packs and re-generates once on a zero-grounded-claim
first pass. This function only judges the result that is finally in hand, and
does not know or care whether a retry already happened.
"""

from __future__ import annotations

from collections.abc import Sequence

from taxverity.generation.claims import ClaimEvent, ClaimType, WithheldEvent
from taxverity.retrieval.evidence import EvidencePack

EVIDENCE_GATE_STAGE_VERSION = 2

# Fixed template, not generated (same discipline as the safety classifier's
# templates, rule 03).
INSUFFICIENT_EVIDENCE_MESSAGE = (
    "I couldn't find anything in the Income-tax Act, 2025 that answers this — "
    "I don't have a grounded basis for an answer here."
)


GROUNDED_CLAIM_TYPES = (ClaimType.STATUTE, ClaimType.ADVICE)


def served_grounded_claims(events: Sequence[ClaimEvent | WithheldEvent]) -> int:
    """STATUTE and ADVICE both cite evidence; COMPUTATION and NO_BASIS do not
    restate the Act, so neither counts as a grounded statutory answer."""
    return sum(
        1
        for event in events
        if isinstance(event, ClaimEvent) and event.type in GROUNDED_CLAIM_TYPES
    )


def gate(
    pack: EvidencePack, events: Sequence[ClaimEvent | WithheldEvent]
) -> str | None:
    """The fixed insufficient-evidence response, or None to serve the answer.

    None means the caller's already-generated claim events stand as they are.
    A returned string replaces the whole answer.
    """
    if not pack.units:
        return INSUFFICIENT_EVIDENCE_MESSAGE
    if served_grounded_claims(events) == 0:
        return INSUFFICIENT_EVIDENCE_MESSAGE
    return None
