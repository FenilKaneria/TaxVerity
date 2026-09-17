"""Step 12.3 — the evidence gate, simplified (ADR-110). Updated for R19 Phase
B's marker-based citation (ADR-120) and R20 Step 20.7's `APPLICATION`/
`UNKNOWN` claim types.

The plan originally called for a score threshold calibrated on the negative
slice. Steps 3.6, 4.6 and 5.6 already measured that no score — lexical,
cosine, or reranked — separates a negative question from an answerable one, so
no calibration is run; that path was not built by decision, not left undone.

The gate instead is the verifier outcome itself: an empty evidence pack, or a
generated answer that served zero claims actually carrying a resolved
citation and grounding something about the Act, means nothing traceable to
the Act was found, and the fixed "insufficient evidence" response is served
instead of whatever text (if any) was generated. `content` and `application`
both count — an `application` line applies a cited, verified rule to the
person's own facts, which is exactly as much a grounded statement about the
Act as a plain `content` line, just phrased for the person rather than the
statute. A `computation` claim never substitutes for this — it restates the
calculator's trace, not a statutory answer — and neither does a `no_basis`
claim, which by construction cites nothing, nor an `unknown` claim, which by
construction declines to conclude anything, nor a connective `content` line
with no marker (rule 04's advisor pivot allows one, per verifier.py's
`STATUTORY_VOCAB` check, but it grounds nothing on its own). A question with
a computation but no cited content or application still gates.

Phase 13's graph runs the corrective retry (rule 02, ADR-110's minimal loop)
before calling this: it re-packs and re-generates once on a zero-grounded-claim
first pass. This function only judges the result that is finally in hand, and
does not know or care whether a retry already happened.
"""

from __future__ import annotations

from collections.abc import Sequence

from taxverity.generation.claims import ClaimEvent, ClaimType, WithheldEvent
from taxverity.retrieval.evidence import EvidencePack

EVIDENCE_GATE_STAGE_VERSION = 4

# Fixed template, not generated (same discipline as the safety classifier's
# templates, rule 03).
INSUFFICIENT_EVIDENCE_MESSAGE = (
    "I couldn't find anything in the Income-tax Act, 2025 that answers this — "
    "I don't have a grounded basis for an answer here."
)


GROUNDED_CLAIM_TYPES = (ClaimType.CONTENT, ClaimType.APPLICATION)


def served_grounded_claims(events: Sequence[ClaimEvent | WithheldEvent]) -> int:
    """A `content` or `application` claim counts only once it actually
    carries a resolved citation — the merged statute/advice voice (ADR-120)
    also covers a connective line with none, which restates nothing about
    the Act. `computation`, `no_basis` and `unknown` never count (see this
    module's docstring)."""
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
    """The fixed insufficient-evidence response, or None to serve the answer.

    None means the caller's already-generated claim events stand as they are.
    A returned string replaces the whole answer.
    """
    if not pack.units:
        return INSUFFICIENT_EVIDENCE_MESSAGE
    if served_grounded_claims(events) == 0:
        return INSUFFICIENT_EVIDENCE_MESSAGE
    return None
