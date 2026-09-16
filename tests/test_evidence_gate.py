"""Step 12.3 — the evidence gate, simplified (ADR-110). Updated for R19 Phase
B's marker-based citation (ADR-120).

Pure and deterministic: no LLM, no corpus. Chunks are built by hand with
`Chunk.create`, which is enough to construct an `EvidencePack` without the
built chunk store.
"""

from __future__ import annotations

from taxverity.chunking.models import Chunk
from taxverity.corpus.nodes import NodeType
from taxverity.generation.claims import Citation, ClaimEvent, ClaimType, WithheldEvent
from taxverity.retrieval.evidence import EvidencePack, EvidenceRole, EvidenceUnit
from taxverity.safety.evidence_gate import (
    EVIDENCE_GATE_STAGE_VERSION,
    INSUFFICIENT_EVIDENCE_MESSAGE,
    gate,
    served_grounded_claims,
)

CORPUS_VERSION = "test-v1"


def make_chunk(node_path: str = "22") -> Chunk:
    return Chunk.create(
        CORPUS_VERSION,
        node_path,
        "Thirty per cent of the annual value as determined under section 21.",
        node_type=NodeType.SECTION,
        section_number=node_path,
        parent_id=None,
        doc_id="act",
        page_start=1,
        page_end=1,
    )


def make_pack(*, with_unit: bool = True) -> EvidencePack:
    units = ()
    if with_unit:
        units = (
            EvidenceUnit(
                chunk=make_chunk(),
                context=(),
                rank=1,
                tokens=20,
                role=EvidenceRole.RETRIEVED,
            ),
        )
    return EvidencePack(units=units, budget=4_000, skipped=())


def content_event(claim_id: int = 1) -> ClaimEvent:
    return ClaimEvent(
        id=claim_id,
        type=ClaimType.CONTENT,
        text="Thirty per cent of the annual value is deductible [1].",
        citations=(Citation(marker=1, path="22", quote="Thirty per cent of the annual value"),),
    )


def uncited_content_event(claim_id: int = 1) -> ClaimEvent:
    """A connective `content` line the verifier let through with no citation
    at all — R19 Phase B removed that allowance (a safety gap found while
    building it, ADR-120), but this shape is still worth a defensive test:
    should any future change reopen an uncited-content path, it still must
    not satisfy the gate on its own."""
    return ClaimEvent(id=claim_id, type=ClaimType.CONTENT, text="Here's what applies.", citations=())


def no_basis_event(claim_id: int = 1) -> ClaimEvent:
    return ClaimEvent(
        id=claim_id,
        type=ClaimType.NO_BASIS,
        text="The Act does not deal with this.",
        citations=(),
    )


def computation_event(claim_id: int = 1) -> ClaimEvent:
    return ClaimEvent(id=claim_id, type=ClaimType.COMPUTATION, text="Tax is 50000.", citations=())


def withheld_event(claim_id: int = 1) -> WithheldEvent:
    return WithheldEvent(id=claim_id, reason="marker_not_in_evidence")


def test_stage_version_is_declared():
    assert EVIDENCE_GATE_STAGE_VERSION == 3


# --- served_grounded_claims ----------------------------------------------------


def test_counts_cited_content_but_not_computation_or_no_basis_or_uncited_content():
    events = [
        content_event(1),
        computation_event(2),
        withheld_event(3),
        uncited_content_event(4),
        no_basis_event(5),
    ]
    assert served_grounded_claims(events) == 1


def test_zero_for_no_events():
    assert served_grounded_claims([]) == 0


def test_zero_when_only_computation_claims_are_served():
    assert served_grounded_claims([computation_event(1)]) == 0


def test_zero_when_only_no_basis_claims_are_served():
    # A no_basis claim cites nothing, by construction - it must not be able
    # to satisfy the gate on its own (advisor pivot).
    assert served_grounded_claims([no_basis_event(1)]) == 0


def test_zero_when_only_an_uncited_content_claim_is_served():
    assert served_grounded_claims([uncited_content_event(1)]) == 0


# --- gate ----------------------------------------------------------------


def test_an_empty_pack_gates_regardless_of_events():
    empty = make_pack(with_unit=False)
    assert gate(empty, [content_event()]) == INSUFFICIENT_EVIDENCE_MESSAGE


def test_a_non_empty_pack_with_a_served_cited_content_claim_does_not_gate():
    assert gate(make_pack(), [content_event()]) is None


def test_a_non_empty_pack_with_only_an_uncited_content_claim_still_gates():
    pack = make_pack()
    assert gate(pack, [uncited_content_event()]) == INSUFFICIENT_EVIDENCE_MESSAGE


def test_a_non_empty_pack_with_only_a_no_basis_claim_still_gates():
    pack = make_pack()
    assert gate(pack, [no_basis_event()]) == INSUFFICIENT_EVIDENCE_MESSAGE


def test_a_non_empty_pack_with_zero_content_claims_gates():
    pack = make_pack()
    assert gate(pack, [computation_event()]) == INSUFFICIENT_EVIDENCE_MESSAGE
    assert gate(pack, [withheld_event()]) == INSUFFICIENT_EVIDENCE_MESSAGE
    assert gate(pack, []) == INSUFFICIENT_EVIDENCE_MESSAGE


def test_one_cited_content_claim_among_several_events_is_enough():
    pack = make_pack()
    events = [withheld_event(1), content_event(2), computation_event(3)]
    assert gate(pack, events) is None


def test_the_message_is_a_fixed_template_not_derived_from_the_pack():
    # Same pack, different outcomes: the message text itself never varies.
    empty_result = gate(make_pack(with_unit=False), [])
    zero_claims_result = gate(make_pack(), [computation_event()])
    assert empty_result == zero_claims_result == INSUFFICIENT_EVIDENCE_MESSAGE
