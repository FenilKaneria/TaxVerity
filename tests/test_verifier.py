"""R19 Phase B (ADR-120) — the grounding gate, marker-based. Adversarial
fixtures must all be caught.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from taxverity.calculator.scope import CalculatorInputs, run
from taxverity.chunking.models import Chunk
from taxverity.corpus.nodes import NodePath, NodeType
from taxverity.facts import Fact, FactField, FactStatus, UserFacts
from taxverity.generation.claims import Claim, ClaimType
from taxverity.generation.verifier import (
    Verifier,
    Violation,
    canonical_path,
    numbers_in,
)
from taxverity.reasoning.models import (
    AnswerPlan,
    CheckStatus,
    ConclusionKind,
    Condition,
    ConditionCheck,
    LegalRule,
)
from taxverity.reasoning.validate import ValidatedAnalysis
from taxverity.retrieval.base import ScoredChunk
from taxverity.retrieval.evidence import EvidencePacker

CORPUS_VERSION = "v" * 64
TYPES = (NodeType.SECTION, NodeType.SUBSECTION, NodeType.CLAUSE)

S22 = (
    "22",
    "22. Deductions from income from house property.",
    [
        (
            "22(1)",
            "(1) The following deductions shall be made from the annual value—",
            [
                ("22(1)(a)", "(a) thirty per cent of the annual value;", []),
                ("22(1)(b)", "(b) interest payable on borrowed capital.", []),
            ],
        ),
        ("22(2)", "(2) No deduction shall be made for any other sum.", []),
    ],
)
S24 = ("24", "24. The deduction shall not exceed Rs. 2,00,000 in a tax year.", [])
S23 = ("23", "23. Arrears of rent received shall be charged.", [])


def full_text(node) -> str:
    _, own, kids = node
    return "\n".join([own, *(full_text(kid) for kid in kids)])


def build(node, root, start=0, parent_id=None):
    citation, own, kids = node
    text = full_text(node)
    made = Chunk.create(
        CORPUS_VERSION,
        citation,
        text,
        parent_id=parent_id,
        doc_id="income-tax-act-2025",
        node_type=TYPES[NodePath.parse(citation).depth - 1],
        section_number=root,
        page_start=1,
        page_end=1,
        char_start=start,
        char_end=start + len(text),
    )
    chunks = [made]
    cursor = start + len(own) + 1
    for kid in kids:
        chunks += build(kid, root, cursor, made.chunk_id)
        cursor += len(full_text(kid)) + 1
    return chunks


CHUNKS = {c.node_path: c for c in (*build(S22, "22"), *build(S24, "24"), *build(S23, "23"))}
# Marker [1] = 22(1) (with 22's lead-in as context), marker [2] = 24. Neither
# 22(2) nor 23 is packed — nothing can cite them.
PACK = EvidencePacker(CHUNKS.values()).pack(
    [ScoredChunk(chunk=CHUNKS["22(1)"], score=2.0), ScoredChunk(chunk=CHUNKS["24"], score=1.0)]
)

QUESTION = "I earn rent of 3,00,000 a year. What can I deduct?"
FACTS = UserFacts(
    facts=(
        Fact(
            field=FactField.SALARY_INCOME,
            status=FactStatus.STATED,
            raw_value="14,00,000",
            value=Decimal("1400000"),
            source_span="salary is 14,00,000",
        ),
    )
)


def content(text: str) -> Claim:
    return Claim(type=ClaimType.CONTENT, text=text)


def heading(text: str) -> Claim:
    return Claim(type=ClaimType.HEADING, text=text)


def no_basis(text: str) -> Claim:
    return Claim(type=ClaimType.NO_BASIS, text=text)


def computation_claim(text: str) -> Claim:
    return Claim(type=ClaimType.COMPUTATION, text=text)


def application(text: str) -> Claim:
    return Claim(type=ClaimType.APPLICATION, text=text)


def unknown(text: str) -> Claim:
    return Claim(type=ClaimType.UNKNOWN, text=text)


# c1's condition sits on rule marker [1] (22(1)), satisfied; c2's on [2]
# (24), unknown — R20 Step 20.7's fixture for APPLICATION/UNKNOWN claims.
ANALYSIS = ValidatedAnalysis(
    legal_rules=(
        LegalRule(
            id="r1",
            markers=(1,),
            rule="Thirty per cent of the annual value is deducted.",
            conditions=(Condition(id="c1", text="The property is let out.", markers=()),),
        ),
        LegalRule(
            id="r2",
            markers=(2,),
            rule="The deduction shall not exceed Rs. 2,00,000 in a tax year.",
            conditions=(Condition(id="c2", text="The claim exceeds the cap.", markers=()),),
        ),
    ),
    applicability=(
        ConditionCheck(condition_id="c1", status=CheckStatus.SATISFIED, fact_refs=("salary_income",)),
        ConditionCheck(condition_id="c2", status=CheckStatus.UNKNOWN),
    ),
    missing_facts=(),
    answer_plan=AnswerPlan(conclusion_kind=ConclusionKind.CONDITIONAL),
)


def violations(claim: Claim, **kwargs) -> set[Violation]:
    verifier = Verifier(PACK, question=QUESTION, facts=FACTS, **kwargs)
    return {finding.violation for finding in verifier.verify(claim).findings}


def test_the_pack_is_what_these_tests_assume():
    assert [unit.citation for unit in PACK.units] == ["22(1)", "24"]
    assert [line.citation for line in PACK.units[0].context] == ["22"]


# --- claims that must pass ---------------------------------------------------


@pytest.mark.parametrize(
    "claim",
    [
        content("Deductions are made from the annual value [1]."),
        # A node inside a packed unit is grounded by the packed unit's own
        # marker — 22(1)(a) has no marker of its own, but its text is inside
        # unit [1] (ADR-055).
        content("Thirty per cent is deducted [1]."),
        # Indian grouping and lakh wording both agree with the source figure.
        content("The cap is 2 lakh [2]."),
        # The Act states this one only in words — a digit claim must still
        # ground against it (R19 Phase B's word-number grounding).
        content("The deduction shall not exceed two lakh rupees [2]."),
    ],
)
def test_a_grounded_claim_passes(claim):
    assert violations(claim) == set()


def test_a_citation_is_resolved_to_the_units_path_and_an_excerpt():
    verifier = Verifier(PACK)
    verdict = verifier.verify(content("Thirty per cent [1]."))
    assert verdict.passed
    assert verdict.claim.citations[0].marker == 1
    assert verdict.claim.citations[0].path == "22(1)"
    assert "thirty per cent" in verdict.claim.citations[0].quote


def test_multiple_markers_on_one_line_both_ground():
    claim = content("Thirty per cent is deducted, capped at 2 lakh [1][2].")
    assert violations(claim) == set()


# --- adversarial fixtures: every one must be caught ---------------------------


@pytest.mark.parametrize(
    ("claim", "expected"),
    [
        # A marker with no evidence at that position at all.
        (content("A deduction exists [99]."), Violation.MARKER_NOT_IN_EVIDENCE),
        (content("A deduction exists [0]."), Violation.MARKER_NOT_IN_EVIDENCE),
        # A real provision that was never packed — the model cannot cite it,
        # because there is no marker number for it. Only whole pack units
        # are citable (22(2) and 23 are both real but unpacked).
        (content("No other deduction [3]."), Violation.MARKER_NOT_IN_EVIDENCE),
        # No marker at all — required unconditionally, no blocklist of
        # trigger words to work around (see the check's own comment for why).
        (content("Thirty per cent of the annual value is deducted."), Violation.NO_CITATION),
        (content("You are entitled to this deduction."), Violation.NO_CITATION),
        # A plain-looking sentence with no statutory vocabulary at all must
        # still be caught uncited — an allowlist-by-absence-of-keywords is
        # exactly the injection-resistance gap this design closes.
        (content("No tax is ever due."), Violation.NO_CITATION),
        # An invented number, with a genuine citation beside it.
        (content("40 per cent is deducted [1]."), Violation.UNSUPPORTED_NUMBER),
        # A changed amount.
        (content("The cap is 5 lakh [2]."), Violation.UNSUPPORTED_NUMBER),
        # A figure from the question cannot ground a statement of law.
        (content("The cap is 3 lakh [2]."), Violation.UNSUPPORTED_NUMBER),
        # Nor can a stated fact.
        (content("The cap is 14,00,000 [2]."), Violation.UNSUPPORTED_NUMBER),
        # A number from a marker that itself failed does not ground the text.
        (content("The cap is 2,00,000 [99]."), Violation.UNSUPPORTED_NUMBER),
        # A computation claim with no computation.
        (computation_claim("Your tax is nil [calc]."), Violation.NO_COMPUTATION),
    ],
)
def test_an_ungrounded_claim_is_caught(claim, expected):
    assert expected in violations(claim)


def test_the_users_figures_ground_a_computation_claim(computation):
    claim = computation_claim("On rent of 3,00,000 and a salary of 14,00,000, see the trace [calc].")
    assert violations(claim, computation=computation) == set()


def test_a_profile_default_grounds_no_number(computation):
    facts = UserFacts(
        facts=(
            Fact(
                field=FactField.SALARY_INCOME,
                status=FactStatus.PROFILE_DEFAULT,
                raw_value="9,99,999",
                value=Decimal("999999"),
                source_span="",
            ),
        )
    )
    claim = computation_claim("A salary of 9,99,999 [calc].")
    verifier = Verifier(PACK, facts=facts, computation=computation)
    assert Violation.UNSUPPORTED_NUMBER in {f.violation for f in verifier.verify(claim).findings}


# --- computation claims ---------------------------------------------------------


@pytest.fixture(scope="module")
def computation():
    return run(
        CalculatorInputs(
            tax_year="2026-27",
            salary=Decimal("1500000"),
            other_income=Decimal("0"),
            resident_individual=True,
            claimed={},
            tax_deducted_at_source=None,
            advance_tax=None,
        )
    )


def test_a_computation_claim_restating_the_trace_passes(computation):
    payable = computation.comparison.under_202_1.payable.amount
    claim = computation_claim(f"For tax year 2026-27 the income-tax payable is {payable:,} [calc].")
    assert violations(claim, computation=computation) == set()


def test_a_computation_claim_with_an_invented_figure_is_caught(computation):
    payable = computation.comparison.under_202_1.payable.amount
    claim = computation_claim(f"The income-tax payable is {payable + 1} [calc].")
    assert violations(claim, computation=computation) == {Violation.UNSUPPORTED_NUMBER}


def test_computation_figures_do_not_ground_a_content_claim(computation):
    payable = computation.comparison.under_202_1.payable.amount
    claim = content(f"The Act fixes the tax at {payable + 7} [1].")
    assert Violation.UNSUPPORTED_NUMBER in violations(claim, computation=computation)


# --- headings ------------------------------------------------------------------


def test_a_clean_heading_passes():
    assert violations(heading("## Deductions from house property")) == set()


def test_a_heading_carrying_a_number_is_malformed():
    assert Violation.MALFORMED_HEADING in violations(heading("## Save up to 2 lakh"))


def test_a_heading_carrying_a_marker_is_malformed():
    assert Violation.MALFORMED_HEADING in violations(heading("## Deductions [1]"))


# --- modal mismatch (new this phase) --------------------------------------------


def test_a_claim_affirming_what_its_source_denies_is_caught():
    # Unit [2] (24) says the deduction "shall not exceed" 2 lakh — a genuine
    # denial. A claim asserting entitlement to more than that, from the same
    # source, gets the fact wrong even though its number and marker both
    # check out.
    claim = content("You are entitled to deduct more than 2 lakh [2].")
    assert Violation.MODAL_MISMATCH in violations(claim)


def test_a_claim_agreeing_with_a_denying_source_is_not_flagged():
    claim = content("You cannot deduct more than 2 lakh [2].")
    assert Violation.MODAL_MISMATCH not in violations(claim)


def test_an_overly_cautious_claim_is_not_flagged():
    # The reverse direction (claim denies, source affirms) is deliberately
    # not gated — see the module's own docstring for why.
    claim = content("You are entitled to deduct interest on borrowed capital [1].")
    assert Violation.MODAL_MISMATCH not in violations(claim)


# --- no_basis claims -------------------------------------------------------------


def test_a_well_formed_no_basis_claim_passes():
    claim = no_basis("The Act does not deal with cryptocurrency donations.")
    assert violations(claim) == set()


@pytest.mark.parametrize(
    "opener",
    ["The Act is silent on gifts of art.", "Nothing in the Act addresses this."],
)
def test_every_registered_opener_passes(opener):
    assert violations(no_basis(opener)) == set()


def test_a_no_basis_claim_citing_evidence_is_malformed():
    claim = no_basis("The Act does not deal with this [1].")
    assert Violation.MALFORMED_NO_BASIS in violations(claim)


def test_a_no_basis_claim_with_the_wrong_opener_is_malformed():
    claim = no_basis("This is not covered by the Act.")
    assert Violation.MALFORMED_NO_BASIS in violations(claim)


def test_a_no_basis_claim_stating_a_number_is_unsupported():
    claim = no_basis("The Act does not deal with gifts of 500 acres.")
    assert Violation.MALFORMED_NO_BASIS in violations(claim)


# --- application claims (R20 Step 20.7) -------------------------------------------


def test_an_application_claim_grounded_by_a_satisfied_condition_passes():
    claim = application("You can deduct thirty per cent of the annual value [1][fact].")
    assert violations(claim, analysis=ANALYSIS) == set()


def test_an_application_claim_may_use_a_stated_fact():
    claim = application("Against your salary of 14,00,000, you can deduct this [1][fact].")
    assert violations(claim, analysis=ANALYSIS) == set()


def test_an_application_claim_with_no_citation_is_uncited():
    assert Violation.NO_CITATION in violations(application("You qualify [fact]."), analysis=ANALYSIS)


def test_an_application_claim_against_an_unknown_condition_is_caught():
    # This is the HRA-to-mother shape: an affirmative conclusion resting on
    # a condition the analysis could not check.
    claim = application("You are entitled to deduct up to 2 lakh [2][fact].")
    assert Violation.UNSUPPORTED_APPLICATION in violations(claim, analysis=ANALYSIS)


def test_an_application_claim_with_no_affirmative_modal_is_not_flagged():
    # No affirmative modal ("you can", "is allowed", ...) at all, so the
    # check never fires even against an unknown condition — the same
    # one-directional, safer-failure-mode design as MODAL_MISMATCH.
    claim = application("This deduction depends on facts not yet known, up to 2 lakh [2][fact].")
    assert Violation.UNSUPPORTED_APPLICATION not in violations(claim, analysis=ANALYSIS)


def test_an_application_claim_with_no_analysis_skips_the_condition_check():
    claim = application("You can deduct thirty per cent [1][fact].")
    assert violations(claim) == set()


def test_an_application_claim_still_needs_a_grounded_number():
    claim = application("You can deduct forty per cent [1][fact].")
    assert Violation.UNSUPPORTED_NUMBER in violations(claim, analysis=ANALYSIS)


# --- unknown claims (R20 Step 20.7) -------------------------------------------------


def test_an_unknown_claim_naming_a_genuinely_unknown_condition_passes():
    claim = unknown("This can't yet be determined because the cap may already be used [2].")
    assert violations(claim, analysis=ANALYSIS) == set()


def test_an_unknown_claim_with_the_wrong_opener_is_malformed():
    assert Violation.MALFORMED_UNKNOWN in violations(
        unknown("We don't know yet [2]."), analysis=ANALYSIS
    )


def test_an_unknown_claim_with_no_citation_is_malformed():
    assert Violation.MALFORMED_UNKNOWN in violations(
        unknown("This can't yet be determined because more is needed."), analysis=ANALYSIS
    )


def test_an_unknown_claim_stating_a_number_is_malformed():
    claim = unknown("This can't yet be determined because the cap is 2 lakh [2].")
    assert Violation.MALFORMED_UNKNOWN in violations(claim, analysis=ANALYSIS)


def test_an_unknown_claim_against_a_satisfied_condition_is_malformed():
    # [1] is c1, and c1 is satisfied, not unknown — this claim is fabricating
    # uncertainty that the analysis does not have.
    claim = unknown("This can't yet be determined because it isn't clear [1].")
    assert Violation.MALFORMED_UNKNOWN in violations(claim, analysis=ANALYSIS)


# --- helpers -----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "path"),
    [
        ("22(2)", "22(2)"),
        ("Section 22(2)", "22(2)"),
        ("s. 22 (2)", "22(2)"),
        ("u/s 354A(1)", "354A(1)"),
        ("Schedule XV(1)", "Schedule XV(1)"),
        ("section twenty-two", None),
    ],
)
def test_canonical_path(raw, path):
    assert canonical_path(raw) == path


def test_numbers_in_reads_indian_grouping_and_words():
    assert numbers_in("Rs. 12,00,000 or 12 lakh or 1.5 crore, 30%") == {
        Decimal("1200000"),
        Decimal("15000000"),
        Decimal("30"),
    }


def test_numbers_in_reads_english_number_words():
    assert numbers_in("fifteen lakh rupees") == {Decimal("1500000")}
    assert numbers_in("one lakh fifty thousand") == {Decimal("150000")}
    assert numbers_in("thirty per cent") == {Decimal("30")}
    assert numbers_in("twenty one") == {Decimal("21")}


def test_a_bare_one_or_zero_is_not_read_as_a_figure():
    # Common non-numeric English usage ("one such condition") must not be
    # treated as stating a quantity — only "one" combined with something else
    # (a scale word, another number word) counts.
    assert numbers_in("one such condition applies") == frozenset()
    assert numbers_in("zero tolerance for late filing") == frozenset()
    assert numbers_in("one lakh rupees") == {Decimal("100000")}
