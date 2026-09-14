"""Step 10.5 — the grounding gate. Adversarial fixtures must all be caught."""

from __future__ import annotations

from decimal import Decimal

import pytest

from taxverity.calculator.scope import CalculatorInputs, run
from taxverity.chunking.models import Chunk
from taxverity.corpus.nodes import NodePath, NodeType
from taxverity.facts import Fact, FactField, FactStatus, UserFacts
from taxverity.generation.claims import Citation, Claim, ClaimType
from taxverity.generation.verifier import (
    Verifier,
    Violation,
    canonical_path,
    numbers_in,
)
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
S24 = ("24", "24. The deduction shall not exceed Rs. 2,00,000 in a tax year.", [])
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
# Retrieved: 22(1) (with 22's lead-in as context) and 24. Not 22(2), not 23.
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


def statute(text: str, *citations: tuple[str, str]) -> Claim:
    return Claim(
        type=ClaimType.STATUTE,
        text=text,
        citations=tuple(Citation(path=p, quote=q) for p, q in citations),
    )


def advice(text: str, *citations: tuple[str, str]) -> Claim:
    return Claim(
        type=ClaimType.ADVICE,
        text=text,
        citations=tuple(Citation(path=p, quote=q) for p, q in citations),
    )


def no_basis(text: str, *citations: tuple[str, str]) -> Claim:
    return Claim(
        type=ClaimType.NO_BASIS,
        text=text,
        citations=tuple(Citation(path=p, quote=q) for p, q in citations),
    )


def violations(claim: Claim, **kwargs) -> set[Violation]:
    verifier = Verifier(PACK, CHUNKS, question=QUESTION, facts=FACTS, **kwargs)
    return {finding.violation for finding in verifier.verify(claim).findings}


def test_the_pack_is_what_these_tests_assume():
    assert [unit.citation for unit in PACK.units] == ["22(1)", "24"]
    assert [line.citation for line in PACK.units[0].context] == ["22"]


# --- claims that must pass ---------------------------------------------------


@pytest.mark.parametrize(
    "claim",
    [
        statute(
            "Deductions are made from the annual value.",
            ("22(1)", "The following deductions shall be made from the annual value"),
        ),
        # A node inside a packed unit is in evidence (ADR-055).
        statute("Thirty per cent is deducted.", ("22(1)(a)", "thirty per cent of the annual value")),
        # A context line is citable for its own lead-in.
        statute("Section 22 concerns house property.", ("22", "Deductions from income from house property")),
        # Figures agree across Indian grouping and lakh wording.
        statute("The cap is 2 lakh.", ("24", "shall not exceed Rs. 2,00,000")),
        # NBSPs in source and quote, a newline in the quote: compared after normalising.
        statute("The cap applies per tax year.", ("24", "The deduction shall\nnot exceed")),
    ],
)
def test_a_grounded_claim_passes(claim):
    assert violations(claim) == set()


def test_a_citation_is_released_in_canonical_form():
    verifier = Verifier(PACK, CHUNKS)
    verdict = verifier.verify(
        statute("Thirty per cent.", ("Section 22 (1)(a)", "thirty per cent of the annual value"))
    )
    assert verdict.passed
    assert verdict.claim.citations[0].path == "22(1)(a)"


# --- adversarial fixtures: every one must be caught ---------------------------


@pytest.mark.parametrize(
    ("claim", "expected"),
    [
        # Fabricated path.
        (statute("A deduction exists.", ("999(1)", "The following deductions shall be made")), Violation.CITATION_NOT_IN_EVIDENCE),
        # Not a path at all.
        (statute("A deduction exists.", ("the house property rules", "The following deductions shall be made")), Violation.CITATION_NOT_IN_EVIDENCE),
        # A real provision that was not retrieved.
        (statute("Arrears are charged.", ("23", "Arrears of rent received shall be charged")), Violation.CITATION_NOT_IN_EVIDENCE),
        # A sibling of a context line: its ancestor's lead-in is in evidence, it is not.
        (statute("No other deduction.", ("22(2)", "No deduction shall be made for any other sum")), Violation.CITATION_NOT_IN_EVIDENCE),
        # A context line's citation quoting text beyond its lead-in.
        (statute("No other deduction.", ("22", "No deduction shall be made for any other sum")), Violation.QUOTE_NOT_IN_SOURCE),
        # Altered quote: one word changed.
        (statute("Forty per cent.", ("22(1)(a)", "forty per cent of the annual value")), Violation.QUOTE_NOT_IN_SOURCE),
        # Right text, wrong provision.
        (statute("Interest is deducted.", ("22(1)(a)", "interest payable on borrowed capital")), Violation.QUOTE_NOT_IN_SOURCE),
        # A quote too short to identify anything.
        (statute("Annual value matters.", ("22(1)(a)", "annual value")), Violation.QUOTE_TOO_SHORT),
        # Uncited statute claim.
        (statute("Thirty per cent of the annual value is deducted."), Violation.NO_CITATION),
        # Invented number, with a genuine citation beside it.
        (statute("40 per cent is deducted.", ("22(1)(a)", "thirty per cent of the annual value")), Violation.UNSUPPORTED_NUMBER),
        # A changed amount.
        (statute("The cap is 5 lakh.", ("24", "shall not exceed Rs. 2,00,000")), Violation.UNSUPPORTED_NUMBER),
        # A figure from the question cannot ground a statement of law.
        (statute("The cap is 3 lakh.", ("24", "shall not exceed Rs. 2,00,000")), Violation.UNSUPPORTED_NUMBER),
        # Nor can a stated fact.
        (statute("The cap is 14,00,000.", ("24", "shall not exceed Rs. 2,00,000")), Violation.UNSUPPORTED_NUMBER),
        # A number from a citation that failed does not ground the text.
        (statute("The cap is 2,00,000.", ("23", "shall not exceed Rs. 2,00,000")), Violation.UNSUPPORTED_NUMBER),
        # A computation claim with no computation.
        (Claim(type=ClaimType.COMPUTATION, text="Your tax is nil."), Violation.NO_COMPUTATION),
    ],
)
def test_an_ungrounded_claim_is_caught(claim, expected):
    assert expected in violations(claim)


def test_the_users_figures_ground_a_computation_claim(computation):
    claim = Claim(
        type=ClaimType.COMPUTATION,
        text="On rent of 3,00,000 and a salary of 14,00,000, see the trace.",
    )
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
    claim = Claim(type=ClaimType.COMPUTATION, text="A salary of 9,99,999.")
    verifier = Verifier(PACK, CHUNKS, facts=facts, computation=computation)
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
    claim = Claim(
        type=ClaimType.COMPUTATION,
        text=f"For tax year 2026-27 the income-tax payable is {payable:,}.",
    )
    assert violations(claim, computation=computation) == set()


def test_a_computation_claim_with_an_invented_figure_is_caught(computation):
    payable = computation.comparison.under_202_1.payable.amount
    claim = Claim(type=ClaimType.COMPUTATION, text=f"The income-tax payable is {payable + 1}.")
    assert violations(claim, computation=computation) == {Violation.UNSUPPORTED_NUMBER}


def test_computation_figures_do_not_ground_a_statute_claim(computation):
    payable = computation.comparison.under_202_1.payable.amount
    claim = statute(
        f"The Act fixes the tax at {payable + 7}.",
        ("22(1)", "deductions shall be made from the annual value"),
    )
    assert Violation.UNSUPPORTED_NUMBER in violations(claim, computation=computation)


# --- advisor pivot: ADVICE claims gated like STATUTE ---------------------------


def test_a_grounded_advice_claim_passes():
    # "shall" in 22(1)'s own text is a statutory modal, so the prescriptive
    # "you may" is backed by a quote that actually imposes/permits something.
    claim = advice(
        "You may deduct thirty per cent of the annual value from your rental income.",
        ("22(1)", "The following deductions shall be made from the annual value"),
    )
    assert violations(claim) == set()


def test_an_uncited_advice_claim_is_caught():
    assert Violation.NO_CITATION in violations(advice("You may claim this deduction."))


def test_advice_cannot_be_grounded_by_the_users_own_figures():
    # Same rule as a statute claim (rule 04's numbers-in-words-not-figures):
    # the question's own 3,00,000 must not ground an advice claim's number.
    claim = advice(
        "You can deduct up to 3,00,000.",
        ("24", "shall not exceed Rs. 2,00,000"),
    )
    assert Violation.UNSUPPORTED_NUMBER in violations(claim)


def test_a_prescription_with_no_modal_quote_is_unsupported_advice():
    # "thirty per cent of the annual value" is a bare description, not a
    # provision imposing or permitting anything - a "you should" resting on
    # it alone is asserted, not shown.
    claim = advice(
        "You should claim thirty per cent as a deduction.",
        ("22(1)(a)", "thirty per cent of the annual value"),
    )
    assert Violation.UNSUPPORTED_ADVICE in violations(claim)


def test_a_prescription_backed_by_a_modal_quote_passes():
    claim = advice(
        "You must deduct interest on your borrowed capital.",
        ("22(1)(a)", "thirty per cent of the annual value"),
        ("22(1)", "The following deductions shall be made from the annual value"),
    )
    assert Violation.UNSUPPORTED_ADVICE not in violations(claim)


def test_a_non_prescriptive_advice_claim_needs_no_modal_quote():
    # No "you should/must/can/may" language - the check does not apply.
    claim = advice(
        "Your annual value is reduced by thirty per cent of it.",
        ("22(1)(a)", "thirty per cent of the annual value"),
    )
    assert Violation.UNSUPPORTED_ADVICE not in violations(claim)


# --- advisor pivot: NO_BASIS claims ---------------------------------------------


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
    claim = no_basis(
        "The Act does not deal with this.",
        ("22(1)", "The following deductions shall be made from the annual value"),
    )
    assert Violation.MALFORMED_NO_BASIS in violations(claim)


def test_a_no_basis_claim_with_the_wrong_opener_is_malformed():
    claim = no_basis("This is not covered by the Act.")
    assert Violation.MALFORMED_NO_BASIS in violations(claim)


def test_a_no_basis_claim_stating_a_number_is_unsupported():
    claim = no_basis("The Act does not deal with gifts of 500 acres.")
    assert Violation.UNSUPPORTED_NUMBER in violations(claim)


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
