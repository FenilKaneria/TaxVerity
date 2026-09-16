"""R19 Phase B (ADR-120) — the claim model and the incremental markdown-line
parser. Same streaming contract as the old NDJSON design (`iter_lines()`,
`LineBuffer` unchanged); only what a line means changed.
"""

from __future__ import annotations

from pathlib import Path

from hypothesis import given, settings
from hypothesis import strategies as st

from taxverity.generation.claims import (
    DISCLAIMER,
    ClaimType,
    LineBuffer,
    MalformedClaim,
    iter_lines,
    parse_claim,
)

HEADING = "## Deductions from house property"
CONTENT = "Interest on borrowed capital is deducted [2]."
COMPUTATION = "Your tax payable is ₹0 [calc]."
NO_BASIS = "The Act does not deal with this."
ANSWER = "\n".join([HEADING, CONTENT, COMPUTATION, NO_BASIS])


def test_a_heading_line_classifies_as_heading():
    claim = parse_claim(HEADING)
    assert claim.type is ClaimType.HEADING
    assert claim.text == HEADING


def test_a_bullet_line_classifies_as_content():
    claim = parse_claim("- " + CONTENT)
    assert claim.type is ClaimType.CONTENT
    assert claim.citations == ()  # filled in by the verifier, not at parse time


def test_a_calc_marked_line_classifies_as_computation():
    claim = parse_claim(COMPUTATION)
    assert claim.type is ClaimType.COMPUTATION


def test_a_no_basis_opener_classifies_as_no_basis():
    claim = parse_claim(NO_BASIS)
    assert claim.type is ClaimType.NO_BASIS


def test_an_ordinary_sentence_classifies_as_content():
    claim = parse_claim("Here's what applies to your rental income.")
    assert claim.type is ClaimType.CONTENT


def test_a_line_that_is_only_a_marker_is_malformed():
    for degenerate in ["[1]", "[calc]", "- ", "## ", "-", "#"]:
        try:
            parse_claim(degenerate)
        except MalformedClaim:
            continue
        raise AssertionError(f"{degenerate!r} should have raised MalformedClaim")


def test_a_line_split_across_deltas_is_yielded_once_complete():
    buffer = LineBuffer()
    assert buffer.feed("## Deduc") == []
    assert buffer.feed("tions\n- Interest is d") == ["## Deductions"]
    assert buffer.feed("eductible [1].") == []
    assert buffer.flush() == ["- Interest is deductible [1]."]
    assert buffer.flush() == []


def test_the_last_line_without_a_newline_is_flushed():
    assert list(iter_lines([ANSWER])) == ANSWER.split("\n")


def test_blank_lines_and_code_fences_carry_no_claim():
    text = "```\n\n" + ANSWER + "\n\n```\n"
    assert list(iter_lines([text])) == ANSWER.split("\n")


def test_every_single_character_split_gives_the_same_lines():
    assert list(iter_lines(list(ANSWER))) == ANSWER.split("\n")


@settings(max_examples=60, deadline=None)
@given(st.lists(st.integers(min_value=0, max_value=len(ANSWER)), max_size=12))
def test_any_split_gives_the_same_claim_sequence(cuts):
    points = sorted(set(cuts))
    pieces = [ANSWER[a:b] for a, b in zip([0, *points], [*points, len(ANSWER)], strict=True)]
    assert [parse_claim(line) for line in iter_lines(pieces)] == [
        parse_claim(line) for line in ANSWER.split("\n")
    ]


def test_closing_the_lines_closes_the_source():
    closed = []

    def source():
        try:
            yield "- a [1].\n"
            yield "- b [2].\n"
        finally:
            closed.append(True)

    lines = iter_lines(source())
    next(lines)
    lines.close()
    assert closed == [True]


# --- Step 12.6: the disclaimer constant --------------------------------------


def test_the_disclaimer_matches_the_safety_policy_doc_verbatim():
    # The doc renders the disclaimer as a markdown blockquote ("> " per line);
    # strip that marker before collapsing whitespace, or it survives as a
    # stray token between words and breaks the substring match.
    lines = Path("docs/SAFETY_POLICY.md").read_text(encoding="utf-8").splitlines()
    policy = " ".join(
        " ".join(line.removeprefix(">").split()) for line in lines
    )
    assert " ".join(DISCLAIMER.split()) in policy


def test_the_disclaimer_is_not_dismissed_as_advice():
    # It must disclaim professional advice, not just describe the product -
    # a frontend rendering "informational" language alone would not satisfy
    # rule 03's legal-posture requirement.
    assert "not" in DISCLAIMER
    assert "professional" in DISCLAIMER.lower()
