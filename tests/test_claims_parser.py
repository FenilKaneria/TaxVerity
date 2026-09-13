"""Steps 10.2 and 10.4 — the claim model and the incremental NDJSON parser."""

from __future__ import annotations

import json

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

from taxverity.generation.claims import (
    Citation,
    ClaimEvent,
    ClaimType,
    LineBuffer,
    MalformedClaim,
    iter_lines,
    parse_claim,
)

STATUTE = {
    "type": "statute",
    "text": "Thirty per cent of the annual value is deducted.",
    "citations": [{"path": "22(1)(a)", "quote": "thirty per cent of the annual value"}],
}
COMPUTATION = {"type": "computation", "text": "The tax payable is 0.", "citations": []}
NDJSON = "\n".join(json.dumps(line) for line in (STATUTE, COMPUTATION, STATUTE))


def test_a_well_formed_line_parses():
    claim = parse_claim(json.dumps(STATUTE))
    assert claim.type is ClaimType.STATUTE
    assert claim.citations == (
        Citation(path="22(1)(a)", quote="thirty per cent of the annual value"),
    )


@pytest.mark.parametrize(
    "line",
    [
        "not json",
        '["a", "list"]',
        '{"type": "statute", "citations": []}',
        '{"type": "opinion", "text": "x"}',
        '{"type": "statute", "text": "x", "id": 7}',
        '{"type": "statute", "text": ""}',
        '{"type": "statute", "text": "x", "citations": [{"path": "22"}]}',
        '{"type": "statute", "text": "x", "citations": [{"path": "22", "quote": ""}]}',
    ],
)
def test_anything_else_is_malformed(line):
    with pytest.raises(MalformedClaim):
        parse_claim(line)


def test_a_released_claim_cannot_carry_verified_false():
    with pytest.raises(ValidationError):
        ClaimEvent(id=1, type=ClaimType.STATUTE, text="x", citations=(), verified=False)
    assert ClaimEvent(id=1, type=ClaimType.STATUTE, text="x", citations=()).verified is True


def test_a_line_split_across_deltas_is_yielded_once_complete():
    buffer = LineBuffer()
    assert buffer.feed('{"type": "stat') == []
    assert buffer.feed('ute", "text": "a"}\n{"ty') == ['{"type": "statute", "text": "a"}']
    assert buffer.feed('pe": "computation", "text": "b"}') == []
    assert buffer.flush() == ['{"type": "computation", "text": "b"}']
    assert buffer.flush() == []


def test_the_last_line_without_a_newline_is_flushed():
    assert list(iter_lines([NDJSON])) == NDJSON.split("\n")


def test_blank_lines_and_code_fences_carry_no_claim():
    text = "```json\n\n" + NDJSON + "\n\n```\n"
    assert list(iter_lines([text])) == NDJSON.split("\n")


def test_every_single_character_split_gives_the_same_lines():
    assert list(iter_lines(list(NDJSON))) == NDJSON.split("\n")


@settings(max_examples=60, deadline=None)
@given(st.lists(st.integers(min_value=0, max_value=len(NDJSON)), max_size=12))
def test_any_split_gives_the_same_claim_sequence(cuts):
    points = sorted(set(cuts))
    pieces = [NDJSON[a:b] for a, b in zip([0, *points], [*points, len(NDJSON)], strict=True)]
    assert [parse_claim(line) for line in iter_lines(pieces)] == [
        parse_claim(line) for line in NDJSON.split("\n")
    ]


def test_closing_the_lines_closes_the_source():
    closed = []

    def source():
        try:
            yield '{"type": "statute", "text": "a"}\n'
            yield '{"type": "statute", "text": "b"}\n'
        finally:
            closed.append(True)

    lines = iter_lines(source())
    next(lines)
    lines.close()
    assert closed == [True]
