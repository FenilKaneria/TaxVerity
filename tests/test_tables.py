"""R22 Part B — in-section tables rendered one row per line for the model.

The bug this guards: a live answer gave a regular individual s.263's 30
November due date, which applies only "where the provisions of section 172
apply". Flat text puts that date on its own line, away from its row."""

from __future__ import annotations

import pytest

from taxverity.chunking.models import Chunk
from taxverity.generation.generate import render_context
from taxverity.retrieval.base import ScoredChunk
from taxverity.retrieval.evidence import EvidencePacker
from taxverity.retrieval.tables import (
    TableRows,
    body_cells,
    load_table_rows,
    locate,
    owner_of,
    with_table_rows,
)

FLAT = """(c) the due date is as in column D of the Table below:
TABLE
Sl.
No.
Person
Due date
A
B
C
1.
A company.
30th
November.
2.
Any other person.
31st July."""

TABLE = TableRows(
    owner="9",
    page=1,
    header=("Sl. No.", "Person", "Due date"),
    rows=(
        ("1.", "A company.", "30th\nNovember."),
        ("2.", "Any other person.", "31st July."),
    ),
)


def _chunk(text: str, section: str = "9") -> Chunk:
    return Chunk.create(
        "v" * 64,
        section,
        text,
        parent_id=None,
        doc_id="income-tax-act-2025",
        node_type="section",
        section_number=section,
        page_start=1,
        page_end=1,
    )


def test_a_matched_table_reads_one_row_per_line():
    rendered = with_table_rows(_chunk(FLAT), [TABLE])
    assert rendered.splitlines()[1:] == [
        "TABLE",
        "Sl. No.: 1. | Person: A company. | Due date: 30th November.",
        "Sl. No.: 2. | Person: Any other person. | Due date: 31st July.",
    ]
    assert rendered.startswith("(c) the due date is as in column D")


def test_cells_out_of_text_order_leave_the_table_flat():
    swapped = TableRows(
        owner="9",
        page=1,
        header=TABLE.header,
        rows=(TABLE.rows[1], TABLE.rows[0]),
    )
    assert locate(FLAT, body_cells(swapped)) is None
    assert with_table_rows(_chunk(FLAT), [swapped]) == FLAT


def test_a_short_cell_never_matches_inside_a_word():
    # "A" is in "TABLE" and "A company", but never a line of its own here.
    assert locate("TABLE\nA company.", ["A"]) is None


def test_a_table_of_another_section_is_not_applied():
    assert with_table_rows(_chunk(FLAT, section="10"), [TABLE]) == FLAT


# --- the real s.263 table, against the built chunk store ------------------------


def _by_path(stored_chunks) -> dict[str, Chunk]:
    _, chunks = stored_chunks
    return {chunk.node_path: chunk for chunk in chunks}


def test_s263_due_dates_stay_with_their_rows(stored_chunks):
    rows = with_table_rows(_by_path(stored_chunks)["263(1)(c)"]).splitlines()
    november = [line for line in rows if "30th November" in line]
    july = [line for line in rows if "31st July" in line]
    assert november == [
        "Sl. No.: 1. | Person: Assessee, including the partners of the firm or the "
        "spouse of such partner (if section 10 applies to such spouse). | "
        "Conditions: Where the provisions of section 172 apply. | "
        "Due date: 30th November."
    ]
    assert july == ["Sl. No.: 4. | Person: Any other assessee. | Due date: 31st July.]"]


def test_the_evidence_block_the_model_reads_shows_s263_as_rows(stored_chunks):
    chunk = _by_path(stored_chunks)["263(1)(c)"]
    pack = EvidencePacker(stored_chunks[1]).pack([ScoredChunk(chunk=chunk, score=1.0)])
    context = render_context("When is my return due?", pack, None, None)
    assert "Person: Any other assessee. | Due date: 31st July.]" in context
    assert "\n30th\nNovember." not in context


@pytest.mark.parametrize("path", ["263", "263(1)", "263(1)(c)"])
def test_every_chunk_carrying_the_s263_table_gets_rows(stored_chunks, path):
    assert "Due date: 31st July." in with_table_rows(_by_path(stored_chunks)[path])


def test_every_recorded_table_still_matches_its_chunk_text(stored_chunks):
    """Staleness guard: a re-built corpus whose text no longer carries a
    recorded table must be caught here, not silently rendered flat."""
    _, chunks = stored_chunks
    largest: dict[str, Chunk] = {}
    for chunk in chunks:
        owner = owner_of(chunk)
        if owner and (
            owner not in largest or len(chunk.text) > len(largest[owner].text)
        ):
            largest[owner] = chunk
    tables = load_table_rows()
    assert tables
    for table in tables:
        assert locate(largest[table.owner].text, body_cells(table)) is not None, (
            table.owner,
            table.page,
        )
