"""R22 Part B — in-section tables, rendered one row per line for the model.

A chunk stores a table the way the PDF's text layer reads it: one cell
fragment per line ("Sl.", "No.", "Person", ..., "30th", "November."). A
reader of that text cannot tell which row a due date belongs to, and a live
answer gave a regular individual s.263's 30 November date, which applies
only "where the provisions of section 172 apply". Number grounding passed it,
because "30" and "November" are in the cited passage.

`scripts/build_table_rows.py` measures each table's grid with pdfplumber
offline and keeps a table only if every body cell is found, in order, in the
chunk text. That check proves the flattened text already lists the cells row
by row, so the row form below only makes the row boundaries visible; it never
moves a figure between rows. Tables that fail the check are not in the file
and stay as flat text, as before.

This changes only what the model reads (`generate.render_context`). Chunk
text, chunk ids and the verifier's grounding are untouched.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from functools import cache
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from taxverity.chunking.models import Chunk

TABLES_STAGE_VERSION = 1
TABLE_ROWS_VERSION = 1
TABLE_ROWS = Path(__file__).with_name("tables_v1.json")

# The line the Act prints above every in-section table.
TABLE_LINE = re.compile(r"^TABLE[ \t]*$", re.MULTILINE)
CELL_SEPARATOR = " | "


class TableRows(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    # The section number, or "Schedule <n>", the table sits in.
    owner: str
    page: int
    header: tuple[str, ...]
    # Raw cell text as pdfplumber read it, "" for an empty cell, one tuple per
    # body row and one entry per header column.
    rows: tuple[tuple[str, ...], ...]


class TableRowsFile(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    version: int
    tables: tuple[TableRows, ...]


def owner_of(chunk: Chunk) -> str | None:
    if chunk.section_number is not None:
        return chunk.section_number
    if chunk.schedule_number is not None:
        return f"Schedule {chunk.schedule_number}"
    return None


def cell_pattern(cell: str) -> str:
    """A cell's words with any whitespace between them, filling whole lines:
    the PDF wraps a cell over several lines, and the chunk text keeps those
    breaks. Whole lines, so a short cell ("A", "1.") can only match a line of
    its own, never a letter inside a word."""
    words = r"\s*".join(re.escape(token) for token in cell.split())
    return rf"(?m)^[ \t]*{words}[ \t]*$"


def locate(text: str, cells: Sequence[str]) -> tuple[int, int] | None:
    """The span from the `TABLE` line before the first cell to the end of the
    last cell, when every non-empty cell is found in order; else None."""
    position = 0
    first: int | None = None
    for cell in cells:
        if not cell.strip():
            continue
        found = re.compile(cell_pattern(cell)).search(text, position)
        if found is None:
            return None
        if first is None:
            first = found.start()
        position = found.end()
    if first is None:
        return None
    table_lines = [m for m in TABLE_LINE.finditer(text, 0, first)]
    if not table_lines:
        return None
    return table_lines[-1].start(), position


def render_rows(table: TableRows) -> str:
    lines = ["TABLE"]
    for row in table.rows:
        cells = [
            f"{label}: {' '.join(value.split())}"
            for label, value in zip(table.header, row, strict=True)
            if value.strip()
        ]
        lines.append(CELL_SEPARATOR.join(cells))
    return "\n".join(lines)


def body_cells(table: TableRows) -> list[str]:
    return [cell for row in table.rows for cell in row]


@cache
def load_table_rows(path: Path = TABLE_ROWS) -> tuple[TableRows, ...]:
    loaded = TableRowsFile.model_validate_json(path.read_text(encoding="utf-8"))
    if loaded.version != TABLE_ROWS_VERSION:
        raise ValueError(
            f"{path} is version {loaded.version}, not {TABLE_ROWS_VERSION}"
        )
    return loaded.tables


def with_table_rows(chunk: Chunk, tables: Sequence[TableRows] | None = None) -> str:
    """The chunk's text with each known table in row form. A table whose cells
    no longer match the text (a re-ingested corpus) is left flat."""
    owner = owner_of(chunk)
    text = chunk.text
    if owner is None or TABLE_LINE.search(text) is None:
        return text
    # Every span is found in the original text before any is replaced, so a
    # rendered table can never be matched again as part of another.
    spans: list[tuple[int, int, TableRows]] = []
    for table in load_table_rows() if tables is None else tables:
        if table.owner != owner:
            continue
        span = locate(text, body_cells(table))
        if span is None:
            continue
        if any(span[0] < end and start < span[1] for start, end, _ in spans):
            continue
        spans.append((*span, table))
    for start, end, table in sorted(spans, key=lambda s: s[0], reverse=True):
        text = text[:start] + render_rows(table) + text[end:]
    return text
