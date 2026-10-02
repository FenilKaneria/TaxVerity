"""R22 Part B — measure in-section tables' grids and keep the ones the chunk
text proves.

For every section or Schedule whose chunk text carries a `TABLE` line,
pdfplumber reads the grids on that chunk's pages. A grid is kept only if:
- it has at least two columns with content;
- every header label is found between the `TABLE` line and the first body
  cell (so the first row really is the header, not a continuation row);
- every body cell is found, in order, in the chunk text
  (`taxverity.retrieval.tables.locate`).

Anything else is reported and left out; those tables stay flat text.
Writes `src/taxverity/retrieval/tables_v1.json`.
"""

from __future__ import annotations

import re
import sys

import pdfplumber

from taxverity.chunking.pipeline import read_corpus_version
from taxverity.chunking.store import load_chunks
from taxverity.config import Settings
from taxverity.retrieval.tables import (
    TABLE_LINE,
    TABLE_ROWS,
    TABLE_ROWS_VERSION,
    TableRows,
    TableRowsFile,
    cell_pattern,
    locate,
    owner_of,
)

# "A", "B", "(1)", "(2)": a column's letter or number under its label.
_COLUMN_KEY = re.compile(r"^\(?[A-Z0-9]{1,2}\)?\.?$")


def _label(cell: str | None) -> str:
    lines = [line for line in (cell or "").splitlines() if line.strip()]
    return " ".join(
        line.strip() for line in lines if not _COLUMN_KEY.match(line.strip())
    )


def _grid(
    raw: list[list[str | None]],
) -> tuple[tuple[str, ...], list[tuple[str, ...]]] | None:
    width = max(len(row) for row in raw)
    rows = [[(cell or "") for cell in row] + [""] * (width - len(row)) for row in raw]
    keep = [c for c in range(width) if any(row[c].strip() for row in rows)]
    if len(keep) < 2:
        return None
    header = tuple(_label(rows[0][c]) for c in keep)
    if not all(header):
        return None
    body = [tuple(row[c] for c in keep) for row in rows[1:]]
    # Drop empty rows and the "A | B | C" row of column keys under the labels.
    body = [
        row
        for row in body
        if any(cell.strip() for cell in row)
        and not all(_COLUMN_KEY.match(cell.strip()) for cell in row if cell.strip())
    ]
    return (header, body) if body else None


def _header_found(text: str, header: tuple[str, ...], first_cell: str) -> bool:
    first = re.compile(cell_pattern(first_cell)).search(text)
    if first is None:
        return False
    lines = list(TABLE_LINE.finditer(text, 0, first.start()))
    if not lines:
        return False
    between = text[lines[-1].end() : first.start()]
    return all(re.search(cell_pattern(label), between) for label in header)


def main() -> int:
    settings = Settings()
    interim = settings.interim_dir
    corpus_version = read_corpus_version(interim / "corpus_manifest.json")
    chunks, _ = load_chunks(interim, corpus_version=corpus_version)

    # The largest chunk of each owner holds every table the owner has.
    largest: dict[str, object] = {}
    for chunk in chunks:
        owner = owner_of(chunk)
        if owner is None or TABLE_LINE.search(chunk.text) is None:
            continue
        if owner not in largest or len(chunk.text) > len(largest[owner].text):  # type: ignore[attr-defined]
            largest[owner] = chunk

    kept: list[TableRows] = []
    rejected: list[str] = []
    with pdfplumber.open(settings.resolve_corpus_pdf()) as pdf:
        for owner, chunk in sorted(largest.items(), key=lambda kv: kv[1].page_start):  # type: ignore[attr-defined]
            for page in range(chunk.page_start, chunk.page_end + 1):  # type: ignore[attr-defined]
                for found in pdf.pages[page].find_tables():
                    grid = _grid(found.extract())
                    if grid is None:
                        continue
                    header, body = grid
                    table = TableRows(
                        owner=owner, page=page, header=header, rows=tuple(body)
                    )
                    cells = [cell for row in body for cell in row if cell.strip()]
                    if not _header_found(chunk.text, header, cells[0]):  # type: ignore[attr-defined]
                        rejected.append(
                            f"{owner} p{page}: header not before the first row"
                        )
                    elif locate(chunk.text, cells) is None:  # type: ignore[attr-defined]
                        rejected.append(f"{owner} p{page}: cells not in text order")
                    else:
                        kept.append(table)

    TABLE_ROWS.write_text(
        TableRowsFile(version=TABLE_ROWS_VERSION, tables=tuple(kept)).model_dump_json(
            indent=1
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"kept {len(kept)} table(s), rejected {len(rejected)}")
    for line in rejected:
        print(f"  rejected {line}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
