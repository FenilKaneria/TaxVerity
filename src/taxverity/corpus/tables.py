from __future__ import annotations

import time
from collections.abc import Iterable
from pathlib import Path

import pdfplumber
from pydantic import BaseModel, ConfigDict

from taxverity.observability import get_logger

logger = get_logger(__name__)

TABLE_STAGE_VERSION = 1

# Measured, not assumed: on a page pdfplumber can genuinely rule into a grid
# (section 206, page 246) it returns real multiple columns. On a borderless
# table (sections 2, 194, 46, 52, 394; Schedule III) it degenerates to one
# whole-page cell — always exactly 2 rows, 1 column. Requiring at least two
# columns is what tells the two apart; PyMuPDF's find_tables() was rejected
# outright for false-positiving on ordinary prose (Step 1.1), so even this
# pdfplumber measurement is only ever run on pages already independently
# known, from our own text, to carry a table — never a blind document scan.
MIN_TABLE_COLUMNS = 2


class TableRegion(BaseModel):
    model_config = ConfigDict(frozen=True)

    page: int
    top: float
    bottom: float

    def contains(self, page: int, top: float | None) -> bool:
        return top is not None and self.page == page and self.top <= top <= self.bottom


def find_table_regions(pdf_path: Path, pages: Iterable[int]) -> tuple[TableRegion, ...]:
    """Genuine table geometry, restricted to pages already known to carry one.

    A page can hold more than one table (e.g. a two-part rate schedule), so
    every grid pdfplumber finds on a candidate page is kept, not just the
    first.
    """
    regions: list[TableRegion] = []
    candidates = list(pages)
    started = time.perf_counter()
    with pdfplumber.open(pdf_path) as pdf:
        for page_number in candidates:
            page = pdf.pages[page_number]
            for table in page.find_tables():
                rows = table.extract()
                columns = max((len(row) for row in rows), default=0)
                if columns < MIN_TABLE_COLUMNS:
                    continue
                regions.append(
                    TableRegion(page=page_number, top=table.bbox[1], bottom=table.bbox[3])
                )
    logger.info(
        "pdfplumber measured %d table regions on %d candidate pages in %.1fs",
        len(regions),
        len(candidates),
        time.perf_counter() - started,
    )
    return tuple(regions)


def in_any_region(regions: Iterable[TableRegion], page: int, top: float | None) -> bool:
    return any(region.contains(page, top) for region in regions)
