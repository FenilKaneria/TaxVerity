
from taxverity.corpus.tables import TableRegion, find_table_regions, in_any_region

# --- TableRegion membership, pure --------------------------------------------


def test_a_region_contains_its_own_boundaries():
    region = TableRegion(page=246, top=343.0, bottom=478.0)
    assert region.contains(246, 343.0)
    assert region.contains(246, 478.0)
    assert region.contains(246, 400.0)


def test_a_region_rejects_the_wrong_page_or_position():
    region = TableRegion(page=246, top=343.0, bottom=478.0)
    assert not region.contains(247, 400.0)
    assert not region.contains(246, 300.0)
    assert not region.contains(246, 500.0)


def test_a_region_rejects_a_missing_position():
    region = TableRegion(page=246, top=343.0, bottom=478.0)
    assert not region.contains(246, None)


def test_in_any_region_checks_every_region():
    regions = (
        TableRegion(page=1, top=0.0, bottom=10.0),
        TableRegion(page=2, top=0.0, bottom=10.0),
    )
    assert in_any_region(regions, 2, 5.0)
    assert not in_any_region(regions, 3, 5.0)
    assert not in_any_region((), 1, 5.0)


# --- against the real PDF ----------------------------------------------------


def test_a_ruled_two_column_table_is_measured(pdf_path):
    """Section 206's table, page 246 — pdfplumber rules this into a clean grid."""
    regions = find_table_regions(pdf_path, [246])
    assert len(regions) == 1
    region = regions[0]
    assert region.page == 246
    assert 340 < region.top < 350
    assert 470 < region.bottom < 480


def test_a_borderless_table_measures_as_nothing(pdf_path):
    """Section 2's table, page 2 — pdfplumber degenerates to one whole-page
    cell here (no ruling lines), which is not a usable grid and is dropped."""
    regions = find_table_regions(pdf_path, [2])
    assert regions == ()


def test_scanning_is_restricted_to_the_given_pages(pdf_path):
    """Never a blind document scan (Step 1.1: PyMuPDF false-positived on
    ordinary prose) — only pages the caller already has textual evidence for."""
    regions = find_table_regions(pdf_path, [246])
    assert all(region.page == 246 for region in regions)
