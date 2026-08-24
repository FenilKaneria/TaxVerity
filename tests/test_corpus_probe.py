from pathlib import Path

import pytest

from taxverity.corpus.models import PageArtifact, Span
from taxverity.corpus.probe import probe, probe_page, title_before

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "pages"


def span(text, *, bold=False, font="LiberationSerif", furniture=False):
    return Span(
        text=text,
        font=font,
        size=8.49,
        bold=bold,
        italic=False,
        bbox=(0.0, 0.0, 1.0, 1.0),
        furniture=furniture,
    )


def page(number, text, spans):
    return PageArtifact(page=number, text=text, spans=tuple(spans))


def load_fixture(page_number):
    path = FIXTURE_DIR / f"page_{page_number:04d}.json"
    return PageArtifact.model_validate_json(path.read_text(encoding="utf-8"))


# --- heading detection ------------------------------------------------------


def test_bold_number_span_is_a_heading_with_its_preceding_title():
    artifact = page(
        10,
        "Method of accounting.\n277. (1) For the purposes",
        [
            span("Method of accounting.", bold=True),
            span("277.", bold=True),
            span("(1) For the purposes"),
        ],
    )
    hits, _, _, _ = probe_page(artifact)
    assert len(hits) == 1
    assert hits[0].number == 277
    assert hits[0].title == "Method of accounting."
    assert hits[0].page == 10


def test_a_soft_hyphen_after_the_number_still_matches():
    artifact = page(
        554,
        "473. Whoever",
        [span("Title here.", bold=True), span("473.\xad", bold=True)],
    )
    hits, _, _, _ = probe_page(artifact)
    assert [hit.number for hit in hits] == [473]


def test_non_bold_numbers_are_not_bold_cue_headings():
    artifact = page(1, "5. something", [span("5."), span("something")])
    hits, _, _, regex_numbers = probe_page(artifact)
    assert hits == []
    assert regex_numbers == {5}


def test_furniture_spans_cannot_be_headings():
    artifact = page(1, "", [span("12.", bold=True, furniture=True)])
    hits, _, _, _ = probe_page(artifact)
    assert hits == []


def test_notosans_bold_is_not_treated_as_a_body_heading():
    artifact = page(1, "", [span("12.", bold=True, font="NotoSans-Bold")])
    hits, _, _, _ = probe_page(artifact)
    assert hits == []


def test_chapter_headings_are_collected_not_treated_as_sections():
    artifact = page(0, "CHAPTER XXII", [span("CHAPTER XXII", bold=True)])
    hits, chapters, _, _ = probe_page(artifact)
    assert hits == []
    assert chapters == ["XXII"]


def test_schedule_headings_are_collected():
    artifact = page(622, "SCHEDULE X\n[See section 49]", [])
    _, _, schedules, _ = probe_page(artifact)
    assert schedules == ["X"]


# --- title association ------------------------------------------------------


def test_title_stops_at_a_chapter_heading():
    spans = [span("CHAPTER XXII", bold=True), span("473.", bold=True)]
    assert title_before(spans, 1) is None


def test_title_strips_surrounding_brackets():
    spans = [
        span("[", bold=True),
        span("Contravention of order.", bold=True),
        span("473.", bold=True),
    ]
    assert title_before(spans, 2) == "Contravention of order."


def test_title_is_none_when_body_text_precedes_the_number():
    spans = [span("ordinary body text"), span("12.", bold=True)]
    assert title_before(spans, 1) is None


# --- aggregation ------------------------------------------------------------


def test_cues_are_recorded_and_combined():
    artifact = page(
        1,
        "Title.\n7. (1) text\n9. other text",
        [span("Title.", bold=True), span("7.", bold=True), span("(1) text")],
    )
    result = probe([artifact])
    by_number = {c.number: c for c in result.candidates}
    assert by_number[7].cues == ("bold", "regex")
    assert by_number[9].cues == ("regex",)
    assert by_number[9].title is None


def test_gaps_and_duplicates_are_reported():
    pages = [
        page(0, "1. a", [span("Title one.", bold=True), span("1.", bold=True)]),
        page(9, "1. a", [span("Sched item.", bold=True), span("1.", bold=True)]),
        page(1, "3. c", [span("Title three.", bold=True), span("3.", bold=True)]),
    ]
    result = probe(pages)
    assert result.gaps(3) == [2]
    assert result.duplicates() == {1: [0, 9]}


def test_probe_of_no_pages_is_empty():
    result = probe([])
    assert result.candidates == ()
    assert result.gaps(3) == [1, 2, 3]


# --- against the committed golden fixtures ----------------------------------


def test_golden_page_333_yields_section_277_with_its_title():
    result = probe([load_fixture(333)])
    assert [c.number for c in result.candidates if "bold" in c.cues] == [277]
    assert result.candidates[0].title == "Method of accounting in certain cases."


def test_golden_page_554_yields_section_473_and_its_chapter():
    result = probe([load_fixture(554)])
    numbers = [c.number for c in result.candidates if "bold" in c.cues]
    assert 473 in numbers
    assert ("XXII", 554) in result.chapters


@pytest.mark.parametrize("page_number", (0, 273, 333, 554, 665))
def test_probe_runs_over_every_golden_fixture(page_number):
    result = probe([load_fixture(page_number)])
    assert all(c.page == page_number for c in result.candidates)
