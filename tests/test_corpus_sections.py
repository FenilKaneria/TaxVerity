from pathlib import Path

import pytest

from taxverity.corpus.loader import read_pages_jsonl
from taxverity.corpus.models import PageArtifact, Span
from taxverity.corpus.nodes import NodeType
from taxverity.corpus.sections import (
    FIRST_SCHEDULE_PAGE,
    LineAlignmentError,
    align_lines,
    parse,
    section_starts,
    title_above,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "pages"
INTERIM = Path(__file__).resolve().parents[1] / "data" / "interim" / "pages.jsonl"

# The gap that separates the amendment apparatus from the body is real geometry,
# so synthetic pages have to lay their lines out rather than share one bbox.
LINE_HEIGHT = 11.7


def span(text, *, bold=False, size=8.49, font=None, furniture=False, y=0.0):
    if font is None:
        font = "LiberationSerif-Bold" if bold else "LiberationSerif"
    return Span(
        text=text,
        font=font,
        size=size,
        bold=bold,
        italic=False,
        bbox=(0.0, y, 100.0, y + 9.4),
        furniture=furniture,
    )


def page(number, rows):
    """rows: list of lists of spans, one list per line, laid out top to bottom."""
    spans = []
    text_lines = []
    y = 60.0
    for row in rows:
        placed = [s.model_copy(update={"bbox": (0.0, y, 100.0, y + 9.4)}) for s in row]
        spans.extend(placed)
        text_lines.append("".join(s.text for s in placed))
        y += LINE_HEIGHT
    return PageArtifact(page=number, text="\n".join(text_lines) + "\n", spans=tuple(spans))


def load(page_number):
    path = FIXTURE_DIR / f"page_{page_number:04d}.json"
    return PageArtifact.model_validate_json(path.read_text(encoding="utf-8"))


def markers_on(artifact):
    return sorted(section_starts(align_lines(artifact)).values())


# --- line alignment ---------------------------------------------------------


def test_align_lines_maps_every_span_onto_its_line():
    artifact = page(1, [[span("Title.", bold=True)], [span("4."), span(" (1) Text")]])
    lines = align_lines(artifact)
    assert [line.content for line in lines] == ["Title.", "4. (1) Text"]
    assert [len(line.spans) for line in lines] == [1, 2]


def test_align_lines_tolerates_the_separators_get_text_inserts():
    artifact = PageArtifact(
        page=1,
        text="(1)\xa0an amount;\n",
        spans=(span("(1)"), span("an amount;")),
    )
    assert align_lines(artifact)[0].spans == artifact.spans


def test_align_lines_raises_when_spans_and_text_disagree():
    artifact = PageArtifact(page=1, text="hello world\n", spans=(span("goodbye"),))
    with pytest.raises(LineAlignmentError):
        align_lines(artifact)


def test_a_line_is_bold_only_when_all_of_its_spans_are():
    artifact = page(1, [[span("Bold.", bold=True), span(" roman")]])
    assert align_lines(artifact)[0].is_bold is False


# --- the two cues must agree ------------------------------------------------


def test_both_cues_together_find_a_section():
    artifact = page(1, [[span("Charge of Income-tax.", bold=True)],
                        [span("4.", bold=True), span(" (1) Where any Central Act")]])
    assert markers_on(artifact) == ["4"]


def test_the_line_regex_alone_does_not_find_a_section():
    """Amendment footnotes run their own 1, 2, 3 sequence in ordinary roman."""
    artifact = page(1, [[span("2. Omtt. by the Act No. 4 of 2026, w.e.f. 1-4-2026.")]])
    assert markers_on(artifact) == []


def test_a_bold_marker_that_does_not_open_the_line_is_not_a_section():
    artifact = page(1, [[span("as set out in "), span("14.", bold=True)]])
    assert markers_on(artifact) == []


def test_a_small_bold_digit_is_a_footnote_reference_not_a_section():
    """Page 548 carries a 7.07pt bold '14' beside a footnote line reading '14. Sub...'."""
    artifact = page(1, [[span("14", bold=True, size=7.07), span(". Sub. for x")]])
    assert markers_on(artifact) == []


def test_a_bold_marker_without_its_period_still_counts():
    """On many pages the bold run stops at the digits: '67' then '. (1) Any profits'."""
    artifact = page(1, [[span("67", bold=True), span(". (1) Any profits or gains")]])
    assert markers_on(artifact) == ["67"]


def test_a_space_between_number_and_period_still_counts():
    artifact = page(1, [[span("94 ", bold=True), span(". (1) Irrespective of anything")]])
    assert markers_on(artifact) == ["94"]


def test_no_space_after_the_period_still_counts():
    artifact = page(1, [[span("428.", bold=True), span("Without prejudice to")]])
    assert markers_on(artifact) == ["428"]


def test_a_soft_hyphen_in_the_marker_does_not_defeat_the_cue():
    artifact = page(1, [[span("473.\xad", bold=True), span(" Whoever fails")]])
    assert markers_on(artifact) == ["473"]


def test_a_letter_suffixed_marker_is_kept_whole():
    artifact = page(1, [[span("354A.", bold=True), span(" Where any registered")]])
    assert markers_on(artifact) == ["354A"]


# --- titles -----------------------------------------------------------------


def test_the_title_is_the_bold_line_above_the_number():
    artifact = page(1, [[span("Charge of Income-tax.", bold=True)],
                        [span("4.", bold=True), span(" (1) Where")]])
    lines = align_lines(artifact)
    assert title_above(lines, 1) == ("Charge of Income-tax.", 0)


def test_a_title_may_run_over_two_lines():
    artifact = page(1, [[span("Penalty for failure to comply with", bold=True)],
                        [span("[sections 262 and 397].", bold=True)],
                        [span("467.", bold=True), span(" (1) If a person")]])
    title, first = title_above(align_lines(artifact), 2)
    assert title == "Penalty for failure to comply with [sections 262 and 397]."
    assert first == 0


def test_the_chapter_banner_does_not_bleed_into_the_title():
    artifact = page(1, [[span("CHAPTER II", bold=True)],
                        [span("BASIS OF CHARGE", bold=True)],
                        [span("Charge of Income-tax.", bold=True)],
                        [span("4.", bold=True), span(" (1) Where")]])
    assert title_above(align_lines(artifact), 3) == ("Charge of Income-tax.", 2)


def test_roman_body_above_a_section_is_not_taken_as_its_title():
    artifact = page(1, [[span("and ending with the said financial year.")],
                        [span("Definitions.", bold=True)],
                        [span("2.", bold=True), span(" In this Act,")]])
    assert title_above(align_lines(artifact), 2) == ("Definitions.", 1)


# --- golden fixture pages: the hazards 1.1 and 1.3 identified ---------------


def test_page_12_yields_its_chapter_and_section():
    act = parse([load(12)])
    assert act.markers == ("4",)
    assert act.section("4").title == "Charge of Income-tax."
    assert [(c.numeral, c.title) for c in act.chapters] == [("II", "BASIS OF CHARGE")]
    assert act.section("4").chapter == "II"


def test_page_91_finds_section_67_whose_bold_span_omits_the_period():
    act = parse([load(91)])
    assert act.markers == ("67",)
    assert act.section("67").title == "Capital gains."


def test_page_91_keeps_the_division_heading_out_of_the_section_text():
    act = parse([load(91)])
    assert [line.content for line in act.divisions] == ["E.—Capital gains"]
    assert "E.—Capital gains" not in act.section("67").text


def test_page_125_finds_section_94_written_with_a_spaced_period():
    act = parse([load(125)])
    assert act.markers == ("94",)
    assert act.section("94").title == "Amounts not deductible."


def test_page_416_finds_the_only_letter_suffixed_section():
    act = parse([load(416)])
    assert act.markers == ("354A",)
    assert act.section("354A").path.render() == "354A"


def test_page_507_separates_the_amendment_apparatus_from_the_section():
    act = parse([load(507)])
    assert act.markers == ("427",)
    text = act.section("427").text
    assert "Sub. by Act No. 4 of 2026" not in text
    assert "w.e.f." not in text
    assert any("Sub. by Act No. 4 of 2026" in f.text for f in act.footnotes)


def test_page_507_does_not_read_the_repealed_section_quoted_in_a_footnote():
    """The footnote quotes the prior 428 at the start of a line; only 427 is live here."""
    act = parse([load(507)])
    assert act.section("428") is None


def test_page_528_keeps_an_omitted_section_as_a_node():
    act = parse([load(528)])
    assert act.markers == ("447",)
    assert "[***]" in act.section("447").text
    assert act.section("447").title == (
        "Penalty for failure to furnish report under section 172."
    )


def test_page_548_ignores_the_bold_footnote_digit():
    act = parse([load(548)])
    assert act.markers == ("467",)
    assert act.section("467").title == (
        "Penalty for failure to comply with the provisions of "
        "[sections 262 and 397]."
    )


@pytest.mark.parametrize("page_number", [12, 91, 125, 416, 507, 528, 548])
def test_every_fixture_page_attributes_all_of_its_lines(page_number):
    act = parse([load(page_number)])
    assert act.attributed_lines() == act.body_lines


# --- against the whole corpus ----------------------------------------------


@pytest.fixture(scope="session")
def act():
    if not INTERIM.exists():
        pytest.skip("run scripts/extract_corpus.py to build data/interim/pages.jsonl")
    return parse(read_pages_jsonl(INTERIM))


def test_every_section_of_the_act_is_found(act):
    assert len(act.sections) == 537
    assert len(set(act.markers)) == 537


def test_section_numbering_is_contiguous(act):
    assert act.gaps() == []


def test_sections_appear_in_strictly_ascending_order(act):
    def key(marker):
        digits = "".join(c for c in marker if c.isdigit())
        return int(digits), marker[len(digits) :]

    ordered = [key(marker) for marker in act.markers]
    assert ordered == sorted(ordered)
    assert len(set(ordered)) == len(ordered)


def test_354a_is_the_only_letter_suffixed_section(act):
    assert [m for m in act.markers if not m.isdigit()] == ["354A"]


def test_every_section_has_a_title_and_a_chapter(act):
    assert [n.marker for n in act.sections if not n.title] == []
    assert [n.marker for n in act.sections if not n.chapter] == []


def test_every_section_has_text_and_page_provenance(act):
    assert [n.marker for n in act.sections if not n.text.strip()] == []
    assert [n.marker for n in act.sections if not n.pages] == []


def test_no_orphan_text(act):
    assert act.attributed_lines() == act.body_lines


def test_every_chapter_is_found_and_titled(act):
    assert len(act.chapters) == 23
    assert [c.numeral for c in act.chapters if not c.title] == []
    assert act.chapters[0].numeral == "I"
    assert act.chapters[-1].numeral == "XXIII"


def test_repealed_wording_stays_out_of_section_text(act):
    """The apparatus quotes prior wording verbatim; ADR-016 must never cite it."""
    leaking = [n.marker for n in act.sections if "w.e.f." in n.text]
    assert leaking == ["393"], "page 464's inline prospective note is the known residue"


def test_schedules_are_excluded_from_section_parsing(act):
    assert max(page for node in act.sections for page in node.pages) < FIRST_SCHEDULE_PAGE


def test_sections_are_section_nodes_carrying_a_citable_path(act):
    assert {node.type for node in act.sections} == {NodeType.SECTION}
    assert act.section("2").citation == "2"
    assert act.section("354A").citation == "354A"


def test_section_text_is_verbatim(act):
    """ADR-016 checks quote fidelity against source text, so nothing may be rewritten."""
    checked = {"4", "94", "427", "428"}
    wanted = {p for m in checked for p in act.section(m).pages}
    source = {p.page: p.text for p in read_pages_jsonl(INTERIM) if p.page in wanted}
    for marker in checked:
        node = act.section(marker)
        page_text = "\n".join(source[p] for p in node.pages)
        for line in node.text.split("\n"):
            assert line in page_text


# --- the footnote block's guards -------------------------------------------


def footnote_page(rows, gap_before_last_block):
    """A page whose apparatus is pushed below a rule, as the printer sets it."""
    artifact = page(1, rows)
    lines = list(align_lines(artifact))
    shift = gap_before_last_block - LINE_HEIGHT
    spans = []
    for line in lines:
        offset = shift if line.index >= len(rows) - 2 else 0.0
        spans.extend(
            s.model_copy(
                update={
                    "bbox": (
                        s.bbox[0],
                        s.bbox[1] + offset,
                        s.bbox[2],
                        s.bbox[3] + offset,
                    )
                }
            )
            for s in line.spans
        )
    return PageArtifact(page=1, text=artifact.text, spans=tuple(spans))


NOTE = "3. Sub. by Act No. 4 of 2026, w.e.f. 1-4-2026. Prior to its substitution,"


def test_a_numbered_table_row_is_not_mistaken_for_the_apparatus():
    """Pages 32 and 455-457 set rate tables as '13. Payment received by...' rows."""
    artifact = footnote_page(
        [
            [span("Exemptions.", bold=True)],
            [span("11.", bold=True), span(" The following are exempt")],
            [span("13. Payment received by an employee of the Central Government")],
            [span(NOTE)],
            [span("'(f) any payment by a company on purchase of its own shares,'")],
        ],
        gap_before_last_block=63.0,
    )
    act = parse([artifact])
    assert "13. Payment received by an employee" in act.section("11").text
    assert [f.text for f in act.footnotes] == [
        NOTE,
        "'(f) any payment by a company on purchase of its own shares,'",
    ]


def test_a_division_heading_is_not_mistaken_for_the_apparatus():
    """Page 365 opens with '3. —Representative assessees—Special cases'."""
    artifact = footnote_page(
        [
            [span("2. —Representative assessees—General provisions")],
            [span("Representative assessee.", bold=True)],
            [span("303.", bold=True), span(" (1) For the purposes of this Act")],
            [span(NOTE)],
            [span("'(v) any advance or loan between two group entities,'")],
        ],
        gap_before_last_block=63.0,
    )
    act = parse([artifact])
    assert [line.content for line in act.divisions] == [
        "2. —Representative assessees—General provisions"
    ]
    assert len(act.footnotes) == 2


def test_the_apparatus_carries_its_quoted_repeal_onto_the_next_page():
    """A repeal quoted at a page foot spills above the next page's own first note."""
    first = footnote_page(
        [
            [span("Definitions.", bold=True)],
            [span("2.", bold=True), span(" In this Act, unless the context")],
            [span("(1) 'accountant' shall have the meaning assigned")],
            [span("1. Sub. by the Act No. 4 of 2026, w.e.f. 1-4-2026. Prior to its")],
            [span("'(32) \"co-operative society\" means a co-operative society")],
        ],
        gap_before_last_block=63.0,
    )
    spill = page(
        2,
        [
            [span("registered under the Co-operative Societies Act, 1912;'")],
            [span("2. Omtt. by the Act No. 4 of 2026, w.e.f. 1-4-2026.")],
        ],
    )
    act = parse([first, spill])
    assert act.markers == ("2",)
    assert "co-operative society" not in act.section("2").text
    assert len(act.footnotes) == 4
