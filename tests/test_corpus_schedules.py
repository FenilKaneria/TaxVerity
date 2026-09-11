
import pytest

from taxverity.corpus.loader import is_furniture
from taxverity.corpus.models import Span
from taxverity.corpus.schedules import (
    FIRST_SCHEDULE_PAGE,
    MARKER_LEFT_MARGIN_MAX,
    is_caption,
    is_plausible_next,
    marker_key,
    paragraph_marker,
    parse_schedules,
    schedule_preamble,
)
from taxverity.corpus.sections import Line, align_lines


def span(text, *, size=8.49, bold=False, x0=27.6, furniture=False):
    return Span(
        text=text,
        font="LiberationSerif-Bold" if bold else "LiberationSerif",
        size=size,
        bold=bold,
        italic=False,
        bbox=(x0, 0.0, x0 + len(text) * 4, 10.0),
        furniture=furniture,
    )


def line(text, *, size=8.49, bold=False, x0=27.6, furniture=False):
    return Line(page=0, index=0, text=text, spans=(span(text, size=size, bold=bold, x0=x0, furniture=furniture),))


# --- marker_key / is_plausible_next -----------------------------------------


def test_marker_key_splits_number_and_suffix():
    assert marker_key("38") == (38, "")
    assert marker_key("38B") == (38, "B")


def test_marker_key_rejects_a_non_marker():
    with pytest.raises(ValueError):
        marker_key("iii")


def test_only_1_opens_a_fresh_schedule_or_part():
    assert is_plausible_next(None, "1")
    assert not is_plausible_next(None, "2")
    assert not is_plausible_next(None, "1A")


def test_exact_succession_is_always_plausible():
    assert is_plausible_next("1", "2")
    assert is_plausible_next("38D", "39")


def test_one_omitted_paragraph_is_tolerated():
    """Schedule XII Part A genuinely omits 28 -- confirmed by direct search."""
    assert is_plausible_next("27", "29")


def test_a_wild_jump_is_rejected():
    """A wrapped body line ('20 or 21),' on its own line) must never open."""
    assert not is_plausible_next("1", "20")


def test_a_letter_suffix_branches_off_the_same_integer():
    assert is_plausible_next("38", "38B")
    assert is_plausible_next("38B", "38C")


def test_a_letter_suffix_must_increase():
    assert not is_plausible_next("38B", "38A")
    assert not is_plausible_next("38B", "38B")


def test_a_marker_never_goes_backward():
    assert not is_plausible_next("39", "38")
    assert not is_plausible_next("39", "37")


# --- paragraph_marker ---------------------------------------------------


def test_a_bold_prose_marker_is_recognised():
    assert paragraph_marker(line("1. (1) The eligible investment fund", bold=True)) == "1"


def test_a_plain_list_marker_is_recognised_without_boldness():
    assert paragraph_marker(line("1. Aluminium ores.")) == "1"


def test_a_missing_trailing_period_is_tolerated():
    """Schedule II row 33 and Schedule VII row 33 both lack the period."""
    assert paragraph_marker(line("33")) == "33"


def test_a_letter_suffixed_marker_is_recognised():
    assert paragraph_marker(line("38B. Some inserted text")) == "38B"


def test_an_ordinal_is_not_a_marker():
    assert paragraph_marker(line("1st April, 2003 to 31st")) is None


def test_a_wrapped_reference_is_rejected_by_position():
    """Measured at x0=55.2 on Schedule XV -- past the left margin, so it is
    the position check, not the regex, that has to reject it: the lookahead
    alone is satisfied by "20" followed by a space."""
    assert paragraph_marker(line("20 or 21), as may be notified", x0=55.2)) is None


def test_a_footnote_reference_digit_is_rejected_by_size():
    """Measured at 7.07pt across every Schedule footnote reference digit."""
    assert paragraph_marker(line("37", size=7.07)) is None


def test_a_footnote_reference_with_lowercase_suffix_is_rejected():
    assert paragraph_marker(line("40a", size=7.07)) is None


def test_a_nested_sub_table_row_is_rejected_by_position():
    """Schedule II paragraph 2's own 'Sl. No.' column sits at x0=239.6."""
    assert paragraph_marker(line("3.", x0=239.6)) is None


def test_a_marker_at_the_margin_cutoff_is_accepted():
    assert paragraph_marker(line("1.", x0=MARKER_LEFT_MARGIN_MAX)) == "1"


def test_furniture_never_yields_a_marker():
    assert paragraph_marker(line("Income Tax Department", furniture=True)) is None


# --- is_caption / schedule_preamble --------------------------------------


def test_a_bold_uppercase_line_is_a_caption():
    assert is_caption(line("SCHEDULE I", bold=True))
    assert is_caption(line("PART A", bold=True))


def test_a_bold_mixed_case_line_is_not_a_caption():
    assert not is_caption(line("Quantum of deduction.", bold=True))


def test_a_plain_uppercase_line_is_not_a_caption():
    assert not is_caption(line("SCHEDULE I"))


def test_preamble_skips_a_reference_line_and_collects_the_title():
    lines = (
        line("SCHEDULE I", bold=True),
        line("[See section 9(12)]"),
        line("CONDITIONS FOR CERTAIN ACTIVITIES", bold=True),
        line("1. (1) The eligible investment fund", bold=True),
    )
    title, preamble, cursor = schedule_preamble(lines, 0)
    assert title == "CONDITIONS FOR CERTAIN ACTIVITIES"
    assert preamble == ["[See section 9(12)]"]
    assert cursor == 3


def test_preamble_stops_at_a_part_heading():
    """A first version of this parser swallowed 'PART A' into the title."""
    lines = (
        line("SCHEDULE XI", bold=True),
        line("PART A", bold=True),
        line("RECOGNISED PROVIDENT FUNDS", bold=True),
    )
    title, preamble, cursor = schedule_preamble(lines, 0)
    assert title is None
    assert preamble == []
    assert cursor == 1


def test_preamble_with_no_title_at_all():
    lines = (line("SCHEDULE II", bold=True), line("Income not to be included."))
    title, _preamble, cursor = schedule_preamble(lines, 0)
    assert title is None
    assert cursor == 1


# --- integration against the real corpus ---------------------------------


PARAGRAPH_COUNTS = {
    "I": 2, "II": 17, "III": 42, "IV": 17, "V": 8, "VI": 12, "VII": 48,
    "VIII": 2, "IX": 6, "X": 6, "XI": 26, "XII": 50, "XIII": 15, "XIV": 6,
    "XV": 6, "XVI": 2,
}


def test_all_sixteen_schedules_are_found(parsed_schedules):
    assert [s.marker for s in parsed_schedules.schedules] == list(PARAGRAPH_COUNTS)


def test_each_schedule_has_its_measured_paragraph_count(parsed_schedules):
    actual = {s.marker: len(s.children) for s in parsed_schedules.schedules}
    assert actual == PARAGRAPH_COUNTS


def test_schedule_iii_carries_its_genuine_letter_suffixed_insertions(parsed_schedules):
    markers = [p.marker for p in parsed_schedules.schedule("III").children]
    assert markers[-4:] == ["38B", "38C", "38D", "39"]
    assert "38A" not in markers


def test_schedule_xii_part_a_omits_28(parsed_schedules):
    """Confirmed by direct search of the text -- not a parsing artefact."""
    markers = [p.marker for p in parsed_schedules.schedule("XII").children]
    assert "A27" in markers
    assert "A28" not in markers
    assert "A29" in markers


def test_a_part_letter_is_folded_into_the_paragraph_marker(parsed_schedules):
    """Paragraph numbering resets per Part, so the bare number alone is not
    a unique citation within the Schedule -- Schedule XI(1) would otherwise
    exist three times over."""
    schedule_xi = parsed_schedules.schedule("XI")
    assert schedule_xi.find("Schedule XI(A1)") is not None
    assert schedule_xi.find("Schedule XI(B1)") is not None
    assert schedule_xi.find("Schedule XI(C1)") is not None


def test_a_clause_rooted_paragraph_opens_at_clause_not_subclause(parsed_schedules):
    """A regression test for the build() depth_shift bug this step found:
    is_clause_rooted's shift assumes a SUBSECTION level exists to skip, which
    a Schedule paragraph's ladder never had, so the shift must not apply."""
    from taxverity.corpus.nodes import NodeType

    paragraph_2 = parsed_schedules.schedule("I").find("Schedule I(2)")
    assert paragraph_2.children[0].type is NodeType.CLAUSE


def test_node_counts_by_type(parsed_schedules):
    from taxverity.corpus.nodes import NodeType

    nodes = [n for s in parsed_schedules.schedules for n in s.walk()]
    assert len(nodes) == 992
    counts = {t: sum(1 for n in nodes if n.type is t) for t in NodeType}
    assert counts[NodeType.SCHEDULE] == 16
    assert counts[NodeType.SCHEDULE_PARAGRAPH] == 265
    assert counts[NodeType.CLAUSE] == 378
    assert counts[NodeType.SUBCLAUSE] == 226
    assert counts[NodeType.ITEM] == 81
    assert counts[NodeType.SUBITEM] == 26


def test_no_citation_nests_deeper_than_the_schedule_ladder(parsed_schedules):
    from taxverity.corpus.nodes import CITATION_DEPTH_TYPES, NodeType

    nodes = [n for s in parsed_schedules.schedules for n in s.walk()]
    ladder = CITATION_DEPTH_TYPES[NodeType.SCHEDULE]
    assert max(n.path.depth for n in nodes) == len(ladder) + 1


def test_the_residue_is_pinned(parsed_schedules):
    """Any change to what the parser cannot place should be loud, not
    silent -- the same discipline substructure.py's own residue carries."""
    assert len(parsed_schedules.anomalies) == 30
    # Citations, not bare markers: "17" alone was two different paragraphs,
    # in Schedules II and III, which is why this list gained an entry
    # without a single new anomaly.
    # Step 1.10: only a duplicate citation invalidates a paragraph's shape.
    # II(17), III(4), III(17), III(39) and IV(14) recovered their children.
    assert parsed_schedules.unreliable == (
        "Schedule III(19)",
        "Schedule III(30)",
        "Schedule V(8)",
    )


def test_footnote_apparatus_is_separated_from_body_text(parsed_schedules):
    """The same running amendment-footnote counter (37-52) that spans the
    main Act continues into the Schedule pages. Only the first line of each
    entry carries the amendment vocabulary -- the rest is quoted repealed
    text -- so it is enough that some lines do, not all of them."""
    assert len(parsed_schedules.footnotes) == 48
    assert any("Act No." in f.text or "w.e.f." in f.text for f in parsed_schedules.footnotes)


def test_schedule_and_part_headings_are_kept_verbatim(parsed_schedules):
    """16 SCHEDULE headings + 5 PART headings (Schedule XI: A/B/C, XII: A/B)."""
    assert len(parsed_schedules.headings) == 21
    assert "SCHEDULE I" in parsed_schedules.headings
    assert "PART A" in parsed_schedules.headings


def test_no_schedule_text_is_lost(act_pages, parsed_schedules):
    """Independent reconstruction of every non-furniture line from the raw
    pages, checked against where the parser actually put it -- verbatim text
    must never simply vanish, even when it lands in a different bucket than
    a stricter model (a proper title/heading field) would prefer."""
    raw_by_schedule: dict[str, list[str]] = {}
    current = None
    for artifact in act_pages:
        if artifact.page < FIRST_SCHEDULE_PAGE:
            continue
        for text_line in align_lines(artifact):
            if is_furniture(text_line.text):
                continue
            if text_line.text in parsed_schedules.headings and is_caption(text_line):
                if text_line.content.startswith("SCHEDULE "):
                    current = text_line.content.removeprefix("SCHEDULE ")
                raw_by_schedule.setdefault(current, []).append(text_line.text)
                continue
            if current:
                raw_by_schedule.setdefault(current, []).append(text_line.text)

    footnote_lines = {f.text for f in parsed_schedules.footnotes}
    for schedule in parsed_schedules.schedules:
        parsed_lines = {text for text in schedule.full_text().split("\n") if text.strip()}
        for raw_line in raw_by_schedule.get(schedule.marker, []):
            if not raw_line.strip():
                continue
            if raw_line in parsed_lines or raw_line in footnote_lines:
                continue
            if raw_line in parsed_schedules.headings:
                continue
            if schedule.title and raw_line.strip() in schedule.title:
                continue
            pytest.fail(f"Schedule {schedule.marker} lost line: {raw_line!r}")


def test_parsing_is_reasonably_fast(act_pages, schedule_table_regions):
    import time

    started = time.perf_counter()
    parse_schedules(act_pages, table_regions=schedule_table_regions)
    assert time.perf_counter() - started < 15
