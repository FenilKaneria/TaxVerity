
import pytest

from taxverity.corpus.nodes import NodePath, NodeType, StatutoryNode
from taxverity.corpus.substructure import (
    LEVEL_TYPES,
    AnomalyReason,
    MarkerKind,
    PageProvenanceError,
    build,
    is_clause_rooted,
    kinds_for,
    opens,
    round_trip_failures,
    successor,
)


def section(marker, *lines, title="A section."):
    return StatutoryNode(
        type=NodeType.SECTION,
        marker=marker,
        path=NodePath.section(marker),
        title=title,
        text="\n".join(lines),
        pages=(0,),
    )


def tree(node, *lines, **kwargs):
    grown, anomalies = build(section(node, *lines, **kwargs))
    return grown, anomalies


def citations(grown):
    return [n.path.render() for n in grown.walk() if n.path.depth > 1]


# --- marker alphabets -------------------------------------------------------


def test_a_marker_carries_every_alphabet_it_could_belong_to():
    assert kinds_for("i") == (MarkerKind.ROMAN_LOWER, MarkerKind.ALPHA_LOWER)
    assert kinds_for("b") == (MarkerKind.ALPHA_LOWER,)
    assert kinds_for("ii") == (MarkerKind.ROMAN_LOWER,)
    assert kinds_for("3") == (MarkerKind.NUMERIC,)


def test_a_marker_in_no_alphabet_has_no_kind():
    assert kinds_for("and") == ()
    assert kinds_for("qq") == ()


def test_the_act_continues_z_with_za_not_aa():
    assert successor(MarkerKind.ALPHA_LOWER, "z") == "za"
    assert successor(MarkerKind.ALPHA_LOWER, "za") == "zb"


def test_roman_sequences_run_past_three_letters():
    assert successor(MarkerKind.ROMAN_LOWER, "vii") == "viii"
    assert successor(MarkerKind.ROMAN_UPPER, "XVII") == "XVIII"


def test_only_the_first_marker_of_an_alphabet_opens_a_level():
    assert opens(MarkerKind.ROMAN_LOWER, "i")
    assert opens(MarkerKind.NUMERIC, "1")
    assert not opens(MarkerKind.ALPHA_LOWER, "b")


# --- placement by sibling sequence ------------------------------------------


def test_a_section_opening_line_carries_its_first_sub_section():
    grown, _ = tree("9", "9. (1) The income referred to.", "(2) The income accruing.")
    assert citations(grown) == ["9(1)", "9(2)"]


def test_a_marker_run_on_one_line_opens_every_level_it_names():
    grown, _ = tree("9", "9. (1)(a) Income by way of interest.", "(i) the Government;")
    assert citations(grown) == ["9(1)", "9(1)(a)", "9(1)(a)(i)"]


def test_a_deeper_list_closes_when_its_parent_resumes():
    grown, _ = tree(
        "9",
        "9. (1) First.",
        "(a) alpha;",
        "(i) roman;",
        "(ii) roman;",
        "(b) alpha;",
        "(2) Second.",
    )
    assert citations(grown) == [
        "9(1)", "9(1)(a)", "9(1)(a)(i)", "9(1)(a)(ii)", "9(1)(b)", "9(2)",
    ]


def test_i_after_h_continues_the_letters_when_j_follows():
    """Page 27 runs a clause sequence a..h, i, j — (i) is a letter there."""
    run = [f"({letter}) {letter};" for letter in "abcdefghij"]
    grown, _ = tree("27", "27. (1) x", *run)
    assert citations(grown) == ["27(1)"] + [f"27(1)({x})" for x in "abcdefghij"]


def test_i_after_h_opens_a_roman_list_when_ii_follows():
    """Section 19(2)(h) opens i, ii, iii directly under (h)."""
    run = [f"({letter}) {letter};" for letter in "abcdefgh"]
    grown, _ = tree("19", "19. (1) x", *run, "(i) i;", "(ii) ii;")
    assert citations(grown)[-3:] == ["19(1)(h)", "19(1)(h)(i)", "19(1)(h)(ii)"]


def test_a_repeated_alphabet_nests_when_the_line_above_announces_a_list():
    """Section 416(5)(g) really does contain (a) and (b)."""
    outer = [f"({letter}) {letter};" for letter in "abcdef"]
    grown, _ = tree(
        "416", "416. (1) x", *outer, "(g) on oath that—", "(a) a;", "(b) b,", "(h) h;"
    )
    assert citations(grown)[-4:] == [
        "416(1)(g)", "416(1)(g)(a)", "416(1)(g)(b)", "416(1)(h)",
    ]


def test_a_repeated_alphabet_restarts_the_level_when_nothing_announces_it():
    grown, _ = tree("1", "1. (1) x", "(a) a;", "(b) b.", "(a) a again.")
    assert citations(grown) == ["1(1)", "1(1)(a)", "1(1)(b)", "1(1)(a)"]


def test_a_marker_may_not_open_a_level_deeper_than_a_citation_can_name():
    lines = ["1. (1) x"] + [f"({m}) deeper—" for m in ("a", "i", "A", "I")]
    grown, anomalies = tree("1", *lines, "(a) too deep—")
    assert max(n.path.depth for n in grown.walk()) == len(LEVEL_TYPES) + 1
    assert [a.reason for a in anomalies] == [AnomalyReason.TOO_DEEP]


# --- markers that are not nodes ---------------------------------------------


def test_a_wrapped_cross_reference_does_not_steal_the_next_marker():
    """Section 44(1) breaks a line on "sub-section" and the next reads "(2)—"."""
    grown, _ = tree(
        "44",
        "44. (1) incurs any expenditure specified in sub-section",
        "(2)—",
        "(a) before the commencement;",
        "(2) The expenditure referred to in sub-section (1) shall be—",
    )
    assert citations(grown) == ["44(1)", "44(1)(a)", "44(2)"]


def test_the_veto_only_applies_to_the_reference_vocabulary():
    grown, _ = tree("1", "1. (1) a list of things—", "(2) the second.")
    assert citations(grown) == ["1(1)", "1(2)"]


def test_a_marker_fenced_by_the_amendment_brackets_still_opens_a_level():
    """A missed "[" cascades: section 66 lost 36 nodes to one of them."""
    grown, _ = tree("66", "66. In this Part,—", '[(1) "a" means x;]', '(2) "b" means y;')
    assert citations(grown) == ["66(1)", "66(2)"]


def test_a_parenthesised_word_is_not_a_marker():
    grown, _ = tree("1", "1. (1) x", "(and) not a marker;")
    assert citations(grown) == ["1(1)"]


# --- tables -----------------------------------------------------------------


def test_markers_inside_a_table_are_not_nodes():
    grown, _ = tree(
        "19",
        "19. (1) deductions in column B of the following Table:—",
        "TABLE",
        "1.",
        "Standard deduction.",
        "(a) Rs. 75000;",
        "(b) Rs. 50000.",
        "(2) For the purposes of the Table,—",
        "(a) in respect of the entries;",
    )
    assert citations(grown) == ["19(1)", "19(2)", "19(2)(a)"]


def test_a_table_ends_when_its_enclosing_sequence_resumes():
    grown, _ = tree("19", "19. (1) x", "TABLE", "(a) cell;", "(2) next.")
    assert grown.find("19(1)").text.endswith("(a) cell;")


# --- measured table geometry (Step 1.7) --------------------------------------


def test_a_measured_line_never_opens_a_marker_however_it_reads():
    """Section 206's two-column table: "(2)" is a real row label on both
    columns, which the "resumes an open sequence" guess alone cannot tell
    from the enclosing sequence's genuine next member."""
    node = section(
        "206",
        "206. (1) x",
        "(a) the Table below shall be added:",
        "TABLE",
        "(1) row one, col A;",
        "(1) row one, col B;",
        "(2) row two, col A;",
        "(2) row two, col B;",
        "(b) resumes the real sequence.",
    )
    grown, _ = build(
        node, in_table=(False, False, False, True, True, True, True, False)
    )
    assert citations(grown) == ["206(1)", "206(1)(a)", "206(1)(b)"]


def test_measured_geometry_only_adds_suppression_never_forces_an_early_close():
    """pdfplumber sometimes measures only part of a table — section 39's ruled
    region covers 6 of its rows and the rest spill onto the next page,
    unmeasured. An unmeasured line right after a measured stretch must not be
    assumed to be past the table; the old resumes-check still decides, exactly
    as it did before any geometry was available."""
    node = section(
        "9",
        "9. (1) x",
        "TABLE",
        "(1) measured row;",
        "(a) unmeasured cell, not a node;",
        "(2) resumes.",
    )
    grown, _ = build(node, in_table=(False, False, True, False, False))
    assert citations(grown) == ["9(1)", "9(2)"]


def test_a_mismatched_table_flag_length_is_refused():
    with pytest.raises(PageProvenanceError):
        build(section("9", "9. (1) First.", "(2) Second."), in_table=(False,))


def test_build_reuses_the_same_ladder_for_a_shallower_root():
    """Schedule paragraphs (Step 1.7's next piece) nest one rung shallower
    than a section — no SUBSECTION level, clauses directly below the root."""
    schedule_ladder = (NodeType.CLAUSE, NodeType.SUBCLAUSE)
    node = section("1", "1. (1) x—", "(a) y;")
    grown, _ = build(node, level_types=schedule_ladder)
    types = {n.path.render(): n.type for n in grown.walk() if n.path.depth > 1}
    assert types["1(1)"] is NodeType.CLAUSE
    assert types["1(1)(a)"] is NodeType.SUBCLAUSE


# --- sub-section or clause --------------------------------------------------


def test_a_section_that_carries_its_marker_inline_is_sub_section_rooted():
    assert not is_clause_rooted("9. (1) The income referred to.")


def test_a_section_that_opens_with_lead_in_prose_is_clause_rooted():
    assert is_clause_rooted("2. In this Act, unless the context otherwise requires,—")


def test_a_bare_opening_line_is_sub_section_rooted():
    """Section 169 prints "169." alone; the marker merely wrapped."""
    assert not is_clause_rooted("169.\n(1) Where a modified return is furnished.")


def test_a_clause_rooted_section_shifts_every_level_down_one_rung():
    grown, _ = tree("2", "2. In this Act,—", "(1) x—", "(a) y—", "(i) z;")
    types = {n.path.render(): n.type for n in grown.walk() if n.path.depth > 1}
    assert types["2(1)"] is NodeType.CLAUSE
    assert types["2(1)(a)"] is NodeType.SUBCLAUSE
    assert types["2(1)(a)(i)"] is NodeType.ITEM


def test_a_sub_section_rooted_section_starts_at_sub_section():
    grown, _ = tree("9", "9. (1) x—", "(a) y—", "(i) z;")
    types = {n.path.render(): n.type for n in grown.walk() if n.path.depth > 1}
    assert types["9(1)"] is NodeType.SUBSECTION
    assert types["9(1)(a)"] is NodeType.CLAUSE
    assert types["9(1)(a)(i)"] is NodeType.SUBCLAUSE


# --- round trip and provenance ----------------------------------------------


def test_the_tree_reproduces_the_section_text_exactly():
    lines = ["9. (1) First.", "(a) alpha;", "wrapped continuation", "(2) Second."]
    grown, _ = tree("9", *lines)
    assert grown.full_text() == "\n".join(lines)


def test_a_line_with_no_marker_belongs_to_the_deepest_open_node():
    grown, _ = tree("9", "9. (1) First.", "(a) alpha;", "wrapped continuation")
    assert grown.find("9(1)(a)").text == "(a) alpha;\nwrapped continuation"


def test_page_provenance_follows_each_line():
    node = section("9", "9. (1) First.", "(a) alpha;", "(2) Second.")
    grown, _ = build(node, pages=(17, 17, 18))
    assert grown.find("9(1)").pages == (17,)
    assert grown.find("9(2)").pages == (18,)


def test_a_marker_only_node_inherits_provenance_from_its_children():
    node = section("9", "9. (1)(a) Income by way of interest.")
    grown, _ = build(node, pages=(17,))
    assert grown.find("9(1)").pages == (17,)


def test_page_provenance_that_does_not_match_the_lines_is_refused():
    with pytest.raises(PageProvenanceError):
        build(section("9", "9. (1) First.", "(2) Second."), pages=(17,))


# --- against the whole corpus -----------------------------------------------


def test_every_section_round_trips(sub, act):
    assert round_trip_failures(sub.sections, act.sections) == []


def test_the_act_yields_its_sub_structure(sub):
    nodes = [node for section in sub.sections for node in section.walk()]
    assert len(nodes) == 7530
    assert sub.count(NodeType.SUBSECTION) == 2061
    assert sub.count(NodeType.CLAUSE) == 3041


def test_no_citation_nests_deeper_than_the_type_ladder(sub):
    deepest = max(
        node.path.depth for section in sub.sections for node in section.walk()
    )
    assert deepest == len(LEVEL_TYPES) + 1


def test_every_node_carries_page_provenance(sub):
    missing = [
        node.path.render()
        for section in sub.sections
        for node in section.walk()
        if not node.pages
    ]
    assert missing == []


def test_no_node_strays_off_its_section_pages(sub):
    strays = [
        node.path.render()
        for section in sub.sections
        for node in section.walk()
        if not set(node.pages) <= set(section.pages)
    ]
    assert strays == []


def test_a_level_only_ever_opens_at_its_alphabets_first_marker(sub):
    """Measured across the whole Act: no list starts part-way through."""
    late = [
        parent.path.render()
        for section in sub.sections
        for parent in section.walk()
        if parent.children
        and not any(opens(kind, parent.children[0].marker)
                    for kind in kinds_for(parent.children[0].marker))
    ]
    assert late == []


def test_the_act_reaches_five_levels_below_a_section(sub):
    assert sub.node("9(9)(b)(i)(A)(I)") is not None


def test_the_definitions_section_is_clause_rooted(sub):
    assert "2" in sub.clause_rooted
    assert sub.node("2(5)").type is NodeType.CLAUSE
    assert sub.node("2(5)(b)").type is NodeType.SUBCLAUSE
    # The Act's own words: "a process of the nature described in item (ii)".
    assert sub.node("2(5)(b)(ii)").type is NodeType.ITEM


def test_sub_section_rooted_sections_are_the_majority(sub):
    assert len(sub.clause_rooted) == 150


def test_the_residue_is_pinned(sub):
    """Any change to what the parser cannot place should be loud, not silent.

    Step 1.7 fixed section 206's 56 anomalies by trusting pdfplumber's
    measured table geometry there (a clean, two-column ruled grid); it is
    gone from this list. The rest are pages where pdfplumber's own extraction
    degenerates to a useless whole-page single cell (borderless tables) or
    were never table-related to begin with — pdfplumber does not help there,
    so the 1.6 heuristic is unchanged for them.
    """
    assert len(sub.anomalies) == 40
    # Only a duplicate citation invalidates a section's shape (Step 1.10).
    # The other 31 anomalies are still recorded and still reported; they
    # simply no longer cost a section its sub-structure.
    assert sub.unreliable == ("194", "376", "393")


def test_each_reason_names_the_refusal_it_describes():
    """The two refusals inside place() are unrelated and must stay distinct."""
    deep = ["1. (1) x"] + [f"({m}) deeper—" for m in ("a", "i", "A", "I")]
    _, too_deep = tree("1", *deep, "(a) below the ladder—")
    assert [a.reason for a in too_deep] == [AnomalyReason.TOO_DEEP]

    _, not_a_marker = tree("1", "1. (1) x", "(zz) fits no alphabet;")
    assert [a.reason for a in not_a_marker] == [AnomalyReason.NOT_A_MARKER]


def test_only_a_duplicate_path_invalidates_a_structure(sub):
    """The invariant is that a citation names exactly one chunk. A duplicate
    path breaks it directly; the other two refusals mint no citation at all."""
    duplicated = {
        anomaly.section
        for anomaly in sub.anomalies
        if anomaly.reason is AnomalyReason.DUPLICATE_PATH
    }
    assert set(sub.unreliable) == duplicated
    assert all(
        anomaly.invalidates_structure
        is (anomaly.reason is AnomalyReason.DUPLICATE_PATH)
        for anomaly in sub.anomalies
    )


def test_the_residue_is_still_mostly_in_section_tables(sub):
    """Down from >0.8 before Step 1.7 fixed 206 — the biggest table-driven
    cluster is gone, so the table share of what's left is a slimmer majority."""
    from_tables = [a for a in sub.anomalies if a.section in sub.table_sections]
    assert len(from_tables) / len(sub.anomalies) > 0.7


def test_a_section_with_no_anomaly_has_a_trusted_shape(sub):
    assert "9" not in sub.unreliable
    assert "44" not in sub.unreliable
    assert "416" not in sub.unreliable


def test_measured_table_geometry_makes_section_206_trustworthy(sub):
    """The dominant case Step 1.7 set out to fix — 56 of 1.6's 96 anomalies."""
    assert "206" not in sub.unreliable
    assert sub.node("206(1)(e)") is not None
