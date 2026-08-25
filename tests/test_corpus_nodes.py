import pytest
from pydantic import ValidationError

from taxverity.corpus.nodes import (
    NodePath,
    NodeType,
    PathComponent,
    StatutoryNode,
)

# --- node types -------------------------------------------------------------


def test_proviso_and_explanation_are_absent_from_the_type_system():
    # Step 1.1 measured zero of either in this Act; the parser must not be able
    # to invent them.
    names = {member.name for member in NodeType}
    assert "PROVISO" not in names
    assert "EXPLANATION" not in names


# --- path rendering ---------------------------------------------------------


@pytest.mark.parametrize(
    "citation",
    [
        "1",
        "277",
        "354A",
        "80C(2)(a)",
        "277(1)(i)",
        "2(72)",
        "Schedule X",
        "Schedule XVI(1)(b)",
    ],
)
def test_citation_round_trips_through_parse_and_render(citation):
    assert NodePath.parse(citation).render() == citation


def test_depth_decides_component_type_when_reading_a_citation():
    path = NodePath.parse("80C(2)(a)")
    assert [component.type for component in path.components] == [
        NodeType.SECTION,
        NodeType.SUBSECTION,
        NodeType.CLAUSE,
    ]


def test_schedule_citations_use_schedule_paragraph_at_the_first_level():
    path = NodePath.parse("Schedule X(1)(b)")
    assert [component.type for component in path.components] == [
        NodeType.SCHEDULE,
        NodeType.SCHEDULE_PARAGRAPH,
        NodeType.CLAUSE,
    ]


def test_a_marker_that_looks_roman_is_not_retyped():
    # (i) is clause "i" on page 27 of the Act and roman one elsewhere. The model
    # stores what it is told and never re-derives the type from the marker.
    explicit = NodePath.section("5").child(NodeType.SUBCLAUSE, "i")
    assert explicit.components[-1].type is NodeType.SUBCLAUSE
    assert explicit.render() == "5(i)"
    assert NodePath.parse("5(i)").components[-1].type is NodeType.SUBSECTION


@pytest.mark.parametrize(
    "bad", ["", "   ", "(2)", "section 5", "5(2", "5()", "80-IA(4)"]
)
def test_malformed_citations_are_rejected(bad):
    with pytest.raises(ValueError):
        NodePath.parse(bad)


def test_citation_deeper_than_the_known_hierarchy_is_rejected():
    with pytest.raises(ValueError, match="deeper"):
        NodePath.parse("5(1)(a)(i)(x)")


# --- path construction ------------------------------------------------------


def test_child_extends_and_parent_retracts():
    section = NodePath.section("277")
    subsection = section.child(NodeType.SUBSECTION, "1")
    clause = subsection.child(NodeType.CLAUSE, "i")
    assert clause.render() == "277(1)(i)"
    assert clause.depth == 3
    assert clause.parent == subsection
    assert section.parent is None


def test_paths_are_frozen_and_comparable():
    assert NodePath.parse("277(1)") == NodePath.parse("277(1)")
    assert NodePath.parse("277(1)") != NodePath.parse("277(2)")
    with pytest.raises(ValidationError):
        NodePath.parse("277(1)").components = ()


def test_a_path_must_start_at_a_section_or_schedule():
    with pytest.raises(ValidationError):
        NodePath(components=(PathComponent(type=NodeType.CLAUSE, marker="a"),))


def test_a_path_cannot_nest_a_second_section():
    with pytest.raises(ValidationError):
        NodePath.section("5").child(NodeType.SECTION, "6")


def test_markers_may_not_carry_their_own_parentheses():
    with pytest.raises(ValidationError):
        PathComponent(type=NodeType.CLAUSE, marker="(a)")


def test_section_markers_must_be_numeric_with_an_optional_suffix():
    assert PathComponent(type=NodeType.SECTION, marker="354A").marker == "354A"
    with pytest.raises(ValidationError):
        PathComponent(type=NodeType.SECTION, marker="IV")


# --- nodes ------------------------------------------------------------------


def build_section() -> StatutoryNode:
    path = NodePath.section("277")
    return StatutoryNode(
        type=NodeType.SECTION,
        marker="277",
        path=path,
        title="Method of accounting in certain cases.",
        text="For the purposes of determining the income chargeable,—",
        chapter="XVII",
        pages=(333,),
        children=(
            StatutoryNode(
                type=NodeType.SUBSECTION,
                marker="1",
                path=path.child(NodeType.SUBSECTION, "1"),
                text="the valuation of inventory shall be made at lower of actual cost",
                pages=(333,),
            ),
            StatutoryNode(
                type=NodeType.SUBSECTION,
                marker="2",
                path=path.child(NodeType.SUBSECTION, "2"),
                text="the comparison shall be made category-wise.",
                pages=(333, 334),
            ),
        ),
    )


def test_node_exposes_its_citation():
    section = build_section()
    assert section.citation == "277"
    assert [child.citation for child in section.children] == ["277(1)", "277(2)"]


def test_walk_yields_the_node_and_every_descendant():
    citations = [node.citation for node in build_section().walk()]
    assert citations == ["277", "277(1)", "277(2)"]


def test_find_locates_a_descendant_by_citation():
    section = build_section()
    assert section.find("277(2)").marker == "2"
    assert section.find("277(9)") is None


def test_full_text_concatenates_own_text_then_children():
    text = build_section().full_text()
    assert text.startswith("For the purposes")
    assert "category-wise." in text
    assert text.count("\n") == 2


def test_provenance_pages_are_retained():
    assert build_section().find("277(2)").pages == (333, 334)


def test_a_node_needs_no_path_before_the_parser_assigns_one():
    orphan = StatutoryNode(type=NodeType.CLAUSE, marker="a")
    assert orphan.citation is None


def test_nodes_serialise_and_round_trip():
    section = build_section()
    restored = StatutoryNode.model_validate_json(section.model_dump_json())
    assert restored == section
    assert restored.find("277(1)").text == section.find("277(1)").text


def test_nodes_are_frozen():
    with pytest.raises(ValidationError):
        build_section().marker = "278"
