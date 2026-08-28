import pytest

from taxverity.chunking.chunker import build_chunks, layout, subtree_refs, terms_in
from taxverity.corpus.crossrefs import (
    CrossReference,
    CrossReferenceIndex,
    GlossaryTerm,
    RefType,
)
from taxverity.corpus.nodes import NodePath, NodeType, StatutoryNode
from taxverity.corpus.sections import Chapter

VERSION = "a" * 64


def node(type, marker, path, text, *children, title=None, chapter=None, pages=(1,)):
    return StatutoryNode(
        type=type,
        marker=marker,
        path=NodePath.parse(path),
        text=text,
        title=title,
        chapter=chapter,
        pages=pages,
        children=tuple(children),
    )


def sample_section() -> StatutoryNode:
    return node(
        NodeType.SECTION,
        "123",
        "123",
        "123. Deductions.",
        node(
            NodeType.SUBSECTION,
            "1",
            "123(1)",
            "(1) The sums are—",
            node(NodeType.CLAUSE, "a", "123(1)(a)", "(a) life insurance premium;"),
            node(NodeType.CLAUSE, "b", "123(1)(b)", "(b) provident fund."),
        ),
        node(NodeType.SUBSECTION, "2", "123(2)", "(2) Nothing else qualifies."),
        title="Deductions",
        chapter="VIII",
        pages=(40, 41),
    )


# --- layout, pure ------------------------------------------------------------


def test_layout_reproduces_full_text():
    section = sample_section()
    text, _spans = layout(section)
    assert text == section.full_text()


def test_every_span_slices_out_its_own_node_text():
    section = sample_section()
    text, spans = layout(section)
    for descendant in section.walk():
        start, end = spans[descendant.citation]
        assert text[start:end] == descendant.full_text()


def test_the_root_span_covers_the_whole_subtree():
    text, spans = layout(sample_section())
    assert spans["123"] == (0, len(text))


def test_identical_sibling_text_gets_distinct_offsets():
    """Searching for a child text inside its parent would collapse these two."""
    section = node(
        NodeType.SECTION,
        "9",
        "9",
        "9. Repeats.",
        node(NodeType.SUBSECTION, "1", "9(1)", "same wording"),
        node(NodeType.SUBSECTION, "2", "9(2)", "same wording"),
    )
    _text, spans = layout(section)
    assert spans["9(1)"] != spans["9(2)"]


def test_a_node_with_no_text_of_its_own_still_spans_its_children():
    section = node(
        NodeType.SECTION,
        "9",
        "9",
        "",
        node(NodeType.SUBSECTION, "1", "9(1)", "(1) Only child."),
    )
    text, spans = layout(section)
    assert text == "(1) Only child."
    assert spans["9"] == spans["9(1)"] == (0, len(text))


# --- reference and glossary attribution, pure --------------------------------


def test_subtree_refs_gathers_every_descendant_reference():
    section = sample_section()
    by_source = {"123(1)(a)": ["80C"], "123(2)": ["12", "80C"]}
    assert subtree_refs(section, by_source) == ("80C", "12")
    assert subtree_refs(section.children[1], by_source) == ("12", "80C")


def test_terms_in_matches_case_insensitively():
    assert terms_in("The Assessee shall pay.", ["assessee", "company"]) == ("assessee",)


# --- build_chunks ------------------------------------------------------------


def chunks_for(section, **kwargs):
    return build_chunks(VERSION, [section], **kwargs)


def test_one_chunk_per_node_parents_first():
    chunks = chunks_for(sample_section())
    assert [chunk.node_path for chunk in chunks] == [
        "123",
        "123(1)",
        "123(1)(a)",
        "123(1)(b)",
        "123(2)",
    ]


def test_every_child_text_is_a_substring_of_its_parent():
    chunks = {chunk.chunk_id: chunk for chunk in chunks_for(sample_section())}
    for chunk in chunks.values():
        if chunk.parent_id is not None:
            assert chunk.text in chunks[chunk.parent_id].text


def test_a_parent_is_covered_by_its_own_lines_plus_its_children():
    chunks = chunks_for(sample_section())
    by_path = {chunk.node_path: chunk for chunk in chunks}
    parent = by_path["123(1)"]
    covered = sum(
        chunk.char_end - chunk.char_start
        for chunk in chunks
        if chunk.parent_id == parent.chunk_id
    )
    own_lines = len("(1) The sums are—")
    separators = 2
    assert covered + own_lines + separators == parent.char_end - parent.char_start


def test_char_offsets_are_relative_to_the_root_chunk():
    chunks = chunks_for(sample_section())
    root = chunks[0]
    assert (root.char_start, root.char_end) == (0, len(root.text))
    for chunk in chunks:
        assert root.text[chunk.char_start : chunk.char_end] == chunk.text


def test_root_metadata_is_carried_down_to_every_descendant():
    chapters = [Chapter(numeral="VIII", title="Deductions", page=39)]
    for chunk in chunks_for(sample_section(), chapters=chapters):
        assert chunk.section_number == "123"
        assert chunk.schedule_number is None
        assert chunk.root_title == "Deductions"
        assert chunk.chapter_numeral == "VIII"
        assert chunk.chapter_title == "Deductions"


def test_a_schedule_root_names_a_schedule_not_a_section():
    schedule = node(
        NodeType.SCHEDULE,
        "III",
        "Schedule III",
        "",
        node(NodeType.SCHEDULE_PARAGRAPH, "1", "Schedule III(1)", "1. A paragraph."),
        title="Insurance business",
    )
    root, paragraph = chunks_for(schedule)
    assert root.schedule_number == "III" and root.section_number is None
    assert paragraph.schedule_number == "III"
    assert paragraph.citation_label == "Schedule III(1) — Insurance business"


def test_pages_fall_back_to_the_root_when_a_node_has_none():
    section = node(
        NodeType.SECTION,
        "9",
        "9",
        "9. Body.",
        node(NodeType.SUBSECTION, "1", "9(1)", "(1) Child.", pages=()),
        pages=(11, 12),
    )
    _root, child = chunks_for(section)
    assert (child.page_start, child.page_end) == (11, 12)


def test_an_untrusted_root_yields_itself_and_nothing_below_it():
    chunks = chunks_for(sample_section(), untrusted={"123"})
    assert [chunk.node_path for chunk in chunks] == ["123"]
    assert chunks[0].text == sample_section().full_text()


def test_pruning_applies_at_any_depth_not_only_at_the_root():
    chunks = chunks_for(sample_section(), untrusted={"123(1)"})
    assert [chunk.node_path for chunk in chunks] == ["123", "123(1)", "123(2)"]


def test_references_and_defined_terms_are_attached():
    crossrefs = CrossReferenceIndex(
        references=(
            CrossReference(
                from_path="123(1)(a)",
                ref_type=RefType.SECTION,
                surface_text="section 80C",
                target_path="80C",
                resolved=True,
            ),
            CrossReference(
                from_path="123(2)",
                ref_type=RefType.SECTION,
                surface_text="section 999",
                resolved=False,
            ),
        ),
        external=(),
        glossary=(GlossaryTerm(term="life insurance premium", node_path="2(5)"),),
    )
    chunks = chunks_for(sample_section(), crossrefs=crossrefs)
    by_path = {chunk.node_path: chunk for chunk in chunks}
    assert by_path["123"].outgoing_refs == ("80C",)
    assert by_path["123(1)(a)"].outgoing_refs == ("80C",)
    assert by_path["123(2)"].outgoing_refs == ()
    assert by_path["123(1)(a)"].defined_terms == ("life insurance premium",)
    assert by_path["123(2)"].defined_terms == ()


def test_chunking_is_deterministic():
    assert chunks_for(sample_section()) == chunks_for(sample_section())


def test_ids_move_with_the_corpus_version():
    other = build_chunks("b" * 64, [sample_section()])
    assert [chunk.chunk_id for chunk in other] != [
        chunk.chunk_id for chunk in chunks_for(sample_section())
    ]


def test_a_root_without_a_citation_path_is_refused():
    orphan = StatutoryNode(type=NodeType.SECTION, marker="9", text="9. Body.")
    with pytest.raises(ValueError, match="citation path"):
        build_chunks(VERSION, [orphan])
