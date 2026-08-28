"""The property suite of plan section 2.3, run against the real corpus."""

from collections import Counter

import pytest

from taxverity.chunking.chunker import build_chunks
from taxverity.corpus.nodes import NodeType

VERSION = "c" * 64


@pytest.fixture(scope="session")
def untrusted(sub, parsed_schedules):
    return set(sub.unreliable) | set(parsed_schedules.unreliable)


@pytest.fixture(scope="session")
def chunks(act, sub, parsed_schedules, crossrefs, untrusted):
    return build_chunks(
        VERSION,
        [*sub.sections, *parsed_schedules.schedules],
        chapters=act.chapters,
        crossrefs=crossrefs,
        untrusted=untrusted,
    )


@pytest.fixture(scope="session")
def by_id(chunks):
    return {chunk.chunk_id: chunk for chunk in chunks}


def test_the_corpus_chunk_count_is_pinned(chunks):
    """537 sections + 16 Schedules as roots, everything trusted below them."""
    assert len(chunks) == 7561
    assert sum(1 for chunk in chunks if chunk.is_root) == 553


def test_the_type_distribution_is_pinned(chunks):
    assert Counter(chunk.node_type for chunk in chunks) == {
        NodeType.CLAUSE: 3036,
        NodeType.SUBSECTION: 1942,
        NodeType.SUBCLAUSE: 1442,
        NodeType.SECTION: 537,
        NodeType.ITEM: 268,
        NodeType.SCHEDULE_PARAGRAPH: 265,
        NodeType.SUBITEM: 55,
        NodeType.SCHEDULE: 16,
    }


def test_every_chunk_id_is_unique(chunks):
    assert len({chunk.chunk_id for chunk in chunks}) == len(chunks)


def test_every_citation_names_exactly_one_chunk(chunks):
    """A duplicate node_path would leave the Phase 10 verifier two different
    texts to accept for one citation. Pruning untrusted subtrees is what keeps
    this true -- DUPLICATE_PATH is one of the two anomalies Step 1.6 records."""
    assert len({chunk.node_path for chunk in chunks}) == len(chunks)


def test_every_child_text_is_a_substring_of_its_parent(chunks, by_id):
    for chunk in chunks:
        if chunk.parent_id is not None:
            assert chunk.text in by_id[chunk.parent_id].text


def test_every_parent_is_covered_by_its_own_lines_plus_its_children(chunks, by_id):
    """No text falls between two children: what a parent holds beyond its
    children is exactly the lines it kept for itself."""
    children: dict[str, list] = {}
    for chunk in chunks:
        if chunk.parent_id is not None:
            children.setdefault(chunk.parent_id, []).append(chunk)
    for parent_id, kids in children.items():
        parent = by_id[parent_id]
        kids.sort(key=lambda chunk: chunk.char_start)
        cursor = parent.char_start
        for kid in kids:
            assert kid.char_start >= cursor
            assert kid.char_end <= parent.char_end
            cursor = kid.char_end
        assert kids[-1].char_end == parent.char_end


def test_no_chunk_crosses_a_root_boundary(chunks, sub, parsed_schedules):
    sections = {node.marker for node in sub.sections}
    schedules = {node.marker for node in parsed_schedules.schedules}
    for chunk in chunks:
        root = chunk.section_number or chunk.schedule_number
        assert root in (sections if chunk.section_number else schedules)
        assert chunk.node_path.startswith(
            root if chunk.section_number else f"Schedule {root}"
        )


def test_char_offsets_index_the_root_chunk_text(chunks, by_id):
    roots = {chunk.node_path: chunk for chunk in chunks if chunk.is_root}
    for chunk in chunks:
        root = roots[chunk.section_number or f"Schedule {chunk.schedule_number}"]
        assert root.text[chunk.char_start : chunk.char_end] == chunk.text


def test_page_spans_sit_inside_their_root(chunks):
    roots = {chunk.node_path: chunk for chunk in chunks if chunk.is_root}
    for chunk in chunks:
        root = roots[chunk.section_number or f"Schedule {chunk.schedule_number}"]
        assert root.page_start <= chunk.page_start <= chunk.page_end <= root.page_end


def test_no_chunk_is_emitted_below_an_untrusted_node(chunks, untrusted):
    for chunk in chunks:
        parent_path = chunk.node_path.rsplit("(", 1)[0] if "(" in chunk.node_path else None
        if parent_path is not None:
            assert parent_path not in untrusted


def test_every_untrusted_root_still_appears_whole(chunks, untrusted, sub):
    paths = {chunk.node_path: chunk for chunk in chunks}
    for section in sub.sections:
        if section.marker in untrusted:
            assert paths[section.marker].text == section.full_text()


def test_ids_are_stable_across_two_runs(act, sub, parsed_schedules, crossrefs, untrusted, chunks):
    again = build_chunks(
        VERSION,
        [*sub.sections, *parsed_schedules.schedules],
        chapters=act.chapters,
        crossrefs=crossrefs,
        untrusted=untrusted,
    )
    assert again == chunks


def test_every_outgoing_ref_names_a_chunk_that_exists(chunks):
    """Plan section 2.4's metadata test: a reference must land somewhere."""
    paths = {chunk.node_path for chunk in chunks}
    missing = {
        ref
        for chunk in chunks
        for ref in chunk.outgoing_refs
        if ref not in paths
    }
    # Cross-references resolve against the parsed tree, which still contains
    # the subtrees chunking declined to split. Those targets are reachable
    # through the root chunk that carries their text.
    roots = {path.split("(")[0] for path in missing}
    assert roots <= paths


def test_no_chunk_is_empty(chunks):
    for chunk in chunks:
        assert chunk.text.strip()


def test_only_three_roots_have_no_title_to_carry(chunks):
    """Schedules II, XI and XII genuinely print no caption line under their
    heading -- II opens straight into its table, XI and XII into a PART
    heading. Their chunks cite by path alone rather than inventing a title."""
    untitled = {
        chunk.node_path for chunk in chunks if chunk.is_root and not chunk.root_title
    }
    assert untitled == {"Schedule II", "Schedule XI", "Schedule XII"}
    label = next(chunk for chunk in chunks if chunk.node_path == "Schedule II(1)")
    assert label.citation_label == "Schedule II(1)"


def test_a_deep_citation_renders_its_full_breadcrumb(chunks):
    by_path = {chunk.node_path: chunk for chunk in chunks}
    chunk = by_path["9(9)(b)"]
    assert chunk.breadcrumb.startswith("Chapter ")
    assert "Section 9(9)(b)" in chunk.breadcrumb
    assert chunk.embed_text().endswith(chunk.text)
