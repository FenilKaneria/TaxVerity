import pytest
from pydantic import ValidationError

from taxverity.chunking.models import (
    CHUNK_ID_CHARS,
    Chunk,
    build_breadcrumb,
    build_citation_label,
    compute_chunk_id,
)
from taxverity.corpus.nodes import NodeType

VERSION = "a" * 64


def section_chunk(**overrides) -> Chunk:
    fields = {
        "parent_id": None,
        "doc_id": "income-tax-act-2025",
        "node_type": NodeType.SECTION,
        "root_title": "Deductions in respect of certain payments",
        "chapter_numeral": "VIII",
        "chapter_title": "Deductions",
        "section_number": "123",
        "page_start": 40,
        "page_end": 41,
    }
    fields.update(overrides)
    text = fields.pop("text", "Body of the section.")
    node_path = fields.pop("node_path", "123")
    return Chunk.create(VERSION, node_path, text, **fields)


def test_chunk_id_is_stable_for_identical_inputs():
    assert compute_chunk_id(VERSION, "123(1)", "text") == compute_chunk_id(
        VERSION, "123(1)", "text"
    )
    assert len(compute_chunk_id(VERSION, "123", "text")) == CHUNK_ID_CHARS


@pytest.mark.parametrize(
    "args",
    [
        ("b" * 64, "123(1)", "text"),
        (VERSION, "123(2)", "text"),
        (VERSION, "123(1)", "other text"),
    ],
)
def test_chunk_id_changes_when_any_input_changes(args):
    assert compute_chunk_id(*args) != compute_chunk_id(VERSION, "123(1)", "text")


def test_chunk_id_does_not_collide_across_input_boundaries():
    assert compute_chunk_id(VERSION, "123", "45text") != compute_chunk_id(
        VERSION, "12345", "text"
    )


def test_chunk_id_ignores_the_soft_hyphen_and_nbsp():
    assert compute_chunk_id(VERSION, "123", "47\xa03. clause") == compute_chunk_id(
        VERSION, "123", "47 3. clause"
    )


def test_a_chunk_rejects_an_id_it_did_not_derive():
    payload = section_chunk().model_dump() | {"text": "tampered"}
    with pytest.raises(ValidationError, match="is not"):
        Chunk.model_validate(payload)


def test_round_trips_through_json():
    chunk = section_chunk(
        outgoing_refs=("2(5)", "Schedule III(1)"),
        defined_terms=("perquisite",),
        token_count=42,
    )
    assert Chunk.model_validate_json(chunk.model_dump_json()) == chunk


def test_citation_label_and_breadcrumb_are_derived():
    chunk = section_chunk(node_path="123(2)(a)", parent_id="0" * CHUNK_ID_CHARS)
    assert chunk.citation_label == (
        "Section 123(2)(a) — Deductions in respect of certain payments"
    )
    assert chunk.breadcrumb == (
        "Chapter VIII — Deductions / Section 123(2)(a) — "
        "Deductions in respect of certain payments"
    )
    assert chunk.embed_text() == f"{chunk.breadcrumb}\nBody of the section."


def test_a_schedule_renders_without_the_section_word():
    assert build_citation_label("Schedule III(1)(b)", None) == "Schedule III(1)(b)"
    assert build_citation_label("Schedule III", "Insurance business") == (
        "Schedule III — Insurance business"
    )


def test_breadcrumb_omits_an_absent_chapter():
    assert build_citation_label("123", None) == "Section 123"
    assert build_breadcrumb("Section 123", None, None) == "Section 123"
    assert build_breadcrumb("Section 123", "IV", None) == "Chapter IV / Section 123"


def test_a_schedule_chunk_names_its_schedule_not_a_section():
    chunk = section_chunk(
        node_path="Schedule III",
        node_type=NodeType.SCHEDULE,
        section_number=None,
        schedule_number="III",
    )
    assert chunk.is_root
    with pytest.raises(ValidationError, match="exactly one root"):
        section_chunk(
            node_path="Schedule III",
            node_type=NodeType.SCHEDULE,
            schedule_number="III",
        )


def test_root_fields_must_match_the_path():
    with pytest.raises(ValidationError, match="does not name"):
        section_chunk(node_path="124")


@pytest.mark.parametrize("node_type", [NodeType.ACT, NodeType.CHAPTER])
def test_uncitable_node_types_are_refused(node_type):
    with pytest.raises(ValidationError, match="not a citable"):
        section_chunk(node_type=node_type)


def test_a_child_names_its_parent_and_a_root_does_not():
    with pytest.raises(ValidationError, match="parent_id is set"):
        section_chunk(node_path="123(1)")
    with pytest.raises(ValidationError, match="parent_id is set"):
        section_chunk(parent_id="0" * CHUNK_ID_CHARS)


def test_outgoing_refs_must_be_citations():
    with pytest.raises(ValidationError, match="not a citation path"):
        section_chunk(outgoing_refs=("section 2(5)",))


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"page_start": 41, "page_end": 40}, "bad page span"),
        ({"page_start": -1, "page_end": 3}, "bad page span"),
        ({"char_start": 5}, "set together"),
        ({"char_start": 9, "char_end": 4}, "bad char span"),
        ({"token_count": -1}, "bad token count"),
        ({"text": "   "}, "must not be blank"),
        ({"parent_id": "zz"}, "not a chunk id"),
    ],
)
def test_invalid_spans_and_ids_are_refused(fields, message):
    with pytest.raises(ValidationError, match=message):
        section_chunk(**fields)


def test_char_span_is_optional_and_accepted_when_ordered():
    chunk = section_chunk(char_start=0, char_end=20)
    assert (chunk.char_start, chunk.char_end) == (0, 20)
    assert section_chunk().char_start is None
