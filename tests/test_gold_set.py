import json

import pytest
from pydantic import ValidationError

from taxverity.config import Settings
from taxverity.evals.gold import (
    GOLD_V1_FILENAME,
    GoldQuery,
    QuerySlice,
    by_slice,
    load_gold_set,
    to_json_line,
)

GOLD_PATH = Settings().evals_dir / "datasets" / GOLD_V1_FILENAME


def query(**fields) -> GoldQuery:
    fields.setdefault("query_id", "q001")
    fields.setdefault("slice", QuerySlice.CITATION)
    fields.setdefault("question", "What does section 22 allow?")
    fields.setdefault("required", ("22",))
    fields.setdefault("notes", "why this label")
    return GoldQuery(**fields)


# --- schema ------------------------------------------------------------------


def test_a_well_formed_query_validates():
    assert query().slice is QuerySlice.CITATION


@pytest.mark.parametrize("bad", ["1", "query-1", "Q001", "q1", "q0001"])
def test_a_query_id_must_be_an_ordinal(bad):
    with pytest.raises(ValidationError, match="query_id"):
        query(query_id=bad)


@pytest.mark.parametrize("field", ["question", "notes"])
def test_prose_fields_may_not_be_blank(field):
    with pytest.raises(ValidationError, match="required"):
        query(**{field: "   "})


def test_a_label_that_is_not_a_citation_is_refused():
    with pytest.raises(ValidationError):
        query(required=("chapter II",))


def test_a_repeated_label_is_refused():
    with pytest.raises(ValidationError, match="duplicate"):
        query(required=("22", "22"))


def test_a_non_negative_query_needs_at_least_one_label():
    with pytest.raises(ValidationError, match="at least one citation"):
        query(required=())


def test_a_negative_query_may_not_carry_a_label():
    """The point of the slice is that nothing in the corpus answers it. A
    labelled negative would silently become a positive with a lenient metric."""
    with pytest.raises(ValidationError, match="no citations"):
        query(slice=QuerySlice.NEGATIVE, required=("22",))


def test_a_negative_query_with_no_label_is_valid():
    assert query(slice=QuerySlice.NEGATIVE, required=()).required == ()


def test_a_query_is_frozen():
    with pytest.raises(ValidationError):
        query().question = "something else"


# --- serialisation -----------------------------------------------------------


def test_a_line_round_trips():
    original = query(required=("22(1)(a)", "21(1)"))
    assert GoldQuery.model_validate_json(to_json_line(original)) == original


def test_a_line_is_canonical_json():
    line = to_json_line(query())
    assert list(json.loads(line)) == sorted(json.loads(line))


def test_grouping_covers_every_slice_even_when_empty():
    grouped = by_slice((query(),))
    assert set(grouped) == set(QuerySlice)
    assert grouped[QuerySlice.PARAPHRASE] == ()


def test_a_duplicate_query_id_in_a_file_is_refused(tmp_path):
    path = tmp_path / GOLD_V1_FILENAME
    path.write_text(to_json_line(query()) + "\n" + to_json_line(query()) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate query_id"):
        load_gold_set(path)


def test_a_missing_file_is_named(tmp_path):
    with pytest.raises(FileNotFoundError, match=GOLD_V1_FILENAME):
        load_gold_set(tmp_path / GOLD_V1_FILENAME)


def test_blank_lines_are_skipped(tmp_path):
    path = tmp_path / GOLD_V1_FILENAME
    path.write_text(to_json_line(query()) + "\n\n", encoding="utf-8")
    assert len(load_gold_set(path)) == 1


# --- the shipped gold set ----------------------------------------------------


def test_the_gold_set_has_thirty_queries_across_four_slices(gold):
    counts = {member: len(queries) for member, queries in by_slice(gold).items()}
    assert sum(counts.values()) == 30
    assert all(count >= 6 for count in counts.values())


def test_query_ids_are_contiguous_from_one(gold):
    assert [q.query_id for q in gold] == [f"q{n:03d}" for n in range(1, len(gold) + 1)]


def test_no_two_queries_ask_the_same_thing(gold):
    assert len({q.question for q in gold}) == len(gold)


def test_a_citation_query_names_a_section_and_a_paraphrase_query_does_not(gold):
    """The slices are only meaningful if they differ in query surface form --
    a paraphrase carrying a section number would be a citation query."""
    grouped = by_slice(gold)
    assert all("section " in q.question.lower() for q in grouped[QuerySlice.CITATION])
    for member in (QuerySlice.PARAPHRASE, QuerySlice.NEGATIVE):
        assert all("section " not in q.question.lower() for q in grouped[member])


def test_a_crossref_query_needs_more_than_one_provision(gold):
    """The slice exists to measure 1-hop expansion. A single-label query would
    be indistinguishable from a paraphrase and would not exercise it."""
    for q in by_slice(gold)[QuerySlice.CROSSREF]:
        assert len(q.required) >= 2


def test_every_labelled_citation_exists_in_the_corpus(gold, chunks):
    """The success criterion of this step: a label naming nothing is a silent
    zero for any retriever, and would look like a retrieval failure."""
    paths = {chunk.node_path for chunk in chunks}
    missing = {citation for q in gold for citation in q.required if citation not in paths}
    assert not missing


def test_the_labels_are_spread_across_the_act(gold):
    """A gold set concentrated in one chapter measures one chapter."""
    roots = {citation.split("(")[0] for q in gold for citation in q.required}
    assert len(roots) >= 15


def test_the_file_on_disk_is_byte_identical_to_what_the_model_serialises(gold):
    """Same discipline as the chunk store: the artifact is canonical, so a
    hand-edit that changes formatting rather than content shows up as a diff."""
    written = GOLD_PATH.read_bytes().decode("utf-8")
    assert written == "".join(to_json_line(q) + "\n" for q in gold)
