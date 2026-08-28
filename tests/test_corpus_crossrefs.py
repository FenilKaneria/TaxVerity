
import pytest

from taxverity.corpus.crossrefs import (
    CrossReference,
    CrossReferenceIndex,
    GlossaryTerm,
    RefType,
    build_node_index,
    extract_crossrefs,
    extract_glossary,
    extract_node_references,
)
from taxverity.corpus.nodes import NodePath, NodeType, StatutoryNode


def referrer(text, marker="99"):
    return StatutoryNode(type=NodeType.SECTION, marker=marker, path=NodePath.section(marker), text=text)


def chain(root_type, root_marker, *rest):
    """A section or Schedule root with a single nested chain of descendants,
    e.g. ``chain(NodeType.SECTION, "10", NodeType.SUBSECTION, "2", NodeType.CLAUSE, "a")``
    builds 10 -> 10(2) -> 10(2)(a)."""
    path = NodePath.section(root_marker) if root_type is NodeType.SECTION else NodePath.schedule(root_marker)
    return _chain_from(root_type, root_marker, path, list(rest))


def _chain_from(node_type, marker, path, rest):
    if not rest:
        return StatutoryNode(type=node_type, marker=marker, path=path)
    child_type, child_marker, *deeper = rest
    child_path = path.child(child_type, child_marker)
    child = _chain_from(child_type, child_marker, child_path, deeper)
    return StatutoryNode(type=node_type, marker=marker, path=path, children=(child,))


def index_of(*roots):
    return build_node_index(roots)


# --- postfix section citations ----------------------------------------------


def test_a_bare_section_reference_resolves():
    target = chain(NodeType.SECTION, "10")
    refs, external = extract_node_references(referrer("as per section 10."), index_of(target))
    assert external == []
    assert len(refs) == 1
    ref = refs[0]
    assert ref.ref_type is RefType.SECTION
    assert ref.target_path == "10"
    assert ref.resolved is True


def test_a_bracket_chain_resolves_to_the_exact_depth():
    target = chain(NodeType.SECTION, "10", NodeType.SUBSECTION, "2", NodeType.CLAUSE, "a")
    refs, _ = extract_node_references(referrer("as per section 10(2)(a)."), index_of(target))
    assert refs[0].target_path == "10(2)(a)"
    assert refs[0].resolved is True


def test_a_numeric_range_expands_to_every_section_in_between():
    targets = [chain(NodeType.SECTION, str(n)) for n in (5, 6, 7)]
    refs, _ = extract_node_references(referrer("as per sections 5 to 7."), index_of(*targets))
    assert [r.target_path for r in refs] == ["5", "6", "7"]
    assert all(r.resolved for r in refs)


def test_a_bracket_only_continuation_inherits_the_full_prefix():
    """'section 70(1)(a), (c) and (d)' means clauses (a), (c), (d) all of
    sub-section (1) -- the whole prefix carries over, not just the number."""
    clause_a = StatutoryNode(
        type=NodeType.CLAUSE, marker="a",
        path=NodePath.section("70").child(NodeType.SUBSECTION, "1").child(NodeType.CLAUSE, "a"),
    )
    clause_c = clause_a.model_copy(update={"marker": "c", "path": NodePath.section("70").child(NodeType.SUBSECTION, "1").child(NodeType.CLAUSE, "c")})
    clause_d = clause_a.model_copy(update={"marker": "d", "path": NodePath.section("70").child(NodeType.SUBSECTION, "1").child(NodeType.CLAUSE, "d")})
    subsection_1 = StatutoryNode(
        type=NodeType.SUBSECTION, marker="1", path=NodePath.section("70").child(NodeType.SUBSECTION, "1"),
        children=(clause_a, clause_c, clause_d),
    )
    section_70 = StatutoryNode(type=NodeType.SECTION, marker="70", path=NodePath.section("70"), children=(subsection_1,))

    refs, _ = extract_node_references(
        referrer("section 70(1)(a), (c) and (d) applies."), index_of(section_70)
    )
    assert [r.target_path for r in refs] == ["70(1)(a)", "70(1)(c)", "70(1)(d)"]
    assert all(r.resolved for r in refs)


def test_a_list_item_without_a_matching_node_is_reported_dangling():
    clause_a = StatutoryNode(
        type=NodeType.CLAUSE, marker="a",
        path=NodePath.section("70").child(NodeType.SUBSECTION, "1").child(NodeType.CLAUSE, "a"),
    )
    subsection_1 = StatutoryNode(
        type=NodeType.SUBSECTION, marker="1", path=NodePath.section("70").child(NodeType.SUBSECTION, "1"),
        children=(clause_a,),
    )
    section_70 = StatutoryNode(type=NodeType.SECTION, marker="70", path=NodePath.section("70"), children=(subsection_1,))

    refs, _ = extract_node_references(referrer("section 70(1)(a) or (z) applies."), index_of(section_70))
    assert [r.target_path for r in refs] == ["70(1)(a)", "70(1)(z)"]
    assert [r.resolved for r in refs] == [True, False]


def test_a_line_wrap_inside_a_bracket_chain_does_not_break_resolution():
    target = chain(NodeType.SECTION, "10", NodeType.SUBSECTION, "15")
    refs, _ = extract_node_references(referrer("section 10(15)\n applies."), index_of(target))
    assert refs[0].target_path == "10(15)"
    assert refs[0].resolved is True


# --- prefix chain -------------------------------------------------------


def test_a_prefix_chain_reverses_into_the_postfix_depth_order():
    target = chain(NodeType.SECTION, "15", NodeType.SUBSECTION, "2", NodeType.CLAUSE, "a")
    text = "the meaning assigned to it in clause (a) of sub-section (2) of section 15;"
    refs, _ = extract_node_references(referrer(text), index_of(target))
    assert refs[0].target_path == "15(2)(a)"
    assert refs[0].resolved is True


def test_a_prefix_chain_with_a_trailing_postfix_bracket():
    """'clause (a) of section 80-ID(6)' -- the prefix chain combines with the
    Act's own postfix style, not just a bare section number."""
    target = chain(NodeType.SECTION, "80", NodeType.SUBSECTION, "6", NodeType.CLAUSE, "a")
    text = "the meaning assigned to it in clause (a) of section 80(6);"
    refs, _ = extract_node_references(referrer(text), index_of(target))
    assert refs[0].target_path == "80(6)(a)"
    assert refs[0].resolved is True


def test_a_hyphenated_1961_act_number_is_not_silently_truncated():
    """'section 80-ID' must never be truncated to a plain '80' that could
    coincidentally collide with an unrelated real section of this Act."""
    decoy = chain(NodeType.SECTION, "80", NodeType.CLAUSE, "a")
    text = "clause (a) of section 80-ID(6) of the Income-tax Act, 1961 (43 of 1961);"
    refs, external = extract_node_references(referrer(text), index_of(decoy))
    assert refs == []
    assert len(external) == 1
    assert external[0].act_name == "Income-tax Act"


# --- Schedules ------------------------------------------------------------


def test_a_bare_schedule_reference_resolves():
    target = StatutoryNode(type=NodeType.SCHEDULE, marker="II", path=NodePath.schedule("II"))
    refs, _ = extract_node_references(referrer("as specified in Schedule II."), index_of(target))
    assert refs[0].ref_type is RefType.SCHEDULE
    assert refs[0].target_path == "Schedule II"
    assert refs[0].resolved is True


def test_a_schedule_paragraph_with_a_part_resolves_to_the_folded_marker():
    paragraph = StatutoryNode(
        type=NodeType.SCHEDULE_PARAGRAPH, marker="A6",
        path=NodePath.schedule("XI").child(NodeType.SCHEDULE_PARAGRAPH, "A6"),
    )
    schedule = StatutoryNode(type=NodeType.SCHEDULE, marker="XI", path=NodePath.schedule("XI"), children=(paragraph,))
    text = "to the extent provided in paragraph 6 of Part A of Schedule XI;"
    refs, _ = extract_node_references(referrer(text), index_of(schedule))
    assert refs[0].ref_type is RefType.SCHEDULE_PARAGRAPH
    assert refs[0].target_path == "Schedule XI(A6)"
    assert refs[0].resolved is True


def test_a_schedule_paragraph_without_a_part_resolves():
    paragraph = StatutoryNode(
        type=NodeType.SCHEDULE_PARAGRAPH, marker="4",
        path=NodePath.schedule("I").child(NodeType.SCHEDULE_PARAGRAPH, "4"),
    )
    schedule = StatutoryNode(type=NodeType.SCHEDULE, marker="I", path=NodePath.schedule("I"), children=(paragraph,))
    refs, _ = extract_node_references(
        referrer("required to furnish a statement under paragraph 4 of Schedule I,"), index_of(schedule)
    )
    assert refs[0].target_path == "Schedule I(4)"
    assert refs[0].resolved is True


def test_a_bare_part_reference_resolves_coarsely_to_the_schedule():
    """A Part is folded into each paragraph's own marker, not a node of its
    own, so a bare 'Part A of Schedule XI' (no paragraph) can only resolve at
    Schedule granularity."""
    schedule = StatutoryNode(type=NodeType.SCHEDULE, marker="XI", path=NodePath.schedule("XI"))
    refs, _ = extract_node_references(referrer("continues to be approved as per Part A of Schedule XI;"), index_of(schedule))
    assert refs[0].ref_type is RefType.SCHEDULE_PART
    assert refs[0].target_path == "Schedule XI"
    assert refs[0].resolved is True


# --- self references ------------------------------------------------------


@pytest.mark.parametrize(
    ("phrase", "expected_type"),
    [
        ("this Act", RefType.THIS_ACT),
        ("this Chapter", RefType.THIS_CHAPTER),
        ("this Part", RefType.THIS_PART),
        ("this Schedule", RefType.THIS_SCHEDULE),
        ("this section", RefType.THIS_SECTION),
        ("this sub-section", RefType.THIS_SUBSECTION),
    ],
)
def test_self_references_resolve_trivially(phrase, expected_type):
    refs, external = extract_node_references(referrer(f"as provided in {phrase}."), index_of())
    assert external == []
    assert len(refs) == 1
    assert refs[0].ref_type is expected_type
    assert refs[0].target_path is None
    assert refs[0].resolved is True


def test_an_of_this_act_tail_confirms_rather_than_excludes():
    target = chain(NodeType.SECTION, "140")
    refs, external = extract_node_references(
        referrer("provisions of section 140 of this Act shall apply."), index_of(target)
    )
    assert external == []
    assert len(refs) == 1
    assert refs[0].target_path == "140"
    assert refs[0].resolved is True
    # "this Act" was consumed as this reference's own tail, not a second,
    # separate self-reference.
    assert not any(r.ref_type is RefType.THIS_ACT for r in refs)


# --- external-Act references ------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected_act"),
    [
        ("a company registered under section 8 of the Companies Act, 2013 (18 of 2013);", "Companies Act"),
        ("established under section 4 of the said Act;", "said Act"),
        (
            "assigned to it in section 2(h) of the Securities Contracts (Regulation) Act, 1956 (42 of 1956);",
            "Securities Contracts (Regulation) Act",
        ),
        (
            "referred to in section 6C of the Employees' Provident Funds and Miscellaneous Provisions Act, 1952;",
            "Employees' Provident Funds and Miscellaneous Provisions Act",
        ),
        (
            "assigned to them in sections 3(19) and 5(1) of the Insolvency and Bankruptcy Code, 2016;",
            "Insolvency and Bankruptcy Code",
        ),
        (
            "within the meaning of section 2(1)(v) of the Bharatiya Nagarik Suraksha Sanhita, 2023;",
            "Bharatiya Nagarik Suraksha Sanhita",
        ),
    ],
)
def test_an_external_act_reference_is_excluded_and_recorded(text, expected_act):
    refs, external = extract_node_references(referrer(text), index_of())
    assert refs == []
    assert len(external) == 1
    assert external[0].act_name == expected_act


def test_a_line_wrapped_act_name_is_still_detected():
    text = "abates under section 245HA of the Income-\ntax Act, 1961 (43 of 1961),"
    refs, external = extract_node_references(referrer(text), index_of())
    assert refs == []
    assert len(external) == 1
    assert "Income" in external[0].act_name and "tax Act" in external[0].act_name


# --- dangling residue, and the aggregate index -------------------------


def test_a_dangling_reference_is_kept_not_dropped():
    refs, _ = extract_node_references(referrer("as per section 999."), index_of())
    assert len(refs) == 1
    assert refs[0].resolved is False
    assert refs[0].target_path == "999"


def test_cross_reference_index_resolution_rate_and_dangling():
    index = CrossReferenceIndex(
        references=(
            CrossReference(from_path="1", ref_type=RefType.SECTION, surface_text="x", target_path="2", resolved=True),
            CrossReference(from_path="1", ref_type=RefType.SECTION, surface_text="y", target_path="3", resolved=True),
            CrossReference(from_path="1", ref_type=RefType.SECTION, surface_text="z", target_path="4", resolved=False),
        ),
        external=(),
        glossary=(),
    )
    assert index.resolution_rate == pytest.approx(2 / 3)
    assert [r.target_path for r in index.dangling] == ["4"]


def test_resolution_rate_of_an_empty_index_is_perfect():
    assert CrossReferenceIndex(references=(), external=(), glossary=()).resolution_rate == 1.0


# --- glossary ---------------------------------------------------------------


def test_the_glossary_maps_a_defined_term_to_its_clause():
    clause = StatutoryNode(
        type=NodeType.CLAUSE, marker="5",
        path=NodePath.section("2").child(NodeType.CLAUSE, "5"),
        text='(5)\xa0"agricultural income" means—',
    )
    section2 = StatutoryNode(type=NodeType.SECTION, marker="2", path=NodePath.section("2"), children=(clause,))
    terms = extract_glossary([section2])
    assert terms == (GlossaryTerm(term="agricultural income", node_path="2(5)"),)


def test_the_glossary_includes_a_delegating_clause():
    """A clause that only points elsewhere for its meaning is still where a
    reader looking the term up lands; the delegation is a separate outgoing
    CrossReference from the same clause."""
    clause = StatutoryNode(
        type=NodeType.CLAUSE, marker="1",
        path=NodePath.section("2").child(NodeType.CLAUSE, "1"),
        text='(1)\xa0"accountant" shall have the meaning assigned to it in section 515(3)(b);',
    )
    section2 = StatutoryNode(type=NodeType.SECTION, marker="2", path=NodePath.section("2"), children=(clause,))
    terms = extract_glossary([section2])
    assert terms == (GlossaryTerm(term="accountant", node_path="2(1)"),)


def test_the_glossary_is_empty_without_a_section_2():
    assert extract_glossary([chain(NodeType.SECTION, "5")]) == ()


# --- end-to-end over a small tree -------------------------------------------


def test_extract_crossrefs_combines_sections_and_schedules():
    target_section = chain(NodeType.SECTION, "10")
    schedule = StatutoryNode(type=NodeType.SCHEDULE, marker="II", path=NodePath.schedule("II"))
    referring_section = StatutoryNode(
        type=NodeType.SECTION, marker="20", path=NodePath.section("20"),
        text="as per section 10 and Schedule II, but not section 999.",
    )
    result = extract_crossrefs([target_section, referring_section], [schedule])
    assert len(result.references) == 3
    assert result.resolution_rate == pytest.approx(2 / 3)
    assert [r.from_path for r in result.references] == ["20", "20", "20"]


# --- integration against the real corpus ---------------------------------


def test_the_resolution_rate_clears_the_plan_s_98_percent_gate(crossrefs):
    assert crossrefs.resolution_rate >= 0.98


def test_the_residue_is_pinned(crossrefs):
    """Any change to what extraction cannot resolve should be loud, not
    silent -- the same discipline substructure.py and schedules.py already
    carry for their own residue."""
    assert len(crossrefs.dangling) == 39


def test_external_act_mentions_are_never_counted_as_dangling(crossrefs):
    assert len(crossrefs.external) == 388
    assert all(ext.act_name for ext in crossrefs.external)


def test_the_definitions_glossary_covers_section_2(crossrefs):
    assert len(crossrefs.glossary) == 109
    terms = {g.term for g in crossrefs.glossary}
    assert "accountant" in terms
    assert "agricultural income" in terms


def test_self_references_never_appear_dangling(crossrefs):
    self_types = {
        RefType.THIS_ACT, RefType.THIS_CHAPTER, RefType.THIS_PART,
        RefType.THIS_SCHEDULE, RefType.THIS_SECTION, RefType.THIS_SUBSECTION,
    }
    assert all(r.resolved for r in crossrefs.references if r.ref_type in self_types)


def test_extraction_is_reasonably_fast(sub, parsed_schedules):
    import time

    started = time.perf_counter()
    extract_crossrefs(sub.sections, parsed_schedules.schedules)
    assert time.perf_counter() - started < 5
