"""Steps 5.3 and 5.4 — evidence delivery: hits deduplicated by ancestry, each
with its ancestors' lead-in lines, within a token budget (ADR-082), plus what
they refer to, one hop out (ADR-083)."""

from __future__ import annotations

import random
from itertools import combinations

import pytest

from taxverity.chunking.models import Chunk
from taxverity.chunking.stats import estimate_tokens
from taxverity.corpus.nodes import NodePath, NodeType
from taxverity.evals.baseline import measure
from taxverity.evals.delivery import judge_delivery, judge_expansion
from taxverity.evals.gold import QuerySlice
from taxverity.evals.metrics import CreditMode, RunReport, Scores, score_run
from taxverity.retrieval.base import ScoredChunk
from taxverity.retrieval.citations import ShortcutRetriever
from taxverity.retrieval.evidence import (
    EVIDENCE_BUDGET,
    EVIDENCE_POOL,
    EXPANSION_HEAD,
    EvidencePacker,
    EvidenceRole,
    EvidenceUnit,
)
from taxverity.retrieval.fusion import FusionRetriever

CORPUS_VERSION = "v" * 64
TYPES = (NodeType.SECTION, NodeType.SUBSECTION, NodeType.CLAUSE)

# (citation, own lines, children), laid out the way the chunker lays out a tree:
# own lines first, then each child's full text, joined by newlines (ADR-055).
S22 = (
    "22",
    "22. Deductions from income from house property.",
    [
        (
            "22(1)",
            "(1) The following deductions shall be made from the annual value—",
            [
                ("22(1)(a)", "(a) thirty per cent of the annual value;", []),
                ("22(1)(b)", "(b) interest payable on borrowed capital.", []),
            ],
        ),
        ("22(2)", "(2) No deduction shall be made for any other sum.", []),
    ],
)
S23 = ("23", "23. Arrears of rent received.", [])
# A giant root whose bulk sits in a child, so its small sibling stays cheap.
S2 = (
    "2",
    "2. Definitions.",
    [
        ("2(1)", "(1) agricultural income means rent.", []),
        ("2(2)", "(2) " + "word " * 3000, []),
    ],
)


def full_text(node) -> str:
    _, own, kids = node
    return "\n".join([own, *(full_text(kid) for kid in kids)])


def build(node, root, start=0, parent_id=None, refs=None):
    citation, own, kids = node
    text = full_text(node)
    made = Chunk.create(
        CORPUS_VERSION,
        citation,
        text,
        parent_id=parent_id,
        doc_id="income-tax-act-2025",
        node_type=TYPES[NodePath.parse(citation).depth - 1],
        section_number=root,
        page_start=1,
        page_end=1,
        char_start=start,
        char_end=start + len(text),
        outgoing_refs=(refs or {}).get(citation, ()),
    )
    chunks = [made]
    cursor = start + len(own) + 1
    for kid in kids:
        chunks += build(kid, root, cursor, made.chunk_id, refs)
        cursor += len(full_text(kid)) + 1
    return chunks


CHUNKS = [*build(S22, "22"), *build(S23, "23"), *build(S2, "2")]
BY_PATH = {chunk.node_path: chunk for chunk in CHUNKS}

# The same tree with a reference graph over it, for Step 5.4. `21(6)(b)` names no
# chunk, the way a reference into an untrusted, unsplit subtree does (ADR-056).
S21 = (
    "21",
    "21. Annual value.",
    [
        ("21(1)", "(1) The annual value shall be the sum for which the property "
                  "might reasonably be expected to let from year to year.", []),
        ("21(6)", "(6) Where the property is self-occupied, its annual value is nil.", []),
    ],
)
REFS = {
    "22(1)(a)": ("21(1)",),
    "22(1)(b)": ("22", "23"),
    "22(2)": ("22",),
    "23": ("21",),
    "2(1)": ("21(6)(b)",),
}
XCHUNKS = [
    *build(S22, "22", refs=REFS),
    *build(S23, "23", refs=REFS),
    *build(S21, "21", refs=REFS),
    *build(S2, "2", refs=REFS),
]
XBY_PATH = {chunk.node_path: chunk for chunk in XCHUNKS}


def hits(*paths, chunks=None):
    by_path = BY_PATH if chunks is None else chunks
    return [
        ScoredChunk(chunk=by_path[path], score=float(len(paths) - i))
        for i, path in enumerate(paths)
    ]


def xhits(*paths):
    return hits(*paths, chunks=XBY_PATH)


def delivered(pack):
    return [unit.citation for unit in pack.units]


def cost(*paths):
    return EvidencePacker(CHUNKS).pack(hits(*paths)).tokens


def contains(outer: str, inner: str) -> bool:
    """`outer` carries `inner`'s text: the same node or an ancestor of it."""
    a, b = NodePath.parse(outer).components, NodePath.parse(inner).components
    return len(a) <= len(b) and b[: len(a)] == a


# --- lead-ins ---------------------------------------------------------------------


def test_a_clause_is_delivered_with_the_lines_above_it():
    (unit,) = EvidencePacker(CHUNKS).pack(hits("22(1)(a)")).units
    assert unit.chunk.text == "(a) thirty per cent of the annual value;"
    assert [(line.citation, line.text) for line in unit.context] == [
        ("22", "22. Deductions from income from house property."),
        ("22(1)", "(1) The following deductions shall be made from the annual value—"),
    ]


def test_a_root_hit_carries_no_context():
    (unit,) = EvidencePacker(CHUNKS).pack(hits("23")).units
    assert unit.context == ()


def test_every_lead_in_is_the_start_of_its_ancestors_own_text():
    """Verbatim, so the Phase 10 quote check can hold it against the ancestor."""
    packer = EvidencePacker(CHUNKS)
    for chunk in CHUNKS:
        (unit,) = packer.pack(hits(chunk.node_path)).units
        for line in unit.context:
            assert BY_PATH[line.citation].text.startswith(line.text + "\n")


def test_a_units_tokens_count_its_context():
    (unit,) = EvidencePacker(CHUNKS).pack(hits("22(1)(b)")).units
    assert unit.tokens == estimate_tokens(unit.chunk.text) + sum(
        estimate_tokens(line.text) for line in unit.context
    )


# --- deduplication by ancestry ----------------------------------------------------


def test_a_hit_inside_a_packed_ancestor_adds_nothing():
    pack = EvidencePacker(CHUNKS).pack(hits("22", "22(1)(a)"))
    assert delivered(pack) == ["22"]
    assert pack.skipped == ()


def test_an_ancestor_absorbs_packed_descendants_and_keeps_their_rank():
    pack = EvidencePacker(CHUNKS).pack(hits("22(1)(a)", "23", "22(1)(b)", "22(1)"))
    assert delivered(pack) == ["22(1)", "23"]
    assert [unit.rank for unit in pack.units] == [1, 2]


def test_absorbing_costs_only_the_difference():
    """Two siblings each repeat their ancestors' lead-ins, so their parent is
    cheaper than the pair. It must fit where the pair did."""
    budget = cost("22(1)(a)") + cost("22(1)(b)")
    pack = EvidencePacker(CHUNKS, budget=budget).pack(hits("22(1)(a)", "22(1)(b)", "22(1)"))
    assert delivered(pack) == ["22(1)"]
    assert pack.tokens <= budget


# --- the budget -------------------------------------------------------------------


def test_a_hit_that_does_not_fit_is_passed_over_and_the_walk_goes_on():
    pack = EvidencePacker(CHUNKS, budget=100).pack(hits("2", "23"))
    assert delivered(pack) == ["23"]
    assert pack.skipped == ("2",)


def test_a_hit_delivered_only_in_part_counts_as_skipped():
    pack = EvidencePacker(CHUNKS, budget=100).pack(hits("2", "2(1)"))
    assert delivered(pack) == ["2(1)"]
    assert pack.skipped == ("2",)


def test_no_unit_carries_another_and_the_budget_always_holds():
    """The two tests the plan names, as properties over shuffled rankings."""
    rng = random.Random(0)
    for budget in (20, 60, 200, 10_000):
        packer = EvidencePacker(CHUNKS, budget=budget)
        for _ in range(200):
            order = [chunk.node_path for chunk in CHUNKS]
            rng.shuffle(order)
            pack = packer.pack(hits(*order))
            assert pack.tokens <= budget
            for a, b in combinations(delivered(pack), 2):
                assert not contains(a, b) and not contains(b, a)
            ranks = [unit.rank for unit in pack.units]
            assert ranks == sorted(set(ranks))
            assert packer.pack(hits(*order)) == pack


def test_an_empty_ranking_gives_an_empty_pack():
    pack = EvidencePacker(CHUNKS).pack([])
    assert pack.units == () and pack.skipped == () and pack.tokens == 0


# --- refusals ---------------------------------------------------------------------


def test_a_budget_below_one_is_refused():
    with pytest.raises(ValueError, match="budget must be"):
        EvidencePacker(CHUNKS, budget=0)


def test_a_hit_outside_the_chunk_set_is_refused():
    with pytest.raises(ValueError, match="not in the packer's chunk set"):
        EvidencePacker(build(S23, "23")).pack(hits("22"))


def test_a_child_without_its_parent_is_refused():
    with pytest.raises(ValueError, match="parent is not in the chunk set"):
        EvidencePacker([BY_PATH["22(1)(a)"]])


def test_a_child_without_offsets_is_refused():
    child = BY_PATH["22(2)"].model_copy(update={"char_start": None, "char_end": None})
    with pytest.raises(ValueError, match="character offsets"):
        EvidencePacker([BY_PATH["22"], child])


# --- cross-reference expansion (Step 5.4) -----------------------------------------


def test_a_retrieved_clause_brings_what_it_refers_to():
    pack = EvidencePacker(XCHUNKS).pack(xhits("22(1)(a)"), expand=True)
    assert delivered(pack) == ["22(1)(a)", "21(1)"]
    hit, referenced = pack.units
    assert hit.role is EvidenceRole.RETRIEVED and hit.cited_by is None
    assert referenced.role is EvidenceRole.REFERENCED
    assert referenced.cited_by == "22(1)(a)" and referenced.rank == 1
    assert [line.citation for line in referenced.context] == ["21"]


def test_expansion_follows_one_hop_only():
    """22(1)(b) cites 23, and 23 cites 21. The second hop is not taken."""
    pack = EvidencePacker(XCHUNKS).pack(xhits("22(1)(b)"), expand=True)
    assert delivered(pack) == ["22(1)(b)", "23"]


def test_a_reference_to_the_citers_own_ancestor_adds_nothing():
    """"this section": the lead-in already carries its lines."""
    pack = EvidencePacker(XCHUNKS).pack(xhits("22(2)"), expand=True)
    assert delivered(pack) == ["22(2)"]


def test_a_reference_already_carried_adds_nothing():
    pack = EvidencePacker(XCHUNKS).pack(xhits("21", "22(1)(a)"), expand=True)
    assert delivered(pack) == ["21", "22(1)(a)"]


def test_a_reference_into_a_pruned_subtree_resolves_to_its_nearest_ancestor():
    pack = EvidencePacker(XCHUNKS).pack(xhits("2(1)"), expand=True)
    assert delivered(pack) == ["2(1)", "21(6)"]


def test_a_referenced_ancestor_of_a_hit_absorbs_it_and_stays_retrieved():
    pack = EvidencePacker(XCHUNKS).pack(xhits("21(1)", "23"), expand=True)
    assert delivered(pack) == ["21", "23"]
    assert [(unit.role, unit.rank) for unit in pack.units] == [
        (EvidenceRole.RETRIEVED, 1),
        (EvidenceRole.RETRIEVED, 2),
    ]


def test_expansion_is_off_by_default():
    """Rejected by its rule (ADR-083): production delivers the Step 5.3 pack."""
    pack = EvidencePacker(XCHUNKS).pack(xhits("22(1)(a)"))
    assert delivered(pack) == ["22(1)(a)"]


def test_referenced_text_outranks_the_tail_of_the_ranking_not_its_head():
    """One budget, three passes: the head, then references, then the tail."""
    both = EvidencePacker(XCHUNKS).pack(xhits("22(1)(a)"), expand=True)
    packer = EvidencePacker(XCHUNKS, budget=both.tokens)
    alone = EvidencePacker(XCHUNKS).pack(xhits("23"), expand=False)
    assert both.units[1].tokens > alone.tokens

    padding = ["22(1)(a)"] * (EXPANSION_HEAD - 1)
    in_head = packer.pack(xhits("22(1)(a)", *padding[:-1], "23"), expand=True)
    assert delivered(in_head) == ["22(1)(a)", "23"]

    in_tail = packer.pack(xhits("22(1)(a)", *padding, "23"), expand=True)
    assert delivered(in_tail) == ["22(1)(a)", "21(1)"]
    assert in_tail.skipped == ("23",)


def test_expansion_never_drops_what_the_head_alone_delivered():
    """Properties over shuffled rankings: the budget holds, no unit carries
    another, the pack is deterministic, every referenced unit names a citer the
    pack carries, and everything the head packs without expansion is still
    carried with it."""
    rng = random.Random(0)
    for budget in (20, 60, 200, 10_000):
        packer = EvidencePacker(XCHUNKS, budget=budget)
        for _ in range(200):
            order = [chunk.node_path for chunk in XCHUNKS]
            rng.shuffle(order)
            order *= 2  # longer than the head, so the tail pass runs too
            pack = packer.pack(xhits(*order), expand=True)
            assert pack.tokens <= budget
            got = delivered(pack)
            for a, b in combinations(got, 2):
                assert not contains(a, b) and not contains(b, a)
            assert packer.pack(xhits(*order), expand=True) == pack
            for unit in pack.units:
                if unit.role is EvidenceRole.REFERENCED:
                    assert any(contains(d, unit.cited_by) for d in got)
            head = packer.pack(xhits(*order[:EXPANSION_HEAD]), expand=False)
            for unit in head.units:
                assert any(contains(d, unit.citation) for d in got)


def test_a_referenced_unit_must_name_its_citer_and_only_it_may():
    unit = EvidencePacker(XCHUNKS).pack(xhits("23")).units[0]
    with pytest.raises(ValueError, match="cited_by"):
        EvidenceUnit(**{**unit.model_dump(), "chunk": unit.chunk, "cited_by": "22"})
    with pytest.raises(ValueError, match="cited_by"):
        EvidenceUnit(
            **{**unit.model_dump(), "chunk": unit.chunk, "role": EvidenceRole.REFERENCED}
        )


# --- unmet references (Step 8.1) --------------------------------------------------


def unmet(*paths):
    packer = EvidencePacker(XCHUNKS)
    return packer.unmet_references(packer.pack(xhits(*paths)))


def test_unmet_references_name_the_citing_unit_by_rank_position():
    assert unmet("22(1)(a)", "23") == ((1, "22(1)(a)", "21(1)"), (2, "23", "21"))


def test_a_target_carried_by_itself_or_an_ancestor_is_met():
    assert unmet("22(1)(a)", "21") == ()


def test_a_packed_descendant_does_not_carry_its_ancestor():
    assert unmet("23", "21(1)") == ((1, "23", "21"),)


def test_a_reference_to_the_units_own_lineage_is_not_unmet():
    assert unmet("22(2)") == ()


def test_a_reference_into_a_pruned_subtree_is_unmet_at_its_nearest_ancestor():
    assert unmet("2(1)") == ((1, "2(1)", "21(6)"),)


# --- the rule ---------------------------------------------------------------------


def scores(recall: float) -> dict[CreditMode, Scores]:
    return {mode: Scores(recall=recall, mrr=0.0, ndcg=0.0) for mode in CreditMode}


def report(recall, *, crossref=None, k=10) -> RunReport:
    return RunReport(
        k=k,
        scored=(),
        overall=scores(recall),
        per_slice={
            QuerySlice.CITATION: scores(1.0),
            QuerySlice.PARAPHRASE: scores(recall),
            QuerySlice.CROSSREF: scores(recall if crossref is None else crossref),
        },
        negatives=0,
    )


def test_delivery_holding_every_slice_is_adopted_without_needing_a_rise():
    verdict = judge_delivery(report(0.797, k=20), report(0.797))
    assert verdict.adopted and verdict.reasons == ()


def test_a_slice_losing_evidence_rejects_even_with_an_overall_rise():
    verdict = judge_delivery(report(0.82, crossref=0.50, k=20), report(0.797, crossref=0.562))
    assert verdict.reasons == ("crossref slice lenient recall fell 0.562 -> 0.500",)


def test_an_overall_fall_rejects():
    verdict = judge_delivery(report(0.70, crossref=0.9, k=20), report(0.797, crossref=0.9))
    assert "lenient recall fell 0.797 -> 0.700" in verdict.reasons


def test_expansion_raising_the_crossref_slice_is_adopted():
    verdict = judge_expansion(report(0.82, crossref=0.70, k=20), report(0.82, crossref=0.594, k=20))
    assert verdict.adopted and verdict.reasons == ()


def test_expansion_that_does_not_raise_the_crossref_slice_rejects():
    verdict = judge_expansion(report(0.83, crossref=0.594, k=20), report(0.82, crossref=0.594, k=20))
    assert verdict.reasons == ("crossref slice lenient recall did not rise (0.594 -> 0.594)",)


def test_expansion_displacing_evidence_elsewhere_rejects_despite_a_crossref_rise():
    verdict = judge_expansion(report(0.80, crossref=0.70, k=20), report(0.82, crossref=0.594, k=20))
    assert "lenient recall fell 0.820 -> 0.800" in verdict.reasons
    assert "paraphrase slice lenient recall fell 0.820 -> 0.800" in verdict.reasons


# --- corpus: the real hybrid + shortcut ranking, no network -----------------------


@pytest.fixture(scope="module")
def ranked_pool(gold, stored_chunks, retrieval_legs):
    index, dense, bm25, shortcut = retrieval_legs
    hybrid = ShortcutRetriever(shortcut, FusionRetriever([dense, bm25]))
    pool = {query.query_id: hybrid.search(query.question, EVIDENCE_POOL) for query in gold}
    return hybrid, index, EvidencePacker(stored_chunks[1]), pool


@pytest.fixture(scope="module")
def packs(ranked_pool):
    hybrid, index, packer, pool = ranked_pool
    return hybrid, index, {qid: packer.pack(results, expand=False) for qid, results in pool.items()}


@pytest.fixture(scope="module")
def expanded_packs(ranked_pool):
    _, _, packer, pool = ranked_pool
    return {qid: packer.pack(results, expand=True) for qid, results in pool.items()}


def test_every_gold_pack_fits_and_carries_no_text_twice(packs, expanded_packs):
    _, _, by_query = packs
    for pack in (*by_query.values(), *expanded_packs.values()):
        assert pack.tokens <= EVIDENCE_BUDGET
        for a, b in combinations(delivered(pack), 2):
            assert not contains(a, b) and not contains(b, a)
        for unit in pack.units:
            for line in unit.context:
                assert line.text.strip()


def test_delivery_loses_no_evidence_the_adopted_ranking_held(gold, packs):
    """The rule registered for Step 5.3 (ADR-082), and a floor under it."""
    hybrid, index, by_query = packs
    ranked = measure("hybrid", hybrid, gold, index, ordinal_scores=True).primary
    runs = {query_id: delivered(pack) for query_id, pack in by_query.items()}
    delivered_report = score_run(gold, runs, EVIDENCE_POOL, index)
    verdict = judge_delivery(delivered_report, ranked)
    assert verdict.adopted, verdict.reasons
    assert delivered_report.overall[CreditMode.LENIENT].recall >= 0.79


def test_expansion_is_rejected_by_its_rule_on_the_real_ranking(gold, packs, expanded_packs):
    """The rule registered for Step 5.4 and its measured outcome (ADR-083): the
    crossref slice rises, but referenced text displaces a paraphrase label from
    the pool's tail. If this flips (Step 5.6's reranker reorders the pool),
    re-run scripts/measure_xref.py and revisit the verdict."""
    _, index, by_query = packs
    runs = {query_id: delivered(pack) for query_id, pack in by_query.items()}
    before = score_run(gold, runs, EVIDENCE_POOL, index)
    runs = {query_id: delivered(pack) for query_id, pack in expanded_packs.items()}
    after = score_run(gold, runs, EVIDENCE_POOL, index)
    verdict = judge_expansion(after, before)
    crossref = QuerySlice.CROSSREF
    assert not verdict.adopted
    assert (
        after.per_slice[crossref][CreditMode.LENIENT].recall
        > before.per_slice[crossref][CreditMode.LENIENT].recall
    )
    assert any(reason.startswith("paraphrase slice") for reason in verdict.reasons)
