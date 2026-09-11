"""Step 5.3 — evidence delivery: hits deduplicated by ancestry, each with its
ancestors' lead-in lines, within a token budget (ADR-082)."""

from __future__ import annotations

import random
from itertools import combinations

import pytest

from taxverity.chunking.models import Chunk
from taxverity.chunking.stats import estimate_tokens
from taxverity.corpus.nodes import NodePath, NodeType
from taxverity.evals.baseline import measure
from taxverity.evals.delivery import judge_delivery
from taxverity.evals.gold import QuerySlice
from taxverity.evals.metrics import CreditMode, RunReport, Scores, score_run
from taxverity.retrieval.base import ScoredChunk
from taxverity.retrieval.citations import ShortcutRetriever
from taxverity.retrieval.evidence import (
    EVIDENCE_BUDGET,
    EVIDENCE_POOL,
    EvidencePacker,
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


def build(node, root, start=0, parent_id=None):
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
    )
    chunks = [made]
    cursor = start + len(own) + 1
    for kid in kids:
        chunks += build(kid, root, cursor, made.chunk_id)
        cursor += len(full_text(kid)) + 1
    return chunks


CHUNKS = [*build(S22, "22"), *build(S23, "23"), *build(S2, "2")]
BY_PATH = {chunk.node_path: chunk for chunk in CHUNKS}


def hits(*paths):
    return [
        ScoredChunk(chunk=BY_PATH[path], score=float(len(paths) - i))
        for i, path in enumerate(paths)
    ]


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


# --- corpus: the real hybrid + shortcut ranking, no network -----------------------


@pytest.fixture(scope="module")
def packs(gold, stored_chunks, retrieval_legs):
    index, dense, bm25, shortcut = retrieval_legs
    hybrid = ShortcutRetriever(shortcut, FusionRetriever([dense, bm25]))
    packer = EvidencePacker(stored_chunks[1])
    return hybrid, index, {
        query.query_id: packer.pack(hybrid.search(query.question, EVIDENCE_POOL))
        for query in gold
    }


def test_every_gold_pack_fits_and_carries_no_text_twice(packs):
    _, _, by_query = packs
    for pack in by_query.values():
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
