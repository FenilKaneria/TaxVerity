"""Step 5.7 — the retrieval failure taxonomy (ADR-085): one category per missed
label, by precedence, and the frozen cohorts later steps are judged on."""

from __future__ import annotations

import pytest

from taxverity.chunking.models import Chunk
from taxverity.config import Settings
from taxverity.corpus.nodes import NodePath, NodeType
from taxverity.evals.gold import QuerySlice
from taxverity.evals.metrics import UnresolvedCitationError
from taxverity.evals.taxonomy import (
    COHORTS_FILENAME,
    TAXONOMY_STAGE_VERSION,
    FailureCategory,
    FailureClassifier,
    LabelFailure,
    cohorts_of,
    load_cohorts,
    write_cohorts,
)

CORPUS_VERSION = "t" * 64
TYPES = (NodeType.SECTION, NodeType.SUBSECTION, NodeType.CLAUSE)

# citation -> outgoing references. A parent carries its subtree's references,
# as the chunker aggregates them (Step 2.2). `21(6)(b)` names no chunk, the way
# a reference into an untrusted, unsplit subtree does (ADR-056).
TREE = {
    "21": ("22",),
    "21(1)": (),
    "21(6)": ("22",),
    "22": ("21",),
    "22(1)": (),
    "22(2)": ("21",),
    "23": ("22(1)",),
    "24": ("21(6)(b)",),
    "25": (),
}


def make() -> list[Chunk]:
    made: dict[str, Chunk] = {}
    for citation, refs in TREE.items():
        path = NodePath.parse(citation)
        parent = path.parent
        made[citation] = Chunk.create(
            CORPUS_VERSION,
            citation,
            f"text of {citation}",
            parent_id=None if parent is None else made[parent.render()].chunk_id,
            doc_id="income-tax-act-2025",
            node_type=TYPES[path.depth - 1],
            section_number=path.components[0].marker,
            page_start=1,
            page_end=1,
            outgoing_refs=refs,
        )
    return list(made.values())


CLASSIFIER = FailureClassifier(make())


def only(required, pool=(), packed=()):
    return [
        (f.label, f.category, f.via)
        for f in CLASSIFIER.classify("q", required, pool, packed)
    ]


def test_a_label_the_pack_credits_is_not_a_failure():
    assert only(["22(2)"], packed=["22(2)"]) == []
    # An ancestor carries its descendant's text (ADR-060).
    assert only(["22(2)"], packed=["22"]) == []


def test_a_delivered_descendant_is_a_labelling_question():
    assert only(["22"], packed=["22(1)"]) == [
        ("22", FailureCategory.LABEL_GRANULARITY, ("22(1)",))
    ]


def test_a_label_in_the_pool_but_not_the_pack_is_budget():
    assert only(["25"], pool=["21", "25"], packed=["21"]) == [
        ("25", FailureCategory.BUDGET, ("25",))
    ]


def test_the_found_part_citing_the_missing_part_is_forward_dangling():
    # 22(2) cites 21, which contains 21(6).
    assert only(["22(2)", "21(6)"], packed=["22(2)"]) == [
        ("21(6)", FailureCategory.DANGLING_FORWARD, ("22(2)",))
    ]


def test_an_ancestor_carrying_the_found_part_can_be_the_citer():
    # 22 carries 22(2) and, through it, the reference to 21.
    assert only(["22(2)", "21(1)"], packed=["22"]) == [
        ("21(1)", FailureCategory.DANGLING_FORWARD, ("22",))
    ]


def test_the_missing_part_citing_the_found_part_is_backward_dangling():
    # 21(6) cites 22, which contains 22(1); 22(1) cites nothing back.
    assert only(["22(1)", "21(6)"], packed=["22(1)"]) == [
        ("21(6)", FailureCategory.DANGLING_BACKWARD, ("22(1)",))
    ]


def test_an_unlinked_part_of_a_two_label_question_is_multi_part():
    assert only(["25", "22(1)"], packed=["22(1)"]) == [("25", FailureCategory.MULTI_PART, ())]


def test_both_parts_missing_is_multi_part_for_each():
    assert only(["25", "21(1)"], packed=["23"]) == [
        ("25", FailureCategory.MULTI_PART, ()),
        ("21(1)", FailureCategory.MULTI_PART, ()),
    ]


def test_a_one_label_question_reaching_nothing_is_vocabulary():
    assert only(["25"], packed=["23"]) == [("25", FailureCategory.VOCABULARY, ())]


def test_an_incidental_link_from_an_unrelated_unit_is_still_vocabulary():
    # 23 cites 22(1), but 23 carries no part of this answer.
    assert only(["22(1)"], packed=["23"]) == [("22(1)", FailureCategory.VOCABULARY, ())]


def test_a_reference_into_an_unsplit_subtree_resolves_to_its_ancestor():
    # 24 cites 21(6)(b), which names no chunk; 21(6) does.
    assert CLASSIFIER.cites("24", "21(6)")
    assert CLASSIFIER.cites("24", "21")
    assert not CLASSIFIER.cites("24", "21(1)")


def test_precedence_granularity_then_budget_then_forward():
    assert only(["22", "23"], pool=["22"], packed=["22(1)", "23"])[0][1] is (
        FailureCategory.LABEL_GRANULARITY
    )
    assert only(["22(2)", "21(6)"], pool=["21(6)"], packed=["22(2)"])[0][1] is (
        FailureCategory.BUDGET
    )
    # 22(2) cites 21 and 21(6) cites 22: both directions hold, forward wins.
    assert CLASSIFIER.cites("21(6)", "22(2)")
    assert only(["22(2)", "21(6)"], packed=["22(2)"]) == [
        ("21(6)", FailureCategory.DANGLING_FORWARD, ("22(2)",))
    ]


def test_every_missed_label_gets_exactly_one_category():
    failures = CLASSIFIER.classify("q", ["25", "21(6)", "22(2)"], [], ["22(2)"])
    assert [f.label for f in failures] == ["25", "21(6)"]


def test_a_negative_is_refused():
    with pytest.raises(ValueError, match="negatives"):
        CLASSIFIER.classify("q", [], [], ["22"])


def test_an_unknown_citation_raises_rather_than_scoring_a_miss():
    with pytest.raises(UnresolvedCitationError):
        CLASSIFIER.classify("q", ["99"], [], [])
    with pytest.raises(UnresolvedCitationError):
        CLASSIFIER.classify("q", ["22"], [], ["99"])


def test_cohorts_group_by_category_and_round_trip(tmp_path):
    failures = [
        LabelFailure(query_id="q2", label="25", category=FailureCategory.VOCABULARY, via=()),
        LabelFailure(query_id="q1", label="21", category=FailureCategory.VOCABULARY, via=()),
        LabelFailure(query_id="q1", label="22", category=FailureCategory.BUDGET, via=("22",)),
    ]
    cohorts = cohorts_of(failures)
    assert cohorts.stage_version == TAXONOMY_STAGE_VERSION
    assert set(cohorts.cohorts) == set(FailureCategory)
    assert cohorts.cohorts[FailureCategory.VOCABULARY] == (("q1", "21"), ("q2", "25"))
    assert cohorts.cohorts[FailureCategory.MULTI_PART] == ()
    path = tmp_path / COHORTS_FILENAME
    write_cohorts(path, cohorts)
    first = path.read_bytes()
    assert load_cohorts(path) == cohorts
    write_cohorts(path, load_cohorts(path))
    assert path.read_bytes() == first


# --- corpus: the Step 5.7 composition, stored scores, no network --------------------


@pytest.fixture(scope="module")
def taxonomy(gold, stored_chunks, retrieval_legs):
    from taxverity.evals.rerank import (
        RERANK_SCORES_FILENAME,
        StoredReranker,
        load_rerank_scores,
    )
    from taxverity.retrieval.citations import ShortcutRetriever
    from taxverity.retrieval.evidence import EVIDENCE_POOL, EvidencePacker
    from taxverity.retrieval.fusion import FusionRetriever
    from taxverity.retrieval.rerank import MODEL_ID, RERANK_DEPTH, RerankRetriever

    path = Settings().data_dir / "rerank" / RERANK_SCORES_FILENAME
    if not path.exists():
        pytest.skip("run scripts/measure_rerank.py to score the gold pool")
    corpus_version, chunks = stored_chunks
    loaded = load_rerank_scores(
        path,
        model_id=MODEL_ID,
        corpus_version=corpus_version,
        depth=RERANK_DEPTH,
        questions=[q.question for q in gold],
    )
    _, dense, bm25, shortcut = retrieval_legs
    # Built here, not imported from production wiring: the frozen cohorts
    # describe this composition, whatever later steps put in front of it.
    retriever = ShortcutRetriever(
        shortcut, RerankRetriever(FusionRetriever([dense, bm25]), StoredReranker(loaded.scores))
    )
    packer = EvidencePacker(chunks)
    classifier = FailureClassifier(chunks)
    failures = []
    for query in gold:
        if query.slice is QuerySlice.NEGATIVE:
            continue
        pool = retriever.search(query.question, EVIDENCE_POOL)
        packed = [unit.citation for unit in packer.pack(pool, expand=False).units]
        pooled = [result.chunk.node_path for result in pool]
        failures += classifier.classify(query.query_id, query.required, pooled, packed)
    return failures


def test_the_real_classification_reproduces_the_frozen_cohorts(taxonomy):
    registered = Settings().evals_dir / "datasets" / COHORTS_FILENAME
    assert cohorts_of(taxonomy) == load_cohorts(registered)


def test_every_cohort_member_is_a_real_answerable_label(gold):
    cohorts = load_cohorts(Settings().evals_dir / "datasets" / COHORTS_FILENAME)
    required = {
        (q.query_id, label) for q in gold if q.slice is not QuerySlice.NEGATIVE for label in q.required
    }
    members = [pair for pairs in cohorts.cohorts.values() for pair in pairs]
    assert len(members) == len(set(members))
    assert set(members) <= required
