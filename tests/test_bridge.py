"""Step 5.8 — the statutory-term bridge (ADR-037, ADR-086): everyday words
mapped to the Act's own, the definitions pull, and the rule both are held to."""

from __future__ import annotations

import json

import pytest

from conftest import VECTOR_STORE, Offline
from taxverity.chunking.models import Chunk
from taxverity.config import Settings
from taxverity.corpus.crossrefs import defined_term
from taxverity.corpus.nodes import NodePath, NodeType
from taxverity.embedding.store import load_vector_store
from taxverity.evals.bridge import (
    BRIDGE_SCORES_FILENAME,
    BRIDGE_VECTORS_FILENAME,
    delivered,
    gained,
    judge_bridge,
    judge_definitions,
    lost,
)
from taxverity.evals.gold import GoldQuery, QuerySlice
from taxverity.evals.query_vectors import (
    QUERY_VECTORS_FILENAME,
    CachedQueryRetriever,
    QueryVectors,
)
from taxverity.evals.rerank import RERANK_SCORES_FILENAME, RerankScores, StoredReranker
from taxverity.evals.taxonomy import COHORTS_FILENAME, FailureCategory, load_cohorts
from taxverity.retrieval.base import ScoredChunk
from taxverity.retrieval.bm25 import BM25Retriever
from taxverity.retrieval.bridge import (
    MAX_ROOT_SHARE,
    BridgedRetriever,
    BridgeEntry,
    BridgeMapError,
    TermBridge,
    load_bridge_map,
    terms,
)
from taxverity.retrieval.citations import CitationRetriever, ShortcutRetriever
from taxverity.retrieval.dense import DenseRetriever
from taxverity.retrieval.evidence import EvidencePacker, EvidenceRole, EvidenceUnit
from taxverity.retrieval.fusion import FusionRetriever
from taxverity.retrieval.rerank import RerankRetriever

CORPUS_VERSION = "b" * 64
TYPES = (NodeType.SECTION, NodeType.CLAUSE)

# (citation, own lines, children, root title), laid out the way the chunker lays
# out a tree (ADR-055). "tax" is in every filler root, so it is too common to
# steer anything; "agricultural income" and "house property" are in one root.
S2 = (
    "2",
    "2. Definitions.",
    [
        ("2(1)", '(1) "agricultural income" means rent from land used for agriculture;', []),
        ("2(2)", '(2) "tax" means income-tax;', []),
    ],
    "Definitions.",
)
S20 = ("20", "20. The annual value of property is chargeable.", [], "Income from house property.")
FILLERS = [(str(n), f"{n}. Tax is charged on income.", [], "Charge.") for n in range(30, 40)]


def full_text(node) -> str:
    _, own, kids, *_ = node
    return "\n".join([own, *(full_text(kid) for kid in kids)])


def build(node, root, title, start=0, parent_id=None):
    citation, own, kids, *_ = node
    text = full_text(node)
    made = Chunk.create(
        CORPUS_VERSION,
        citation,
        text,
        parent_id=parent_id,
        doc_id="income-tax-act-2025",
        node_type=TYPES[NodePath.parse(citation).depth - 1],
        section_number=root,
        root_title=title,
        page_start=1,
        page_end=1,
        char_start=start,
        char_end=start + len(text),
    )
    chunks = [made]
    cursor = start + len(own) + 1
    for kid in kids:
        chunks += build(kid, root, title, cursor, made.chunk_id)
        cursor += len(full_text(kid)) + 1
    return chunks


CHUNKS = [
    chunk
    for node in (S2, S20, *FILLERS)
    for chunk in build(node, node[0], node[3])
]
BY_PATH = {chunk.node_path: chunk for chunk in CHUNKS}

AGRICULTURE = BridgeEntry(statutory="agricultural income", source="2(1)", lay=("farming", "farm income"))
HOUSE = BridgeEntry(statutory="house property", source="20", lay=("flat", "rented out"))
BRIDGE = TermBridge([AGRICULTURE, HOUSE], CHUNKS)


def test_terms_fold_a_trailing_plural_only():
    assert terms("Instalments of Taxes") == ("instalment", "of", "taxe")
    assert terms("business its gas") == ("business", "its", "gas")


def test_everyday_words_map_to_the_statute():
    assert BRIDGE.expand("Is my farming profit taxable?") == ("agricultural income",)
    assert BRIDGE.expand("Is farm income taxable?") == ("agricultural income",)


def test_an_already_statutory_question_is_left_alone():
    question = "Is agricultural income from farming taxable?"
    assert BRIDGE.expand(question) == ()
    assert BRIDGE.rewrite(question) == question


def test_terms_come_once_in_the_order_the_question_uses_them():
    assert BRIDGE.expand("I rented out a flat and do farming, farming") == (
        "house property",
        "agricultural income",
    )


def test_matching_is_by_whole_word():
    assert BRIDGE.expand("my farmhouse and my flatmate") == ()


def test_rewrite_appends_the_statutory_terms():
    assert BRIDGE.rewrite("Is farming taxed?") == "Is farming taxed? (agricultural income)"


@pytest.mark.parametrize(
    "entries",
    [
        pytest.param([BridgeEntry(statutory="house property", source="99", lay=("flat",))], id="no-source"),
        pytest.param([BridgeEntry(statutory="house property", source="2(1)", lay=("flat",))], id="not-in-source"),
        pytest.param([BridgeEntry(statutory="tax", source="2(2)", lay=("levy",))], id="too-common"),
        pytest.param(
            [AGRICULTURE, BridgeEntry(statutory="house property", source="20", lay=("farming",))],
            id="lay-twice",
        ),
        pytest.param(
            [BridgeEntry(statutory="house property", source="20", lay=("my house property",))],
            id="lay-says-target",
        ),
        pytest.param(
            [HOUSE, BridgeEntry(statutory="house property", source="20", lay=("apartment",))],
            id="target-twice",
        ),
        pytest.param([BridgeEntry(statutory="house property", source="20", lay=("?",))], id="empty-lay"),
    ],
)
def test_a_map_disagreeing_with_the_corpus_is_refused(entries):
    with pytest.raises(BridgeMapError):
        TermBridge(entries, CHUNKS)


def test_the_specificity_threshold_is_a_share_of_roots():
    assert BRIDGE.root_share("tax") > MAX_ROOT_SHARE
    assert BRIDGE.root_share("house property") == pytest.approx(1 / 12)


def test_only_a_specific_term_is_pullable():
    assert BRIDGE.pullable == ("2(1)",)
    assert BRIDGE.definitions("what is agricultural income tax") == (BY_PATH["2(1)"],)
    assert BRIDGE.definitions("what is tax") == ()


def test_a_bridged_term_is_defined_too():
    assert BRIDGE.definitions(BRIDGE.rewrite("Is farming taxed?")) == (BY_PATH["2(1)"],)


def test_a_map_of_another_version_is_refused(tmp_path):
    path = tmp_path / "map.json"
    path.write_text(json.dumps({"version": 2, "entries": []}), encoding="utf-8")
    with pytest.raises(BridgeMapError):
        load_bridge_map(path)


class Recorder:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def search(self, query, k):
        self.queries.append(query)
        return [ScoredChunk(chunk=BY_PATH["30"], score=1.0)]


def test_the_bridge_rewrites_what_the_ranking_sees():
    inner = Recorder()
    retriever = ShortcutRetriever(CitationRetriever(CHUNKS), BridgedRetriever(BRIDGE, inner))
    results = retriever.search("Is farming taxed?", 5)
    assert inner.queries == ["Is farming taxed? (agricultural income)"]
    assert [r.chunk.node_path for r in results] == ["30"]


@pytest.mark.parametrize(
    "question", ["What does section 20 say about farming?", "Is farming covered u/s 20?"]
)
def test_a_question_naming_a_provision_is_left_alone(question):
    inner = Recorder()
    retriever = ShortcutRetriever(CitationRetriever(CHUNKS), BridgedRetriever(BRIDGE, inner))
    results = retriever.search(question, 5)
    assert BRIDGE.expand(question) == ()
    assert inner.queries == [question]
    assert [r.chunk.node_path for r in results] == ["20", "30"]


# --- the definitions pull in the packer ----------------------------------------


def hits(*citations):
    return [ScoredChunk(chunk=BY_PATH[c], score=float(len(citations) - i)) for i, c in enumerate(citations)]


def test_a_definition_comes_after_every_hit():
    pack = EvidencePacker(CHUNKS).pack(hits("20", "30"), definitions=[BY_PATH["2(1)"]])
    assert [(u.citation, u.role, u.rank) for u in pack.units] == [
        ("20", EvidenceRole.RETRIEVED, 1),
        ("30", EvidenceRole.RETRIEVED, 2),
        ("2(1)", EvidenceRole.DEFINITION, 3),
    ]


def test_a_definition_never_displaces_a_hit():
    packer = EvidencePacker(CHUNKS)
    alone = packer.pack(hits("20"))
    tight = EvidencePacker(CHUNKS, budget=alone.tokens)
    pack = tight.pack(hits("20"), definitions=[BY_PATH["2(1)"]])
    assert [u.citation for u in pack.units] == ["20"]


def test_a_definition_already_carried_is_not_repeated():
    pack = EvidencePacker(CHUNKS).pack(hits("2"), definitions=[BY_PATH["2(1)"]])
    assert [(u.citation, u.role) for u in pack.units] == [("2", EvidenceRole.RETRIEVED)]


def test_a_definition_outside_the_chunk_set_is_refused():
    (stranger,) = build(("50", "50. Elsewhere.", [], "Elsewhere."), "50", "Elsewhere.")
    with pytest.raises(ValueError):
        EvidencePacker(CHUNKS).pack([], definitions=[stranger])


def test_a_definition_is_cited_by_nothing():
    with pytest.raises(ValueError):
        EvidenceUnit(
            chunk=BY_PATH["2(1)"], context=(), rank=1, tokens=1,
            role=EvidenceRole.DEFINITION, cited_by="20",
        )


# --- the rule ------------------------------------------------------------------


def gold_query(query_id, required, slice_=QuerySlice.PARAPHRASE):
    return GoldQuery(
        query_id=query_id, question="a question?", required=required, slice=slice_, notes="n"
    )


GOLD = [
    gold_query("q001", ["20"]),
    gold_query("q002", ["2(1)", "30"], QuerySlice.CROSSREF),
    gold_query("q003", [], QuerySlice.NEGATIVE),
]


def test_delivered_credits_an_ancestor_and_skips_negatives():
    runs = {"q001": ["30"], "q002": ["2", "31"], "q003": ["20"]}
    assert delivered(GOLD, runs) == {"q001": frozenset(), "q002": frozenset({"2(1)"})}


def test_the_bridge_is_adopted_on_a_recovery_with_no_loss():
    before = {"q1": frozenset(), "q2": frozenset({"30"})}
    after = {"q1": frozenset({"20"}), "q2": frozenset({"30"})}
    assert gained(before, after) == (("q1", "20"),)
    assert judge_bridge(before, after, [("q1", "20")], []).adopted


def test_the_bridge_is_rejected_on_any_loss_or_citation_rewrite():
    before = {"q1": frozenset(), "q2": frozenset({"30"})}
    after = {"q1": frozenset({"20"}), "q2": frozenset()}
    assert lost(before, after) == (("q2", "30"),)
    verdict = judge_bridge(before, after, [("q1", "20")], ["q9"])
    assert not verdict.adopted
    assert verdict.reasons == ("q2 lost 30", "rewrote citation-slice questions: q9")


def test_the_bridge_is_rejected_if_it_recovers_nothing():
    same = {"q1": frozenset(), "q2": frozenset({"30"})}
    assert judge_bridge(same, same, [("q1", "20")], []).reasons == (
        "recovered none of the 1 vocabulary labels",
    )


def test_the_definitions_pull_needs_a_gain():
    same = {"q1": frozenset(), "q2": frozenset({"30"})}
    assert not judge_definitions(same, same).adopted
    assert judge_definitions(same, {"q1": frozenset({"20"}), "q2": frozenset({"30"})}).adopted


# --- the real map, against the real corpus ------------------------------------


@pytest.fixture(scope="module")
def real_bridge(stored_chunks):
    _, stored = stored_chunks
    return TermBridge(load_bridge_map(), stored)


@pytest.mark.parametrize(
    ("question", "term"),
    [
        ("How is bitcoin taxed?", "virtual digital asset"),
        ("Can I set off a loss on my flat?", "house property"),
        ("When is the TDS certificate issued?", "deducted at source"),
        ("Is there a late fee for filing my ITR?", "return of income"),
    ],
)
def test_known_mappings_resolve(real_bridge, question, term):
    assert term in real_bridge.expand(question)


def test_no_citation_question_is_rewritten(real_bridge, gold):
    rewritten = [
        q.query_id for q in gold if q.slice is QuerySlice.CITATION and real_bridge.expand(q.question)
    ]
    assert rewritten == []


def test_the_measured_result_reproduces_offline(real_bridge, gold, stored_chunks):
    """Pins the Step 5.8 verdict (ADR-086) on the stored question vectors and
    rerank scores, gold and bridged, so a later change to retrieval or to the
    map that moves it fails here rather than silently."""
    rerank_dir = Settings().data_dir / "rerank"
    sources = [
        VECTOR_STORE / QUERY_VECTORS_FILENAME,
        VECTOR_STORE / BRIDGE_VECTORS_FILENAME,
        rerank_dir / RERANK_SCORES_FILENAME,
        rerank_dir / BRIDGE_SCORES_FILENAME,
    ]
    if not all(path.exists() for path in [VECTOR_STORE / "vector_manifest.json", *sources]):
        pytest.skip("run scripts/measure_hybrid.py, measure_rerank.py, then measure_bridge.py")
    corpus_version, stored = stored_chunks
    vectors, ids, manifest = load_vector_store(VECTOR_STORE, corpus_version=corpus_version)
    dense = DenseRetriever(stored, vectors, ids, manifest, Offline(manifest.model))
    questions: dict = {}
    for path in sources[:2]:
        questions |= QueryVectors.model_validate_json(path.read_text(encoding="utf-8")).vectors
    scores: dict = {}
    for path in sources[2:]:
        scores |= RerankScores.model_validate_json(path.read_text(encoding="utf-8")).scores
    fusion = FusionRetriever([CachedQueryRetriever(dense, questions), BM25Retriever(stored)])
    reranked = RerankRetriever(fusion, StoredReranker(scores))
    shortcut = CitationRetriever(stored)
    packer = EvidencePacker(stored)

    def packs(retriever):
        return {
            q.query_id: [u.citation for u in packer.pack(retriever.search(q.question, 20)).units]
            for q in gold
        }

    before = delivered(gold, packs(ShortcutRetriever(shortcut, reranked)))
    after = delivered(gold, packs(ShortcutRetriever(shortcut, BridgedRetriever(real_bridge, reranked))))
    assert gained(before, after) == (("q048", "194"), ("q058", "403(3)"), ("q059", "437(1)"))
    assert lost(before, after) == ()
    cohort = load_cohorts(Settings().evals_dir / "datasets" / COHORTS_FILENAME)
    assert judge_bridge(before, after, cohort.cohorts[FailureCategory.VOCABULARY], []).adopted


def test_the_glossary_reads_the_same_off_chunks(chunks, crossrefs):
    from_chunks = {
        term: chunk.node_path
        for chunk in chunks
        if chunk.section_number == "2" and NodePath.parse(chunk.node_path).depth == 2
        if (term := defined_term(chunk.text)) is not None
    }
    assert from_chunks == {g.term: g.node_path for g in crossrefs.glossary}
