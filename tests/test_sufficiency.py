"""Step 8.1 — the evidence-sufficiency grader (ADR-099): its rules, the
registered ladder and adoption rule, and the measured rejection."""

from __future__ import annotations

import pytest

from conftest import VECTOR_STORE, Offline
from taxverity.config import Settings
from taxverity.embedding.store import load_vector_store
from taxverity.evals.bridge import BRIDGE_SCORES_FILENAME, BRIDGE_VECTORS_FILENAME
from taxverity.evals.gold import QuerySlice
from taxverity.evals.query_vectors import (
    QUERY_VECTORS_FILENAME,
    CachedQueryRetriever,
    QueryVectors,
)
from taxverity.evals.rerank import RERANK_SCORES_FILENAME, RerankScores, StoredReranker
from taxverity.evals.sufficiency import (
    Case,
    Rung,
    RungResult,
    Tally,
    candidates,
    cross_validate,
    judge_sufficiency,
    select,
    tally,
)
from taxverity.evals.taxonomy import FailureClassifier
from taxverity.retrieval.bm25 import BM25Retriever
from taxverity.retrieval.bridge import BridgedRetriever, TermBridge, load_bridge_map
from taxverity.retrieval.citations import CitationRetriever, ShortcutRetriever
from taxverity.retrieval.dense import DenseRetriever
from taxverity.retrieval.evidence import EVIDENCE_POOL, EvidencePacker
from taxverity.retrieval.fusion import FusionRetriever
from taxverity.retrieval.rerank import RERANK_DEPTH, RerankRetriever
from taxverity.retrieval.sufficiency import (
    GraderConfig,
    Reason,
    Signals,
    Sufficiency,
    grade,
)


def signals(*, hit=False, top=0.5, unmet=()):
    return Signals(citation_hit=hit, top_relevance=top, unmet=unmet)


# --- the grader -------------------------------------------------------------------


def test_an_unused_rule_never_fires():
    graded = grade(signals(top=-1.0, unmet=((1, "22", "21"),)), GraderConfig())
    assert graded.sufficiency is Sufficiency.SUFFICIENT
    assert graded.reasons == ()


def test_low_relevance_is_insufficient_and_the_threshold_itself_is_not():
    config = GraderConfig(min_relevance=0.3)
    assert grade(signals(top=0.29), config).reasons == (Reason.LOW_RELEVANCE,)
    assert grade(signals(top=0.3), config).sufficiency is Sufficiency.SUFFICIENT


def test_a_degraded_reranker_is_recorded_and_never_fires_the_loop():
    graded = grade(signals(top=None), GraderConfig(min_relevance=0.3))
    assert graded.sufficiency is Sufficiency.SUFFICIENT
    assert graded.reasons == (Reason.NO_RELEVANCE,)


def test_an_unmet_reference_counts_only_within_the_depth():
    unmet = ((2, "23", "21"),)
    assert grade(signals(unmet=unmet), GraderConfig(unmet_within=1)).sufficiency is Sufficiency.SUFFICIENT
    assert grade(signals(unmet=unmet), GraderConfig(unmet_within=2)).reasons == (Reason.UNMET_REFERENCE,)


def test_both_failures_are_reported():
    config = GraderConfig(min_relevance=0.3, unmet_within=1)
    assert grade(signals(top=0.1, unmet=((1, "22", "21"),)), config).reasons == (
        Reason.LOW_RELEVANCE,
        Reason.UNMET_REFERENCE,
    )


def test_a_citation_hit_overrides_only_when_asked():
    weak = signals(hit=True, top=0.0)
    assert grade(weak, GraderConfig(min_relevance=0.3)).sufficiency is Sufficiency.INSUFFICIENT
    graded = grade(weak, GraderConfig(min_relevance=0.3, citation_override=True))
    assert graded.sufficiency is Sufficiency.SUFFICIENT
    assert graded.reasons == (Reason.CITATION_HIT,)


def test_a_depth_below_one_is_refused():
    with pytest.raises(ValueError, match="unmet_within"):
        GraderConfig(unmet_within=0)


# --- the ladder and the rule --------------------------------------------------------


def case(query_id, *, missed=(), slice_=QuerySlice.PARAPHRASE, **kwargs):
    return Case(query_id=query_id, slice=slice_, signals=signals(**kwargs), missed=missed)


def test_tally_keeps_negatives_out_of_catches_and_false_flags():
    cases = [
        case("q001", missed=("20",)),
        case("q002"),
        case("q003", slice_=QuerySlice.NEGATIVE),
    ]
    assert tally(cases, {"q001", "q002", "q003"}) == Tally(
        caught=1, failing=1, false=1, sufficient=1, negatives_flagged=1, negatives=1
    )


def test_the_gate_is_half_caught_and_a_tenth_flagged_inclusive():
    assert Tally(caught=6, failing=12, false=5, sufficient=50, negatives_flagged=0, negatives=0).passes
    assert not Tally(caught=5, failing=12, false=0, sufficient=50, negatives_flagged=0, negatives=0).passes
    assert not Tally(caught=12, failing=12, false=6, sufficient=50, negatives_flagged=0, negatives=0).passes
    assert not Tally(caught=0, failing=0, false=0, sufficient=50, negatives_flagged=0, negatives=0).passes


def test_the_grid_holds_the_rule_unused_and_every_answerable_score():
    cases = [case("q001", top=0.2), case("q002", top=0.4), case("q003", top=0.9, slice_=QuerySlice.NEGATIVE)]
    assert [c.min_relevance for c in candidates(Rung.RELEVANCE, cases)] == [None, 0.2, 0.4]
    unmet = candidates(Rung.UNMET, cases)
    assert len(unmet) == 6 and not any(c.citation_override for c in unmet)
    assert all(c.citation_override for c in candidates(Rung.CITATION, cases))


def test_selection_separates_when_it_can():
    cases = [case("q001", missed=("20",), top=0.1), case("q002", top=0.5), case("q003", top=0.7)]
    assert select(Rung.RELEVANCE, cases) == GraderConfig(min_relevance=0.5)


def test_a_tie_goes_to_the_setting_that_fires_least():
    # Nothing separates, so every threshold scores at most zero: never firing wins.
    cases = [case("q001", missed=("20",), top=0.5), case("q002", top=0.5)]
    assert select(Rung.RELEVANCE, cases) == GraderConfig()


def test_each_question_is_graded_by_settings_chosen_on_the_other_fold():
    # Trained on the even fold, 0.5 separates and flags both odd questions;
    # trained on the odd fold, nothing separates, so it never fires.
    cases = [
        case("q001", top=0.3),
        case("q003", top=0.3, missed=("20",)),
        case("q002", top=0.1, missed=("20",)),
        case("q004", top=0.5),
    ]
    result = cross_validate(Rung.RELEVANCE, cases)
    assert result.fold_configs == (GraderConfig(), GraderConfig(min_relevance=0.5))
    assert result.flagged == ("q001", "q003")


def result(rung, caught, false):
    return RungResult(
        rung=rung,
        fold_configs=(GraderConfig(), GraderConfig()),
        held_out=Tally(caught=caught, failing=10, false=false, sufficient=50, negatives_flagged=0, negatives=0),
        flagged=(),
        shipped=GraderConfig(),
    )


def test_the_simplest_passing_rung_is_adopted():
    rung, verdict = judge_sufficiency([result(Rung.RELEVANCE, 5, 5), result(Rung.UNMET, 5, 5), result(Rung.CITATION, 5, 1)])
    assert rung is Rung.RELEVANCE and verdict.adopted


def test_a_later_rung_replaces_only_by_catching_more_with_no_more_false_flags():
    rung, _ = judge_sufficiency([result(Rung.RELEVANCE, 5, 2), result(Rung.UNMET, 7, 3), result(Rung.CITATION, 6, 2)])
    assert rung is Rung.CITATION


def test_no_passing_rung_rejects_with_every_reason():
    rung, verdict = judge_sufficiency([result(Rung.RELEVANCE, 4, 0), result(Rung.UNMET, 9, 6), result(Rung.CITATION, 9, 6)])
    assert rung is None and not verdict.adopted
    assert verdict.reasons[0] == "relevance: caught 4/10, flagged 0/50 sufficient"
    assert len(verdict.reasons) == 3


def test_the_rungs_must_come_in_ladder_order():
    with pytest.raises(ValueError, match="ladder order"):
        judge_sufficiency([result(Rung.UNMET, 5, 0), result(Rung.RELEVANCE, 5, 0), result(Rung.CITATION, 5, 0)])


# --- the measured result, on the real corpus ----------------------------------------


def test_the_grader_is_rejected_on_production_retrieval(gold, stored_chunks):
    """Pins ADR-099's measurement on the stored vectors and rerank scores: no
    rung passes, and the best relevance threshold within the false-flag cap
    catches 2 of 12. Insufficient packs do not score lower than sufficient ones.
    If retrieval changes and this moves, re-run scripts/measure_sufficiency.py."""
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
    bridge = TermBridge(load_bridge_map(), stored)
    fusion = FusionRetriever([CachedQueryRetriever(dense, questions), BM25Retriever(stored)])
    reranker = StoredReranker(scores)
    shortcut = CitationRetriever(stored)
    production = ShortcutRetriever(shortcut, BridgedRetriever(bridge, RerankRetriever(fusion, reranker)))
    packer = EvidencePacker(stored)
    classifier = FailureClassifier(stored)

    cases = []
    for query in gold:
        results = production.search(query.question, EVIDENCE_POOL)
        pack = packer.pack(results)
        rewritten = bridge.rewrite(query.question)
        relevance = reranker.score(rewritten, [r.chunk for r in fusion.search(rewritten, RERANK_DEPTH)])
        missed = ()
        if query.slice is not QuerySlice.NEGATIVE:
            failures = classifier.classify(
                query.query_id,
                query.required,
                [r.chunk.node_path for r in results],
                [u.citation for u in pack.units],
            )
            missed = tuple(f.label for f in failures)
        cases.append(
            Case(
                query_id=query.query_id,
                slice=query.slice,
                signals=Signals(
                    citation_hit=bool(shortcut.search(query.question, EVIDENCE_POOL)),
                    top_relevance=max(relevance.values()),
                    unmet=packer.unmet_references(pack),
                ),
                missed=missed,
            )
        )

    results = [cross_validate(rung, cases) for rung in Rung]
    rung, verdict = judge_sufficiency(results)
    assert rung is None and not verdict.adopted
    held = {r.rung: (r.held_out.caught, r.held_out.failing, r.held_out.false, r.held_out.sufficient) for r in results}
    assert held == {
        Rung.RELEVANCE: (6, 12, 22, 52),
        Rung.UNMET: (9, 12, 40, 52),
        Rung.CITATION: (9, 12, 28, 52),
    }
    capped = [
        t
        for t in (
            tally(cases, {c.query_id for c in cases if grade(c.signals, config).sufficiency is Sufficiency.INSUFFICIENT})
            for config in candidates(Rung.RELEVANCE, cases)
        )
        if t.false <= 0.1 * t.sufficient
    ]
    assert max(t.caught for t in capped) == 2
