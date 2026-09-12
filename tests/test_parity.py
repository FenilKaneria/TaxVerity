"""Step 6.5 — the parity comparison and the HNSW rule, both pure.

The measurement itself needs a loaded database and lives in
`scripts/measure_parity.py`; what is tested here is the arithmetic that decides
what the measurement means.
"""

from __future__ import annotations

import psycopg
import pytest

from conftest import MICRO_V1 as V1
from conftest import VECTOR_STORE, Offline, micro_chunks
from taxverity.config import Settings
from taxverity.embedding.store import load_vector_store
from taxverity.evals.metrics import CreditMode, RunReport, Scores
from taxverity.evals.parity import (
    EXACT_P95_BUDGET_MS,
    agreement,
    judge_hnsw,
    judge_parity,
)
from taxverity.evals.query_vectors import (
    QUERY_VECTORS_FILENAME,
    load_query_vectors,
)
from taxverity.retrieval.base import ScoredChunk
from taxverity.retrieval.dense import DenseRetriever
from taxverity.retrieval.pgvector import PgVectorIndex

CHUNKS = {chunk.node_path: chunk for chunk in micro_chunks(V1)}
PATHS = sorted(CHUNKS)


def run(*specs: tuple[str, float]) -> list[ScoredChunk]:
    return [ScoredChunk(chunk=CHUNKS[path], score=score) for path, score in specs]


def report(recall: float, ndcg: float, k: int = 10) -> RunReport:
    score = Scores(recall=recall, mrr=recall, ndcg=ndcg)
    return RunReport(
        k=k,
        scored=(),
        overall={CreditMode.STRICT: score, CreditMode.LENIENT: score},
        per_slice={},
        negatives=0,
    )


def test_identical_runs_agree_completely():
    a = {"q1": run((PATHS[0], 0.9), (PATHS[1], 0.8))}
    found = agreement(a, dict(a), 10)
    assert found.identical == ("q1",)
    assert found.reordered == () and found.different == ()
    assert found.mean_overlap == 1.0
    assert found.max_score_delta == 0.0
    assert found.worst_query is None
    assert found.identical_share == 1.0


def test_same_chunks_in_another_order_is_reordered_not_identical():
    a = {"q1": run((PATHS[0], 0.9), (PATHS[1], 0.9))}
    b = {"q1": run((PATHS[1], 0.9), (PATHS[0], 0.9))}
    found = agreement(a, b, 10)
    assert found.reordered == ("q1",)
    assert found.identical == () and found.different == ()
    assert found.mean_overlap == 1.0


def test_a_swapped_chunk_is_different_and_halves_the_overlap():
    a = {"q1": run((PATHS[0], 0.9), (PATHS[1], 0.8))}
    b = {"q1": run((PATHS[0], 0.9), (PATHS[2], 0.8))}
    found = agreement(a, b, 10)
    assert found.different == ("q1",)
    assert found.mean_overlap == 0.5


def test_score_delta_is_taken_over_shared_chunks_and_names_the_worst_query():
    a = {
        "q1": run((PATHS[0], 0.9000000)),
        "q2": run((PATHS[1], 0.5000000)),
    }
    b = {
        "q1": run((PATHS[0], 0.9000001)),
        "q2": run((PATHS[1], 0.5001000)),
    }
    found = agreement(a, b, 10)
    assert found.worst_query == "q2"
    assert found.max_score_delta == pytest.approx(1e-4)


def test_agreement_truncates_to_k_before_comparing():
    a = {"q1": run((PATHS[0], 0.9), (PATHS[1], 0.8))}
    b = {"q1": run((PATHS[0], 0.9), (PATHS[2], 0.7))}
    assert agreement(a, b, 1).identical == ("q1",)
    assert agreement(a, b, 2).different == ("q1",)


def test_agreement_refuses_runs_over_different_queries():
    a = {"q1": run((PATHS[0], 0.9))}
    b = {"q2": run((PATHS[0], 0.9))}
    with pytest.raises(ValueError, match="same queries"):
        agreement(a, b, 10)


def test_parity_holds_when_only_the_scores_differ_within_tolerance():
    a = {"q1": run((PATHS[0], 0.9))}
    b = {"q1": run((PATHS[0], 0.9 + 1e-7))}
    verdict = judge_parity(agreement(a, b, 10), score_tolerance=1e-5)
    assert verdict.adopted and verdict.reasons == ()


def test_parity_fails_on_a_reordering_even_with_identical_scores():
    a = {"q1": run((PATHS[0], 0.9), (PATHS[1], 0.9))}
    b = {"q1": run((PATHS[1], 0.9), (PATHS[0], 0.9))}
    verdict = judge_parity(agreement(a, b, 10), score_tolerance=1e-5)
    assert not verdict.adopted
    assert "different order" in verdict.reasons[0]


def test_parity_fails_on_a_score_difference_above_tolerance():
    a = {"q1": run((PATHS[0], 0.9))}
    b = {"q1": run((PATHS[0], 0.91))}
    verdict = judge_parity(agreement(a, b, 10), score_tolerance=1e-5)
    assert not verdict.adopted
    assert "max score difference" in verdict.reasons[0]


def test_hnsw_is_rejected_when_exact_search_is_inside_the_budget():
    verdict = judge_hnsw(
        EXACT_P95_BUDGET_MS - 1, report(0.9, 0.9), report(0.9, 0.9)
    )
    assert not verdict.adopted
    assert "no latency constraint" in verdict.reasons[0]


def test_hnsw_is_rejected_when_it_costs_recall_even_if_exact_is_over_budget():
    verdict = judge_hnsw(
        EXACT_P95_BUDGET_MS + 1, report(0.85, 0.9), report(0.9, 0.9)
    )
    assert not verdict.adopted
    assert any("recall fell" in reason for reason in verdict.reasons)


def test_hnsw_is_rejected_when_it_costs_ndcg_alone():
    verdict = judge_hnsw(
        EXACT_P95_BUDGET_MS + 1, report(0.9, 0.85), report(0.9, 0.9)
    )
    assert not verdict.adopted
    assert any("nDCG fell" in reason for reason in verdict.reasons)


def test_hnsw_is_adopted_only_when_it_is_both_needed_and_free():
    verdict = judge_hnsw(
        EXACT_P95_BUDGET_MS + 1, report(0.9, 0.9), report(0.9, 0.9)
    )
    assert verdict.adopted and verdict.reasons == ()


def test_hnsw_rule_refuses_to_compare_two_different_k():
    with pytest.raises(ValueError, match="k="):
        judge_hnsw(200.0, report(0.9, 0.9, k=5), report(0.9, 0.9, k=10))


# --- against the loaded dev database ------------------------------------------
# The micro-corpus agreement test in test_pgvector.py pins the SQL on three
# chunks. This pins it on the real 8,351 and the whole gold set, which is where
# a tie-break or an ordering bug actually shows.


@pytest.fixture
def served_set(stored_chunks):
    """The embedding set holding the local store's own vectors. Looked up by the
    manifest's sha256, never by taking whatever set happens to be first."""
    url = Settings().database_url
    if url is None:
        pytest.skip("TAXVERITY_DATABASE_URL not set; run `docker compose up -d`")
    corpus_version, _ = stored_chunks
    _, _, manifest = load_vector_store(VECTOR_STORE, corpus_version=corpus_version)
    with psycopg.connect(url.get_secret_value()) as conn:
        row = conn.execute(
            "SELECT embedding_set_id FROM embedding_sets "
            "WHERE corpus_version = %s AND vectors_sha256 = %s",
            (corpus_version, manifest.vectors_sha256),
        ).fetchone()
        if row is None:
            pytest.skip("run scripts/ingest_corpus.py to load this corpus")
        yield conn, row[0]


def test_the_postgres_index_ranks_the_real_corpus_exactly_like_numpy(
    served_set, stored_chunks, gold, retrieval_legs
):
    conn, embedding_set_id = served_set
    corpus_version, stored = stored_chunks
    vectors, ids, manifest = load_vector_store(
        VECTOR_STORE, corpus_version=corpus_version
    )
    cached = load_query_vectors(
        VECTOR_STORE / QUERY_VECTORS_FILENAME,
        model=manifest.model,
        questions=[q.question for q in gold],
    )
    numpy_index = DenseRetriever(stored, vectors, ids, manifest, Offline(manifest.model))
    pg = PgVectorIndex(conn, embedding_set_id, Offline(manifest.model))

    numpy_runs, pg_runs = {}, {}
    for query in gold:
        vector = cached.vectors[query.question]
        numpy_runs[query.query_id] = numpy_index.search_vector(vector, 20)
        pg_runs[query.query_id] = pg.search_vector(vector, 20)

    found = agreement(numpy_runs, pg_runs, 20)
    assert judge_parity(found, score_tolerance=1e-5).adopted
    assert len(found.identical) == len(gold)
