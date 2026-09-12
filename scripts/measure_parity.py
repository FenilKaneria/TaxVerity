"""Step 6.5 — the Postgres index against the NumPy one on the gold set, and the
HNSW decision Step 6.2 deferred.

Reads the gold question vectors embedded once at Step 5.2, so it bills nothing
and needs no API key. Requires a loaded database: run `scripts/ingest_corpus.py`
first, and name the embedding set to serve — the schema cannot say which one is
current (ADR-090).

Builds an HNSW index on `chunk_embeddings`, measures it, and drops it again.

Writes `evals/reports/<corpus_version>/pgvector_parity.json` and
`reports/pgvector_parity.md`.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections.abc import Sequence
from pathlib import Path

import psycopg
from pydantic import BaseModel, ConfigDict

from taxverity.chunking.pipeline import read_corpus_version
from taxverity.chunking.store import load_chunks
from taxverity.config import Settings
from taxverity.db.ingest import vector_literal
from taxverity.embedding.backends import ModelInfo, describe
from taxverity.embedding.store import StaleVectorStoreError, load_vector_store
from taxverity.evals.baseline import PRIMARY_K, RetrieverBaseline, measure
from taxverity.evals.gold import GOLD_V2_FILENAME, GoldQuery, QuerySlice, load_gold_set
from taxverity.evals.ladder import Verdict
from taxverity.evals.metrics import CitationIndex, CreditMode
from taxverity.evals.parity import (
    EXACT_P95_BUDGET_MS,
    HNSW_EF_CONSTRUCTION,
    HNSW_EF_SEARCH,
    HNSW_M,
    Agreement,
    agreement,
    judge_hnsw,
    judge_parity,
)
from taxverity.evals.query_vectors import (
    QUERY_VECTORS_FILENAME,
    CachedQueryRetriever,
    load_query_vectors,
)
from taxverity.observability import configure_logging, get_logger
from taxverity.retrieval.base import ScoredChunk
from taxverity.retrieval.bm25 import BM25Retriever
from taxverity.retrieval.citations import CitationRetriever, ShortcutRetriever
from taxverity.retrieval.dense import DenseRetriever
from taxverity.retrieval.fusion import FusionRetriever
from taxverity.retrieval.pgvector import SEARCH_SQL, PgVectorIndex

logger = get_logger(__name__)

REPORT = Path("reports") / "pgvector_parity.md"
ARTIFACT = "pgvector_parity.json"
VECTORS_KEY = "jina-api"
SEARCH_REPEATS = 3
AGREEMENT_K = 20
# float32 both sides, cosine computed by Postgres against a float32 dot product
# in NumPy: a few ulps apart, not a different answer.
SCORE_TOLERANCE = 1e-5


class Offline:
    """The served identity with no way to embed. Every search here runs from a
    stored vector, so reaching the network would be a bug, not a slow path."""

    def __init__(self, info: ModelInfo) -> None:
        self._info = info

    def info(self) -> ModelInfo:
        return self._info

    def embed(self, texts, kind):
        raise AssertionError("the parity measurement must not embed")


class Arm(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    plan: str
    search_ms: tuple[float, float]
    dense: RetrieverBaseline
    hybrid: RetrieverBaseline


class ParityReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    corpus_version: str
    chunk_count: int
    gold_count: int
    primary_k: int
    model: str
    embedding_set_id: int
    stored_vectors: int
    numpy: Arm
    exact: Arm
    hnsw: Arm
    hnsw_rebuild: Arm
    hnsw_build_seconds: float
    hnsw_index_size: str
    exact_parity: Agreement
    exact_verdict: Verdict
    hnsw_parity: Agreement
    hnsw_rebuild_parity: Agreement
    hnsw_verdict: Verdict


def percentiles(samples: Sequence[float]) -> tuple[float, float]:
    ordered = sorted(samples)
    return ordered[len(ordered) // 2], ordered[math.ceil(0.95 * len(ordered)) - 1]


def timed(
    retriever: CachedQueryRetriever, gold: Sequence[GoldQuery]
) -> tuple[tuple[float, float], dict[str, Sequence[ScoredChunk]]]:
    """Best of three per query: the floor is the search, the spread above it is
    the machine."""
    samples: list[float] = []
    runs: dict[str, Sequence[ScoredChunk]] = {}
    for query in gold:
        best: float = math.inf
        hits: Sequence[ScoredChunk] = ()
        for _ in range(SEARCH_REPEATS):
            began = time.perf_counter()
            hits = retriever.search(query.question, AGREEMENT_K)
            best = min(best, (time.perf_counter() - began) * 1000)
        samples.append(best)
        runs[query.query_id] = hits
    return percentiles(samples), runs


def arm(
    name: str,
    plan: str,
    dense: CachedQueryRetriever,
    bm25: BM25Retriever,
    shortcut: CitationRetriever,
    gold: Sequence[GoldQuery],
    index: CitationIndex,
) -> tuple[Arm, dict[str, Sequence[ScoredChunk]]]:
    search_ms, runs = timed(dense, gold)
    hybrid = ShortcutRetriever(shortcut, FusionRetriever([dense, bm25]))
    return (
        Arm(
            name=name,
            plan=plan,
            search_ms=search_ms,
            dense=measure(f"{name} dense", dense, gold, index),
            hybrid=measure(
                f"{name} hybrid+shortcut", hybrid, gold, index, ordinal_scores=True
            ),
        ),
        runs,
    )


def explain(conn: psycopg.Connection, index: PgVectorIndex, vector: Sequence[float]) -> str:
    """The scan node the planner actually chose, so exact and approximate are
    observed rather than asserted."""
    literal = vector_literal(tuple(vector))
    rows = conn.execute(
        "EXPLAIN " + SEARCH_SQL,
        (literal, index.embedding_set_id, literal, PRIMARY_K),
    ).fetchall()
    for (line,) in rows:
        stripped = line.strip().removeprefix("->").strip()
        # The plan carries a scan per table; the vector one is the question here,
        # and the chunk join's own scan would answer a different one.
        if "Scan" in stripped and "chunk_embeddings" in stripped:
            return stripped.split("  (cost")[0]
    return "unknown"


def overall_row(name: str, baseline: RetrieverBaseline) -> str:
    at10 = baseline.reports[PRIMARY_K].overall
    at20 = baseline.reports[20].overall[CreditMode.LENIENT]
    return (
        f"| {name} | {at10[CreditMode.STRICT].recall:.3f} "
        f"| {at10[CreditMode.LENIENT].recall:.3f} | {at10[CreditMode.LENIENT].mrr:.3f} "
        f"| {at10[CreditMode.LENIENT].ndcg:.3f} | {at20.recall:.3f} |"
    )


def agreement_row(name: str, found: Agreement) -> str:
    return (
        f"| {name} | {len(found.identical)}/{found.queries} | {len(found.reordered)} "
        f"| {len(found.different)} | {found.mean_overlap:.3f} "
        f"| {found.max_score_delta:.2e} |"
    )


def render(report: ParityReport, elapsed: float) -> str:
    arms = (report.numpy, report.exact, report.hnsw, report.hnsw_rebuild)
    lines = [
        "# pgvector parity — Income-tax Act, 2025",
        "",
        "Auto-generated by `scripts/measure_parity.py` (Step 6.5). The Postgres",
        "index against the NumPy one on the gold v2 set, searched with the query",
        "vectors embedded once at Step 5.2 — this run bills no tokens.",
        "",
        "| Measure | Value |",
        "|---|---|",
        f"| corpus_version | `{report.corpus_version[:16]}…` |",
        f"| Chunks | {report.chunk_count} |",
        f"| Gold queries | {report.gold_count} |",
        f"| Primary k | {report.primary_k} |",
        f"| Model | `{report.model}` |",
        f"| Embedding set | {report.embedding_set_id} ({report.stored_vectors} vectors) |",
        f"| HNSW | m={HNSW_M}, ef_construction={HNSW_EF_CONSTRUCTION}, "
        f"ef_search={HNSW_EF_SEARCH} |",
        f"| HNSW build | {report.hnsw_build_seconds:.1f}s, {report.hnsw_index_size} |",
        f"| Run time | {elapsed:.1f}s |",
        "",
        "## Search latency, k=20",
        "",
        "Best of three per query, one query at a time, from this machine against",
        "the local container. The chunk join is included: it is what serving",
        "returns.",
        "",
        "| index | plan | p50 | p95 |",
        "|---|---|---|---|",
        *[
            f"| {a.name} | `{a.plan}` | {a.search_ms[0]:.2f} ms | {a.search_ms[1]:.2f} ms |"
            for a in arms
        ],
        "",
        "## Ranking agreement against NumPy, top 20",
        "",
        "By chunk id. **Reordered** is the same chunks in a different order — for",
        "an exact index that is a broken tie-break, not noise.",
        "",
        "| index | identical | reordered | different | mean overlap | max score delta |",
        "|---|---|---|---|---|---|",
        agreement_row("pgvector exact", report.exact_parity),
        agreement_row("pgvector HNSW", report.hnsw_parity),
        "",
        "The last row compares the two HNSW builds with each other, not with",
        "NumPy: the same vectors, the same settings, a second randomised graph.",
        "",
        "| comparison | identical | reordered | different | mean overlap | max score delta |",
        "|---|---|---|---|---|---|",
        agreement_row("HNSW build 1 vs build 2", report.hnsw_rebuild_parity),
        "",
        "## Gold-set scores",
        "",
        "Every score twice (ADR-060): **strict** credits only the labelled unit,",
        "**lenient** also credits an ancestor.",
        "",
        "| retriever | strict R@10 | lenient R@10 | lenient MRR | lenient nDCG@10 "
        "| lenient R@20 |",
        "|---|---|---|---|---|---|",
    ]
    for a in arms:
        lines.append(overall_row(f"{a.name} — dense", a.dense))
    for a in arms:
        lines.append(overall_row(f"{a.name} — hybrid + shortcut", a.hybrid))

    lines += [
        "",
        "Per slice, lenient recall / lenient nDCG at k=10, dense only:",
        "",
        "| slice | " + " | ".join(a.name for a in arms) + " |",
        "|---|" + "---|" * len(arms),
    ]
    for member in QuerySlice:
        if member is QuerySlice.NEGATIVE:
            continue
        cells = []
        for a in arms:
            scores = a.dense.primary.per_slice[member][CreditMode.LENIENT]
            cells.append(f"{scores.recall:.3f} / {scores.ndcg:.3f}")
        lines.append(f"| {member.value} | " + " | ".join(cells) + " |")

    parity = report.exact_verdict
    lines += [
        "",
        "## Verdict — parity",
        "",
        f"**{'Identical' if parity.adopted else 'NOT identical'}.** Two exact searches",
        "of the same vectors must return the same ranking; a score difference up to",
        f"{SCORE_TOLERANCE:.0e} is float noise.",
        "",
    ]
    lines += (
        ["The Postgres index is a drop-in for the NumPy one on this corpus."]
        if parity.adopted
        else [f"- {reason}" for reason in parity.reasons]
    )

    hnsw = report.hnsw_verdict
    lines += [
        "",
        "## Verdict — HNSW",
        "",
        "Registered in `evals/parity.py` before this run: an approximate index is",
        f"adopted only if exact search p95 exceeds **{EXACT_P95_BUDGET_MS:.0f} ms**",
        "— the budget argued from the query path's own vendor round trips, query",
        "embedding at p50 324 ms (Step 4.6) and reranking at p95 1,126 ms (Step",
        "5.6) — **and** costs no lenient recall@10 or nDCG@10.",
        "",
        f"**{'ADOPTED' if hnsw.adopted else 'REJECTED'}.**",
        "",
        *[f"- {reason}" for reason in hnsw.reasons],
        "",
        "A third cost the rule does not price, measured above: two builds of the",
        "same index over the same vectors do not rank alike, so an approximate",
        "index would make every retrieval measurement in this project depend on",
        "which graph happened to be built.",
        "",
        "Both indexes were dropped after the measurement. The numbers above are",
        "what HNSW would cost and buy if Step 6.6, or a larger corpus, ever needs",
        "it.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--embedding-set",
        type=int,
        required=True,
        help="the embedding set to serve; the schema cannot say which is current",
    )
    args = parser.parse_args()

    configure_logging()
    settings = Settings()
    started = time.perf_counter()
    store = settings.vectors_dir / VECTORS_KEY

    if settings.database_url is None:
        print("error: TAXVERITY_DATABASE_URL is not set", file=sys.stderr)
        return 1
    corpus_version = read_corpus_version(settings.interim_dir / "corpus_manifest.json")
    gold = load_gold_set(settings.evals_dir / "datasets" / GOLD_V2_FILENAME)
    try:
        chunks, _ = load_chunks(settings.interim_dir, corpus_version=corpus_version)
        vectors, ids, manifest = load_vector_store(store, corpus_version=corpus_version)
        cached = load_query_vectors(
            store / QUERY_VECTORS_FILENAME,
            model=manifest.model,
            questions=[q.question for q in gold],
        )
    except (RuntimeError, StaleVectorStoreError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    embedder = Offline(manifest.model)
    index = CitationIndex(chunks)
    bm25 = BM25Retriever(chunks)
    shortcut = CitationRetriever(chunks)
    numpy_dense = CachedQueryRetriever(
        DenseRetriever(chunks, vectors, ids, manifest, embedder), cached.vectors
    )
    probe = cached.vectors[gold[0].question]

    with psycopg.connect(settings.database_url.get_secret_value()) as conn:
        try:
            pg = PgVectorIndex(conn, args.embedding_set, embedder)
        except (StaleVectorStoreError, RuntimeError) as error:
            print(f"error: {error}", file=sys.stderr)
            return 1
        pg_dense = CachedQueryRetriever(pg, cached.vectors)

        numpy_arm, numpy_runs = arm(
            "numpy", "in-process", numpy_dense, bm25, shortcut, gold, index
        )
        exact_arm, exact_runs = arm(
            "pgvector exact",
            explain(conn, pg, probe),
            pg_dense,
            bm25,
            shortcut,
            gold,
            index,
        )

        # Built twice from the same vectors. HNSW's graph construction is
        # randomised, so a rebuild is a second sample of the same configuration:
        # whether the two agree is a property of the index, not of this machine.
        hnsw_arms, hnsw_runs_by_build = [], []
        build_seconds, size = 0.0, ""
        for build in (1, 2):
            began = time.perf_counter()
            conn.execute(
                f"CREATE INDEX chunk_embeddings_hnsw ON chunk_embeddings "
                f"USING hnsw (embedding vector_cosine_ops) "
                f"WITH (m = {HNSW_M}, ef_construction = {HNSW_EF_CONSTRUCTION})"
            )
            conn.commit()
            elapsed_build = time.perf_counter() - began
            index_size = conn.execute(
                "SELECT pg_size_pretty(pg_relation_size('chunk_embeddings_hnsw'))"
            ).fetchone()[0]
            logger.info(
                "built HNSW index %d in %.1fs, %s", build, elapsed_build, index_size
            )
            try:
                conn.execute(f"SET hnsw.ef_search = {HNSW_EF_SEARCH}")
                built, runs = arm(
                    "pgvector HNSW" if build == 1 else "pgvector HNSW, rebuilt",
                    explain(conn, pg, probe),
                    pg_dense,
                    bm25,
                    shortcut,
                    gold,
                    index,
                )
            finally:
                conn.execute("DROP INDEX chunk_embeddings_hnsw")
                conn.commit()
                logger.info("dropped HNSW index %d", build)
            hnsw_arms.append(built)
            hnsw_runs_by_build.append(runs)
            if build == 1:
                build_seconds, size = elapsed_build, index_size
        hnsw_arm, rebuilt_arm = hnsw_arms
        hnsw_runs, rebuilt_runs = hnsw_runs_by_build

        stored_vectors = len(pg)

    exact_parity = agreement(numpy_runs, exact_runs, AGREEMENT_K)
    hnsw_parity = agreement(numpy_runs, hnsw_runs, AGREEMENT_K)
    report = ParityReport(
        corpus_version=corpus_version,
        chunk_count=len(chunks),
        gold_count=len(gold),
        primary_k=PRIMARY_K,
        model=describe(manifest.model),
        embedding_set_id=args.embedding_set,
        stored_vectors=stored_vectors,
        numpy=numpy_arm,
        exact=exact_arm,
        hnsw=hnsw_arm,
        hnsw_rebuild=rebuilt_arm,
        hnsw_build_seconds=build_seconds,
        hnsw_index_size=size,
        exact_parity=exact_parity,
        exact_verdict=judge_parity(exact_parity, score_tolerance=SCORE_TOLERANCE),
        hnsw_parity=hnsw_parity,
        hnsw_rebuild_parity=agreement(hnsw_runs, rebuilt_runs, AGREEMENT_K),
        hnsw_verdict=judge_hnsw(
            exact_arm.search_ms[1],
            hnsw_arm.dense.primary,
            exact_arm.dense.primary,
        ),
    )

    elapsed = time.perf_counter() - started
    artifact = settings.evals_dir / "reports" / corpus_version / ARTIFACT
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text(
        json.dumps(report.model_dump(mode="json"), sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
        newline="",
    )
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(render(report, elapsed), encoding="utf-8", newline="")

    print(f"wrote {artifact}")
    print(f"wrote {REPORT}")
    print(
        f"parity: {'identical' if report.exact_verdict.adopted else 'BROKEN'}; "
        f"HNSW: {'adopted' if report.hnsw_verdict.adopted else 'rejected'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
