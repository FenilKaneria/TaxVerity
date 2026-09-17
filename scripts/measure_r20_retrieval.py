"""Step 20.1 — the R20 retrieval-quality and latency diagnosis.

Read-only against production's real composition (the R19/R18 stack, no
changes made here): measures Recall@k / Precision@k / MRR / Legal Rule
Coverage@k on `evals/datasets/retrieval_gold_r20.jsonl` (governing
citation, ranked list at k=10/20, and the delivered evidence pack), then
attributes every missed rule unit to one cause (chunking / context /
ranking / recall / budget), then times each retrieval sub-stage separately
for one simple and one complex question.

Not wired into CI: this is a diagnosis script per PLAN's R20 section,
producing `reports/r20_retrieval_diagnosis.md`. Bills Jina/Groq tokens —
run deliberately, not in a loop.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import psycopg

from taxverity.config import REPO_ROOT, Settings
from taxverity.db.chunks import load_chunks_from_db
from taxverity.db.serving import resolve_serving
from taxverity.embedding.jina_api import JinaAPIEmbedder
from taxverity.evals.gold_r20 import R20Slice, load_r20_gold_set
from taxverity.evals.metrics import CitationIndex, CreditMode, credits
from taxverity.evals.rule_coverage import score_coverage, score_coverage_run
from taxverity.observability import configure_logging, get_logger
from taxverity.retrieval.bm25 import BM25Retriever
from taxverity.retrieval.bridge import BridgedRetriever, TermBridge, load_bridge_map
from taxverity.retrieval.citations import (
    CitationRetriever,
    ShortcutRetriever,
    extract_query_citations,
)
from taxverity.retrieval.evidence import EVIDENCE_POOL, EvidencePacker
from taxverity.retrieval.fallback import FallbackRetriever
from taxverity.retrieval.fusion import FusionRetriever
from taxverity.retrieval.pgvector import PgVectorIndex
from taxverity.retrieval.rerank import CachedReranker, JinaReranker, RerankRetriever

logger = get_logger(__name__)

RANK_DEPTH = 100  # matches FusionRetriever/RerankRetriever's own read depth
PRODUCTION_BUDGET = 2_500  # graph/build.py's PRODUCTION_EVIDENCE_BUDGET

REPORT_PATH = REPO_ROOT / "reports" / "r20_retrieval_diagnosis.md"

# One "simple" (single governing provision, no explicit "section N" in the
# text so the citation shortcut does not intercept it) and one "complex"
# (multi-issue, two governing provisions) probe for the latency breakdown.
LATENCY_SIMPLE = "r009"
LATENCY_COMPLEX = "r007"


@dataclass
class FailureCause:
    unit: str
    cause: str
    detail: str


def classify_miss(
    unit: str,
    ranked_100: list[str],
    pool_k: int,
    pack_citations: set[str],
    index: CitationIndex,
    mode: CreditMode,
) -> FailureCause:
    if unit not in index:
        return FailureCause(unit, "chunking", "citation resolves to no chunk at all")
    in_100 = any(credits(c, unit, mode) for c in ranked_100)
    if not in_100:
        return FailureCause(unit, "recall", "not in fusion's top 100 at all")
    in_pool = any(credits(c, unit, mode) for c in ranked_100[:pool_k])
    if not in_pool:
        return FailureCause(unit, "ranking", f"in top 100 but outside the top {pool_k} pool")
    in_pack = any(credits(c, unit, mode) for c in pack_citations)
    if not in_pack:
        return FailureCause(unit, "budget", "in the pool but cut by the evidence-pack token budget")
    return FailureCause(unit, "context", "in the delivered pack under a lenient match, but not exact")


def main() -> None:
    configure_logging()
    settings = Settings()
    gold = load_r20_gold_set(REPO_ROOT / "evals" / "datasets" / "retrieval_gold_r20.jsonl")
    answerable = [q for q in gold if q.slice is not R20Slice.NEGATIVE]
    negatives = [q for q in gold if q.slice is R20Slice.NEGATIVE]

    conn = psycopg.connect(settings.require("database_url"), autocommit=True)
    embedding_set_id = settings.require("serving_embedding_set_id")
    served = resolve_serving(
        conn,
        corpus_version=settings.require("serving_corpus_version"),
        embedding_set_id=embedding_set_id,
    )
    chunks = load_chunks_from_db(conn, served.corpus_version)
    index = CitationIndex(chunks)

    embedder = JinaAPIEmbedder.from_settings(settings)
    bm25 = BM25Retriever(chunks)
    dense = PgVectorIndex(conn, embedding_set_id, embedder)
    fusion = FallbackRetriever(FusionRetriever([dense, bm25]), bm25)
    bridge = TermBridge(load_bridge_map(), chunks)
    reranker = CachedReranker(JinaReranker.from_settings(settings))
    ranked_retriever = BridgedRetriever(bridge, RerankRetriever(fusion, reranker, depth=EVIDENCE_POOL))
    retriever = ShortcutRetriever(CitationRetriever(chunks), ranked_retriever)
    packer = EvidencePacker(chunks, budget=PRODUCTION_BUDGET)

    runs_ranked: dict[str, list[str]] = {}
    packs: dict[str, list[str]] = {}
    causes: list[FailureCause] = []

    for query in answerable:
        results = retriever.search(query.question, RANK_DEPTH)
        citations = [r.chunk.node_path for r in results]
        runs_ranked[query.query_id] = citations
        pack = packer.pack(results[:EVIDENCE_POOL], expand=False)
        pack_citations = [unit.citation for unit in pack.units]
        packs[query.query_id] = pack_citations

        missing = [
            unit
            for unit in query.rule_units
            if not any(credits(c, unit, CreditMode.LENIENT) for c in pack_citations)
        ]
        for unit in missing:
            causes.append(
                classify_miss(
                    unit, citations, EVIDENCE_POOL, set(pack_citations), index, CreditMode.LENIENT
                )
            )
        logger.info(
            "%s: %d/%d rule units in pack, %d in ranked@100",
            query.query_id,
            len(query.rule_units) - len(missing),
            len(query.rule_units),
            sum(1 for u in query.rule_units if any(credits(c, u, CreditMode.LENIENT) for c in citations)),
        )

    # Recall@k / Precision@k / MRR / Coverage@k, on the ranked list at k=10
    # and k=20, and on the delivered pack (pack size varies per query, so it
    # is scored at its own length).
    report_ranked_10 = score_coverage_run(gold, runs_ranked, 10, index)
    report_ranked_20 = score_coverage_run(gold, runs_ranked, EVIDENCE_POOL, index)
    pack_scores = [
        score_coverage(q, packs[q.query_id], max(len(packs[q.query_id]), 1), index)
        for q in answerable
    ]

    # Negative-question regression check (R19 Phase D's q029 finding): does
    # anything in the pack cite a rule unit at all (it should not, since
    # these are out-of-corpus questions with no rule_units to find, but a
    # tangential hit like section 39's "customs duty" is exactly what a
    # relevance-blind retriever can surface).
    negative_pack_sizes = {}
    for query in negatives:
        results = retriever.search(query.question, RANK_DEPTH)
        pack = packer.pack(results[:EVIDENCE_POOL], expand=False)
        negative_pack_sizes[query.query_id] = len(pack.units)

    # --- Latency: sub-stage timing for one simple and one complex question ---
    # An UNCACHED reranker, deliberately: the main loop above already scored
    # every gold question through `reranker` (the CachedReranker used by the
    # real retriever chain), so timing that same instance again would report
    # a cache hit's near-zero cost, not a real rerank call's cost.
    raw_reranker = JinaReranker.from_settings(settings)
    latency = {}
    for label, qid in (("simple", LATENCY_SIMPLE), ("complex", LATENCY_COMPLEX)):
        query = next(q for q in gold if q.query_id == qid)
        latency[label] = _time_stages(query.question, dense, bm25, raw_reranker, packer)

    _write_report(
        gold_count=len(gold),
        report_ranked_10=report_ranked_10,
        report_ranked_20=report_ranked_20,
        pack_scores=pack_scores,
        causes=causes,
        negative_pack_sizes=negative_pack_sizes,
        latency=latency,
    )


def _time_stages(question, dense, bm25, reranker, packer):
    times = {}

    t0 = time.perf_counter()
    if extract_query_citations(question):
        # Shortcut would fire in production; still time embed/bm25/rerank so
        # the report shows what a non-shortcut question costs.
        pass
    vector = dense.embed_query(question)
    times["embed"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    dense_results = dense.search_vector(vector, RANK_DEPTH)
    times["pgvector"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    bm25_results = bm25.search(question, RANK_DEPTH)
    times["bm25"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    from taxverity.retrieval.fusion import reciprocal_rank_fusion

    fused = reciprocal_rank_fusion([dense_results, bm25_results])[:RANK_DEPTH]
    times["fusion"] = (time.perf_counter() - t0) * 1000

    head = fused[: EVIDENCE_POOL]
    t0 = time.perf_counter()
    try:
        scores = reranker.score(question, [r.chunk for r in head])
        order = sorted(range(len(head)), key=lambda i: -scores[head[i].chunk.chunk_id])
        reranked = [head[i] for i in order] + fused[EVIDENCE_POOL:]
    except Exception as error:  # noqa: BLE001 - diagnosis only
        logger.warning("rerank failed during measurement: %s", error)
        reranked = fused
    times["rerank"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    pack = packer.pack(reranked[:EVIDENCE_POOL], expand=False)
    times["pack"] = (time.perf_counter() - t0) * 1000

    times["total"] = sum(times.values())
    times["pack_units"] = len(pack.units)
    times["pack_tokens_proxy"] = sum(
        len(unit.chunk.embed_text().split()) for unit in pack.units
    )
    return times


def _write_report(*, gold_count, report_ranked_10, report_ranked_20, pack_scores, causes, negative_pack_sizes, latency):
    lines = ["# R20 retrieval-quality and latency diagnosis (Step 20.1)", ""]
    lines.append(f"Measured against the real Supabase corpus/production retriever composition, {gold_count} R20 gold questions.")
    lines.append("")
    lines.append("## Recall@k / Coverage@k on the ranked list")
    lines.append("")
    lines.append("| k | mean governing recall | mean precision | mean rule coverage | coverage_complete rate |")
    lines.append("|---|---|---|---|---|")
    for r in (report_ranked_10, report_ranked_20):
        lines.append(
            f"| {r.k} | {r.mean_governing_recall:.3f} | {r.mean_precision:.3f} | "
            f"{r.mean_rule_coverage:.3f} | {r.coverage_complete_rate:.3f} |"
        )
    lines.append("")
    lines.append("## Coverage on the delivered evidence pack (what the model actually sees)")
    lines.append("")
    mean_cov = sum(s.rule_coverage for s in pack_scores) / len(pack_scores)
    complete_rate = sum(1 for s in pack_scores if s.coverage_complete) / len(pack_scores)
    lines.append(f"Mean rule coverage: **{mean_cov:.3f}**. Coverage-complete rate: **{complete_rate:.3f}**.")
    lines.append("")
    lines.append("| query | slice | pack size | coverage | complete | missing units |")
    lines.append("|---|---|---|---|---|---|")
    for s in pack_scores:
        lines.append(
            f"| {s.query_id} | {s.slice.value} | {s.retrieved} | {s.rule_coverage:.2f} | "
            f"{s.coverage_complete} | {', '.join(s.missing_units) or '-'} |"
        )
    lines.append("")
    lines.append("## Failure attribution (per missed rule unit, lenient credit)")
    lines.append("")
    from collections import Counter

    counts = Counter(c.cause for c in causes)
    lines.append(f"Total misses: {len(causes)}. {dict(counts)}")
    lines.append("")
    lines.append("| unit | cause | detail |")
    lines.append("|---|---|---|")
    for c in causes:
        lines.append(f"| {c.unit} | {c.cause} | {c.detail} |")
    lines.append("")
    lines.append("## Negative-question regression check")
    lines.append("")
    for qid, size in negative_pack_sizes.items():
        lines.append(f"- {qid}: pack size {size} (0 expected; a non-zero pack is a false-positive relevance risk, same shape as the R19 Phase D q029 finding)")
    lines.append("")
    lines.append("## Latency, per sub-stage (ms)")
    lines.append("")
    lines.append("| stage | simple | complex |")
    lines.append("|---|---|---|")
    for stage in ("embed", "pgvector", "bm25", "fusion", "rerank", "pack", "total"):
        lines.append(f"| {stage} | {latency['simple'][stage]:.1f} | {latency['complex'][stage]:.1f} |")
    lines.append(f"| pack_units | {latency['simple']['pack_units']} | {latency['complex']['pack_units']} |")
    lines.append("")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    print(f"wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
