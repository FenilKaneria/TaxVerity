"""R20 Step 20.2 — live benchmark: does classifier-driven sub-query
decomposition (ADR-123) actually change Recall@k / Legal Rule Coverage@k /
coverage_complete_rate on the 12-question R20 gold set, versus the plain
single-query baseline Step 20.1 measured?

Read-only against production's real composition plus the real
`IntentClassifier` (which 20.1's script did not call — it measured
retrieval only). For every answerable question: get `sub_queries` from a
live classification call, then score both the plain baseline and the
sub-query-merged ranking (reusing `graph.nodes._merge_subquery_results`,
the exact production merge) against the same ranked-list and pack metrics
20.1 used. Bills Jina/Groq tokens — run deliberately, not in a loop.
"""

from __future__ import annotations

import psycopg

from taxverity.config import REPO_ROOT, Settings
from taxverity.db.chunks import load_chunks_from_db
from taxverity.db.serving import resolve_serving
from taxverity.embedding.jina_api import JinaAPIEmbedder
from taxverity.evals.gold_r20 import R20Slice, load_r20_gold_set
from taxverity.evals.metrics import CitationIndex
from taxverity.evals.rule_coverage import score_coverage, score_coverage_run
from taxverity.graph.nodes import _merge_subquery_results
from taxverity.observability import configure_logging, get_logger
from taxverity.retrieval.bm25 import BM25Retriever
from taxverity.retrieval.bridge import BridgedRetriever, TermBridge, load_bridge_map
from taxverity.retrieval.citations import CitationRetriever, ShortcutRetriever
from taxverity.retrieval.evidence import EVIDENCE_POOL, EvidencePacker
from taxverity.retrieval.fallback import FallbackRetriever
from taxverity.retrieval.fusion import FusionRetriever
from taxverity.retrieval.pgvector import PgVectorIndex
from taxverity.retrieval.rerank import CachedReranker, JinaReranker, RerankRetriever
from taxverity.safety.classifier import IntentClassifier

logger = get_logger(__name__)

RANK_DEPTH = 100
PRODUCTION_BUDGET = 2_500  # graph/build.py's PRODUCTION_EVIDENCE_BUDGET

REPORT_PATH = REPO_ROOT / "reports" / "r20_subquery_benchmark.md"


def main() -> None:
    configure_logging()
    settings = Settings()
    gold = load_r20_gold_set(REPO_ROOT / "evals" / "datasets" / "retrieval_gold_r20.jsonl")
    answerable = [q for q in gold if q.slice is not R20Slice.NEGATIVE]

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

    classifier = IntentClassifier.from_settings(settings, cache=False, trace=False)

    baseline_ranked: dict[str, list[str]] = {}
    decomposed_ranked: dict[str, list[str]] = {}
    baseline_pack: dict[str, list[str]] = {}
    decomposed_pack: dict[str, list[str]] = {}
    sub_query_counts: dict[str, int] = {}

    for query in answerable:
        classification = classifier.classify(query.question)
        sub_queries = classification.sub_queries
        sub_query_counts[query.query_id] = len(sub_queries)

        baseline = retriever.search(query.question, RANK_DEPTH)
        baseline_ranked[query.query_id] = [r.chunk.node_path for r in baseline]
        b_pack = packer.pack(baseline[:EVIDENCE_POOL], expand=False)
        baseline_pack[query.query_id] = [u.citation for u in b_pack.units]

        if sub_queries:
            per_query = [retriever.search(sq, RANK_DEPTH) for sq in sub_queries]
            merged = _merge_subquery_results(per_query, RANK_DEPTH)
        else:
            merged = list(baseline)
        decomposed_ranked[query.query_id] = [r.chunk.node_path for r in merged]
        d_pack = packer.pack(merged[:EVIDENCE_POOL], expand=False)
        decomposed_pack[query.query_id] = [u.citation for u in d_pack.units]

        logger.info(
            "%s: %d sub_queries (%s)",
            query.query_id,
            len(sub_queries),
            "; ".join(sub_queries) if sub_queries else "none",
        )

    report_baseline_10 = score_coverage_run(gold, baseline_ranked, 10, index)
    report_baseline_20 = score_coverage_run(gold, baseline_ranked, EVIDENCE_POOL, index)
    report_decomposed_10 = score_coverage_run(gold, decomposed_ranked, 10, index)
    report_decomposed_20 = score_coverage_run(gold, decomposed_ranked, EVIDENCE_POOL, index)

    baseline_pack_scores = [
        score_coverage(q, baseline_pack[q.query_id], max(len(baseline_pack[q.query_id]), 1), index)
        for q in answerable
    ]
    decomposed_pack_scores = [
        score_coverage(q, decomposed_pack[q.query_id], max(len(decomposed_pack[q.query_id]), 1), index)
        for q in answerable
    ]

    _write_report(
        answerable=answerable,
        sub_query_counts=sub_query_counts,
        report_baseline_10=report_baseline_10,
        report_baseline_20=report_baseline_20,
        report_decomposed_10=report_decomposed_10,
        report_decomposed_20=report_decomposed_20,
        baseline_pack_scores=baseline_pack_scores,
        decomposed_pack_scores=decomposed_pack_scores,
    )


def _write_report(
    *,
    answerable,
    sub_query_counts,
    report_baseline_10,
    report_baseline_20,
    report_decomposed_10,
    report_decomposed_20,
    baseline_pack_scores,
    decomposed_pack_scores,
):
    lines = ["# R20 Step 20.2 — sub-query decomposition benchmark", ""]
    lines.append(
        "Live re-run against the real Supabase corpus, production retriever "
        "composition and the real `IntentClassifier`, comparing the plain "
        "single-query baseline against sub-query decomposition (ADR-123) on "
        f"the same {len(answerable)} answerable R20 gold questions."
    )
    lines.append("")
    lines.append("## Sub-queries the live classifier actually emitted")
    lines.append("")
    lines.append("| query | sub_queries |")
    lines.append("|---|---|")
    for qid, count in sub_query_counts.items():
        lines.append(f"| {qid} | {count} |")
    lines.append("")
    lines.append("## Recall@k / Coverage@k on the ranked list")
    lines.append("")
    lines.append("| variant | k | mean governing recall | mean precision | mean rule coverage | coverage_complete rate |")
    lines.append("|---|---|---|---|---|---|")
    for label, r in (
        ("baseline", report_baseline_10),
        ("baseline", report_baseline_20),
        ("decomposed", report_decomposed_10),
        ("decomposed", report_decomposed_20),
    ):
        lines.append(
            f"| {label} | {r.k} | {r.mean_governing_recall:.3f} | {r.mean_precision:.3f} | "
            f"{r.mean_rule_coverage:.3f} | {r.coverage_complete_rate:.3f} |"
        )
    lines.append("")
    lines.append("## Coverage on the delivered evidence pack")
    lines.append("")
    for label, scores in (("baseline", baseline_pack_scores), ("decomposed", decomposed_pack_scores)):
        mean_cov = sum(s.rule_coverage for s in scores) / len(scores)
        complete_rate = sum(1 for s in scores if s.coverage_complete) / len(scores)
        lines.append(f"- {label}: mean rule coverage **{mean_cov:.3f}**, coverage-complete rate **{complete_rate:.3f}**")
    lines.append("")
    lines.append("| query | slice | baseline pack coverage | decomposed pack coverage | baseline missing | decomposed missing |")
    lines.append("|---|---|---|---|---|---|")
    by_id = {s.query_id: s for s in decomposed_pack_scores}
    for s in baseline_pack_scores:
        d = by_id[s.query_id]
        lines.append(
            f"| {s.query_id} | {s.slice.value} | {s.rule_coverage:.2f} | {d.rule_coverage:.2f} | "
            f"{', '.join(s.missing_units) or '-'} | {', '.join(d.missing_units) or '-'} |"
        )
    lines.append("")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    print(f"wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
