"""Step 5.2 — hybrid retrieval (RRF over dense and BM25, Step 5.1) against each
single retriever, per slice, judged by the rule registered in
`taxverity.evals.hybrid` before this script first ran (ADR-081).

Question vectors are read from `data/vectors/jina-api/gold_query_vectors.json`
when it matches the store's model. Otherwise each question is embedded once
through the API (~1,700 tokens) and the file is written. Requires
`TAXVERITY_JINA_API_KEY` either way: the embedder's identity is checked against
the store even when nothing is embedded.

Writes `evals/reports/<corpus_version>/hybrid_measurement.json` and
`reports/hybrid_measurement.md`.
"""

from __future__ import annotations

import json
import math
import sys
import time
from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from taxverity.chunking.pipeline import read_corpus_version
from taxverity.chunking.store import load_chunks
from taxverity.config import MissingSettingError, Settings
from taxverity.embedding.backends import describe
from taxverity.embedding.jina_api import JinaAPIEmbedder
from taxverity.embedding.store import StaleVectorStoreError
from taxverity.evals.baseline import PRIMARY_K, RetrieverBaseline, measure
from taxverity.evals.comparison import HeadToHead, head_to_head
from taxverity.evals.gold import GOLD_V2_FILENAME, GoldQuery, QuerySlice, load_gold_set
from taxverity.evals.hybrid import judge_hybrid
from taxverity.evals.ladder import Verdict
from taxverity.evals.metrics import CitationIndex, CreditMode
from taxverity.evals.query_vectors import (
    QUERY_VECTORS_FILENAME,
    CachedQueryRetriever,
    QueryVectors,
    load_query_vectors,
    write_query_vectors,
)
from taxverity.observability import configure_logging, get_logger
from taxverity.retrieval.bm25 import BM25Retriever
from taxverity.retrieval.citations import CitationRetriever, ShortcutRetriever
from taxverity.retrieval.dense import DenseRetrievalError, DenseRetriever
from taxverity.retrieval.fusion import FUSION_DEPTH, RRF_K, FusionRetriever

logger = get_logger(__name__)

REPORT = Path("reports") / "hybrid_measurement.md"
ARTIFACT = "hybrid_measurement.json"
VECTORS_KEY = "jina-api"
SEARCH_REPEATS = 3
INCUMBENTS = ("bm25+citation_shortcut", "dense+citation_shortcut")
CANDIDATE = "hybrid+citation_shortcut"


class HybridReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    corpus_version: str
    chunk_count: int
    gold_count: int
    primary_k: int
    rrf_k: int
    depth: int
    model: str
    vectors_source: str
    fingerprint_cosine: float
    tokens_billed: int
    fused_search_ms: tuple[float, float]
    baselines: tuple[RetrieverBaseline, ...]
    verdict: Verdict
    vs_dense: HeadToHead
    vs_bm25: HeadToHead

    def baseline(self, name: str) -> RetrieverBaseline:
        return next(b for b in self.baselines if b.name == name)


def percentiles(samples: Sequence[float]) -> tuple[float, float]:
    ordered = sorted(samples)
    return ordered[len(ordered) // 2], ordered[math.ceil(0.95 * len(ordered)) - 1]


def question_vectors(
    dense: DenseRetriever, embedder: JinaAPIEmbedder, path: Path, gold: Sequence[GoldQuery]
) -> tuple[QueryVectors, str]:
    questions = [q.question for q in gold]
    if path.exists():
        try:
            return load_query_vectors(path, model=embedder.info(), questions=questions), "cache"
        except StaleVectorStoreError as error:
            logger.warning("re-embedding the gold questions: %s", error)
    vectors = {question: tuple(dense.embed_query(question)) for question in questions}
    cached = QueryVectors(
        model=embedder.info(),
        fingerprint_cosine=dense.fingerprint_cosine,
        vectors=vectors,
    )
    write_query_vectors(path, cached)
    logger.info("embedded %d gold questions, wrote %s", len(vectors), path)
    return cached, "api"


def overall_row(baseline: RetrieverBaseline) -> str:
    at10 = baseline.reports[PRIMARY_K].overall
    at20 = baseline.reports[20].overall[CreditMode.LENIENT]
    return (
        f"| {baseline.name} | {at10[CreditMode.STRICT].recall:.3f} "
        f"| {at10[CreditMode.LENIENT].recall:.3f} | {at10[CreditMode.LENIENT].mrr:.3f} "
        f"| {at10[CreditMode.LENIENT].ndcg:.3f} | {at20.recall:.3f} |"
    )


def duel_rows(title: str, ids: Sequence[str], mine: RetrieverBaseline, theirs: RetrieverBaseline) -> list[str]:
    mine_by = {o.query_id: o for o in mine.outcomes}
    theirs_by = {o.query_id: o for o in theirs.outcomes}
    lines = [
        "",
        f"### {title}",
        "",
        f"| query | slice | required | {mine.name} | {theirs.name} | hybrid top 3 "
        f"| other top 3 | question |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for query_id in ids:
        m, t = mine_by[query_id], theirs_by[query_id]
        lines.append(
            f"| {query_id} | {m.slice.value} | {', '.join(m.required)} "
            f"| {m.lenient_recall:.2f} | {t.lenient_recall:.2f} "
            f"| {', '.join(m.retrieved[:3])} | {', '.join(t.retrieved[:3])} | {m.question} |"
        )
    if not ids:
        lines.append("| — | | | | | | | |")
    return lines


def render(report: HybridReport, elapsed: float) -> str:
    shortcut_arms = [report.baseline(n) for n in (*INCUMBENTS, CANDIDATE)]
    verdict = report.verdict
    lines = [
        "# Hybrid retrieval measurement — Income-tax Act, 2025",
        "",
        "Auto-generated by `scripts/measure_hybrid.py` (Step 5.2). RRF over dense and",
        "BM25 (Step 5.1, ADR-080) against each single retriever, over the gold v2 set.",
        "",
        "| Measure | Value |",
        "|---|---|",
        f"| corpus_version | `{report.corpus_version[:16]}…` |",
        f"| Chunks | {report.chunk_count} |",
        f"| Gold queries | {report.gold_count} |",
        f"| Primary k | {report.primary_k} |",
        f"| RRF constant / depth per leg | {report.rrf_k} / {report.depth} |",
        f"| Dense model | `{report.model}` |",
        f"| Question vectors | {report.vectors_source} |",
        f"| Fingerprint when embedded, lowest probe cosine | "
        f"{report.fingerprint_cosine:.6f} |",
        f"| Tokens billed this run | {report.tokens_billed} |",
        f"| Hybrid + shortcut search p50 / p95 (excl. embedding) | "
        f"{report.fused_search_ms[0]:.2f} / {report.fused_search_ms[1]:.2f} ms |",
        f"| Run time | {elapsed:.1f}s |",
        "",
        "## The rule (registered before the run, ADR-081)",
        "",
        "Hybrid + shortcut is adopted only if, against **each** of bm25 + shortcut",
        "and dense + shortcut, at k=10: lenient recall and lenient nDCG both hold,",
        "at least one rises, and **no slice's lenient recall falls**.",
        "",
        f"**Verdict: {'ADOPTED' if verdict.adopted else 'REJECTED'}.**",
        "",
        *([f"- {reason}" for reason in verdict.reasons] or ["- every clause held"]),
        "",
        "## Overall",
        "",
        "| retriever | strict R@10 | lenient R@10 | lenient MRR | lenient nDCG@10 "
        "| lenient R@20 |",
        "|---|---|---|---|---|---|",
        *[overall_row(b) for b in report.baselines],
        "",
        "## Per slice, k=10, with the shortcut",
        "",
        "Lenient recall / lenient nDCG.",
        "",
        "| slice | " + " | ".join(a.name for a in shortcut_arms) + " |",
        "|---|" + "---|" * len(shortcut_arms),
    ]
    for member in QuerySlice:
        if member is QuerySlice.NEGATIVE:
            continue
        cells = []
        for a in shortcut_arms:
            s = a.primary.per_slice[member][CreditMode.LENIENT]
            cells.append(f"{s.recall:.3f} / {s.ndcg:.3f}")
        lines.append(f"| {member.value} | " + " | ".join(cells) + " |")

    hybrid = report.baseline(CANDIDATE)
    lines += [
        "",
        "## Query by query, with the shortcut",
        "",
        "By lenient recall@10. **Union** is the share of labels either top 10",
        "credits: an upper bound over up to 20 results.",
        "",
        "| comparison | hybrid wins | other wins | both complete | tied, incomplete "
        "| union recall |",
        "|---|---|---|---|---|---|",
    ]
    for label, duel in (("vs dense", report.vs_dense), ("vs bm25", report.vs_bm25)):
        lines.append(
            f"| {label} | {len(duel.a_wins)} | {len(duel.b_wins)} "
            f"| {len(duel.both_complete)} | {len(duel.tied_incomplete)} "
            f"| {duel.union_recall:.3f} |"
        )
    dense = report.baseline(INCUMBENTS[1])
    lines += duel_rows("Hybrid wins over dense", report.vs_dense.a_wins, hybrid, dense)
    lines += duel_rows("Dense wins over hybrid", report.vs_dense.b_wins, hybrid, dense)
    lines += [
        "",
        "## Hybrid + shortcut failures at k=10",
        "",
        "| query | slice | required | missed | first hit | question |",
        "|---|---|---|---|---|---|",
        *[
            f"| {o.query_id} | {o.slice.value} | {', '.join(o.required)} "
            f"| {', '.join(o.missed) or '—'} | {o.first_hit_rank or '—'} | {o.question} |"
            for o in hybrid.failures
        ],
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    configure_logging()
    settings = Settings()
    started = time.perf_counter()
    store = settings.vectors_dir / VECTORS_KEY

    corpus_version = read_corpus_version(settings.interim_dir / "corpus_manifest.json")
    try:
        chunks, _ = load_chunks(settings.interim_dir, corpus_version=corpus_version)
        embedder = JinaAPIEmbedder.from_settings(settings)
    except (RuntimeError, MissingSettingError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    gold = load_gold_set(settings.evals_dir / "datasets" / GOLD_V2_FILENAME)
    if len({q.question for q in gold}) != len(gold):
        print("error: gold questions must be unique to key their vectors", file=sys.stderr)
        return 1
    index = CitationIndex(chunks)
    bm25 = BM25Retriever(chunks)
    shortcut = CitationRetriever(chunks)

    with embedder:
        try:
            dense = DenseRetriever.from_store(
                store, chunks, embedder, corpus_version=corpus_version
            )
            cached, source = question_vectors(
                dense, embedder, store / QUERY_VECTORS_FILENAME, gold
            )
        except (RuntimeError, DenseRetrievalError) as error:
            print(f"error: {error}", file=sys.stderr)
            return 1
        tokens = embedder.tokens_used

    dense_leg = CachedQueryRetriever(dense, cached.vectors)
    # Dense first: input order breaks exact RRF ties, and dense is the stronger
    # single retriever (ADR-079, ADR-080).
    hybrid = FusionRetriever([dense_leg, bm25])

    baselines = []
    for name, retriever in (("bm25", bm25), ("dense", dense_leg), ("hybrid", hybrid)):
        # An RRF score is a function of rank alone; it measures no similarity.
        baselines.append(measure(name, retriever, gold, index, ordinal_scores=name == "hybrid"))
        baselines.append(
            measure(
                f"{name}+citation_shortcut",
                ShortcutRetriever(shortcut, retriever),
                gold,
                index,
                ordinal_scores=True,
            )
        )
    by_name = {b.name: b for b in baselines}

    fused = ShortcutRetriever(shortcut, hybrid)
    samples = []
    for query in gold:
        best = math.inf
        for _ in range(SEARCH_REPEATS):
            began = time.perf_counter()
            fused.search(query.question, PRIMARY_K)
            best = min(best, (time.perf_counter() - began) * 1000)
        samples.append(best)

    candidate = by_name[CANDIDATE]
    report = HybridReport(
        corpus_version=corpus_version,
        chunk_count=len(chunks),
        gold_count=len(gold),
        primary_k=PRIMARY_K,
        rrf_k=RRF_K,
        depth=FUSION_DEPTH,
        model=describe(embedder.info()),
        vectors_source=source,
        fingerprint_cosine=cached.fingerprint_cosine,
        tokens_billed=tokens,
        fused_search_ms=percentiles(samples),
        baselines=tuple(baselines),
        verdict=judge_hybrid(
            candidate.primary, {name: by_name[name].primary for name in INCUMBENTS}
        ),
        vs_dense=head_to_head(candidate.outcomes, by_name[INCUMBENTS[1]].outcomes),
        vs_bm25=head_to_head(candidate.outcomes, by_name[INCUMBENTS[0]].outcomes),
    )

    artifact_dir = settings.evals_dir / "reports" / corpus_version
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact = artifact_dir / ARTIFACT
    with artifact.open("w", encoding="utf-8", newline="") as handle:
        handle.write(
            json.dumps(
                report.model_dump(mode="json"),
                sort_keys=True,
                ensure_ascii=False,
                separators=(",", ":"),
            )
        )
        handle.write("\n")

    elapsed = time.perf_counter() - started
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(render(report, elapsed), encoding="utf-8", newline="")

    for baseline in baselines:
        overall = baseline.primary.overall[CreditMode.LENIENT]
        print(
            f"{baseline.name}: lenient recall@{PRIMARY_K} {overall.recall:.3f}, "
            f"nDCG {overall.ndcg:.3f}"
        )
    verdict = report.verdict
    print(f"verdict: {'ADOPTED' if verdict.adopted else 'REJECTED'} {list(verdict.reasons)}")
    print(f"question vectors: {source}, {tokens} tokens billed")
    print(f"wrote {artifact}")
    print(f"wrote {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
