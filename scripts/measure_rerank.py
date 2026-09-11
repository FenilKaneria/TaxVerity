"""Step 5.6 — the reranker measured against the un-reranked hybrid + shortcut
ranking (ADR-081), judged by the rule registered in `taxverity.evals.rerank`
before this script first ran (ADR-084).

The first run bills the Jina rerank API once per gold question, paced under the
free tier's 100k tokens a minute, and stores the scores together with each
request's latency in `data/rerank/gold_rerank_scores.json`. Every later run
reads that file and bills nothing. Requires `TAXVERITY_JINA_API_KEY` either way:
the embedder's identity is checked against the vector store.

Writes `evals/reports/<corpus_version>/rerank.json` and
`reports/rerank_measurement.md`.
"""

from __future__ import annotations

import json
import math
import sys
import time
from collections import deque
from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from taxverity.chunking.models import Chunk
from taxverity.chunking.pipeline import read_corpus_version
from taxverity.chunking.stats import estimate_tokens
from taxverity.chunking.store import load_chunks
from taxverity.config import MissingSettingError, Settings
from taxverity.embedding.jina_api import JinaAPIEmbedder
from taxverity.embedding.store import StaleVectorStoreError
from taxverity.evals.baseline import RetrieverBaseline, measure
from taxverity.evals.delivery import judge_expansion
from taxverity.evals.gold import GOLD_V2_FILENAME, GoldQuery, QuerySlice, load_gold_set
from taxverity.evals.ladder import Verdict
from taxverity.evals.metrics import CitationIndex, CreditMode, RunReport, score_run
from taxverity.evals.query_vectors import (
    QUERY_VECTORS_FILENAME,
    CachedQueryRetriever,
    load_query_vectors,
)
from taxverity.evals.rerank import (
    GUARD_K,
    RERANK_P95_BUDGET_MS,
    RERANK_SCORES_FILENAME,
    RerankScores,
    StaleRerankScoresError,
    StoredReranker,
    judge_rerank,
    load_rerank_scores,
    write_rerank_scores,
)
from taxverity.observability import configure_logging, get_logger
from taxverity.retrieval.base import Retriever
from taxverity.retrieval.bm25 import BM25Retriever
from taxverity.retrieval.citations import CitationRetriever, ShortcutRetriever
from taxverity.retrieval.dense import DenseRetrievalError, DenseRetriever
from taxverity.retrieval.evidence import EVIDENCE_POOL, EvidencePacker
from taxverity.retrieval.fusion import FusionRetriever
from taxverity.retrieval.rerank import (
    MODEL_ID,
    RERANK_DEPTH,
    JinaReranker,
    RerankRetriever,
)

logger = get_logger(__name__)

REPORT = Path("reports") / "rerank_measurement.md"
ARTIFACT = "rerank.json"
VECTORS_KEY = "jina-api"

# The free tier allows 100k tokens a minute; stay under it with margin. The
# proxy runs 1.145x short of Jina's count (Step 4.4), rounded up for safety.
TOKENS_PER_MINUTE = 80_000
PROXY_FACTOR = 1.2
# Wide enough to see the true latency distribution: the guard is on p95, and a
# query-path timeout would censor exactly the tail it measures.
MEASUREMENT_TIMEOUT = 60.0
MEASUREMENT_ATTEMPTS = 6


class Paced:
    """Keeps a gold run under the tokens-a-minute limit by waiting, not failing."""

    def __init__(self, inner: JinaReranker) -> None:
        self._inner = inner
        self._window: deque[tuple[float, int]] = deque()

    def score(self, query: str, chunks: Sequence[Chunk]) -> dict[str, float]:
        estimate = int(PROXY_FACTOR * sum(estimate_tokens(c.embed_text()) for c in chunks))
        while True:
            now = time.monotonic()
            while self._window and now - self._window[0][0] > 60:
                self._window.popleft()
            used = sum(tokens for _, tokens in self._window)
            if not self._window or used + estimate <= TOKENS_PER_MINUTE:
                break
            time.sleep(60 - (now - self._window[0][0]) + 0.5)
        before = self._inner.tokens_used
        result = self._inner.score(query, chunks)
        self._window.append((time.monotonic(), self._inner.tokens_used - before))
        return result


class RerankReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    corpus_version: str
    chunk_count: int
    gold_count: int
    depth: int
    fusion: dict[int, RunReport]
    reranked: dict[int, RunReport]
    verdict: Verdict
    latency_ms: tuple[float, float, float]
    tokens_billed: int
    requests: int
    packed_fusion: RunReport
    packed_reranked: RunReport
    expansion_after_rerank: Verdict
    expanded_reranked: RunReport
    top_score_answerable: tuple[float, float]
    top_score_negative: tuple[float, float]
    negatives_above_weakest_answerable: int


def quantiles(samples: Sequence[float]) -> tuple[float, float, float]:
    ordered = sorted(samples)
    rank = lambda q: ordered[math.ceil(q * len(ordered)) - 1]  # noqa: E731
    return ordered[len(ordered) // 2], rank(0.95), ordered[-1]


def score_live(
    settings: Settings, fusion: Retriever, gold: Sequence[GoldQuery], corpus_version: str
) -> RerankScores:
    live = JinaReranker.from_settings(
        settings, timeout=MEASUREMENT_TIMEOUT, max_attempts=MEASUREMENT_ATTEMPTS
    )
    paced = Paced(live)
    scores: dict[str, dict[str, float]] = {}
    with live:
        for position, query in enumerate(gold, start=1):
            pool = [result.chunk for result in fusion.search(query.question, RERANK_DEPTH)]
            scores[query.question] = paced.score(query.question, pool)
            logger.info(
                "scored %d/%d, %d tokens billed so far", position, len(gold), live.tokens_used
            )
    return RerankScores(
        model_id=MODEL_ID,
        corpus_version=corpus_version,
        depth=RERANK_DEPTH,
        scores=scores,
        latencies_ms=tuple(live.latencies_ms),
        tokens_billed=live.tokens_used,
    )


def packed(retriever: Retriever, gold, index, packer, *, expand: bool) -> RunReport:
    runs = {
        q.query_id: [
            unit.citation
            for unit in packer.pack(retriever.search(q.question, EVIDENCE_POOL), expand=expand).units
        ]
        for q in gold
    }
    return score_run(gold, runs, EVIDENCE_POOL, index)


def render(report: RerankReport, base: RetrieverBaseline, after: RetrieverBaseline, elapsed: float) -> str:
    verdict = report.verdict
    lenient, strict = CreditMode.LENIENT, CreditMode.STRICT
    lines = [
        "# Reranker — Income-tax Act, 2025",
        "",
        "Auto-generated by `scripts/measure_rerank.py` (Step 5.6).",
        f"`{MODEL_ID}` via Jina's hosted API reorders the fused pool's top {report.depth},",
        "inside the citation shortcut (ADR-084). Compared against the same",
        "hybrid + shortcut ranking un-reranked (ADR-081).",
        "",
        "| Measure | Value |",
        "|---|---|",
        f"| corpus_version | `{report.corpus_version[:16]}…` |",
        f"| Chunks | {report.chunk_count} |",
        f"| Gold queries | {report.gold_count} |",
        f"| Rerank depth | top {report.depth} |",
        f"| Requests / tokens billed (first run) | {report.requests} / {report.tokens_billed} |",
        f"| Rerank latency p50 / p95 / max | {report.latency_ms[0]:.0f} / "
        f"{report.latency_ms[1]:.0f} / {report.latency_ms[2]:.0f} ms |",
        f"| Run time | {elapsed:.1f}s |",
        "",
        "## The rule (registered before the first paid run, ADR-084)",
        "",
        f"Lenient nDCG@{GUARD_K} must not fall against the un-reranked order, and the",
        f"rerank call's p95 must be within {RERANK_P95_BUDGET_MS:.0f} ms. No rise is required.",
        "",
        f"**Verdict: {'ADOPTED' if verdict.adopted else 'REJECTED'}.**",
        "",
        *([f"- {reason}" for reason in verdict.reasons] or ["- every clause held"]),
        "",
        "## Ranking quality",
        "",
        "| k | measure | fusion | reranked |",
        "|---|---|---|---|",
    ]
    for k in sorted(report.fusion):
        for label, mode, field in (
            ("lenient nDCG", lenient, "ndcg"),
            ("strict nDCG", strict, "ndcg"),
            ("lenient recall", lenient, "recall"),
            ("strict recall", strict, "recall"),
            ("lenient MRR", lenient, "mrr"),
        ):
            a = getattr(report.fusion[k].overall[mode], field)
            b = getattr(report.reranked[k].overall[mode], field)
            lines.append(f"| {k} | {label} | {a:.3f} | {b:.3f} |")
    lines += [
        "",
        f"## Per slice at k = {GUARD_K}, lenient nDCG / recall",
        "",
        "| slice | fusion | reranked |",
        "|---|---|---|",
    ]
    for member, scores in report.fusion[GUARD_K].per_slice.items():
        other = report.reranked[GUARD_K].per_slice[member][lenient]
        lines.append(
            f"| {member.value} | {scores[lenient].ndcg:.3f} / {scores[lenient].recall:.3f} "
            f"| {other.ndcg:.3f} / {other.recall:.3f} |"
        )
    before = {s.query_id: s.lenient.ndcg for s in report.fusion[GUARD_K].scored}
    now = {s.query_id: s.lenient.ndcg for s in report.reranked[GUARD_K].scored}
    moved = sorted(
        (q for q in after.outcomes if abs(now[q.query_id] - before[q.query_id]) > 1e-9),
        key=lambda q: now[q.query_id] - before[q.query_id],
    )
    lines += [
        "",
        f"## Queries whose lenient nDCG@{GUARD_K} moved",
        "",
        "| query | slice | fusion | reranked | reranked top 5 | question |",
        "|---|---|---|---|---|---|",
        *[
            f"| {q.query_id} | {q.slice.value} | {before[q.query_id]:.2f} "
            f"| {now[q.query_id]:.2f} | {', '.join(q.retrieved[:5])} | {q.question} |"
            for q in moved
        ],
        *([] if moved else ["| — | | | | | |"]),
        "",
        "## Evidence delivery on the reranked pool (reported, not gated)",
        "",
        "Lenient coverage of the Step 5.3 pack, and Step 5.4's expansion re-checked",
        "against its own registered rule now that the pool's order has changed (ADR-083).",
        "",
        "| slice | 5.3 pack, fusion | 5.3 pack, reranked | 5.4 expanded, reranked |",
        "|---|---|---|---|",
        f"| overall | {report.packed_fusion.overall[lenient].recall:.3f} "
        f"| {report.packed_reranked.overall[lenient].recall:.3f} "
        f"| {report.expanded_reranked.overall[lenient].recall:.3f} |",
    ]
    for member in report.packed_fusion.per_slice:
        lines.append(
            f"| {member.value} | {report.packed_fusion.per_slice[member][lenient].recall:.3f} "
            f"| {report.packed_reranked.per_slice[member][lenient].recall:.3f} "
            f"| {report.expanded_reranked.per_slice[member][lenient].recall:.3f} |"
        )
    expansion = report.expansion_after_rerank
    lines += [
        "",
        f"Step 5.4's rule on the reranked pool: **{'would adopt' if expansion.adopted else 'still rejects'}**"
        + (f" ({'; '.join(expansion.reasons)})." if expansion.reasons else "."),
        "",
        "## Does the relevance score separate negatives? (reported for Phase 12)",
        "",
        "Top relevance score in each question's pool. Unlike fusion scores these are",
        "on one scale across questions, from one model.",
        "",
        "| group | min | max |",
        "|---|---|---|",
        f"| answerable | {report.top_score_answerable[0]:.3f} | {report.top_score_answerable[1]:.3f} |",
        f"| negative | {report.top_score_negative[0]:.3f} | {report.top_score_negative[1]:.3f} |",
        "",
        f"Negatives scoring above the weakest answerable question: "
        f"{report.negatives_above_weakest_answerable}.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    configure_logging()
    settings = Settings()
    started = time.perf_counter()
    store = settings.vectors_dir / VECTORS_KEY
    scores_path = settings.data_dir / "rerank" / RERANK_SCORES_FILENAME

    corpus_version = read_corpus_version(settings.interim_dir / "corpus_manifest.json")
    try:
        chunks, _ = load_chunks(settings.interim_dir, corpus_version=corpus_version)
        embedder = JinaAPIEmbedder.from_settings(settings)
    except (RuntimeError, MissingSettingError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    gold = load_gold_set(settings.evals_dir / "datasets" / GOLD_V2_FILENAME)
    questions = [q.question for q in gold]
    with embedder:
        try:
            dense = DenseRetriever.from_store(store, chunks, embedder, corpus_version=corpus_version)
            cached = load_query_vectors(
                store / QUERY_VECTORS_FILENAME, model=embedder.info(), questions=questions
            )
        except (RuntimeError, DenseRetrievalError, StaleVectorStoreError) as error:
            print(f"error: {error} (run scripts/measure_hybrid.py first)", file=sys.stderr)
            return 1

    fusion = FusionRetriever([CachedQueryRetriever(dense, cached.vectors), BM25Retriever(chunks)])
    try:
        stored = load_rerank_scores(
            scores_path,
            model_id=MODEL_ID,
            corpus_version=corpus_version,
            depth=RERANK_DEPTH,
            questions=questions,
        )
        logger.info("read stored rerank scores from %s; billing nothing", scores_path)
    except (FileNotFoundError, StaleRerankScoresError) as reason:
        logger.info("scoring the gold pool through the API (%s)", reason)
        stored = score_live(settings, fusion, gold, corpus_version)
        write_rerank_scores(scores_path, stored)

    index = CitationIndex(chunks)
    shortcut = CitationRetriever(chunks)
    base = ShortcutRetriever(shortcut, fusion)
    reranked = ShortcutRetriever(shortcut, RerankRetriever(fusion, StoredReranker(stored.scores)))
    before = measure("hybrid+citation_shortcut", base, gold, index, ordinal_scores=True)
    after = measure("reranked", reranked, gold, index, ordinal_scores=True)
    latency = quantiles(stored.latencies_ms)

    packer = EvidencePacker(chunks)
    packed_reranked = packed(reranked, gold, index, packer, expand=False)
    expanded_reranked = packed(reranked, gold, index, packer, expand=True)

    tops = {question: max(pool.values()) for question, pool in stored.scores.items() if pool}
    answerable = [tops[q.question] for q in gold if q.slice is not QuerySlice.NEGATIVE]
    negative = [tops[q.question] for q in gold if q.slice is QuerySlice.NEGATIVE]

    report = RerankReport(
        corpus_version=corpus_version,
        chunk_count=len(chunks),
        gold_count=len(gold),
        depth=RERANK_DEPTH,
        fusion=before.reports,
        reranked=after.reports,
        verdict=judge_rerank(after.reports[GUARD_K], before.reports[GUARD_K], latency[1]),
        latency_ms=latency,
        tokens_billed=stored.tokens_billed,
        requests=len(stored.latencies_ms),
        packed_fusion=packed(base, gold, index, packer, expand=False),
        packed_reranked=packed_reranked,
        expansion_after_rerank=judge_expansion(expanded_reranked, packed_reranked),
        expanded_reranked=expanded_reranked,
        top_score_answerable=(min(answerable), max(answerable)),
        top_score_negative=(min(negative), max(negative)),
        negatives_above_weakest_answerable=sum(1 for s in negative if s > min(answerable)),
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
    REPORT.write_text(render(report, before, after, elapsed), encoding="utf-8", newline="")

    for label, run in (("fusion", before), ("reranked", after)):
        overall = run.reports[GUARD_K].overall[CreditMode.LENIENT]
        print(f"{label}: lenient nDCG@{GUARD_K} {overall.ndcg:.3f}, recall@{GUARD_K} {overall.recall:.3f}")
    print(f"rerank latency p50/p95/max: {latency[0]:.0f}/{latency[1]:.0f}/{latency[2]:.0f} ms")
    verdict = report.verdict
    print(f"verdict: {'ADOPTED' if verdict.adopted else 'REJECTED'} {list(verdict.reasons)}")
    print(f"wrote {artifact}")
    print(f"wrote {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
