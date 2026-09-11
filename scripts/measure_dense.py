"""Step 4.6 — dense retrieval against the lexical path, per slice, plus the
long-chunk rule pre-registered at R15 (PLAN 4.6).

Embeds each gold question once through the Jina API (80 requests, a few
thousand tokens), then searches the stored corpus vectors with those same
query vectors for every index variant. Requires `TAXVERITY_JINA_API_KEY`.

Writes `evals/reports/<corpus_version>/dense_measurement.json` and
`reports/dense_measurement.md`.
"""

from __future__ import annotations

import json
import math
import sys
import time
from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from taxverity.chunking.models import Chunk
from taxverity.chunking.pipeline import read_corpus_version
from taxverity.chunking.store import load_chunks
from taxverity.config import MissingSettingError, Settings
from taxverity.embedding.backends import describe
from taxverity.embedding.jina_api import JinaAPIEmbedder
from taxverity.embedding.store import load_vector_store
from taxverity.evals.baseline import PRIMARY_K, QueryOutcome, RetrieverBaseline, measure
from taxverity.evals.comparison import (
    INTRUSION_THRESHOLD,
    WATCH_SET,
    HeadToHead,
    head_to_head,
    intrusions,
    judge_exclusion,
    rule_triggered,
)
from taxverity.evals.gold import GOLD_V2_FILENAME, GoldQuery, QuerySlice, load_gold_set
from taxverity.evals.ladder import Verdict
from taxverity.evals.metrics import CitationIndex, CreditMode
from taxverity.evals.query_vectors import CachedQueryRetriever
from taxverity.observability import configure_logging, get_logger
from taxverity.retrieval.base import Retriever, as_ranked_citations
from taxverity.retrieval.bm25 import K1, B, BM25Retriever
from taxverity.retrieval.citations import CitationRetriever, ShortcutRetriever
from taxverity.retrieval.dense import DenseRetrievalError, DenseRetriever

logger = get_logger(__name__)

REPORT = Path("reports") / "dense_measurement.md"
ARTIFACT = "dense_measurement.json"
VECTORS_KEY = "jina-api"
SEARCH_REPEATS = 3


class Arm(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    bare: RetrieverBaseline
    shortcut: RetrieverBaseline


class DenseReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    corpus_version: str
    chunk_count: int
    gold_count: int
    primary_k: int
    model: str
    fingerprint_cosine: float
    tokens_billed: int
    embed_ms: tuple[float, float]
    search_ms: tuple[float, float]
    bm25: Arm
    dense: Arm
    bare_duel: HeadToHead
    shortcut_duel: HeadToHead
    intrusions: dict[str, int]
    triggered: bool
    excluded: tuple[str, ...]
    reduced: Arm | None
    exclusion: Verdict | None


def percentiles(samples: Sequence[float]) -> tuple[float, float]:
    ordered = sorted(samples)
    return ordered[len(ordered) // 2], ordered[math.ceil(0.95 * len(ordered)) - 1]


def arm(
    name: str,
    retriever: Retriever,
    shortcut: CitationRetriever,
    gold: Sequence[GoldQuery],
    index: CitationIndex,
) -> Arm:
    return Arm(
        name=name,
        bare=measure(name, retriever, gold, index),
        shortcut=measure(
            f"{name}+citation_shortcut",
            ShortcutRetriever(shortcut, retriever),
            gold,
            index,
            ordinal_scores=True,
        ),
    )


def reduce_index(
    store: Path,
    chunks: Sequence[Chunk],
    excluded: set[str],
    dense: DenseRetriever,
    embedder: JinaAPIEmbedder,
    corpus_version: str,
) -> DenseRetriever:
    """The dense index without the excluded roots. BM25 keeps them (PLAN 4.6)."""
    vectors, ids, manifest = load_vector_store(store, corpus_version=corpus_version)
    keep = [c for c in chunks if c.node_path not in excluded]
    keep_ids = {c.chunk_id for c in keep}
    rows = [i for i, chunk_id in enumerate(ids) if chunk_id in keep_ids]
    return DenseRetriever(
        keep, vectors[rows], tuple(ids[i] for i in rows), manifest, embedder
    )


def overall_row(name: str, baseline: RetrieverBaseline) -> str:
    at10 = baseline.reports[PRIMARY_K].overall
    at20 = baseline.reports[20].overall[CreditMode.LENIENT]
    return (
        f"| {name} | {at10[CreditMode.STRICT].recall:.3f} "
        f"| {at10[CreditMode.LENIENT].recall:.3f} | {at10[CreditMode.LENIENT].mrr:.3f} "
        f"| {at10[CreditMode.LENIENT].ndcg:.3f} | {at20.recall:.3f} |"
    )


def spread(values: list[float]) -> str:
    if not values:
        return "—"
    ordered = sorted(values)
    return f"{ordered[0]:.3f} / {ordered[len(ordered) // 2]:.3f} / {ordered[-1]:.3f}"


def slice_union(duel: HeadToHead, outcomes: Sequence[QueryOutcome], member: QuerySlice) -> float:
    ids = [o.query_id for o in outcomes if o.slice is member]
    return sum(duel.union[i] for i in ids) / len(ids)


def render(report: DenseReport, elapsed: float) -> str:
    arms = [report.bm25, report.dense] + ([report.reduced] if report.reduced else [])
    lines = [
        "# Dense retrieval measurement — Income-tax Act, 2025",
        "",
        "Auto-generated by `scripts/measure_dense.py` (Step 4.6). Dense retrieval",
        "against the tuned lexical path, per slice, over the gold v2 set.",
        "",
        "| Measure | Value |",
        "|---|---|",
        f"| corpus_version | `{report.corpus_version[:16]}…` |",
        f"| Chunks | {report.chunk_count} |",
        f"| Gold queries | {report.gold_count} |",
        f"| Primary k | {report.primary_k} |",
        f"| Dense model | `{report.model}` |",
        f"| BM25 | k1={K1}, b={B} (ADR-077) |",
        f"| Fingerprint, lowest probe cosine | {report.fingerprint_cosine:.6f} |",
        f"| Tokens billed (queries + probes) | {report.tokens_billed} |",
        f"| Query embedding round trip p50 / p95 | {report.embed_ms[0]:.0f} / "
        f"{report.embed_ms[1]:.0f} ms |",
        f"| Vector search p50 / p95 | {report.search_ms[0]:.2f} / "
        f"{report.search_ms[1]:.2f} ms |",
        f"| Run time | {elapsed:.1f}s |",
        "",
        "Every score is reported twice (ADR-060): **strict** credits only the",
        "labelled unit, **lenient** also credits an ancestor.",
        "",
        "## Overall",
        "",
        "| retriever | strict R@10 | lenient R@10 | lenient MRR | lenient nDCG@10 "
        "| lenient R@20 |",
        "|---|---|---|---|---|---|",
    ]
    for a in arms:
        lines.append(overall_row(a.name, a.bare))
        lines.append(overall_row(f"{a.name} + shortcut", a.shortcut))

    lines += [
        "",
        "## Per slice, k=10",
        "",
        "Lenient recall / lenient nDCG.",
        "",
        "| slice | " + " | ".join(
            f"{a.name} | {a.name} + shortcut" for a in arms
        ) + " |",
        "|---|" + "---|" * (2 * len(arms)),
    ]
    for member in QuerySlice:
        if member is QuerySlice.NEGATIVE:
            continue
        cells = []
        for a in arms:
            for baseline in (a.bare, a.shortcut):
                scores = baseline.primary.per_slice[member][CreditMode.LENIENT]
                cells.append(f"{scores.recall:.3f} / {scores.ndcg:.3f}")
        lines.append(f"| {member.value} | " + " | ".join(cells) + " |")

    lines += [
        "",
        "## Where each wins",
        "",
        "Query by query, by lenient recall@10. **Union** is the share of labels",
        "credited by either top 10: an upper bound on what fusing the two could",
        "reach, over up to 20 results.",
        "",
        "| comparison | dense wins | bm25 wins | both complete | tied, incomplete "
        "| union recall |",
        "|---|---|---|---|---|---|",
    ]
    for label, duel in (("bare", report.bare_duel), ("with shortcut", report.shortcut_duel)):
        lines.append(
            f"| {label} | {len(duel.a_wins)} | {len(duel.b_wins)} "
            f"| {len(duel.both_complete)} | {len(duel.tied_incomplete)} "
            f"| {duel.union_recall:.3f} |"
        )

    lines += [
        "",
        "Union recall per slice, with shortcut, against each retriever alone:",
        "",
        "| slice | bm25 + shortcut | dense + shortcut | union |",
        "|---|---|---|---|",
    ]
    for member in QuerySlice:
        if member is QuerySlice.NEGATIVE:
            continue
        b = report.bm25.shortcut.primary.per_slice[member][CreditMode.LENIENT].recall
        d = report.dense.shortcut.primary.per_slice[member][CreditMode.LENIENT].recall
        u = slice_union(report.shortcut_duel, report.dense.shortcut.outcomes, member)
        lines.append(f"| {member.value} | {b:.3f} | {d:.3f} | {u:.3f} |")

    bm25_by_id = {o.query_id: o for o in report.bm25.bare.outcomes}
    dense_by_id = {o.query_id: o for o in report.dense.bare.outcomes}
    for title, ids in (
        ("Dense wins (bare)", report.bare_duel.a_wins),
        ("BM25 wins (bare)", report.bare_duel.b_wins),
    ):
        lines += [
            "",
            f"### {title}",
            "",
            "| query | slice | required | dense | bm25 | dense top 3 | question |",
            "|---|---|---|---|---|---|---|",
        ]
        for query_id in ids:
            d, b = dense_by_id[query_id], bm25_by_id[query_id]
            lines.append(
                f"| {query_id} | {d.slice.value} | {', '.join(d.required)} "
                f"| {d.lenient_recall:.2f} | {b.lenient_recall:.2f} "
                f"| {', '.join(d.retrieved[:3])} | {d.question} |"
            )
        if not ids:
            lines.append("| — | | | | | | |")

    lines += [
        "",
        "## Long-chunk rule (pre-registered at R15)",
        "",
        "For each watched chunk, the queries (all 80) where it is in the bare dense",
        "top 10 but is neither a label nor an ancestor of one. If any chunk scores",
        f"**≥ {INTRUSION_THRESHOLD}**, the roots in the watch set are removed from",
        "the dense index only and re-measured; the exclusion is adopted only if",
        "lenient recall@10 and lenient nDCG@10 of bare dense do not fall.",
        "",
        "| chunk | intrusions |",
        "|---|---|",
        *[
            f"| {citation} | {report.intrusions[citation]} |"
            for citation in sorted(
                report.intrusions, key=lambda c: (-report.intrusions[c], WATCH_SET.index(c))
            )
        ],
        "",
    ]
    if not report.triggered:
        lines.append(
            f"No chunk reached {INTRUSION_THRESHOLD}. The rule does not fire and the "
            "dense index keeps every chunk."
        )
    else:
        verdict = report.exclusion
        why = "; ".join(verdict.reasons) or "neither measure fell"
        lines.append(
            f"The rule fired. Excluded from the dense index: "
            f"{', '.join(f'`{c}`' for c in report.excluded)}. "
            f"Verdict: **{'adopted' if verdict.adopted else 'rejected'}** ({why})."
        )

    answerable = [o.top_score for o in report.dense.bare.outcomes if o.top_score is not None]
    negative = [n.top_score for n in report.dense.bare.negatives if n.top_score is not None]
    beaten = sum(1 for s in answerable if negative and s < max(negative))
    lines += [
        "",
        "## Negative separation (bare dense, cosine)",
        "",
        "Top-1 cosine, min / median / max. Comparable only within dense. Reported,",
        "not gated; a threshold is Phase 12's.",
        "",
        "| set | queries | min / median / max |",
        "|---|---|---|",
        f"| answerable | {len(answerable)} | {spread(answerable)} |",
        f"| negative | {len(negative)} | {spread(negative)} |",
        "",
        f"The best negative outscores {beaten} of {len(answerable)} answerable queries.",
        "",
        "| query | top citation | cosine | question |",
        "|---|---|---|---|",
        *[
            f"| {n.query_id} | {n.top_citation or '—'} | {n.top_score:.3f} | {n.question} |"
            for n in report.dense.bare.negatives
        ],
        "",
        "## Bare dense failures at k=10",
        "",
        "| query | slice | required | missed | first hit | question |",
        "|---|---|---|---|---|---|",
        *[
            f"| {o.query_id} | {o.slice.value} | {', '.join(o.required)} "
            f"| {', '.join(o.missed) or '—'} | {o.first_hit_rank or '—'} | {o.question} |"
            for o in report.dense.bare.failures
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
            vectors: dict[str, list[float]] = {}
            embed_samples = []
            for query in gold:
                began = time.perf_counter()
                vectors[query.question] = dense.embed_query(query.question)
                embed_samples.append((time.perf_counter() - began) * 1000)
        except (RuntimeError, DenseRetrievalError) as error:
            print(f"error: {error}", file=sys.stderr)
            return 1
        tokens = embedder.tokens_used
        # The first sample carries the fingerprint check; it is not a query.
        embed_ms = percentiles(embed_samples[1:])
        logger.info("embedded %d questions, %d tokens billed", len(vectors), tokens)

        full = CachedQueryRetriever(dense, vectors)
        search_samples = []
        for query in gold:
            best = math.inf
            for _ in range(SEARCH_REPEATS):
                began = time.perf_counter()
                full.search(query.question, PRIMARY_K)
                best = min(best, (time.perf_counter() - began) * 1000)
            search_samples.append(best)

        bm25_arm = arm("bm25", bm25, shortcut, gold, index)
        dense_arm = arm("dense", full, shortcut, gold, index)

        runs = {
            q.query_id: as_ranked_citations(full.search(q.question, PRIMARY_K))
            for q in gold
        }
        counts = intrusions(runs, gold)
        triggered = rule_triggered(counts)
        watched = set(WATCH_SET)
        excluded = tuple(
            c.node_path for c in chunks if c.parent_id is None and c.node_path in watched
        )
        reduced_arm = verdict = None
        if triggered:
            reduced = reduce_index(
                store, chunks, set(excluded), dense, embedder, corpus_version
            )
            reduced_arm = arm(
                "dense − long roots", CachedQueryRetriever(reduced, vectors), shortcut, gold, index
            )
            verdict = judge_exclusion(reduced_arm.bare.primary, dense_arm.bare.primary)

    report = DenseReport(
        corpus_version=corpus_version,
        chunk_count=len(chunks),
        gold_count=len(gold),
        primary_k=PRIMARY_K,
        model=describe(embedder.info()),
        fingerprint_cosine=dense.fingerprint_cosine,
        tokens_billed=tokens,
        embed_ms=embed_ms,
        search_ms=percentiles(search_samples),
        bm25=bm25_arm,
        dense=dense_arm,
        bare_duel=head_to_head(dense_arm.bare.outcomes, bm25_arm.bare.outcomes),
        shortcut_duel=head_to_head(
            dense_arm.shortcut.outcomes, bm25_arm.shortcut.outcomes
        ),
        intrusions=counts,
        triggered=triggered,
        excluded=excluded,
        reduced=reduced_arm,
        exclusion=verdict,
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

    for a in [bm25_arm, dense_arm] + ([reduced_arm] if reduced_arm else []):
        for baseline in (a.bare, a.shortcut):
            overall = baseline.primary.overall[CreditMode.LENIENT]
            print(
                f"{baseline.name}: lenient recall@{PRIMARY_K} {overall.recall:.3f}, "
                f"nDCG {overall.ndcg:.3f}"
            )
    print(
        f"dense wins {len(report.bare_duel.a_wins)}, bm25 wins "
        f"{len(report.bare_duel.b_wins)}, union recall {report.shortcut_duel.union_recall:.3f}"
    )
    print(f"long-chunk rule: max intrusions {max(counts.values())}, fired={triggered}")
    if verdict is not None:
        print(f"exclusion {'adopted' if verdict.adopted else 'rejected'}: {verdict.reasons}")
    print(f"wrote {artifact}")
    print(f"wrote {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
