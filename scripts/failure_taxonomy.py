"""Step 5.7 — why each gold label the delivered evidence misses is missing.

Runs the Step 5.6 composition (shortcut over the reranked hybrid ranking) and
the Step 5.3 pack over every gold question, classifies each missed label with
`taxverity.evals.taxonomy` (ADR-085), and adds the diagnostics the rules do not
read: each leg's rank for the label, its size, and whether Step 5.4's expansion
or a second hop would reach it.

Offline: the question vectors and rerank scores are the stored Step 5.2 and
Step 5.6 files, so this bills nothing. Requires `TAXVERITY_JINA_API_KEY` only
for the embedder identity check against the vector store.

Writes `evals/reports/<corpus_version>/failure_taxonomy.json` and
`reports/failure_taxonomy.md`. Registers `evals/datasets/failure_cohorts_v1.json`
on the first run and never overwrites it: later steps are judged on it.
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from taxverity.chunking.pipeline import read_corpus_version
from taxverity.chunking.stats import estimate_tokens
from taxverity.chunking.store import load_chunks
from taxverity.config import MissingSettingError, Settings
from taxverity.embedding.jina_api import JinaAPIEmbedder
from taxverity.embedding.store import StaleVectorStoreError
from taxverity.evals.gold import GOLD_V2_FILENAME, QuerySlice, load_gold_set
from taxverity.evals.metrics import CreditMode, credits
from taxverity.evals.query_vectors import (
    QUERY_VECTORS_FILENAME,
    CachedQueryRetriever,
    load_query_vectors,
)
from taxverity.evals.rerank import (
    RERANK_SCORES_FILENAME,
    StaleRerankScoresError,
    StoredReranker,
    load_rerank_scores,
)
from taxverity.evals.taxonomy import (
    COHORTS_FILENAME,
    FailureCategory,
    FailureClassifier,
    LabelFailure,
    cohorts_of,
    load_cohorts,
    write_cohorts,
)
from taxverity.observability import configure_logging, get_logger
from taxverity.retrieval.base import ScoredChunk
from taxverity.retrieval.bm25 import BM25Retriever
from taxverity.retrieval.citations import CitationRetriever, ShortcutRetriever
from taxverity.retrieval.dense import DenseRetrievalError, DenseRetriever
from taxverity.retrieval.evidence import EVIDENCE_BUDGET, EVIDENCE_POOL, EvidencePacker
from taxverity.retrieval.fusion import FUSION_DEPTH, FusionRetriever
from taxverity.retrieval.rerank import MODEL_ID, RERANK_DEPTH, RerankRetriever

logger = get_logger(__name__)

REPORT = Path("reports") / "failure_taxonomy.md"
ARTIFACT = "failure_taxonomy.json"
VECTORS_KEY = "jina-api"

# What each category would need, and which step owns it. Reporting only.
REMEDY = {
    FailureCategory.LABEL_GRANULARITY: "re-label pass on the gold set, in its own pass",
    FailureCategory.BUDGET: "giant untrusted roots (Step 1.7 residue)",
    FailureCategory.DANGLING_FORWARD: "one-hop expansion (5.4, rejected as built)",
    FailureCategory.DANGLING_BACKWARD: "an inbound reference index (Phase 6 edge table)",
    FailureCategory.MULTI_PART: "query decomposition (8.4, gated on this count)",
    FailureCategory.VOCABULARY: "statutory-term bridge (5.8)",
}


class Diagnosed(BaseModel):
    model_config = ConfigDict(frozen=True)

    failure: LabelFailure
    slice: QuerySlice
    question: str
    # 1-based rank of the first result crediting the label leniently; None if
    # beyond the depth searched (pool 20, fusion 100, each leg the whole corpus).
    pool_rank: int | None
    fusion_rank: int | None
    bm25_rank: int | None
    dense_rank: int | None
    label_tokens: int
    expansion_recovers: bool
    # Fewest outgoing hops from any delivered unit to the label, up to two.
    hops: int | None


class NegativeDelivery(BaseModel):
    model_config = ConfigDict(frozen=True)

    query_id: str
    units: int
    tokens: int


class TaxonomyReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    corpus_version: str
    chunk_count: int
    gold_count: int
    answerable: int
    slice_sizes: dict[QuerySlice, int]
    labels: int
    budget: int
    pool: int
    diagnosed: tuple[Diagnosed, ...]
    negatives: tuple[NegativeDelivery, ...]


def rank_of(results: Sequence[ScoredChunk], label: str) -> int | None:
    for rank, result in enumerate(results, start=1):
        if credits(result.chunk.node_path, label, CreditMode.LENIENT):
            return rank
    return None


def hops_to(classifier: FailureClassifier, chunks_by_path, packed: Sequence[str], label: str) -> int | None:
    if any(classifier.cites(unit, label) for unit in packed):
        return 1
    for unit in packed:
        for reference in chunks_by_path[unit].outgoing_refs:
            target = classifier.resolve(reference)
            if target is not None and classifier.cites(target.node_path, label):
                return 2
    return None


def diagnose(settings: Settings) -> TaxonomyReport | str:
    store = settings.vectors_dir / VECTORS_KEY
    corpus_version = read_corpus_version(settings.interim_dir / "corpus_manifest.json")
    try:
        chunks, _ = load_chunks(settings.interim_dir, corpus_version=corpus_version)
        embedder = JinaAPIEmbedder.from_settings(settings)
    except (RuntimeError, MissingSettingError) as error:
        return str(error)
    gold = load_gold_set(settings.evals_dir / "datasets" / GOLD_V2_FILENAME)
    questions = [q.question for q in gold]
    with embedder:
        try:
            dense = DenseRetriever.from_store(store, chunks, embedder, corpus_version=corpus_version)
            cached = load_query_vectors(
                store / QUERY_VECTORS_FILENAME, model=embedder.info(), questions=questions
            )
            stored = load_rerank_scores(
                settings.data_dir / "rerank" / RERANK_SCORES_FILENAME,
                model_id=MODEL_ID,
                corpus_version=corpus_version,
                depth=RERANK_DEPTH,
                questions=questions,
            )
        except (
            RuntimeError,
            DenseRetrievalError,
            StaleVectorStoreError,
            StaleRerankScoresError,
            FileNotFoundError,
        ) as error:
            return f"{error} (run scripts/measure_hybrid.py, then scripts/measure_rerank.py)"

    dense_leg = CachedQueryRetriever(dense, cached.vectors)
    bm25 = BM25Retriever(chunks)
    fusion = FusionRetriever([dense_leg, bm25])
    retriever = ShortcutRetriever(
        CitationRetriever(chunks), RerankRetriever(fusion, StoredReranker(stored.scores))
    )
    packer = EvidencePacker(chunks)
    classifier = FailureClassifier(chunks)
    by_path = {chunk.node_path: chunk for chunk in chunks}

    diagnosed: list[Diagnosed] = []
    negatives: list[NegativeDelivery] = []
    for query in gold:
        pool = retriever.search(query.question, EVIDENCE_POOL)
        pack = packer.pack(pool, expand=False)
        packed = [unit.citation for unit in pack.units]
        if query.slice is QuerySlice.NEGATIVE:
            negatives.append(
                NegativeDelivery(query_id=query.query_id, units=len(packed), tokens=pack.tokens)
            )
            continue
        pooled = [result.chunk.node_path for result in pool]
        failures = classifier.classify(query.query_id, query.required, pooled, packed)
        if not failures:
            continue
        expanded = [unit.citation for unit in packer.pack(pool, expand=True).units]
        fused = fusion.search(query.question, FUSION_DEPTH)
        lexical = bm25.search(query.question, len(chunks))
        semantic = dense_leg.search(query.question, len(chunks))
        for failure in failures:
            label = failure.label
            diagnosed.append(
                Diagnosed(
                    failure=failure,
                    slice=query.slice,
                    question=query.question,
                    pool_rank=rank_of(pool, label),
                    fusion_rank=rank_of(fused, label),
                    bm25_rank=rank_of(lexical, label),
                    dense_rank=rank_of(semantic, label),
                    label_tokens=estimate_tokens(by_path[label].text),
                    expansion_recovers=any(
                        credits(unit, label, CreditMode.LENIENT) for unit in expanded
                    ),
                    hops=hops_to(classifier, by_path, packed, label),
                )
            )
    logger.info(
        "classified %d missed labels across %d queries",
        len(diagnosed),
        len({d.failure.query_id for d in diagnosed}),
    )
    return TaxonomyReport(
        corpus_version=corpus_version,
        chunk_count=len(chunks),
        gold_count=len(gold),
        answerable=sum(1 for q in gold if q.slice is not QuerySlice.NEGATIVE),
        slice_sizes=Counter(q.slice for q in gold if q.slice is not QuerySlice.NEGATIVE),
        labels=sum(len(q.required) for q in gold),
        budget=EVIDENCE_BUDGET,
        pool=EVIDENCE_POOL,
        diagnosed=tuple(diagnosed),
        negatives=tuple(negatives),
    )


def rank(value: int | None) -> str:
    return "—" if value is None else str(value)


def named(members: Sequence[Diagnosed]) -> str:
    return (
        ", ".join(
            f"{d.failure.query_id} `{d.failure.label}` ({d.failure.category.value})"
            for d in members
        )
        or "none"
    )


def render(report: TaxonomyReport, cohort_note: str, elapsed: float) -> str:
    diagnosed = report.diagnosed
    failing = {d.failure.query_id for d in diagnosed}
    by_category = {c: [d for d in diagnosed if d.failure.category is c] for c in FailureCategory}
    lines = [
        "# Retrieval failure taxonomy — Income-tax Act, 2025",
        "",
        "Auto-generated by `scripts/failure_taxonomy.py` (Step 5.7, ADR-085).",
        "Every gold label the delivered evidence misses, and why. The composition is",
        "production's as of Step 5.6: the citation shortcut over the reranked hybrid",
        f"ranking, packed by Step 5.3 into {report.budget} proxy tokens from a pool of",
        f"{report.pool}. Cross-reference expansion stays off (ADR-083).",
        "",
        "| Measure | Value |",
        "|---|---|",
        f"| corpus_version | `{report.corpus_version[:16]}…` |",
        f"| Chunks | {report.chunk_count} |",
        f"| Gold queries / answerable | {report.gold_count} / {report.answerable} |",
        f"| Answerable labels | {report.labels} |",
        f"| Queries missing a label | {len(failing)} |",
        f"| Labels missed | {len(diagnosed)} |",
        f"| Run time | {elapsed:.1f}s |",
        "",
        "## How a label is classified",
        "",
        "A label is missed when no delivered unit credits it leniently, so an",
        "ancestor carrying its text counts as found (ADR-060). Each missed label",
        "takes the first category that fits, in this order:",
        "",
        "1. **label granularity**: the pack carries a descendant of the label but",
        "   not the label.",
        "2. **budget**: the label is in the pool, and the pack could not fit it.",
        "3. **dangling, forward**: a delivered unit carrying another label of the",
        "   same question cites the missing one.",
        "4. **dangling, backward**: the missing label cites such a unit, and",
        "   nothing points back.",
        "5. **multi-part**: the question has more than one label, and the missing",
        "   part is unlinked.",
        "6. **vocabulary**: a one-label question whose wording reached nothing",
        "   that leads to the answer.",
        "",
        "\"Genuinely absent\" is not a category here. Every answerable label names",
        "a chunk, so absence arises only for the negatives, reported last.",
        "",
        "## Counts",
        "",
        "| category | labels | queries | paraphrase | crossref | citation | remedy |",
        "|---|---|---|---|---|---|---|",
    ]
    for category, members in by_category.items():
        slices = Counter(d.slice for d in members)
        lines.append(
            f"| {category.value} | {len(members)} | {len({d.failure.query_id for d in members})} "
            f"| {slices[QuerySlice.PARAPHRASE]} | {slices[QuerySlice.CROSSREF]} "
            f"| {slices[QuerySlice.CITATION]} | {REMEDY[category]} |"
        )
    lines += [
        f"| **total** | **{len(diagnosed)}** | **{len(failing)}** | | | | |",
        "",
        "## Reading the counts",
        "",
        "| slice | answerable queries | missing a label |",
        "|---|---|---|",
        *[
            f"| {member.value} | {size} "
            f"| {len({d.failure.query_id for d in diagnosed if d.slice is member})} |"
            for member, size in sorted(report.slice_sizes.items())
        ],
        "",
        "Read the categories per slice, not as shares of one total. The crossref",
        "slice was written around provisions that cite each other (ADR-064), so",
        "dangling dependency is over-represented against real traffic by",
        "construction, and every one of those failures sits in that slice. The",
        "vocabulary cohort Step 5.8 is judged on is small, so its result will be",
        "a count of labels recovered, not a rate.",
        "",
        "## Every missed label",
        "",
        "Ranks are 1-based: pool is the reranked top 20, fusion is read to",
        f"{FUSION_DEPTH}, and each leg over the whole corpus. \"5.4\" is whether",
        "Step 5.4's expansion would have delivered it; \"hops\" the fewest outgoing",
        "references from any delivered unit, up to two.",
        "",
        "| query | slice | label | category | via | pool | fusion | bm25 | dense "
        "| tokens | 5.4 | hops | question |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for category, members in by_category.items():
        for d in members:
            lines.append(
                f"| {d.failure.query_id} | {d.slice.value} | {d.failure.label} | {category.value} "
                f"| {', '.join(d.failure.via) or '—'} | {rank(d.pool_rank)} "
                f"| {rank(d.fusion_rank)} | {rank(d.bm25_rank)} | {rank(d.dense_rank)} "
                f"| {d.label_tokens} | {'yes' if d.expansion_recovers else 'no'} "
                f"| {rank(d.hops)} | {d.question} |"
            )
    forward = by_category[FailureCategory.DANGLING_FORWARD]
    edge = [d for d in diagnosed if d.pool_rank is None and d.fusion_rank is not None]
    second = [d for d in diagnosed if d.hops == 2]
    lines += [
        "",
        "## What each fix could be worth (upper bounds)",
        "",
        f"- **Dangling, forward: {len(forward)}.** Step 5.4's expansion, as built,",
        f"  would deliver {sum(d.expansion_recovers for d in forward)} of them; the rest are",
        "  crowded out by the budget or by a large target. It was rejected because",
        "  its referenced text displaced paraphrase labels (ADR-083).",
        f"- **Within fusion's top {FUSION_DEPTH} but outside the rerank pool: {len(edge)}** of",
        f"  {len(diagnosed)}, whatever their category: {named(edge)}.",
        "  A deeper pool would reach them, and every extra candidate is billed.",
        f"- **Two hops from the delivered evidence: {len(second)}**: {named(second)}.",
        "  Step 8.5's two-hop widening is gated on this count. A label-granularity",
        "  case already has its answer delivered through a descendant.",
        "",
        "## Negatives (reported for Phase 12)",
        "",
    ]
    tokens = [n.tokens for n in report.negatives]
    lines += [
        f"{len(report.negatives)} negative questions. Every one received evidence: "
        f"{min(n.units for n in report.negatives)}–{max(n.units for n in report.negatives)} "
        f"units, {min(tokens)}–{max(tokens)} tokens, median {statistics.median(tokens):.0f}.",
        "Retrieval never returns nothing, so scope is a gate after it, not a",
        "retrieval failure.",
        "",
        "## Frozen cohorts",
        "",
        f"`evals/datasets/{COHORTS_FILENAME}`: {cohort_note}",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    configure_logging()
    settings = Settings()
    started = time.perf_counter()
    report = diagnose(settings)
    if isinstance(report, str):
        print(f"error: {report}", file=sys.stderr)
        return 1

    cohorts = cohorts_of(d.failure for d in report.diagnosed)
    registered = settings.evals_dir / "datasets" / COHORTS_FILENAME
    if not registered.exists():
        write_cohorts(registered, cohorts)
        cohort_note = "registered by this run."
    elif load_cohorts(registered) == cohorts:
        cohort_note = "this run reproduces it exactly."
    else:
        cohort_note = "**this run differs from it; the registered file was not changed.**"
        logger.warning("classification differs from the registered cohorts in %s", registered)

    artifact_dir = settings.evals_dir / "reports" / report.corpus_version
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
    REPORT.write_text(render(report, cohort_note, elapsed), encoding="utf-8", newline="")

    counts = Counter(d.failure.category for d in report.diagnosed)
    for category in FailureCategory:
        print(f"{category.value}: {counts[category]}")
    print(f"cohorts: {cohort_note}")
    print(f"wrote {artifact}")
    print(f"wrote {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
