"""Step 5.8 — the statutory-term bridge measured against production's Step 5.6
composition, judged by the rule in `taxverity.evals.bridge`, registered before
this script first ran (ADR-086). The definitions pull is judged separately,
against the bridged pack.

A rewritten question needs its own vector and rerank scores. The first run
embeds and reranks only the rewritten questions and stores them in
`data/vectors/jina-api/bridge_query_vectors.json` and
`data/rerank/bridge_rerank_scores.json`, keyed by the rewritten text. Every
later run reads those and bills nothing. Requires `TAXVERITY_JINA_API_KEY`.

Writes `evals/reports/<corpus_version>/term_bridge.json` and
`reports/term_bridge.md`.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from measure_rerank import MEASUREMENT_ATTEMPTS, MEASUREMENT_TIMEOUT, Paced
from pydantic import BaseModel, ConfigDict

from taxverity.chunking.pipeline import read_corpus_version
from taxverity.chunking.store import load_chunks
from taxverity.config import MissingSettingError, Settings
from taxverity.embedding.jina_api import JinaAPIEmbedder
from taxverity.embedding.store import StaleVectorStoreError
from taxverity.evals.bridge import (
    BRIDGE_SCORES_FILENAME,
    BRIDGE_VECTORS_FILENAME,
    delivered,
    gained,
    judge_bridge,
    judge_definitions,
    lost,
)
from taxverity.evals.gold import GOLD_V2_FILENAME, QuerySlice, load_gold_set
from taxverity.evals.ladder import Verdict
from taxverity.evals.metrics import CreditMode, credits
from taxverity.evals.query_vectors import (
    QUERY_VECTORS_FILENAME,
    CachedQueryRetriever,
    QueryVectors,
    load_query_vectors,
    write_query_vectors,
)
from taxverity.evals.rerank import (
    RERANK_SCORES_FILENAME,
    RerankScores,
    StaleRerankScoresError,
    StoredReranker,
    load_rerank_scores,
    write_rerank_scores,
)
from taxverity.evals.taxonomy import (
    COHORTS_FILENAME,
    FailureCategory,
    FailureClassifier,
    load_cohorts,
)
from taxverity.observability import configure_logging, get_logger
from taxverity.retrieval.base import Retriever
from taxverity.retrieval.bm25 import BM25Retriever
from taxverity.retrieval.bridge import (
    BRIDGE_MAP,
    BridgedRetriever,
    TermBridge,
    load_bridge_map,
)
from taxverity.retrieval.citations import CitationRetriever, ShortcutRetriever
from taxverity.retrieval.dense import DenseRetrievalError, DenseRetriever
from taxverity.retrieval.evidence import EVIDENCE_POOL, EvidencePacker, EvidenceRole
from taxverity.retrieval.fusion import FusionRetriever
from taxverity.retrieval.rerank import (
    MODEL_ID,
    RERANK_DEPTH,
    JinaReranker,
    RerankRetriever,
)

logger = get_logger(__name__)

REPORT = Path("reports") / "term_bridge.md"
ARTIFACT = "term_bridge.json"
VECTORS_KEY = "jina-api"


class Rewritten(BaseModel):
    model_config = ConfigDict(frozen=True)

    query_id: str
    slice: QuerySlice
    question: str
    labels: tuple[str, ...]
    added: tuple[str, ...]
    # Labels the pack credits, before and after; empty for a negative.
    before: tuple[str, ...]
    after: tuple[str, ...]
    # Units inside a label's subtree that the pack carried before and no longer
    # does. Lenient credit never counts a descendant (ADR-060), so a question
    # labelled too coarsely loses its real answer invisibly to the rule.
    dropped: tuple[str, ...]


class DefinitionUse(BaseModel):
    model_config = ConfigDict(frozen=True)

    query_id: str
    delivered: tuple[str, ...]
    tokens: int


class BridgeReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    corpus_version: str
    chunk_count: int
    gold_count: int
    map_sha256: str
    statutory_terms: int
    lay_phrases: int
    pullable: int
    tokens_embedded: int
    tokens_reranked: int
    rewritten: tuple[Rewritten, ...]
    cohort: tuple[tuple[str, str], ...]
    recovered: tuple[tuple[str, str], ...]
    gained: tuple[tuple[str, str], ...]
    lost: tuple[tuple[str, str], ...]
    verdict: Verdict
    categories_before: dict[FailureCategory, int]
    categories_after: dict[FailureCategory, int]
    definitions: tuple[DefinitionUse, ...]
    definitions_gained: tuple[tuple[str, str], ...]
    definitions_lost: tuple[tuple[str, str], ...]
    definitions_verdict: Verdict


def rerank_live(settings: Settings, fusion: Retriever, questions: Sequence[str], corpus_version: str) -> RerankScores:
    live = JinaReranker.from_settings(settings, timeout=MEASUREMENT_TIMEOUT, max_attempts=MEASUREMENT_ATTEMPTS)
    paced = Paced(live)
    scores: dict[str, dict[str, float]] = {}
    with live:
        for position, question in enumerate(questions, start=1):
            pool = [result.chunk for result in fusion.search(question, RERANK_DEPTH)]
            scores[question] = paced.score(question, pool)
            logger.info("reranked %d/%d, %d tokens billed so far", position, len(questions), live.tokens_used)
    return RerankScores(
        model_id=MODEL_ID,
        corpus_version=corpus_version,
        depth=RERANK_DEPTH,
        scores=scores,
        latencies_ms=tuple(live.latencies_ms),
        tokens_billed=live.tokens_used,
    )


def pairs(members: Sequence[tuple[str, str]]) -> str:
    return ", ".join(f"{q} `{label}`" for q, label in members) or "none"


def render(report: BridgeReport, elapsed: float) -> str:
    verdict = report.verdict
    by_id = {r.query_id: r for r in report.rewritten}
    answerable = [r for r in report.rewritten if r.slice is not QuerySlice.NEGATIVE]
    negatives = [r for r in report.rewritten if r.slice is QuerySlice.NEGATIVE]
    lines = [
        "# Statutory-term bridge — Income-tax Act, 2025",
        "",
        "Auto-generated by `scripts/measure_bridge.py` (Step 5.8, ADR-086).",
        "A question's everyday words are mapped to the Act's own and appended",
        "before retrieval, inside the citation shortcut. Measured against",
        "production's Step 5.6 composition and the Step 5.3 pack.",
        "",
        "| Measure | Value |",
        "|---|---|",
        f"| corpus_version | `{report.corpus_version[:16]}…` |",
        f"| Chunks | {report.chunk_count} |",
        f"| Gold queries | {report.gold_count} |",
        f"| Map | `term_bridge_v1.json`, sha256 `{report.map_sha256[:16]}…` |",
        f"| Statutory terms / lay phrases | {report.statutory_terms} / {report.lay_phrases} |",
        f"| Pullable definitions | {report.pullable} |",
        f"| Questions rewritten (answerable / negative) | {len(answerable)} / {len(negatives)} |",
        f"| Tokens billed: embedding this run / rerank when first scored "
        f"| {report.tokens_embedded} / {report.tokens_reranked} |",
        f"| Run time | {elapsed:.1f}s |",
        "",
        "## The rule (registered before the first run, ADR-086)",
        "",
        "At least one label of the frozen vocabulary cohort (ADR-085) is recovered,",
        "no answerable question loses a label it was delivered, and no",
        "citation-slice question is rewritten.",
        "",
        f"**Verdict: {'ADOPTED' if verdict.adopted else 'REJECTED'}.**",
        "",
        *([f"- {reason}" for reason in verdict.reasons] or ["- every clause held"]),
        "",
        "**Caveat.** The map's lay side was written after the four cohort",
        "questions had been read (Step 5.7 names them). It was written by walking",
        "the section 2 glossary and the section titles and frozen before this",
        "script first ran, but a recovery count on those four is not evidence of",
        "how the map does on questions it was not written for. The no-loss clause",
        "over every other answerable question is the stronger check.",
        "",
        "## Vocabulary cohort",
        "",
        f"Recovered **{len(report.recovered)} of {len(report.cohort)}**.",
        "",
        "| query | label | recovered | added terms | question |",
        "|---|---|---|---|---|",
    ]
    for q, label in report.cohort:
        row = by_id.get(q)
        added = "; ".join(row.added) if row else "—"
        question = row.question if row else ""
        lines.append(
            f"| {q} | {label} | {'yes' if (q, label) in report.recovered else 'no'} "
            f"| {added} | {question} |"
        )
    lines += [
        "",
        "## Every rewritten question",
        "",
        "Labels are those the delivered pack credits leniently.",
        "",
        "| query | slice | added terms | before | after | label evidence dropped | question |",
        "|---|---|---|---|---|---|---|",
        *[
            f"| {r.query_id} | {r.slice.value} | {'; '.join(r.added)} "
            f"| {', '.join(r.before) or '—'} | {', '.join(r.after) or '—'} "
            f"| {', '.join(r.dropped) or '—'} | {r.question} |"
            for r in report.rewritten
        ],
        "",
        f"Gained: {pairs(report.gained)}.",
        "",
        f"Lost: {pairs(report.lost)}.",
        "",
        "## Evidence the rule cannot see (reported, not gated)",
        "",
        "\"Label evidence dropped\" lists units inside a label's subtree that the",
        "pack carried before the bridge and no longer does. Lenient credit never",
        "counts a descendant (ADR-060), so for a question labelled coarser than its",
        "answer, losing the answering descendant is not a lost label.",
        "",
        *[
            f"- {r.query_id} labels {', '.join(f'`{label}`' for label in r.labels)} "
            f"and lost {', '.join(f'`{u}`' for u in r.dropped)}."
            for r in report.rewritten
            if r.dropped
        ],
        *([] if any(r.dropped for r in report.rewritten) else ["- none"]),
        "",
        "## Failure categories after the bridge",
        "",
        "Step 5.7's classifier re-run on the bridged pack. Reported, not gated.",
        "",
        "| category | 5.7 (frozen) | bridged |",
        "|---|---|---|",
        *[
            f"| {c.value} | {report.categories_before.get(c, 0)} | {report.categories_after.get(c, 0)} |"
            for c in FailureCategory
        ],
        "",
        "## Definitions pull (judged separately, ADR-086)",
        "",
        "The section 2 clause for each specific term the rewritten question uses,",
        "placed after every retrieved hit. Judged against the bridged pack: it must",
        "deliver a label that pack missed and lose none. Gold v2 labels no",
        "definition clause, so a gain is not reachable on this gold set.",
        "",
        f"**Verdict: {'ADOPTED' if report.definitions_verdict.adopted else 'REJECTED'}.**",
        "",
        *[f"- {reason}" for reason in report.definitions_verdict.reasons],
        "",
    ]
    used = [d for d in report.definitions if d.delivered]
    lines += [
        f"Delivered a definition on {len(used)} of {report.gold_count} questions, "
        f"{sum(d.tokens for d in used)} tokens in all.",
        "",
        "| query | definitions delivered | tokens |",
        "|---|---|---|",
        *[f"| {d.query_id} | {', '.join(d.delivered)} | {d.tokens} |" for d in used],
        *([] if used else ["| — | | |"]),
        "",
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
    questions = [q.question for q in gold]
    entries = load_bridge_map()
    bridge = TermBridge(entries, chunks)
    rewritten = {q.query_id: bridge.rewrite(q.question) for q in gold}
    changed = sorted({rewritten[q.query_id] for q in gold} - set(questions))

    tokens_embedded = 0
    with embedder:
        try:
            dense = DenseRetriever.from_store(store, chunks, embedder, corpus_version=corpus_version)
            gold_vectors = load_query_vectors(
                store / QUERY_VECTORS_FILENAME, model=embedder.info(), questions=questions
            )
            gold_scores = load_rerank_scores(
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
            print(f"error: {error} (run scripts/measure_hybrid.py, then scripts/measure_rerank.py)", file=sys.stderr)
            return 1
        vectors_path = store / BRIDGE_VECTORS_FILENAME
        try:
            bridged_vectors = load_query_vectors(vectors_path, model=embedder.info(), questions=changed)
            logger.info("read stored bridged-question vectors from %s; billing nothing", vectors_path)
        except (FileNotFoundError, StaleVectorStoreError) as reason:
            logger.info("embedding %d rewritten questions (%s)", len(changed), reason)
            embedded = {text: tuple(dense.embed_query(text)) for text in changed}
            bridged_vectors = QueryVectors(
                model=embedder.info(), fingerprint_cosine=dense.fingerprint_cosine, vectors=embedded
            )
            write_query_vectors(vectors_path, bridged_vectors)
            tokens_embedded = embedder.tokens_used

    fusion = FusionRetriever(
        [CachedQueryRetriever(dense, {**gold_vectors.vectors, **bridged_vectors.vectors}), BM25Retriever(chunks)]
    )
    scores_path = settings.data_dir / "rerank" / BRIDGE_SCORES_FILENAME
    try:
        bridged_scores = load_rerank_scores(
            scores_path,
            model_id=MODEL_ID,
            corpus_version=corpus_version,
            depth=RERANK_DEPTH,
            questions=changed,
        )
        logger.info("read stored bridged-question rerank scores from %s; billing nothing", scores_path)
    except (FileNotFoundError, StaleRerankScoresError) as reason:
        logger.info("reranking %d rewritten questions through the API (%s)", len(changed), reason)
        bridged_scores = rerank_live(settings, fusion, changed, corpus_version)
        write_rerank_scores(scores_path, bridged_scores)

    reranker = StoredReranker({**gold_scores.scores, **bridged_scores.scores})
    shortcut = CitationRetriever(chunks)
    base = ShortcutRetriever(shortcut, RerankRetriever(fusion, reranker))
    bridged = ShortcutRetriever(shortcut, BridgedRetriever(bridge, RerankRetriever(fusion, reranker)))
    packer = EvidencePacker(chunks)
    classifier = FailureClassifier(chunks)

    packs_before: dict[str, list[str]] = {}
    packs_after: dict[str, list[str]] = {}
    packs_defined: dict[str, list[str]] = {}
    definitions: list[DefinitionUse] = []
    categories_after: Counter[FailureCategory] = Counter()
    for query in gold:
        pool = bridged.search(query.question, EVIDENCE_POOL)
        packs_before[query.query_id] = [
            unit.citation for unit in packer.pack(base.search(query.question, EVIDENCE_POOL)).units
        ]
        packs_after[query.query_id] = [unit.citation for unit in packer.pack(pool).units]
        defined = packer.pack(pool, definitions=bridge.definitions(rewritten[query.query_id]))
        packs_defined[query.query_id] = [unit.citation for unit in defined.units]
        pulled = [unit for unit in defined.units if unit.role is EvidenceRole.DEFINITION]
        definitions.append(
            DefinitionUse(
                query_id=query.query_id,
                delivered=tuple(unit.citation for unit in pulled),
                tokens=sum(unit.tokens for unit in pulled),
            )
        )
        if query.slice is not QuerySlice.NEGATIVE:
            pooled = [result.chunk.node_path for result in pool]
            for failure in classifier.classify(
                query.query_id, query.required, pooled, packs_after[query.query_id]
            ):
                categories_after[failure.category] += 1

    before = delivered(gold, packs_before)
    after = delivered(gold, packs_after)
    defined = delivered(gold, packs_defined)
    frozen = load_cohorts(settings.evals_dir / "datasets" / COHORTS_FILENAME)
    cohort = frozen.cohorts[FailureCategory.VOCABULARY]
    expanded_citation = [
        q.query_id for q in gold if q.slice is QuerySlice.CITATION and rewritten[q.query_id] != q.question
    ]
    report = BridgeReport(
        corpus_version=corpus_version,
        chunk_count=len(chunks),
        gold_count=len(gold),
        map_sha256=hashlib.sha256(BRIDGE_MAP.read_bytes()).hexdigest(),
        statutory_terms=len(entries),
        lay_phrases=sum(len(entry.lay) for entry in entries),
        pullable=len(bridge.pullable),
        tokens_embedded=tokens_embedded,
        tokens_reranked=bridged_scores.tokens_billed,
        rewritten=tuple(
            Rewritten(
                query_id=q.query_id,
                slice=q.slice,
                question=q.question,
                labels=q.required,
                added=bridge.expand(q.question),
                before=tuple(sorted(before.get(q.query_id, ()))),
                after=tuple(sorted(after.get(q.query_id, ()))),
                dropped=tuple(
                    unit
                    for unit in packs_before[q.query_id]
                    if any(credits(label, unit, CreditMode.LENIENT) for label in q.required)
                    and not any(
                        credits(kept, unit, CreditMode.LENIENT) for kept in packs_after[q.query_id]
                    )
                ),
            )
            for q in gold
            if rewritten[q.query_id] != q.question
        ),
        cohort=cohort,
        recovered=tuple(pair for pair in cohort if pair[1] in after[pair[0]]),
        gained=gained(before, after),
        lost=lost(before, after),
        verdict=judge_bridge(before, after, cohort, expanded_citation),
        categories_before={c: len(members) for c, members in frozen.cohorts.items()},
        categories_after=dict(categories_after),
        definitions=tuple(definitions),
        definitions_gained=gained(after, defined),
        definitions_lost=lost(after, defined),
        definitions_verdict=judge_definitions(after, defined),
    )

    artifact_dir = settings.evals_dir / "reports" / corpus_version
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact = artifact_dir / ARTIFACT
    with artifact.open("w", encoding="utf-8", newline="") as handle:
        handle.write(
            json.dumps(report.model_dump(mode="json"), sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        )
        handle.write("\n")
    elapsed = time.perf_counter() - started
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(render(report, elapsed), encoding="utf-8", newline="")

    print(f"rewritten: {len(report.rewritten)} questions")
    print(f"vocabulary cohort recovered: {len(report.recovered)} of {len(cohort)}")
    print(f"gained {pairs(report.gained)}; lost {pairs(report.lost)}")
    print(f"bridge: {'ADOPTED' if report.verdict.adopted else 'REJECTED'} {list(report.verdict.reasons)}")
    print(
        f"definitions: {'ADOPTED' if report.definitions_verdict.adopted else 'REJECTED'} "
        f"{list(report.definitions_verdict.reasons)}"
    )
    print(f"tokens billed: embed {tokens_embedded}, rerank {bridged_scores.tokens_billed}")
    print(f"wrote {artifact}")
    print(f"wrote {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
