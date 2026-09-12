"""Step 8.1 — the evidence-sufficiency grader measured against production's
Step 5.8 composition, judged by the rule in `taxverity.evals.sufficiency`,
registered before this script first ran (ADR-099).

Offline: the question vectors and rerank scores are the stored Step 5.2, 5.6
and 5.8 files, so this bills nothing and refuses to run without them. Requires
`TAXVERITY_JINA_API_KEY` only for the embedder identity check.

Writes `evals/reports/<corpus_version>/evidence_sufficiency.json` and
`reports/evidence_sufficiency.md`.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from taxverity.chunking.pipeline import read_corpus_version
from taxverity.chunking.store import load_chunks
from taxverity.config import MissingSettingError, Settings
from taxverity.embedding.jina_api import JinaAPIEmbedder
from taxverity.embedding.store import StaleVectorStoreError
from taxverity.evals.bridge import BRIDGE_SCORES_FILENAME, BRIDGE_VECTORS_FILENAME
from taxverity.evals.gold import GOLD_V2_FILENAME, QuerySlice, load_gold_set
from taxverity.evals.ladder import Verdict
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
from taxverity.evals.sufficiency import (
    MAX_FALSE_SHARE,
    MIN_CATCH_SHARE,
    Case,
    Rung,
    RungResult,
    Tally,
    candidates,
    cross_validate,
    flags,
    judge_sufficiency,
    tally,
)
from taxverity.evals.taxonomy import FailureCategory, FailureClassifier
from taxverity.observability import configure_logging, get_logger
from taxverity.retrieval.bm25 import BM25Retriever
from taxverity.retrieval.bridge import BridgedRetriever, TermBridge, load_bridge_map
from taxverity.retrieval.citations import CitationRetriever, ShortcutRetriever
from taxverity.retrieval.dense import DenseRetrievalError, DenseRetriever
from taxverity.retrieval.evidence import EVIDENCE_POOL, EvidencePacker
from taxverity.retrieval.fusion import FusionRetriever
from taxverity.retrieval.rerank import MODEL_ID, RERANK_DEPTH, RerankRetriever
from taxverity.retrieval.sufficiency import GraderConfig, signals_of

logger = get_logger(__name__)

REPORT = Path("reports") / "evidence_sufficiency.md"
ARTIFACT = "evidence_sufficiency.json"
VECTORS_KEY = "jina-api"


class Row(BaseModel):
    model_config = ConfigDict(frozen=True)

    case: Case
    question: str
    categories: tuple[FailureCategory, ...]
    tokens: int


class SufficiencyReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    corpus_version: str
    chunk_count: int
    gold_count: int
    rows: tuple[Row, ...]
    results: tuple[RungResult, ...]
    adopted: Rung | None
    verdict: Verdict
    # Diagnostic, not gated: the most any setting catches over every question,
    # in sample, while flagging no more than the rule allows.
    bounds: dict[Rung, Tally | None]


def build_cases(settings: Settings) -> tuple[str, int, int, list[Row]] | str:
    store = settings.vectors_dir / VECTORS_KEY
    corpus_version = read_corpus_version(settings.interim_dir / "corpus_manifest.json")
    try:
        chunks, _ = load_chunks(settings.interim_dir, corpus_version=corpus_version)
        embedder = JinaAPIEmbedder.from_settings(settings)
    except (RuntimeError, MissingSettingError) as error:
        return str(error)

    gold = load_gold_set(settings.evals_dir / "datasets" / GOLD_V2_FILENAME)
    questions = [q.question for q in gold]
    bridge = TermBridge(load_bridge_map(), chunks)
    rewritten = {q.query_id: bridge.rewrite(q.question) for q in gold}
    changed = sorted(set(rewritten.values()) - set(questions))
    rerank_dir = settings.data_dir / "rerank"
    try:
        with embedder:
            dense = DenseRetriever.from_store(store, chunks, embedder, corpus_version=corpus_version)
            vectors = {
                **load_query_vectors(store / QUERY_VECTORS_FILENAME, model=embedder.info(), questions=questions).vectors,
                **load_query_vectors(store / BRIDGE_VECTORS_FILENAME, model=embedder.info(), questions=changed).vectors,
            }
        scores = {
            **load_rerank_scores(
                rerank_dir / RERANK_SCORES_FILENAME,
                model_id=MODEL_ID, corpus_version=corpus_version, depth=RERANK_DEPTH, questions=questions,
            ).scores,
            **load_rerank_scores(
                rerank_dir / BRIDGE_SCORES_FILENAME,
                model_id=MODEL_ID, corpus_version=corpus_version, depth=RERANK_DEPTH, questions=changed,
            ).scores,
        }
    except (
        RuntimeError,
        DenseRetrievalError,
        StaleVectorStoreError,
        StaleRerankScoresError,
        FileNotFoundError,
    ) as error:
        return f"{error} (run scripts/measure_hybrid.py, measure_rerank.py, then measure_bridge.py)"

    fusion = FusionRetriever([CachedQueryRetriever(dense, vectors), BM25Retriever(chunks)])
    reranker = StoredReranker(scores)
    shortcut = CitationRetriever(chunks)
    production = ShortcutRetriever(shortcut, BridgedRetriever(bridge, RerankRetriever(fusion, reranker)))
    packer = EvidencePacker(chunks)
    classifier = FailureClassifier(chunks)

    rows = []
    for query in gold:
        results = production.search(query.question, EVIDENCE_POOL)
        pack = packer.pack(results)
        packed = [unit.citation for unit in pack.units]
        head = [result.chunk for result in fusion.search(rewritten[query.query_id], RERANK_DEPTH)]
        relevance = reranker.score(rewritten[query.query_id], head)
        signals = signals_of(
            packer,
            pack,
            citation_hit=bool(shortcut.search(query.question, EVIDENCE_POOL)),
            top_relevance=max(relevance.values()) if relevance else None,
        )
        failures = ()
        if query.slice is not QuerySlice.NEGATIVE:
            pooled = [result.chunk.node_path for result in results]
            failures = classifier.classify(query.query_id, query.required, pooled, packed)
        rows.append(
            Row(
                case=Case(
                    query_id=query.query_id,
                    slice=query.slice,
                    signals=signals,
                    missed=tuple(f.label for f in failures),
                ),
                question=query.question,
                categories=tuple(f.category for f in failures),
                tokens=pack.tokens,
            )
        )
    return corpus_version, len(chunks), len(gold), rows


def bound(rung: Rung, cases: list[Case]) -> Tally | None:
    allowed = [
        t
        for t in (tally(cases, {c.query_id for c in cases if flags(c, config)}) for config in candidates(rung, cases))
        if t.false <= MAX_FALSE_SHARE * t.sufficient
    ]
    return max(allowed, key=lambda t: (t.caught, -t.false), default=None)


def describe(config: GraderConfig) -> str:
    parts = []
    if config.min_relevance is not None:
        parts.append(f"relevance < {config.min_relevance:.3f}")
    if config.unmet_within is not None:
        parts.append(f"unmet reference in first {config.unmet_within}")
    rule = " or ".join(parts) or "never fires"
    return f"{rule}{', citation hit overrides' if config.citation_override else ''}"


def render(report: SufficiencyReport, elapsed: float) -> str:
    cases = [row.case for row in report.rows]
    answerable = [c for c in cases if c.answerable]
    lines = [
        "# Evidence sufficiency — Income-tax Act, 2025",
        "",
        "Auto-generated by `scripts/measure_sufficiency.py` (Step 8.1, ADR-099).",
        "Whether the delivered pack is enough to answer, graded without a model",
        "call from signals retrieval already produced. Measured on production's",
        "Step 5.8 composition and the Step 5.3 pack.",
        "",
        "| Measure | Value |",
        "|---|---|",
        f"| corpus_version | `{report.corpus_version[:16]}…` |",
        f"| Chunks | {report.chunk_count} |",
        f"| Gold queries | {report.gold_count} |",
        f"| Answerable: insufficient / sufficient | {sum(c.insufficient for c in answerable)} "
        f"/ {sum(not c.insufficient for c in answerable)} |",
        f"| Negatives | {len(cases) - len(answerable)} |",
        f"| Run time | {elapsed:.1f}s |",
        "",
        "## The rule (registered before the first run, ADR-099)",
        "",
        "Insufficient means a gold label the pack does not credit leniently (ADR-085).",
        "Cumulative ladder: (1) best relevance below a threshold; (2) or a reference",
        "from one of the first N retrieved units to a provision the pack lacks;",
        "(3) either, overridden by a citation-shortcut hit. Settings chosen on one",
        "parity fold, scored on the other. A rung passes when, held out, it flags at",
        f"least {MIN_CATCH_SHARE:.0%} of insufficient answerable questions and at most",
        f"{MAX_FALSE_SHARE:.0%} of sufficient ones. The simplest passing rung is adopted;",
        "a later one replaces it only by catching more with no more false flags.",
        "",
        f"**Verdict: {'ADOPTED, rung ' + report.adopted.value if report.adopted else 'REJECTED'}.**",
        "",
        *([f"- {reason}" for reason in report.verdict.reasons] or ["- every rung passed"]),
        "",
        "## Held out",
        "",
        "Negatives flagged is exposure to false rescue (ADR-036), not a success.",
        "",
        "| rung | caught | false flags | negatives flagged | passes | settings (odd-trained / even-trained) | settings on all |",
        "|---|---|---|---|---|---|---|",
    ]
    for result in report.results:
        held = result.held_out
        lines.append(
            f"| {result.rung.value} | {held.caught}/{held.failing} | {held.false}/{held.sufficient} "
            f"| {held.negatives_flagged}/{held.negatives} | {'yes' if held.passes else 'no'} "
            f"| {describe(result.fold_configs[0])} / {describe(result.fold_configs[1])} "
            f"| {describe(result.shipped)} |"
        )
    lines += [
        "",
        "## In-sample bound (diagnostic, not gated)",
        "",
        "The most any setting in a rung's grid catches over all 64 answerable",
        f"questions at once, while flagging at most {MAX_FALSE_SHARE:.0%} of sufficient ones.",
        "A bound below the catch threshold means no threshold choice could have",
        "passed, so the rejection is not an artefact of the fold split.",
        "",
        "| rung | caught | false flags | negatives flagged |",
        "|---|---|---|---|",
        *[
            f"| {rung.value} | {t.caught}/{t.failing} | {t.false}/{t.sufficient} | {t.negatives_flagged}/{t.negatives} |"
            if t
            else f"| {rung.value} | — | — | — |"
            for rung, t in report.bounds.items()
        ],
    ]
    flagged = {result.rung: set(result.flagged) for result in report.results}
    lines += [
        "",
        "## Every question",
        "",
        "Unmet counts references from retrieved units to provisions the pack lacks,",
        "with the first unit's shown. Rung columns mark a held-out flag.",
        "",
        "| query | slice | truth | missed (category) | citation hit | top relevance | unmet (first 1 / 3 / all) | "
        + " | ".join(r.value for r in Rung)
        + " | tokens |",
        "|---|---|---|---|---|---|---|" + "---|" * len(Rung) + "---|",
    ]
    for row in report.rows:
        case = row.case
        unmet = case.signals.unmet
        truth = "negative" if not case.answerable else ("insufficient" if case.insufficient else "sufficient")
        missed = ", ".join(f"{label} ({cat.value})" for label, cat in zip(case.missed, row.categories, strict=True)) or "—"
        top = "—" if case.signals.top_relevance is None else f"{case.signals.top_relevance:.3f}"
        counts = " / ".join(
            str(sum(1 for position, _, _ in unmet if position <= n)) for n in (1, 3)
        ) + f" / {len(unmet)}"
        marks = " | ".join("x" if case.query_id in flagged[r] else "" for r in Rung)
        lines.append(
            f"| {case.query_id} | {case.slice.value} | {truth} | {missed} "
            f"| {'yes' if case.signals.citation_hit else ''} | {top} | {counts} | {marks} | {row.tokens} |"
        )
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    configure_logging()
    settings = Settings()
    started = time.perf_counter()
    built = build_cases(settings)
    if isinstance(built, str):
        print(f"error: {built}", file=sys.stderr)
        return 1
    corpus_version, chunk_count, gold_count, rows = built
    cases = [row.case for row in rows]
    results = tuple(cross_validate(rung, cases) for rung in Rung)
    adopted, verdict = judge_sufficiency(results)
    report = SufficiencyReport(
        corpus_version=corpus_version,
        chunk_count=chunk_count,
        gold_count=gold_count,
        rows=tuple(rows),
        results=results,
        adopted=adopted,
        verdict=verdict,
        bounds={rung: bound(rung, cases) for rung in Rung},
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
    logger.info("graded %d questions in %.1fs", len(rows), elapsed)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(render(report, elapsed), encoding="utf-8", newline="")

    for result in results:
        held = result.held_out
        print(
            f"{result.rung.value}: caught {held.caught}/{held.failing}, false {held.false}/{held.sufficient}, "
            f"negatives {held.negatives_flagged}/{held.negatives}, passes={held.passes}"
        )
    for rung, t in report.bounds.items():
        print(f"{rung.value} in-sample bound: " + (f"caught {t.caught}/{t.failing}, false {t.false}/{t.sufficient}" if t else "none"))
    print(f"grader: {'ADOPTED ' + adopted.value if adopted else 'REJECTED'} {list(verdict.reasons)}")
    print(f"wrote {artifact}")
    print(f"wrote {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
