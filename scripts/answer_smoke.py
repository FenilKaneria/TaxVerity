"""Step 10.7 — the answer smoke set, run through production's retrieval and the
Step 10.6 generator.

Retrieval is production's composition (ADR-086) read from stored vectors and
rerank scores, so it bills no Jina tokens and needs no Jina key. Generation
bills Groq on the first run, paced under the free tier's 8,000 tokens a
minute. The Step 7.3 cache stores every finished stream, so a re-run bills
nothing; `--stored` only re-reports the stored run.

Writes `data/answers/answer_smoke_v1.json` and `reports/answer_smoke.md`.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from taxverity.chunking.models import Chunk
from taxverity.chunking.pipeline import read_corpus_version
from taxverity.chunking.store import load_chunks
from taxverity.config import MissingSettingError, Settings
from taxverity.embedding.store import load_vector_store
from taxverity.evals.answers import (
    ANSWER_EVAL_VERSION,
    ANSWER_RUN_FILENAME,
    SMOKE_QUERY_IDS,
    AnswerRecord,
    AnswerRun,
    EvidenceText,
    SmokeSummary,
    load_answer_run,
    store_answer_run,
    summarise,
)
from taxverity.evals.bridge import BRIDGE_SCORES_FILENAME, BRIDGE_VECTORS_FILENAME
from taxverity.evals.gold import GOLD_V2_FILENAME, QuerySlice, load_gold_set
from taxverity.evals.query_vectors import (
    QUERY_VECTORS_FILENAME,
    CachedQueryRetriever,
    load_query_vectors,
)
from taxverity.evals.rerank import (
    RERANK_SCORES_FILENAME,
    StoredReranker,
    load_rerank_scores,
)
from taxverity.generation.claims import ClaimEvent, WithheldEvent
from taxverity.generation.generate import (
    GENERATION_PROMPT_VERSION,
    GENERATION_STAGE_VERSION,
    AnswerGenerator,
)
from taxverity.llm.cache import CachedLLMClient
from taxverity.llm.client import LLMClient, LLMError
from taxverity.llm.tracing import LangfuseTracer, TracedLLMClient
from taxverity.observability import configure_logging, get_logger
from taxverity.retrieval.base import Retriever
from taxverity.retrieval.bm25 import BM25Retriever
from taxverity.retrieval.bridge import BridgedRetriever, TermBridge, load_bridge_map
from taxverity.retrieval.citations import CitationRetriever, ShortcutRetriever
from taxverity.retrieval.dense import DenseRetriever
from taxverity.retrieval.evidence import EVIDENCE_POOL, EvidencePack, EvidencePacker
from taxverity.retrieval.fusion import FusionRetriever
from taxverity.retrieval.rerank import MODEL_ID, RERANK_DEPTH, RerankRetriever

logger = get_logger(__name__)

REPORT = Path("reports") / "answer_smoke.md"
# One answer is roughly 5-7k tokens against 8,000 a minute (Step 7.1).
DEFAULT_PAUSE = 50.0


class Offline:
    """The served embedding identity with no way to embed: every question here
    has a stored vector, so reaching the network would be a bug."""

    def __init__(self, info) -> None:
        self._info = info

    def info(self):
        return self._info

    def embed(self, texts, kind):
        raise RuntimeError("the smoke set must not embed; its vectors are stored")


def build_retrieval(
    settings: Settings, questions: list[str]
) -> tuple[list[Chunk], Retriever]:
    """Production's composition (ADR-086) over stored vectors and rerank scores.

    The dense failure fallback is left out: nothing here can fail over the network.
    """
    corpus_version = read_corpus_version(settings.interim_dir / "corpus_manifest.json")
    chunks, _ = load_chunks(settings.interim_dir, corpus_version=corpus_version)
    store = settings.vectors_dir / "jina-api"
    vectors, ids, manifest = load_vector_store(store, corpus_version=corpus_version)
    bridge = TermBridge(load_bridge_map(), chunks)
    rewritten = sorted({bridge.rewrite(q) for q in questions} - set(questions))

    gold_vectors = load_query_vectors(
        store / QUERY_VECTORS_FILENAME, model=manifest.model, questions=questions
    )
    bridge_vectors = load_query_vectors(
        store / BRIDGE_VECTORS_FILENAME, model=manifest.model, questions=rewritten
    )
    rerank = settings.data_dir / "rerank"
    scores = {
        **load_rerank_scores(
            rerank / RERANK_SCORES_FILENAME, model_id=MODEL_ID, corpus_version=corpus_version,
            depth=RERANK_DEPTH, questions=questions,
        ).scores,
        **load_rerank_scores(
            rerank / BRIDGE_SCORES_FILENAME, model_id=MODEL_ID, corpus_version=corpus_version,
            depth=RERANK_DEPTH, questions=rewritten,
        ).scores,
    }  # fmt: skip

    dense = DenseRetriever(chunks, vectors, ids, manifest, Offline(manifest.model))
    fusion = FusionRetriever(
        [
            CachedQueryRetriever(
                dense, {**gold_vectors.vectors, **bridge_vectors.vectors}
            ),
            BM25Retriever(chunks),
        ]
    )
    ranked = BridgedRetriever(bridge, RerankRetriever(fusion, StoredReranker(scores)))
    return chunks, ShortcutRetriever(CitationRetriever(chunks), ranked)


def evidence_texts(pack: EvidencePack) -> tuple[EvidenceText, ...]:
    return tuple(
        EvidenceText(
            citation=unit.citation,
            text="\n".join([*(line.text for line in unit.context), unit.chunk.text]),
        )
        for unit in pack.units
    )


def render(run: AnswerRun, summary: SmokeSummary) -> str:
    gate = "PASS" if not summary.served_on_negative else "FAIL"
    negatives = sum(1 for r in run.records if r.slice is QuerySlice.NEGATIVE)
    lines = [
        "# Answer smoke set — Step 10.7",
        "",
        "Auto-generated by `scripts/answer_smoke.py`. Production retrieval from",
        "stored vectors and rerank scores, the Step 5.3 pack, and the Step 10.6",
        "generator with its verifier gate. No facts and no computation are passed:",
        "these are statutory questions, and Phase 13 wires extraction and the",
        "calculator in.",
        "",
        "| Measure | Value |",
        "|---|---|",
        f"| Model | `{run.model}` |",
        f"| Generation stage / prompt version | {run.generation_stage_version} / {run.prompt_version} |",
        f"| Questions (answerable / negative) | {summary.questions - negatives} / {negatives} |",
        f"| Answerable questions with a served claim | {summary.answered} |",
        f"| Claims served / withheld | {summary.claims_served} / {summary.claims_withheld} |",
        f"| Provider errors | {summary.errors} |",
        f"| Groq tokens this run | {run.tokens:,} |",
        "",
        "## Gate: no negative question gets a served statute claim",
        "",
        f"**{gate}.** Served on a negative: {', '.join(summary.served_on_negative) or 'none'}.",
        "",
        f"Answerable questions with nothing served (reported, not gated): "
        f"{', '.join(summary.silent_answerable) or 'none'}.",
        "",
        "## Every answer",
        "",
    ]
    for record in run.records:
        lines += [
            f"### {record.query_id} ({record.slice.value}) — {record.question}",
            "",
            f"Evidence: {', '.join(e.citation for e in record.evidence) or 'none'}. "
            f"{record.seconds:.1f}s.",
            "",
        ]
        if record.error:
            lines += [f"Provider error: {record.error}", ""]
        for claim in record.claims:
            cites = "; ".join(f'`{c.path}` "{c.quote}"' for c in claim.citations)
            lines.append(f"- [{claim.id}, {claim.type.value}] {claim.text} — {cites}")
        for withheld in record.withheld:
            lines.append(f"- [{withheld.id}, withheld] {withheld.reason}")
        if not record.claims and not record.withheld:
            lines.append("- nothing served")
        lines.append("")
    return "\n".join(lines)


def run_live(settings: Settings, pause: float) -> AnswerRun:
    gold = {
        q.query_id: q
        for q in load_gold_set(settings.evals_dir / "datasets" / GOLD_V2_FILENAME)
    }
    chunks, retriever = build_retrieval(
        settings, [gold[i].question for i in SMOKE_QUERY_IDS]
    )
    packer = EvidencePacker(chunks)
    base = LLMClient.from_settings(settings)
    llm = TracedLLMClient(
        CachedLLMClient(base, settings.llm_cache_dir),
        LangfuseTracer.from_settings(settings),
    )
    generator = AnswerGenerator(llm, {chunk.node_path: chunk for chunk in chunks})

    records = []
    for index, query_id in enumerate(SMOKE_QUERY_IDS, start=1):
        query = gold[query_id]
        pack = packer.pack(retriever.search(query.question, EVIDENCE_POOL))
        before = sum(base.tokens_used.values())
        started = time.perf_counter()
        claims: list[ClaimEvent] = []
        withheld: list[WithheldEvent] = []
        error = None
        try:
            for event in generator.generate(query.question, pack):
                (claims if isinstance(event, ClaimEvent) else withheld).append(event)
        except LLMError as failure:
            error = str(failure)
            logger.warning("%s: provider failed: %s", query_id, failure)
        seconds = time.perf_counter() - started
        spent = sum(base.tokens_used.values()) - before
        records.append(
            AnswerRecord(
                query_id=query_id,
                slice=query.slice,
                question=query.question,
                evidence=evidence_texts(pack),
                claims=tuple(claims),
                withheld=tuple(withheld),
                error=error,
                seconds=round(seconds, 1),
            )
        )
        logger.info(
            "%s (%d/%d): %d served, %d withheld, %d tokens",
            query_id, index, len(SMOKE_QUERY_IDS), len(claims), len(withheld), spent,
        )  # fmt: skip
        if spent and index < len(SMOKE_QUERY_IDS):
            time.sleep(pause)
    return AnswerRun(
        eval_version=ANSWER_EVAL_VERSION,
        generation_stage_version=GENERATION_STAGE_VERSION,
        prompt_version=GENERATION_PROMPT_VERSION,
        model=base.primary.model,
        tokens=sum(base.tokens_used.values()),
        records=tuple(records),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pause", type=float, default=DEFAULT_PAUSE)
    parser.add_argument(
        "--stored", action="store_true", help="re-report the stored run"
    )
    args = parser.parse_args()
    configure_logging()
    settings = Settings()
    path = settings.data_dir / "answers" / ANSWER_RUN_FILENAME

    if args.stored:
        run = load_answer_run(path)
    else:
        try:
            run = run_live(settings, args.pause)
        except (MissingSettingError, FileNotFoundError, RuntimeError) as error:
            print(f"error: {error}", file=sys.stderr)
            return 1
        store_answer_run(path, run)

    summary = summarise(run.records)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(render(run, summary), encoding="utf-8", newline="")
    negatives = sum(1 for r in run.records if r.slice is QuerySlice.NEGATIVE)
    print(
        f"answerable with a served claim: {summary.answered}/{summary.questions - negatives}"
    )
    print(f"claims served {summary.claims_served}, withheld {summary.claims_withheld}")
    print(f"served on negative: {', '.join(summary.served_on_negative) or 'none'}")
    print(f"provider errors: {summary.errors}")
    print(f"tokens {run.tokens:,}")
    print(f"wrote {path}")
    print(f"wrote {REPORT}")
    return 0 if not summary.served_on_negative else 2


if __name__ == "__main__":
    raise SystemExit(main())
