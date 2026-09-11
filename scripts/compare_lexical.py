"""Step 4.4b — choose the no-network retrieval path by measurement.

A cumulative ladder over BM25: digit grouping, k1/b tuning, RM3 feedback. Each
rung is judged against the last adopted one by the rule pre-registered in
PLAN Step 4.4b, on the citation-shortcut composition, since that is what
production falls back to when the embedding API fails (ADR-075).

Writes `evals/reports/<corpus_version>/lexical_comparison.json` and
`reports/lexical_comparison.md`.
"""

from __future__ import annotations

import json
import math
import re
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from taxverity.chunking.models import Chunk
from taxverity.chunking.store import load_chunks
from taxverity.config import Settings
from taxverity.corpus.loader import normalise
from taxverity.evals.baseline import PRIMARY_K, RetrieverBaseline, measure
from taxverity.evals.gold import GOLD_V2_FILENAME, GoldQuery, QuerySlice, load_gold_set
from taxverity.evals.ladder import Params, Verdict, judge, select_params, two_fold_split
from taxverity.evals.metrics import CitationIndex, CreditMode, score_run
from taxverity.observability import configure_logging, get_logger
from taxverity.retrieval.base import Retriever, ScoredChunk, as_ranked_citations
from taxverity.retrieval.bm25 import BM25Retriever, tokenize
from taxverity.retrieval.citations import CitationRetriever, ShortcutRetriever

logger = get_logger(__name__)

REPORT = Path("reports") / "lexical_comparison.md"
ARTIFACT = "lexical_comparison.json"

# Step 3.4's untuned Okapi defaults, the ladder's starting point. Fixed here
# rather than read from bm25.py, whose defaults are now the ladder's winner.
DEFAULTS: Params = (1.5, 0.75)
K1_GRID = (0.9, 1.2, 1.5, 1.8, 2.1)
B_GRID = (0.3, 0.5, 0.75, 0.9)
LATENCY_BUDGET_MS = 100.0
LATENCY_REPEATS = 3

GROUPED_NUMBER = re.compile(r"\d[\d,]*\d")
AMOUNT = re.compile(r"\d{1,3}(?:,\d{2})*,\d{3}|\d{1,3}(?:,\d{3})+")

# --- rejected rungs (ADR-077) -------------------------------------------------
# Kept in this script, not in src/, so the recorded result stays reproducible
# without shipping code production never runs.

# A comma between digits with a 2- or 3-digit group after it: Indian (2,00,000)
# and Western (200,000) grouping both collapse to 200000.
DIGIT_GROUPING = re.compile(r"(?<=\d),(?=\d{2,3}(?!\d))")


def tokenize_grouped(text: str) -> list[str]:
    return tokenize(DIGIT_GROUPING.sub("", normalise(text)))


# Anserini's RM3 defaults, untuned: every free parameter tuned against 64
# answerable queries is another way to fit the ruler.
FB_DOCS = 10
FB_TERMS = 10
ORIGINAL_WEIGHT = 0.5


class FeedbackRetriever:
    """RM3 pseudo-relevance feedback: the first pass's top chunks lend their
    vocabulary to a second pass."""

    def __init__(self, bm25: BM25Retriever) -> None:
        self._bm25 = bm25

    def expand(self, query: str) -> dict[str, float]:
        original = list(dict.fromkeys(self._bm25.tokenize(query)))
        feedback = self._bm25.search(query, FB_DOCS) if original else []
        if not feedback:
            return {}

        total = sum(result.score for result in feedback)
        relevance: dict[str, float] = {}
        for result in feedback:
            frequencies = self._bm25.term_frequencies(result.chunk)
            length = sum(frequencies.values())
            share = result.score / total
            for token, frequency in frequencies.items():
                relevance[token] = relevance.get(token, 0.0) + share * frequency / length

        # Ranked by P(w|R) x idf: ADR-061 keeps no stop list, and by relevance
        # mass alone the top expansion terms are "the" and "of".
        expansion = sorted(
            relevance, key=lambda token: (-relevance[token] * self._bm25.idf(token), token)
        )[:FB_TERMS]

        weights = dict.fromkeys(original, ORIGINAL_WEIGHT / len(original))
        mass = sum(relevance[token] for token in expansion)
        for token in expansion:
            weights[token] = (
                weights.get(token, 0.0) + (1 - ORIGINAL_WEIGHT) * relevance[token] / mass
            )
        return weights

    def search(self, query: str, k: int) -> Sequence[ScoredChunk]:
        weights = self.expand(query)
        return self._bm25.search_weighted(weights, k) if weights else []


# -----------------------------------------------------------------------------


class Rung(BaseModel):
    model_config = ConfigDict(frozen=True)

    rung: str
    change: str
    config: str
    against: str | None
    bare: RetrieverBaseline
    shortcut: RetrieverBaseline
    p50_ms: float
    p95_ms: float
    verdict: Verdict | None


class GridPoint(BaseModel):
    model_config = ConfigDict(frozen=True)

    k1: float
    b: float
    even: float
    odd: float
    whole: float


class Fold(BaseModel):
    model_config = ConfigDict(frozen=True)

    train: str
    test: str
    picked: Params
    held_out: float
    default_held_out: float


class DigitAudit(BaseModel):
    model_config = ConfigDict(frozen=True)

    hits: int
    non_amounts: tuple[str, ...]


class LadderReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    corpus_version: str
    chunk_count: int
    gold_count: int
    primary_k: int
    digit_audit: DigitAudit
    grid: tuple[GridPoint, ...]
    folds: tuple[Fold, ...]
    whole_pick: Params
    rungs: tuple[Rung, ...]
    adopted: str


def audit_digits(chunks: Sequence[Chunk]) -> DigitAudit:
    hits = 0
    odd: set[str] = set()
    for chunk in chunks:
        if chunk.parent_id is not None:
            continue
        text = normalise(chunk.text)
        hits += len(DIGIT_GROUPING.findall(text))
        odd.update(
            token
            for token in GROUPED_NUMBER.findall(text)
            if "," in token and not AMOUNT.fullmatch(token)
        )
    return DigitAudit(hits=hits, non_amounts=tuple(sorted(odd)))


def latency_ms(retriever: Retriever, gold: Sequence[GoldQuery]) -> tuple[float, float]:
    samples = []
    for query in gold:
        best = math.inf
        for _ in range(LATENCY_REPEATS):
            started = time.perf_counter()
            retriever.search(query.question, PRIMARY_K)
            best = min(best, (time.perf_counter() - started) * 1000)
        samples.append(best)
    ordered = sorted(samples)
    return ordered[len(ordered) // 2], ordered[math.ceil(0.95 * len(ordered)) - 1]


def rung(
    name: str,
    change: str,
    config: str,
    primary: Retriever,
    shortcut: CitationRetriever,
    gold: Sequence[GoldQuery],
    index: CitationIndex,
    *,
    against: Rung | None = None,
    extra: Callable[[float], list[str]] = lambda p95: [],
) -> Rung:
    composed = ShortcutRetriever(shortcut, primary)
    bare = measure(f"{name}: {config}", primary, gold, index)
    with_shortcut = measure(
        f"{name}: {config} + citation shortcut", composed, gold, index, ordinal_scores=True
    )
    p50, p95 = latency_ms(composed, gold)
    verdict = (
        judge(with_shortcut.primary, against.shortcut.primary, extra=extra(p95))
        if against is not None
        else None
    )
    return Rung(
        rung=name,
        change=change,
        config=config,
        against=against.rung if against is not None else None,
        bare=bare,
        shortcut=with_shortcut,
        p50_ms=p50,
        p95_ms=p95,
        verdict=verdict,
    )


def lenient_ndcg(
    queries: Sequence[GoldQuery], runs: dict[str, list[str]], index: CitationIndex
) -> float:
    return score_run(queries, runs, PRIMARY_K, index).overall[CreditMode.LENIENT].ndcg


def tune(
    chunks: Sequence[Chunk],
    tokenizer: Callable[[str], list[str]],
    shortcut: CitationRetriever,
    gold: Sequence[GoldQuery],
    index: CitationIndex,
) -> tuple[tuple[GridPoint, ...], tuple[Fold, ...], Params]:
    even, odd = two_fold_split(gold)
    answerable = even + odd
    grid: list[GridPoint] = []
    for k1 in K1_GRID:
        for b in B_GRID:
            composed = ShortcutRetriever(
                shortcut, BM25Retriever(chunks, k1=k1, b=b, tokenizer=tokenizer)
            )
            runs = {
                q.query_id: as_ranked_citations(composed.search(q.question, PRIMARY_K))
                for q in answerable
            }
            grid.append(
                GridPoint(
                    k1=k1,
                    b=b,
                    even=lenient_ndcg(even, runs, index),
                    odd=lenient_ndcg(odd, runs, index),
                    whole=lenient_ndcg(answerable, runs, index),
                )
            )

    folds = []
    for train, test in (("even", "odd"), ("odd", "even")):
        picked = select_params(
            {(p.k1, p.b): getattr(p, train) for p in grid}, prefer=DEFAULTS
        )
        held = {(p.k1, p.b): getattr(p, test) for p in grid}
        folds.append(
            Fold(
                train=train,
                test=test,
                picked=picked,
                held_out=held[picked],
                default_held_out=held[DEFAULTS],
            )
        )
    whole = select_params({(p.k1, p.b): p.whole for p in grid}, prefer=DEFAULTS)
    return tuple(grid), tuple(folds), whole


def render(report: LadderReport, elapsed: float) -> str:
    rungs = report.rungs
    lines = [
        "# Lexical path comparison — Income-tax Act, 2025",
        "",
        "Auto-generated by `scripts/compare_lexical.py` (Step 4.4b). The path",
        "measured here is what retrieval falls back to when the embedding API",
        "fails (ADR-075), and the lexical leg of Phase 5's hybrid.",
        "",
        "| Measure | Value |",
        "|---|---|",
        f"| corpus_version | `{report.corpus_version[:16]}…` |",
        f"| Chunks | {report.chunk_count} |",
        f"| Gold queries | {report.gold_count} |",
        f"| Primary k | {report.primary_k} |",
        f"| Run time | {elapsed:.1f}s |",
        f"| **Adopted path** | {report.adopted} |",
        "",
        "## The rule, pre-registered before any run",
        "",
        "Each rung is judged against the last adopted rung, on the citation-",
        "shortcut composition, over the answerable queries at k=10:",
        "",
        "1. Lenient recall@10 and lenient nDCG@10 must both not fall, and at",
        "   least one must rise.",
        "2. No slice loses more than one query's worth of lenient recall.",
        "3. Tuning only: both held-out folds must not fall against the defaults.",
        f"4. RM3 only: search p95 at most {LATENCY_BUDGET_MS:.0f} ms per query.",
        "",
        "## Ladder",
        "",
        "Scores are for the shortcut composition except the `bare` column.",
        "",
        "| rung | change | config | bare lenient R@10 | strict R@10 | lenient R@10 "
        "| lenient MRR | lenient nDCG@10 | p50 ms | p95 ms | verdict |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rungs:
        overall = r.shortcut.primary.overall
        verdict = (
            "baseline"
            if r.verdict is None
            else ("**adopted**" if r.verdict.adopted else "rejected")
        )
        lines.append(
            f"| {r.rung} | {r.change} | `{r.config}` "
            f"| {r.bare.primary.overall[CreditMode.LENIENT].recall:.3f} "
            f"| {overall[CreditMode.STRICT].recall:.3f} "
            f"| {overall[CreditMode.LENIENT].recall:.3f} "
            f"| {overall[CreditMode.LENIENT].mrr:.3f} "
            f"| {overall[CreditMode.LENIENT].ndcg:.3f} "
            f"| {r.p50_ms:.1f} | {r.p95_ms:.1f} | {verdict} |"
        )

    lines += ["", "### Verdicts", ""]
    for r in rungs:
        if r.verdict is None:
            continue
        why = "; ".join(r.verdict.reasons) or "every rule passed"
        lines.append(f"- **{r.rung}** against {r.against}: {why}.")

    lines += [
        "",
        "## Per slice, lenient recall@10 (shortcut composition)",
        "",
        "| slice | " + " | ".join(r.rung for r in rungs) + " |",
        "|---|" + "---|" * len(rungs),
    ]
    for member in QuerySlice:
        if member is QuerySlice.NEGATIVE:
            continue
        lines.append(
            f"| {member.value} | "
            + " | ".join(
                f"{r.shortcut.primary.per_slice[member][CreditMode.LENIENT].recall:.3f}"
                for r in rungs
            )
            + " |"
        )

    lines += [
        "",
        "## Queries that moved, against the rung each was judged against",
        "",
        "| rung | gained | lost |",
        "|---|---|---|",
    ]
    by_name = {r.rung: r for r in rungs}
    for r in rungs:
        if r.against is None:
            continue
        before = {o.query_id: o.lenient_recall for o in by_name[r.against].shortcut.outcomes}
        after = {o.query_id: o.lenient_recall for o in r.shortcut.outcomes}
        gained = [q for q in after if after[q] > before[q]]
        lost = [q for q in after if after[q] < before[q]]
        lines.append(f"| {r.rung} | {', '.join(gained) or '—'} | {', '.join(lost) or '—'} |")

    lines += [
        "",
        "## k1 / b tuning",
        "",
        "Lenient nDCG@10 of the shortcut composition per setting, on each",
        "half of the answerable queries (split by query-id parity) and on all",
        "of them. A tie keeps the defaults.",
        "",
        "| k1 | b | even | odd | all |",
        "|---|---|---|---|---|",
        *[
            f"| {p.k1} | {p.b} | {p.even:.3f} | {p.odd:.3f} | {p.whole:.3f} |"
            for p in report.grid
        ],
        "",
        "| trained on | tested on | picked (k1, b) | held-out nDCG | defaults held-out |",
        "|---|---|---|---|---|",
        *[
            f"| {f.train} | {f.test} | {f.picked} | {f.held_out:.3f} "
            f"| {f.default_held_out:.3f} |"
            for f in report.folds
        ],
        "",
        f"Picked on all answerable queries: {report.whole_pick}.",
        "",
        "## Digit-grouping audit",
        "",
        f"The fix fires {report.digit_audit.hits} times across the corpus root",
        "chunks. Comma-separated digit runs that are not amounts, which the fix",
        "would wrongly merge: "
        + (", ".join(f"`{t}`" for t in report.digit_audit.non_amounts) or "none")
        + ".",
        "",
        "## Negative separation (bare retrievers)",
        "",
        "Top-1 score per rung. A score is comparable only within one retriever",
        "(Step 3.3). Reported, not gated; a threshold is Phase 12's.",
        "",
        "| rung | answerable min / median / max | negative min / median / max |",
        "|---|---|---|",
    ]
    for r in rungs:
        answerable = sorted(o.top_score for o in r.bare.outcomes if o.top_score is not None)
        negative = sorted(n.top_score for n in r.bare.negatives if n.top_score is not None)
        lines.append(f"| {r.rung} | {spread(answerable)} | {spread(negative)} |")
    return "\n".join(lines) + "\n"


def spread(values: list[float]) -> str:
    if not values:
        return "—"
    return f"{values[0]:.2f} / {values[len(values) // 2]:.2f} / {values[-1]:.2f}"


def main() -> int:
    configure_logging()
    settings = Settings()
    started = time.perf_counter()

    try:
        chunks, manifest = load_chunks(settings.interim_dir)
    except RuntimeError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    gold = load_gold_set(settings.evals_dir / "datasets" / GOLD_V2_FILENAME)
    index = CitationIndex(chunks)
    shortcut = CitationRetriever(chunks)

    k1, b = DEFAULTS
    base = rung(
        "A",
        "BM25 as-is",
        f"k1={k1} b={b}",
        BM25Retriever(chunks, k1=k1, b=b),
        shortcut,
        gold,
        index,
    )
    grouped = rung(
        "B",
        "+ digit grouping",
        f"grouped k1={k1} b={b}",
        BM25Retriever(chunks, k1=k1, b=b, tokenizer=tokenize_grouped),
        shortcut,
        gold,
        index,
        against=base,
    )
    incumbent = grouped if grouped.verdict.adopted else base
    tokenizer = tokenize_grouped if incumbent is grouped else tokenize
    label = "grouped " if incumbent is grouped else ""

    grid, folds, whole_pick = tune(chunks, tokenizer, shortcut, gold, index)
    cv_failures = [
        f"held-out nDCG on {f.test} fell {f.default_held_out:.3f} -> {f.held_out:.3f}"
        for f in folds
        if f.held_out < f.default_held_out - 1e-9
    ]
    k1, b = whole_pick
    tuned = rung(
        "C",
        "+ k1/b tuned",
        f"{label}k1={k1} b={b}",
        BM25Retriever(chunks, k1=k1, b=b, tokenizer=tokenizer),
        shortcut,
        gold,
        index,
        against=incumbent,
        extra=lambda p95: list(cv_failures),
    )
    if tuned.verdict.adopted:
        incumbent = tuned
    else:
        k1, b = DEFAULTS

    lexical = BM25Retriever(chunks, k1=k1, b=b, tokenizer=tokenizer)
    feedback = rung(
        "D",
        "+ RM3 feedback",
        f"{label}k1={k1} b={b} + rm3",
        FeedbackRetriever(lexical),
        shortcut,
        gold,
        index,
        against=incumbent,
        extra=lambda p95: (
            [f"p95 {p95:.1f} ms exceeds {LATENCY_BUDGET_MS:.0f} ms"]
            if p95 > LATENCY_BUDGET_MS
            else []
        ),
    )
    if feedback.verdict.adopted:
        incumbent = feedback

    report = LadderReport(
        corpus_version=manifest.corpus_version,
        chunk_count=len(chunks),
        gold_count=len(gold),
        primary_k=PRIMARY_K,
        digit_audit=audit_digits(chunks),
        grid=grid,
        folds=folds,
        whole_pick=whole_pick,
        rungs=(base, grouped, tuned, feedback),
        adopted=f"rung {incumbent.rung}: `{incumbent.config}` + citation shortcut",
    )

    artifact_dir = settings.evals_dir / "reports" / manifest.corpus_version
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

    for r in report.rungs:
        overall = r.shortcut.primary.overall[CreditMode.LENIENT]
        verdict = "baseline" if r.verdict is None else (
            "adopted" if r.verdict.adopted else "rejected"
        )
        print(
            f"{r.rung} {r.config}: lenient recall@{PRIMARY_K} {overall.recall:.3f}, "
            f"nDCG {overall.ndcg:.3f}, p95 {r.p95_ms:.1f} ms: {verdict}"
        )
    print(f"adopted: {report.adopted}")
    print(f"wrote {artifact}")
    print(f"wrote {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
