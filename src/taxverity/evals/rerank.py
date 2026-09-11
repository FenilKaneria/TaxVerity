"""Step 5.6 — the rule the reranker is held to, written before
`scripts/measure_rerank.py` first ran (ADR-051, ADR-084), and the stored gold
rerank scores that let it be re-measured and floor-tested without billing."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from taxverity.chunking.models import Chunk
from taxverity.evals.ladder import TOLERANCE, Verdict
from taxverity.evals.metrics import CreditMode, RunReport

# ADR-051's two build-blocking guards: nDCG at 5 against the un-reranked fusion
# order, and a p95 latency budget on the rerank call. The budget is argued from
# the streamed answer (a stalled evidence stage reads as a frozen screen), not
# from any measurement, and was fixed before the first paid run.
GUARD_K = 5
RERANK_P95_BUDGET_MS = 1_500.0

RERANK_SCORES_FILENAME = "gold_rerank_scores.json"


class StaleRerankScoresError(RuntimeError):
    pass


def judge_rerank(reranked: RunReport, fusion: RunReport, p95_ms: float) -> Verdict:
    """Adopted only if lenient nDCG@5 does not fall against the fusion order and
    the rerank call's p95 is within budget. No rise is required: ADR-051 made
    the reranker a scheduled build, and the guards ask only whether it is
    configured well."""
    if reranked.k != GUARD_K or fusion.k != GUARD_K:
        raise ValueError(f"the guard is nDCG@{GUARD_K}; got k={reranked.k} and k={fusion.k}")
    if {s.query_id for s in reranked.scored} != {s.query_id for s in fusion.scored}:
        raise ValueError("the two runs do not cover the same queries")
    reasons = []
    before = fusion.overall[CreditMode.LENIENT].ndcg
    after = reranked.overall[CreditMode.LENIENT].ndcg
    if after < before - TOLERANCE:
        reasons.append(f"lenient nDCG@{GUARD_K} fell {before:.3f} -> {after:.3f}")
    if p95_ms > RERANK_P95_BUDGET_MS:
        reasons.append(
            f"rerank p95 {p95_ms:.0f} ms exceeds the {RERANK_P95_BUDGET_MS:.0f} ms budget"
        )
    return Verdict(adopted=not reasons, reasons=tuple(reasons))


class RerankScores(BaseModel):
    model_config = ConfigDict(frozen=True)

    model_id: str
    # Scores are keyed by chunk id, and a chunk id moves with corpus_version.
    corpus_version: str
    depth: int
    # Keyed by the exact question text, then chunk id: an edited gold question
    # misses rather than reusing the scores of its old wording.
    scores: dict[str, dict[str, float]]
    # The live run's per-request latency and bill, so the verdict can be
    # reproduced offline rather than only on the day it was paid for.
    latencies_ms: tuple[float, ...]
    tokens_billed: int


def write_rerank_scores(path: Path, scores: RerankScores) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(
        scores.model_dump(mode="json"),
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(text + "\n")


def load_rerank_scores(
    path: Path,
    *,
    model_id: str,
    corpus_version: str,
    depth: int,
    questions: Sequence[str],
) -> RerankScores:
    """Refuses scores from another model, corpus or depth, or missing any
    question asked for."""
    loaded = RerankScores.model_validate_json(path.read_text(encoding="utf-8"))
    for name, want in (
        ("model_id", model_id),
        ("corpus_version", corpus_version),
        ("depth", depth),
    ):
        got = getattr(loaded, name)
        if got != want:
            raise StaleRerankScoresError(f"{path} was scored with {name}={got!r}, not {want!r}")
    missing = [q for q in questions if q not in loaded.scores]
    if missing:
        raise StaleRerankScoresError(
            f"{path} has no scores for {len(missing)} question(s), first {missing[0]!r}"
        )
    return loaded


class StoredReranker:
    """Satisfies `Reranker` from stored scores. A question or candidate not on
    file raises KeyError; it never reaches the network."""

    def __init__(self, scores: Mapping[str, Mapping[str, float]]) -> None:
        self._scores = scores

    def score(self, query: str, chunks: Sequence[Chunk]) -> dict[str, float]:
        stored = self._scores[query]
        return {chunk.chunk_id: stored[chunk.chunk_id] for chunk in chunks}
