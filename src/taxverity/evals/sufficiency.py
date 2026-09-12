"""Step 8.1 — the evidence-sufficiency grader's ladder and adoption rule,
registered before `scripts/measure_sufficiency.py` first ran (ADR-099).

Ground truth is the Step 5.7 oracle: an answerable question's pack is
insufficient when some gold label is not credited leniently (ADR-085). A
negative carries no labels, so it is neither caught nor missed. How often the
grader fires on one is reported as exposure to false rescue (ADR-036), never
counted as a success.

The ladder is cumulative, simplest first:

1. relevance: the reranker's best score below a threshold;
2. unmet: rung 1, or a reference from one of the first N retrieved units to a
   provision the pack lacks, N in UNMET_DEPTHS;
3. citation: rung 2, overridden to sufficient by a citation-shortcut hit.

Each rung's settings are chosen on one parity fold and scored on the other
(ADR-077's split), so every question is graded by settings that never saw it.
A rung passes when, pooled over both held-out folds, it flags at least half the
insufficient answerable questions and at most a tenth of the sufficient ones.
The simplest passing rung is adopted. A later rung replaces it only by also
passing, catching strictly more and flagging no more sufficient questions.
When none passes, the grader is not adopted, and escalating to an LLM grader
is a separate decision, since it would put a model call on the query path.
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum

from pydantic import BaseModel, ConfigDict

from taxverity.evals.gold import QuerySlice
from taxverity.evals.ladder import Verdict
from taxverity.retrieval.sufficiency import GraderConfig, Signals, Sufficiency, grade

UNMET_DEPTHS = (1, 3)
MIN_CATCH_SHARE = 0.5
MAX_FALSE_SHARE = 0.1


class Rung(StrEnum):
    RELEVANCE = "relevance"
    UNMET = "unmet"
    CITATION = "citation"


class Case(BaseModel):
    model_config = ConfigDict(frozen=True)

    query_id: str
    slice: QuerySlice
    signals: Signals
    # Gold labels the pack does not credit leniently. Empty for a negative.
    missed: tuple[str, ...]

    @property
    def answerable(self) -> bool:
        return self.slice is not QuerySlice.NEGATIVE

    @property
    def insufficient(self) -> bool:
        return self.answerable and bool(self.missed)


class Tally(BaseModel):
    model_config = ConfigDict(frozen=True)

    caught: int
    failing: int
    false: int
    sufficient: int
    negatives_flagged: int
    negatives: int

    @property
    def objective(self) -> float:
        catch = self.caught / self.failing if self.failing else 0.0
        false = self.false / self.sufficient if self.sufficient else 0.0
        return catch - false

    @property
    def passes(self) -> bool:
        return (
            self.failing > 0
            and self.caught >= MIN_CATCH_SHARE * self.failing
            and self.false <= MAX_FALSE_SHARE * self.sufficient
        )


class RungResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    rung: Rung
    # Settings chosen on the odd fold (grading the even one), then the even.
    fold_configs: tuple[GraderConfig, GraderConfig]
    held_out: Tally
    flagged: tuple[str, ...]
    # Settings chosen on every answerable question: what would ship.
    shipped: GraderConfig


def flags(case: Case, config: GraderConfig) -> bool:
    return grade(case.signals, config).sufficiency is Sufficiency.INSUFFICIENT


def tally(cases: Sequence[Case], flagged: set[str]) -> Tally:
    answerable = [c for c in cases if c.answerable]
    negatives = [c for c in cases if not c.answerable]
    return Tally(
        caught=sum(1 for c in answerable if c.insufficient and c.query_id in flagged),
        failing=sum(1 for c in answerable if c.insufficient),
        false=sum(1 for c in answerable if not c.insufficient and c.query_id in flagged),
        sufficient=sum(1 for c in answerable if not c.insufficient),
        negatives_flagged=sum(1 for c in negatives if c.query_id in flagged),
        negatives=len(negatives),
    )


def candidates(rung: Rung, cases: Sequence[Case]) -> list[GraderConfig]:
    """Thresholds are the training fold's own answerable scores, plus None for
    the rule unused, so rung 1's "never fires" is in every rung's grid."""
    scores = sorted(
        {c.signals.top_relevance for c in cases if c.answerable and c.signals.top_relevance is not None}
    )
    thresholds: list[float | None] = [None, *scores]
    if rung is Rung.RELEVANCE:
        return [GraderConfig(min_relevance=t) for t in thresholds]
    return [
        GraderConfig(min_relevance=t, unmet_within=n, citation_override=rung is Rung.CITATION)
        for t in thresholds
        for n in UNMET_DEPTHS
    ]


def select(rung: Rung, cases: Sequence[Case]) -> GraderConfig:
    """Best objective on the answerable cases. A tie goes to the settings that
    flag fewer answerable questions, then the lower threshold, then the shallower
    depth, so the pick fires least and never depends on grid order."""
    answerable = [c for c in cases if c.answerable]

    def key(config: GraderConfig) -> tuple:
        flagged = {c.query_id for c in answerable if flags(c, config)}
        return (
            -tally(answerable, flagged).objective,
            len(flagged),
            config.min_relevance is not None,
            config.min_relevance or 0.0,
            config.unmet_within or 0,
        )

    return min(candidates(rung, answerable), key=key)


def fold_of(query_id: str) -> int:
    return int(query_id[1:]) % 2


def cross_validate(rung: Rung, cases: Sequence[Case]) -> RungResult:
    configs = []
    flagged: set[str] = set()
    for held in (0, 1):
        training = [c for c in cases if fold_of(c.query_id) != held]
        config = select(rung, training)
        configs.append(config)
        flagged |= {c.query_id for c in cases if fold_of(c.query_id) == held and flags(c, config)}
    return RungResult(
        rung=rung,
        fold_configs=(configs[0], configs[1]),
        held_out=tally(cases, flagged),
        flagged=tuple(sorted(flagged)),
        shipped=select(rung, cases),
    )


def judge_sufficiency(results: Sequence[RungResult]) -> tuple[Rung | None, Verdict]:
    if [r.rung for r in results] != list(Rung):
        raise ValueError("every rung, in ladder order")
    adopted: RungResult | None = None
    reasons: list[str] = []
    for result in results:
        held = result.held_out
        if not held.passes:
            reasons.append(
                f"{result.rung.value}: caught {held.caught}/{held.failing}, "
                f"flagged {held.false}/{held.sufficient} sufficient"
            )
            continue
        if adopted is None:
            adopted = result
        elif held.caught > adopted.held_out.caught and held.false <= adopted.held_out.false:
            adopted = result
    if adopted is None:
        return None, Verdict(adopted=False, reasons=tuple(reasons))
    return adopted.rung, Verdict(adopted=True, reasons=tuple(reasons))
