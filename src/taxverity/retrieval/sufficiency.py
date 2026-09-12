"""Step 8.1 — whether a delivered evidence pack is enough to answer, decided
from signals retrieval already produced: no model call, no gold labels (ADR-099).

Three signals, each a rung of the registered ladder in
`taxverity.evals.sufficiency`:

- the reranker's best relevance score over the pool, the one retrieval score on
  a single scale across questions (ADR-084);
- references a delivered unit makes to provisions the pack does not carry, the
  query-time trace of a dangling dependency (ADR-085);
- whether the citation shortcut fired, so the question named what it asks about.

The grader decides; it does not act. Step 8.2 routes an insufficient pack.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, model_validator

from taxverity.retrieval.evidence import EvidencePack, EvidencePacker

SUFFICIENCY_STAGE_VERSION = 1


class Sufficiency(StrEnum):
    SUFFICIENT = "sufficient"
    INSUFFICIENT = "insufficient"


class Reason(StrEnum):
    CITATION_HIT = "citation_hit"
    LOW_RELEVANCE = "low_relevance"
    UNMET_REFERENCE = "unmet_reference"
    # The reranker degraded to fusion order (ADR-052), so there is no score to
    # read. Recorded, never a failure: grading cannot fire the loop on nothing.
    NO_RELEVANCE = "no_relevance"


class Signals(BaseModel):
    model_config = ConfigDict(frozen=True)

    citation_hit: bool
    top_relevance: float | None
    # EvidencePacker.unmet_references: (position, unit, target).
    unmet: tuple[tuple[int, str, str], ...]


class GraderConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    # Insufficient when the best relevance score is below this. None: unused.
    min_relevance: float | None = None
    # Insufficient when one of the first this-many retrieved units references a
    # provision the pack lacks. None: unused.
    unmet_within: int | None = None
    # A citation-shortcut hit is sufficient whatever the other rules say.
    citation_override: bool = False

    @model_validator(mode="after")
    def a_depth_is_positive(self) -> GraderConfig:
        if self.unmet_within is not None and self.unmet_within < 1:
            raise ValueError(f"unmet_within must be at least 1, not {self.unmet_within}")
        return self


class Graded(BaseModel):
    model_config = ConfigDict(frozen=True)

    sufficiency: Sufficiency
    reasons: tuple[Reason, ...]


def signals_of(
    packer: EvidencePacker,
    pack: EvidencePack,
    *,
    citation_hit: bool,
    top_relevance: float | None,
) -> Signals:
    return Signals(
        citation_hit=citation_hit,
        top_relevance=top_relevance,
        unmet=packer.unmet_references(pack),
    )


def grade(signals: Signals, config: GraderConfig) -> Graded:
    if config.citation_override and signals.citation_hit:
        return Graded(sufficiency=Sufficiency.SUFFICIENT, reasons=(Reason.CITATION_HIT,))
    reasons: list[Reason] = []
    failed = False
    if config.min_relevance is not None:
        if signals.top_relevance is None:
            reasons.append(Reason.NO_RELEVANCE)
        elif signals.top_relevance < config.min_relevance:
            reasons.append(Reason.LOW_RELEVANCE)
            failed = True
    if config.unmet_within is not None and any(
        position <= config.unmet_within for position, _, _ in signals.unmet
    ):
        reasons.append(Reason.UNMET_REFERENCE)
        failed = True
    return Graded(
        sufficiency=Sufficiency.INSUFFICIENT if failed else Sufficiency.SUFFICIENT,
        reasons=tuple(reasons),
    )
