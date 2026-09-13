"""Step 10.7 — the answer smoke set: a stored run of the generator over a
handful of gold-v2 questions (ADR-110).

The only gate is that no negative question gets a served statute claim. The
rest is reported so a regression is visible, not judged against a threshold.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict, TypeAdapter

from taxverity.evals.gold import QuerySlice
from taxverity.generation.claims import ClaimEvent, ClaimType, WithheldEvent

ANSWER_EVAL_VERSION = 1
ANSWER_RUN_FILENAME = "answer_smoke_v1.json"

# Chosen by rule, not by reading outputs: the first three citation, four
# paraphrase and three crossref questions of gold v2, and its first five
# negatives. Phase 11 appends its follow-up cases.
SMOKE_QUERY_IDS: tuple[str, ...] = (
    "q001", "q002", "q003",
    "q009", "q010", "q011", "q012",
    "q019", "q020", "q021",
    "q025", "q026", "q027", "q028", "q029",
)  # fmt: skip


class EvidenceText(BaseModel):
    """A packed unit as the model saw it: ancestor lead-ins, then its text."""

    model_config = ConfigDict(frozen=True)

    citation: str
    text: str


class AnswerRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    query_id: str
    slice: QuerySlice
    question: str
    evidence: tuple[EvidenceText, ...]
    claims: tuple[ClaimEvent, ...]
    withheld: tuple[WithheldEvent, ...]
    # A provider failure ends the answer; what was already released stands.
    error: str | None = None
    seconds: float

    @property
    def statute_claims(self) -> tuple[ClaimEvent, ...]:
        return tuple(c for c in self.claims if c.type is ClaimType.STATUTE)

    def answer_text(self) -> str:
        return " ".join(claim.text for claim in self.claims)


class AnswerRun(BaseModel):
    model_config = ConfigDict(frozen=True)

    eval_version: int
    generation_stage_version: int
    prompt_version: int
    model: str
    tokens: int
    records: tuple[AnswerRecord, ...]


class SmokeSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    questions: int
    answered: int
    claims_served: int
    claims_withheld: int
    errors: int
    # The gate: negative questions that got at least one served statute claim.
    served_on_negative: tuple[str, ...]
    # Answerable questions that got no served claim at all. Reported only.
    silent_answerable: tuple[str, ...]


def summarise(records: Sequence[AnswerRecord]) -> SmokeSummary:
    negatives = [r for r in records if r.slice is QuerySlice.NEGATIVE]
    answerable = [r for r in records if r.slice is not QuerySlice.NEGATIVE]
    return SmokeSummary(
        questions=len(records),
        answered=sum(1 for r in answerable if r.claims),
        claims_served=sum(len(r.claims) for r in records),
        claims_withheld=sum(len(r.withheld) for r in records),
        errors=sum(1 for r in records if r.error is not None),
        served_on_negative=tuple(r.query_id for r in negatives if r.statute_claims),
        silent_answerable=tuple(r.query_id for r in answerable if not r.claims),
    )


def store_answer_run(path: Path, run: AnswerRun) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = run.model_dump(mode="json")
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(
            json.dumps(payload, sort_keys=True, ensure_ascii=False, indent=1) + "\n"
        )


def load_answer_run(path: Path) -> AnswerRun:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("eval_version") != ANSWER_EVAL_VERSION:
        raise ValueError(
            f"{path.name} was written by eval version "
            f"{payload.get('eval_version')}, not {ANSWER_EVAL_VERSION}"
        )
    return TypeAdapter(AnswerRun).validate_python(payload)


def by_query(run: AnswerRun) -> Mapping[str, AnswerRecord]:
    return {record.query_id: record for record in run.records}
