"""Step 7.7 — the extraction eval: how well does 7.6's node read a turn?

Two numbers rather than one, for the same reason ADR-060 reports retrieval
strict and lenient: field-level precision and recall say whether the node
*noticed* a fact, and value accuracy says whether it copied it correctly. A
single rate hides which of the two failed, and they are repaired by different
changes — the first by the prompt, the second by `normalise_value`.

Nothing here calls a model. The script runs the node and stores what came back;
this module judges a stored run, so a scoring change costs no tokens.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from taxverity.facts import (
    FIELDS,
    Fact,
    FactField,
    FactIssue,
    FactStatus,
    Rejection,
    UnmappedFact,
    UserFacts,
    fact_value,
    normalise_value,
)
from taxverity.llm.client import Completion
from taxverity.llm.extract import ExtractionResult
from taxverity.observability import get_logger

logger = get_logger(__name__)

EXTRACTION_EVAL_VERSION = 2

GOLD_FILENAME = "extraction_gold_v1.jsonl"
# Written and frozen before the loss-sign fix touched the prompt or the parser,
# so the re-measure is not scored on the turns the fix was aimed at (ADR-098).
LOSS_HOLDOUT_FILENAME = "extraction_loss_holdout_v1.jsonl"
TURN_ID = re.compile(r"^t\d{3}$")


class TurnSlice(StrEnum):
    SIMPLE = "simple"
    FORMATTING = "formatting"
    INFERRED = "inferred"
    LOSS = "loss"
    MULTI = "multi"
    NONE = "none"
    OUT_OF_VOCABULARY = "out_of_vocabulary"
    PII = "pii"


class LabelledFact(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: FactField
    status: FactStatus
    # Written as the user would, normalised on the way in: the label must be the
    # value the node is expected to produce, not the spelling it came in as.
    value: str

    @model_validator(mode="after")
    def _label_is_reachable(self) -> LabelledFact:
        if self.status not in (FactStatus.STATED, FactStatus.INFERRED):
            raise ValueError(f"a labelled fact cannot be {self.status}")
        if normalise_value(FIELDS[self.name].kind, self.value) is None:
            raise ValueError(f"{self.value!r} does not normalise for {self.name}")
        return self

    @property
    def normalised(self) -> Decimal | int | str:
        value = normalise_value(FIELDS[self.name].kind, self.value)
        if value is None:
            raise ValueError(f"{self.value!r} does not normalise for {self.name}")
        return value


class LabelledTurn(BaseModel):
    model_config = ConfigDict(frozen=True)

    turn_id: str
    slice: TurnSlice
    turn: str
    facts: tuple[LabelledFact, ...]
    notes: str

    @field_validator("turn_id")
    @classmethod
    def _ids_are_ordinals(cls, value: str) -> str:
        if not TURN_ID.match(value):
            raise ValueError(f"turn_id {value!r} is not of the form t001")
        return value

    @field_validator("turn", "notes")
    @classmethod
    def _prose_is_present(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("turn and notes are both required")
        return value

    @model_validator(mode="after")
    def _one_label_per_field(self) -> LabelledTurn:
        names = [fact.name for fact in self.facts]
        if len(set(names)) != len(names):
            raise ValueError(f"{self.turn_id} labels a field twice")
        return self

    @property
    def expected(self) -> dict[FactField, LabelledFact]:
        return {fact.name: fact for fact in self.facts}


def load_extraction_gold(
    directory: Path, filename: str = GOLD_FILENAME
) -> tuple[LabelledTurn, ...]:
    path = directory / filename
    turns = tuple(
        LabelledTurn.model_validate(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    expected = [f"t{index:03d}" for index in range(1, len(turns) + 1)]
    if [turn.turn_id for turn in turns] != expected:
        raise ValueError("turn ids must be contiguous ordinals from t001")
    return turns


@dataclass(frozen=True)
class Counts:
    """Detection only: did the node find this field at all."""

    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0

    @property
    def precision(self) -> float:
        found = self.true_positives + self.false_positives
        return self.true_positives / found if found else 0.0

    @property
    def recall(self) -> float:
        real = self.true_positives + self.false_negatives
        return self.true_positives / real if real else 0.0

    @property
    def f1(self) -> float:
        total = self.precision + self.recall
        return 2 * self.precision * self.recall / total if total else 0.0

    def __add__(self, other: Counts) -> Counts:
        return Counts(
            self.true_positives + other.true_positives,
            self.false_positives + other.false_positives,
            self.false_negatives + other.false_negatives,
        )


@dataclass(frozen=True)
class TurnJudgement:
    turn_id: str
    slice: TurnSlice
    counts: Counts
    found: tuple[FactField, ...]
    missed: tuple[FactField, ...]
    spurious: tuple[FactField, ...]
    value_wrong: tuple[tuple[FactField, str, str], ...]
    status_wrong: tuple[tuple[FactField, str, str], ...]
    # Found, with the right value and the right status. The single summary
    # number, reported beside the two diagnostic ones rather than instead.
    strict: tuple[FactField, ...]
    # Refused by parse_facts, so no fabricated span ever reaches a fact. The
    # rate is of attempts, which is the only honest denominator.
    fabricated_spans: int
    unquoted: int
    stated_attempts: int
    repaired: bool
    repair_helped: bool
    tokens: int

    @property
    def value_checked(self) -> int:
        return len(self.found)

    @property
    def value_correct(self) -> int:
        return len(self.found) - len(self.value_wrong)

    @property
    def status_correct(self) -> int:
        return len(self.found) - len(self.status_wrong)


@dataclass(frozen=True)
class ExtractionScore:
    turns: tuple[TurnJudgement, ...]
    counts: Counts
    per_slice: dict[TurnSlice, Counts]
    per_field: dict[FactField, Counts]

    @property
    def value_accuracy(self) -> float:
        checked = sum(turn.value_checked for turn in self.turns)
        correct = sum(turn.value_correct for turn in self.turns)
        return correct / checked if checked else 0.0

    @property
    def status_accuracy(self) -> float:
        checked = sum(turn.value_checked for turn in self.turns)
        correct = sum(turn.status_correct for turn in self.turns)
        return correct / checked if checked else 0.0

    @property
    def strict_rate(self) -> float:
        """Field, value and status all right, over every labelled fact."""
        labelled = self.counts.true_positives + self.counts.false_negatives
        strict = sum(len(turn.strict) for turn in self.turns)
        return strict / labelled if labelled else 0.0

    @property
    def fabricated_span_rate(self) -> float:
        attempts = sum(turn.stated_attempts for turn in self.turns)
        fabricated = sum(turn.fabricated_spans for turn in self.turns)
        return fabricated / attempts if attempts else 0.0

    @property
    def clean_turns(self) -> int:
        return sum(
            1
            for turn in self.turns
            if not (turn.missed or turn.spurious or turn.value_wrong)
        )

    @property
    def repaired_turns(self) -> int:
        return sum(1 for turn in self.turns if turn.repaired)

    @property
    def repairs_that_helped(self) -> int:
        return sum(1 for turn in self.turns if turn.repair_helped)

    @property
    def tokens(self) -> int:
        return sum(turn.tokens for turn in self.turns)


def judge_turn(label: LabelledTurn, result: ExtractionResult) -> TurnJudgement:
    expected = label.expected
    # MISSING is the complement of what was reported, not a prediction: counting
    # it would make precision a statement about the vocabulary's size.
    predicted = {fact.field: fact for fact in result.facts.known()}

    found = tuple(
        field for field in FactField if field in expected and field in predicted
    )
    missed = tuple(
        field for field in FactField if field in expected and field not in predicted
    )
    spurious = tuple(
        field for field in FactField if field in predicted and field not in expected
    )

    value_wrong = tuple(
        (field, str(expected[field].normalised), str(predicted[field].value))
        for field in found
        if predicted[field].value != expected[field].normalised
    )
    status_wrong = tuple(
        (field, expected[field].status.value, predicted[field].status.value)
        for field in found
        if predicted[field].status is not expected[field].status
    )
    wrong = {field for field, *_ in value_wrong} | {field for field, *_ in status_wrong}

    fabricated = _count(result, FactIssue.SPAN_NOT_IN_TURN)
    unquoted = _count(result, FactIssue.MISSING_SPAN)
    stated_kept = sum(
        1 for fact in predicted.values() if fact.status is FactStatus.STATED
    )
    survived = result.rejections
    return TurnJudgement(
        turn_id=label.turn_id,
        slice=label.slice,
        counts=Counts(len(found), len(spurious), len(missed)),
        found=found,
        missed=missed,
        spurious=spurious,
        value_wrong=value_wrong,
        status_wrong=status_wrong,
        strict=tuple(field for field in found if field not in wrong),
        fabricated_spans=fabricated,
        unquoted=unquoted,
        stated_attempts=stated_kept + fabricated + unquoted,
        repaired=result.repaired,
        repair_helped=any(rejection not in survived for rejection in result.repairable),
        tokens=result.tokens,
    )


def judge_extraction(
    labels: Sequence[LabelledTurn], run: Mapping[str, ExtractionResult]
) -> ExtractionScore:
    judgements: list[TurnJudgement] = []
    for label in labels:
        if label.turn_id not in run:
            # Same reason ADR-060 refuses a missing run entry: an absent turn
            # scores exactly like a turn the node got wrong, and it is not.
            raise KeyError(f"{label.turn_id} is not in the run")
        judgements.append(judge_turn(label, run[label.turn_id]))

    per_slice: dict[TurnSlice, Counts] = {}
    per_field: dict[FactField, Counts] = {}
    for judgement in judgements:
        per_slice[judgement.slice] = per_slice.get(judgement.slice, Counts()) + (
            judgement.counts
        )
        for field in judgement.found:
            per_field[field] = per_field.get(field, Counts()) + Counts(1, 0, 0)
        for field in judgement.spurious:
            per_field[field] = per_field.get(field, Counts()) + Counts(0, 1, 0)
        for field in judgement.missed:
            per_field[field] = per_field.get(field, Counts()) + Counts(0, 0, 1)

    total = Counts()
    for judgement in judgements:
        total = total + judgement.counts
    return ExtractionScore(
        turns=tuple(judgements),
        counts=total,
        per_slice=per_slice,
        per_field=per_field,
    )


def _count(result: ExtractionResult, issue: FactIssue) -> int:
    """Attempts, not survivors: parse_facts drops a fabricated span, so the
    surviving facts can never show one and the rejections are the only record."""
    seen = list(result.rejections) + [
        rejection
        for rejection in result.repairable
        if rejection not in result.rejections
    ]
    return sum(1 for rejection in seen if rejection.issue is issue)


def store_run(path: Path, run: Mapping[str, ExtractionResult]) -> None:
    payload = {
        "eval_version": EXTRACTION_EVAL_VERSION,
        "turns": {turn_id: _result_json(result) for turn_id, result in run.items()},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(
            json.dumps(payload, sort_keys=True, ensure_ascii=False, indent=1) + "\n"
        )


def load_run(path: Path) -> dict[str, ExtractionResult]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("eval_version") != EXTRACTION_EVAL_VERSION:
        raise ValueError(
            f"{path.name} was written by eval version "
            f"{payload.get('eval_version')}, not {EXTRACTION_EVAL_VERSION}"
        )
    return {
        turn_id: _result_from_json(entry) for turn_id, entry in payload["turns"].items()
    }


def _result_json(result: ExtractionResult) -> dict[str, object]:
    return {
        "facts": [
            {
                "field": fact.field.value,
                "status": fact.status.value,
                # The value is not stored: it is re-derived from raw_value by
                # the same deterministic normalisation that produced it, which
                # keeps a Decimal a Decimal across a JSON round trip.
                "raw_value": fact.raw_value,
                "source_span": fact.source_span,
            }
            for fact in result.facts.facts
        ],
        "unmapped": [fact.model_dump(mode="json") for fact in result.facts.unmapped],
        "rejections": [_rejection_json(r) for r in result.rejections],
        "repairable": [_rejection_json(r) for r in result.repairable],
        "repaired": result.repaired,
        "completions": [c.model_dump(mode="json") for c in result.completions],
    }


def _rejection_json(rejection: Rejection) -> dict[str, object]:
    return {
        "issue": rejection.issue.value,
        "detail": rejection.detail,
        "entry": rejection.entry,
    }


def _result_from_json(entry: Mapping[str, object]) -> ExtractionResult:
    facts = []
    for stored in entry["facts"]:  # type: ignore[index]
        field = FactField(stored["field"])
        facts.append(
            Fact(
                field=field,
                status=FactStatus(stored["status"]),
                raw_value=stored["raw_value"],
                value=fact_value(field, stored["raw_value"], stored["source_span"]),
                source_span=stored["source_span"],
            )
        )
    return ExtractionResult(
        facts=UserFacts(
            facts=tuple(facts),
            unmapped=tuple(
                UnmappedFact.model_validate(u)
                for u in entry["unmapped"]  # type: ignore[union-attr]
            ),
        ),
        rejections=tuple(_rejection_from_json(r) for r in entry["rejections"]),  # type: ignore[union-attr]
        repairable=tuple(_rejection_from_json(r) for r in entry["repairable"]),  # type: ignore[union-attr]
        repaired=bool(entry["repaired"]),
        completions=tuple(
            Completion.model_validate(c)
            for c in entry["completions"]  # type: ignore[union-attr]
        ),
    )


def _rejection_from_json(entry: Mapping[str, object]) -> Rejection:
    return Rejection(
        issue=FactIssue(entry["issue"]),
        detail=str(entry["detail"]),
        entry=dict(entry["entry"]),  # type: ignore[arg-type]
    )
