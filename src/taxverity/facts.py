"""Step 7.5 — the structured user-fact schema.

What the extraction step (7.6) must produce, what its eval (7.7) counts, what
Phase 9 computes from and what Phase 11 merges across turns. Pure data and
deterministic normalisation: nothing here calls a model, and this module does
not import the LLM layer at all — it hands 7.6 a JSON schema and parses what
comes back.

Two measurements shape it. Step 7.1 found that value *formatting* is not stable
across runs — one input came back as "1400000" on one run and "14,00,000" on the
next — so normalisation is this module's job and not the prompt's. And the field
vocabulary is closed, because 7.7 cannot count recall against a set with no
members, Phase 11 cannot merge `salary` against `gross_salary`, and 9.7's
materiality probe has nothing to sweep without a declared domain.

Field names are lay-neutral and carry the **2025** Act's section as data. The
Act renumbers: the ₹1,50,000 savings-and-insurance deduction is section 123, not
the 1961 Act's 80C, and a field named `deduction_80c` would bake repealed
vocabulary into a system whose whole purpose is the current statute.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, model_validator

from taxverity.corpus.loader import normalise

FACTS_STAGE_VERSION = 4


class FactStatus(StrEnum):
    STATED = "stated"
    INFERRED = "inferred"
    MISSING = "missing"
    # Phase 11 only, and never assertable by a model: rule 04 makes silently
    # reusing a stale profile fact as `stated` a correctness bug, so the
    # distinction is unforgeable here rather than enforced by a later convention.
    PROFILE_DEFAULT = "profile_default"


MODEL_STATUSES = (FactStatus.STATED, FactStatus.INFERRED, FactStatus.MISSING)


class ValueKind(StrEnum):
    MONEY = "money"
    COUNT = "count"
    CHOICE = "choice"
    YEAR_RANGE = "year_range"


class Regime(StrEnum):
    OLD = "old"
    NEW = "new"


class ResidentialStatus(StrEnum):
    RESIDENT = "resident"
    RESIDENT_NOT_ORDINARILY_RESIDENT = "resident_not_ordinarily_resident"
    NON_RESIDENT = "non_resident"


class FactField(StrEnum):
    TAX_YEAR = "tax_year"
    REGIME = "regime"
    AGE = "age"
    RESIDENTIAL_STATUS = "residential_status"
    SALARY_INCOME = "salary_income"
    HOUSE_PROPERTY_INCOME = "house_property_income"
    BUSINESS_INCOME = "business_income"
    CAPITAL_GAINS_SHORT_TERM = "capital_gains_short_term"
    CAPITAL_GAINS_LONG_TERM = "capital_gains_long_term"
    OTHER_SOURCES_INCOME = "other_sources_income"
    DEDUCTION_SAVINGS_INSURANCE = "deduction_savings_insurance"
    DEDUCTION_HEALTH_INSURANCE = "deduction_health_insurance"
    DEDUCTION_OTHER = "deduction_other"
    TDS_PAID = "tds_paid"
    ADVANCE_TAX_PAID = "advance_tax_paid"


@dataclass(frozen=True)
class FieldSpec:
    kind: ValueKind
    description: str
    # The 2025 Act's own provision, where this project has already confirmed one
    # against the corpus. None means unverified, not absent — a guess here would
    # be a citation nobody checked.
    section: str | None = None
    choices: tuple[str, ...] = ()
    # A head of income can be a loss; money paid in never can.
    allows_negative: bool = False
    minimum: int | None = None
    maximum: int | None = None


FIELDS: dict[FactField, FieldSpec] = {
    FactField.TAX_YEAR: FieldSpec(
        ValueKind.YEAR_RANGE,
        "The tax year the question is about, as 2026-27: the financial year "
        "from 1 April. Copy the year as the user wrote it, even when they call "
        "it an assessment year.",
        section="3(1)",
    ),
    FactField.REGIME: FieldSpec(
        ValueKind.CHOICE,
        "Which rate regime the person is taxed under.",
        section="202(1)",
        choices=tuple(Regime),
    ),
    FactField.AGE: FieldSpec(
        ValueKind.COUNT,
        "Age in completed years.",
        minimum=0,
        maximum=120,
    ),
    FactField.RESIDENTIAL_STATUS: FieldSpec(
        ValueKind.CHOICE,
        "Residential status for the tax year.",
        section="6",
        choices=tuple(ResidentialStatus),
    ),
    FactField.SALARY_INCOME: FieldSpec(
        ValueKind.MONEY,
        "Gross salary or pension for the tax year, before deductions.",
        section="19",
    ),
    FactField.HOUSE_PROPERTY_INCOME: FieldSpec(
        ValueKind.MONEY,
        "Income from house property; negative for a loss.",
        section="21",
        allows_negative=True,
    ),
    FactField.BUSINESS_INCOME: FieldSpec(
        ValueKind.MONEY,
        "Income from business or profession; negative for a loss.",
        section="26",
        allows_negative=True,
    ),
    FactField.CAPITAL_GAINS_SHORT_TERM: FieldSpec(
        ValueKind.MONEY,
        "Short-term capital gains; negative for a loss.",
        section="67",
        allows_negative=True,
    ),
    FactField.CAPITAL_GAINS_LONG_TERM: FieldSpec(
        ValueKind.MONEY,
        "Long-term capital gains; negative for a loss.",
        section="67",
        allows_negative=True,
    ),
    FactField.OTHER_SOURCES_INCOME: FieldSpec(
        ValueKind.MONEY,
        "Income from other sources, such as interest or winnings.",
        section="92",
    ),
    FactField.DEDUCTION_SAVINGS_INSURANCE: FieldSpec(
        ValueKind.MONEY,
        "Amounts paid into savings and insurance sums that qualify for the "
        "capped deduction.",
        section="123",
    ),
    FactField.DEDUCTION_HEALTH_INSURANCE: FieldSpec(
        ValueKind.MONEY,
        "Health insurance premium and preventive check-up amounts paid.",
        section="126",
    ),
    FactField.DEDUCTION_OTHER: FieldSpec(
        ValueKind.MONEY,
        "Any other claimed deduction that is not one of the named ones.",
    ),
    FactField.TDS_PAID: FieldSpec(
        ValueKind.MONEY,
        "Tax already deducted at source.",
    ),
    FactField.ADVANCE_TAX_PAID: FieldSpec(
        ValueKind.MONEY,
        "Advance tax already paid for the tax year.",
        section="403",
    ),
}


class FactIssue(StrEnum):
    UNKNOWN_FIELD = "unknown_field"
    DUPLICATE_FIELD = "duplicate_field"
    BAD_STATUS = "bad_status"
    UNPARSABLE_VALUE = "unparsable_value"
    VALUE_OUT_OF_DOMAIN = "value_out_of_domain"
    MISSING_SPAN = "missing_span"
    SPAN_NOT_IN_TURN = "span_not_in_turn"
    SIGN_CONTRADICTS_SPAN = "sign_contradicts_span"
    MALFORMED_ENTRY = "malformed_entry"


class Fact(BaseModel):
    model_config = ConfigDict(frozen=True)

    field: FactField
    status: FactStatus
    # What the model wrote, kept verbatim beside the normalised value: 7.7's
    # eval reads it, and a repair prompt has to quote what was actually said.
    raw_value: str
    value: Decimal | int | str | None
    source_span: str

    @model_validator(mode="after")
    def _check(self) -> Fact:
        spec = FIELDS[self.field]
        if self.status is FactStatus.MISSING:
            if self.value is not None:
                raise ValueError("a missing fact carries no value")
            if self.source_span:
                raise ValueError("a missing fact points at no span")
            return self
        if self.value is None:
            raise ValueError(f"{self.status} fact for {self.field} carries no value")
        if self.status is FactStatus.STATED and not self.source_span:
            raise ValueError("a stated fact must quote the span it was read from")
        # A profile fact comes from another thread, so there is no span in this
        # turn for it to point at. Rule 04's distinction, enforced structurally.
        if self.status is FactStatus.PROFILE_DEFAULT and self.source_span:
            raise ValueError("a profile fact points at no span in this turn")
        _check_domain(spec, self.field, self.value)
        return self


class UnmappedFact(BaseModel):
    """A fact the model named outside the vocabulary, kept verbatim.

    Never reaches the calculator. It exists so a closed field set drops nothing
    silently: 7.7 can count what the vocabulary is missing, rather than the
    vocabulary quietly deciding what the user said.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    raw_value: str
    source_span: str


class UserFacts(BaseModel):
    model_config = ConfigDict(frozen=True)

    facts: tuple[Fact, ...] = ()
    unmapped: tuple[UnmappedFact, ...] = ()

    @model_validator(mode="after")
    def _one_per_field(self) -> UserFacts:
        seen = [fact.field for fact in self.facts]
        if len(set(seen)) != len(seen):
            raise ValueError("a field may be carried at most once")
        return self

    def get(self, field: FactField) -> Fact | None:
        for fact in self.facts:
            if fact.field is field:
                return fact
        return None

    def known(self) -> tuple[Fact, ...]:
        """Facts with a value. What Phase 9 computes from."""
        return tuple(
            fact for fact in self.facts if fact.status is not FactStatus.MISSING
        )

    def missing(self) -> tuple[FactField, ...]:
        """Fields the extractor reported as absent. What 9.7 probes for materiality."""
        return tuple(
            fact.field for fact in self.facts if fact.status is FactStatus.MISSING
        )


@dataclass(frozen=True)
class Rejection:
    """One entry the parser refused, and why. 7.6's repair prompt is built from these."""

    issue: FactIssue
    detail: str
    entry: dict[str, Any]


@dataclass(frozen=True)
class Extraction:
    facts: UserFacts
    rejections: tuple[Rejection, ...]


def parse_facts(payload: Any, turn: str) -> Extraction:
    """Turn a model payload into facts, refusing rather than raising.

    A single bad entry must not cost the whole turn, and 7.6 needs to know which
    entry to repair — so every refusal is returned, not thrown.
    """
    facts: list[Fact] = []
    unmapped: list[UnmappedFact] = []
    rejections: list[Rejection] = []
    normalised_turn = normalise(turn)
    seen: set[FactField] = set()

    for entry in _entries(payload, rejections):
        name = str(entry.get("name", "")).strip().casefold()
        raw_value = str(entry.get("value", "") or "")
        span = str(entry.get("source_span", "") or "")
        raw_status = str(entry.get("status", "") or "").strip().casefold()

        try:
            status = FactStatus(raw_status)
        except ValueError:
            rejections.append(
                Rejection(FactIssue.BAD_STATUS, f"unknown status {raw_status!r}", entry)
            )
            continue
        if status not in MODEL_STATUSES:
            rejections.append(
                Rejection(
                    FactIssue.BAD_STATUS,
                    f"{status} may not be asserted by a model",
                    entry,
                )
            )
            continue

        try:
            field = FactField(name)
        except ValueError:
            unmapped.append(
                UnmappedFact(name=name, raw_value=raw_value, source_span=span)
            )
            rejections.append(
                Rejection(FactIssue.UNKNOWN_FIELD, f"{name!r} is not a field", entry)
            )
            continue
        if field in seen:
            rejections.append(
                Rejection(FactIssue.DUPLICATE_FIELD, f"{field} given twice", entry)
            )
            continue

        if status is FactStatus.MISSING:
            facts.append(
                Fact(
                    field=field,
                    status=status,
                    raw_value="",
                    value=None,
                    source_span="",
                )
            )
            seen.add(field)
            continue

        # Told apart from a fabricated span on purpose: 7.6 repairs the two
        # differently — one turn was quoted badly, the other was not quoted.
        if status is FactStatus.STATED and not span:
            rejections.append(
                Rejection(
                    FactIssue.MISSING_SPAN,
                    f"{field} is stated but quotes nothing",
                    entry,
                )
            )
            continue

        # A span the turn does not contain is a fabricated quotation, which is
        # the same failure the Phase 10 verifier exists to catch. The fact is
        # refused rather than downgraded: a citation nobody can check is worse
        # than a field nobody extracted.
        if span and normalise(span) not in normalised_turn:
            rejections.append(
                Rejection(
                    FactIssue.SPAN_NOT_IN_TURN, f"{span!r} is not in the turn", entry
                )
            )
            continue

        value = fact_value(field, raw_value, span)
        if value is None:
            rejections.append(
                Rejection(
                    FactIssue.UNPARSABLE_VALUE,
                    f"{raw_value!r} is not a {FIELDS[field].kind}",
                    entry,
                )
            )
            continue
        # Step 7.7 measured the model quoting "loss" and writing the amount
        # positive, which inverts the tax. Refused, never flipped: the words can
        # sit beside a genuine positive figure ("no loss this time"), and a
        # refusal costs a repair call where a wrong flip costs a wrong answer.
        if (
            FIELDS[field].allows_negative
            and isinstance(value, Decimal)
            and value > 0
            and _LOSS_WORDS.search(normalise(span))
        ):
            rejections.append(
                Rejection(
                    FactIssue.SIGN_CONTRADICTS_SPAN,
                    f"{field} is positive but {span!r} describes a loss",
                    entry,
                )
            )
            continue
        try:
            fact = Fact(
                field=field,
                status=status,
                raw_value=raw_value,
                value=value,
                source_span=span,
            )
        except ValueError as error:
            rejections.append(
                Rejection(FactIssue.VALUE_OUT_OF_DOMAIN, str(error), entry)
            )
            continue
        facts.append(fact)
        seen.add(field)

    return Extraction(
        facts=UserFacts(facts=tuple(facts), unmapped=tuple(unmapped)),
        rejections=tuple(rejections),
    )


_LOSS_WORDS = re.compile(r"\b(?:loss(?:es)?|lost|minus|negative|deficit)\b", re.IGNORECASE)

_MULTIPLIERS = {
    "lakh": 100_000,
    "lakhs": 100_000,
    "crore": 10_000_000,
    "crores": 10_000_000,
    "cr": 10_000_000,
    "k": 1_000,
}
_MONEY = re.compile(
    r"^(?P<sign>-)?\s*(?:rs\.?|inr|₹)?\s*(?P<number>[\d,]*\d(?:\.\d+)?)\s*"
    r"(?P<multiplier>lakhs?|crores?|cr|k)?$"
)
_ASSESSMENT_YEAR_MARKER = r"(?:\bassessment\s+year|\ba\.?\s?y\.?)\s*(?:of\s+|:\s*)?"
_YEAR_RANGE = re.compile(
    r"^(?:a\.?y\.?\s*)?(?P<start>\d{4})\s*[-/]\s*(?P<end>\d{2,4})$"
)


def fact_value(field: FactField, raw_value: str, span: str) -> Decimal | int | str | None:
    """The value a fact carries, from what the model wrote and the words it quoted.

    Recomputed from both wherever a fact is rebuilt, so a stored run re-derives
    exactly the value the parser produced.
    """
    value = normalise_value(FIELDS[field].kind, raw_value)
    # The 2025 Act has no assessment year: section 3(1) makes the tax year the
    # financial year itself. In the repealed Act's usage assessment year 2026-27
    # is the financial year 2025-26, so taking the figure literally would put a
    # question one year late — and possibly under the wrong statute. Shifted in
    # code, not by the model, because the rule is arithmetic.
    #
    # The marker must name this figure, not merely appear nearby: "FY 2025-26
    # (AY 2026-27)" gives 2025-26 as a tax year, and shifting it because the span
    # also says "AY" would move the question a year early.
    if (
        isinstance(value, str)
        and FIELDS[field].kind is ValueKind.YEAR_RANGE
        and _names_assessment_year(value[:4], raw_value, span)
    ):
        start = int(value[:4]) - 1
        value = f"{start}-{(start + 1) % 100:02d}"
    return value


def _names_assessment_year(start: str, *texts: str) -> bool:
    marked = re.compile(rf"{_ASSESSMENT_YEAR_MARKER}{start}\s*[-/]")
    return any(marked.search(normalise(text).casefold()) for text in texts)


def normalise_value(kind: ValueKind, raw: str) -> Decimal | int | str | None:
    """Step 7.1 measured that the model's formatting is not stable. This is the fix."""
    text = normalise(raw).strip().casefold()
    if not text:
        return None
    if kind is ValueKind.MONEY:
        return _money(text)
    if kind is ValueKind.COUNT:
        return _count(text)
    if kind is ValueKind.YEAR_RANGE:
        return _year_range(text)
    return text.replace(" ", "_").replace("-", "_")


def _money(text: str) -> Decimal | None:
    match = _MONEY.match(text)
    if match is None:
        return None
    # Decimal from the digit string, never through float: a binary intermediate
    # is exactly what the calculator's Decimal rule exists to keep out.
    digits = match.group("number").replace(",", "")
    if not digits or digits.startswith("."):
        return None
    try:
        amount = Decimal(digits)
    except InvalidOperation:
        return None
    multiplier = match.group("multiplier")
    if multiplier:
        amount *= _MULTIPLIERS[multiplier]
    return -amount if match.group("sign") else amount


def _count(text: str) -> int | None:
    digits = text.replace(",", "")
    if not digits.isdigit():
        return None
    return int(digits)


def _year_range(text: str) -> str | None:
    match = _YEAR_RANGE.match(text.replace(" ", ""))
    if match is None:
        return None
    start, end = match.group("start"), match.group("end")
    end = end[-2:]
    if (int(start) + 1) % 100 != int(end):
        return None
    return f"{start}-{end}"


def _check_domain(spec: FieldSpec, field: FactField, value: object) -> None:
    if spec.kind is ValueKind.MONEY:
        if not isinstance(value, Decimal):
            raise ValueError(f"{field} takes an amount")
        if value < 0 and not spec.allows_negative:
            raise ValueError(f"{field} cannot be negative")
    elif spec.kind is ValueKind.COUNT:
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(f"{field} takes a whole number")
        if spec.minimum is not None and value < spec.minimum:
            raise ValueError(f"{field} below {spec.minimum}")
        if spec.maximum is not None and value > spec.maximum:
            raise ValueError(f"{field} above {spec.maximum}")
    elif spec.kind is ValueKind.CHOICE:
        if value not in spec.choices:
            raise ValueError(f"{field} must be one of {', '.join(spec.choices)}")
    elif spec.kind is ValueKind.YEAR_RANGE and not _YEAR_RANGE.match(str(value)):
        raise ValueError(f"{field} must read like 2026-27")


def _entries(payload: Any, rejections: list[Rejection]) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        rejections.append(
            Rejection(FactIssue.MALFORMED_ENTRY, "payload is not an object", {})
        )
        return []
    raw = payload.get("fields")
    if not isinstance(raw, list):
        rejections.append(
            Rejection(FactIssue.MALFORMED_ENTRY, "fields is not a list", {})
        )
        return []
    entries: list[dict[str, Any]] = []
    for entry in raw:
        if isinstance(entry, dict):
            entries.append(entry)
        else:
            rejections.append(
                Rejection(FactIssue.MALFORMED_ENTRY, "entry is not an object", {})
            )
    return entries


def _field_description() -> str:
    return " ".join(
        f"{field.value}: {FIELDS[field].description}"
        + (f" (Act section {FIELDS[field].section}.)" if FIELDS[field].section else "")
        for field in FactField
    )


FACTS_SCHEMA_NAME = "user_facts"

# Handed to the model as `response_format`. `name` is an enum so strict mode
# refuses an invented field outright; `parse_facts` still handles one, because
# 7.1 measured `json_object` as the fallback mode and nothing is enforced there.
FACTS_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["fields"],
    "properties": {
        "fields": {
            "type": "array",
            "description": _field_description(),
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["name", "value", "status", "source_span"],
                "properties": {
                    "name": {"type": "string", "enum": [f.value for f in FactField]},
                    "value": {
                        "type": "string",
                        "description": (
                            "Digits only for an amount, with no separators, "
                            "currency symbol or words. Empty when status is "
                            "missing."
                        ),
                    },
                    "status": {
                        "type": "string",
                        "enum": [s.value for s in MODEL_STATUSES],
                    },
                    "source_span": {
                        "type": "string",
                        "description": (
                            "Copied verbatim from the user's own words. Empty "
                            "unless the status is stated."
                        ),
                    },
                },
            },
        }
    },
}
