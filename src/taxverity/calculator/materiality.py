"""Step 9.7 — which unknown facts are worth asking about, decided by arithmetic.

A field whose values are bounded is swept through the calculator, and it is
asked about only when the rounded tax differs across them. A field whose values
are not bounded cannot be shown not to matter, so each class gets a fixed
policy instead: income is asked, income under another head or an unnamed
deduction is asked because any non-zero amount changes the route, and tax
already paid leaves the balance uncomputed rather than guessed (ADR-108).
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum

from taxverity.calculator.rates import supported_tax_years
from taxverity.calculator.scope import (
    HEADS_NOT_COMPUTED,
    INPUT_FIELDS,
    CalculatorInputs,
    Route,
    ScopeDecision,
    inputs_from,
    run,
)
from taxverity.facts import FactField, ResidentialStatus, UserFacts

MATERIALITY_STAGE_VERSION = 1


class Outcome(StrEnum):
    ASK = "ask"
    ASSUME = "assume"
    # Not asked and not assumed: the balance is left out of the answer.
    NOT_COMPUTED = "not_computed"
    # Swept once the income it depends on is known, so it is not asked early.
    DEFERRED = "deferred"


class Reason(StrEnum):
    INCOME_UNBOUNDED = "income_unbounded"
    CHANGES_ROUTE = "changes_route"
    TAX_DIFFERS = "tax_differs"
    TAX_SAME = "tax_same"
    ONLY_TAX_YEAR_WITH_DATA = "only_tax_year_with_data"
    NOT_ALLOWED_UNDER_202_1 = "not_allowed_under_202_1"
    BALANCE_ONLY = "balance_only"
    WAITS_ON_INCOME = "waits_on_income"


INCOME_FIELDS = (FactField.SALARY_INCOME, FactField.OTHER_SOURCES_INCOME)
ROUTE_CHANGING_FIELDS = (*HEADS_NOT_COMPUTED, FactField.DEDUCTION_OTHER)
# Section 202(2)(a)(xii) excludes both from a 202(1) computation, so no amount
# can move its tax. The opted-out side's total income still assumes them nil.
CLAIM_FIELDS = (FactField.DEDUCTION_SAVINGS_INSURANCE, FactField.DEDUCTION_HEALTH_INSURANCE)
PAID_FIELDS = (FactField.TDS_PAID, FactField.ADVANCE_TAX_PAID)

_OUTCOMES = {
    Reason.INCOME_UNBOUNDED: Outcome.ASK,
    Reason.CHANGES_ROUTE: Outcome.ASK,
    Reason.TAX_DIFFERS: Outcome.ASK,
    Reason.TAX_SAME: Outcome.ASSUME,
    Reason.ONLY_TAX_YEAR_WITH_DATA: Outcome.ASSUME,
    Reason.NOT_ALLOWED_UNDER_202_1: Outcome.ASSUME,
    Reason.BALANCE_ONLY: Outcome.NOT_COMPUTED,
    Reason.WAITS_ON_INCOME: Outcome.DEFERRED,
}


@dataclass(frozen=True)
class Finding:
    field: FactField
    reason: Reason
    # The lowest and highest rounded tax payable across the field's values,
    # present exactly when the field was swept.
    spread: tuple[Decimal, Decimal] | None = None
    assumed: object = None
    # A profile default is surfaced for confirmation whatever the probe finds
    # (rule 04), so the answer needs to know which unknowns were one.
    unconfirmed_profile_default: bool = False

    def __post_init__(self) -> None:
        swept = self.reason in (Reason.TAX_DIFFERS, Reason.TAX_SAME)
        if swept != (self.spread is not None):
            raise ValueError("a spread is carried exactly when the field was swept")
        if self.spread is not None and (self.spread[0] != self.spread[1]) != (self.reason is Reason.TAX_DIFFERS):
            raise ValueError("a field is material exactly when its spread is non-zero")
        if (self.outcome is Outcome.ASSUME) != (self.assumed is not None):
            raise ValueError("only an assumed field carries an assumed value")

    @property
    def outcome(self) -> Outcome:
        return _OUTCOMES[self.reason]


@dataclass(frozen=True)
class Probe:
    findings: tuple[Finding, ...]
    inputs: CalculatorInputs | None

    def __post_init__(self) -> None:
        waiting = any(f.outcome in (Outcome.ASK, Outcome.DEFERRED) for f in self.findings)
        if waiting == (self.inputs is not None):
            raise ValueError("inputs are built exactly when nothing is asked or deferred")

    def by_outcome(self, outcome: Outcome) -> tuple[Finding, ...]:
        return tuple(f for f in self.findings if f.outcome is outcome)


def probe(facts: UserFacts, decision: ScopeDecision) -> Probe:
    if decision.route is Route.TEXT_ONLY:
        raise ValueError("a text-only decision has nothing to probe")
    if decision.route is Route.COMPUTE:
        return Probe(findings=(), inputs=decision.inputs)

    unknown = set(decision.unknown)
    # Only stated and inferred facts reach here as known; route() already
    # counted every other status as unknown.
    values = {name: facts.get(name).value for name in INPUT_FIELDS if name not in unknown}
    reasons: dict[FactField, Reason] = {}
    assumed: dict[FactField, object] = {}
    spreads: dict[FactField, tuple[Decimal, Decimal]] = {}

    for name in decision.unknown:
        if name in INCOME_FIELDS:
            reasons[name] = Reason.INCOME_UNBOUNDED
        elif name in ROUTE_CHANGING_FIELDS:
            reasons[name] = Reason.CHANGES_ROUTE
        elif name in CLAIM_FIELDS:
            reasons[name], assumed[name] = Reason.NOT_ALLOWED_UNDER_202_1, Decimal(0)
        elif name in PAID_FIELDS:
            reasons[name] = Reason.BALANCE_ONLY

    if FactField.TAX_YEAR in unknown:
        years = supported_tax_years()
        if len(years) != 1:
            raise NotImplementedError("a second tax year needs the tax year swept, not assumed")
        reasons[FactField.TAX_YEAR], assumed[FactField.TAX_YEAR] = Reason.ONLY_TAX_YEAR_WITH_DATA, years[0]

    if FactField.RESIDENTIAL_STATUS in unknown:
        if any(name in unknown for name in INCOME_FIELDS):
            reasons[FactField.RESIDENTIAL_STATUS] = Reason.WAITS_ON_INCOME
        else:
            spread = _sweep_residential_status({**values, **assumed})
            spreads[FactField.RESIDENTIAL_STATUS] = spread
            if spread[0] == spread[1]:
                # Either value gives the same figure. Non-resident is used so no
                # rebate line claims an eligibility nobody stated.
                reasons[FactField.RESIDENTIAL_STATUS] = Reason.TAX_SAME
                assumed[FactField.RESIDENTIAL_STATUS] = ResidentialStatus.NON_RESIDENT
            else:
                reasons[FactField.RESIDENTIAL_STATUS] = Reason.TAX_DIFFERS

    findings = tuple(
        Finding(
            field=name,
            reason=reasons[name],
            spread=spreads.get(name),
            assumed=assumed.get(name),
            unconfirmed_profile_default=name in decision.unconfirmed,
        )
        for name in decision.unknown
    )
    waiting = any(f.outcome in (Outcome.ASK, Outcome.DEFERRED) for f in findings)
    inputs = None if waiting else inputs_from({**values, **assumed})
    return Probe(findings=findings, inputs=inputs)


def _sweep_residential_status(values: dict[FactField, object]) -> tuple[Decimal, Decimal]:
    # Heads and the unnamed deduction are taken as nil only for the sweep: were
    # any non-zero, the question would route text-only and no tax would exist.
    base = {name: Decimal(0) for name in (*ROUTE_CHANGING_FIELDS, *CLAIM_FIELDS)} | values
    taxes = [
        run(inputs_from({**base, FactField.RESIDENTIAL_STATUS: status})).comparison.under_202_1.payable.amount
        for status in ResidentialStatus
    ]
    return min(taxes), max(taxes)

