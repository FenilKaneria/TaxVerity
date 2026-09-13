"""Step 9.6 — decide whether the calculator may answer, before it is asked.

The calculator computes salary and other income under section 202(1), the
section 123 and 126 claims, and the set-off of tax already paid. A fact it does
not model is never approximated: income under any other head, an unnamed
deduction, a tax year with no data, or a fact outside the extraction vocabulary
routes the question to a text-only answer. A calculator input that is not yet
known makes the decision incomplete, and Step 9.7 decides whether it is worth
asking (ADR-107).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from enum import StrEnum

from taxverity.calculator.comparison import RegimeComparison, compare_regimes
from taxverity.calculator.rates import OutsideAct, load_rates, supported_tax_years
from taxverity.calculator.settlement import Settlement, settle
from taxverity.facts import Fact, FactField, FactStatus, ResidentialStatus, UserFacts

SCOPE_STAGE_VERSION = 2


class Route(StrEnum):
    COMPUTE = "compute"
    TEXT_ONLY = "text_only"
    INCOMPLETE = "incomplete"


class OutOfScope(StrEnum):
    UNSUPPORTED_TAX_YEAR = "unsupported_tax_year"
    HEAD_NOT_COMPUTED = "head_not_computed"
    DEDUCTION_NOT_IDENTIFIED = "deduction_not_identified"
    UNMAPPED_FACT = "unmapped_fact"


# Income the calculator has no computation for. A stated nil is not income, so
# only a non-zero amount, gain or loss, routes out.
HEADS_NOT_COMPUTED = (
    FactField.HOUSE_PROPERTY_INCOME,
    FactField.BUSINESS_INCOME,
    FactField.CAPITAL_GAINS_SHORT_TERM,
    FactField.CAPITAL_GAINS_LONG_TERM,
)

# Neither changes a section 202(1) computation. The option under 202(4) is shown
# beside it whatever the person says, and age reaches only section 126, whose
# limit the comparison already leaves undetermined.
NOT_INPUTS = (FactField.REGIME, FactField.AGE)

INPUT_FIELDS = tuple(f for f in FactField if f not in NOT_INPUTS)

# Section 156 says "individual resident in India", which a resident but not
# ordinarily resident individual is. The vocabulary carries no person type, so
# the person is taken to be an individual.
_RESIDENT = {ResidentialStatus.RESIDENT, ResidentialStatus.RESIDENT_NOT_ORDINARILY_RESIDENT}


@dataclass(frozen=True)
class Blocker:
    reason: OutOfScope
    detail: str
    field: FactField | None = None


@dataclass(frozen=True)
class CalculatorInputs:
    tax_year: str
    salary: Decimal
    other_income: Decimal
    resident_individual: bool
    claimed: dict[str, Decimal]
    # None when not known. The tax is still computed, but no balance is: a
    # settlement needs everything paid, and a nil guess would be wrong for most
    # salaried people (ADR-108).
    tax_deducted_at_source: Decimal | None
    advance_tax: Decimal | None


@dataclass(frozen=True)
class ScopeDecision:
    route: Route
    blockers: tuple[Blocker, ...] = ()
    unknown: tuple[FactField, ...] = ()
    # Profile facts wait for the person to confirm them (rule 04), so they are
    # listed here and counted as unknown, never used.
    unconfirmed: tuple[FactField, ...] = ()
    inferred: tuple[FactField, ...] = ()
    inputs: CalculatorInputs | None = field(default=None)

    def __post_init__(self) -> None:
        if (self.route is Route.TEXT_ONLY) != bool(self.blockers):
            raise ValueError("a text-only route is exactly one that carries a blocker")
        if self.route is Route.INCOMPLETE and not self.unknown:
            raise ValueError("an incomplete route names what is unknown")
        if (self.route is Route.COMPUTE) != (self.inputs is not None):
            raise ValueError("only a computed route carries calculator inputs")
        if not set(self.unconfirmed) <= set(self.unknown):
            raise ValueError("an unconfirmed profile fact is still unknown")


@dataclass(frozen=True)
class Computation:
    comparison: RegimeComparison
    settlement: Settlement | None
    # The Act defers both to "the Central Acts", so no computed tax here is the
    # whole liability, and an answer must say so every time.
    not_computed: tuple[OutsideAct, ...]


def route(facts: UserFacts) -> ScopeDecision:
    blockers = [
        Blocker(OutOfScope.UNMAPPED_FACT, f"{fact.name} is not a fact the calculator models")
        for fact in facts.unmapped
    ]
    usable: dict[FactField, Fact] = {}
    unknown, unconfirmed = [], []
    for name in INPUT_FIELDS:
        fact = facts.get(name)
        if fact is None or fact.status is FactStatus.MISSING:
            unknown.append(name)
        elif fact.status is FactStatus.PROFILE_DEFAULT:
            unknown.append(name)
            unconfirmed.append(name)
        else:
            usable[name] = fact

    for name, fact in usable.items():
        if name is FactField.TAX_YEAR and fact.value not in supported_tax_years():
            blockers.append(Blocker(OutOfScope.UNSUPPORTED_TAX_YEAR, f"no rate data for tax year {fact.value}", name))
        elif name in HEADS_NOT_COMPUTED and fact.value != 0:
            blockers.append(Blocker(OutOfScope.HEAD_NOT_COMPUTED, f"{name} is not computed", name))
        elif name is FactField.DEDUCTION_OTHER and fact.value != 0:
            # 202(2)(a)(xii) still allows 124(1), 124(2), 125(2) and 146, so an
            # unnamed deduction may or may not change the tax.
            blockers.append(Blocker(OutOfScope.DEDUCTION_NOT_IDENTIFIED, "the claimed deduction is not named", name))

    inferred = tuple(name for name, fact in usable.items() if fact.status is FactStatus.INFERRED)
    common = {"unknown": tuple(unknown), "unconfirmed": tuple(unconfirmed), "inferred": inferred}
    if blockers:
        return ScopeDecision(Route.TEXT_ONLY, blockers=tuple(blockers), **common)
    if unknown:
        return ScopeDecision(Route.INCOMPLETE, **common)

    inputs = inputs_from({name: fact.value for name, fact in usable.items()})
    return ScopeDecision(Route.COMPUTE, inputs=inputs, **common)


def inputs_from(values: dict[FactField, object]) -> CalculatorInputs:
    """Calculator inputs from field values; tax already paid may be absent."""
    return CalculatorInputs(
        tax_year=values[FactField.TAX_YEAR],
        salary=values[FactField.SALARY_INCOME],
        other_income=values[FactField.OTHER_SOURCES_INCOME],
        resident_individual=values[FactField.RESIDENTIAL_STATUS] in _RESIDENT,
        claimed={
            FactField.DEDUCTION_SAVINGS_INSURANCE.value: values[FactField.DEDUCTION_SAVINGS_INSURANCE],
            FactField.DEDUCTION_HEALTH_INSURANCE.value: values[FactField.DEDUCTION_HEALTH_INSURANCE],
        },
        tax_deducted_at_source=values.get(FactField.TDS_PAID),
        advance_tax=values.get(FactField.ADVANCE_TAX_PAID),
    )


def compute(decision: ScopeDecision) -> Computation:
    if decision.inputs is None:
        raise ValueError(f"a {decision.route} decision is not computed")
    return run(decision.inputs)


def run(inputs: CalculatorInputs) -> Computation:
    rates = load_rates(inputs.tax_year)
    comparison = compare_regimes(
        rates,
        salary=inputs.salary,
        other_income=inputs.other_income,
        resident_individual=inputs.resident_individual,
        claimed=inputs.claimed,
    )
    settlement = None
    if inputs.tax_deducted_at_source is not None and inputs.advance_tax is not None:
        settlement = settle(
            rates,
            comparison.under_202_1,
            tax_deducted_at_source=inputs.tax_deducted_at_source,
            advance_tax=inputs.advance_tax,
        )
    return Computation(
        comparison=comparison,
        settlement=settlement,
        not_computed=(rates.outside_act["surcharge"], rates.outside_act["health_and_education_cess"]),
    )
