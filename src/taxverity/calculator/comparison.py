"""Step 9.4 — the two routes section 202 offers, compared as far as the Act goes.

Section 202(1) is computed in full (Step 9.3). Section 202(4) lets a person opt
out of it, but the rates that would then apply are outside the Act (ADR-100), so
that side stops at total income: the section 19(1) standard deduction for "any
other case", the section 123 deduction within its cap, and the section 122(2)
limit to gross total income. Its tax is carried as the `OutsideAct` declaration,
never as a number, and a section 126 claim is left undetermined because its
limit turns on whose health is insured (ADR-104).
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from taxverity.calculator.new_regime import NewRegimeTax, new_regime_tax
from taxverity.calculator.rates import OutsideAct, Provenance, TaxYearRates
from taxverity.calculator.slabs import LineItem, round_to_multiple

COMPARISON_STAGE_VERSION = 2

_ZERO = Decimal(0)
_SAVINGS = "deduction_savings_insurance"
_HEALTH = "deduction_health_insurance"


@dataclass(frozen=True)
class Undetermined:
    name: str
    claimed: Decimal
    reason: str
    provenance: Provenance


@dataclass(frozen=True)
class OptedOut:
    option: Provenance
    salary: Decimal
    other_income: Decimal
    standard_deduction: LineItem | None
    gross_total_income: Decimal
    deductions: tuple[LineItem, ...]
    undetermined: tuple[Undetermined, ...]
    total_income: Decimal
    rounded_total_income: LineItem
    tax: OutsideAct

    def __post_init__(self) -> None:
        deducted = self.standard_deduction.amount if self.standard_deduction else _ZERO
        if self.gross_total_income != self.salary - deducted + self.other_income:
            raise ValueError("gross total income is not salary less the standard deduction plus other income")
        allowed = sum((line.amount for line in self.deductions), _ZERO)
        if allowed > self.gross_total_income:
            raise ValueError("the deductions exceed gross total income")
        if self.total_income != self.gross_total_income - allowed:
            raise ValueError("total income is not gross total income less the deductions")
        if self.rounded_total_income.amount != round_to_multiple(self.total_income, Decimal(10)):
            raise ValueError("the rounded total income is not total income rounded under section 516")

    @property
    def total_income_is_upper_bound(self) -> bool:
        # An undetermined deduction can only lower total income further.
        return bool(self.undetermined)


@dataclass(frozen=True)
class RegimeComparison:
    tax_year: str
    under_202_1: NewRegimeTax
    opted_out: OptedOut


def compare_regimes(
    rates: TaxYearRates,
    *,
    salary: Decimal,
    other_income: Decimal,
    resident_individual: bool,
    claimed: dict[str, Decimal] | None = None,
) -> RegimeComparison:
    claimed = claimed or {}
    # Validates every input, and refuses a claim it does not know, before the
    # opted-out side reads any of them.
    under = new_regime_tax(
        rates,
        salary=salary,
        other_income=other_income,
        resident_individual=resident_individual,
        claimed=claimed,
    )
    return RegimeComparison(
        tax_year=rates.tax_year,
        under_202_1=under,
        opted_out=_opted_out(rates, salary, other_income, claimed),
    )


def _opted_out(rates: TaxYearRates, salary: Decimal, other_income: Decimal, claimed: dict[str, Decimal]) -> OptedOut:
    standard = None
    if salary > 0:
        cap = rates.value("standard_deduction_other")
        standard = LineItem(label="Standard deduction from salary", amount=min(cap.value, salary), provenance=cap.provenance)
    gross = salary - (standard.amount if standard else _ZERO) + other_income

    deductions = []
    if _SAVINGS in claimed:
        cap = rates.value("savings_insurance_deduction_cap")
        within_cap = min(claimed[_SAVINGS], cap.value)
        if within_cap > gross > 0:
            deductions.append(
                LineItem(
                    label="Deduction under section 123, limited to gross total income",
                    amount=gross,
                    provenance=rates.rule("chapter_viii_within_gross_total_income"),
                )
            )
        elif 0 < within_cap <= gross:
            deductions.append(
                LineItem(label=f"Deduction under section 123, up to Rs. {cap.value}", amount=within_cap, provenance=cap.provenance)
            )

    undetermined = ()
    if claimed.get(_HEALTH, _ZERO) > 0:
        undetermined = (
            Undetermined(
                name=_HEALTH,
                claimed=claimed[_HEALTH],
                reason="the section 126 limit turns on whose health is insured and whether they are a senior citizen",
                provenance=rates.rule("health_insurance_limit_depends_on_who_is_insured"),
            ),
        )

    allowed = sum((line.amount for line in deductions), _ZERO)
    # Section 516 rounds "the amount of total income computed", whichever route
    # computed it, so this side is rounded exactly as the 202(1) side is.
    rounding = rates.value("rounding_multiple")
    rounded = LineItem(
        label="Total income rounded off to the nearest multiple of Rs. 10",
        amount=round_to_multiple(gross - allowed, rounding.value),
        provenance=rounding.provenance,
    )
    return OptedOut(
        option=rates.rule("section_202_1_option"),
        salary=salary,
        other_income=other_income,
        standard_deduction=standard,
        gross_total_income=gross,
        deductions=tuple(deductions),
        undetermined=undetermined,
        total_income=gross - allowed,
        rounded_total_income=rounded,
        tax=rates.outside_act["old_regime_slabs"],
    )
