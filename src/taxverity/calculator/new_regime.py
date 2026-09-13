"""Step 9.3 — income-tax under section 202(1): standard deduction, the section
156(2) rebate with its marginal relief, and the section 516 rounding of the
amount payable.

Only what section 202(1) allows is applied. Section 202(2)(a)(xii) computes
total income without Chapter VIII deductions other than 124(1), 124(2), 125(2)
and 146, so a claimed section 123 or 126 deduction is recorded as not allowed
and changes nothing. The other regime's rates are outside the Act (ADR-100), so
no deduction is computed for it here (ADR-103).
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from taxverity.calculator.rates import TaxYearRates
from taxverity.calculator.slabs import (
    LineItem,
    SlabTax,
    require_amount,
    round_to_multiple,
    slab_tax,
)

NEW_REGIME_STAGE_VERSION = 2

_ZERO = Decimal(0)


@dataclass(frozen=True)
class NewRegimeTax:
    tax_year: str
    salary: Decimal
    other_income: Decimal
    standard_deduction: LineItem | None
    slab: SlabTax
    rebate: LineItem | None
    payable: LineItem
    not_allowed: tuple[LineItem, ...]

    def __post_init__(self) -> None:
        deducted = self.standard_deduction.amount if self.standard_deduction else _ZERO
        if deducted > self.salary:
            raise ValueError("the standard deduction exceeds the salary")
        if self.slab.total_income != self.salary - deducted + self.other_income:
            raise ValueError("total income is not salary less the standard deduction plus other income")
        rebated = self.rebate.amount if self.rebate else _ZERO
        if not _ZERO <= rebated <= self.slab.tax:
            raise ValueError("the rebate exceeds the income-tax it is deducted from")
        if self.payable.amount != round_to_multiple(self.slab.tax - rebated, Decimal(10)):
            raise ValueError("the amount payable is not the rounded tax after rebate")

    def lines(self) -> tuple[LineItem, ...]:
        return tuple(
            line
            for line in (self.standard_deduction, *self.slab.lines(), self.rebate, self.payable)
            if line is not None
        )


def new_regime_tax(
    rates: TaxYearRates,
    *,
    salary: Decimal,
    other_income: Decimal,
    resident_individual: bool,
    claimed: dict[str, Decimal] | None = None,
) -> NewRegimeTax:
    # `resident_individual` has no default: the rebate turns on it, and a
    # forgotten argument must fail rather than quietly grant or withhold it.
    require_amount(salary, "salary")
    require_amount(other_income, "other income")
    if not isinstance(resident_individual, bool):
        raise TypeError("resident_individual must be True or False")

    standard = None
    if salary > 0:
        cap = rates.value("standard_deduction_new_regime")
        standard = LineItem(
            label="Standard deduction from salary",
            amount=min(cap.value, salary),
            provenance=cap.provenance,
        )

    deducted = standard.amount if standard else _ZERO
    slab = slab_tax(salary - deducted + other_income, rates)
    rebate = _rebate(rates, slab) if resident_individual else None
    rebated = rebate.amount if rebate else _ZERO
    rounding = rates.value("rounding_multiple")
    payable = LineItem(
        label="Income-tax payable rounded off to the nearest multiple of Rs. 10",
        amount=round_to_multiple(slab.tax - rebated, rounding.value),
        provenance=rounding.provenance,
    )

    return NewRegimeTax(
        tax_year=rates.tax_year,
        salary=salary,
        other_income=other_income,
        standard_deduction=standard,
        slab=slab,
        rebate=rebate,
        payable=payable,
        not_allowed=_not_allowed(rates, claimed or {}),
    )


def _rebate(rates: TaxYearRates, slab: SlabTax) -> LineItem | None:
    income = slab.rounded_income.amount
    tax = slab.tax
    limit = rates.value("rebate_new_regime_income_limit")
    if income <= limit.value:
        maximum = rates.value("rebate_new_regime_maximum")
        amount = min(tax, maximum.value)
        if amount == 0:
            return None
        return LineItem(label="Rebate on income not exceeding twelve lakh rupees", amount=amount, provenance=maximum.provenance)

    threshold = rates.value("rebate_new_regime_marginal_relief_threshold")
    excess = income - threshold.value
    if tax <= excess:
        return None
    # Section 156(2)(b) leaves the tax equal to the income above twelve lakh; the
    # rebate is then the tax less a positive excess, so the 156(3) cap cannot bind.
    return LineItem(
        label="Rebate by which the tax exceeds the income above twelve lakh rupees",
        amount=tax - excess,
        provenance=threshold.provenance,
    )


def _not_allowed(rates: TaxYearRates, claimed: dict[str, Decimal]) -> tuple[LineItem, ...]:
    lines = []
    for name, amount in sorted(claimed.items()):
        require_amount(amount, name)
        if name not in rates.not_allowed_under_202_1:
            raise KeyError(f"{name} is not a deduction this computation knows to exclude")
        if amount == 0:
            continue
        entry = rates.not_allowed_under_202_1[name]
        lines.append(
            LineItem(
                label=f"Deduction under section {entry.section} (Chapter {entry.chapter}) not allowed",
                amount=amount,
                provenance=entry.provenance,
            )
        )
    return tuple(lines)
