"""Step 9.2 — tax on total income under the section 202(1) slabs, with a trace.

Input is a total income already computed; deductions and the rebate are Step 9.3.
Every figure the result carries is a `LineItem` that names the provision it came
from and quotes it, so Phase 10's numeric-provenance check and the audit table
read one structure. The result re-derives its own arithmetic on construction, so
a trace that does not add up cannot exist (ADR-101).
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_DOWN, ROUND_HALF_UP, Decimal

from taxverity.calculator.rates import Provenance, TaxYearRates

SLABS_STAGE_VERSION = 1

_HUNDRED = Decimal(100)


class SlabInputError(ValueError):
    pass


@dataclass(frozen=True)
class LineItem:
    label: str
    amount: Decimal
    provenance: Provenance
    basis: Decimal | None = None
    rate_percent: Decimal | None = None

    def __post_init__(self) -> None:
        if (self.basis is None) != (self.rate_percent is None):
            raise ValueError("a rate line carries both its basis and its rate")
        if self.rate_percent is not None and self.amount != self.basis * self.rate_percent / _HUNDRED:
            raise ValueError(f"{self.label}: {self.amount} is not {self.rate_percent}% of {self.basis}")


@dataclass(frozen=True)
class SlabTax:
    tax_year: str
    total_income: Decimal
    rounded_income: LineItem
    bands: tuple[LineItem, ...]
    tax: Decimal

    def __post_init__(self) -> None:
        if sum((band.basis for band in self.bands), Decimal(0)) != self.rounded_income.amount:
            raise ValueError("the bands do not cover the rounded total income exactly")
        if sum((band.amount for band in self.bands), Decimal(0)) != self.tax:
            raise ValueError("the tax is not the sum of its bands")

    def lines(self) -> tuple[LineItem, ...]:
        return (self.rounded_income, *self.bands)


def round_to_multiple(amount: Decimal, multiple: Decimal) -> Decimal:
    # Section 516's order: ignore the paise, then a last figure of five or more
    # goes up to the next multiple of ten.
    rupees = amount.to_integral_value(rounding=ROUND_DOWN)
    return (rupees / multiple).quantize(Decimal(1), rounding=ROUND_HALF_UP) * multiple


def require_amount(value: Decimal, what: str) -> Decimal:
    # A Decimal only: an int would pass, but a float would too once converted,
    # and the calculator admits no binary intermediate (ADR-017).
    if not isinstance(value, Decimal):
        raise SlabInputError(f"{what} must be a Decimal, not {type(value).__name__}")
    if not value.is_finite() or value < 0:
        raise SlabInputError(f"{what} must be a finite amount of at least zero, not {value}")
    return value


def slab_tax(total_income: Decimal, rates: TaxYearRates) -> SlabTax:
    require_amount(total_income, "total income")

    rounding = rates.value("rounding_multiple")
    income = round_to_multiple(total_income, rounding.value)
    rounded = LineItem(
        label="Total income rounded off to the nearest multiple of Rs. 10",
        amount=income,
        provenance=rounding.provenance,
    )

    bands = []
    for slab in rates.new_regime_slabs:
        if income <= slab.lower:
            break
        top = income if slab.upper is None else min(income, slab.upper)
        basis = top - slab.lower
        bands.append(
            LineItem(
                label=_band_label(slab.lower, slab.upper),
                amount=basis * slab.rate_percent / _HUNDRED,
                provenance=slab.provenance,
                basis=basis,
                rate_percent=slab.rate_percent,
            )
        )

    return SlabTax(
        tax_year=rates.tax_year,
        total_income=total_income,
        rounded_income=rounded,
        bands=tuple(bands),
        tax=sum((band.amount for band in bands), Decimal(0)),
    )


def _band_label(lower: Decimal, upper: Decimal | None) -> str:
    if lower == 0:
        return f"Income up to Rs. {upper}"
    if upper is None:
        return f"Income above Rs. {lower}"
    return f"Income from Rs. {lower + 1} to Rs. {upper}"
