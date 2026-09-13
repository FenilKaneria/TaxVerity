"""Step 9.6 — the section 202(1) tax set off against tax already paid.

Section 270(1)(c) determines the sum payable, or the refund due, after adjusting
the tax by any tax deducted at source and any advance tax paid. The net amount is
then rounded under section 516, which reaches "any amount payable or refundable".
Interest and fee enter the same adjustment, but they turn on dates of payment and
of filing that no fact carries, so they are named as not computed rather than
taken as nil (ADR-107).
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from taxverity.calculator.new_regime import NewRegimeTax
from taxverity.calculator.rates import Provenance, TaxYearRates
from taxverity.calculator.slabs import LineItem, require_amount, round_to_multiple

SETTLEMENT_STAGE_VERSION = 1

_ZERO = Decimal(0)


@dataclass(frozen=True)
class Settlement:
    payable: LineItem
    tax_deducted_at_source: LineItem | None
    advance_tax: LineItem | None
    balance_payable: LineItem | None
    refund_due: LineItem | None
    adjustment: Provenance

    def __post_init__(self) -> None:
        if (self.balance_payable is None) == (self.refund_due is None):
            raise ValueError("a settlement is either a balance payable or a refund due")
        net = self.payable.amount - self.prepaid
        if self.balance_payable and (net < 0 or self.balance_payable.amount != round_to_multiple(net, Decimal(10))):
            raise ValueError("the balance payable is not the rounded tax less what was paid")
        if self.refund_due and (net >= 0 or self.refund_due.amount != round_to_multiple(-net, Decimal(10))):
            raise ValueError("the refund due is not the rounded excess of what was paid")

    @property
    def prepaid(self) -> Decimal:
        return sum((line.amount for line in (self.tax_deducted_at_source, self.advance_tax) if line), _ZERO)

    def lines(self) -> tuple[LineItem, ...]:
        return tuple(
            line
            for line in (self.payable, self.tax_deducted_at_source, self.advance_tax, self.balance_payable, self.refund_due)
            if line is not None
        )


def settle(
    rates: TaxYearRates,
    under_202_1: NewRegimeTax,
    *,
    tax_deducted_at_source: Decimal,
    advance_tax: Decimal,
) -> Settlement:
    require_amount(tax_deducted_at_source, "tax deducted at source")
    require_amount(advance_tax, "advance tax")

    deducted = None
    if tax_deducted_at_source > 0:
        deducted = LineItem(
            label="Less tax deducted at source",
            amount=tax_deducted_at_source,
            provenance=rates.rule("tax_deducted_at_source_adjusted"),
        )
    advance = None
    if advance_tax > 0:
        advance = LineItem(label="Less advance tax paid", amount=advance_tax, provenance=rates.rule("advance_tax_adjusted"))

    net = under_202_1.payable.amount - tax_deducted_at_source - advance_tax
    rounding = rates.value("rounding_multiple")
    balance = refund = None
    if net >= 0:
        balance = LineItem(
            label="Balance payable rounded off to the nearest multiple of Rs. 10",
            amount=round_to_multiple(net, rounding.value),
            provenance=rounding.provenance,
        )
    else:
        refund = LineItem(
            label="Refund due rounded off to the nearest multiple of Rs. 10",
            amount=round_to_multiple(-net, rounding.value),
            provenance=rounding.provenance,
        )

    return Settlement(
        payable=under_202_1.payable,
        tax_deducted_at_source=deducted,
        advance_tax=advance,
        balance_payable=balance,
        refund_due=refund,
        adjustment=rates.rule("sum_payable_or_refund_after_adjustment"),
    )
