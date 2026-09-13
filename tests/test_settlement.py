"""Step 9.6 — the 202(1) tax set off against tax already paid, section 270(1)(c)."""

from __future__ import annotations

from dataclasses import replace
from decimal import Decimal

import pytest

from taxverity.calculator.new_regime import new_regime_tax
from taxverity.calculator.rates import load_rates
from taxverity.calculator.settlement import SETTLEMENT_STAGE_VERSION, settle
from taxverity.calculator.slabs import SlabInputError

D = Decimal


@pytest.fixture(scope="module")
def rates():
    return load_rates("2026-27")


def settled(rates, salary, tds="0", advance="0"):
    under = new_regime_tax(rates, salary=D(salary), other_income=D(0), resident_individual=True)
    return settle(rates, under, tax_deducted_at_source=D(tds), advance_tax=D(advance))


def test_the_stage_version_is_declared():
    assert SETTLEMENT_STAGE_VERSION == 1


# Salary 1800000 gives payable 145000 (worked in test_comparison_regimes).
@pytest.mark.parametrize(
    ("tds", "advance", "balance", "refund"),
    [
        ("0", "0", "145000", None),
        ("100000", "0", "45000", None),
        ("100000", "45000", "0", None),
        ("120000", "30000", None, "5000"),
        ("144995.50", "0", "0", None),  # 4.50 left: paise ignored, then 4 rounds down
        ("144994", "0", "10", None),  # last figure 6 rounds up
        ("145004.99", "0", None, "0"),  # a refund of 4.99 rounds to nil
        ("145005", "0", None, "10"),
    ],
)
def test_balance_or_refund_worked_by_hand(rates, tds, advance, balance, refund):
    result = settled(rates, "1800000", tds, advance)
    assert result.payable.amount == D(145000)
    assert (result.balance_payable.amount if result.balance_payable else None) == (D(balance) if balance else None)
    assert (result.refund_due.amount if result.refund_due else None) == (D(refund) if refund else None)


def test_every_line_names_its_provision(rates):
    result = settled(rates, "1800000", "100000", "20000")
    assert [line.provenance.citation for line in result.lines()] == ["516", "270(1)(c)(i)", "270(1)(c)(iii)", "516"]
    assert result.adjustment.citation == "270(1)(c)"
    assert "interest and fee" in result.adjustment.source_text


def test_nil_payments_write_no_lines(rates):
    result = settled(rates, "1800000")
    assert result.tax_deducted_at_source is None
    assert result.advance_tax is None


def test_tax_paid_on_nil_payable_is_all_refunded(rates):
    result = settled(rates, "900000", "8000")
    assert result.payable.amount == D(0)
    assert result.refund_due.amount == D(8000)


@pytest.mark.parametrize("bad", [D(-1), D("NaN"), 5])
def test_a_payment_that_is_not_a_non_negative_decimal_is_refused(rates, bad):
    under = new_regime_tax(rates, salary=D(0), other_income=D(0), resident_individual=True)
    with pytest.raises(SlabInputError):
        settle(rates, under, tax_deducted_at_source=bad, advance_tax=D(0))


def test_a_settlement_that_does_not_add_up_cannot_be_built(rates):
    result = settled(rates, "1800000", "100000")
    with pytest.raises(ValueError, match="balance payable"):
        replace(result, balance_payable=replace(result.balance_payable, amount=D(40000)))
    with pytest.raises(ValueError, match="either"):
        replace(result, refund_due=result.balance_payable)
    with pytest.raises(ValueError, match="refund due"):
        replace(result, balance_payable=None, refund_due=result.balance_payable)
