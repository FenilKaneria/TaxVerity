"""Step 9.4 — section 202(1) against the section 202(4) option, as far as the Act
carries each (ADR-104)."""

from __future__ import annotations

import re
from dataclasses import replace
from decimal import Decimal

import pytest

from taxverity.calculator.comparison import COMPARISON_STAGE_VERSION, compare_regimes
from taxverity.calculator.rates import load_rates

D = Decimal


@pytest.fixture(scope="module")
def rates():
    return load_rates("2026-27")


def compare(rates, salary="0", other="0", resident=True, **claimed):
    return compare_regimes(
        rates,
        salary=D(salary),
        other_income=D(other),
        resident_individual=resident,
        claimed={name: D(amount) for name, amount in claimed.items()} or None,
    )


def test_the_stage_version_is_declared():
    assert COMPARISON_STAGE_VERSION == 2


def test_the_202_1_side_is_the_step_9_3_computation(rates):
    # 1800000 less 75000 is 1725000: the bands to sixteen lakh give 120000, then
    # 20% of 125000 is 25000. No rebate above twelve lakh, and section 123 is
    # not allowed under 202(1).
    result = compare(rates, "1800000", deduction_savings_insurance="150000")
    assert result.under_202_1.payable.amount == D(145000)
    assert [line.provenance.citation for line in result.under_202_1.not_allowed] == ["202(2)(a)(xii)"]


# Worked by hand: salary less Rs. 50000 (19(1) "any other case"), plus other
# income, less the section 123 deduction within Rs. 150000 and within gross
# total income (122(2)).
@pytest.mark.parametrize(
    ("salary", "other", "savings", "gross", "deducted", "total"),
    [
        ("1800000", "0", None, "1750000", "0", "1750000"),
        ("1800000", "0", "200000", "1750000", "150000", "1600000"),
        ("1800000", "0", "90000", "1750000", "90000", "1660000"),
        ("30000", "0", "150000", "0", "0", "0"),
        ("0", "100000", "150000", "100000", "100000", "0"),
        ("600000", "200000", "150000", "750000", "150000", "600000"),
    ],
)
def test_opted_out_total_income_matches_the_act_worked_by_hand(rates, salary, other, savings, gross, deducted, total):
    claimed = {"deduction_savings_insurance": savings} if savings else {}
    side = compare(rates, salary, other, **claimed).opted_out
    assert side.gross_total_income == D(gross)
    assert sum((line.amount for line in side.deductions), D(0)) == D(deducted)
    assert side.total_income == D(total)


def test_the_opted_out_tax_is_never_a_number(rates):
    side = compare(rates, "1800000").opted_out
    assert side.tax is rates.outside_act["old_regime_slabs"]
    assert not re.search(r"\d", side.tax.provenance.source_text)
    assert side.option.citation == "202(4)"


def test_the_limit_to_gross_total_income_cites_section_122_2(rates):
    side = compare(rates, "0", "100000", deduction_savings_insurance="150000").opted_out
    assert [(line.amount, line.provenance.citation) for line in side.deductions] == [(D(100000), "122(2)")]


def test_a_capped_deduction_cites_section_123_and_states_only_figures_its_quote_prints(rates):
    (line,) = compare(rates, "1800000", deduction_savings_insurance="200000").opted_out.deductions
    assert line.provenance.citation == "123"
    for figure in re.findall(r"\d+", line.label.replace("section 123", "")):
        assert figure in line.provenance.source_text


def test_a_health_insurance_claim_is_undetermined_and_bounds_total_income(rates):
    plain = compare(rates, "1800000").opted_out
    side = compare(rates, "1800000", deduction_health_insurance="40000").opted_out
    assert side.total_income == plain.total_income
    assert side.total_income_is_upper_bound
    assert not plain.total_income_is_upper_bound
    (entry,) = side.undetermined
    assert entry.claimed == D(40000)
    assert entry.provenance.citation == "126(2)"


def test_the_standard_deduction_differs_between_the_two_routes(rates):
    result = compare(rates, "1000000")
    assert result.under_202_1.standard_deduction.amount == D(75000)
    assert result.opted_out.standard_deduction.amount == D(50000)
    assert result.opted_out.standard_deduction.provenance.citation == "19(1)"


def test_an_unknown_claim_is_refused(rates):
    with pytest.raises(KeyError):
        compare(rates, "1000000", deduction_other="1000")


def test_an_opted_out_side_that_does_not_add_up_cannot_be_built(rates):
    side = compare(rates, "1800000", deduction_savings_insurance="150000").opted_out
    with pytest.raises(ValueError, match="gross total income less"):
        replace(side, total_income=side.total_income + 1)
    with pytest.raises(ValueError, match="exceed gross total income"):
        replace(side, gross_total_income=D(100), salary=D(50100), total_income=D(-149900))
