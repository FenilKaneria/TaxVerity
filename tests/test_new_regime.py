"""Step 9.3 — the section 202(1) computation: standard deduction, the section 156(2)
rebate and its marginal relief, the rounded amount payable (ADR-103)."""

from __future__ import annotations

from dataclasses import replace
from decimal import Decimal

import pytest

from taxverity.calculator.new_regime import NEW_REGIME_STAGE_VERSION, new_regime_tax
from taxverity.calculator.rates import load_rates
from taxverity.calculator.slabs import SlabInputError, round_to_multiple

D = Decimal


@pytest.fixture(scope="module")
def rates():
    return load_rates("2026-27")


def tax(rates, salary="0", other="0", resident=True, **claimed):
    return new_regime_tax(
        rates,
        salary=D(salary),
        other_income=D(other),
        resident_individual=resident,
        claimed={name: D(amount) for name, amount in claimed.items()} or None,
    )


def test_the_stage_version_is_declared():
    assert NEW_REGIME_STAGE_VERSION == 2


# Worked by hand from sections 19(1), 202(1), 156(2) and 516, not from the engine.
@pytest.mark.parametrize(
    ("salary", "other", "resident", "slab_tax", "rebate", "payable"),
    [
        ("0", "0", True, "0", None, "0"),
        ("50000", "0", True, "0", None, "0"),
        ("1275000", "0", True, "60000", "60000", "0"),
        ("0", "1200000", True, "60000", "60000", "0"),
        ("0", "1200000", False, "60000", None, "60000"),
        ("1285000", "0", True, "61500", "51500", "10000"),
        ("0", "1270580", True, "70587", "7", "70580"),
        ("0", "1270590", True, "70588.5", None, "70590"),
        ("0", "400070", False, "3.5", None, "0"),
        ("0", "400100", False, "5", None, "10"),
        ("3075000", "0", True, "480000", None, "480000"),
        ("1000000", "500000", True, "93750", None, "93750"),
    ],
)
def test_the_tax_matches_the_act_worked_by_hand(rates, salary, other, resident, slab_tax, rebate, payable):
    result = tax(rates, salary, other, resident)
    assert result.slab.tax == D(slab_tax)
    assert (result.rebate.amount if result.rebate else None) == (D(rebate) if rebate else None)
    assert result.payable.amount == D(payable)


def test_the_standard_deduction_is_the_cap_or_the_salary_whichever_is_less(rates):
    assert tax(rates, "40000").standard_deduction.amount == D(40000)
    assert tax(rates, "900000").standard_deduction.amount == D(75000)
    assert tax(rates, "0", "900000").standard_deduction is None


def test_the_standard_deduction_reaches_salary_only(rates):
    # Other income is not "income chargeable under the head Salaries".
    assert tax(rates, "10000", "900000").slab.total_income == D(900000)


def test_each_line_cites_the_provision_it_applies(rates):
    result = tax(rates, "1285000")
    assert result.standard_deduction.provenance.citation == "19(1)"
    assert result.rebate.provenance.citation == "156(2)(b)"
    assert result.payable.provenance.citation == "516"
    assert tax(rates, "900000").rebate.provenance.citation == "156(2)(a)"
    assert result.lines()[0] is result.standard_deduction
    assert result.lines()[-1] is result.payable


def test_a_claimed_chapter_viii_deduction_is_recorded_and_changes_nothing(rates):
    plain = tax(rates, "1800000")
    claimed = tax(rates, "1800000", deduction_savings_insurance="150000", deduction_health_insurance="25000")
    assert claimed.payable == plain.payable
    assert claimed.slab == plain.slab
    assert [line.label for line in claimed.not_allowed] == [
        "Deduction under section 126 (Chapter VIII) not allowed",
        "Deduction under section 123 (Chapter VIII) not allowed",
    ]
    assert {line.provenance.citation for line in claimed.not_allowed} == {"202(2)(a)(xii)"}
    assert [line.amount for line in claimed.not_allowed] == [D(25000), D(150000)]


def test_an_unknown_claimed_deduction_is_refused_rather_than_ignored(rates):
    with pytest.raises(KeyError, match="deduction_other"):
        tax(rates, "900000", deduction_other="1000")


# --- properties over a sweep --------------------------------------------------


@pytest.fixture(scope="module")
def sweep(rates):
    incomes = sorted({D(n) for n in range(1_100_000, 1_400_001, 370)} | {D(1_200_000), D(1_200_010), D(1_270_590)})
    return [tax(rates, "0", str(income)) for income in incomes]


def test_tax_payable_never_falls_as_income_rises(sweep):
    for lower, higher in zip(sweep, sweep[1:], strict=False):
        assert higher.payable.amount >= lower.payable.amount


def test_marginal_relief_keeps_the_tax_within_the_income_above_twelve_lakh(sweep):
    checked = 0
    for result in sweep:
        above = result.slab.rounded_income.amount - D(1_200_000)
        if above > 0:
            checked += 1
            assert result.payable.amount == round_to_multiple(min(result.slab.tax, above), D(10))
    assert checked


def test_the_rebate_never_exceeds_the_slab_tax(sweep):
    # Section 156(3): the rebate is bounded by the tax at the 202(1) rates.
    for result in sweep:
        if result.rebate:
            assert result.rebate.amount <= result.slab.tax


def test_a_resident_never_pays_more_than_a_non_resident(rates):
    for income in range(1_150_000, 1_300_001, 1_010):
        assert tax(rates, "0", str(income)).payable.amount <= tax(rates, "0", str(income), False).payable.amount


# --- refusals -----------------------------------------------------------------


def test_residence_has_no_default(rates):
    with pytest.raises(TypeError):
        new_regime_tax(rates, salary=D(0), other_income=D(0))  # type: ignore[call-arg]
    with pytest.raises(TypeError, match="True or False"):
        new_regime_tax(rates, salary=D(0), other_income=D(0), resident_individual="yes")  # type: ignore[arg-type]


@pytest.mark.parametrize(("field", "bad"), [("salary", 100.0), ("other_income", D(-1)), ("salary", D("NaN"))])
def test_a_bad_amount_is_refused(rates, field, bad):
    arguments = {"salary": D(0), "other_income": D(0), field: bad}
    with pytest.raises(SlabInputError):
        new_regime_tax(rates, resident_individual=True, **arguments)


def test_a_computation_that_does_not_add_up_cannot_be_built(rates):
    result = tax(rates, "1285000")
    with pytest.raises(ValueError, match="rounded tax after rebate"):
        replace(result, payable=replace(result.payable, amount=D(0)))
    with pytest.raises(ValueError, match="exceeds the income-tax"):
        replace(result, rebate=replace(result.rebate, amount=result.slab.tax + 1))
    with pytest.raises(ValueError, match="total income"):
        replace(result, other_income=D(1))
