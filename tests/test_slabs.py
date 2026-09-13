"""Step 9.2 — the section 202(1) slab engine and its line-item trace (ADR-101)."""

from __future__ import annotations

import re
from dataclasses import replace
from decimal import Decimal

import pytest

from taxverity.calculator.rates import load_rates
from taxverity.calculator.slabs import (
    SLABS_STAGE_VERSION,
    LineItem,
    SlabInputError,
    SlabTax,
    round_to_multiple,
    slab_tax,
)

D = Decimal


@pytest.fixture(scope="module")
def rates():
    return load_rates("2026-27")


def test_the_stage_version_is_declared():
    assert SLABS_STAGE_VERSION == 1


# Worked by hand from the seven rows of section 202(1), not from the engine.
@pytest.mark.parametrize(
    ("income", "tax"),
    [
        ("0", "0"),
        ("400000", "0"),
        ("400010", "0.50"),
        ("800000", "20000"),
        ("1200000", "60000"),
        ("1275000", "71250"),
        ("1600000", "120000"),
        ("2000000", "200000"),
        ("2400000", "300000"),
        ("2500000", "330000"),
    ],
)
def test_the_tax_matches_the_table_worked_by_hand(rates, income, tax):
    assert slab_tax(D(income), rates).tax == D(tax)


@pytest.mark.parametrize(
    ("amount", "rounded"),
    [
        ("0", "0"),
        ("4", "0"),
        ("5", "10"),
        ("15", "20"),
        ("25", "30"),
        ("400004.99", "400000"),
        ("400005", "400010"),
        ("400005.01", "400010"),
        ("1234567.89", "1234570"),
    ],
)
def test_total_income_is_rounded_as_section_516_says(amount, rounded):
    assert round_to_multiple(D(amount), D(10)) == D(rounded)


def test_the_slabs_apply_to_the_rounded_income_not_the_raw_one(rates):
    result = slab_tax(D("400004.99"), rates)
    assert result.total_income == D("400004.99")
    assert result.rounded_income.amount == D(400000)
    assert result.tax == 0


def test_the_trace_names_each_band_the_income_reaches(rates):
    result = slab_tax(D(1275000), rates)
    assert [band.basis for band in result.bands] == [D(400000), D(400000), D(400000), D(75000)]
    assert [band.rate_percent for band in result.bands] == [D(0), D(5), D(10), D(15)]
    assert [band.amount for band in result.bands] == [D(0), D(20000), D(40000), D(11250)]
    assert result.lines() == (result.rounded_income, *result.bands)


def test_no_band_is_listed_for_zero_income(rates):
    assert slab_tax(D(0), rates).bands == ()


def test_every_line_cites_and_quotes_its_provision(rates):
    result = slab_tax(D(3000000), rates)
    assert result.rounded_income.provenance.citation == "516"
    assert len(result.bands) == 7
    for band in result.bands:
        assert band.provenance.citation == "202(1)"
        # Every figure a band label states is printed in the row it quotes.
        for figure in re.findall(r"\d+", band.label):
            assert figure in band.provenance.source_text


def test_the_result_carries_its_tax_year(rates):
    assert slab_tax(D(1), rates).tax_year == "2026-27"


# --- properties over a sweep --------------------------------------------------


@pytest.fixture(scope="module")
def sweep(rates):
    incomes = [D(n) for n in range(0, 3_000_001, 1_730)]
    incomes += [slab.upper + delta for slab in rates.new_regime_slabs[:-1] for delta in (-10, 0, 10)]
    return [slab_tax(income, rates) for income in sorted(incomes)]


def test_tax_never_falls_as_income_rises(sweep):
    for lower, higher in zip(sweep, sweep[1:], strict=False):
        assert higher.tax >= lower.tax


def test_the_marginal_rate_never_exceeds_the_top_rate(sweep):
    for lower, higher in zip(sweep, sweep[1:], strict=False):
        gained = higher.rounded_income.amount - lower.rounded_income.amount
        if gained:
            assert (higher.tax - lower.tax) / gained <= D("0.30")


def test_tax_is_continuous_across_every_slab_boundary(rates):
    slabs = rates.new_regime_slabs
    for slab, following in zip(slabs, slabs[1:], strict=False):
        at = slab_tax(slab.upper, rates).tax
        above = slab_tax(slab.upper + 10, rates).tax
        assert above - at == D(10) * following.rate_percent / 100


def test_the_arithmetic_is_exact_in_decimal(sweep):
    for result in sweep:
        assert isinstance(result.tax, Decimal)
        assert sum(band.basis for band in result.bands) == result.rounded_income.amount


# --- refusals -----------------------------------------------------------------


@pytest.mark.parametrize("bad", [1200000, 1200000.0, "1200000"])
def test_an_amount_that_is_not_a_decimal_is_refused(rates, bad):
    with pytest.raises(SlabInputError, match="Decimal"):
        slab_tax(bad, rates)


@pytest.mark.parametrize("bad", ["-10", "NaN", "Infinity"])
def test_a_negative_or_non_finite_amount_is_refused(rates, bad):
    with pytest.raises(SlabInputError, match="at least zero"):
        slab_tax(D(bad), rates)


def test_a_rate_line_whose_amount_is_wrong_cannot_be_built(rates):
    band = slab_tax(D(800000), rates).bands[1]
    with pytest.raises(ValueError, match="is not 5% of"):
        replace(band, amount=band.amount + 1)


def test_a_rate_line_needs_both_its_basis_and_its_rate(rates):
    band = slab_tax(D(800000), rates).bands[1]
    with pytest.raises(ValueError, match="both"):
        LineItem(label=band.label, amount=band.amount, provenance=band.provenance, basis=band.basis)


def test_a_trace_that_does_not_add_up_cannot_be_built(rates):
    result = slab_tax(D(800000), rates)
    with pytest.raises(ValueError, match="sum of its bands"):
        replace(result, tax=result.tax + 1)
    with pytest.raises(ValueError, match="cover"):
        SlabTax(result.tax_year, result.total_income, result.rounded_income, result.bands[:1], D(0))
