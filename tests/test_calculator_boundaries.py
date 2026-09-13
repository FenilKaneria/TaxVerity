"""Step 9.5 — boundaries, derived from the rate data file rather than typed in.

Each edge the data file states is exercised at, just below and just above
itself, so a changed figure moves its own tests with it. What is asserted at an
edge is which side of it an amount lands on — the Act's rows are inclusive at the
top ("Upto Rs. 400000", "does not exceed twelve lakh") — never a total the test
would have to compute by re-implementing the engine (ADR-105).
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from taxverity.calculator.comparison import compare_regimes
from taxverity.calculator.new_regime import new_regime_tax
from taxverity.calculator.rates import load_rates
from taxverity.calculator.slabs import round_to_multiple, slab_tax

D = Decimal
RATES = load_rates("2026-27")
STEP = RATES.value("rounding_multiple").value
EDGES = [slab.upper for slab in RATES.new_regime_slabs if slab.upper is not None]


def tax(salary="0", other="0", resident=True, **claimed):
    return new_regime_tax(
        RATES,
        salary=D(salary),
        other_income=D(other),
        resident_individual=resident,
        claimed={name: D(amount) for name, amount in claimed.items()} or None,
    )


def opted_out(salary="0", other="0", **claimed):
    return compare_regimes(
        RATES,
        salary=D(salary),
        other_income=D(other),
        resident_individual=True,
        claimed={name: D(amount) for name, amount in claimed.items()},
    ).opted_out


def test_the_edges_are_read_from_the_data_file():
    assert len(EDGES) == len(RATES.new_regime_slabs) - 1
    assert EDGES == sorted(EDGES)


# --- section 202(1) slab edges ------------------------------------------------


@pytest.mark.parametrize("index", range(len(EDGES)))
def test_an_income_at_a_slab_edge_belongs_to_the_lower_row(index):
    edge, row = EDGES[index], RATES.new_regime_slabs[index]
    top = slab_tax(edge, RATES).bands[-1]
    assert (top.rate_percent, top.basis) == (row.rate_percent, edge - row.lower)


@pytest.mark.parametrize("index", range(len(EDGES)))
def test_one_rounding_step_above_an_edge_opens_the_next_row(index):
    edge, above = EDGES[index], RATES.new_regime_slabs[index + 1]
    at, beyond = slab_tax(edge, RATES), slab_tax(edge + STEP, RATES)
    assert len(beyond.bands) == len(at.bands) + 1
    assert (beyond.bands[-1].rate_percent, beyond.bands[-1].basis) == (above.rate_percent, STEP)
    assert beyond.tax - at.tax == STEP * above.rate_percent / 100


@pytest.mark.parametrize("index", range(len(EDGES)))
def test_one_rounding_step_below_an_edge_stays_in_its_row(index):
    edge, row = EDGES[index], RATES.new_regime_slabs[index]
    below, at = slab_tax(edge - STEP, RATES), slab_tax(edge, RATES)
    assert len(below.bands) == len(at.bands)
    assert at.tax - below.tax == STEP * row.rate_percent / 100


# --- section 516 rounding -----------------------------------------------------


@pytest.mark.parametrize("edge", [D(0), *EDGES])
@pytest.mark.parametrize(
    ("offset", "lands_on"),
    [("0", "0"), ("0.99", "0"), ("4", "0"), ("4.99", "0"), ("5", "1"), ("9.99", "1")],
)
def test_rounding_ignores_paise_then_takes_five_upward(edge, offset, lands_on):
    # Ignoring paise first and rounding the whole amount half-up agree for every
    # remainder (floor(r) >= 5 exactly when r >= 5), so no input separates the
    # two orders. Measured: a mutant keeping the paise survives every suite.
    assert slab_tax(edge + D(offset), RATES).rounded_income.amount == edge + D(lands_on) * STEP


@pytest.mark.parametrize(
    ("offset", "lands_on"),
    [("0", "0"), ("0.99", "0"), ("4.99", "0"), ("5", "1"), ("9.99", "1")],
)
def test_the_opted_out_total_income_is_rounded_the_same_way(offset, lands_on):
    # Section 516 names "the amount of total income computed", not one route.
    base = CAP_123 * 3
    side = opted_out(other=str(base + D(offset)))
    assert side.total_income == base + D(offset)
    assert side.rounded_total_income.amount == base + D(lands_on) * STEP
    assert side.rounded_total_income.provenance.citation == "516"


def test_rounding_below_one_step_is_rounding_of_the_last_figure_only():
    assert round_to_multiple(D("5"), STEP) == STEP
    assert round_to_multiple(D("4.999"), STEP) == 0


# --- section 156(2) rebate and marginal relief --------------------------------


LIMIT = RATES.value("rebate_new_regime_income_limit").value
MAXIMUM = RATES.value("rebate_new_regime_maximum").value


def test_the_rebate_limit_is_inclusive_and_cites_clause_a():
    at = tax(other=str(LIMIT))
    assert at.rebate.provenance.citation == "156(2)(a)"
    assert at.payable.amount == 0


def test_one_step_above_the_rebate_limit_moves_to_marginal_relief():
    above = tax(other=str(LIMIT + STEP))
    assert above.rebate.provenance.citation == "156(2)(b)"
    assert above.slab.tax - above.rebate.amount == STEP


def test_income_that_rounds_down_to_the_limit_keeps_the_full_rebate():
    assert tax(other=str(LIMIT + 4)).rebate.provenance.citation == "156(2)(a)"
    assert tax(other=str(LIMIT + 5)).rebate.provenance.citation == "156(2)(b)"


def crossover():
    # The smallest whole-step income above the limit at which the slab tax no
    # longer exceeds the income above the limit, found by walking the data.
    income = LIMIT + STEP
    while slab_tax(income, RATES).tax > income - LIMIT:
        income += STEP
    return income


def test_marginal_relief_ends_exactly_where_the_tax_meets_the_excess():
    end = crossover()
    assert tax(other=str(end - STEP)).rebate is not None
    assert tax(other=str(end)).rebate is None


def test_the_rebate_is_the_maximum_or_the_tax_whichever_is_less_at_the_limit():
    assert tax(other=str(LIMIT)).rebate.amount == min(MAXIMUM, slab_tax(LIMIT, RATES).tax)


def test_no_rebate_line_is_written_when_there_is_no_tax_to_rebate():
    first_edge = EDGES[0]
    assert tax(other=str(first_edge)).rebate is None
    assert tax(other=str(first_edge + STEP)).rebate is not None


def test_a_non_resident_gets_no_rebate_on_either_side_of_the_limit():
    for income in (LIMIT - STEP, LIMIT, LIMIT + STEP, crossover() - STEP):
        assert tax(other=str(income), resident=False).rebate is None


# --- section 19(1) standard deduction -----------------------------------------


@pytest.mark.parametrize(
    ("name", "route"),
    [("standard_deduction_new_regime", "under_202_1"), ("standard_deduction_other", "opted_out")],
)
def test_the_standard_deduction_is_the_salary_up_to_its_cap_and_the_cap_beyond(name, route):
    cap = RATES.value(name).value

    def deducted(salary):
        result = compare_regimes(RATES, salary=salary, other_income=D(0), resident_individual=True)
        return getattr(result, route).standard_deduction.amount

    assert deducted(cap - 1) == cap - 1
    assert deducted(cap) == cap
    assert deducted(cap + 1) == cap
    assert deducted(D("0.01")) == D("0.01")


# --- section 123 cap and section 122(2) ----------------------------------------


CAP_123 = RATES.value("savings_insurance_deduction_cap").value


def test_a_claim_is_allowed_up_to_the_section_123_cap_and_no_further():
    gross = CAP_123 * 10
    for claim, allowed in ((CAP_123 - 1, CAP_123 - 1), (CAP_123, CAP_123), (CAP_123 + 1, CAP_123)):
        (line,) = opted_out(other=str(gross), deduction_savings_insurance=str(claim)).deductions
        assert (line.amount, line.provenance.citation) == (allowed, "123")


def test_section_122_2_binds_only_once_the_claim_exceeds_gross_total_income():
    claim = CAP_123
    (at,) = opted_out(other=str(claim), deduction_savings_insurance=str(claim)).deductions
    (below,) = opted_out(other=str(claim - 1), deduction_savings_insurance=str(claim)).deductions
    assert (at.amount, at.provenance.citation) == (claim, "123")
    assert (below.amount, below.provenance.citation) == (claim - 1, "122(2)")


def test_a_nil_claim_writes_no_deduction_line():
    assert opted_out(other="500000", deduction_savings_insurance="0").deductions == ()


def test_nil_gross_total_income_writes_no_deduction_line():
    assert opted_out(salary="50000", deduction_savings_insurance=str(CAP_123)).deductions == ()
    (line,) = opted_out(other="0.01", deduction_savings_insurance=str(CAP_123)).deductions
    assert (line.amount, line.provenance.citation) == (D("0.01"), "122(2)")


def test_a_nil_claim_is_not_recorded_as_a_disallowed_deduction():
    for name in RATES.not_allowed_under_202_1:
        assert tax("900000", **{name: "0"}).not_allowed == ()
        assert tax("900000", **{name: "1"}).not_allowed[0].amount == 1
