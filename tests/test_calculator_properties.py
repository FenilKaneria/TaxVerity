"""Step 9.5 — properties the Act guarantees across every income, checked with
generated inputs (ADR-105).

The hand-worked cases pin particular figures; these pin the shape the Act gives
the whole function. Marginal relief exists so that earning more never leaves a
person with less, and that is asserted directly rather than inferred from a
sweep. Incomes run from nil to ten crore, with paise.
"""

from __future__ import annotations

from decimal import Decimal

from hypothesis import given, settings
from hypothesis import strategies as st

from taxverity.calculator.comparison import compare_regimes
from taxverity.calculator.new_regime import new_regime_tax
from taxverity.calculator.rates import load_rates

D = Decimal
RATES = load_rates("2026-27")
STEP = RATES.value("rounding_multiple").value
STANDARD = RATES.value("standard_deduction_new_regime").value
TOP = D(100_000_000)

# Uniform draws over ten crore almost never land in the Rs. 70588 of marginal
# relief above twelve lakh: measured, removing relief entirely passed every
# property. So half the draws sit within a lakh of an edge the data file states.
EDGES = sorted(
    {slab.upper for slab in RATES.new_regime_slabs if slab.upper is not None}
    | {RATES.value("rebate_new_regime_income_limit").value, STANDARD}
)
NEAR = D(100_000)
anywhere = st.decimals(min_value=0, max_value=TOP, places=2, allow_nan=False, allow_infinity=False)
near_edge = st.builds(
    lambda edge, offset: max(D(0), edge + offset),
    st.sampled_from(EDGES),
    st.decimals(min_value=-NEAR, max_value=NEAR, places=2, allow_nan=False, allow_infinity=False),
)
amounts = st.one_of(near_edge, anywhere)
whole_steps = amounts.map(lambda amount: (amount / STEP).to_integral_value() * STEP)
small_steps = st.integers(min_value=0, max_value=int(NEAR / STEP)).map(lambda n: D(n) * STEP)
raises = st.one_of(small_steps, whole_steps)
# A pair of incomes either side of one edge. A raise from an arbitrary income
# rarely starts close enough below twelve lakh for a lost rebate to show:
# measured, with relief removed the pair form still passed all 300 examples.
straddles = st.builds(
    lambda edge, below, above: (max(D(0), edge - below), edge + above),
    st.sampled_from(EDGES),
    small_steps,
    small_steps,
)
pairs = st.one_of(straddles, st.tuples(whole_steps, raises).map(lambda pair: (pair[0], pair[0] + pair[1])))
residence = st.booleans()

# Derandomized: the examples CI checks are the same on every run, the same
# reproducibility the corpus pipeline holds itself to. No example database, so a
# failure found on one machine is not silently replayed only there.
PROPERTY = settings(derandomize=True, database=None, max_examples=300)


ZERO = D(0)


def payable(income, resident, salary=ZERO):
    return new_regime_tax(RATES, salary=salary, other_income=income, resident_individual=resident).payable.amount


@PROPERTY
@given(pairs, residence)
def test_tax_payable_never_falls_as_income_rises(pair, resident):
    low, high = pair
    assert payable(low, resident) <= payable(high, resident)


@PROPERTY
@given(pairs, residence)
def test_what_is_left_after_tax_never_falls_by_more_than_one_rounding_step(pair, resident):
    # Without marginal relief this fails just above twelve lakh, where one more
    # rupee of income would cost the whole Rs. 60000 rebate. Rounding the amount
    # payable to Rs. 10 can still cost one step, and no more. Incomes are whole
    # steps so that rounding total income adds nothing further.
    low, high = pair
    assert (high - payable(high, resident)) - (low - payable(low, resident)) >= -STEP


@PROPERTY
@given(amounts, residence)
def test_tax_never_exceeds_income(income, resident):
    assert payable(income, resident) <= income


@PROPERTY
@given(amounts)
def test_a_resident_individual_never_pays_more_than_a_non_resident(income):
    assert payable(income, True) <= payable(income, False)


@PROPERTY
@given(amounts, residence)
def test_salary_is_never_taxed_more_than_the_same_amount_of_other_income(income, resident):
    assert payable(D(0), resident, salary=income) <= payable(income, resident)


@PROPERTY
@given(st.decimals(min_value=STANDARD, max_value=TOP, places=2), amounts, residence)
def test_above_the_cap_salary_is_other_income_less_the_standard_deduction(salary, other, resident):
    as_salary = new_regime_tax(RATES, salary=salary, other_income=other, resident_individual=resident)
    as_other = new_regime_tax(RATES, salary=D(0), other_income=salary - STANDARD + other, resident_individual=resident)
    assert as_salary.payable == as_other.payable
    assert as_salary.slab == as_other.slab


@PROPERTY
@given(amounts, amounts, residence, st.dictionaries(st.sampled_from(sorted(RATES.not_allowed_under_202_1)), amounts))
def test_every_line_on_both_routes_carries_a_provenance_the_data_file_holds(salary, other, resident, claimed):
    held = set(RATES.provenances())
    result = compare_regimes(RATES, salary=salary, other_income=other, resident_individual=resident, claimed=claimed)
    lines = (*result.under_202_1.lines(), *result.under_202_1.not_allowed, *result.opted_out.deductions)
    standard = result.opted_out.standard_deduction
    assert all(line.provenance in held for line in (*lines, *([standard] if standard else [])))
    assert all(entry.provenance in held for entry in result.opted_out.undetermined)
    assert result.opted_out.tax in RATES.outside_act.values()


@PROPERTY
@given(amounts, amounts, residence, st.dictionaries(st.sampled_from(sorted(RATES.not_allowed_under_202_1)), amounts))
def test_a_claimed_chapter_viii_deduction_never_changes_the_202_1_tax(salary, other, resident, claimed):
    plain = new_regime_tax(RATES, salary=salary, other_income=other, resident_individual=resident)
    with_claims = new_regime_tax(RATES, salary=salary, other_income=other, resident_individual=resident, claimed=claimed)
    assert with_claims.lines() == plain.lines()


@PROPERTY
@given(amounts, amounts, amounts, raises)
def test_opted_out_total_income_stays_between_nil_and_gross_and_falls_with_a_larger_claim(salary, other, small, raise_):
    large = small + raise_

    def side(claim):
        return compare_regimes(
            RATES,
            salary=salary,
            other_income=other,
            resident_individual=True,
            claimed={"deduction_savings_insurance": claim},
        ).opted_out

    lesser, greater = side(small), side(large)
    for result in (lesser, greater):
        assert 0 <= result.total_income <= result.gross_total_income
    assert greater.total_income <= lesser.total_income


@PROPERTY
@given(amounts, amounts, raises)
def test_opted_out_total_income_never_falls_as_income_rises(salary, other, extra):
    def total(income):
        return compare_regimes(
            RATES,
            salary=salary,
            other_income=income,
            resident_individual=True,
            claimed={"deduction_savings_insurance": D(150000)},
        ).opted_out.total_income

    assert total(other) <= total(other + extra)


@PROPERTY
@given(amounts, amounts, amounts)
def test_no_section_123_line_exceeds_its_cap_or_gross_total_income(salary, other, claim):
    cap = RATES.value("savings_insurance_deduction_cap").value
    side = compare_regimes(
        RATES,
        salary=salary,
        other_income=other,
        resident_individual=True,
        claimed={"deduction_savings_insurance": claim},
    ).opted_out
    for line in side.deductions:
        assert line.amount <= min(cap, claim, side.gross_total_income)


@PROPERTY
@given(amounts, amounts, residence, st.dictionaries(st.sampled_from(sorted(RATES.not_allowed_under_202_1)), amounts))
def test_no_deduction_or_disallowance_line_is_written_for_nothing(salary, other, resident, claimed):
    # Only a rate band, a rounded amount or a nil payable may carry zero; a zero
    # deduction line states a figure no provision produced.
    result = compare_regimes(RATES, salary=salary, other_income=other, resident_individual=resident, claimed=claimed)
    for line in (*result.under_202_1.not_allowed, *result.opted_out.deductions):
        assert line.amount > 0


@PROPERTY
@given(amounts, amounts, amounts)
def test_opted_out_rounded_total_income_is_within_half_a_step_of_total_income(salary, other, claim):
    side = compare_regimes(
        RATES,
        salary=salary,
        other_income=other,
        resident_individual=True,
        claimed={"deduction_savings_insurance": claim},
    ).opted_out
    rounded = side.rounded_total_income.amount
    assert rounded % STEP == 0
    # Paise go first, then five goes up: 15 becomes 20 and 14.99 becomes 10.
    assert -STEP / 2 <= side.total_income - rounded < STEP / 2 + 1
