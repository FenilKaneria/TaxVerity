"""Step 9.6 — the out-of-scope detector routes rather than approximates (ADR-107)."""

from __future__ import annotations

from decimal import Decimal

import pytest

from taxverity.calculator.scope import (
    HEADS_NOT_COMPUTED,
    INPUT_FIELDS,
    SCOPE_STAGE_VERSION,
    Blocker,
    OutOfScope,
    Route,
    ScopeDecision,
    compute,
    route,
)
from taxverity.facts import Fact, FactField, FactStatus, UnmappedFact, UserFacts

D = Decimal
F = FactField

# Every input known: a resident with a salary, nothing else. The shape a
# complete extraction takes, so each test changes one thing from it.
BASE = {
    F.TAX_YEAR: "2026-27",
    F.RESIDENTIAL_STATUS: "resident",
    F.SALARY_INCOME: D("1000000"),
    F.HOUSE_PROPERTY_INCOME: D(0),
    F.BUSINESS_INCOME: D(0),
    F.CAPITAL_GAINS_SHORT_TERM: D(0),
    F.CAPITAL_GAINS_LONG_TERM: D(0),
    F.OTHER_SOURCES_INCOME: D(0),
    F.DEDUCTION_SAVINGS_INSURANCE: D(0),
    F.DEDUCTION_HEALTH_INSURANCE: D(0),
    F.DEDUCTION_OTHER: D(0),
    F.TDS_PAID: D(0),
    F.ADVANCE_TAX_PAID: D(0),
}


def fact(name, value, status=FactStatus.STATED):
    if status is FactStatus.MISSING:
        return Fact(field=name, status=status, raw_value="", value=None, source_span="")
    span = "" if status is FactStatus.PROFILE_DEFAULT else str(value)
    return Fact(field=name, status=status, raw_value=str(value), value=value, source_span=span)


def facts(unmapped=(), statuses=None, **changes):
    values = {**BASE, **{F(name): value for name, value in changes.items()}}
    statuses = statuses or {}
    return UserFacts(
        facts=tuple(
            fact(name, value, statuses.get(name, FactStatus.STATED))
            for name, value in values.items()
            if value is not None
        ),
        unmapped=tuple(unmapped),
    )


def test_the_stage_version_is_declared():
    assert SCOPE_STAGE_VERSION == 1


def test_regime_and_age_are_the_only_facts_that_are_not_inputs():
    assert set(FactField) - set(INPUT_FIELDS) == {F.REGIME, F.AGE}
    assert set(BASE) == set(INPUT_FIELDS)


def test_a_complete_salary_case_is_computed():
    decision = route(facts())
    assert decision.route is Route.COMPUTE
    assert decision.inputs.salary == D("1000000")
    assert decision.inputs.resident_individual is True
    assert decision.unknown == decision.blockers == decision.inferred == ()


@pytest.mark.parametrize("head", HEADS_NOT_COMPUTED)
@pytest.mark.parametrize("amount", ["250000", "-300000"])
def test_income_under_another_head_routes_text_only_gain_or_loss(head, amount):
    # A loss read with its sign dropped is still non-zero, so it routes out too.
    decision = route(facts(**{head.value: D(amount)}))
    assert decision.route is Route.TEXT_ONLY
    assert decision.blockers == (Blocker(OutOfScope.HEAD_NOT_COMPUTED, f"{head} is not computed", head),)
    assert decision.inputs is None


def test_a_stated_nil_under_another_head_is_not_income():
    assert route(facts(business_income=D(0))).route is Route.COMPUTE


def test_an_unnamed_deduction_routes_text_only():
    decision = route(facts(deduction_other=D("50000")))
    assert [b.reason for b in decision.blockers] == [OutOfScope.DEDUCTION_NOT_IDENTIFIED]


@pytest.mark.parametrize("year", ["2025-26", "2027-28"])
def test_a_tax_year_without_rate_data_routes_text_only(year):
    decision = route(facts(tax_year=year))
    assert [b.reason for b in decision.blockers] == [OutOfScope.UNSUPPORTED_TAX_YEAR]


def test_an_unmapped_fact_routes_text_only():
    unmapped = UnmappedFact(name="agricultural_income", raw_value="200000", source_span="2 lakh from farming")
    decision = route(facts(unmapped=[unmapped]))
    assert decision.route is Route.TEXT_ONLY
    assert decision.blockers[0].reason is OutOfScope.UNMAPPED_FACT
    assert decision.blockers[0].field is None


def test_every_blocker_is_reported_not_just_the_first():
    unmapped = UnmappedFact(name="hra", raw_value="1", source_span="1")
    decision = route(facts(unmapped=[unmapped], tax_year="2025-26", business_income=D(1), deduction_other=D(1)))
    assert {b.reason for b in decision.blockers} == set(OutOfScope)


def test_a_missing_input_makes_the_decision_incomplete():
    decision = route(facts(residential_status=None))
    assert decision.route is Route.INCOMPLETE
    assert decision.unknown == (F.RESIDENTIAL_STATUS,)
    assert decision.inputs is None


def test_a_missing_fact_entry_counts_as_unknown_like_an_absent_one():
    decision = route(facts(statuses={F.SALARY_INCOME: FactStatus.MISSING}))
    assert decision.unknown == (F.SALARY_INCOME,)


def test_text_only_wins_over_incomplete_and_still_reports_the_unknown():
    decision = route(facts(salary_income=None, capital_gains_long_term=D("100000")))
    assert decision.route is Route.TEXT_ONLY
    assert decision.unknown == (F.SALARY_INCOME,)


def test_an_empty_extraction_is_incomplete_on_every_input():
    decision = route(UserFacts())
    assert decision.route is Route.INCOMPLETE
    assert decision.unknown == INPUT_FIELDS


def test_a_profile_default_is_never_used_and_waits_for_confirmation():
    decision = route(facts(statuses={F.RESIDENTIAL_STATUS: FactStatus.PROFILE_DEFAULT}))
    assert decision.route is Route.INCOMPLETE
    assert decision.unconfirmed == (F.RESIDENTIAL_STATUS,)
    assert decision.unknown == (F.RESIDENTIAL_STATUS,)


def test_an_inferred_fact_is_used_and_named():
    decision = route(facts(statuses={F.RESIDENTIAL_STATUS: FactStatus.INFERRED}))
    assert decision.route is Route.COMPUTE
    assert decision.inferred == (F.RESIDENTIAL_STATUS,)


@pytest.mark.parametrize(
    ("status", "resident"),
    [("resident", True), ("resident_not_ordinarily_resident", True), ("non_resident", False)],
)
def test_residential_status_maps_onto_the_section_156_individual_resident(status, resident):
    assert route(facts(residential_status=status)).inputs.resident_individual is resident


@pytest.mark.parametrize("regime", ["old", "new"])
def test_regime_and_age_never_change_the_route(regime):
    extra = UserFacts(facts=(*facts().facts, fact(F.REGIME, regime), fact(F.AGE, 67)))
    assert route(extra).route is Route.COMPUTE


@pytest.mark.parametrize(
    "kwargs",
    [
        {"route": Route.TEXT_ONLY},
        {"route": Route.COMPUTE, "blockers": (Blocker(OutOfScope.UNMAPPED_FACT, "x"),)},
        {"route": Route.INCOMPLETE},
        {"route": Route.COMPUTE},
        {"route": Route.INCOMPLETE, "unknown": (F.AGE,), "unconfirmed": (F.REGIME,)},
    ],
)
def test_a_decision_that_contradicts_itself_cannot_be_built(kwargs):
    with pytest.raises(ValueError):
        ScopeDecision(**kwargs)


def test_only_a_computed_decision_can_be_computed():
    with pytest.raises(ValueError, match="incomplete"):
        compute(route(UserFacts()))


def test_compute_runs_the_comparison_and_settles_what_was_paid():
    # 1000000 salary less 75000 is 925000: 20000 to eight lakh, then 10% of
    # 125000 is 12500, so 32500 slab tax, all rebated below twelve lakh. The
    # 12000 deducted at source is all refunded.
    result = compute(route(facts(tds_paid=D("12000"))))
    assert result.comparison.under_202_1.payable.amount == D(0)
    assert result.settlement.refund_due.amount == D("12000")
    assert [c.provenance.citation for c in result.not_computed] == ["206(1)(c)(i)(B)", "206(1)(c)(i)(C)"]


def test_compute_passes_the_claims_through_to_both_routes():
    result = compute(route(facts(deduction_savings_insurance=D("150000"), deduction_health_insurance=D("20000"))))
    assert len(result.comparison.under_202_1.not_allowed) == 2
    assert result.comparison.opted_out.total_income == D("800000")
    assert result.comparison.opted_out.total_income_is_upper_bound
