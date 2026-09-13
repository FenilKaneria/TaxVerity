"""Step 9.7 — unknown facts are asked about only when arithmetic says so (ADR-108)."""

from __future__ import annotations

from decimal import Decimal

import pytest

from taxverity.calculator.materiality import (
    CLAIM_FIELDS,
    INCOME_FIELDS,
    MATERIALITY_STAGE_VERSION,
    PAID_FIELDS,
    ROUTE_CHANGING_FIELDS,
    Finding,
    Outcome,
    Probe,
    Reason,
    probe,
)
from taxverity.calculator.new_regime import new_regime_tax
from taxverity.calculator.rates import load_rates
from taxverity.calculator.scope import INPUT_FIELDS, Route, route, run
from taxverity.facts import FactField, FactStatus, UnmappedFact, UserFacts
from test_scope import BASE, facts

D = Decimal
F = FactField


def probed(**changes):
    extracted = facts(**changes)
    return probe(extracted, route(extracted))


def outcomes(result):
    return {finding.field: finding.outcome for finding in result.findings}


def test_the_stage_version_is_declared():
    assert MATERIALITY_STAGE_VERSION == 1


def test_every_input_field_has_exactly_one_policy():
    classes = (INCOME_FIELDS, ROUTE_CHANGING_FIELDS, CLAIM_FIELDS, PAID_FIELDS)
    policed = [name for group in classes for name in group]
    policed += [F.TAX_YEAR, F.RESIDENTIAL_STATUS]
    assert sorted(policed) == sorted(INPUT_FIELDS)


def test_an_empty_extraction_asks_for_income_and_each_head():
    result = probe(UserFacts(), route(UserFacts()))
    assert result.inputs is None
    assert outcomes(result) == {
        F.TAX_YEAR: Outcome.ASSUME,
        F.RESIDENTIAL_STATUS: Outcome.DEFERRED,
        F.SALARY_INCOME: Outcome.ASK,
        F.OTHER_SOURCES_INCOME: Outcome.ASK,
        F.HOUSE_PROPERTY_INCOME: Outcome.ASK,
        F.BUSINESS_INCOME: Outcome.ASK,
        F.CAPITAL_GAINS_SHORT_TERM: Outcome.ASK,
        F.CAPITAL_GAINS_LONG_TERM: Outcome.ASK,
        F.DEDUCTION_OTHER: Outcome.ASK,
        F.DEDUCTION_SAVINGS_INSURANCE: Outcome.ASSUME,
        F.DEDUCTION_HEALTH_INSURANCE: Outcome.ASSUME,
        F.TDS_PAID: Outcome.NOT_COMPUTED,
        F.ADVANCE_TAX_PAID: Outcome.NOT_COMPUTED,
    }


@pytest.mark.parametrize("name", INPUT_FIELDS)
def test_only_unknown_fields_are_probed_and_a_stated_one_never_is(name):
    result = probed(**{name.value: None})
    assert [finding.field for finding in result.findings] == [name]


def test_a_computed_decision_passes_through_unprobed():
    result = probed()
    assert result.findings == ()
    assert result.inputs == route(facts()).inputs


def test_a_text_only_decision_is_refused():
    extracted = facts(business_income=D("1"))
    with pytest.raises(ValueError, match="text-only"):
        probe(extracted, route(extracted))


@pytest.mark.parametrize("name", INCOME_FIELDS)
def test_unknown_income_is_asked_because_no_sweep_can_bound_it(name):
    result = probed(**{name.value: None})
    (finding,) = result.findings
    assert (finding.outcome, finding.reason) == (Outcome.ASK, Reason.INCOME_UNBOUNDED)
    assert result.inputs is None


@pytest.mark.parametrize("name", ROUTE_CHANGING_FIELDS)
def test_an_unknown_head_or_unnamed_deduction_is_asked_on_its_own(name):
    (finding,) = probed(**{name.value: None}).findings
    assert (finding.outcome, finding.reason) == (Outcome.ASK, Reason.CHANGES_ROUTE)


def test_unknown_heads_are_asked_one_question_each():
    result = probed(**{name.value: None for name in ROUTE_CHANGING_FIELDS})
    assert [f.field for f in result.by_outcome(Outcome.ASK)] == list(ROUTE_CHANGING_FIELDS)


@pytest.mark.parametrize("name", CLAIM_FIELDS)
def test_an_unknown_claim_is_assumed_nil_because_202_1_ignores_it(name):
    result = probed(**{name.value: None})
    (finding,) = result.findings
    assert (finding.outcome, finding.assumed) == (Outcome.ASSUME, D(0))
    assert result.inputs.claimed[name.value] == D(0)


@pytest.mark.parametrize("amount", ["1", "150000", "9999999"])
def test_no_claimed_amount_moves_the_202_1_tax(amount):
    # The reason the claim fields are never asked: if this fails, they must be.
    nil = run(probed(**{n.value: None for n in CLAIM_FIELDS}).inputs)
    claimed = run(route(facts(**{n.value: D(amount) for n in CLAIM_FIELDS})).inputs)
    assert nil.comparison.under_202_1.payable == claimed.comparison.under_202_1.payable


@pytest.mark.parametrize(
    "unknown",
    [(F.TDS_PAID,), (F.ADVANCE_TAX_PAID,), PAID_FIELDS],
)
def test_unknown_tax_paid_computes_the_tax_but_no_balance(unknown):
    result = probed(**{name.value: None for name in unknown})
    assert {f.outcome for f in result.findings} == {Outcome.NOT_COMPUTED}
    computation = run(result.inputs)
    assert computation.settlement is None
    assert computation.comparison.under_202_1.payable.amount == D(0)


def test_an_unknown_tax_year_is_the_only_year_with_data():
    result = probed(tax_year=None)
    (finding,) = result.findings
    assert (finding.reason, finding.assumed) == (Reason.ONLY_TAX_YEAR_WITH_DATA, "2026-27")
    assert result.inputs.tax_year == "2026-27"


def test_residential_status_waits_until_income_is_known():
    result = probed(residential_status=None, other_sources_income=None)
    by_field = {f.field: f for f in result.findings}
    assert by_field[F.RESIDENTIAL_STATUS].outcome is Outcome.DEFERRED
    assert by_field[F.RESIDENTIAL_STATUS].spread is None


def test_residential_status_is_asked_when_the_rebate_is_at_stake():
    # Ten lakh salary is 925000 total income: 32500 slab tax, all rebated for a
    # resident and none for a non-resident.
    (finding,) = probed(residential_status=None).findings
    assert (finding.outcome, finding.reason) == (Outcome.ASK, Reason.TAX_DIFFERS)
    assert finding.spread == (D(0), D("32500"))


@pytest.mark.parametrize("salary", ["0", "300000", "475000", "3000000"])
def test_residential_status_is_assumed_where_the_rebate_cannot_apply(salary):
    result = probed(residential_status=None, salary_income=D(salary))
    (finding,) = result.findings
    assert finding.reason is Reason.TAX_SAME
    assert finding.spread[0] == finding.spread[1]
    assert result.inputs.resident_individual is False


@pytest.mark.parametrize(
    "salary",
    ["475001", "800000", "1275000", "1285000", "1320000", "1345000", "1400000"],
)
def test_the_sweep_agrees_with_computing_both_statuses_directly(salary):
    # Straddles the rebate limit and the end of marginal relief, where a
    # hand-picked threshold would drift from the Act's arithmetic.
    rates = load_rates("2026-27")
    taxes = {
        new_regime_tax(rates, salary=D(salary), other_income=D(0), resident_individual=resident).payable.amount
        for resident in (True, False)
    }
    (finding,) = probed(residential_status=None, salary_income=D(salary)).findings
    assert (finding.reason is Reason.TAX_DIFFERS) == (len(taxes) == 2)
    assert finding.spread == (min(taxes), max(taxes))


def test_a_profile_default_is_probed_but_never_used_and_is_flagged():
    extracted = facts(
        salary_income=D("3000000"),
        statuses={F.RESIDENTIAL_STATUS: FactStatus.PROFILE_DEFAULT},
    )
    result = probe(extracted, route(extracted))
    (finding,) = result.findings
    assert finding.unconfirmed_profile_default
    # The profile says resident; the probe found it immaterial and did not use it.
    assert result.inputs.resident_individual is False


def test_an_inferred_fact_is_used_as_a_value_in_the_sweep():
    extracted = facts(
        residential_status=None,
        statuses={F.SALARY_INCOME: FactStatus.INFERRED},
    )
    (finding,) = probe(extracted, route(extracted)).findings
    assert finding.spread == (D(0), D("32500"))


def test_a_probed_result_is_computed_with_its_assumptions():
    changes = {n.value: None for n in (*CLAIM_FIELDS, *PAID_FIELDS)}
    result = probed(tax_year=None, residential_status=None, salary_income=D("3000000"), **changes)
    assert {f.outcome for f in result.findings} == {Outcome.ASSUME, Outcome.NOT_COMPUTED}
    computation = run(result.inputs)
    assert computation.settlement is None
    assert computation.comparison.under_202_1.rebate is None


def test_unmapped_facts_never_reach_the_probe():
    unmapped = UnmappedFact(name="agricultural_income", raw_value="1", source_span="1")
    extracted = facts(unmapped=[unmapped], salary_income=None)
    assert route(extracted).route is Route.TEXT_ONLY


@pytest.mark.parametrize(
    "kwargs",
    [
        {"field": F.RESIDENTIAL_STATUS, "reason": Reason.TAX_DIFFERS},
        {"field": F.RESIDENTIAL_STATUS, "reason": Reason.TAX_DIFFERS, "spread": (D(1), D(1))},
        {"field": F.RESIDENTIAL_STATUS, "reason": Reason.TAX_SAME, "spread": (D(0), D(1)), "assumed": "x"},
        {"field": F.SALARY_INCOME, "reason": Reason.INCOME_UNBOUNDED, "spread": (D(0), D(0))},
        {"field": F.TAX_YEAR, "reason": Reason.ONLY_TAX_YEAR_WITH_DATA},
        {"field": F.SALARY_INCOME, "reason": Reason.INCOME_UNBOUNDED, "assumed": D(0)},
    ],
)
def test_a_finding_that_contradicts_itself_cannot_be_built(kwargs):
    with pytest.raises(ValueError):
        Finding(**kwargs)


def test_a_probe_cannot_carry_inputs_while_a_question_is_open():
    asked = Finding(field=F.SALARY_INCOME, reason=Reason.INCOME_UNBOUNDED)
    with pytest.raises(ValueError):
        Probe(findings=(asked,), inputs=route(facts()).inputs)
    with pytest.raises(ValueError):
        Probe(findings=(), inputs=None)


def test_base_is_the_complete_case_the_probe_starts_from():
    assert set(BASE) == set(INPUT_FIELDS)
