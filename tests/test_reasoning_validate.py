"""R20 Step 20.5 — `reasoning/validate.py`'s deterministic checks. Adversarial
cases per PLAN R20's own verification list: a fabricated marker, an
ungrounded limit number, `satisfied` with no fact, a clarify question
carrying a number, a missing fact pointing at a satisfied condition.

Reuses `test_verifier.py`'s `PACK` (marker [1] = section 22(1), marker
[2] = section 24) rather than building a second evidence fixture — same
discipline `test_graph_paths.py` already uses, importing another test
module's constants instead of re-deriving them.
"""

from __future__ import annotations

from decimal import Decimal

from taxverity.facts import Fact, FactField, FactStatus, SituationFact, UserFacts
from taxverity.memory.fact_state import ThreadFactState, merge_turn
from taxverity.reasoning.models import (
    AnswerPlan,
    CheckStatus,
    ConclusionKind,
    Condition,
    ConditionCheck,
    LegalRule,
    MissingFact,
    ReasoningAnalysis,
)
from taxverity.reasoning.validate import MAX_QUESTION_LENGTH, validate
from test_verifier import PACK

FRESH = ThreadFactState()


def with_facts(*facts: Fact, situation: tuple[SituationFact, ...] = ()) -> ThreadFactState:
    return merge_turn(FRESH, UserFacts(facts=facts), turn=1, situation_facts=situation)


def plan(kind: ConclusionKind = ConclusionKind.DETERMINED, **kwargs) -> AnswerPlan:
    return AnswerPlan(conclusion_kind=kind, **kwargs)


def rule(
    id: str = "r1",
    markers: tuple[int, ...] = (2,),
    text: str = "The deduction is capped at Rs. 2,00,000.",
    **kwargs,
) -> LegalRule:
    return LegalRule(id=id, markers=markers, rule=text, **kwargs)


def analysis(**kwargs) -> ReasoningAnalysis:
    kwargs.setdefault("answer_plan", plan())
    return ReasoningAnalysis(**kwargs)


# --- markers must fall inside the pack ---------------------------------------


def test_a_rule_citing_a_marker_outside_the_pack_is_dropped():
    result = validate(analysis(legal_rules=(rule(markers=(99,)),)), PACK, fact_state=FRESH)
    assert result.legal_rules == ()
    assert result.has_governing_rule is False


def test_a_rule_with_one_valid_and_one_fabricated_marker_keeps_the_valid_one():
    result = validate(analysis(legal_rules=(rule(markers=(2, 99)),)), PACK, fact_state=FRESH)
    (survivor,) = result.legal_rules
    assert survivor.markers == (2,)


# --- numbers must be grounded in the cited units ------------------------------


def test_a_rule_stating_an_invented_number_is_dropped_whole():
    bad = rule(text="The deduction is capped at Rs. 9,99,999.")
    result = validate(analysis(legal_rules=(bad,)), PACK, fact_state=FRESH)
    assert result.legal_rules == ()


def test_a_rule_with_a_real_number_survives():
    result = validate(analysis(legal_rules=(rule(),)), PACK, fact_state=FRESH)
    assert len(result.legal_rules) == 1


def test_an_ungrounded_limit_is_trimmed_the_rule_survives():
    good_rule = rule(limits=("Capped at Rs. 2,00,000.", "Capped at Rs. 9,99,999."))
    (survivor,) = validate(analysis(legal_rules=(good_rule,)), PACK, fact_state=FRESH).legal_rules
    assert survivor.limits == ("Capped at Rs. 2,00,000.",)


def test_ungrounded_exceptions_and_definitions_are_trimmed_too():
    good_rule = rule(
        exceptions=("Except up to 9,99,999.",),
        definitions=("The figure is Rs. 2,00,000.",),
    )
    (survivor,) = validate(analysis(legal_rules=(good_rule,)), PACK, fact_state=FRESH).legal_rules
    assert survivor.exceptions == ()
    assert survivor.definitions == ("The figure is Rs. 2,00,000.",)


def test_a_condition_with_an_invented_number_is_dropped_the_rule_survives():
    good = Condition(id="c1", text="Applies to the standard case.", markers=(2,))
    bad = Condition(id="c2", text="Applies above Rs. 9,99,999.", markers=(2,))
    (survivor,) = validate(
        analysis(legal_rules=(rule(conditions=(good, bad)),)), PACK, fact_state=FRESH
    ).legal_rules
    assert [c.id for c in survivor.conditions] == ["c1"]


def test_a_condition_with_no_markers_inherits_the_rules_own_markers():
    inherited = Condition(id="c1", text="Capped at Rs. 2,00,000.", markers=())
    (survivor,) = validate(
        analysis(legal_rules=(rule(conditions=(inherited,)),)), PACK, fact_state=FRESH
    ).legal_rules
    assert len(survivor.conditions) == 1


def test_a_condition_whose_own_marker_is_fabricated_and_no_rule_fallback_applies():
    # markers=(99,) is genuinely fabricated, not "empty", so it is NOT treated
    # as "inherit the rule's markers" — it is filtered to nothing and dropped.
    orphan = Condition(id="c1", text="Capped at Rs. 2,00,000.", markers=(99,))
    (survivor,) = validate(
        analysis(legal_rules=(rule(conditions=(orphan,)),)), PACK, fact_state=FRESH
    ).legal_rules
    assert survivor.conditions == ()


# --- a satisfied/not_satisfied check needs a real fact ------------------------


def test_satisfied_with_no_fact_ref_is_downgraded_to_unknown():
    good_rule = rule(conditions=(Condition(id="c1", text="x", markers=(2,)),))
    check = ConditionCheck(condition_id="c1", status=CheckStatus.SATISFIED, fact_refs=())
    result = validate(analysis(legal_rules=(good_rule,), applicability=(check,)), PACK, fact_state=FRESH)
    (survivor,) = result.applicability
    assert survivor.status is CheckStatus.UNKNOWN


def test_satisfied_with_a_fact_ref_naming_an_unknown_fact_is_downgraded():
    good_rule = rule(conditions=(Condition(id="c1", text="x", markers=(2,)),))
    check = ConditionCheck(
        condition_id="c1", status=CheckStatus.SATISFIED, fact_refs=("some_unrelated_thing",)
    )
    result = validate(analysis(legal_rules=(good_rule,), applicability=(check,)), PACK, fact_state=FRESH)
    (survivor,) = result.applicability
    assert survivor.status is CheckStatus.UNKNOWN


def test_satisfied_with_a_real_closed_fact_ref_survives():
    state = with_facts(
        Fact(
            field=FactField.SALARY_INCOME,
            status=FactStatus.STATED,
            raw_value="1400000",
            value=Decimal("1400000"),
            source_span="salary 1400000",
        )
    )
    good_rule = rule(conditions=(Condition(id="c1", text="x", markers=(2,)),))
    check = ConditionCheck(
        condition_id="c1", status=CheckStatus.SATISFIED, fact_refs=("salary_income",)
    )
    result = validate(analysis(legal_rules=(good_rule,), applicability=(check,)), PACK, fact_state=state)
    (survivor,) = result.applicability
    assert survivor.status is CheckStatus.SATISFIED


def test_satisfied_with_a_situation_fact_ref_survives_case_insensitively():
    state = with_facts(
        situation=(
            SituationFact(
                name="Rent Recipient",
                status=FactStatus.STATED,
                raw_value="my mother",
                source_span="pay rent to my mother",
            ),
        )
    )
    good_rule = rule(conditions=(Condition(id="c1", text="x", markers=(2,)),))
    check = ConditionCheck(
        condition_id="c1", status=CheckStatus.NOT_SATISFIED, fact_refs=("  RENT RECIPIENT  ",)
    )
    result = validate(analysis(legal_rules=(good_rule,), applicability=(check,)), PACK, fact_state=state)
    (survivor,) = result.applicability
    assert survivor.status is CheckStatus.NOT_SATISFIED


def test_unknown_and_ambiguous_checks_need_no_fact_ref():
    good_rule = rule(conditions=(Condition(id="c1", text="x", markers=(2,)),))
    check = ConditionCheck(condition_id="c1", status=CheckStatus.AMBIGUOUS, fact_refs=())
    result = validate(analysis(legal_rules=(good_rule,), applicability=(check,)), PACK, fact_state=FRESH)
    (survivor,) = result.applicability
    assert survivor.status is CheckStatus.AMBIGUOUS


def test_a_check_naming_a_condition_that_did_not_survive_is_dropped():
    check = ConditionCheck(condition_id="ghost", status=CheckStatus.UNKNOWN)
    result = validate(analysis(applicability=(check,)), PACK, fact_state=FRESH)
    assert result.applicability == ()


# --- missing facts must point at a surviving unknown condition ---------------


def test_a_missing_fact_pointing_at_a_satisfied_condition_is_dropped():
    state = with_facts(
        Fact(
            field=FactField.SALARY_INCOME,
            status=FactStatus.STATED,
            raw_value="1400000",
            value=Decimal("1400000"),
            source_span="salary 1400000",
        )
    )
    good_rule = rule(conditions=(Condition(id="c1", text="x", markers=(2,)),))
    check = ConditionCheck(
        condition_id="c1", status=CheckStatus.SATISFIED, fact_refs=("salary_income",)
    )
    missing = MissingFact(condition_id="c1", question="What is your salary?", material=True)
    result = validate(
        analysis(legal_rules=(good_rule,), applicability=(check,), missing_facts=(missing,)),
        PACK,
        fact_state=state,
    )
    assert result.missing_facts == ()


def test_a_missing_fact_pointing_at_a_surviving_unknown_condition_survives():
    good_rule = rule(conditions=(Condition(id="c1", text="x", markers=(2,)),))
    check = ConditionCheck(condition_id="c1", status=CheckStatus.UNKNOWN)
    missing = MissingFact(condition_id="c1", question="Who did you pay the rent to?", material=True)
    result = validate(
        analysis(legal_rules=(good_rule,), applicability=(check,), missing_facts=(missing,)),
        PACK,
        fact_state=FRESH,
    )
    assert [m.question for m in result.missing_facts] == ["Who did you pay the rent to?"]


def test_a_clarify_question_carrying_an_invented_legal_number_is_dropped():
    good_rule = rule(conditions=(Condition(id="c1", text="x", markers=(2,)),))
    check = ConditionCheck(condition_id="c1", status=CheckStatus.UNKNOWN)
    missing = MissingFact(
        condition_id="c1", question="Is the amount above Rs. 9,99,999?", material=True
    )
    result = validate(
        analysis(legal_rules=(good_rule,), applicability=(check,), missing_facts=(missing,)),
        PACK,
        fact_state=FRESH,
    )
    assert result.missing_facts == ()


def test_a_clarify_question_asking_for_the_persons_own_figure_is_allowed():
    """Amendment 3: the question may ask the person for their own number —
    it is only forbidden to *state* a legal number the pack doesn't ground."""
    good_rule = rule(conditions=(Condition(id="c1", text="x", markers=(2,)),))
    check = ConditionCheck(condition_id="c1", status=CheckStatus.UNKNOWN)
    missing = MissingFact(condition_id="c1", question="How much rent did you pay?", material=True)
    result = validate(
        analysis(legal_rules=(good_rule,), applicability=(check,), missing_facts=(missing,)),
        PACK,
        fact_state=FRESH,
    )
    assert len(result.missing_facts) == 1


def test_a_long_clarify_question_is_capped_not_dropped():
    good_rule = rule(conditions=(Condition(id="c1", text="x", markers=(2,)),))
    check = ConditionCheck(condition_id="c1", status=CheckStatus.UNKNOWN)
    long_question = "Who did you pay it to, " * 30
    missing = MissingFact(condition_id="c1", question=long_question, material=True)
    result = validate(
        analysis(legal_rules=(good_rule,), applicability=(check,), missing_facts=(missing,)),
        PACK,
        fact_state=FRESH,
    )
    (survivor,) = result.missing_facts
    assert len(survivor.question) <= MAX_QUESTION_LENGTH + 1  # +1 for the ellipsis
    assert survivor.question.endswith("…")


# --- the answer plan -----------------------------------------------------------


def test_a_plan_step_with_an_invented_number_is_dropped_others_kept():
    good_plan = plan(
        steps=("The cap is Rs. 2,00,000.", "It applies above Rs. 9,99,999."),
        next_step="Confirm the amount claimed.",
    )
    result = validate(analysis(answer_plan=good_plan), PACK, fact_state=FRESH)
    assert result.answer_plan.steps == ("The cap is Rs. 2,00,000.",)


def test_a_plan_next_step_with_an_invented_number_is_blanked():
    good_plan = plan(next_step="Only above Rs. 9,99,999.")
    result = validate(analysis(answer_plan=good_plan), PACK, fact_state=FRESH)
    assert result.answer_plan.next_step == ""


def test_a_plan_number_grounded_in_a_stated_fact_survives():
    state = with_facts(
        Fact(
            field=FactField.SALARY_INCOME,
            status=FactStatus.STATED,
            raw_value="1234567",
            value=Decimal("1234567"),
            source_span="salary 1234567",
        )
    )
    good_plan = plan(next_step="Your salary of 1234567 is above the threshold.")
    result = validate(analysis(answer_plan=good_plan), PACK, fact_state=state)
    assert result.answer_plan.next_step == good_plan.next_step


# --- overall fallback ---------------------------------------------------------


def test_no_legal_rules_at_all_means_no_governing_rule():
    result = validate(analysis(), PACK, fact_state=FRESH)
    assert result.has_governing_rule is False


def test_a_no_basis_plan_with_no_rules_is_allowed_through_unchanged():
    result = validate(analysis(answer_plan=plan(ConclusionKind.NO_BASIS)), PACK, fact_state=FRESH)
    assert result.answer_plan.conclusion_kind is ConclusionKind.NO_BASIS
    assert result.has_governing_rule is False
