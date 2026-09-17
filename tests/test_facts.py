"""Step 7.5 — the user-fact schema.

Pure: no model, no network, no corpus. Every case is a payload the extractor
could plausibly return, including the ones Step 7.1 actually observed.
"""

from __future__ import annotations

from decimal import Decimal

import pytest
from pydantic import ValidationError

from taxverity.facts import (
    FACTS_JSON_SCHEMA,
    FIELDS,
    MODEL_STATUSES,
    Fact,
    FactField,
    FactIssue,
    FactStatus,
    Regime,
    ResidentialStatus,
    UserFacts,
    ValueKind,
    fact_value,
    normalise_value,
    parse_facts,
)

TURN = "I'm 41, salaried, earned 14,00,000 last year and paid 1,50,000 into PPF."


def entry(
    name: str = "salary_income",
    value: str = "1400000",
    status: str = "stated",
    span: str = "earned 14,00,000",
) -> dict:
    return {"name": name, "value": value, "status": status, "source_span": span}


def payload(*entries: dict) -> dict:
    return {"fields": list(entries or (entry(),))}


def issues(extraction) -> list[FactIssue]:
    return [rejection.issue for rejection in extraction.rejections]


# --- the field vocabulary --------------------------------------------------


def test_every_field_has_a_spec():
    assert set(FIELDS) == set(FactField)


def test_no_field_carries_1961_act_vocabulary():
    # The 2025 Act renumbers: the capped savings-and-insurance deduction is
    # section 123, not 80C. A field named for the repealed section would bake
    # the wrong statute into the schema.
    for field in FactField:
        assert "80" not in field.value


def test_declared_sections_are_the_2025_acts_own_numbers():
    assert FIELDS[FactField.DEDUCTION_SAVINGS_INSURANCE].section == "123"
    assert FIELDS[FactField.DEDUCTION_HEALTH_INSURANCE].section == "126"
    assert FIELDS[FactField.REGIME].section == "202(1)"
    assert FIELDS[FactField.ADVANCE_TAX_PAID].section == "403"


def test_step_9_1_confirmed_the_four_sections_step_7_5_left_open():
    assert FIELDS[FactField.RESIDENTIAL_STATUS].section == "6"
    assert FIELDS[FactField.BUSINESS_INCOME].section == "26"
    assert FIELDS[FactField.CAPITAL_GAINS_SHORT_TERM].section == "67"
    assert FIELDS[FactField.CAPITAL_GAINS_LONG_TERM].section == "67"
    assert FIELDS[FactField.TAX_YEAR].section == "3(1)"


def test_the_year_field_uses_the_acts_own_term():
    # The 2025 Act has no "assessment year" (ADR-100).
    assert "assessment_year" not in {field.value for field in FactField}
    assert FactField.TAX_YEAR.value == "tax_year"


def test_only_heads_of_income_may_be_negative():
    negative = {f for f in FactField if FIELDS[f].allows_negative}

    assert negative == {
        FactField.HOUSE_PROPERTY_INCOME,
        FactField.BUSINESS_INCOME,
        FactField.CAPITAL_GAINS_SHORT_TERM,
        FactField.CAPITAL_GAINS_LONG_TERM,
    }


def test_choice_fields_declare_their_domain():
    assert FIELDS[FactField.REGIME].choices == tuple(Regime)
    assert FIELDS[FactField.RESIDENTIAL_STATUS].choices == tuple(ResidentialStatus)


# --- the status enum -------------------------------------------------------


def test_a_model_may_not_assert_a_profile_default():
    assert FactStatus.PROFILE_DEFAULT not in MODEL_STATUSES
    statuses = FACTS_JSON_SCHEMA["properties"]["fields"]["items"]["properties"][
        "status"
    ]["enum"]

    assert statuses == ["stated", "inferred", "missing"]


def test_a_profile_default_in_a_payload_is_refused():
    extraction = parse_facts(payload(entry(status="profile_default")), TURN)

    assert extraction.facts.facts == ()
    assert issues(extraction) == [FactIssue.BAD_STATUS]


def test_a_profile_fact_may_be_constructed_but_quotes_no_span():
    fact = Fact(
        field=FactField.RESIDENTIAL_STATUS,
        status=FactStatus.PROFILE_DEFAULT,
        raw_value="resident",
        value="resident",
        source_span="",
    )

    assert fact.status is FactStatus.PROFILE_DEFAULT
    with pytest.raises(ValidationError, match="no span"):
        Fact(
            field=FactField.RESIDENTIAL_STATUS,
            status=FactStatus.PROFILE_DEFAULT,
            raw_value="resident",
            value="resident",
            source_span="I live in India",
        )


# --- value normalisation (Step 7.1's unstable formatting) ------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("1400000", Decimal("1400000")),
        ("14,00,000", Decimal("1400000")),
        ("1,400,000", Decimal("1400000")),
        ("Rs. 1,50,000", Decimal("150000")),
        ("₹150000", Decimal("150000")),
        ("INR 150000", Decimal("150000")),
        ("14 lakh", Decimal("1400000")),
        ("1.4 lakhs", Decimal("140000.0")),
        ("2 crore", Decimal("20000000")),
        ("-30000", Decimal("-30000")),
        ("2,50,000.50", Decimal("250000.50")),
    ],
)
def test_the_two_formats_one_run_produced_both_parse(raw, expected):
    assert normalise_value(ValueKind.MONEY, raw) == expected


def test_an_amount_is_a_decimal_never_a_float():
    value = normalise_value(ValueKind.MONEY, "0.1")

    assert isinstance(value, Decimal)
    assert value + Decimal("0.2") == Decimal("0.3")


@pytest.mark.parametrize("raw", ["", "a lot", "14,00,000 rupees odd", "Rs", "."])
def test_an_unparsable_amount_is_none(raw):
    assert normalise_value(ValueKind.MONEY, raw) is None


@pytest.mark.parametrize(
    ("raw", "expected"), [("2026-27", "2026-27"), ("AY 2026-2027", "2026-27")]
)
def test_a_year_range_normalises(raw, expected):
    assert normalise_value(ValueKind.YEAR_RANGE, raw) == expected


def test_a_year_range_that_is_not_consecutive_is_refused():
    assert normalise_value(ValueKind.YEAR_RANGE, "2026-29") is None


@pytest.mark.parametrize(
    ("raw", "span", "expected"),
    [
        ("2026-27", "tax year 2026-27", "2026-27"),
        ("2026-27", "for 2026-27", "2026-27"),
        ("2026-27", "assessment year 2026-27", "2025-26"),
        ("2025-26", "For A.Y. 2025-26", "2024-25"),
        ("AY 2026-27", "AY 2026-27", "2025-26"),
        ("2026-27", "ay2026-27", "2025-26"),
        ("2000-01", "Assessment Year 2000-01", "1999-00"),
        ("2026-27", "in May 2026-27", "2026-27"),
        ("2026-27", "assessment year of 2026-27", "2025-26"),
        ("2026-27", "A.Y.: 2026-27", "2025-26"),
        # The marker names another figure in the same span, so this one stays.
        ("2025-26", "for FY 2025-26 (AY 2026-27)", "2025-26"),
        ("2026-27", "AY 2027-28, that is tax year 2026-27", "2026-27"),
        ("2026-27", "tax year 2026-27, not the assessment year", "2026-27"),
        ("2027-28", "for FY 2026-27 (AY 2027-28)", "2026-27"),
    ],
)
def test_an_assessment_year_is_shifted_back_to_its_tax_year(raw, span, expected):
    assert fact_value(FactField.TAX_YEAR, raw, span) == expected


def test_the_shift_touches_no_other_field():
    assert fact_value(FactField.SALARY_INCOME, "800000", "A.Y. 2025-26 salary 800000") == Decimal(800000)


def test_a_parsed_assessment_year_comes_out_as_a_tax_year():
    turn = "For A.Y. 2025-26 my salary was 8,00,000."
    payload = {"fields": [entry("tax_year", "2025-26", "stated", "A.Y. 2025-26")]}

    fact = parse_facts(payload, turn).facts.get(FactField.TAX_YEAR)

    assert fact.value == "2024-25"
    assert fact.raw_value == "2025-26"


def test_a_choice_normalises_case_and_spacing():
    assert normalise_value(ValueKind.CHOICE, "Non Resident") == "non_resident"


def test_a_count_takes_digits_only():
    assert normalise_value(ValueKind.COUNT, "41") == 41
    assert normalise_value(ValueKind.COUNT, "forty-one") is None


# --- parsing ---------------------------------------------------------------


def test_a_clean_payload_parses():
    extraction = parse_facts(payload(entry()), TURN)
    fact = extraction.facts.get(FactField.SALARY_INCOME)

    assert extraction.rejections == ()
    assert fact.value == Decimal("1400000")
    assert fact.status is FactStatus.STATED
    assert fact.source_span == "earned 14,00,000"


def test_the_raw_value_is_kept_verbatim_beside_the_normalised_one():
    extraction = parse_facts(payload(entry(value="Rs. 14,00,000")), TURN)
    fact = extraction.facts.get(FactField.SALARY_INCOME)

    assert fact.raw_value == "Rs. 14,00,000"
    assert fact.value == Decimal("1400000")


def test_a_missing_fact_carries_no_value_and_no_span():
    extraction = parse_facts(
        payload(entry(name="age", value="", status="missing", span="")), TURN
    )
    fact = extraction.facts.get(FactField.AGE)

    assert extraction.rejections == ()
    assert fact.value is None
    assert fact.source_span == ""
    assert extraction.facts.missing() == (FactField.AGE,)


def test_an_inferred_fact_needs_no_span():
    extraction = parse_facts(
        payload(entry(name="regime", value="new", status="inferred", span="")), TURN
    )

    assert extraction.rejections == ()
    assert extraction.facts.get(FactField.REGIME).value == "new"


def test_a_stated_fact_without_a_span_is_refused():
    extraction = parse_facts(payload(entry(span="")), TURN)

    assert extraction.facts.facts == ()
    # Told apart from a fabricated span: 7.6 repairs the two differently.
    assert issues(extraction) == [FactIssue.MISSING_SPAN]


def test_a_fabricated_span_is_refused_not_downgraded():
    extraction = parse_facts(payload(entry(span="earned 25,00,000")), TURN)

    assert extraction.facts.facts == ()
    assert issues(extraction) == [FactIssue.SPAN_NOT_IN_TURN]


def test_a_span_differing_only_by_a_non_breaking_space_is_accepted():
    # The corpus carries NBSPs (Step 1.1) and pasted user text can too, so the
    # span check runs over normalise()d text on both sides.
    extraction = parse_facts(payload(entry(span="earned 14,00,000")), TURN)

    assert extraction.rejections == ()
    assert extraction.facts.get(FactField.SALARY_INCOME) is not None


def test_an_unknown_field_is_kept_verbatim_rather_than_dropped():
    extraction = parse_facts(payload(entry(name="gratuity_received")), TURN)

    assert extraction.facts.facts == ()
    assert issues(extraction) == [FactIssue.UNKNOWN_FIELD]
    assert extraction.facts.unmapped[0].name == "gratuity_received"
    assert extraction.facts.unmapped[0].raw_value == "1400000"


def test_a_repeated_field_keeps_the_first_and_reports_the_second():
    extraction = parse_facts(payload(entry(value="1400000"), entry(value="99")), TURN)

    assert extraction.facts.get(FactField.SALARY_INCOME).value == Decimal("1400000")
    assert issues(extraction) == [FactIssue.DUPLICATE_FIELD]


def test_an_unparsable_value_is_reported_against_its_entry():
    extraction = parse_facts(payload(entry(value="a lot")), TURN)

    assert issues(extraction) == [FactIssue.UNPARSABLE_VALUE]
    assert extraction.rejections[0].entry["value"] == "a lot"


def test_a_negative_amount_is_refused_where_the_field_forbids_it():
    extraction = parse_facts(payload(entry(value="-1400000")), TURN)

    assert issues(extraction) == [FactIssue.VALUE_OUT_OF_DOMAIN]


def test_a_loss_is_accepted_on_a_head_of_income():
    turn = "my rented flat lost 30000 this year"
    extraction = parse_facts(
        payload(entry(name="house_property_income", value="-30000", span="lost 30000")),
        turn,
    )

    assert extraction.facts.get(FactField.HOUSE_PROPERTY_INCOME).value == Decimal(
        "-30000"
    )


def test_a_positive_amount_quoted_as_a_loss_is_refused_not_flipped():
    turn = "my rented flat lost 30000 this year"
    extraction = parse_facts(
        payload(entry(name="house_property_income", value="30000", span="lost 30000")),
        turn,
    )

    assert issues(extraction) == [FactIssue.SIGN_CONTRADICTS_SPAN]
    assert extraction.facts.get(FactField.HOUSE_PROPERTY_INCOME) is None


def test_loss_words_do_not_touch_a_field_that_cannot_be_negative():
    turn = "I lost my job, but my salary was 600000"
    extraction = parse_facts(payload(entry(value="600000", span=turn)), turn)

    assert issues(extraction) == []


def test_loss_words_match_whole_words_only():
    turn = "Blossom Traders, my firm, earned 30000"
    extraction = parse_facts(
        payload(entry(name="business_income", value="30000", span=turn)), turn
    )

    assert issues(extraction) == []


def test_an_inferred_loss_head_has_no_span_to_check():
    extraction = parse_facts(
        payload(entry(name="business_income", value="30000", status="inferred", span="")),
        "my shop did fine after last year's losses",
    )

    assert issues(extraction) == []


def test_an_out_of_domain_choice_is_refused():
    extraction = parse_facts(
        payload(entry(name="regime", value="hybrid", status="inferred", span="")), TURN
    )

    assert issues(extraction) == [FactIssue.VALUE_OUT_OF_DOMAIN]


def test_an_impossible_age_is_refused():
    extraction = parse_facts(
        payload(entry(name="age", value="411", span="I'm 41")), TURN
    )

    assert issues(extraction) == [FactIssue.VALUE_OUT_OF_DOMAIN]


def test_one_bad_entry_does_not_cost_the_whole_turn():
    extraction = parse_facts(
        payload(entry(), entry(name="age", value="not a number", span="I'm 41")), TURN
    )

    assert extraction.facts.get(FactField.SALARY_INCOME) is not None
    assert issues(extraction) == [FactIssue.UNPARSABLE_VALUE]


@pytest.mark.parametrize(
    "bad", [None, [], {"fields": "salary"}, {"fields": ["salary"]}, {}]
)
def test_a_malformed_payload_is_reported_not_raised(bad):
    extraction = parse_facts(bad, TURN)

    assert extraction.facts.facts == ()
    assert FactIssue.MALFORMED_ENTRY in issues(extraction)


def test_an_unknown_status_is_reported():
    extraction = parse_facts(payload(entry(status="guessed")), TURN)

    assert issues(extraction) == [FactIssue.BAD_STATUS]


# --- the accessors Phase 9 and 9.7 use -------------------------------------


def test_known_excludes_missing_fields():
    extraction = parse_facts(
        payload(entry(), entry(name="age", value="", status="missing", span="")), TURN
    )

    assert {fact.field for fact in extraction.facts.known()} == {
        FactField.SALARY_INCOME
    }


def test_a_field_may_be_carried_at_most_once():
    fact = Fact(
        field=FactField.AGE,
        status=FactStatus.INFERRED,
        raw_value="41",
        value=41,
        source_span="",
    )
    with pytest.raises(ValidationError, match="at most once"):
        UserFacts(facts=(fact, fact))


def test_an_empty_userfacts_is_valid():
    facts = UserFacts()

    assert facts.known() == ()
    assert facts.missing() == ()
    assert facts.get(FactField.AGE) is None


# --- the schema handed to the model ----------------------------------------


def test_the_schema_names_every_field_and_refuses_an_invented_one():
    names = FACTS_JSON_SCHEMA["properties"]["fields"]["items"]["properties"]["name"]
    items = FACTS_JSON_SCHEMA["properties"]["fields"]["items"]

    assert names["enum"] == [field.value for field in FactField]
    assert items["additionalProperties"] is False
    assert set(items["required"]) == {"name", "value", "status", "source_span"}


def test_the_schema_carries_the_field_descriptions_the_model_needs():
    description = FACTS_JSON_SCHEMA["properties"]["fields"]["description"]

    assert "Act section 123" in description
    assert "negative for a loss" in description


# --- Step 7.8: the loss word is read in the span's clause (ADR-109) ---------


def sign_issues(turn, name, value, span):
    return issues(parse_facts(payload(entry(name=name, value=value, span=span)), turn))


@pytest.mark.parametrize(
    ("turn", "name", "span"),
    [
        ("My business made a loss of 3,00,000 last year.", "business_income", "3,00,000"),
        ("My tuition classes lost 60,000 during the year.", "business_income", "60,000"),
        ("Long-term capital gains: 2.5 lakh loss on my old plot.", "capital_gains_long_term", "2.5 lakh"),
        ("My let-out flat ran up a net loss of Rs. 1,60,000.", "house_property_income", "1,60,000"),
        ("Salary 14,20,000, business loss of 2,10,000 and TDS of 1,05,000.", "business_income", "2,10,000"),
        ("Rs. 90,000 was the deficit on my shop.", "business_income", "90,000"),
    ],
)
def test_a_positive_figure_in_a_loss_clause_is_refused_when_the_span_omits_the_word(turn, name, span):
    assert sign_issues(turn, name, "1", span) == [FactIssue.SIGN_CONTRADICTS_SPAN]


@pytest.mark.parametrize(
    ("turn", "name", "span"),
    [
        ("My house property income is 2,10,000, no loss this time.", "house_property_income", "2,10,000"),
        ("Last year's business loss is behind me; this year the business earned 3,40,000.", "business_income", "3,40,000"),
        ("No capital loss this time, and my long-term capital gain was 1,25,000.", "capital_gains_long_term", "1,25,000"),
        ("I lost my tenant for two months, but my house property income this year is 1,80,000.", "house_property_income", "1,80,000"),
        ("My shop made a profit of 2,20,000, unlike the losses of the last two years.", "business_income", "2,20,000"),
        ("We lost money. The business then earned 50,000.", "business_income", "50,000"),
        ("Last year was a loss but this year the shop earned 3,40,000.", "business_income", "3,40,000"),
    ],
)
def test_a_loss_word_in_a_neighbouring_clause_does_not_fire(turn, name, span):
    assert sign_issues(turn, name, "1", span) == []


def test_a_span_that_crosses_a_clause_is_read_whole():
    turn = "My house property income is 2,10,000, no loss this time."
    assert sign_issues(turn, "house_property_income", "210000", turn) == [FactIssue.SIGN_CONTRADICTS_SPAN]


def test_a_figure_repeated_in_a_loss_clause_and_a_gain_clause_is_refused():
    turn = "I made 50,000 on shares and lost 50,000 on bonds."
    assert sign_issues(turn, "capital_gains_short_term", "50000", "50,000") == [FactIssue.SIGN_CONTRADICTS_SPAN]


def test_a_negative_figure_in_a_loss_clause_is_accepted():
    turn = "My business made a loss of 3,00,000 last year."
    assert sign_issues(turn, "business_income", "-300000", "3,00,000") == []


# --- R20 Step 20.4: open-vocabulary situation facts --------------------------


def situation_entry(
    name: str = "rent recipient",
    value: str = "my mother",
    status: str = "stated",
    span: str = "pay rent to my mother",
) -> dict:
    return {"name": name, "value": value, "status": status, "source_span": span}


def situation_payload(*entries: dict) -> dict:
    return {"fields": [], "situation_facts": list(entries or (situation_entry(),))}


SITUATION_TURN = "I pay rent to my mother for a flat we both live in."


def test_a_situation_fact_parses_alongside_the_closed_fields():
    result = parse_facts(situation_payload(), SITUATION_TURN)
    assert result.situation_rejections == ()
    (fact,) = result.situation_facts
    assert fact.name == "rent recipient"
    assert fact.raw_value == "my mother"
    assert fact.status is FactStatus.STATED
    assert fact.source_span == "pay rent to my mother"


def test_a_missing_situation_facts_key_is_simply_empty():
    result = parse_facts({"fields": []}, SITUATION_TURN)
    assert result.situation_facts == ()
    assert result.situation_rejections == ()


def test_a_stated_situation_fact_without_a_span_is_refused():
    result = parse_facts(situation_payload(situation_entry(span="")), SITUATION_TURN)
    assert result.situation_facts == ()
    assert [r.issue for r in result.situation_rejections] == [FactIssue.MISSING_SPAN]


def test_a_fabricated_situation_span_is_refused():
    result = parse_facts(
        situation_payload(situation_entry(span="invented words")), SITUATION_TURN
    )
    assert result.situation_facts == ()
    assert [r.issue for r in result.situation_rejections] == [FactIssue.SPAN_NOT_IN_TURN]


def test_an_inferred_situation_fact_needs_no_span():
    result = parse_facts(
        situation_payload(situation_entry(status="inferred", span="")), SITUATION_TURN
    )
    (fact,) = result.situation_facts
    assert fact.status is FactStatus.INFERRED
    assert fact.source_span == ""


def test_a_profile_default_situation_status_is_refused():
    result = parse_facts(
        situation_payload(situation_entry(status="profile_default")), SITUATION_TURN
    )
    assert result.situation_facts == ()
    assert [r.issue for r in result.situation_rejections] == [FactIssue.BAD_STATUS]


def test_a_missing_situation_status_is_refused_there_is_no_missing_situation_fact():
    result = parse_facts(
        situation_payload(situation_entry(status="missing")), SITUATION_TURN
    )
    assert result.situation_facts == ()
    assert [r.issue for r in result.situation_rejections] == [FactIssue.BAD_STATUS]


def test_a_situation_fact_with_an_empty_name_or_value_is_refused():
    result = parse_facts(situation_payload(situation_entry(name="  ")), SITUATION_TURN)
    assert result.situation_facts == ()
    assert [r.issue for r in result.situation_rejections] == [FactIssue.MALFORMED_ENTRY]


def test_situation_facts_never_touch_the_closed_field_rejections():
    """A bad situation entry must not be treated as repairable by
    `llm/extract.py`'s closed-field repair pass."""
    result = parse_facts(
        situation_payload(situation_entry(span="")), SITUATION_TURN
    )
    assert result.rejections == ()
    assert result.situation_rejections != ()


def test_the_schema_requires_both_arrays():
    assert set(FACTS_JSON_SCHEMA["required"]) == {"fields", "situation_facts"}


def test_the_situation_schema_has_no_missing_status():
    statuses = FACTS_JSON_SCHEMA["properties"]["situation_facts"]["items"]["properties"][
        "status"
    ]["enum"]
    assert set(statuses) == {"stated", "inferred"}
