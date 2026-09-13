"""Step 9.1 — the rate data (ADR-100). Pure tests on the loader, then the
provenance check: every quote is a verbatim substring of the chunk it cites."""

from __future__ import annotations

import copy
import json
import re
from decimal import Decimal

import pytest

from taxverity.calculator.rates import (
    RATES_DIR,
    RATES_STAGE_VERSION,
    OutsideActError,
    RatesError,
    Unit,
    UnsupportedTaxYearError,
    load_rates,
    parse_rates,
    supported_tax_years,
)
from taxverity.corpus.loader import normalise
from taxverity.facts import FIELDS

OUTSIDE_ACT = {
    "old_regime_slabs",
    "age_based_exemption_limits",
    "surcharge",
    "marginal_relief_on_surcharge",
    "health_and_education_cess",
}


@pytest.fixture(scope="module")
def payload():
    return json.loads(
        (RATES_DIR / "tax_year_2026_27.json").read_text(encoding="utf-8")
    )


@pytest.fixture(scope="module")
def rates():
    return load_rates("2026-27")


def test_the_stage_version_is_declared():
    assert RATES_STAGE_VERSION == 1


def test_the_only_supported_tax_year_is_the_first_one_the_act_governs():
    assert supported_tax_years() == ("2026-27",)


def test_a_tax_year_without_data_is_refused():
    with pytest.raises(UnsupportedTaxYearError, match="2025-26"):
        load_rates("2025-26")


@pytest.mark.parametrize("year", ["2026", "AY 2026-27", "2026-2027", ""])
def test_a_malformed_tax_year_is_refused(year):
    with pytest.raises(UnsupportedTaxYearError):
        load_rates(year)


def test_the_tax_year_does_not_begin_before_the_act_commences(rates):
    assert "1st April, 2026" in rates.commencement.source_text
    assert int(rates.tax_year[:4]) >= 2026


def test_the_new_regime_slabs_are_the_seven_rows_of_section_202_1(rates):
    assert [(s.lower, s.upper, s.rate_percent) for s in rates.new_regime_slabs] == [
        (Decimal(0), Decimal(400000), Decimal(0)),
        (Decimal(400000), Decimal(800000), Decimal(5)),
        (Decimal(800000), Decimal(1200000), Decimal(10)),
        (Decimal(1200000), Decimal(1600000), Decimal(15)),
        (Decimal(1600000), Decimal(2000000), Decimal(20)),
        (Decimal(2000000), Decimal(2400000), Decimal(25)),
        (Decimal(2400000), None, Decimal(30)),
    ]
    assert {s.provenance.citation for s in rates.new_regime_slabs} == {"202(1)"}


def test_every_amount_is_a_decimal(rates):
    for slab in rates.new_regime_slabs:
        assert isinstance(slab.lower, Decimal)
        assert isinstance(slab.rate_percent, Decimal)
    for entry in rates.values.values():
        assert isinstance(entry.value, Decimal)


@pytest.mark.parametrize(
    ("name", "value", "citation"),
    [
        ("standard_deduction_new_regime", 75000, "19(1)"),
        ("standard_deduction_other", 50000, "19(1)"),
        ("rebate_new_regime_income_limit", 1200000, "156(2)(a)"),
        ("rebate_new_regime_maximum", 60000, "156(2)(a)"),
        ("rebate_other_income_limit", 500000, "156(1)"),
        ("rebate_other_maximum", 12500, "156(1)"),
        ("savings_insurance_deduction_cap", 150000, "123"),
        ("senior_citizen_age", 60, "2(100)"),
        ("rounding_multiple", 10, "516"),
        ("rebate_new_regime_marginal_relief_threshold", 1200000, "156(2)(b)"),
    ],
)
def test_a_named_value_carries_its_section(rates, name, value, citation):
    entry = rates.value(name)
    assert entry.value == Decimal(value)
    assert entry.provenance.citation == citation


def test_what_the_act_does_not_print_is_declared_outside_it(rates):
    assert set(rates.outside_act) == OUTSIDE_ACT


@pytest.mark.parametrize("name", sorted(OUTSIDE_ACT))
def test_asking_for_a_value_outside_the_act_raises_rather_than_answering(rates, name):
    with pytest.raises(OutsideActError) as raised:
        rates.value(name)
    assert raised.value.component == name
    assert raised.value.citation == rates.outside_act[name].provenance.citation


def test_an_unknown_rate_is_a_key_error_not_an_outside_act_one(rates):
    with pytest.raises(KeyError):
        rates.value("surcharg")


def test_number_words_are_read_back(rates):
    assert rates.value("rebate_new_regime_income_limit").unit is Unit.RUPEES
    assert "twelve lakh rupees" in rates.value("rebate_new_regime_income_limit").provenance.source_text
    assert rates.value("senior_citizen_age").unit is Unit.YEARS


# --- refusals ----------------------------------------------------------------


def edited(payload, change):
    copied = copy.deepcopy(payload)
    change(copied)
    return copied


def test_a_value_that_is_not_what_its_quote_says_is_refused(payload):
    bad = edited(payload, lambda p: p["values"]["rebate_new_regime_maximum"].update(value="65000"))
    with pytest.raises(RatesError, match="not what its quote says"):
        parse_rates(bad)


def test_a_number_word_that_disagrees_with_the_value_is_refused(payload):
    bad = edited(payload, lambda p: p["values"]["senior_citizen_age"].update(value="65"))
    with pytest.raises(RatesError, match="not what its quote says"):
        parse_rates(bad)


def test_a_json_number_is_refused_because_it_was_a_float_first(payload):
    bad = edited(payload, lambda p: p["values"]["rebate_other_maximum"].update(value=12500))
    with pytest.raises(RatesError, match="must be a string"):
        parse_rates(bad)


def test_a_slab_that_disagrees_with_its_row_is_refused(payload):
    bad = edited(payload, lambda p: p["new_regime_slabs"][1].update(rate_percent="6"))
    with pytest.raises(RatesError, match="slab 1"):
        parse_rates(bad)


def test_a_gap_between_slabs_is_refused(payload):
    bad = edited(payload, lambda p: p["new_regime_slabs"].pop(2))
    with pytest.raises(RatesError, match="slab 2"):
        parse_rates(bad)


def test_an_open_ended_slab_below_the_top_is_refused(payload):
    bad = edited(payload, lambda p: p["new_regime_slabs"][0].update(upper=None))
    with pytest.raises(RatesError, match="open-ended"):
        parse_rates(bad)


def test_a_capped_top_slab_is_refused(payload):
    bad = edited(payload, lambda p: p["new_regime_slabs"][-1].update(upper="9999999"))
    with pytest.raises(RatesError, match="open-ended"):
        parse_rates(bad)


def test_an_entry_without_a_quote_is_refused(payload):
    bad = edited(payload, lambda p: p["values"]["savings_insurance_deduction_cap"].pop("source_text"))
    with pytest.raises(RatesError, match="citation or source text"):
        parse_rates(bad)


def test_a_component_cannot_be_both_supported_and_outside_the_act(payload):
    def both(p):
        p["values"]["surcharge"] = copy.deepcopy(p["values"]["rebate_other_maximum"])

    with pytest.raises(RatesError, match="both supported and outside"):
        parse_rates(edited(payload, both))


def test_an_outside_act_entry_cannot_smuggle_a_number(payload):
    def smuggle(p):
        p["outside_act"]["health_and_education_cess"]["source_text"] = "cess at 4%"

    with pytest.raises(RatesError, match="carries a number"):
        parse_rates(edited(payload, smuggle))


def test_the_data_file_is_canonical(payload):
    text = (RATES_DIR / "tax_year_2026_27.json").read_text(encoding="utf-8")
    assert text == json.dumps(payload, sort_keys=True, ensure_ascii=False, indent=1) + "\n"


# --- provenance against the corpus ------------------------------------------


@pytest.fixture(scope="module")
def chunk_text(stored_chunks):
    _, chunks = stored_chunks
    return {chunk.node_path: normalise(chunk.text) for chunk in chunks}


def test_every_quote_is_verbatim_in_the_chunk_it_cites(rates, chunk_text):
    for provenance in rates.provenances():
        assert provenance.citation in chunk_text, provenance.citation
        assert normalise(provenance.source_text) in chunk_text[provenance.citation], (
            provenance
        )


def test_the_act_prints_no_surcharge_or_cess_rate(chunk_text):
    # The absence the outside_act declarations rest on, measured rather than
    # asserted: if a re-extraction ever surfaced a rate, a declaration is wrong.
    rate = re.compile(r"(?:cess|surcharge)[^.;]*?\d+(?:\.\d+)?\s*(?:%|per cent)", re.I)
    assert not any(rate.search(" ".join(text.split())) for text in chunk_text.values())


def test_every_fact_field_section_names_a_chunk(chunk_text):
    for field, spec in FIELDS.items():
        if spec.section is not None:
            assert spec.section in chunk_text, field


def test_a_deduction_not_allowed_under_202_1_sits_in_the_chapter_the_quote_excludes(rates, stored_chunks):
    # Section 202(2)(a)(xii) names a chapter, not these sections; the chunk's own
    # chapter metadata is what ties section 123 and 126 to that quote.
    _, chunks = stored_chunks
    chapter_of = {chunk.node_path: chunk.chapter_numeral for chunk in chunks}
    assert set(rates.not_allowed_under_202_1) == {"deduction_savings_insurance", "deduction_health_insurance"}
    for entry in rates.not_allowed_under_202_1.values():
        assert chapter_of[entry.section] == entry.chapter
        assert f"Chapter {entry.chapter} other than" in entry.provenance.source_text
        assert entry.section not in re.findall(r"section (\d+)", entry.provenance.source_text)
