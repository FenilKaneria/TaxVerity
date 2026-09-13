"""Step 9.5 — golden cases: full traces worked by hand from the Act, frozen in
`evals/datasets/calculator_golden_v1.jsonl` (ADR-105).

Each case pins every line of both routes — citation, amount, and a rate line's
basis and rate — not only the final figure, so a line that moves to the wrong
provision or a band that splits differently fails even when the total survives.
"""

from __future__ import annotations

import json
from decimal import Decimal

import pytest

from taxverity.calculator.comparison import compare_regimes
from taxverity.calculator.rates import load_rates
from taxverity.config import Settings

D = Decimal
GOLDEN_FILENAME = "calculator_golden_v1.jsonl"


def golden_path():
    return Settings().evals_dir / "datasets" / GOLDEN_FILENAME


def load_golden():
    with golden_path().open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


CASES = load_golden()


def expected_line(entry):
    return (
        entry["citation"],
        D(entry["amount"]),
        D(entry["basis"]) if "basis" in entry else None,
        D(entry["rate_percent"]) if "rate_percent" in entry else None,
    )


def actual_line(line):
    return (line.provenance.citation, line.amount, line.basis, line.rate_percent)


def run(case):
    inputs = case["inputs"]
    return compare_regimes(
        load_rates(case["tax_year"]),
        salary=D(inputs["salary"]),
        other_income=D(inputs["other_income"]),
        resident_individual=inputs["resident_individual"],
        claimed={name: D(amount) for name, amount in inputs["claimed"].items()},
    )


def test_the_file_is_canonical_jsonl():
    # Same discipline as the retrieval gold sets: the bytes on disk are exactly
    # what re-serialising them produces, so a hand edit cannot hide in formatting.
    raw = golden_path().read_bytes()
    assert b"\r" not in raw
    canonical = "".join(json.dumps(case, sort_keys=True, ensure_ascii=False) + "\n" for case in CASES)
    assert raw.decode("utf-8") == canonical


def test_case_ids_are_contiguous_and_every_case_explains_its_working():
    assert [case["case_id"] for case in CASES] == [f"c{n:03d}" for n in range(1, len(CASES) + 1)]
    assert all(case["working"].strip() for case in CASES)


def test_amounts_are_strings_never_json_numbers():
    # A JSON number is parsed as a float before the test sees it (ADR-017).
    def walk(node):
        if isinstance(node, dict):
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)
        else:
            assert not isinstance(node, (int, float)) or isinstance(node, bool)

    walk(CASES)


def test_the_cases_reach_every_branch_they_exist_for():
    cited = {entry["citation"] for case in CASES for entry in case["under_202_1"]["lines"]}
    assert {"19(1)", "516", "202(1)", "156(2)(a)", "156(2)(b)"} <= cited
    rates = {entry["rate_percent"] for case in CASES for entry in case["under_202_1"]["lines"] if "rate_percent" in entry}
    assert rates == {"0", "5", "10", "15", "20", "25", "30"}
    deductions = {entry["citation"] for case in CASES for entry in case["opted_out"]["deductions"]}
    assert deductions == {"123", "122(2)"}
    assert any(case["opted_out"]["total_income_is_upper_bound"] for case in CASES)
    assert {case["inputs"]["resident_individual"] for case in CASES} == {True, False}


@pytest.mark.parametrize("case", CASES, ids=[case["case_id"] for case in CASES])
def test_the_202_1_trace_matches_the_hand_worked_case(case):
    result = run(case).under_202_1
    expected = case["under_202_1"]
    assert [actual_line(line) for line in result.lines()] == [expected_line(entry) for entry in expected["lines"]]
    assert [(line.provenance.citation, line.amount) for line in result.not_allowed] == [
        (entry["citation"], D(entry["amount"])) for entry in expected["not_allowed"]
    ]


@pytest.mark.parametrize("case", CASES, ids=[case["case_id"] for case in CASES])
def test_the_opted_out_route_matches_the_hand_worked_case(case):
    side = run(case).opted_out
    expected = case["opted_out"]
    standard = side.standard_deduction.amount if side.standard_deduction else None
    assert standard == (D(expected["standard_deduction"]) if expected["standard_deduction"] else None)
    assert side.gross_total_income == D(expected["gross_total_income"])
    assert [(line.provenance.citation, line.amount) for line in side.deductions] == [
        (entry["citation"], D(entry["amount"])) for entry in expected["deductions"]
    ]
    assert side.total_income == D(expected["total_income"])
    assert (side.rounded_total_income.amount, side.rounded_total_income.provenance.citation) == (
        D(expected["rounded_total_income"]),
        "516",
    )
    assert side.total_income_is_upper_bound is expected["total_income_is_upper_bound"]
    assert side.tax.provenance.citation == "4(1)"
