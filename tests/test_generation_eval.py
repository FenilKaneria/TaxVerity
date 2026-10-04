"""The generation eval's scoring, offline: dataset shape, judge parsing,
citation credit, kappa, and the summary over hand-built records."""

from __future__ import annotations

import pytest

from taxverity.evals.generation import (
    AnswerGoldItem,
    ClaimLabel,
    ItemKind,
    JudgeParseError,
    cites_gold,
    cohen_kappa,
    gold_citation_coverage,
    headline_tax,
    load_answer_gold,
    parse_claim_label,
    parse_coverage,
    passages_for,
    percentile,
    summarise,
)
from taxverity.generation.generate import AnswerGenerator, DraftOutcome


def test_the_answer_gold_set_loads_and_is_what_the_report_says():
    items = load_answer_gold()
    counts = {kind: sum(i.kind is kind for i in items) for kind in ItemKind}
    assert counts == {
        ItemKind.ANSWERABLE: 49,
        ItemKind.NEGATIVE: 16,
        ItemKind.CALCULATION: 8,
        ItemKind.SAFETY: 8,
    }


def test_no_flagged_gold_label_is_in_the_answer_set():
    # q001, q052, q062, q064 and q067 have labels awaiting a re-check.
    sources = {item.source for item in load_answer_gold()}
    for query_id in ("q001", "q052", "q062", "q064", "q067"):
        assert f"retrieval_gold_v2:{query_id}" not in sources


@pytest.mark.parametrize(
    ("cited", "gold", "expected"),
    [
        ("22(2)", "22(2)", True),
        ("22", "22(2)", True),  # an ancestor holds the answer
        ("22(2)(a)", "22(2)", True),  # a part of it is precise, not wrong
        ("22(1)", "22(2)", False),
        ("21(6)", "22(2)", False),
    ],
)
def test_an_answer_citation_is_credited_up_and_down_its_path(cited, gold, expected):
    assert cites_gold(cited, gold) is expected


def test_gold_coverage_is_the_share_of_gold_provisions_cited():
    assert gold_citation_coverage(["22(2)(a)"], ["22(2)", "21(6)"]) == 0.5
    assert gold_citation_coverage([], ["22(2)"]) == 0.0


def test_the_headline_tax_comes_from_the_first_computation_line():
    served = [
        {"type": "heading", "text": "### Your tax"},
        {"type": "computation", "text": "Your tax payable is ₹1,55,000 [calc]"},
        {"type": "computation", "text": "Tax before rebate is ₹60,000 [calc]"},
    ]
    assert {int(n) for n in headline_tax(served)} == {155000}


@pytest.mark.parametrize(
    ("reply", "label"),
    [
        ('{"label": "supported", "reason": "x"}', ClaimLabel.SUPPORTED),
        ('```json\n{"label": "contradicted"}\n```', ClaimLabel.CONTRADICTED),
    ],
)
def test_a_claim_label_parses(reply, label):
    assert parse_claim_label(reply) is label


@pytest.mark.parametrize(
    "reply", ["not json", '{"label": "probably"}', '{"label": "uncited"}', "[]"]
)
def test_an_unusable_claim_label_is_refused_not_guessed(reply):
    with pytest.raises(JudgeParseError):
        parse_claim_label(reply)


def test_coverage_must_answer_every_point_with_a_boolean():
    assert parse_coverage('{"covered": [true, false]}', 2) == (True, False)
    for reply in ('{"covered": [true]}', '{"covered": [true, "yes"]}', "{}"):
        with pytest.raises(JudgeParseError):
            parse_coverage(reply, 2)


def test_passages_follow_the_markers_a_line_cites():
    evidence = {1: ("22(1)", "a"), 2: ("24", "b")}
    assert passages_for("x [2][1][2] [9].", evidence) == [("24", "b"), ("22(1)", "a")]
    assert passages_for("no markers", evidence) == []


def test_kappa():
    assert cohen_kappa([True, False, True, False], [True, False, True, False]) == 1.0
    assert cohen_kappa([True, True, False, False], [True, False, True, False]) == 0.0
    with pytest.raises(ValueError):
        cohen_kappa([True], [True, False])


def test_percentile_is_nearest_rank():
    assert percentile([1.0, 2.0, 3.0, 4.0, 5.0], 0.5) == 3.0
    assert percentile([], 0.95) == 0.0


def _item(item_id, kind, **kwargs):
    return AnswerGoldItem(
        item_id=item_id,
        kind=kind,
        source="test",
        question="q",
        slice=kind.value,
        **kwargs,
    )


def _claim(claim_id, type_, text, *paths):
    return {
        "id": claim_id,
        "type": type_,
        "text": text,
        "citations": [{"marker": 1, "path": p, "quote": "q"} for p in paths],
    }


def _line(claim_id, passed, type_="content"):
    return {"claim_id": claim_id, "line": "l", "type": type_, "passed": passed}


def test_the_summary_over_hand_built_records():
    items = [
        _item(
            "g001",
            ItemKind.ANSWERABLE,
            gold_citations=("22(2)",),
            key_points=("a", "b"),
        ),
        _item("g002", ItemKind.NEGATIVE),
        _item("g003", ItemKind.CALCULATION, expected_tax="155000"),
        _item("g004", ItemKind.SAFETY, expected_category="prohibited"),
    ]
    records = {
        "g001": {
            "seconds": 4.0,
            "tokens": 1000,
            "category": "in_scope",
            "served": [_claim(1, "content", "x [1]", "22(2)(a)")],
            # Line 2 failed, was repaired; line 3 failed and stayed withheld.
            "first_pass": [_line(1, True), _line(2, False), _line(3, False)],
            "final": [_line(1, True), _line(2, True), _line(3, False)],
            "judge": {
                "served": {"1": "supported", "2": "partially_supported"},
                "first_pass": {"1": "supported", "2": "unsupported", "3": "uncited"},
                "key_points": [True, False],
            },
        },
        "g002": {
            "seconds": 1.0,
            "tokens": 0,
            "category": "adjacent",
            "served": [],
            "first_pass": [],
            "final": [],
        },
        "g003": {
            "seconds": 6.0,
            "tokens": 2000,
            "category": "in_scope",
            "served": [
                _claim(1, "computation", "Your tax payable is ₹1,55,000 [calc]")
            ],
            "first_pass": [_line(1, True, "computation")],
            "final": [_line(1, True, "computation")],
        },
        "g004": {
            "seconds": 1.0,
            "tokens": 300,
            "category": "prohibited",
            "served": [],
            "first_pass": [],
            "final": [],
        },
    }
    summary = summarise(items, records)
    assert summary["items_run"] == 4
    assert summary["first_pass_pass_rate"] == 2 / 4
    assert summary["repair_success_rate"] == 1 / 2
    assert summary["withheld_line_rate"] == 1 / 4
    assert summary["answer_rate"] == 1.0
    assert summary["citation_hit_rate"] == 1.0
    assert summary["abstention_recall"] == 1.0
    assert summary["calculator_exact"] == 1.0
    assert summary["safety_correct"] == 1.0
    assert summary["supported_rate"] == 0.5
    assert summary["unsupported_rate_served"] == 0.0
    assert summary["unsupported_rate_ungated"] == 2 / 3
    assert summary["key_point_recall"] == 0.5


def test_a_negative_with_a_served_statute_claim_is_not_an_abstention():
    items = [_item("g001", ItemKind.NEGATIVE)]
    records = {
        "g001": {
            "seconds": 1.0,
            "served": [_claim(1, "content", "x [1]", "22")],
            "first_pass": [],
            "final": [],
        }
    }
    assert summarise(items, records)["abstention_recall"] == 0.0


def test_judged_rates_are_absent_until_the_judge_runs():
    items = [_item("g001", ItemKind.NEGATIVE)]
    records = {"g001": {"seconds": 1.0, "served": [], "first_pass": [], "final": []}}
    summary = summarise(items, records)
    assert summary["unsupported_rate_served"] is None
    assert summary["key_point_recall"] is None


def test_the_generator_observer_sees_both_passes_and_changes_nothing():
    from test_generation import PACK, FakeLLM, answer

    seen: list[tuple[str, list[DraftOutcome]]] = []
    llm = FakeLLM(
        answer("- Thirty per cent is deducted [1].", "- The cap is 9 lakh [2]."),
        answer("- The cap is 2 lakh [2]."),
    )
    generator = AnswerGenerator(llm, {})
    generator.observer = lambda stage, outcomes: seen.append((stage, outcomes))
    events = generator.generate("q", PACK)
    assert [stage for stage, _ in seen] == ["first_pass", "final"]
    assert [o.passed for o in seen[0][1]] == [True, False]
    assert seen[0][1][1].violations == ("unsupported_number",)
    assert [o.passed for o in seen[1][1]] == [True, True]
    assert len(events) == 2
