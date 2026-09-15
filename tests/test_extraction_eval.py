"""Step 7.7 — the extraction gold set and the judge that scores a run against it.

No network and no model: the judge takes a stored run, so every test here is
pure. The floor tests read the measured run under `data/` and skip when it is
absent, the same way the dense and rerank floors do.
"""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import pytest
from pydantic import ValidationError

from taxverity.config import Settings
from taxverity.corpus.loader import normalise
from taxverity.evals.extraction import (
    EXTRACTION_EVAL_VERSION,
    GOLD_FILENAME,
    Counts,
    LabelledFact,
    LabelledTurn,
    TurnSlice,
    distinct_models,
    judge_extraction,
    judge_turn,
    load_extraction_gold,
    load_run,
    store_run,
)
from taxverity.facts import (
    Fact,
    FactField,
    FactIssue,
    FactStatus,
    Rejection,
    UnmappedFact,
    UserFacts,
)
from taxverity.llm.client import GROQ, Completion, Usage
from taxverity.llm.extract import ExtractionResult
from taxverity.observability import redact

DATASETS = Path("evals/datasets")
# The node as it stands: Step 7.7 plus the loss-sign fix (ADR-099).
STORED_RUN = Settings().data_dir / "extraction" / "extraction_run_v4.json"


@pytest.fixture(scope="session")
def gold() -> tuple[LabelledTurn, ...]:
    return load_extraction_gold(DATASETS)


def fact(
    field: FactField,
    value,
    status: FactStatus = FactStatus.STATED,
    raw: str = "x",
    span: str = "x",
) -> Fact:
    return Fact(
        field=field, status=status, raw_value=raw, value=value, source_span=span
    )


def completion(text: str = "{}", tokens: int = 100) -> Completion:
    return Completion(
        text=text,
        provider="groq",
        model="openai/gpt-oss-120b",
        finish_reason="stop",
        usage=Usage(prompt_tokens=tokens, completion_tokens=0),
        degraded=False,
    )


def result(
    *facts: Fact,
    rejections: tuple[Rejection, ...] = (),
    repairable: tuple[Rejection, ...] = (),
    repaired: bool = False,
    unmapped: tuple[UnmappedFact, ...] = (),
) -> ExtractionResult:
    return ExtractionResult(
        facts=UserFacts(facts=facts, unmapped=unmapped),
        rejections=rejections,
        repairable=repairable,
        repaired=repaired,
        completions=(completion(),),
    )


def labelled(
    *facts: LabelledFact, slice_: TurnSlice = TurnSlice.SIMPLE
) -> LabelledTurn:
    return LabelledTurn(
        turn_id="t001",
        slice=slice_,
        turn="My salary is 1400000.",
        facts=facts,
        notes="a fixture",
    )


SALARY_LABEL = LabelledFact(
    name=FactField.SALARY_INCOME, status=FactStatus.STATED, value="1400000"
)


# --- the dataset ------------------------------------------------------------


def test_the_gold_set_is_at_least_forty_turns(gold):
    assert len(gold) >= 40


def test_every_field_is_labelled_at_least_once(gold):
    labelled_fields = {label.name for turn in gold for label in turn.facts}
    assert labelled_fields == set(FactField)


def test_every_slice_is_represented(gold):
    assert {turn.slice for turn in gold} == set(TurnSlice)


def test_every_label_normalises(gold):
    for turn in gold:
        for label in turn.facts:
            assert label.normalised is not None


def test_a_stated_label_is_a_value_the_turn_can_support(gold):
    # A label the turn cannot possibly justify would make recall unreachable
    # rather than measuring the node. Digits are checked loosely, since the turn
    # may write them grouped, as a multiplier word, or as a loss.
    for turn in gold:
        for label in turn.facts:
            if label.status is FactStatus.STATED and label.name is FactField.AGE:
                assert str(label.normalised) in turn.turn


def test_turns_with_no_labels_exist(gold):
    # A turn carrying no fact is the only way to measure invention.
    assert any(not turn.facts for turn in gold)


def test_the_out_of_vocabulary_turns_label_nothing(gold):
    for turn in gold:
        if turn.slice is TurnSlice.OUT_OF_VOCABULARY:
            assert turn.facts == ()


def test_the_file_on_disk_is_canonical(gold):
    lines = (DATASETS / GOLD_FILENAME).read_text(encoding="utf-8").splitlines()
    for line, turn in zip(lines, gold, strict=True):
        rewritten = json.dumps(
            turn.model_dump(mode="json"),
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        assert line == rewritten


def test_a_label_may_not_claim_a_status_the_model_cannot_assert():
    with pytest.raises(ValidationError):
        LabelledFact(name=FactField.AGE, status=FactStatus.PROFILE_DEFAULT, value="40")


def test_a_label_that_does_not_normalise_is_refused():
    with pytest.raises(ValidationError):
        LabelledFact(
            name=FactField.SALARY_INCOME, status=FactStatus.STATED, value="lots"
        )


def test_a_turn_may_not_label_a_field_twice():
    with pytest.raises(ValidationError):
        labelled(SALARY_LABEL, SALARY_LABEL)


def test_turn_ids_must_be_contiguous(tmp_path):
    row = {
        "turn_id": "t002",
        "slice": "simple",
        "turn": "hello",
        "facts": [],
        "notes": "n",
    }
    (tmp_path / GOLD_FILENAME).write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="contiguous"):
        load_extraction_gold(tmp_path)


# --- the judge --------------------------------------------------------------


def test_a_perfect_turn_scores_one():
    judgement = judge_turn(
        labelled(SALARY_LABEL),
        result(fact(FactField.SALARY_INCOME, Decimal("1400000"))),
    )
    assert judgement.counts == Counts(1, 0, 0)
    assert judgement.strict == (FactField.SALARY_INCOME,)


def test_a_missed_field_is_a_false_negative():
    judgement = judge_turn(labelled(SALARY_LABEL), result())
    assert judgement.counts == Counts(0, 0, 1)
    assert judgement.missed == (FactField.SALARY_INCOME,)


def test_an_invented_field_is_a_false_positive():
    judgement = judge_turn(labelled(), result(fact(FactField.AGE, 40)))
    assert judgement.counts == Counts(0, 1, 0)
    assert judgement.spurious == (FactField.AGE,)


def test_a_missing_fact_is_not_a_prediction():
    # The node completes every unreported field as MISSING; counting those would
    # make precision a statement about how many fields the vocabulary has.
    missing = fact(FactField.AGE, None, FactStatus.MISSING, raw="", span="")
    judgement = judge_turn(labelled(), result(missing))
    assert judgement.counts == Counts(0, 0, 0)


def test_a_wrong_value_is_found_but_not_strict():
    judgement = judge_turn(
        labelled(SALARY_LABEL), result(fact(FactField.SALARY_INCOME, Decimal("9")))
    )
    assert judgement.counts == Counts(1, 0, 0)
    assert judgement.strict == ()
    assert judgement.value_wrong == ((FactField.SALARY_INCOME, "1400000", "9"),)


def test_a_wrong_status_is_found_but_not_strict():
    judgement = judge_turn(
        labelled(SALARY_LABEL),
        result(
            fact(
                FactField.SALARY_INCOME,
                Decimal("1400000"),
                FactStatus.INFERRED,
                span="",
            )
        ),
    )
    assert judgement.status_wrong == ((FactField.SALARY_INCOME, "stated", "inferred"),)
    assert judgement.strict == ()


def test_formatting_is_not_measured():
    # "14 lakh" and "1400000" are the same correct answer; ADR-096 put the
    # normalisation in code precisely so the eval measures extraction.
    label = LabelledFact(
        name=FactField.SALARY_INCOME, status=FactStatus.STATED, value="14 lakh"
    )
    judgement = judge_turn(
        labelled(label), result(fact(FactField.SALARY_INCOME, Decimal("1400000")))
    )
    assert judgement.value_wrong == ()


def test_a_fabricated_span_is_counted_as_an_attempt():
    rejection = Rejection(FactIssue.SPAN_NOT_IN_TURN, "made up", {"name": "age"})
    judgement = judge_turn(labelled(), result(rejections=(rejection,)))
    assert judgement.fabricated_spans == 1
    assert judgement.stated_attempts == 1


def test_a_span_repaired_before_the_end_still_counts_as_an_attempt():
    # The rejection is gone from the final result, so only `repairable` records
    # that the model fabricated one at all.
    rejection = Rejection(FactIssue.SPAN_NOT_IN_TURN, "made up", {"name": "age"})
    judgement = judge_turn(
        labelled(),
        result(repairable=(rejection,), repaired=True),
    )
    assert judgement.fabricated_spans == 1
    assert judgement.repair_helped is True


def test_a_repair_that_changed_nothing_is_reported_as_such():
    rejection = Rejection(FactIssue.MISSING_SPAN, "no span", {"name": "age"})
    judgement = judge_turn(
        labelled(),
        result(rejections=(rejection,), repairable=(rejection,), repaired=True),
    )
    assert judgement.repaired is True
    assert judgement.repair_helped is False


def test_scores_aggregate_across_turns():
    turns = [
        LabelledTurn(
            turn_id="t001",
            slice=TurnSlice.SIMPLE,
            turn="a",
            facts=(SALARY_LABEL,),
            notes="n",
        ),
        LabelledTurn(
            turn_id="t002", slice=TurnSlice.NONE, turn="b", facts=(), notes="n"
        ),
    ]
    run = {
        "t001": result(fact(FactField.SALARY_INCOME, Decimal("1400000"))),
        "t002": result(fact(FactField.AGE, 40)),
    }
    score = judge_extraction(turns, run)
    assert score.counts == Counts(1, 1, 0)
    assert score.per_slice[TurnSlice.NONE] == Counts(0, 1, 0)
    assert score.per_field[FactField.SALARY_INCOME] == Counts(1, 0, 0)
    assert score.strict_rate == 1.0
    assert score.clean_turns == 1
    assert score.tokens == 200


def test_a_turn_absent_from_the_run_is_refused():
    with pytest.raises(KeyError):
        judge_extraction([labelled(SALARY_LABEL)], {})


def test_counts_with_nothing_labelled_do_not_divide_by_zero():
    assert Counts().precision == 0.0
    assert Counts().recall == 0.0
    assert Counts().f1 == 0.0


# --- the stored run ---------------------------------------------------------


def test_a_run_round_trips_through_disk(tmp_path):
    run = {
        "t001": result(
            fact(FactField.SALARY_INCOME, Decimal("1400000"), raw="14,00,000"),
            fact(FactField.AGE, 40, raw="40"),
            rejections=(Rejection(FactIssue.MISSING_SPAN, "no span", {"a": 1}),),
            repairable=(Rejection(FactIssue.MISSING_SPAN, "no span", {"a": 1}),),
            repaired=True,
        )
    }
    path = tmp_path / "run.json"
    store_run(path, run)
    restored = load_run(path)["t001"]
    salary = restored.facts.get(FactField.SALARY_INCOME)
    assert salary is not None
    # A Decimal that came back as a string would compare equal to nothing and
    # silently tank value accuracy, so the value is re-derived, never stored.
    assert salary.value == Decimal("1400000")
    assert isinstance(salary.value, Decimal)
    assert restored.rejections[0].issue is FactIssue.MISSING_SPAN
    assert restored.repaired is True


def test_a_run_from_another_eval_version_is_refused(tmp_path):
    path = tmp_path / "run.json"
    path.write_text(json.dumps({"eval_version": 99, "turns": {}}), encoding="utf-8")
    with pytest.raises(ValueError, match="eval version"):
        load_run(path)


def test_the_eval_version_is_declared():
    assert EXTRACTION_EVAL_VERSION == 2


# --- floors, measured. They skip without the stored run ---------------------


@pytest.fixture(scope="session")
def measured(gold):
    if not STORED_RUN.exists():
        pytest.skip(f"no measured extraction run at {STORED_RUN}")
    run = load_run(STORED_RUN)
    # R18.3: the stored completions already carry provider/model; nothing
    # asserted them before. FactExtractor.from_settings() calls GROQ with no
    # override today, so a mismatch here means the floors below are about to
    # be checked against a different model's numbers.
    models = distinct_models(run)
    assert models == {(GROQ.name, GROQ.model)}, (
        f"{STORED_RUN.name} was measured against {sorted(models)}, not the "
        f"currently configured ({GROQ.name!r}, {GROQ.model!r}) — re-measure "
        "before trusting these floors"
    )
    return judge_extraction(gold, run)


# Floors sit below the measured run (precision 1.000, recall 1.000, value
# 0.959, strict 0.939, 0 fabricated spans over 46 stated attempts), the same
# way the retrieval floors do. They are a regression guard, not a target.
FIELD_PRECISION_FLOOR = 0.95
FIELD_RECALL_FLOOR = 0.95
VALUE_ACCURACY_FLOOR = 0.90
STRICT_FLOOR = 0.88
FABRICATED_SPAN_CEILING = 0.05

def test_field_detection_holds_its_floor(measured):
    assert measured.counts.precision >= FIELD_PRECISION_FLOOR
    assert measured.counts.recall >= FIELD_RECALL_FLOOR


def test_value_accuracy_holds_its_floor(measured):
    assert measured.value_accuracy >= VALUE_ACCURACY_FLOOR


def test_the_strict_rate_holds_its_floor(measured):
    assert measured.strict_rate >= STRICT_FLOOR


def test_the_node_does_not_invent_facts_from_a_turn_that_states_none(measured):
    for judgement in measured.turns:
        if judgement.slice in (TurnSlice.NONE, TurnSlice.OUT_OF_VOCABULARY):
            assert judgement.spurious == ()


def test_fabricated_spans_stay_rare(measured):
    assert measured.fabricated_span_rate <= FABRICATED_SPAN_CEILING


def test_the_only_wrong_value_is_the_pinned_residue(measured):
    # ADR-102 pinned t041 here. Step 7.8's fresh run (ADR-109) wrote its sign
    # right on the first pass, and the clause guard now refuses the positive it
    # once returned. A wrong value admits no residue.
    assert [j.turn_id for j in measured.turns if j.value_wrong] == []


def test_every_surviving_span_is_in_the_turn_the_model_saw(gold):
    # Structural, not statistical: parse_facts refuses a span the turn does not
    # contain, so this must hold for every fact in the measured run. It guards
    # the parser rather than the model.
    if not STORED_RUN.exists():
        pytest.skip(f"no measured extraction run at {STORED_RUN}")
    run = load_run(STORED_RUN)
    for turn in gold:
        seen = normalise(redact(turn.turn))
        for fact_ in run[turn.turn_id].facts.known():
            if fact_.source_span:
                assert normalise(fact_.source_span) in seen


def test_no_stored_span_carries_a_pan():
    if not STORED_RUN.exists():
        pytest.skip(f"no measured extraction run at {STORED_RUN}")
    stored = STORED_RUN.read_text(encoding="utf-8")
    assert "ABCDE1234F" not in stored
