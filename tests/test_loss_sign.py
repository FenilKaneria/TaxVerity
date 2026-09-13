"""The loss-sign fix (ADR-099): the held-out turns and the rule the fix is held
to. The measured verdict is pinned against the stored runs and skips without
them, like the Step 7.7 floors."""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import pytest

from taxverity.config import Settings
from taxverity.evals.extraction import (
    GOLD_FILENAME,
    LOSS_HOLDOUT_FILENAME,
    Counts,
    ExtractionScore,
    TurnJudgement,
    TurnSlice,
    judge_extraction,
    load_extraction_gold,
    load_run,
)
from taxverity.evals.loss_sign import judge_loss_fix, unclean
from taxverity.facts import FactField

DATASETS = Path("evals/datasets")
RUNS = Settings().data_dir / "extraction"


@pytest.fixture(scope="module")
def holdout():
    return load_extraction_gold(DATASETS, LOSS_HOLDOUT_FILENAME)


def is_loss(turn) -> bool:
    return any(isinstance(f.normalised, Decimal) and f.normalised < 0 for f in turn.facts)


def test_the_holdout_is_eight_losses_and_four_controls(holdout):
    assert sum(is_loss(t) for t in holdout) == 8
    assert len(holdout) == 12


def test_every_control_puts_loss_vocabulary_beside_a_positive_amount(holdout):
    for turn in holdout:
        if not is_loss(turn):
            assert any(word in turn.turn.lower() for word in ("loss", "lost"))


def test_the_holdout_shares_no_turn_with_the_gold_set(holdout):
    gold = {t.turn for t in load_extraction_gold(DATASETS, GOLD_FILENAME)}
    assert not gold & {t.turn for t in holdout}


def test_the_holdout_file_is_canonical(holdout):
    lines = (DATASETS / LOSS_HOLDOUT_FILENAME).read_text(encoding="utf-8").splitlines()
    for line, turn in zip(lines, holdout, strict=True):
        assert line == json.dumps(
            turn.model_dump(mode="json"), sort_keys=True, ensure_ascii=False, separators=(",", ":")
        )


# --- the rule ---------------------------------------------------------------------


def judgement(turn_id, *, wrong=False, missed=False):
    field = FactField.BUSINESS_INCOME
    return TurnJudgement(
        turn_id=turn_id,
        slice=TurnSlice.LOSS,
        counts=Counts(0 if missed else 1, 0, 1 if missed else 0),
        found=() if missed else (field,),
        missed=(field,) if missed else (),
        spurious=(),
        value_wrong=((field, "-1", "1"),) if wrong else (),
        status_wrong=(),
        strict=() if wrong or missed else (field,),
        fabricated_spans=0,
        unquoted=0,
        stated_attempts=1,
        repaired=False,
        repair_helped=False,
        tokens=0,
    )


def score(*judgements):
    return ExtractionScore(turns=judgements, counts=Counts(), per_slice={}, per_field={})


BEFORE = score(judgement("t001"), judgement("t013", wrong=True), judgement("t017", wrong=True))
AFTER = score(judgement("t001"), judgement("t013"), judgement("t017"))


def test_a_clean_holdout_that_fixes_both_and_breaks_nothing_is_adopted():
    assert judge_loss_fix(score(judgement("t001")), BEFORE, AFTER).adopted


def test_any_unclean_holdout_turn_rejects():
    verdict = judge_loss_fix(score(judgement("t001"), judgement("t002", missed=True)), BEFORE, AFTER)
    assert verdict.reasons == ("held-out t002 is not clean",)


def test_a_sign_still_wrong_rejects():
    after = score(judgement("t001"), judgement("t013", wrong=True), judgement("t017"))
    assert judge_loss_fix(score(), BEFORE, after).reasons == ("t013 still has a wrong value",)


def test_a_turn_that_was_clean_and_is_not_now_rejects():
    after = score(judgement("t001", missed=True), judgement("t013"), judgement("t017"))
    assert judge_loss_fix(score(), BEFORE, after).reasons == ("t001 was clean and is not now",)


def test_runs_over_different_turns_are_refused():
    with pytest.raises(ValueError, match="same turns"):
        judge_loss_fix(score(), BEFORE, score(judgement("t001")))


# --- measured -----------------------------------------------------------------------


def test_the_fix_still_holds_on_the_held_out_turns(holdout):
    # ADR-099's own verdict was judged on runs the tax_year rename (ADR-100) made
    # unloadable; reports/loss_sign_fix.md is its frozen record. What must still
    # hold is clause 1, re-measured through the node as it now stands. Clause 2
    # is test_extraction_eval's "no value is wrong".
    path = RUNS / "loss_holdout_run_v3.json"
    if not path.exists():
        pytest.skip("run scripts/measure_loss_sign.py --stage recheck")
    score = judge_extraction(holdout, load_run(path))
    # No sign is ever inverted on a held-out turn; that part admits no residue.
    assert [t.turn_id for t in score.turns if t.value_wrong] == []
    # t006's span was mis-copied ("1,110,000"), refused, and the loss missed.
    # A miss, not a wrong number: pinned exactly (ADR-102).
    assert [t.turn_id for t in score.turns if unclean(t)] == ["t006"]
