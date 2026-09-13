"""Step 11.8 — the follow-up cases and their run/report models.

No LLM and no retrieval here: `scripts/answer_smoke.py` is the live runner.
This file pins the cases against the real gold set and the real chunk store
(ADR-059's same discipline the gold set itself uses), and round-trips the
report models.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from taxverity.config import Settings
from taxverity.evals.followups import (
    FOLLOW_UP_CASES,
    FOLLOWUP_EVAL_VERSION,
    FOLLOWUP_RUN_FILENAME,
    FollowUpRecord,
    FollowUpRun,
    load_followup_run,
    store_followup_run,
)
from taxverity.evals.metrics import CitationIndex, UnresolvedCitationError

RECORD = FollowUpRecord(
    case_id="f001",
    prior_question="What does section 6(5) substitute for the sixty-day period?",
    follow_up="What if my income is below that threshold instead?",
    contextualized=True,
    rewritten_query="What is the residency day threshold below fifteen lakh rupees?",
    expected=("6(4)", "6(5)"),
    retrieved=("6(5)", "6(4)"),
)


def test_case_ids_are_unique():
    ids = [case.case_id for case in FOLLOW_UP_CASES]
    assert len(ids) == len(set(ids)) == 5


def test_every_case_names_a_real_gold_v2_question(gold):
    ids = {q.query_id for q in gold}
    for case in FOLLOW_UP_CASES:
        assert case.prior_query_id in ids


def test_every_expected_citation_resolves_to_a_real_chunk(chunks):
    index = CitationIndex(chunks)
    for case in FOLLOW_UP_CASES:
        for citation in case.expected:
            index.resolve(citation)  # raises UnresolvedCitationError if not


def test_a_bogus_citation_would_have_failed_that_check(chunks):
    index = CitationIndex(chunks)
    with pytest.raises(UnresolvedCitationError):
        index.resolve("999(1)")


def test_a_follow_up_shares_no_content_word_with_its_prior_question():
    """The point of each case: the raw follow-up alone gives retrieval nothing."""
    for case in FOLLOW_UP_CASES:
        follow_up_words = set(case.follow_up.casefold().split())
        assert not follow_up_words & {"section", "deduction", "cent", "percent"}


# --- report models -----------------------------------------------------------


def test_hit_is_true_when_an_expected_citation_was_retrieved():
    assert RECORD.hit is True


def test_hit_is_false_when_none_of_the_expected_citations_were_retrieved():
    missed = RECORD.model_copy(update={"retrieved": ("22(2)",)})
    assert missed.hit is False


def test_hit_credits_a_retrieved_section_root_leniently():
    """ADR-060: a section root carries a sub-section's text too (ADR-055), so
    it is imprecise, not wrong — the same credit every other retrieval eval in
    this project gives."""
    root_only = RECORD.model_copy(update={"expected": ("134(1)",), "retrieved": ("134",)})
    assert root_only.hit is True


def test_hit_never_credits_a_descendant_of_the_expected_citation():
    descendant = RECORD.model_copy(update={"expected": ("134",), "retrieved": ("134(1)",)})
    assert descendant.hit is False


def test_run_round_trips_through_disk(tmp_path: Path):
    run = FollowUpRun(eval_version=FOLLOWUP_EVAL_VERSION, records=(RECORD,))
    path = tmp_path / "followup_smoke_v1.json"
    store_followup_run(path, run)
    assert load_followup_run(path) == run


def test_load_refuses_a_run_from_another_eval_version(tmp_path: Path):
    run = FollowUpRun(eval_version=FOLLOWUP_EVAL_VERSION, records=(RECORD,))
    path = tmp_path / "followup_smoke_v1.json"
    store_followup_run(path, run)
    path.write_text(
        path.read_text(encoding="utf-8").replace(
            f'"eval_version": {FOLLOWUP_EVAL_VERSION}', '"eval_version": 999'
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="eval version"):
        load_followup_run(path)


def test_an_uncontextualized_record_reports_the_query_unchanged():
    record = RECORD.model_copy(
        update={"contextualized": False, "rewritten_query": RECORD.follow_up}
    )
    assert record.rewritten_query == record.follow_up


# --- the stored run (Step 7.7 pattern: skip without it, no live LLM in CI) --

STORED = Settings().data_dir / "answers" / FOLLOWUP_RUN_FILENAME


@pytest.fixture(scope="module")
def stored_followup_run():
    if not STORED.exists():
        pytest.skip("run scripts/answer_smoke.py to store a follow-up run")
    return load_followup_run(STORED)


def test_the_stored_run_covers_every_case(stored_followup_run):
    assert tuple(r.case_id for r in stored_followup_run.records) == tuple(
        c.case_id for c in FOLLOW_UP_CASES
    )


def test_every_stored_follow_up_was_rewritten(stored_followup_run):
    """Every case carries a marker and a prior turn by construction, so the
    deterministic skip should never fire here."""
    assert all(r.contextualized for r in stored_followup_run.records)


def test_the_stored_run_hit_at_least_the_unambiguous_case(stored_followup_run):
    by_id = {r.case_id: r for r in stored_followup_run.records}
    assert by_id["f005"].hit is True
