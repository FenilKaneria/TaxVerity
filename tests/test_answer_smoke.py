"""Step 10.7 — the answer smoke set.

Pure tests pin the summary and the stored-run format. The stored-run tests read
`data/answers/` and skip without it (the Step 7.7 pattern), so no live LLM runs
in CI. Beyond the gate, every served claim in the stored run is re-verified
against its own evidence by a fresh `Verifier`, so a stored answer that slipped
past the gate would fail here too.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from taxverity.config import Settings
from taxverity.evals.answers import (
    ANSWER_EVAL_VERSION,
    ANSWER_RUN_FILENAME,
    SMOKE_QUERY_IDS,
    AnswerRecord,
    AnswerRun,
    load_answer_run,
    store_answer_run,
    summarise,
)
from taxverity.evals.gold import QuerySlice
from taxverity.generation.claims import (
    Citation,
    Claim,
    ClaimEvent,
    ClaimType,
    WithheldEvent,
)
from taxverity.generation.verifier import Verifier
from taxverity.retrieval.base import ScoredChunk
from taxverity.retrieval.evidence import EvidencePacker

SRC = Path(__file__).resolve().parents[1] / "src"
STORED = Settings().data_dir / "answers" / ANSWER_RUN_FILENAME


def statute(claim_id: int = 1) -> ClaimEvent:
    return ClaimEvent(
        id=claim_id,
        type=ClaimType.STATUTE,
        text="Thirty per cent of the annual value is deducted.",
        citations=(
            Citation(path="22(1)(a)", quote="thirty per cent of the annual value"),
        ),
    )


def computation(claim_id: int = 1) -> ClaimEvent:
    return ClaimEvent(
        id=claim_id, type=ClaimType.COMPUTATION, text="Tax is 0.", citations=()
    )


def record(query_id, slice_, claims=(), withheld=(), error=None) -> AnswerRecord:
    return AnswerRecord(
        query_id=query_id,
        slice=slice_,
        question="q?",
        evidence=(),
        claims=tuple(claims),
        withheld=tuple(withheld),
        error=error,
        seconds=1.0,
    )


def test_the_smoke_set_is_ten_answerable_and_five_negative(gold):
    by_id = {q.query_id: q for q in gold}
    slices = [by_id[query_id].slice for query_id in SMOKE_QUERY_IDS]
    assert len(set(SMOKE_QUERY_IDS)) == 15
    assert slices.count(QuerySlice.NEGATIVE) == 5
    assert {s for s in slices if s is not QuerySlice.NEGATIVE} == {
        QuerySlice.CITATION,
        QuerySlice.PARAPHRASE,
        QuerySlice.CROSSREF,
    }


def test_a_served_statute_claim_on_a_negative_fails_the_gate():
    summary = summarise(
        [
            record("q025", QuerySlice.NEGATIVE, claims=[statute()]),
            record("q026", QuerySlice.NEGATIVE),
        ]
    )
    assert summary.served_on_negative == ("q025",)


def test_a_withheld_claim_or_computation_on_a_negative_does_not_fail_the_gate():
    summary = summarise(
        [
            record(
                "q025", QuerySlice.NEGATIVE, withheld=[WithheldEvent(id=1, reason="x")]
            ),
            record("q026", QuerySlice.NEGATIVE, claims=[computation()]),
        ]
    )
    assert summary.served_on_negative == ()


def test_the_summary_counts_and_reports_silent_answerable_questions():
    summary = summarise(
        [
            record("q001", QuerySlice.CITATION, claims=[statute(1), statute(2)]),
            record(
                "q009",
                QuerySlice.PARAPHRASE,
                withheld=[WithheldEvent(id=1, reason="x")],
                error="boom",
            ),
            record("q025", QuerySlice.NEGATIVE),
        ]
    )
    assert summary.questions == 3
    assert summary.answered == 1
    assert (summary.claims_served, summary.claims_withheld) == (2, 1)
    assert summary.errors == 1
    assert summary.silent_answerable == ("q009",)


def test_a_run_round_trips_and_refuses_another_eval_version(tmp_path):
    run = AnswerRun(
        eval_version=ANSWER_EVAL_VERSION,
        generation_stage_version=1,
        prompt_version=1,
        model="m",
        tokens=10,
        records=(record("q001", QuerySlice.CITATION, claims=[statute()]),),
    )
    path = tmp_path / "run.json"
    store_answer_run(path, run)
    assert load_answer_run(path) == run

    path.write_text(
        path.read_text().replace(
            f'"eval_version": {ANSWER_EVAL_VERSION}', '"eval_version": 99'
        )
    )
    with pytest.raises(ValueError, match="eval version"):
        load_answer_run(path)


def test_deepeval_is_never_imported_from_src():
    offenders = [
        path
        for path in SRC.rglob("*.py")
        if "deepeval" in path.read_text(encoding="utf-8")
    ]
    assert offenders == []


@pytest.fixture(scope="module")
def stored_run():
    if not STORED.exists():
        pytest.skip("run scripts/answer_smoke.py to store a smoke-set run")
    return load_answer_run(STORED)


def test_the_stored_run_covers_the_smoke_set(stored_run):
    assert tuple(r.query_id for r in stored_run.records) == SMOKE_QUERY_IDS


def test_no_negative_question_gets_a_served_statute_claim(stored_run):
    assert summarise(stored_run.records).served_on_negative == ()


def test_every_stored_claim_re_verifies_against_its_own_evidence(
    stored_run, stored_chunks
):
    _, chunks = stored_chunks
    by_path = {chunk.node_path: chunk for chunk in chunks}
    packer = EvidencePacker(chunks)
    for answer in stored_run.records:
        cited = [by_path[unit.citation] for unit in answer.evidence]
        pack = packer.pack(
            [
                ScoredChunk(chunk=chunk, score=float(len(cited) - i))
                for i, chunk in enumerate(cited)
            ]
        )
        assert [u.citation for u in pack.units] == [u.citation for u in answer.evidence]
        verifier = Verifier(pack, by_path, question=answer.question)
        for claim in answer.claims:
            verdict = verifier.verify(
                Claim(type=claim.type, text=claim.text, citations=claim.citations)
            )
            assert verdict.passed, (answer.query_id, claim.id, verdict.findings)
