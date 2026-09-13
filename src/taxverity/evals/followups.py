"""Step 11.8 — multi-turn retrieval, simplified into ~5 follow-up cases run
through the 10.7 smoke script (ADR-110): no recall study, no gold-set slice,
no threshold. Each case pairs one already-labelled gold-v2 question with a
follow-up phrasing that only makes sense after it — the same shape rule 04's
recent-turns window exists for. Step 11.7's `QueryContextualizer` rewrites the
follow-up before it reaches retrieval, exactly as Phase 13's graph will do per
turn.

`expected` is recorded so a human reading the rendered report can see whether
the rewrite kept retrieval on the right provision. It is not scored — the full
rescue-rate study stays deferred with Phase 8's corrective loop.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, TypeAdapter

from taxverity.evals.metrics import CreditMode, credits

FOLLOWUP_EVAL_VERSION = 1
FOLLOWUP_RUN_FILENAME = "followup_smoke_v1.json"


class FollowUpCase(BaseModel):
    model_config = ConfigDict(frozen=True)

    case_id: str
    # A gold-v2 query id: its question text seeds the recent-turns window.
    prior_query_id: str
    follow_up: str
    # Citations the rewrite should still reach. Reported, never gated.
    expected: tuple[str, ...]


# Each follow-up shares no content word with the provision that answers it —
# the point of the case. f002 is rule 03's own canonical lawful-planning
# example ("can I pay rent to my mother"), asked as a follow-up on purpose.
FOLLOW_UP_CASES: tuple[FollowUpCase, ...] = (
    FollowUpCase(
        case_id="f001",
        prior_query_id="q003",
        follow_up="What if my income is below that threshold instead?",
        expected=("6(4)", "6(5)"),
    ),
    FollowUpCase(
        case_id="f002",
        prior_query_id="q010",
        follow_up="Can I pay it to my mother instead?",
        expected=("134(1)", "134(2)"),
    ),
    FollowUpCase(
        case_id="f003",
        prior_query_id="q014",
        follow_up="What if I buy it two years later instead?",
        expected=("82(1)",),
    ),
    FollowUpCase(
        case_id="f004",
        prior_query_id="q013",
        follow_up="What about after that period ends?",
        expected=("129(2)",),
    ),
    FollowUpCase(
        case_id="f005",
        prior_query_id="q006",
        follow_up="Does the same limit apply to a plug-in hybrid instead?",
        expected=("132",),
    ),
)


class FollowUpRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    case_id: str
    prior_question: str
    follow_up: str
    contextualized: bool
    rewritten_query: str
    expected: tuple[str, ...]
    retrieved: tuple[str, ...]
    # A provider or retrieval failure ends the case; reported, not raised.
    error: str | None = None

    @property
    def hit(self) -> bool:
        """Lenient (ADR-060): a retrieved section root covers a required
        sub-section's text too (ADR-055), so it counts, same as everywhere
        else in this project a citation is scored."""
        return any(
            credits(retrieved, expected, CreditMode.LENIENT)
            for retrieved in self.retrieved
            for expected in self.expected
        )


class FollowUpRun(BaseModel):
    model_config = ConfigDict(frozen=True)

    eval_version: int
    records: tuple[FollowUpRecord, ...]


def store_followup_run(path: Path, run: FollowUpRun) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = run.model_dump(mode="json")
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(
            json.dumps(payload, sort_keys=True, ensure_ascii=False, indent=1) + "\n"
        )


def load_followup_run(path: Path) -> FollowUpRun:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("eval_version") != FOLLOWUP_EVAL_VERSION:
        raise ValueError(
            f"{path.name} was written by eval version "
            f"{payload.get('eval_version')}, not {FOLLOWUP_EVAL_VERSION}"
        )
    return TypeAdapter(FollowUpRun).validate_python(payload)
