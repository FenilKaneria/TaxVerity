"""Step 20.1 — the R20 retrieval-quality benchmark: a small, hand-verified
set of reasoning-shaped questions, each carrying not just the governing
citation (as `evals.gold.GoldQuery` already does) but the full set of
provisions a correct *legal rule extraction* needs — the governing
provision plus its conditions, limits, exceptions/provisos, and any
cross-referenced provision the rule cannot be stated correctly without.

This is deliberately a separate file from `retrieval_gold_v2.jsonl`
(ADR-064's file stays frozen and untouched — Step 3.7's own anti-
overfitting rule) rather than an edit to it. Several rows reuse a v2
`query_id` and `question` verbatim (the benchmark quotes them, not the
frozen file) precisely because they are already reasoning-shaped questions
(HRA, house-property loss set-off, salary standard deduction under the
default regime) — widening what is measured about them, not re-labelling
what v2 already measures.

`rule_units` is a superset of `required`: `required` names the smallest
node that answers a plain citation lookup (what `evals.gold.GoldQuery`
already measures); `rule_units` names every provision a rule extraction
step needs to state the rule *completely* — its conditions, its limits, an
exception it does not apply under, and a cross-reference it names by
number. Every citation in every row was checked against the real corpus
text before being written here (see `reports/r20_retrieval_diagnosis.md`
for the verification transcript) — this file does not restate anything
that has not been read.
"""

from __future__ import annotations

import json
import re
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from taxverity.corpus.nodes import NodePath
from taxverity.observability import get_logger

logger = get_logger(__name__)

R20_GOLD_FILENAME = "retrieval_gold_r20.jsonl"
QUERY_ID = re.compile(r"^r\d{3}$")


class R20Slice(StrEnum):
    ELIGIBILITY = "eligibility"
    APPLICABILITY = "applicability"
    CALCULATION = "calculation"
    DEDUCTION_LIMIT = "deduction_limit"
    MULTI_ISSUE = "multi_issue"
    NEGATIVE = "negative"


class R20GoldQuery(BaseModel):
    model_config = ConfigDict(frozen=True)

    query_id: str
    slice: R20Slice
    question: str
    # The governing provision(s) alone — same meaning as evals.gold's `required`.
    required: tuple[str, ...]
    # Governing provision + its conditions/limits/exceptions/cross-references —
    # everything a complete legal-rule extraction needs. Always a superset of
    # `required` (checked below).
    rule_units: tuple[str, ...]
    notes: str

    @field_validator("query_id")
    @classmethod
    def ids_are_ordinals(cls, value: str) -> str:
        if not QUERY_ID.match(value):
            raise ValueError(f"query_id {value!r} is not of the form r001")
        return value

    @field_validator("question", "notes")
    @classmethod
    def prose_is_present(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("question and notes are both required")
        return value

    @field_validator("required", "rule_units")
    @classmethod
    def labels_are_citations(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(set(value)) != len(value):
            raise ValueError(f"duplicate citation in {value}")
        for citation in value:
            NodePath.parse(citation)
        return value

    @model_validator(mode="after")
    def only_a_negative_has_no_answer(self) -> R20GoldQuery:
        negative = self.slice is R20Slice.NEGATIVE
        if negative and (self.required or self.rule_units):
            raise ValueError(f"{self.query_id}: a negative query must have no citations")
        if not negative and not self.required:
            raise ValueError(f"{self.query_id}: {self.slice} needs at least one citation")
        return self

    @model_validator(mode="after")
    def rule_units_cover_required(self) -> R20GoldQuery:
        if self.slice is R20Slice.NEGATIVE:
            return self
        missing = set(self.required) - set(self.rule_units)
        if missing:
            raise ValueError(f"{self.query_id}: rule_units omits required citation(s) {missing}")
        if len(self.rule_units) <= len(self.required):
            raise ValueError(
                f"{self.query_id}: rule_units must add at least one condition/limit/"
                f"exception/cross-reference beyond the governing citation(s)"
            )
        return self


def load_r20_gold_set(path: Path) -> tuple[R20GoldQuery, ...]:
    if not path.is_file():
        raise FileNotFoundError(f"missing R20 gold set at {path}")
    queries = tuple(
        R20GoldQuery.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    seen = {query.query_id for query in queries}
    if len(seen) != len(queries):
        raise ValueError(f"duplicate query_id in {path}")
    logger.info("loaded %d R20 gold queries from %s", len(queries), path)
    return queries


def to_json_line(query: R20GoldQuery) -> str:
    return json.dumps(query.model_dump(mode="json"), sort_keys=True, ensure_ascii=False)
