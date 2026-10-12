"""The retrieval gold set: hand-labelled questions with the citations a correct
retriever must return."""

from __future__ import annotations

import json
import re
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from taxverity.corpus.nodes import NodePath
from taxverity.observability import get_logger

logger = get_logger(__name__)

GOLD_V1_FILENAME = "retrieval_gold_v1.jsonl"
GOLD_V2_FILENAME = "retrieval_gold_v2.jsonl"
QUERY_ID = re.compile(r"^q\d{3}$")


class QuerySlice(StrEnum):
    CITATION = "citation"
    PARAPHRASE = "paraphrase"
    CROSSREF = "crossref"
    NEGATIVE = "negative"


class GoldQuery(BaseModel):
    model_config = ConfigDict(frozen=True)

    query_id: str
    slice: QuerySlice
    question: str
    required: tuple[str, ...]
    notes: str

    @field_validator("query_id")
    @classmethod
    def ids_are_ordinals(cls, value: str) -> str:
        if not QUERY_ID.match(value):
            raise ValueError(f"query_id {value!r} is not of the form q001")
        return value

    @field_validator("question", "notes")
    @classmethod
    def prose_is_present(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("question and notes are both required")
        return value

    @field_validator("required")
    @classmethod
    def labels_are_citations(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(set(value)) != len(value):
            raise ValueError(f"duplicate citation in {value}")
        for citation in value:
            NodePath.parse(citation)
        return value

    @model_validator(mode="after")
    def only_a_negative_has_no_answer(self) -> GoldQuery:
        negative = self.slice is QuerySlice.NEGATIVE
        if negative and self.required:
            raise ValueError(
                f"{self.query_id}: a negative query must have no citations"
            )
        if not negative and not self.required:
            raise ValueError(
                f"{self.query_id}: {self.slice} needs at least one citation"
            )
        return self


def to_json_line(query: GoldQuery) -> str:
    return json.dumps(query.model_dump(mode="json"), sort_keys=True, ensure_ascii=False)


def load_gold_set(path: Path) -> tuple[GoldQuery, ...]:
    if not path.is_file():
        raise FileNotFoundError(f"missing gold set at {path}")
    queries = tuple(
        GoldQuery.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    seen = {query.query_id for query in queries}
    if len(seen) != len(queries):
        raise ValueError(f"duplicate query_id in {path}")
    logger.info("loaded %d gold queries from %s", len(queries), path)
    return queries


def by_slice(queries: tuple[GoldQuery, ...]) -> dict[QuerySlice, tuple[GoldQuery, ...]]:
    return {
        member: tuple(query for query in queries if query.slice is member)
        for member in QuerySlice
    }
