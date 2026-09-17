"""R20 Step 20.5 — what the `reason` LLM call returns, before validation.

Plain pydantic, `extra="forbid"` throughout — the same discipline
`safety/classifier.py` and `facts.py` already use for a model's strict-schema
output: an unexpected field is a parse failure, not silently ignored data.

**This module's shapes are data, not truth (PLAN R20's "Reasoning
validation" section).** Nothing here is served to a person directly — a
`LegalRule` or `AnswerPlan` only ever reaches `reasoning/validate.py`'s
deterministic checks, and only what survives those checks ever reaches
`generate` (20.7). A `markers` field is a list of 1-based positions into the
`EvidencePack` the reasoning call was shown — the same numbering
`generation/claims.py`'s `[n]` markers already use, so a marker here and a
citation marker in the eventual answer name the same passage.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

REASONING_STAGE_VERSION = 1


class CheckStatus(StrEnum):
    SATISFIED = "satisfied"
    NOT_SATISFIED = "not_satisfied"
    UNKNOWN = "unknown"
    NOT_APPLICABLE = "not_applicable"
    AMBIGUOUS = "ambiguous"


class ConclusionKind(StrEnum):
    DETERMINED = "determined"
    CONDITIONAL = "conditional"
    UNDETERMINABLE = "undeterminable"
    NO_BASIS = "no_basis"


class Condition(BaseModel):
    """One condition, limit or exception a `LegalRule` sets — the unit
    `ConditionCheck`/`MissingFact` refer to by `id`. `markers` may be empty,
    in which case validation grounds it against the parent rule's own
    markers (a condition drawn from the same passage as its rule needs no
    separate citation)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    markers: tuple[int, ...] = ()


class LegalRule(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(min_length=1)
    # The passages this rule is drawn from. Grounds `rule`, `limits`,
    # `exceptions` and `definitions` below; a `Condition` may cite these too
    # by leaving its own `markers` empty.
    markers: tuple[int, ...]
    rule: str = Field(min_length=1)
    conditions: tuple[Condition, ...] = ()
    limits: tuple[str, ...] = ()
    exceptions: tuple[str, ...] = ()
    definitions: tuple[str, ...] = ()


class ConditionCheck(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    condition_id: str = Field(min_length=1)
    status: CheckStatus
    # A `FactField` value or a situation fact's name, whichever the check was
    # decided from. Required by validation whenever status is `satisfied` or
    # `not_satisfied` (PLAN R20's own rule); `validate.py` downgrades the
    # check to `unknown` rather than dropping it when this is missing or
    # names no known fact.
    fact_refs: tuple[str, ...] = ()
    note: str = ""


class MissingFact(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    condition_id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    material: bool


class AnswerPlan(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    conclusion_kind: ConclusionKind
    steps: tuple[str, ...] = ()
    next_step: str = ""


class ReasoningAnalysis(BaseModel):
    """The whole raw completion, before `reasoning/validate.py` runs."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    legal_rules: tuple[LegalRule, ...] = ()
    applicability: tuple[ConditionCheck, ...] = ()
    missing_facts: tuple[MissingFact, ...] = ()
    answer_plan: AnswerPlan


REASONING_SCHEMA_NAME = "legal_reasoning"

# Hand-authored, not derived from the pydantic models above — the same
# choice `facts.py`/`safety/classifier.py` already made: a `$ref`-based
# schema pydantic would emit is not guaranteed to satisfy every provider's
# strict-mode subset, and a flat, explicit schema is what every other
# strict-schema call in this codebase already sends.
REASONING_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["legal_rules", "applicability", "missing_facts", "answer_plan"],
    "properties": {
        "legal_rules": {
            "type": "array",
            "description": (
                "The provisions that govern this question, one entry per "
                "distinct rule. Only rules you can point to a passage for."
            ),
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["id", "markers", "rule", "conditions", "limits", "exceptions", "definitions"],
                "properties": {
                    "id": {"type": "string", "description": "A short id you invent, referenced by applicability and missing_facts."},
                    "markers": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "The passage numbers, exactly as shown, this rule is drawn from.",
                    },
                    "rule": {"type": "string", "description": "The rule itself, in your own words, close to the cited passage."},
                    "conditions": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["id", "text", "markers"],
                            "properties": {
                                "id": {"type": "string"},
                                "text": {"type": "string"},
                                "markers": {
                                    "type": "array",
                                    "items": {"type": "integer"},
                                    "description": "Leave empty to reuse the rule's own markers.",
                                },
                            },
                        },
                    },
                    "limits": {"type": "array", "items": {"type": "string"}},
                    "exceptions": {"type": "array", "items": {"type": "string"}},
                    "definitions": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
        "applicability": {
            "type": "array",
            "description": (
                "For each condition above, whether the person's own facts "
                "(given below) satisfy it."
            ),
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["condition_id", "status", "fact_refs", "note"],
                "properties": {
                    "condition_id": {"type": "string"},
                    "status": {"type": "string", "enum": [s.value for s in CheckStatus]},
                    "fact_refs": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "The field name or situation fact this check was decided "
                            "from. Required when status is satisfied or not_satisfied."
                        ),
                    },
                    "note": {"type": "string"},
                },
            },
        },
        "missing_facts": {
            "type": "array",
            "description": (
                "A question to ask the person, for each condition whose status is "
                "unknown and would change the conclusion. Never invent a rate, "
                "threshold or amount in the question."
            ),
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["condition_id", "question", "material"],
                "properties": {
                    "condition_id": {"type": "string"},
                    "question": {"type": "string"},
                    "material": {
                        "type": "boolean",
                        "description": "True only if knowing the answer would change the conclusion.",
                    },
                },
            },
        },
        "answer_plan": {
            "type": "object",
            "additionalProperties": False,
            "required": ["conclusion_kind", "steps", "next_step"],
            "properties": {
                "conclusion_kind": {
                    "type": "string",
                    "enum": [k.value for k in ConclusionKind],
                    "description": (
                        "determined: the facts fully decide it. conditional: it "
                        "depends on something the person hasn't said. "
                        "undeterminable: the Act doesn't give enough to decide "
                        "even with more facts. no_basis: nothing here answers this."
                    ),
                },
                "steps": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "The order the answer should walk through: what applies, then how, then the result.",
                },
                "next_step": {"type": "string", "description": "What the answer should say is still needed, if anything."},
            },
        },
    },
}
