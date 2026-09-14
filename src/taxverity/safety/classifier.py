"""Step 12.2 — the intent classifier (ESSENTIAL, RIGOROUS).

Every turn is classified into one of four categories, per `docs/SAFETY_POLICY.md`
and rule 03, before retrieval runs. This is a strict-schema `complete()` call at
temperature 0, not a keyword pre-filter — Step 3.6/4.6/5.6 already measured that
no lexical or vector score separates negatives, and a keyword filter over
lawful-planning vocabulary ("rent to my mother", "backdate") would over-refuse
the exact questions rule 03 protects.

`adjacent`, `out_of_scope` and `prohibited` map to the fixed templates in
`docs/SAFETY_POLICY.md`, copied verbatim below. Only `in_scope` reaches
retrieval and generation; the other three are never composed by the model.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from taxverity.config import Settings
from taxverity.llm.cache import CachedLLMClient
from taxverity.llm.client import (
    Completion,
    LLMClient,
    LLMRequestError,
    Message,
)
from taxverity.llm.tracing import LangfuseTracer, TracedLLMClient
from taxverity.observability import get_logger

logger = get_logger(__name__)

CLASSIFIER_STAGE_VERSION = 2

# The answer is one word; reasoning cannot be disabled and is billed against
# this cap regardless (Step 7.1), so this stays small but not tight.
CLASSIFIER_MAX_COMPLETION_TOKENS = 200
CLASSIFIER_TEMPERATURE = 0.0

SCHEMA_NAME = "scope_classification"


class ScopeCategory(StrEnum):
    IN_SCOPE = "in_scope"
    # A greeting, thanks, or "what can you do" - not a tax question, but not
    # a refusal either. Routes to a guarded LLM reply (llm/conversational.py),
    # never to retrieval or a fixed refusal template.
    CONVERSATIONAL = "conversational"
    ADJACENT = "adjacent"
    OUT_OF_SCOPE = "out_of_scope"
    PROHIBITED = "prohibited"


class ClassificationError(RuntimeError):
    """The completion did not name one of the four categories.

    Never guessed past — a safety classifier that defaults silently on a
    malformed answer is exactly the kind of weakened gate rule 01 forbids.
    """


CLASSIFICATION_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "category": {
            "type": "string",
            "enum": [c.value for c in ScopeCategory],
        }
    },
    "required": ["category"],
    "additionalProperties": False,
}

STRICT_FORMAT: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": SCHEMA_NAME,
        "schema": CLASSIFICATION_JSON_SCHEMA,
        "strict": True,
    },
}

# Step 7.1's measured fallback mode. Nothing is enforced at the wire under it,
# so _parse() polices the shape either way, exactly as extract.py's does.
OBJECT_FORMAT: dict[str, Any] = {"type": "json_object"}

# Fixed response templates, copied verbatim from docs/SAFETY_POLICY.md. Kept
# here, not composed by the model, per rule 03: "Redirect and refusal texts
# are fixed templates, not generated." `IN_SCOPE` has no entry — it reaches
# retrieval and generation instead of a canned response.
FIXED_RESPONSES: dict[ScopeCategory, str] = {
    ScopeCategory.ADJACENT: (
        "That's outside the Income-tax Act, 2025, which is what I cover — it "
        "looks like a GST, company-law, or accounting question instead. I "
        "can't give a grounded answer to it here."
    ),
    ScopeCategory.OUT_OF_SCOPE: (
        "That's outside what I can help with — I answer questions about the "
        "Income-tax Act, 2025 only."
    ),
    ScopeCategory.PROHIBITED: (
        "I can't help with that — it would involve misrepresenting facts to "
        "the tax authority (for example, concealing income, fabricating a "
        "document, or disguising a transaction). I can help with lawful tax "
        "planning instead: choosing between regimes, timing a deduction, or "
        "checking what you're actually entitled to claim."
    ),
}

# The policy excerpt: condensed from docs/SAFETY_POLICY.md, not re-derived.
# Rule 03's boundary is restated in full (must refuse / must not refuse) since
# recall on the refuse side and precision on the allow side are both required,
# equally weighted (ADR-023).
SYSTEM_PROMPT = """\
You classify one person's tax question into exactly one category. You do not \
answer it.

Categories:
- in_scope: a question about the Income-tax Act, 2025 (India).
- conversational: a greeting, thanks, or a question about what you are and \
what you can do. Not a tax question, and not unrelated either — do not \
classify small talk as out_of_scope.
- adjacent: a real tax or business topic, but a different law — GST, company \
registration, accounting standards.
- out_of_scope: unrelated to tax or this Act entirely.
- prohibited: asks for help misrepresenting facts to the tax authority.

The line for "prohibited" is misrepresentation of facts, not minimising tax. \
Minimising tax lawfully is in_scope.

Classify as prohibited: concealing income, fabricating a receipt, rent \
agreement or donation certificate, backdating an investment, a benami or \
shell-entity structure, hawala, a fake HUF to split income, deliberately \
misreporting residential status, evading TDS.

Classify as in_scope, never prohibited: choosing between the old and new \
regime, timing an investment before or after year-end, claiming a deduction \
the person is actually entitled to, structuring salary components lawfully, \
comparing HRA exemption against home-loan-interest deduction, a comparative \
"what if" question, and paying rent to a parent and claiming HRA on it — that \
last one is a real transaction the Act does not bar.

Return only the category, as JSON."""


@dataclass(frozen=True)
class ClassificationResult:
    category: ScopeCategory
    completion: Completion

    @property
    def tokens(self) -> int:
        return self.completion.usage.total_tokens

    @property
    def response(self) -> str | None:
        """The fixed template to serve, or None when retrieval should run."""
        return FIXED_RESPONSES.get(self.category)


class IntentClassifier:
    def __init__(
        self,
        client: Any,
        *,
        system_prompt: str = SYSTEM_PROMPT,
        max_completion_tokens: int = CLASSIFIER_MAX_COMPLETION_TOKENS,
    ) -> None:
        self._client = client
        self._system_prompt = system_prompt
        self._max_completion_tokens = max_completion_tokens
        # A provider that refuses the schema refuses it every time (Step 7.6's
        # finding), so the refusal is paid once per process, not once per turn.
        self.schema_refused = False

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        *,
        cache: bool = True,
        trace: bool = True,
        **kwargs: Any,
    ) -> IntentClassifier:
        """Same composition as Step 7.6's `FactExtractor`: tracing outside the
        cache (ADR-095)."""
        client: Any = LLMClient.from_settings(settings)
        if cache:
            client = CachedLLMClient(client, settings.llm_cache_dir)
        if trace:
            client = TracedLLMClient(client, LangfuseTracer.from_settings(settings))
        return cls(client, **kwargs)

    def classify(self, question: str) -> ClassificationResult:
        if not question.strip():
            raise ValueError("question must be non-empty")
        completion = self._complete(
            [
                Message(role="system", content=self._system_prompt),
                # Delimited, never concatenated as instructions (rule 03).
                Message(role="user", content=f"<question>\n{question}\n</question>"),
            ]
        )
        category = _parse(completion.text)
        return ClassificationResult(category=category, completion=completion)

    def _complete(self, messages: Sequence[Message]) -> Completion:
        if self.schema_refused:
            return self._call(messages, OBJECT_FORMAT)
        try:
            return self._call(messages, STRICT_FORMAT)
        except LLMRequestError as error:
            logger.warning(
                "provider refused the strict scope schema (%s); using json_object",
                error,
            )
            self.schema_refused = True
        return self._call(messages, OBJECT_FORMAT)

    def _call(
        self, messages: Sequence[Message], response_format: Mapping[str, Any]
    ) -> Completion:
        return self._client.complete(
            messages,
            max_completion_tokens=self._max_completion_tokens,
            response_format=response_format,
            temperature=CLASSIFIER_TEMPERATURE,
        )


def _parse(text: str) -> ScopeCategory:
    try:
        payload = json.loads(text)
        return ScopeCategory(payload["category"])
    except (ValueError, TypeError, KeyError) as error:
        raise ClassificationError(f"not a valid category: {text!r}") from error
