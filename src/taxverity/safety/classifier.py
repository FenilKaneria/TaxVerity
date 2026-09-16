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
import re
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
    Provider,
    Usage,
)
from taxverity.llm.tracing import LangfuseTracer, TracedLLMClient
from taxverity.observability import get_logger

logger = get_logger(__name__)

CLASSIFIER_STAGE_VERSION = 4

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
        },
        # R19 Phase B (ADR-120): the question restated in the Act's own
        # vocabulary, used for retrieval instead of the person's raw wording.
        "search_query": {"type": "string"},
    },
    "required": ["category", "search_query"],
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

# R19 Phase C: a deterministic short-circuit for canonical small talk, per
# rule 01 ("prefer a deterministic check where one is possible"). ADR-117's
# 34-case re-measure never exercised the exact combined phrasing "Hello what
# can you do?", and Groq's model classified it out_of_scope on a live check —
# the model prompt already names this example, but a live model call cannot
# be trusted to honour its own instructions every time. Deliberately narrow:
# every pattern requires the WHOLE message (after light punctuation/whitespace
# normalisation) to match one exact small-talk shape, so a real tax question
# cannot collide with it just for containing a greeting word.
_GREETING = r"(?:hi|hello|hey|hiya|yo|greetings)"
_THANKS = r"(?:thanks?|thank you|thx|ty)"
_CAPABILITY = r"(?:what (?:can|do) you (?:do|help(?: me)? with)|who are you|what are you)"
_SMALL_TALK_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        rf"{_GREETING}( there)?",
        _THANKS,
        _CAPABILITY,
        rf"{_GREETING}[,!\s]*{_CAPABILITY}",
    )
)

# A shortcut never touches the wire, so it carries no usage to bill.
_SHORTCUT_COMPLETION = Completion(
    text="",
    provider="shortcut",
    model="none",
    finish_reason="stop",
    usage=Usage(),
    degraded=False,
)


def _looks_conversational(question: str) -> bool:
    normalised = re.sub(r"[.!?]+$", "", question.strip())
    normalised = re.sub(r"\s+", " ", normalised).strip()
    return any(pattern.fullmatch(normalised) for pattern in _SMALL_TALK_PATTERNS)


# The policy excerpt: condensed from docs/SAFETY_POLICY.md, not re-derived.
# Rule 03's boundary is restated in full (must refuse / must not refuse) since
# recall on the refuse side and precision on the allow side are both required,
# equally weighted (ADR-023).
SYSTEM_PROMPT = """\
You classify one person's message into exactly one category. You do not \
answer it.

Categories:
- in_scope: a question about the Income-tax Act, 2025 (India).
- conversational: a greeting, thanks, or a question about what you are and \
what you can do — for example "hi", "what can you help me with?", or "hello \
what can you do?". Not a tax question, and not unrelated either — do not \
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

Also give "search_query": the message restated in the Income-tax Act's own \
vocabulary, for retrieval — not an answer, and not for conversational, \
adjacent, out_of_scope or prohibited messages, where it may just repeat the \
message. For example "tax benefits for a home loan" becomes something like \
"interest on borrowed capital for acquisition or construction of a house \
property; deduction". Do not invent a section number.

Return only the category and search_query, as JSON."""


@dataclass(frozen=True)
class ClassificationResult:
    category: ScopeCategory
    # R19 Phase B (ADR-120): the question restated in the Act's own
    # vocabulary, used for retrieval in place of the person's raw wording.
    search_query: str
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
        primary: Provider | None = None,
        fallback: Provider | None = None,
        **kwargs: Any,
    ) -> IntentClassifier:
        """Same composition as Step 7.6's `FactExtractor`: tracing outside the
        cache (ADR-095). `primary`/`fallback` (R18) default to
        `LLMClient`'s own class defaults (Groq 120b / Gemini) — pass them to
        run this node against a different pair, e.g. Groq's 20b model."""
        client: Any = LLMClient.from_settings(settings, primary=primary, fallback=fallback)
        if cache:
            client = CachedLLMClient(client, settings.llm_cache_dir)
        if trace:
            client = TracedLLMClient(client, LangfuseTracer.from_settings(settings))
        return cls(client, **kwargs)

    def classify(self, question: str) -> ClassificationResult:
        if not question.strip():
            raise ValueError("question must be non-empty")
        if _looks_conversational(question):
            return ClassificationResult(
                category=ScopeCategory.CONVERSATIONAL,
                search_query=question,
                completion=_SHORTCUT_COMPLETION,
            )
        completion = self._complete(
            [
                Message(role="system", content=self._system_prompt),
                # Delimited, never concatenated as instructions (rule 03).
                Message(role="user", content=f"<question>\n{question}\n</question>"),
            ]
        )
        category, search_query = _parse(completion.text, question)
        return ClassificationResult(category=category, search_query=search_query, completion=completion)

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


def _parse(text: str, question: str) -> tuple[ScopeCategory, str]:
    try:
        payload = json.loads(text)
        category = ScopeCategory(payload["category"])
    except (ValueError, TypeError, KeyError) as error:
        raise ClassificationError(f"not a valid category: {text!r}") from error
    # A missing or blank search_query (the fallback json_object mode enforces
    # nothing at the wire, Step 7.1's finding) degrades to the raw question
    # rather than failing the whole classification over a field only
    # retrieval consumes.
    search_query = payload.get("search_query")
    if not isinstance(search_query, str) or not search_query.strip():
        search_query = question
    return category, search_query
