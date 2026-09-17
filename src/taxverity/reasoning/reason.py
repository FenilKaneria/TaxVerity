"""R20 Step 20.5 — the `reason` call itself: one strict-schema completion,
composed exactly like `safety/classifier.py`'s `IntentClassifier` and
`llm/extract.py`'s `FactExtractor` (tracing outside the cache, ADR-095;
strict `json_schema` first, `json_object` on a provider refusal, Step 7.1's
measured fallback). No repair retry — unlike extraction's closed 15 fields,
a malformed or ungrounded entry here is simply dropped by
`reasoning/validate.py`, and 20.5's own scope note commits to no new
mechanism beyond what that deterministic pass already buys.

A malformed completion (not valid JSON, or failing the pydantic shape even
under the lenient `json_object` fallback) is not an exception — `reason()`
returns `analysis=None`, and the caller (the `reason` graph node) falls
back to plain generation over the pack, the same "nothing survives" path
`reasoning/validate.py` already defines for a well-formed-but-empty
analysis. A reasoning failure must never stop a turn from answering (rule
01: `reason` is additive, not a grounding gate — the verifier is)."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from taxverity.calculator.scope import Computation
from taxverity.config import Settings
from taxverity.llm.cache import CachedLLMClient
from taxverity.llm.client import (
    Completion,
    LLMClient,
    LLMRequestError,
    Message,
    Provider,
)
from taxverity.llm.tracing import LangfuseTracer, TracedLLMClient
from taxverity.memory.fact_state import ThreadFactState
from taxverity.observability import get_logger
from taxverity.reasoning.models import (
    REASONING_JSON_SCHEMA,
    REASONING_SCHEMA_NAME,
    ReasoningAnalysis,
)
from taxverity.reasoning.prompt import SYSTEM_PROMPT, render_reason_prompt
from taxverity.retrieval.evidence import EvidencePack

logger = get_logger(__name__)

REASON_STAGE_VERSION = 1

# Reasoning cannot be disabled and is billed against this cap regardless
# (Step 7.1). The completion is structurally larger than extraction's or
# the classifier's — several rules, each with several conditions — so this
# starts above both of theirs.
REASON_MAX_COMPLETION_TOKENS = 1_536
REASON_TEMPERATURE = 0.0

STRICT_FORMAT: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": REASONING_SCHEMA_NAME,
        "schema": REASONING_JSON_SCHEMA,
        "strict": True,
    },
}

# Step 7.1's measured fallback mode. Nothing is enforced at the wire under
# it, so `_parse` polices the shape either way, exactly as extract.py's does.
OBJECT_FORMAT: dict[str, Any] = {"type": "json_object"}


@dataclass(frozen=True)
class ReasonResult:
    # None means the completion did not parse as a `ReasoningAnalysis` at
    # all — the caller falls back to plain generation, it never raises.
    analysis: ReasoningAnalysis | None
    completion: Completion

    @property
    def tokens(self) -> int:
        return self.completion.usage.total_tokens


class Reasoner:
    def __init__(
        self,
        client: Any,
        *,
        system_prompt: str = SYSTEM_PROMPT,
        max_completion_tokens: int = REASON_MAX_COMPLETION_TOKENS,
    ) -> None:
        self._client = client
        self._system_prompt = system_prompt
        self._max_completion_tokens = max_completion_tokens
        # A provider that refuses the schema refuses it every time (Step
        # 7.6's finding), so the refusal is paid once per process.
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
    ) -> Reasoner:
        """Same composition as `FactExtractor`/`IntentClassifier`: tracing
        outside the cache (ADR-095). `primary`/`fallback` default to
        `LLMClient`'s own class defaults (Groq 120b / Gemini)."""
        client: Any = LLMClient.from_settings(settings, primary=primary, fallback=fallback)
        if cache:
            client = CachedLLMClient(client, settings.llm_cache_dir)
        if trace:
            client = TracedLLMClient(client, LangfuseTracer.from_settings(settings))
        return cls(client, **kwargs)

    def reason(
        self,
        question: str,
        pack: EvidencePack,
        fact_state: ThreadFactState,
        computation: Computation | None,
    ) -> ReasonResult:
        prompt = render_reason_prompt(question, pack, fact_state, computation)
        completion = self._complete(
            [
                Message(role="system", content=self._system_prompt),
                Message(role="user", content=prompt),
            ]
        )
        return ReasonResult(analysis=_parse(completion.text), completion=completion)

    def _complete(self, messages: Sequence[Message]) -> Completion:
        if self.schema_refused:
            return self._call(messages, OBJECT_FORMAT)
        try:
            return self._call(messages, STRICT_FORMAT)
        except LLMRequestError as error:
            logger.warning(
                "provider refused the strict reasoning schema (%s); using json_object",
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
            temperature=REASON_TEMPERATURE,
        )


def _parse(text: str) -> ReasoningAnalysis | None:
    try:
        payload = json.loads(text)
        return ReasoningAnalysis.model_validate(payload)
    except (ValueError, ValidationError) as error:
        logger.warning("reasoning completion did not parse: %s", error)
        return None
