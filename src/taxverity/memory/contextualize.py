"""Step 11.7 — query contextualization, simplified (ADR-110).

A follow-up turn ("what if I'm 65 instead?") is unretrievable on its own: it
shares no vocabulary with the statute. The fix is a rewrite into a standalone
question, using only the recent-turns window (Step 11.6's
`threads.store.list_messages(last=...)`) to resolve what it refers to — never
as a source of fact truth, exactly as rule 04 requires.

Rule 01's recurring test applies here too: a deterministic check beats a model
call wherever one exists. Most turns are not follow-ups at all — no prior
turn, or no pronoun/reference marker — so `needs_contextualization` decides
that in code, for free, before any request is built. Only a genuine follow-up
pays for one `complete()` call.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from taxverity.config import Settings
from taxverity.llm.cache import CachedLLMClient
from taxverity.llm.client import Completion, LLMClient, Message, Provider
from taxverity.llm.tracing import LangfuseTracer, TracedLLMClient
from taxverity.observability import get_logger

logger = get_logger(__name__)

CONTEXTUALIZE_STAGE_VERSION = 1

# A rewritten question is short; reasoning cannot be disabled (Step 7.1) but
# needs far less room here than an extraction's JSON object does.
CONTEXTUALIZE_MAX_COMPLETION_TOKENS = 300
CONTEXTUALIZE_TEMPERATURE = 0.0

# Pronouns and follow-up phrasing a standalone question would not need. Not
# exhaustive — a missed marker just means one more question answered as if it
# were standalone, the same failure mode a first turn already has.
FOLLOW_UP_MARKERS = re.compile(
    r"\b(it|this|that|these|those|them|he|she|instead|same|"
    r"also|again|otherwise)\b|what if|what about",
    re.IGNORECASE,
)

SYSTEM_PROMPT = (
    "You rewrite a person's latest message as one standalone question, using "
    "the prior turns only to resolve what it refers to.\n\n"
    "Do not answer the question. Do not add any fact the person did not "
    "state. Return only the rewritten question, nothing else."
)


def needs_contextualization(query: str, prior_turns: Sequence[str]) -> bool:
    """No prior turn, or no follow-up marker: the query already stands alone."""
    if not prior_turns:
        return False
    return bool(FOLLOW_UP_MARKERS.search(query))


@dataclass(frozen=True)
class ContextualizationResult:
    query: str
    rewritten: bool
    completion: Completion | None

    @property
    def tokens(self) -> int:
        return self.completion.usage.total_tokens if self.completion else 0


class QueryContextualizer:
    def __init__(
        self,
        client: Any,
        *,
        system_prompt: str = SYSTEM_PROMPT,
        max_completion_tokens: int = CONTEXTUALIZE_MAX_COMPLETION_TOKENS,
    ) -> None:
        self._client = client
        self._system_prompt = system_prompt
        self._max_completion_tokens = max_completion_tokens

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
    ) -> QueryContextualizer:
        """Same composition as Step 7.6's `FactExtractor`: tracing outside the
        cache (ADR-095), so a cache hit is still traced as an ordinary
        generation. `primary`/`fallback` (R18) default to `LLMClient`'s own
        class defaults (Groq 120b / Gemini) — pass them to run this node
        against a different pair, e.g. Groq's 20b model."""
        client: Any = LLMClient.from_settings(settings, primary=primary, fallback=fallback)
        if cache:
            client = CachedLLMClient(client, settings.llm_cache_dir)
        if trace:
            client = TracedLLMClient(client, LangfuseTracer.from_settings(settings))
        return cls(client, **kwargs)

    def contextualize(
        self, query: str, prior_turns: Sequence[str]
    ) -> ContextualizationResult:
        if not needs_contextualization(query, prior_turns):
            return ContextualizationResult(query=query, rewritten=False, completion=None)

        completion = self._client.complete(
            [
                Message(role="system", content=self._system_prompt),
                Message(role="user", content=_prompt(query, prior_turns)),
            ],
            max_completion_tokens=self._max_completion_tokens,
            temperature=CONTEXTUALIZE_TEMPERATURE,
        )
        rewritten = completion.text.strip()
        if not rewritten:
            # The query itself is never logged (Rule 03) — only that the
            # fallback fired.
            logger.warning(
                "contextualization returned no text; answering the original query"
            )
            return ContextualizationResult(query=query, rewritten=False, completion=completion)
        return ContextualizationResult(query=rewritten, rewritten=True, completion=completion)


def _prompt(query: str, prior_turns: Sequence[str]) -> str:
    history = "\n".join(f"- {turn}" for turn in prior_turns)
    return f"Prior turns:\n{history}\n\nLatest message: {query}"
