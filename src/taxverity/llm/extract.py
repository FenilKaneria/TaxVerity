"""Step 7.6 — the extraction node: one turn in, a `UserFacts` out.

The first code that composes the Phase 7 stack. Step 7.2 built the client, 7.3
the offline cache and 7.4 the tracer with no caller between them; this is the
caller. Step 7.5 built the schema and the parser with no producer; this is the
producer.

Two things shape it. The parser refuses rather than raises, so a bad entry
arrives as a `Rejection` carrying the entry itself — which is exactly what a
repair prompt needs to quote back. And the repair is bounded at one attempt:
rule 02 rejects unbounded agentic loops, and an entry the model could not fix
when told precisely what was wrong is better reported than re-asked.

Which fields are `missing` is decided here, in code, not by the model. Absence
is a set difference, and a deterministic check beats a model call wherever one
exists (rule 01). It also keeps every turn's output to the fields actually
found, which matters on a free tier where tokens bind before requests do.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from taxverity.config import Settings
from taxverity.facts import (
    FACTS_JSON_SCHEMA,
    FACTS_SCHEMA_NAME,
    Extraction,
    Fact,
    FactField,
    FactIssue,
    FactStatus,
    Rejection,
    UnmappedFact,
    UserFacts,
    parse_facts,
)
from taxverity.llm.cache import CachedLLMClient
from taxverity.llm.client import (
    Completion,
    LLMClient,
    LLMRequestError,
    Message,
)
from taxverity.llm.tracing import LangfuseTracer, TracedLLMClient
from taxverity.observability import get_logger, redact

logger = get_logger(__name__)

EXTRACTION_STAGE_VERSION = 3

# Step 7.1: reasoning cannot be disabled and is billed against this cap, so a
# cap sized for the visible JSON alone truncates it mid-object.
EXTRACTION_MAX_COMPLETION_TOKENS = 2_048

# Extraction is a copying task, not a creative one.
EXTRACTION_TEMPERATURE = 0.0

STRICT_FORMAT: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": FACTS_SCHEMA_NAME,
        "schema": FACTS_JSON_SCHEMA,
        "strict": True,
    },
}

# Step 7.1 measured this as the fallback mode. Nothing is enforced at the wire
# under it, which is why parse_facts() polices the shape either way.
OBJECT_FORMAT: dict[str, Any] = {"type": "json_object"}

_SHAPE = FACTS_JSON_SCHEMA["properties"]["fields"]["description"]

SYSTEM_PROMPT = (
    "You read one message from a person describing their tax situation and "
    "report the facts it contains, as JSON.\n\n"
    'Return an object with one key, "fields", holding an array. Each entry has '
    '"name", "value", "status" and "source_span".\n\n'
    "Report only fields the message actually contains. Do not list a field the "
    "message says nothing about; a field you omit is recorded as missing.\n\n"
    'Use status "stated" when the person says the value outright, and copy '
    '"source_span" from their words character for character. Use "inferred" '
    "when the value follows from what they wrote but is not written there, and "
    'leave "source_span" empty. Never write a source_span the message does not '
    "contain.\n\n"
    'Write "value" as digits only for an amount: no separators, no currency '
    "symbol, no words such as lakh. A field that can be negative takes a "
    "leading minus sign when the person describes the amount as a loss or as "
    'negative, for example "-50000".\n\n'
    "The fields are: " + _SHAPE
)

# An unknown field is a measurement of what the closed vocabulary misses (7.5),
# and asking the model to rename it into the enum would force a wrong field in
# and destroy the thing 7.7 counts. A duplicate is not repaired either: the
# first occurrence is already kept, and a second value is a conflict, which
# Phase 11 owns.
REPAIRABLE = frozenset(
    {
        FactIssue.BAD_STATUS,
        FactIssue.MALFORMED_ENTRY,
        FactIssue.MISSING_SPAN,
        FactIssue.SIGN_CONTRADICTS_SPAN,
        FactIssue.SPAN_NOT_IN_TURN,
        FactIssue.UNPARSABLE_VALUE,
        FactIssue.VALUE_OUT_OF_DOMAIN,
    }
)

REPAIR_HINTS: dict[FactIssue, str] = {
    FactIssue.BAD_STATUS: 'status must be exactly "stated", "inferred" or "missing".',
    FactIssue.MALFORMED_ENTRY: (
        'Return one JSON object whose only key is "fields", holding an array of '
        "entries."
    ),
    FactIssue.MISSING_SPAN: (
        "A stated fact must quote the person's own words. Copy them into "
        'source_span, or use status "inferred" and leave source_span empty.'
    ),
    FactIssue.SPAN_NOT_IN_TURN: (
        "source_span must appear in the message character for character. Copy it "
        "from the message, or drop this field if the message does not contain it."
    ),
    FactIssue.SIGN_CONTRADICTS_SPAN: (
        "The words around the quoted figure describe a loss but the value is "
        "positive. Write a loss with a leading minus sign. If this figure is not a "
        "loss, quote only the words that give it; a positive figure stays refused "
        "while its own clause describes a loss."
    ),
    FactIssue.UNPARSABLE_VALUE: (
        "Write the value as digits only, with no separators, currency symbol or words."
    ),
    FactIssue.VALUE_OUT_OF_DOMAIN: (
        "The value is outside what this field accepts. Re-read the field's "
        "description and correct it, or drop the field."
    ),
}


@dataclass(frozen=True)
class ExtractionResult:
    facts: UserFacts
    # What one repair could not fix, plus everything repair is not allowed to
    # touch. 7.7 counts these; it never sees a silent drop.
    rejections: tuple[Rejection, ...]
    # What the repair was asked to fix, kept so 7.7 can say whether it helped:
    # `rejections` alone only shows what survived.
    repairable: tuple[Rejection, ...]
    repaired: bool
    completions: tuple[Completion, ...]

    @property
    def tokens(self) -> int:
        return sum(completion.usage.total_tokens for completion in self.completions)

    @property
    def degraded(self) -> bool:
        return any(completion.degraded for completion in self.completions)


class FactExtractor:
    def __init__(
        self,
        client: Any,
        *,
        repair: bool = True,
        system_prompt: str = SYSTEM_PROMPT,
        max_completion_tokens: int = EXTRACTION_MAX_COMPLETION_TOKENS,
    ) -> None:
        self._client = client
        self._repair = repair
        self._system_prompt = system_prompt
        self._max_completion_tokens = max_completion_tokens
        # A provider that refuses the schema refuses it every time, so the
        # refusal is paid once per process rather than once per turn.
        self.schema_refused = False

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        *,
        cache: bool = True,
        trace: bool = True,
        **kwargs: Any,
    ) -> FactExtractor:
        """Builds the Phase 7 stack: tracing outside the cache, per ADR-095.

        A cache hit is traced as an ordinary generation, because a hit returns
        the stored completion and the tracer cannot tell the difference. An eval
        report reads tokens from the client, never from a count of traces.
        """
        client: Any = LLMClient.from_settings(settings)
        if cache:
            client = CachedLLMClient(client, settings.llm_cache_dir)
        if trace:
            client = TracedLLMClient(client, LangfuseTracer.from_settings(settings))
        return cls(client, **kwargs)

    def extract(self, turn: str) -> ExtractionResult:
        if not turn.strip():
            raise ValueError("turn must be non-empty")

        # The span check runs against the turn as the model saw it, not as the
        # user typed it. redact() masks the PAN inside LLMClient, so a quotation
        # spanning one can only ever match the masked text — and a span that did
        # carry the real PAN would put it straight into the fact state Phase 11
        # persists. Rule 03's egress obligation reaches what we store, not only
        # what we send.
        seen = redact(turn)

        completions: list[Completion] = []
        first = self._ask(
            [
                Message(role="system", content=self._system_prompt),
                Message(role="user", content=turn),
            ],
            completions,
        )
        extraction = self._parse(first.text, seen)

        repairable = tuple(r for r in extraction.rejections if r.issue in REPAIRABLE)
        if not (repairable and self._repair):
            return ExtractionResult(
                facts=_complete_missing(extraction.facts),
                rejections=extraction.rejections,
                repairable=repairable,
                repaired=False,
                completions=tuple(completions),
            )

        # A retry, which rule 01 logs without exception.
        logger.warning(
            "repairing %d of %d rejected fact entries",
            len(repairable),
            len(extraction.rejections),
        )
        second = self._ask(
            [
                Message(role="system", content=self._system_prompt),
                Message(role="user", content=turn),
                Message(role="assistant", content=first.text),
                Message(role="user", content=_repair_prompt(repairable)),
            ],
            completions,
        )
        merged = _merge(extraction, self._parse(second.text, seen), repairable)
        return ExtractionResult(
            facts=_complete_missing(merged.facts),
            rejections=merged.rejections,
            repairable=repairable,
            repaired=True,
            completions=tuple(completions),
        )

    def _ask(
        self, messages: Sequence[Message], completions: list[Completion]
    ) -> Completion:
        completion = self._complete(messages)
        completions.append(completion)
        return completion

    def _complete(self, messages: Sequence[Message]) -> Completion:
        if self.schema_refused:
            return self._call(messages, OBJECT_FORMAT)
        try:
            return self._call(messages, STRICT_FORMAT)
        except LLMRequestError as error:
            # ADR-093 does not fail over on a request error, so without this the
            # whole node dies against a provider that cannot do strict schemas.
            # The shape is then policed by parse_facts() alone.
            logger.warning(
                "provider refused the strict fact schema (%s); using json_object",
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
            temperature=EXTRACTION_TEMPERATURE,
        )

    def _parse(self, text: str, turn: str) -> Extraction:
        """A body that is not JSON is a rejection, the same as a bad field."""
        try:
            payload = json.loads(text)
        except ValueError as error:
            return Extraction(
                facts=UserFacts(),
                rejections=(
                    Rejection(
                        FactIssue.MALFORMED_ENTRY,
                        f"the completion is not JSON: {error}",
                        {},
                    ),
                ),
            )
        return parse_facts(payload, turn)


def _repair_prompt(rejections: Sequence[Rejection]) -> str:
    lines = [
        "These entries were rejected. Return the corrected entries and nothing "
        "else: do not repeat the entries that were accepted, and drop any entry "
        "you cannot correct from the message itself.",
        "",
    ]
    for rejection in rejections:
        entry = json.dumps(rejection.entry, sort_keys=True, ensure_ascii=False)
        lines.append(f"{entry}\n  rejected: {rejection.detail}")
        lines.append(f"  {REPAIR_HINTS[rejection.issue]}")
    return "\n".join(lines)


def _merge(
    first: Extraction, second: Extraction, repairable: Sequence[Rejection]
) -> Extraction:
    """Pass one owns every field it accepted; pass two may only fill the gaps.

    Repair was asked for the rejected entries alone, so a field pass one already
    accepted is not pass two's to change — and a changed one is recorded as a
    duplicate rather than dropped quietly.
    """
    kept = {fact.field: fact for fact in first.facts.facts}
    rejections = [r for r in first.rejections if r.issue not in REPAIRABLE]
    unmapped: list[UnmappedFact] = list(first.facts.unmapped)
    repaired: set[FactField] = set()

    for fact in second.facts.facts:
        if fact.field in kept:
            rejections.append(
                Rejection(
                    FactIssue.DUPLICATE_FIELD,
                    f"{fact.field} was already accepted before the repair",
                    {"name": fact.field.value, "value": fact.raw_value},
                )
            )
            continue
        kept[fact.field] = fact
        repaired.add(fact.field)

    unmapped.extend(second.facts.unmapped)
    rejections.extend(second.rejections)
    # An entry the repair simply did not return is still unrepaired, and 7.7
    # must see it, reported as it was rejected the first time. A rejection
    # naming no field is a whole-payload failure rather than a field's, and the
    # second attempt already carries its own verdict on that — re-reporting the
    # first would say the same thing twice.
    for rejection in repairable:
        field = _entry_field(rejection)
        if field is not None and field not in repaired:
            rejections.append(rejection)
    return Extraction(
        facts=UserFacts(facts=tuple(_ordered(kept.values())), unmapped=tuple(unmapped)),
        rejections=tuple(rejections),
    )


def _entry_field(rejection: Rejection) -> FactField | None:
    try:
        return FactField(str(rejection.entry.get("name", "")).strip().casefold())
    except ValueError:
        return None


def _complete_missing(facts: UserFacts) -> UserFacts:
    """Absence is a set difference, not something a model has to be asked for."""
    reported = {fact.field for fact in facts.facts}
    filled = list(facts.facts) + [
        Fact(
            field=field,
            status=FactStatus.MISSING,
            raw_value="",
            value=None,
            source_span="",
        )
        for field in FactField
        if field not in reported
    ]
    return UserFacts(facts=tuple(_ordered(filled)), unmapped=facts.unmapped)


def _ordered(facts: Iterable[Fact]) -> list[Fact]:
    """Declaration order, so one turn extracts to one byte-stable object."""
    order = {field: index for index, field in enumerate(FactField)}
    return sorted(facts, key=lambda fact: order[fact.field])
