"""Step 10.5 — the grounding gate (ADR-016). Plain Python, no model.

A claim is released only if every mechanical check passes:

- **Citation existence.** The path names a packed unit, one of its context
  lines, or a node inside a packed unit that exists as a chunk. A unit carries
  its whole subtree (ADR-055), so `22(2)` is in evidence when unit `22` is. A
  context line is only an ancestor's lead-in, so a quote from it is checked
  against the lead-in, not the ancestor's whole text.
- **Quote fidelity.** The quote is a verbatim substring of the cited node's
  text, compared after `normalise()` and whitespace collapsing only.
- **Numeric provenance.** Every number in a statute claim's text appears in one
  of its own quotes or citation paths. A user's figure must not be able to
  ground a statement of law: "the cap is 3 lakh" cannot pass because the
  question mentioned 3,00,000. A computation claim may also use the calculator's
  result, the user's question and their stated or inferred facts.

What it cannot see: a claim that quotes real text and misreads it. That is the
omission critic's territory (deferred, ADR-110), not this gate's.
"""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from enum import StrEnum

from taxverity.calculator.scope import Computation
from taxverity.chunking.models import Chunk
from taxverity.corpus.loader import normalise
from taxverity.corpus.nodes import NodePath
from taxverity.facts import FactStatus, UserFacts
from taxverity.generation.claims import Citation, Claim, ClaimType
from taxverity.retrieval.evidence import EvidencePack

VERIFIER_STAGE_VERSION = 2

# A quote of one or two words ("the", "income") is in almost every provision,
# so it proves nothing about which one the claim rests on.
MIN_QUOTE_WORDS = 3

_CITATION_PREFIX = re.compile(r"^(?:sections?|sec\.?|s\.|u/s\.?)\s*", re.IGNORECASE)
_SPACE_BEFORE_BRACKET = re.compile(r"\s+\(")
_WHITESPACE = re.compile(r"\s+")
_NUMBER = re.compile(
    r"(?<![\d.])(?P<digits>\d+(?:,\d+)*)(?:\.(?P<fraction>\d+))?"
    r"(?:\s*(?P<multiplier>lakhs?|crores?)\b)?",
    re.IGNORECASE,
)
_MULTIPLIERS = {"lakh": 100_000, "crore": 10_000_000}

# NO_BASIS claims (Step 1, advisor pivot) carry no citation, so they must be
# recognisable as "the Act is silent" from their own opening words alone —
# otherwise the one uncited claim type becomes a free-text channel.
NO_BASIS_OPENERS = ("The Act does not", "The Act is silent on", "Nothing in the Act")

# ADVICE claims assert a prescription; a quote must show the Act actually
# imposing/permitting one, not merely defining a term the claim leans on.
_PRESCRIPTIVE = re.compile(
    r"\b(you (?:should|must|can|may|are entitled)|i recommend|it is advisable)\b",
    re.IGNORECASE,
)
_STATUTORY_MODAL = re.compile(
    r"\b(shall not|shall|may|is not|no deduction|entitled|allowed|required|"
    r"liable|exempt)\b",
    re.IGNORECASE,
)


class Violation(StrEnum):
    MALFORMED_CLAIM = "malformed_claim"
    NO_CITATION = "no_citation"
    CITATION_NOT_IN_EVIDENCE = "citation_not_in_evidence"
    QUOTE_TOO_SHORT = "quote_too_short"
    QUOTE_NOT_IN_SOURCE = "quote_not_in_source"
    NO_COMPUTATION = "no_computation"
    UNSUPPORTED_NUMBER = "unsupported_number"
    MALFORMED_NO_BASIS = "malformed_no_basis"
    UNSUPPORTED_ADVICE = "unsupported_advice"


@dataclass(frozen=True)
class Finding:
    violation: Violation
    detail: str


@dataclass(frozen=True)
class Verdict:
    # The claim with its citation paths in canonical form, so what is released
    # names the node exactly as the chunk store does.
    claim: Claim
    findings: tuple[Finding, ...]

    @property
    def passed(self) -> bool:
        return not self.findings


class Verifier:
    def __init__(
        self,
        pack: EvidencePack,
        chunks: Mapping[str, Chunk],
        *,
        question: str = "",
        facts: UserFacts | None = None,
        computation: Computation | None = None,
    ) -> None:
        """`chunks` maps node path to chunk: the whole corpus, not just the pack."""
        self._units = {unit.citation: unit.chunk.text for unit in pack.units}
        self._context = {
            line.citation: line.text for unit in pack.units for line in unit.context
        }
        self._chunks = chunks
        self._computation = computation
        self._user_numbers = numbers_in(question) | _fact_numbers(facts)
        self._computation_numbers = (
            _computation_numbers(computation) if computation is not None else frozenset()
        )

    def verify(self, claim: Claim) -> Verdict:
        findings: list[Finding] = []
        citations: list[Citation] = []
        allowed: set[Decimal] = set()

        if claim.type in (ClaimType.STATUTE, ClaimType.ADVICE) and not claim.citations:
            findings.append(
                Finding(Violation.NO_CITATION, f"a {claim.type.value} claim cites nothing")
            )
        if claim.type is ClaimType.COMPUTATION:
            if self._computation is None:
                findings.append(
                    Finding(Violation.NO_COMPUTATION, "no computation was provided")
                )
            allowed |= self._computation_numbers | self._user_numbers
        if claim.type is ClaimType.NO_BASIS:
            if claim.citations:
                findings.append(
                    Finding(Violation.MALFORMED_NO_BASIS, "a no_basis claim cites evidence")
                )
            if not claim.text.startswith(NO_BASIS_OPENERS):
                findings.append(
                    Finding(
                        Violation.MALFORMED_NO_BASIS,
                        "a no_basis claim must open by naming the Act's silence",
                    )
                )

        for citation in claim.citations:
            path = canonical_path(citation.path)
            citations.append(Citation(path=path or citation.path, quote=citation.quote))
            source = self._source(path) if path is not None else None
            if source is None:
                findings.append(
                    Finding(
                        Violation.CITATION_NOT_IN_EVIDENCE,
                        f"{citation.path!r} is not in the evidence",
                    )
                )
                continue
            allowed |= numbers_in(path)
            if len(citation.quote.split()) < MIN_QUOTE_WORDS:
                findings.append(
                    Finding(Violation.QUOTE_TOO_SHORT, f"the quote for {path} is too short")
                )
                continue
            if _squash(citation.quote) not in _squash(source):
                findings.append(
                    Finding(
                        Violation.QUOTE_NOT_IN_SOURCE,
                        f"the quote for {path} is not in its text",
                    )
                )
                continue
            allowed |= numbers_in(citation.quote)

        if claim.type is ClaimType.ADVICE and _PRESCRIPTIVE.search(claim.text):
            if not any(_STATUTORY_MODAL.search(c.quote) for c in claim.citations):
                findings.append(
                    Finding(
                        Violation.UNSUPPORTED_ADVICE,
                        "no cited quote supports the prescription",
                    )
                )

        unsupported = sorted(numbers_in(claim.text) - allowed)
        if unsupported:
            findings.append(
                Finding(
                    Violation.UNSUPPORTED_NUMBER,
                    "no source for " + ", ".join(str(number) for number in unsupported),
                )
            )
        return Verdict(
            claim=claim.model_copy(update={"citations": tuple(citations)}),
            findings=tuple(findings),
        )

    def _source(self, path: str) -> str | None:
        if path in self._units:
            return self._units[path]
        chunk = self._chunks.get(path)
        if chunk is not None:
            ancestor = NodePath.parse(path).parent
            while ancestor is not None:
                if ancestor.render() in self._units:
                    return chunk.text
                ancestor = ancestor.parent
        return self._context.get(path)


def canonical_path(raw: str) -> str | None:
    """`Section 22 (2)` and `s. 22(2)` both name `22(2)`; anything else is None."""
    text = _SPACE_BEFORE_BRACKET.sub("(", _CITATION_PREFIX.sub("", raw.strip()))
    try:
        return NodePath.parse(text).render()
    except ValueError:
        return None


def numbers_in(text: str) -> frozenset[Decimal]:
    """Every figure a text states, as a value: `12,00,000` and `12 lakh` agree."""
    numbers = set()
    for match in _NUMBER.finditer(normalise(text)):
        digits = match.group("digits").replace(",", "")
        fraction = match.group("fraction")
        try:
            value = Decimal(f"{digits}.{fraction}" if fraction else digits)
        except InvalidOperation:
            continue
        multiplier = match.group("multiplier")
        if multiplier:
            value *= _MULTIPLIERS[multiplier.lower().rstrip("s")]
        numbers.add(value.normalize())
    return frozenset(numbers)


def _squash(text: str) -> str:
    return _WHITESPACE.sub(" ", normalise(text)).strip()


def _fact_numbers(facts: UserFacts | None) -> frozenset[Decimal]:
    if facts is None:
        return frozenset()
    numbers: set[Decimal] = set()
    for fact in facts.facts:
        # A profile default is unconfirmed (rule 04), so it grounds nothing.
        if fact.status not in (FactStatus.STATED, FactStatus.INFERRED):
            continue
        if isinstance(fact.value, bool):
            continue
        if isinstance(fact.value, (Decimal, int)):
            numbers.add(Decimal(fact.value).normalize())
            numbers.add(abs(Decimal(fact.value)).normalize())
        elif isinstance(fact.value, str):
            numbers |= numbers_in(fact.value)
    return frozenset(numbers)


def _computation_numbers(computation: Computation) -> frozenset[Decimal]:
    """Every figure anywhere in the result, its provenance quotes included."""
    numbers: set[Decimal] = set()
    for value in _walk(computation):
        if isinstance(value, bool):
            continue
        if isinstance(value, (Decimal, int)):
            numbers.add(Decimal(value).normalize())
        elif isinstance(value, str):
            numbers |= numbers_in(value)
    return frozenset(numbers)


def _walk(value: object) -> Iterator[object]:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        for field in dataclasses.fields(value):
            yield from _walk(getattr(value, field.name))
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _walk(item)
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from _walk(key)
            yield from _walk(item)
    else:
        yield value
