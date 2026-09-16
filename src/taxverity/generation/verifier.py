"""R19 Phase B (ADR-120) — the grounding gate, rewritten for marker-based
citation. Plain Python, no model.

A claim is released only if every mechanical check passes:

- **Citation existence.** Every `[n]` marker in a `content` claim's text must
  name a position in the evidence pack (`pack.units[n-1]`). Because a pack
  unit already carries its whole subtree (ADR-055), citing unit `n` grounds
  anything inside it — there is no separate path-parsing step anymore, and
  no way to cite a node that was never packed.
- **Numeric provenance.** Every number in a claim's text — digits or English
  number words, Indian scale included ("fifteen lakh rupees" is a real
  figure the Act states that way; see `numbers_in()`) — must appear the same
  way in one of its cited units' own text, its context lead-ins, or its
  citation path. A user's figure must not be able to ground a statement of
  the Act: "the cap is 3 lakh" cannot pass because the question mentioned
  3,00,000. A `computation` claim may also use the calculator's result, the
  user's question and their stated or inferred facts.
- **Modal mismatch (new this phase).** If a cited passage denies something
  ("shall not", "is not allowed") and the claim's own text affirms it
  anyway, the claim is caught even though every number and marker it carries
  checks out. This is deliberately one-directional — a claim being *more*
  cautious than its source is not gated, since that is the safer failure
  mode. It is a targeted guard against the one failure the number/marker
  checks cannot see, not an entailment checker: quote fidelity guaranteed
  the old scheme could never misstate a passage's plain meaning this way; the
  new scheme allows paraphrase, per an explicit user instruction ("the LLM
  can modify it but should not change the original meaning"), so this is the
  one deterministic check that stands in the fidelity check's place.

What it still cannot see: a paraphrase that drifts in some way this modal
check doesn't cover. That is exactly the boundary DeepEval's offline sample
exists to watch (ADR-053) — never the hot path's correctness mechanism.
"""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Iterator
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from enum import StrEnum

from taxverity.calculator.scope import Computation
from taxverity.corpus.loader import normalise
from taxverity.corpus.nodes import NodePath
from taxverity.facts import FactStatus, UserFacts
from taxverity.generation.claims import (
    CALC_MARKER,
    MARKER,
    NO_BASIS_OPENERS,
    Citation,
    Claim,
    ClaimType,
)
from taxverity.retrieval.evidence import EvidencePack, EvidenceUnit

VERIFIER_STAGE_VERSION = 3

_CITATION_PREFIX = re.compile(r"^(?:sections?|sec\.?|s\.|u/s\.?)\s*", re.IGNORECASE)
_SPACE_BEFORE_BRACKET = re.compile(r"\s+\(")
_NUMBER = re.compile(
    r"(?<![\d.])(?P<digits>\d+(?:,\d+)*)(?:\.(?P<fraction>\d+))?"
    r"(?:\s*(?P<multiplier>lakhs?|crores?)\b)?",
    re.IGNORECASE,
)
_MULTIPLIERS = {"lakh": 100_000, "crore": 10_000_000}

# English number words, including the Indian scale. "one" and "zero" are
# deliberately excluded as *single-token* triggers below — both are common
# non-numeric English words ("one such condition"), and a false trigger here
# would make an ordinary sentence look like it stated a figure it must then
# ground. They still count inside a longer run ("one lakh", "twenty one").
_ONES = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19,
}  # fmt: skip
_TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
    "seventy": 70, "eighty": 80, "ninety": 90,
}  # fmt: skip
_SCALES = {
    "hundred": 100, "thousand": 1_000, "lakh": 100_000, "lakhs": 100_000,
    "crore": 10_000_000, "crores": 10_000_000, "million": 1_000_000,
}  # fmt: skip
_NUMBER_WORD_TOKEN = frozenset({*_ONES, *_TENS, *_SCALES, "and", "zero"})
_WORD = re.compile(r"[a-zA-Z]+")

# Shared with conversational.py's `_looks_statutory`: a word that only
# belongs in a statement about what the Act provides. One list, one place —
# a divergent copy is how a guard silently stops covering what it claims to.
STATUTORY_VOCAB = re.compile(
    r"\b(section|schedule|deduct\w*|exempt\w*|taxable|rebate|slab|allow\w*|"
    r"regime|shall|entitled|liable|provision|TDS|surcharge|cess)\b",
    re.IGNORECASE,
)

_NEGATIVE_MODAL = re.compile(
    r"\b(shall not|cannot|can not|is not allowed|are not allowed|"
    r"not entitled|not permitted|no deduction|not deductible)\b",
    re.IGNORECASE,
)
_AFFIRMATIVE_MODAL = re.compile(
    r"\b(you (?:can|may|are entitled|should|must)|is allowed|is deductible|"
    r"is permitted|shall be)\b",
    re.IGNORECASE,
)


class Violation(StrEnum):
    MALFORMED_CLAIM = "malformed_claim"
    MALFORMED_HEADING = "malformed_heading"
    MALFORMED_NO_BASIS = "malformed_no_basis"
    NO_CITATION = "no_citation"
    MARKER_NOT_IN_EVIDENCE = "marker_not_in_evidence"
    NO_COMPUTATION = "no_computation"
    UNSUPPORTED_NUMBER = "unsupported_number"
    MODAL_MISMATCH = "modal_mismatch"


@dataclass(frozen=True)
class Finding:
    violation: Violation
    detail: str


@dataclass(frozen=True)
class Verdict:
    # The claim with its citations resolved from `[n]` markers, so what is
    # released carries the actual path/quote a viewer can inspect.
    claim: Claim
    findings: tuple[Finding, ...]

    @property
    def passed(self) -> bool:
        return not self.findings


class Verifier:
    def __init__(
        self,
        pack: EvidencePack,
        *,
        question: str = "",
        facts: UserFacts | None = None,
        computation: Computation | None = None,
    ) -> None:
        self._units: dict[int, EvidenceUnit] = dict(enumerate(pack.units, start=1))
        self._computation = computation
        self._user_numbers = numbers_in(question) | _fact_numbers(facts)
        self._computation_numbers = (
            _computation_numbers(computation) if computation is not None else frozenset()
        )

    def verify(self, claim: Claim) -> Verdict:
        if claim.type is ClaimType.HEADING:
            return self._verify_heading(claim)
        if claim.type is ClaimType.NO_BASIS:
            return self._verify_no_basis(claim)
        if claim.type is ClaimType.COMPUTATION:
            return self._verify_computation(claim)
        return self._verify_content(claim)

    def _verify_heading(self, claim: Claim) -> Verdict:
        findings = []
        if numbers_in(claim.text) or MARKER.search(claim.text):
            findings.append(
                Finding(Violation.MALFORMED_HEADING, "a heading carries a figure or citation")
            )
        return Verdict(claim=claim, findings=tuple(findings))

    def _verify_no_basis(self, claim: Claim) -> Verdict:
        findings = []
        if MARKER.search(claim.text):
            findings.append(
                Finding(Violation.MALFORMED_NO_BASIS, "a no_basis claim cites evidence")
            )
        if numbers_in(claim.text):
            findings.append(
                Finding(Violation.MALFORMED_NO_BASIS, "a no_basis claim states a number")
            )
        if not claim.text.startswith(NO_BASIS_OPENERS):
            findings.append(
                Finding(
                    Violation.MALFORMED_NO_BASIS,
                    "a no_basis claim must open by naming the Act's silence",
                )
            )
        return Verdict(claim=claim, findings=tuple(findings))

    def _verify_computation(self, claim: Claim) -> Verdict:
        findings = []
        if self._computation is None:
            findings.append(Finding(Violation.NO_COMPUTATION, "no computation was provided"))
        allowed = self._computation_numbers | self._user_numbers
        text = claim.text.replace(CALC_MARKER, "")
        unsupported = sorted(numbers_in(text) - allowed)
        if unsupported:
            findings.append(_unsupported_finding(unsupported))
        return Verdict(claim=claim, findings=tuple(findings))

    def _verify_content(self, claim: Claim) -> Verdict:
        findings: list[Finding] = []
        citations: list[Citation] = []
        allowed: set[Decimal] = set()
        markers = [int(m) for m in MARKER.findall(claim.text)]

        for marker in markers:
            unit = self._units.get(marker)
            if unit is None:
                findings.append(
                    Finding(
                        Violation.MARKER_NOT_IN_EVIDENCE,
                        f"[{marker}] does not name any passage shown",
                    )
                )
                continue
            citations.append(Citation(marker=marker, path=unit.citation, quote=_excerpt(unit)))
            allowed |= _ground_numbers(unit)

        # Every content line needs a citation, no exceptions — deliberately
        # not "unless it looks connective": a blocklist of trigger words a
        # statement must contain to require grounding can never be
        # exhaustive, and the failure mode of getting it wrong (an
        # ungrounded assertion served as fact) is exactly the one thing
        # rule 03's core invariant exists to prevent. A genuinely connective
        # line belongs in the heading, not as an uncited bullet.
        if not markers:
            findings.append(Finding(Violation.NO_CITATION, "cites nothing"))

        # `[n]` marker brackets are citation syntax, not a stated figure — the
        # digit inside one must never be read as a number the sentence itself
        # asserts (and then have to "find a source" for).
        unsupported = sorted(numbers_in(MARKER.sub("", claim.text)) - allowed)
        if unsupported:
            findings.append(_unsupported_finding(unsupported))

        cited_units = [self._units[m] for m in markers if m in self._units]
        if cited_units and _asserts_the_opposite_of_its_source(claim.text, cited_units):
            findings.append(
                Finding(
                    Violation.MODAL_MISMATCH,
                    "the cited passage says this is not allowed, but the claim asserts it is",
                )
            )

        return Verdict(
            claim=claim.model_copy(update={"citations": tuple(citations)}),
            findings=tuple(findings),
        )


def _unsupported_finding(unsupported: list[Decimal]) -> Finding:
    return Finding(
        Violation.UNSUPPORTED_NUMBER,
        "no source for " + ", ".join(str(number) for number in unsupported),
    )


def _excerpt(unit: EvidenceUnit, limit: int = 600) -> str:
    text = unit.chunk.text
    return text if len(text) <= limit else text[:limit].rstrip() + "…"


def _ground_numbers(unit: EvidenceUnit) -> frozenset[Decimal]:
    numbers = numbers_in(unit.chunk.text) | numbers_in(unit.citation)
    for line in unit.context:
        numbers |= numbers_in(line.text)
    return numbers


def _asserts_the_opposite_of_its_source(text: str, units: list[EvidenceUnit]) -> bool:
    """One-directional: a claim affirming what a cited passage denies. See
    this module's docstring for why the reverse (an overly cautious claim)
    is not gated."""
    if not (_AFFIRMATIVE_MODAL.search(text) and not _NEGATIVE_MODAL.search(text)):
        return False
    for unit in units:
        source = unit.chunk.text
        if _NEGATIVE_MODAL.search(source) and not _AFFIRMATIVE_MODAL.search(source):
            return True
    return False


def canonical_path(raw: str) -> str | None:
    """`Section 22 (2)` and `s. 22(2)` both name `22(2)`; anything else is
    None. Used independently by `conversational.py`'s own guard, not by
    verification here — a marker resolves by position, never by parsing a
    path out of a claim's text."""
    text = _SPACE_BEFORE_BRACKET.sub("(", _CITATION_PREFIX.sub("", raw.strip()))
    try:
        return NodePath.parse(text).render()
    except ValueError:
        return None


def numbers_in(text: str) -> frozenset[Decimal]:
    """Every figure a text states, in digits or in English words: `12,00,000`,
    `12 lakh` and `twelve lakh` all agree. The Act states some amounts only in
    words ("fifteen lakh rupees"), so a digit-only reading of a claim would
    never find that grounding — this is what closes that gap."""
    numbers = set()
    normalised = normalise(text)
    for match in _NUMBER.finditer(normalised):
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
    numbers |= _word_numbers_in(normalised)
    return frozenset(numbers)


def _word_numbers_in(text: str) -> frozenset[Decimal]:
    tokens = [t.lower() for t in _WORD.findall(text)]
    values: set[Decimal] = set()
    i, n = 0, len(tokens)
    while i < n:
        if tokens[i] not in _NUMBER_WORD_TOKEN:
            i += 1
            continue
        j = i
        total = Decimal(0)
        current = Decimal(0)
        length = 0
        while j < n and tokens[j] in _NUMBER_WORD_TOKEN:
            tok = tokens[j]
            if tok == "and":
                j += 1
                continue
            length += 1
            if tok in _ONES:
                current += _ONES[tok]
            elif tok in _TENS:
                current += _TENS[tok]
            elif tok == "zero":
                pass
            elif tok in _SCALES:
                # A scale word with nothing before it in this run ("lakh" on
                # its own, e.g. the leftover from "2 lakh" — the digit "2" is
                # not a *word* token, so the word-scanner never saw it) states
                # no figure by itself; only commit a contribution once a
                # preceding word-number actually set `current`.
                if current > 0:
                    scale = _SCALES[tok]
                    if scale == 100:
                        current *= scale
                    else:
                        total += current * scale
                        current = Decimal(0)
            j += 1
        total += current
        # A bare "one" or "zero" is almost always the ordinary English word,
        # not a figure ("one such condition") — only trust it as a number
        # once it combines with something else (a scale word, another digit
        # word).
        if length > 1 or (length == 1 and tokens[i] not in ("one", "zero")):
            if total > 0:
                values.add(total.normalize())
        i = j
    return frozenset(values)


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
