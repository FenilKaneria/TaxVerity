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
from typing import TYPE_CHECKING

from taxverity.calculator.scope import Computation
from taxverity.corpus.loader import normalise
from taxverity.corpus.nodes import NodePath
from taxverity.facts import FactStatus, UserFacts
from taxverity.generation.claims import (
    CALC_MARKER,
    EXAMPLE_MARKER,
    EXAMPLE_OPENERS,
    FACT_MARKER,
    GUIDE_MARKER,
    MARKER,
    NO_BASIS_OPENERS,
    UNKNOWN_OPENERS,
    Citation,
    Claim,
    ClaimType,
    line_body,
    strip_list_number,
    strip_non_citation_markers,
)
from taxverity.reasoning.models import CheckStatus
from taxverity.retrieval.evidence import EvidencePack, EvidenceUnit

# `reasoning/validate.py` imports this module (`ground_numbers`), so importing
# `ValidatedAnalysis` here at runtime would be circular — it is used only for
# the type hint below, which `from __future__ import annotations` (this
# module's first import) already makes a deferred string.
if TYPE_CHECKING:
    from taxverity.reasoning.validate import ValidatedAnalysis

VERIFIER_STAGE_VERSION = 9

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

# R21 (ADR-127) — the EXAMPLE line's "can't invent law" guard. A figure in a
# legal position (right after a limit/rate word or a provision name, or any
# percentage) states the law, never a hypothetical, so it must ground in the
# cited passage however the line frames it.
_LEGAL_KEYWORD = re.compile(
    r"\b(?:up\s*to|upto|limit\w*|maximum|max\.|cap(?:ped)?|at\s+most|not\s+exceeding|"
    r"exceed\w*|threshold|rates?|slab|sections?|sub-sections?|clauses?|schedule|rules?|"
    r"paragraph)\b",
    re.IGNORECASE,
)
# A clause ends at `.`/`;`/`,` followed by space or end — never at the commas
# inside "2,00,000".
_CLAUSE_END = re.compile(r"[.;,](?=\s|$)")
_PERCENT = re.compile(
    r"(?P<digits>\d+(?:\.\d+)?)\s*(?:%|per\s*cent\b|percent\b)"
    r"|(?P<words>(?:[a-z]+[\s-]){1,3})per\s*cent\b",
    re.IGNORECASE,
)
# The premise — "Suppose your loss is ₹3,00,000 and your salary ₹12,00,000" —
# runs from the opener to the first sentence break, "then", or a comma that
# hands over to the consequence ("…, you can…"). Figures stated there are the
# example's own inputs. A premise that itself says what is *allowed* ("Suppose
# you can set off ₹5,00,000") is stating law, so it grounds nothing.
_OPENER_PREFIX = re.compile(
    r"^(?:for example|for instance|e\.g\.|say|suppose|imagine)[,:]?\s*(?:(?:suppose|say|imagine|if)\b,?\s*)?",
    re.IGNORECASE,
)
_PREMISE_END = re.compile(
    r"[.;:—](?=\s|$)|\bthen\b|,\s*(?=(?:you|so|this|that|which|it|only|the)\b)",
    re.IGNORECASE,
)
_PERMISSION = re.compile(
    r"\b(?:can|could|may|allowed|entitled|permitted|eligible)\b", re.IGNORECASE
)
_OPERAND = (
    r"(?:₹\s*|Rs\.?\s*|INR\s*)?\d[\d,]*(?:\.\d+)?(?:\s*(?:lakhs?|crores?))?(?:\s*%)?"
)
_OPERATOR = r"\s*(?:[+\-−–‑×*/÷]|\sx\s)\s*"
_TERM = rf"\(*\s*{_OPERAND}\)*"
_EXPRESSION = rf"{_TERM}(?:{_OPERATOR}{_TERM})*"
# A worked equation, possibly chained and parenthesised: "(a × 5%) + (b × 10%)
# = ₹20,000 + ₹20,000 = ₹40,000". The first side must do some arithmetic.
_EQUATION = re.compile(rf"{_TERM}(?:{_OPERATOR}{_TERM})+(?:\s*=\s*{_EXPRESSION})+")
_TOKEN = re.compile(
    rf"(?P<operand>{_OPERAND})|(?P<paren>[()])|(?P<op>[+\-−–‑×*/÷]|(?<=\s)x(?=\s))"
)
_EXAMPLE_BULLET = re.compile(r"^[-*•\s]+")
_NIL = re.compile(r"\bnil\b", re.IGNORECASE)
# R22 Part B: words naming an online system. The Act sets who files and by
# when, not how a website works, so a cited line using one of these words is
# withheld unless a passage it cites uses the same word. A no_basis line
# ("The Act does not describe the portal") cites nothing and is not checked.
_ONLINE_PROCEDURE = re.compile(
    r"\b(?:portal|website|web site|log ?in|sign in|click|app|online|"
    r"e[-‐‑–]?filing|upload|download|otp|net ?banking)\b",
    re.IGNORECASE,
)

# R22 Part C (ADR-128): what a GUIDANCE line may never contain. Guidance is
# served without a citation, so it may describe process (where to file, what
# to keep, how to e-verify) but never state law: no figure, date, provision,
# tax treatment or obligation. A blocklist can only withhold, never serve, so
# a word it over-catches costs one line, never correctness.
GUIDANCE_MAX_WORDS = 40
# Form and statement names carry digits but state no figure. Stripped before
# the figure check; any other digit withholds the line.
_FORM_NAME = re.compile(
    r"\bITR[-\s]?(?:[1-7]|U|V)\b|\bForm\s*(?:No\.?\s*)?(?:16A|16|26AS)\b|"
    r"\b26AS\b|\bAIS\b|\bTIS\b",
    re.IGNORECASE,
)
_GUIDANCE_LAW = re.compile(
    r"\b(?:deduct\w*|exempt\w*|taxable|tax[-\s]free|rebate\w*|slabs?|entitle\w*|"
    r"liab\w*|allow\w*|penalt\w*|rates?|limits?|regimes?|interest|fees?|fines?|"
    r"due\s+date|deadlines?|last\s+date|"
    r"january|february|march|april|june|july|august|september|october|"
    r"november|december|sections?|sub-?sections?|schedules?|clauses?|chapters?|"
    r"provisos?|provisions?|rules?|laws?|legal\w*|statut\w*|mandatory|"
    r"compulsor\w*|obligat\w*|required\s+by|eligib\w*|qualif\w*)\b",
    re.IGNORECASE,
)
# The statute, matched case-sensitively so "act on it" still passes. ("May"
# is left out of the month names above for the same reason; a real date
# carries a figure the figure check already catches.)
_ACT = re.compile(r"\bAct\b")
_GUIDANCE_EVASION = re.compile(
    r"\b(?:conceal\w*|hid(?:e|es|ing|den)|backdat\w*|fake\w*|fabricat\w*|forg(?:e|ed|ing|ery)|"
    r"benami|hawala|undisclosed|unreported|evad\w*|evasion|cash|"
    r"(?:not|never|avoid|without)\s+(?:report\w*|disclos\w*|declar\w*|show\w*))\b",
    re.IGNORECASE,
)
_URL = re.compile(
    r"https?://|\bwww\.|\b[\w-]+\.(?:gov|com|in|org|net)\b", re.IGNORECASE
)


class Violation(StrEnum):
    MALFORMED_CLAIM = "malformed_claim"
    MALFORMED_HEADING = "malformed_heading"
    MALFORMED_NO_BASIS = "malformed_no_basis"
    MALFORMED_UNKNOWN = "malformed_unknown"
    NO_CITATION = "no_citation"
    MARKER_NOT_IN_EVIDENCE = "marker_not_in_evidence"
    NO_COMPUTATION = "no_computation"
    UNSUPPORTED_NUMBER = "unsupported_number"
    MODAL_MISMATCH = "modal_mismatch"
    # R20 Step 20.7: an APPLICATION line affirms a positive conclusion
    # ("you can claim X") while the analysis has the condition it cites as
    # `unknown`, `not_satisfied` or `ambiguous` — not a text-vs-text
    # mismatch like MODAL_MISMATCH, a claim-vs-analysis one.
    UNSUPPORTED_APPLICATION = "unsupported_application"
    # R21 (ADR-127): EXAMPLE-line checks.
    MALFORMED_EXAMPLE = "malformed_example"
    # A rate, limit, threshold or provision number no cited passage states —
    # the one thing an illustration may never do.
    INVENTED_LAW = "invented_law"
    BAD_ARITHMETIC = "bad_arithmetic"
    # R22 Part B: a step on a website, portal or app ("log in to the
    # e-filing portal [1]") attributed to a passage that never mentions one.
    OFF_ACT_PROCEDURE = "off_act_procedure"
    # R22 Part C (ADR-128): GUIDANCE-line checks.
    MALFORMED_GUIDANCE = "malformed_guidance"
    GUIDANCE_STATES_LAW = "guidance_states_law"
    GUIDANCE_UNSAFE = "guidance_unsafe"
    GUIDANCE_TOO_LONG = "guidance_too_long"


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
        analysis: ValidatedAnalysis | None = None,
    ) -> None:
        self._units: dict[int, EvidenceUnit] = dict(enumerate(pack.units, start=1))
        self.pack_size = len(pack.units)
        self.has_computation = computation is not None
        # R21 Part B: section number -> pack positions, for the generator's
        # deterministic "[408] means the passage of section 408" rewrite.
        sections: dict[str, list[int]] = {}
        for number, unit in self._units.items():
            if unit.chunk.section_number is not None:
                sections.setdefault(unit.chunk.section_number, []).append(number)
        self.section_markers = {k: tuple(v) for k, v in sections.items()}
        self._computation = computation
        self._user_numbers = numbers_in(question) | _fact_numbers(facts)
        self._computation_numbers = (
            _computation_numbers(computation)
            if computation is not None
            else frozenset()
        )
        # R20 Step 20.7: which condition each pack marker speaks to, and that
        # condition's checked status — an APPLICATION or UNKNOWN claim is
        # gated against these, never against the model's own prose.
        self._condition_status: dict[str, CheckStatus] = {}
        self._condition_ids_by_marker: dict[int, set[str]] = {}
        if analysis is not None:
            for rule in analysis.legal_rules:
                for condition in rule.conditions:
                    for marker in condition.markers or rule.markers:
                        self._condition_ids_by_marker.setdefault(marker, set()).add(
                            condition.id
                        )
            for check in analysis.applicability:
                self._condition_status[check.condition_id] = check.status

    def verify(self, claim: Claim) -> Verdict:
        """R22 Part B: every check reads the line without its step number;
        the released claim keeps the line exactly as written."""
        body = strip_list_number(claim.text)
        if body == claim.text.strip():
            return self._verify(claim)
        verdict = self._verify(claim.model_copy(update={"text": body}))
        return dataclasses.replace(
            verdict, claim=verdict.claim.model_copy(update={"text": claim.text})
        )

    def _verify(self, claim: Claim) -> Verdict:
        if claim.type is ClaimType.HEADING:
            return self._verify_heading(claim)
        if claim.type is ClaimType.NO_BASIS:
            return self._verify_no_basis(claim)
        if claim.type is ClaimType.UNKNOWN:
            return self._verify_unknown(claim)
        if claim.type is ClaimType.COMPUTATION:
            return self._verify_computation(claim)
        if claim.type is ClaimType.APPLICATION:
            return self._verify_application(claim)
        if claim.type is ClaimType.EXAMPLE:
            return self._verify_example(claim)
        if claim.type is ClaimType.GUIDANCE:
            return self._verify_guidance(claim)
        return self._verify_content(claim)

    def _verify_guidance(self, claim: Claim) -> Verdict:
        """R22 Part C (ADR-128): the one served line with no source, so it
        may state nothing a source would be needed for. Every check reads the
        whole line; none consults the pack, the facts or the computation, so
        nothing the person wrote can make a guidance line pass."""
        findings: list[Finding] = []
        text = claim.text.replace(GUIDE_MARKER, " ")
        if MARKER.search(text) or any(
            marker in text for marker in (CALC_MARKER, FACT_MARKER, EXAMPLE_MARKER)
        ):
            findings.append(
                Finding(
                    Violation.MALFORMED_GUIDANCE,
                    "a guidance line carries a citation or another line's marker",
                )
            )
        if numbers_in(_FORM_NAME.sub(" ", MARKER.sub(" ", text))):
            findings.append(
                Finding(
                    Violation.GUIDANCE_STATES_LAW, "a guidance line states a figure"
                )
            )
        law = _GUIDANCE_LAW.search(text) or _ACT.search(text)
        if law is not None:
            findings.append(
                Finding(
                    Violation.GUIDANCE_STATES_LAW,
                    f'a guidance line states law ("{law.group(0)}")',
                )
            )
        if _URL.search(text) or _GUIDANCE_EVASION.search(text):
            findings.append(
                Finding(
                    Violation.GUIDANCE_UNSAFE,
                    "a guidance line carries a link or describes hiding something",
                )
            )
        if len(_WORD.findall(line_body(text))) > GUIDANCE_MAX_WORDS:
            findings.append(
                Finding(
                    Violation.GUIDANCE_TOO_LONG,
                    f"a guidance line runs past {GUIDANCE_MAX_WORDS} words",
                )
            )
        return Verdict(claim=claim, findings=tuple(findings))

    def _verify_example(self, claim: Claim) -> Verdict:
        """R21 (ADR-127): the model may illustrate, never invent law.

        - It must open with a fixed hypothetical phrase and cite the rule it
          illustrates.
        - Any figure in a legal position (after a limit/rate word or a
          provision name, or a percentage) must ground in a cited passage.
        - Figures in the premise ("Suppose your loss is ₹3,00,000") are the
          example's own inputs and need no source.
        - Any other figure is either grounded in a cited passage or the
          result of an equation the line shows and that adds up.
        """
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
            citations.append(
                Citation(marker=marker, path=unit.citation, quote=_excerpt(unit))
            )
            allowed |= ground_numbers(unit)
        if not markers:
            findings.append(Finding(Violation.NO_CITATION, "cites nothing"))

        body = normalise(
            _EXAMPLE_BULLET.sub(
                "", MARKER.sub("", strip_non_citation_markers(claim.text))
            )
        ).strip()
        if not body.startswith(EXAMPLE_OPENERS):
            findings.append(
                Finding(
                    Violation.MALFORMED_EXAMPLE,
                    'an example must open with "For example" or "Suppose"',
                )
            )

        invented = sorted(_legal_numbers(body) - allowed)
        if invented:
            findings.append(
                Finding(
                    Violation.INVENTED_LAW,
                    "states a rate, limit or provision no cited passage gives: "
                    + ", ".join(str(n) for n in invented),
                )
            )

        opener = _OPENER_PREFIX.match(body)
        premise_start = opener.end() if opener else 0
        end = _PREMISE_END.search(body, premise_start)
        premise_end = end.start() if end else len(body)
        results: set[Decimal] = set()
        equations = list(_EQUATION.finditer(body))
        premise = _mask(body, equations)[:premise_end]
        if _PERMISSION.search(premise):
            # The "premise" states law, so its figures are checked like any
            # other rather than accepted as the example's own inputs.
            hypothetical: frozenset[Decimal] = frozenset()
            premise_end = 0
        else:
            hypothetical = numbers_in(premise) - _legal_numbers(body)
        for equation in equations:
            sides = equation.group(0).split("=")
            values = [_evaluate(side) for side in sides]
            if any(v is None for v in values) or max(values) - min(values) > 1:  # type: ignore[type-var]
                findings.append(
                    Finding(
                        Violation.BAD_ARITHMETIC,
                        f'"{equation.group(0)}" does not add up',
                    )
                )
                continue
            # Only the first side works from sourced figures; every later side
            # restates it (each is checked equal to it above), so its
            # sub-totals are the line's own results, not new claims.
            known_now = allowed | hypothetical | results
            unsourced = sorted(
                n
                for n in numbers_in(sides[0]) - known_now
                if not _is_sum_or_difference(n, known_now)
            )
            if unsourced:
                findings.append(_unsupported_finding(unsourced))
            for side, value in zip(sides, values, strict=True):
                results |= numbers_in(side)
                results.add(value.normalize())  # type: ignore[union-attr]
        masked = _mask(body, equations)

        known = allowed | hypothetical | results
        derived = sorted(
            n
            for n in numbers_in(masked[premise_end:]) - known
            if not _is_sum_or_difference(n, known)
        )
        if derived:
            findings.append(
                Finding(
                    Violation.UNSUPPORTED_NUMBER,
                    "no source or shown working for "
                    + ", ".join(str(n) for n in derived),
                )
            )

        cited_units = [self._units[m] for m in markers if m in self._units]
        if _names_an_unsourced_online_step(claim.text, cited_units):
            findings.append(
                Finding(
                    Violation.OFF_ACT_PROCEDURE,
                    "describes a website or portal step no cited passage mentions",
                )
            )
        if cited_units and _asserts_the_opposite_of_its_source(claim.text, cited_units):
            findings.append(
                Finding(
                    Violation.MODAL_MISMATCH,
                    "the cited passage says this is not allowed, but the example asserts it is",
                )
            )

        return Verdict(
            claim=claim.model_copy(update={"citations": tuple(citations)}),
            findings=tuple(findings),
        )

    def _verify_heading(self, claim: Claim) -> Verdict:
        findings = []
        if numbers_in(claim.text) or MARKER.search(claim.text):
            findings.append(
                Finding(
                    Violation.MALFORMED_HEADING,
                    "a heading carries a figure or citation",
                )
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
                Finding(
                    Violation.MALFORMED_NO_BASIS, "a no_basis claim states a number"
                )
            )
        if not line_body(claim.text).startswith(NO_BASIS_OPENERS):
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
            findings.append(
                Finding(Violation.NO_COMPUTATION, "no computation was provided")
            )
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
            citations.append(
                Citation(marker=marker, path=unit.citation, quote=_excerpt(unit))
            )
            allowed |= ground_numbers(unit)

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
        if _names_an_unsourced_online_step(claim.text, cited_units):
            findings.append(
                Finding(
                    Violation.OFF_ACT_PROCEDURE,
                    "describes a website or portal step no cited passage mentions",
                )
            )
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

    def _verify_application(self, claim: Claim) -> Verdict:
        findings: list[Finding] = []
        citations: list[Citation] = []
        allowed: set[Decimal] = set(self._user_numbers)
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
            citations.append(
                Citation(marker=marker, path=unit.citation, quote=_excerpt(unit))
            )
            allowed |= ground_numbers(unit)

        if not markers:
            findings.append(Finding(Violation.NO_CITATION, "cites nothing"))

        text = MARKER.sub("", claim.text.replace(FACT_MARKER, ""))
        unsupported = sorted(numbers_in(text) - allowed)
        if unsupported:
            findings.append(_unsupported_finding(unsupported))

        cited_units = [self._units[m] for m in markers if m in self._units]
        # A person's own figures may ground an application line, but a
        # provision number never comes from the person: "section 999" typed
        # into the question must not ground through `_user_numbers`.
        from_law: set[Decimal] = set()
        for unit in cited_units:
            from_law |= ground_numbers(unit)
        unsourced_provisions = sorted(_provision_numbers(text) - from_law)
        if unsourced_provisions:
            findings.append(_unsupported_finding(unsourced_provisions))

        if _names_an_unsourced_online_step(claim.text, cited_units):
            findings.append(
                Finding(
                    Violation.OFF_ACT_PROCEDURE,
                    "describes a website or portal step no cited passage mentions",
                )
            )
        if cited_units and _asserts_the_opposite_of_its_source(claim.text, cited_units):
            findings.append(
                Finding(
                    Violation.MODAL_MISMATCH,
                    "the cited passage says this is not allowed, but the claim asserts it is",
                )
            )

        if _affirms(claim.text):
            unresolved = {
                condition_id
                for marker in markers
                for condition_id in self._condition_ids_by_marker.get(marker, ())
                if self._condition_status.get(condition_id)
                in (
                    CheckStatus.UNKNOWN,
                    CheckStatus.NOT_SATISFIED,
                    CheckStatus.AMBIGUOUS,
                )
            }
            if unresolved:
                findings.append(
                    Finding(
                        Violation.UNSUPPORTED_APPLICATION,
                        "condition(s) "
                        + ", ".join(sorted(unresolved))
                        + " are not satisfied, so this cannot state a positive conclusion",
                    )
                )

        return Verdict(
            claim=claim.model_copy(update={"citations": tuple(citations)}),
            findings=tuple(findings),
        )

    def _verify_unknown(self, claim: Claim) -> Verdict:
        findings: list[Finding] = []
        if not line_body(claim.text).startswith(UNKNOWN_OPENERS):
            findings.append(
                Finding(
                    Violation.MALFORMED_UNKNOWN,
                    "an unknown claim must open by naming what can't yet be determined",
                )
            )
        if numbers_in(MARKER.sub("", claim.text)):
            findings.append(
                Finding(Violation.MALFORMED_UNKNOWN, "an unknown claim states a number")
            )

        markers = [int(m) for m in MARKER.findall(claim.text)]
        citations: list[Citation] = []
        names_unknown_condition = False
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
            citations.append(
                Citation(marker=marker, path=unit.citation, quote=_excerpt(unit))
            )
            if any(
                self._condition_status.get(condition_id) is CheckStatus.UNKNOWN
                for condition_id in self._condition_ids_by_marker.get(marker, ())
            ):
                names_unknown_condition = True

        if not markers:
            findings.append(
                Finding(
                    Violation.MALFORMED_UNKNOWN,
                    "an unknown claim must cite the condition it can't yet determine",
                )
            )
        elif not names_unknown_condition:
            findings.append(
                Finding(
                    Violation.MALFORMED_UNKNOWN,
                    "no cited passage names a condition the analysis marked unknown",
                )
            )

        return Verdict(
            claim=claim.model_copy(update={"citations": tuple(citations)}),
            findings=tuple(findings),
        )


def _affirms(text: str) -> bool:
    """An affirmative modal with no negative one alongside it — shared by the
    content/application modal-mismatch check and the application-vs-analysis
    check, so a line reading e.g. "you can claim X, but not Y" is not
    mistaken for a bare affirmation."""
    return bool(_AFFIRMATIVE_MODAL.search(text)) and not _NEGATIVE_MODAL.search(text)


def _legal_numbers(text: str) -> frozenset[Decimal]:
    """Figures a text states *as law*: whatever follows a limit/rate word or a
    provision name within its own clause, and every percentage."""
    numbers: set[Decimal] = set()
    for keyword in _LEGAL_KEYWORD.finditer(text):
        tail = text[keyword.end() : keyword.end() + 40]
        clause_end = _CLAUSE_END.search(tail)
        numbers |= numbers_in(tail[: clause_end.start()] if clause_end else tail)
    for percent in _PERCENT.finditer(text):
        numbers |= numbers_in(percent.group("digits") or percent.group("words"))
    return frozenset(numbers)


def _is_sum_or_difference(value: Decimal, known: set[Decimal]) -> bool:
    """A figure the example worked out in its head ("the remaining ₹1,00,000"):
    accepted only if it is exactly a + b or a − b of two figures the line
    already carries — checked arithmetic, never a new legal number."""
    return any(
        a + b == value or a - b == value for a in known for b in known if a is not b
    )


def _mask(text: str, spans: list[re.Match[str]]) -> str:
    """Blank out equation spans, keeping every other offset unchanged."""
    chars = list(text)
    for span in spans:
        chars[span.start() : span.end()] = " " * (span.end() - span.start())
    return "".join(chars)


def _operand_value(raw: str) -> Decimal | None:
    text = re.sub(r"^(?:₹|Rs\.?|INR)\s*", "", raw.strip(), flags=re.IGNORECASE)
    percent = text.endswith("%")
    text = text.rstrip("% ").strip()
    match = re.fullmatch(
        r"(\d[\d,]*(?:\.\d+)?)\s*(lakhs?|crores?)?", text, re.IGNORECASE
    )
    if match is None:
        return None
    try:
        value = Decimal(match.group(1).replace(",", ""))
    except InvalidOperation:
        return None
    if match.group(2):
        value *= _MULTIPLIERS[match.group(2).lower().rstrip("s")]
    return value / 100 if percent else value


def _evaluate(expression: str) -> Decimal | None:
    """One side of an equation: `+ − × ÷` with the usual precedence and
    parentheses. Anything it can't parse is None — an unreadable equation is a
    failed one, never a passed one."""
    tokens: list[tuple[str, str]] = []
    for token in _TOKEN.finditer(expression):
        kind = token.lastgroup or ""
        tokens.append((kind, token.group(0).strip()))
    position = 0

    def peek() -> tuple[str, str] | None:
        return tokens[position] if position < len(tokens) else None

    def factor() -> Decimal | None:
        nonlocal position
        token = peek()
        if token is None:
            return None
        position += 1
        kind, text = token
        if kind == "operand":
            return _operand_value(text)
        if kind == "paren" and text == "(":
            value = add()
            closing = peek()
            if value is None or closing != ("paren", ")"):
                return None
            position += 1
            return value
        return None

    def multiply() -> Decimal | None:
        nonlocal position
        value = factor()
        while value is not None and (token := peek()) and token[0] == "op":
            if token[1] in ("+", "-", "−", "–", "‑"):
                break
            position += 1
            right = factor()
            if right is None or (token[1] in ("/", "÷") and right == 0):
                return None
            value = value / right if token[1] in ("/", "÷") else value * right
        return value

    def add() -> Decimal | None:
        nonlocal position
        value = multiply()
        while value is not None and (token := peek()) and token[0] == "op":
            position += 1
            right = multiply()
            if right is None:
                return None
            value = value + right if token[1] == "+" else value - right
        return value

    result = add()
    return result if position == len(tokens) else None


_PROVISION_NUMBER = re.compile(
    r"\b(?:sub-?sections?|sections?|sec\.|s\.|u/s\.?|clauses?|rules?|schedules?)"
    r"\s*\(?(\d+)",
    re.IGNORECASE,
)


def _provision_numbers(text: str) -> frozenset[Decimal]:
    return frozenset(Decimal(m) for m in _PROVISION_NUMBER.findall(text))


def _unsupported_finding(unsupported: list[Decimal]) -> Finding:
    return Finding(
        Violation.UNSUPPORTED_NUMBER,
        "no source for " + ", ".join(str(number) for number in unsupported),
    )


def _excerpt(unit: EvidenceUnit, limit: int = 600) -> str:
    text = unit.chunk.text
    return text if len(text) <= limit else text[:limit].rstrip() + "…"


def ground_numbers(unit: EvidenceUnit) -> frozenset[Decimal]:
    """Every number an evidence unit itself grounds: its own text, its
    citation path, and its ancestor lead-in lines. R20 Step 20.5's
    deterministic reasoning validator reuses this unchanged — a legal
    number the model reports in a rule, condition or limit must be
    grounded exactly the way a claim's own number already is."""
    numbers = numbers_in(unit.chunk.text) | numbers_in(unit.citation)
    for line in unit.context:
        numbers |= numbers_in(line.text)
    # R21 Part B: the Act prints a zero rate as "Nil" (section 202(1)'s first
    # slab), so a passage saying "Nil" grounds 0 — and only such a passage.
    if _NIL.search(unit.chunk.text) or any(
        _NIL.search(line.text) for line in unit.context
    ):
        numbers |= {Decimal(0)}
    return numbers


def _names_an_unsourced_online_step(text: str, units: list[EvidenceUnit]) -> bool:
    sources = " ".join(
        [unit.chunk.text for unit in units]
        + [line.text for unit in units for line in unit.context]
    ).lower()
    return any(
        re.search(rf"\b{re.escape(found.group(0).lower())}", sources) is None
        for found in _ONLINE_PROCEDURE.finditer(normalise(text))
    )


def _asserts_the_opposite_of_its_source(text: str, units: list[EvidenceUnit]) -> bool:
    """One-directional: a claim affirming what a cited passage denies. See
    this module's docstring for why the reverse (an overly cautious claim)
    is not gated."""
    if not _affirms(text):
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
