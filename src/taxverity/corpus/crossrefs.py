from __future__ import annotations

import re
from collections.abc import Sequence
from enum import StrEnum

from pydantic import BaseModel, ConfigDict

from taxverity.corpus.loader import normalise
from taxverity.corpus.nodes import StatutoryNode
from taxverity.observability import get_logger

logger = get_logger(__name__)

CROSSREF_STAGE_VERSION = 1

# The optional hyphen segment is never real in *this* Act -- 1.1/1.4 measured
# zero hyphenated section numbers here -- but the repealed 1961 Act numbers
# some sections that way ("80-IA", "80-ID"), and this Act's own text quotes
# them by cross-reference. Consuming the whole token (rather than stopping at
# the hyphen) keeps the constructed citation honestly unresolvable instead of
# silently truncating to a short prefix ("80") that could coincidentally
# collide with an unrelated real section of this Act.
NUMBER = r"\d{1,3}(?:-[A-Z]{1,3})?[A-Z]{0,3}"
BRACKET_ONE = r"\([^()]{1,10}\)"
# A PDF line wrap can fall between two brackets of the same citation
# ("section 10(15)\n(iii)"), so whitespace -- including a newline -- is
# tolerated between consecutive groups, not just around the whole chain.
BRACKET_CHAIN = rf"(?:\s*{BRACKET_ONE})*"
BRACKET_ONE_PATTERN = re.compile(BRACKET_ONE)
SCHEDULE_ROMAN = r"[IVXLCDM]+"

# Measured, not assumed (scratch_crossrefs.py against the real corpus): the
# 2025 Act's own drafting convention is the postfix citation form used
# throughout this codebase's own NodePath.render() -- "section 515(3)(b)" --
# not the older prefix style "sub-section (3) of section 515". The prefix
# style survives only as a rare minority (~30 hits across 1.68M chars) and is
# handled separately by PREFIX_CHAIN below.
SELF_REF = re.compile(r"\bthis\s+(Act|Chapter|Part|Schedule|sub-section|section)\b")

PREFIX_WORD = r"(?:sub-section|clause|sub-clause|item)"
# The chained prefix and the Act's own postfix style combine ("clause (a) of
# section 80-ID(6)" -- clause (a) of sub-something (6) of section 80-ID), so a
# trailing bracket chain on the section number is captured too, not just the
# bare number.
PREFIX_CHAIN = re.compile(
    rf"(?P<chain>(?:{PREFIX_WORD}\s*{BRACKET_ONE}\s*of\s*)+)"
    rf"section\s+(?P<num>{NUMBER})(?P<brackets>{BRACKET_CHAIN})"
)
PREFIX_TOKEN = re.compile(rf"{PREFIX_WORD}\s*\(([^()]{{1,10}})\)")

# "Part A of Schedule XI, paragraph 8" -- deferred from Step 1.7 (schedules.py)
# because a Part is folded into the paragraph's own marker (A8), not a node
# of its own. These three forms recover that fold from the Act's own prose,
# most-specific first so a paragraph reference is never re-matched by the
# coarser bare-Part or bare-Schedule passes below.
PART_PARAGRAPH = re.compile(
    rf"\bparagraph\s+(?P<num>{NUMBER})(?P<brackets>{BRACKET_CHAIN})\s+of\s+Part\s+"
    rf"(?P<part>[A-Z])\s+of\s+(?:the\s+)?Schedule\s+(?P<sched>{SCHEDULE_ROMAN})\b"
)
PLAIN_PARAGRAPH = re.compile(
    rf"\bparagraph\s+(?P<num>{NUMBER})(?P<brackets>{BRACKET_CHAIN})\s+of\s+(?:the\s+)?"
    rf"Schedule\s+(?P<sched>{SCHEDULE_ROMAN})\b"
)
BARE_PART = re.compile(
    rf"\bPart\s+(?P<part>[A-Z])\s+of\s+(?:the\s+)?Schedule\s+(?P<sched>{SCHEDULE_ROMAN})\b"
)
SCHEDULE_GROUP = re.compile(
    rf"\b[Ss]chedules?\s+{SCHEDULE_ROMAN}(?:\s*(?:,|and|or)\s*{SCHEDULE_ROMAN})*"
)
SCHEDULE_TOKEN = re.compile(rf"(?P<sep>,|and|or)?\s*(?P<num>{SCHEDULE_ROMAN})")

# A later list item may drop the section number and give only its own bracket
# ("section 2(a), (c) and (h) of the ... Act" -- clauses (a), (c), (h) all of
# section 2), so a continuation is either a fresh number or a bare, non-empty
# bracket chain that inherits the number last seen.
SECTION_GROUP = re.compile(
    rf"\b[Ss]ections?\s+{NUMBER}{BRACKET_CHAIN}"
    rf"(?:\s*(?:,|and|or|to)\s*(?:{NUMBER}{BRACKET_CHAIN}|{BRACKET_ONE}{BRACKET_CHAIN}))*"
)
SECTION_TOKEN = re.compile(
    rf"(?P<sep>,|and|or|to)?\s*"
    rf"(?:(?P<num>{NUMBER})(?P<brackets>{BRACKET_CHAIN})|(?P<inherited>{BRACKET_ONE}{BRACKET_CHAIN}))"
)

# A reference immediately followed by "of/to <Act name>" points outside this
# corpus entirely -- a different statute (Companies Act, FEMA...) or an
# anaphoric "the said Act"/"that Act" naming one mentioned earlier in the same
# clause. Neither is a node this corpus can resolve to, so both are kept out
# of the edge table and reported separately (ADR-049), exactly as 1.5/1.6
# separated the amendment apparatus and 1961-Act quotations rather than
# letting them corrupt the real count.
# No leading ``^`` here: this is used with ``Pattern.match(text, pos)``, whose
# match already starts exactly at ``pos`` -- an explicit ``^`` would instead
# re-anchor to the real start of ``text`` (or just after a newline) and never
# match at an arbitrary offset, which is the whole point of a lookahead here.
# The name class allows ``()`` ("Securities Contracts (Regulation) Act"), an
# apostrophe ("Employees' Provident Funds ... Act"), and ``\s`` rather than a
# literal space -- a PDF line wrap can fall mid-name ("Income-\ntax Act"), and
# ``\s`` matches that newline. A named statute occasionally ends in "Code"
# rather than "Act" (the Insolvency and Bankruptcy Code) or in "Sanhita" --
# India's 2023 recodified criminal statutes (Bharatiya Nagarik Suraksha
# Sanhita and siblings) use that word where an older Act would say "Code" --
# measured, not assumed, against every external mention in the real corpus.
# The interior quantifier is generous (some real names run past 90 chars --
# "National Trust for Welfare of Persons with Autism, Cerebral Palsy, Mental
# Retardation and Multiple Disabilities Act") but still bounded, so an
# ordinary sentence with no Act name never accidentally spans into one.
ACT_TAIL = re.compile(
    r"\s*(?:of|to)\s+(?:the\s+)?(?P<name>this Act|said Act|that Act"
    r"|[A-Za-z][A-Za-z&,.\-'()\s]{2,140}?\s+(?:Act|Code|Sanhita))\b"
)

# Section numbering has zero gaps 1-536 (+354A) -- Step 1.1/1.4 measured
# this -- so "sections 28 to 33" can be expanded to every integer in between
# without risking a fabricated citation, but only between two bare, unbracketed
# numbers; a 50-integer cap keeps a stray match from exploding.
MAX_RANGE_SPAN = 50

# A clause opening this Act's own definitions convention: `(12)"term" means...`
# -- measured against every child of section 2 (scratch_section2.py). The
# marker prefix is stripped by the caller before this is applied.
GLOSSARY_TERM = re.compile(r'^"([^"]{1,80})"')


class RefType(StrEnum):
    SECTION = "section"
    SCHEDULE = "schedule"
    SCHEDULE_PART = "schedule_part"
    SCHEDULE_PARAGRAPH = "schedule_paragraph"
    THIS_ACT = "this_act"
    THIS_CHAPTER = "this_chapter"
    THIS_PART = "this_part"
    THIS_SECTION = "this_section"
    THIS_SUBSECTION = "this_subsection"
    THIS_SCHEDULE = "this_schedule"


SELF_REF_TYPES = {
    "act": RefType.THIS_ACT,
    "chapter": RefType.THIS_CHAPTER,
    "part": RefType.THIS_PART,
    "schedule": RefType.THIS_SCHEDULE,
    "section": RefType.THIS_SECTION,
    "sub-section": RefType.THIS_SUBSECTION,
}


class CrossReference(BaseModel):
    model_config = ConfigDict(frozen=True)

    from_path: str
    ref_type: RefType
    surface_text: str
    target_path: str | None = None
    resolved: bool = False


class ExternalReference(BaseModel):
    """A reference this corpus cannot resolve because it names another Act,
    or refers anaphorically ("the said Act") to one named earlier. Kept
    verbatim, never silently dropped -- same discipline as ``ParsedAct.footnotes``.
    """

    model_config = ConfigDict(frozen=True)

    from_path: str
    surface_text: str
    act_name: str


class GlossaryTerm(BaseModel):
    model_config = ConfigDict(frozen=True)

    term: str
    node_path: str


class CrossReferenceIndex(BaseModel):
    model_config = ConfigDict(frozen=True)

    references: tuple[CrossReference, ...]
    external: tuple[ExternalReference, ...]
    glossary: tuple[GlossaryTerm, ...]

    @property
    def dangling(self) -> tuple[CrossReference, ...]:
        return tuple(ref for ref in self.references if not ref.resolved)

    @property
    def resolution_rate(self) -> float:
        if not self.references:
            return 1.0
        resolved = sum(1 for ref in self.references if ref.resolved)
        return resolved / len(self.references)


def build_node_index(roots: Sequence[StatutoryNode]) -> dict[str, StatutoryNode]:
    return {
        node.citation: node
        for root in roots
        for node in root.walk()
        if node.path is not None
    }


def _overlaps(span: tuple[int, int], consumed: list[tuple[int, int]]) -> bool:
    start, end = span
    return any(start < c_end and c_start < end for c_start, c_end in consumed)


def _act_tail(text: str, end: int) -> tuple[str | None, int]:
    """Whether the reference ending at ``end`` names an Act, and where that name ends."""
    match = ACT_TAIL.match(text, end, min(end + 180, len(text)))
    if not match:
        return None, end
    return match.group("name"), match.end()


_WHITESPACE = re.compile(r"\s+")


def _clean(brackets: str) -> str:
    """Collapse a line-wrap inside a bracket chain before it becomes part of a
    citation string -- ``NodePath`` citations never contain whitespace."""
    return _WHITESPACE.sub("", brackets)


def _is_plain(number: str) -> bool:
    return number.isdigit()


def _int_range(low: str, high: str) -> list[str]:
    lo, hi = int(low), int(high)
    if lo > hi or hi - lo > MAX_RANGE_SPAN:
        return [low, high]
    return [str(n) for n in range(lo, hi + 1)]


def _section_citations(group_text: str) -> list[str]:
    """A later list item may give only its own trailing bracket, inheriting
    everything before it from the item before ("section 70(1)(a), (c) and
    (d)" means clauses (a), (c), (d) all of sub-section (1) of section 70 --
    the *whole* prefix up to the varying bracket carries over, not just the
    section number, or "(c)" would wrongly collapse to depth-1 ``70(c)``).
    """
    citations: list[str] = []
    previous_plain: str | None = None
    prefix: str | None = None
    for match in SECTION_TOKEN.finditer(group_text):
        sep = match.group("sep")
        number = match.group("num")
        brackets = _clean(match.group("brackets") or match.group("inherited") or "")

        if number is not None:
            if sep and sep.lower() == "to" and previous_plain and not brackets and _is_plain(number):
                # ``previous_plain`` was already appended as its own citation
                # on the prior iteration -- only the rest of the range is new.
                citations.extend(_int_range(previous_plain, number)[1:])
                prefix = number
                previous_plain = number
                continue
            citations.append(f"{number}{brackets}")
            kept = BRACKET_ONE_PATTERN.findall(brackets)[:-1]
            prefix = number + "".join(kept)
            previous_plain = number if not brackets else None
            continue

        if prefix is None or not brackets:
            continue
        citations.append(f"{prefix}{brackets}")
        previous_plain = None
    return citations


def _schedule_citations(group_text: str) -> list[str]:
    return [f"Schedule {match.group('num')}" for match in SCHEDULE_TOKEN.finditer(group_text)]


def extract_node_references(
    node: StatutoryNode, index: dict[str, StatutoryNode]
) -> tuple[list[CrossReference], list[ExternalReference]]:
    """Regex over one node's own text (never its children's -- each line of the
    Act belongs to exactly one node's ``text``, so walking every node once
    covers the whole tree without double-counting).
    """
    text = normalise(node.text)
    references: list[CrossReference] = []
    external: list[ExternalReference] = []
    consumed: list[tuple[int, int]] = []

    def resolve(ref_type: RefType, surface: str, citation: str) -> None:
        target = index.get(citation)
        references.append(
            CrossReference(
                from_path=node.citation,
                ref_type=ref_type,
                surface_text=surface,
                target_path=citation,
                resolved=target is not None,
            )
        )

    def dispatch(
        ref_type: RefType, span: tuple[int, int], citations: list[str], name: str | None
    ) -> None:
        surface = text[span[0] : span[1]]
        if name is not None and name.lower() != "this act":
            external.append(
                ExternalReference(from_path=node.citation, surface_text=surface, act_name=name)
            )
            return
        for citation in citations:
            resolve(ref_type, surface, citation)

    for match in PREFIX_CHAIN.finditer(text):
        if _overlaps(match.span(), consumed):
            continue
        markers = PREFIX_TOKEN.findall(match.group("chain"))
        citation = (
            match.group("num")
            + _clean(match.group("brackets"))
            + "".join(f"({marker})" for marker in reversed(markers))
        )
        name, tail_end = _act_tail(text, match.end())
        consumed.append((match.start(), tail_end))
        dispatch(RefType.SECTION, (match.start(), match.end()), [citation], name)

    for match in PART_PARAGRAPH.finditer(text):
        if _overlaps(match.span(), consumed):
            continue
        marker = f"{match.group('part')}{match.group('num')}"
        citation = f"Schedule {match.group('sched')}({marker}){_clean(match.group('brackets'))}"
        name, tail_end = _act_tail(text, match.end())
        consumed.append((match.start(), tail_end))
        dispatch(RefType.SCHEDULE_PARAGRAPH, (match.start(), match.end()), [citation], name)

    for match in PLAIN_PARAGRAPH.finditer(text):
        if _overlaps(match.span(), consumed):
            continue
        citation = f"Schedule {match.group('sched')}({match.group('num')}){_clean(match.group('brackets'))}"
        name, tail_end = _act_tail(text, match.end())
        consumed.append((match.start(), tail_end))
        dispatch(RefType.SCHEDULE_PARAGRAPH, (match.start(), match.end()), [citation], name)

    for match in BARE_PART.finditer(text):
        if _overlaps(match.span(), consumed):
            continue
        citation = f"Schedule {match.group('sched')}"
        name, tail_end = _act_tail(text, match.end())
        consumed.append((match.start(), tail_end))
        dispatch(RefType.SCHEDULE_PART, (match.start(), match.end()), [citation], name)

    for match in SCHEDULE_GROUP.finditer(text):
        if _overlaps(match.span(), consumed):
            continue
        name, tail_end = _act_tail(text, match.end())
        consumed.append((match.start(), tail_end))
        dispatch(RefType.SCHEDULE, (match.start(), match.end()), _schedule_citations(match.group(0)), name)

    for match in SECTION_GROUP.finditer(text):
        if _overlaps(match.span(), consumed):
            continue
        name, tail_end = _act_tail(text, match.end())
        consumed.append((match.start(), tail_end))
        dispatch(RefType.SECTION, (match.start(), match.end()), _section_citations(match.group(0)), name)

    for match in SELF_REF.finditer(text):
        if _overlaps(match.span(), consumed):
            continue
        ref_type = SELF_REF_TYPES[match.group(1).lower()]
        references.append(
            CrossReference(
                from_path=node.citation,
                ref_type=ref_type,
                surface_text=text[match.start() : match.end()],
                target_path=None,
                resolved=True,
            )
        )

    return references, external


def extract_glossary(sections: Sequence[StatutoryNode]) -> tuple[GlossaryTerm, ...]:
    """Section 2's definitions, term -> the clause node that names it.

    A clause that delegates ("shall have the meaning assigned to it in section
    X") is still the right target: it is where a reader looking up the term
    lands, and the delegation itself is captured as an ordinary outgoing
    ``CrossReference`` from that same clause.
    """
    section2 = next((section for section in sections if section.marker == "2"), None)
    if section2 is None:
        return ()
    terms: list[GlossaryTerm] = []
    for clause in section2.children:
        first_line = next((line for line in clause.text.split("\n") if line.strip()), "")
        content = normalise(first_line).strip()
        content = re.sub(r"^\(\d{1,3}[A-Za-z]{0,3}\)\s*", "", content)
        match = GLOSSARY_TERM.match(content)
        if match and clause.path is not None:
            terms.append(GlossaryTerm(term=match.group(1), node_path=clause.citation))
    return tuple(terms)


def extract_crossrefs(
    sections: Sequence[StatutoryNode], schedules: Sequence[StatutoryNode]
) -> CrossReferenceIndex:
    roots = (*sections, *schedules)
    index = build_node_index(roots)
    references: list[CrossReference] = []
    external: list[ExternalReference] = []
    for root in roots:
        for node in root.walk():
            if not node.text:
                continue
            found_refs, found_external = extract_node_references(node, index)
            references.extend(found_refs)
            external.extend(found_external)
    result = CrossReferenceIndex(
        references=tuple(references),
        external=tuple(external),
        glossary=extract_glossary(sections),
    )
    logger.info(
        "extracted %d references (%.2f%% resolved), %d external-Act mentions, "
        "%d glossary terms",
        len(result.references),
        result.resolution_rate * 100,
        len(result.external),
        len(result.glossary),
    )
    if result.dangling:
        logger.warning("%d references did not resolve", len(result.dangling))
    return result
