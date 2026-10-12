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

NUMBER = r"\d{1,3}(?:-[A-Z]{1,3})?[A-Z]{0,3}"
BRACKET_ONE = r"\([^()]{1,10}\)"
# A PDF line wrap can fall between two brackets of the same citation
# ("section 10(15)\n(iii)"), so whitespace -- including a newline -- is
# tolerated between consecutive groups, not just around the whole chain.
BRACKET_CHAIN = rf"(?:\s*{BRACKET_ONE})*"
BRACKET_ONE_PATTERN = re.compile(BRACKET_ONE)
SCHEDULE_ROMAN = r"[IVXLCDM]+"

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

ACT_TAIL = re.compile(
    r"\s*(?:of|to)\s+(?:the\s+)?(?P<name>this Act|said Act|that Act"
    r"|[A-Za-z][A-Za-z&,.\-'()\s]{2,140}?\s+(?:Act|Code|Sanhita))\b"
)

MAX_RANGE_SPAN = 50

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
    """A reference this corpus cannot resolve because it names another Act, or refers
    anaphorically ("the said Act") to one named earlier."""

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
    """Collapse a line-wrap inside a bracket chain before it becomes part of a citation
    string -- ``NodePath`` citations never contain whitespace."""
    return _WHITESPACE.sub("", brackets)


def _is_plain(number: str) -> bool:
    return number.isdigit()


def _int_range(low: str, high: str) -> list[str]:
    lo, hi = int(low), int(high)
    if lo > hi or hi - lo > MAX_RANGE_SPAN:
        return [low, high]
    return [str(n) for n in range(lo, hi + 1)]


def _section_citations(group_text: str) -> list[str]:
    """A later list item may give only its own trailing bracket, inheriting everything
    before it from the item before ("section 70(1)(a), (c) and (d)" means clauses (a),
    (c), (d) all of sub-section (1) of section 70 -- the *whole* prefix up to the
    varying bracket carries over, not just the section number, or "(c)" would wrongly
    collapse to depth-1 ``70(c)``)."""
    citations: list[str] = []
    previous_plain: str | None = None
    prefix: str | None = None
    for match in SECTION_TOKEN.finditer(group_text):
        sep = match.group("sep")
        number = match.group("num")
        brackets = _clean(match.group("brackets") or match.group("inherited") or "")

        if number is not None:
            if (
                sep
                and sep.lower() == "to"
                and previous_plain
                and not brackets
                and _is_plain(number)
            ):
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
    return [
        f"Schedule {match.group('num')}"
        for match in SCHEDULE_TOKEN.finditer(group_text)
    ]


_SELF_REF_TYPE_SET = frozenset(SELF_REF_TYPES.values())


class ScannedReference(BaseModel):
    """One reference found in a run of text, before anything decides what it means."""

    model_config = ConfigDict(frozen=True)

    ref_type: RefType
    surface_text: str
    citations: tuple[str, ...]
    act_name: str | None = None


def scan_references(text: str) -> list[ScannedReference]:
    """Every citation surface form this corpus uses, over arbitrary text."""
    text = normalise(text)
    found: list[ScannedReference] = []
    consumed: list[tuple[int, int]] = []

    def take(ref_type: RefType, match: re.Match[str], citations: list[str]) -> None:
        name, tail_end = _act_tail(text, match.end())
        consumed.append((match.start(), tail_end))
        found.append(
            ScannedReference(
                ref_type=ref_type,
                surface_text=text[match.start() : match.end()],
                citations=tuple(citations),
                act_name=name,
            )
        )

    for match in PREFIX_CHAIN.finditer(text):
        if _overlaps(match.span(), consumed):
            continue
        markers = PREFIX_TOKEN.findall(match.group("chain"))
        citation = (
            match.group("num")
            + _clean(match.group("brackets"))
            + "".join(f"({marker})" for marker in reversed(markers))
        )
        take(RefType.SECTION, match, [citation])

    for match in PART_PARAGRAPH.finditer(text):
        if _overlaps(match.span(), consumed):
            continue
        marker = f"{match.group('part')}{match.group('num')}"
        take(
            RefType.SCHEDULE_PARAGRAPH,
            match,
            [
                f"Schedule {match.group('sched')}({marker}){_clean(match.group('brackets'))}"
            ],
        )

    for match in PLAIN_PARAGRAPH.finditer(text):
        if _overlaps(match.span(), consumed):
            continue
        take(
            RefType.SCHEDULE_PARAGRAPH,
            match,
            [
                f"Schedule {match.group('sched')}({match.group('num')})"
                f"{_clean(match.group('brackets'))}"
            ],
        )

    for match in BARE_PART.finditer(text):
        if _overlaps(match.span(), consumed):
            continue
        take(RefType.SCHEDULE_PART, match, [f"Schedule {match.group('sched')}"])

    for match in SCHEDULE_GROUP.finditer(text):
        if _overlaps(match.span(), consumed):
            continue
        take(RefType.SCHEDULE, match, _schedule_citations(match.group(0)))

    for match in SECTION_GROUP.finditer(text):
        if _overlaps(match.span(), consumed):
            continue
        take(RefType.SECTION, match, _section_citations(match.group(0)))

    for match in SELF_REF.finditer(text):
        if _overlaps(match.span(), consumed):
            continue
        found.append(
            ScannedReference(
                ref_type=SELF_REF_TYPES[match.group(1).lower()],
                surface_text=text[match.start() : match.end()],
                citations=(),
            )
        )

    return found


def extract_node_references(
    node: StatutoryNode, index: dict[str, StatutoryNode]
) -> tuple[list[CrossReference], list[ExternalReference]]:
    """Regex over one node's own text (never its children's -- each line of the Act
    belongs to exactly one node's ``text``, so walking every node once covers the whole
    tree without double-counting)."""
    references: list[CrossReference] = []
    external: list[ExternalReference] = []

    for scanned in scan_references(node.text):
        if scanned.act_name is not None and scanned.act_name.lower() != "this act":
            external.append(
                ExternalReference(
                    from_path=node.citation,
                    surface_text=scanned.surface_text,
                    act_name=scanned.act_name,
                )
            )
            continue
        if scanned.ref_type in _SELF_REF_TYPE_SET:
            references.append(
                CrossReference(
                    from_path=node.citation,
                    ref_type=scanned.ref_type,
                    surface_text=scanned.surface_text,
                    target_path=None,
                    resolved=True,
                )
            )
            continue
        for citation in scanned.citations:
            references.append(
                CrossReference(
                    from_path=node.citation,
                    ref_type=scanned.ref_type,
                    surface_text=scanned.surface_text,
                    target_path=citation,
                    resolved=citation in index,
                )
            )

    return references, external


def extract_glossary(sections: Sequence[StatutoryNode]) -> tuple[GlossaryTerm, ...]:
    """Section 2's definitions, term -> the clause node that names it."""
    section2 = next((section for section in sections if section.marker == "2"), None)
    if section2 is None:
        return ()
    terms: list[GlossaryTerm] = []
    for clause in section2.children:
        term = defined_term(clause.text)
        if term is not None and clause.path is not None:
            terms.append(GlossaryTerm(term=term, node_path=clause.citation))
    return tuple(terms)


def defined_term(clause_text: str) -> str | None:
    """The term a section 2 clause defines, from its first non-empty line."""
    first_line = next((line for line in clause_text.split("\n") if line.strip()), "")
    content = normalise(first_line).strip()
    content = re.sub(r"^\(\d{1,3}[A-Za-z]{0,3}\)\s*", "", content)
    match = GLOSSARY_TERM.match(content)
    return match.group(1) if match else None


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
