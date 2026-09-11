from __future__ import annotations

import re
from collections.abc import Iterable
from enum import StrEnum

from pydantic import BaseModel, ConfigDict

from taxverity.corpus.nodes import (
    CITATION_DEPTH_TYPES,
    NodePath,
    NodeType,
    StatutoryNode,
)
from taxverity.corpus.sections import ParsedAct
from taxverity.corpus.tables import TableRegion, in_any_region
from taxverity.observability import get_logger

logger = get_logger(__name__)

SUBSTRUCTURE_STAGE_VERSION = 2

LEVEL_TYPES = CITATION_DEPTH_TYPES[NodeType.SECTION]

# Two things this has to get right. The marker may be wrapped in the square
# brackets the Act uses to fence substituted text ("[(4) "commodities
# transactions tax" ... ;]"), and missing one is not a local error — the level
# never opens, so every later sibling in that run fails to place too, which cost
# section 66 thirty-six nodes. And the length is deliberately loose, because
# "(viii)" and "(xviii)" are markers the Act really uses; what narrows it is
# ``kinds_for``, which admits only members of the five alphabets below.
LEAD_MARKER = re.compile(r"^\s*\[?\((\d{1,3}|[a-z]{1,8}|[A-Z]{1,8})\)")
SECTION_HEAD = re.compile(r"^\d{1,3}[A-Z]{0,3}\s*\.?\s*")

# A line-initial marker is a citation fragment, not a node, when the line above
# breaks off on the word that introduces it ("... specified in sub-section" then
# "(2)—"). Left alone such a fragment steals the next real sibling's marker:
# section 44's subsection (2) was consumed by a wrapped reference to it.
REFERENCE_TAIL = re.compile(
    r"\b(?:(?:sub-)?(?:sections?|clauses?)|items?|paragraphs?|schedules?|under)$",
    re.IGNORECASE,
)

TABLE_LINE = "TABLE"

# A list that announces itself ("... objects to it by a statement on oath that—")
# opens a child even when its alphabet is already open higher up: section
# 416(5)(g) really does contain (a) and (b). Without the announcement, a repeat
# of an open alphabet is a restart of that level instead.
LEAD_IN = re.compile(r"[—:]\s*$")

_ROMAN_UNITS = ["", "i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix"]
_ROMAN_TENS = ["", "x", "xx", "xxx", "xl", "l", "lx", "lxx", "lxxx", "xc"]
ROMAN_LOWER = tuple(
    _ROMAN_TENS[n // 10] + _ROMAN_UNITS[n % 10] for n in range(1, 100)
)
# The Act continues z with za..zl rather than aa..al — measured, not assumed.
ALPHA_LOWER = tuple(chr(ord("a") + n) for n in range(26)) + tuple(
    "z" + chr(ord("a") + n) for n in range(26)
)


class MarkerKind(StrEnum):
    NUMERIC = "numeric"
    ROMAN_LOWER = "roman_lower"
    ALPHA_LOWER = "alpha_lower"
    ROMAN_UPPER = "roman_upper"
    ALPHA_UPPER = "alpha_upper"


SEQUENCES: dict[MarkerKind, tuple[str, ...]] = {
    MarkerKind.ROMAN_LOWER: ROMAN_LOWER,
    MarkerKind.ALPHA_LOWER: ALPHA_LOWER,
    MarkerKind.ROMAN_UPPER: tuple(marker.upper() for marker in ROMAN_LOWER),
    MarkerKind.ALPHA_UPPER: tuple(marker.upper() for marker in ALPHA_LOWER),
}


class AnomalyReason(StrEnum):
    NOT_A_MARKER = "not_a_marker"
    TOO_DEEP = "too_deep"
    DUPLICATE_PATH = "duplicate_path"


# Only a duplicate path invalidates a root's structure. The invariant at stake
# is that citation -> chunk is injective: the verifier resolves a claim's
# citation to exactly one text before checking quote fidelity, so two nodes
# rendering one citation give it two texts for one claim. DUPLICATE_PATH is
# literally that check. TOO_DEEP creates no node, so it mints no citation and
# cannot collide — what it loses is addressability at a depth NodePath cannot
# render, and the line still sits in its nearest valid ancestor, whose text
# carries it. NOT_A_MARKER is the parser working correctly.
INVALIDATING_REASONS = frozenset({AnomalyReason.DUPLICATE_PATH})


class PageProvenanceError(ValueError):
    pass


class Anomaly(BaseModel):
    model_config = ConfigDict(frozen=True)

    section: str
    line: int
    marker: str
    reason: AnomalyReason
    context: str

    @property
    def invalidates_structure(self) -> bool:
        return self.reason in INVALIDATING_REASONS


class Substructure(BaseModel):
    model_config = ConfigDict(frozen=True)

    sections: tuple[StatutoryNode, ...]
    anomalies: tuple[Anomaly, ...]
    clause_rooted: tuple[str, ...]
    table_sections: tuple[str, ...]

    def node(self, citation: str) -> StatutoryNode | None:
        for section in self.sections:
            found = section.find(citation)
            if found is not None:
                return found
        return None

    @property
    def unreliable(self) -> tuple[str, ...]:
        """Sections whose tree carries a structure-invalidating anomaly."""
        return tuple(
            sorted(
                {
                    anomaly.section
                    for anomaly in self.anomalies
                    if anomaly.invalidates_structure
                }
            )
        )

    def count(self, type: NodeType) -> int:
        return sum(
            1 for section in self.sections for node in section.walk() if node.type is type
        )


def kinds_for(marker: str) -> tuple[MarkerKind, ...]:
    """Every alphabet the marker could belong to, romans first.

    ``(i)`` is roman one and alpha nine; ``(v)`` is roman five and alpha
    twenty-two. Nothing about the marker itself resolves that — only its place in
    a sequence does — so this returns candidates, never an answer.
    """
    if marker.isdigit():
        return (MarkerKind.NUMERIC,)
    if marker.islower():
        candidates = (MarkerKind.ROMAN_LOWER, MarkerKind.ALPHA_LOWER)
    else:
        candidates = (MarkerKind.ROMAN_UPPER, MarkerKind.ALPHA_UPPER)
    return tuple(kind for kind in candidates if marker in SEQUENCES[kind])


def successor(kind: MarkerKind, marker: str) -> str | None:
    if kind is MarkerKind.NUMERIC:
        return str(int(marker) + 1)
    sequence = SEQUENCES[kind]
    if marker not in sequence:
        return None
    position = sequence.index(marker) + 1
    return sequence[position] if position < len(sequence) else None


def opens(kind: MarkerKind, marker: str) -> bool:
    if kind is MarkerKind.NUMERIC:
        return marker == "1"
    return marker == SEQUENCES[kind][0]


def is_clause_rooted(text: str) -> bool:
    """Whether the level below this section is clauses rather than sub-sections.

    The Act calls section 9's ``(1)`` a sub-section and section 2's ``(22)`` a
    clause, and the two markers are typographically identical. What separates
    them is the section's own opening line: a clause run continues a sentence the
    section began ("2. In this Act, unless the context otherwise requires,—"),
    whereas a sub-section carries its marker inline ("9. (1) The income referred
    to ..."). A bare opening line means the marker merely wrapped, so it is a
    sub-section. Measured against the Act's own vocabulary, 536 of 537 sections
    agree; the exception (428) refers to a *different* section's sub-section.
    """
    first = next((line for line in text.split("\n") if line.strip()), "")
    lead = SECTION_HEAD.sub("", first, count=1).strip()
    return bool(lead) and not LEAD_MARKER.match(lead)


class _Level:
    __slots__ = ("kind", "marker", "type", "path", "lines", "pages", "children")

    def __init__(self, kind: MarkerKind, marker: str, type: NodeType, path: NodePath):
        self.kind = kind
        self.marker = marker
        self.type = type
        self.path = path
        self.lines: list[str] = []
        self.pages: list[int] = []
        self.children: list[StatutoryNode] = []

    def close(self) -> StatutoryNode:
        children = tuple(self.children)
        # A marker sharing its line with a deeper one ("(5)(a) Income by way of
        # ...") holds no line of its own, so its provenance is its children's.
        pages = self.pages or [page for child in children for page in child.pages]
        return StatutoryNode(
            type=self.type,
            marker=self.marker,
            path=self.path,
            text="\n".join(self.lines),
            pages=tuple(dict.fromkeys(pages)),
            children=children,
        )


class _Builder:
    """Grows one section's flat lines into a tree, by sibling sequence alone.

    A marker that continues an open level closes everything below it; a marker
    that opens an alphabet not already on the stack pushes a new level; anything
    else is prose belonging to the deepest open node. Marker *shape* never
    decides type — page 27 runs a clause sequence ``a..h, i, j``, so ``(i)`` is a
    sub-clause only when no alpha level is waiting for it.
    """

    def __init__(
        self,
        section: StatutoryNode,
        depth_shift: int,
        level_types: tuple[NodeType, ...] = LEVEL_TYPES,
    ):
        self.section = section
        self.depth_shift = depth_shift
        self.level_types = level_types
        self.stack: list[_Level] = []
        self.root = _Level(MarkerKind.NUMERIC, section.marker, section.type, section.path)
        self.seen: set[str] = set()
        self.anomalies: list[Anomaly] = []

    def type_at(self, depth: int) -> NodeType:
        return self.level_types[min(depth + self.depth_shift, len(self.level_types) - 1)]

    def path_at(self, depth: int, marker: str) -> NodePath:
        parent = self.stack[depth - 1].path if depth else self.section.path
        return parent.child(self.type_at(depth), marker)

    def collapse_to(self, depth: int) -> None:
        while len(self.stack) > depth:
            done = self.stack.pop().close()
            parent = self.stack[-1] if self.stack else self.root
            parent.children.append(done)

    def push(self, depth: int, kind: MarkerKind, marker: str) -> None:
        self.collapse_to(depth)
        self.stack.append(
            _Level(kind, marker, self.type_at(depth), self.path_at(depth, marker))
        )

    def resumes_open_sequence(self, marker: str) -> bool:
        return any(
            successor(level.kind, level.marker) == marker for level in self.stack
        )

    def place(self, marker: str, ahead: str | None, announced: bool) -> AnomalyReason | None:
        """Place the marker, or name the refusal made.

        Returns ``None`` on success. The two refusals are unrelated and their
        consequences differ, so the caller is told which one happened rather
        than re-deriving a cause it cannot see from here.
        """
        for depth in range(len(self.stack) - 1, -1, -1):
            if successor(self.stack[depth].kind, self.stack[depth].marker) == marker:
                if self.nests_instead(depth, marker, ahead):
                    break
                self.push(depth, self.stack[depth].kind, marker)
                return None
        open_kinds = [level.kind for level in self.stack]
        for kind in kinds_for(marker):
            if not opens(kind, marker):
                continue
            reopening = kind in open_kinds and not announced
            depth = open_kinds.index(kind) if reopening else len(self.stack)
            # Below this the Act has no vocabulary and a citation has no
            # component to name, so a deeper marker is a parse artefact rather
            # than a node. Refusing it here keeps the runaway visible as an
            # anomaly instead of burying it in a 14-deep path.
            if depth >= len(self.level_types):
                return AnomalyReason.TOO_DEEP
            self.push(depth, kind, marker)
            return None
        return AnomalyReason.NOT_A_MARKER

    def nests_instead(self, depth: int, marker: str, ahead: str | None) -> bool:
        """Whether ``(h)`` -> ``(i)`` opens a roman list rather than continuing a-z.

        This is the one marker the sequence rule cannot settle on its own: page 27
        runs a clause sequence ``a..h, i, j`` while section 19(2)(h) opens a
        sub-clause list ``i, ii, iii``. Both are the successor of ``(h)``, and
        both are a roman opener. Only what comes next separates them, so that is
        what is consulted — never the marker's shape.
        """
        roman = next(
            (kind for kind in kinds_for(marker) if opens(kind, marker)), None
        )
        if roman is None or self.stack[depth].kind is roman:
            return False
        return ahead is not None and ahead == successor(roman, marker)

    def note(self, line: int, marker: str, reason: AnomalyReason, context: str) -> None:
        self.anomalies.append(
            Anomaly(
                # The citation, not the bare marker: a Schedule paragraph's
                # marker repeats across Schedules, so "2" alone names no one
                # node. For a section the two are identical.
                section=self.section.citation or self.section.marker,
                line=line,
                marker=marker,
                reason=reason,
                context=context[:120],
            )
        )

    def attach(self, raw: str, page: int | None) -> None:
        target = self.stack[-1] if self.stack else self.root
        target.lines.append(raw)
        if page is not None:
            target.pages.append(page)

    def finish(self) -> StatutoryNode:
        self.collapse_to(0)
        # The section keeps the page span Step 1.5 measured over its whole body,
        # not just the lead-in lines left behind once children took theirs.
        return self.section.model_copy(
            update={
                "text": "\n".join(self.root.lines),
                "children": tuple(self.root.children),
            }
        )


def build(
    section: StatutoryNode,
    pages: tuple[int, ...] = (),
    level_types: tuple[NodeType, ...] = LEVEL_TYPES,
    in_table: tuple[bool, ...] = (),
    depth_shift: int | None = None,
) -> tuple[StatutoryNode, tuple[Anomaly, ...]]:
    lines = section.text.split("\n") if section.text else []
    if pages and len(pages) != len(lines):
        raise PageProvenanceError(
            f"section {section.marker}: {len(pages)} pages for {len(lines)} lines"
        )
    if in_table and len(in_table) != len(lines):
        raise PageProvenanceError(
            f"section {section.marker}: {len(in_table)} table flags for {len(lines)} lines"
        )

    # is_clause_rooted's shift exists to skip a SUBSECTION level that a
    # clause-rooted section has no use for -- section 2 has no "(1)" before
    # its "(a)". A Schedule paragraph's ladder never had that level to begin
    # with (it starts at CLAUSE directly, per CITATION_DEPTH_TYPES), so
    # applying the same shift there skips CLAUSE itself instead, mistyping
    # every schedule paragraph's first level as SUBCLAUSE. Callers with a
    # ladder that starts below the root (schedules.py) pass depth_shift=0
    # explicitly rather than relying on this section-specific inference.
    if depth_shift is None:
        depth_shift = int(is_clause_rooted(section.text))
    builder = _Builder(section, depth_shift=depth_shift, level_types=level_types)
    tokens = lead_markers(lines)
    guessing_table = False
    previous = ""

    for index, raw in enumerate(lines):
        scan = SECTION_HEAD.sub("", raw, count=1) if index == 0 else raw
        announced = bool(LEAD_IN.search(previous))
        if scan.strip() == TABLE_LINE:
            guessing_table = True

        # A line pdfplumber has actually measured as table geometry (Step 1.7)
        # is trusted outright — no marker on it is structural, however it
        # happens to read. This is what section 206's two-column table needs:
        # "(2)" is the *enclosing* sequence's real next member, but it is also
        # the second column's row label, so letting the "resumes an open
        # sequence" guess below see it closes the table one row early.
        # Deliberately one-directional: a measured line only ever adds
        # suppression, it never resets ``guessing_table`` on the strength of
        # its *absence* — pdfplumber sometimes measures only part of a table
        # (section 39's ruled region covers 6 of its rows; the rest spill onto
        # the next page unmeasured), and closing early there reintroduces
        # duplicate-path anomalies in sections 1.6 already got right.
        measured = bool(in_table) and in_table[index]

        opened = 0
        if not measured:
            while (match := LEAD_MARKER.match(scan)) and opened < len(tokens[index]):
                marker = match.group(1)
                ahead = next_marker(tokens, index, opened)
                if guessing_table:
                    # Inside a table only a marker resuming an open sequence is
                    # structural; the rest is cell content. That also ends the
                    # table — section 19's runs 124 lines and closes on its own
                    # "(2)".
                    if not builder.resumes_open_sequence(marker):
                        break
                    guessing_table = False
                refusal = builder.place(marker, ahead, announced and not opened)
                if refusal is not None:
                    if not opened:
                        builder.note(index, marker, refusal, scan.strip())
                    break
                citation = builder.stack[-1].path.render()
                if citation in builder.seen:
                    builder.note(index, marker, AnomalyReason.DUPLICATE_PATH, citation)
                builder.seen.add(citation)
                scan = scan[match.end() :]
                opened += 1

        builder.attach(raw, pages[index] if pages else None)
        if raw.strip():
            previous = raw.rstrip()

    return builder.finish(), tuple(builder.anomalies)


def lead_markers(lines: list[str]) -> list[list[str]]:
    """Every line's run of leading markers, in document order.

    Read once up front because placement needs to see the *next* marker before
    deciding the current one (see ``_Builder.nests_instead``).
    """
    runs: list[list[str]] = []
    previous = ""
    for index, raw in enumerate(lines):
        scan = SECTION_HEAD.sub("", raw, count=1) if index == 0 else raw
        run: list[str] = []
        while (match := LEAD_MARKER.match(scan)) and kinds_for(match.group(1)) and not (
            REFERENCE_TAIL.search(previous) and not run
        ):
            run.append(match.group(1))
            scan = scan[match.end() :]
        runs.append(run)
        if raw.strip():
            previous = raw.rstrip()
    return runs


def next_marker(tokens: list[list[str]], index: int, position: int) -> str | None:
    if position + 1 < len(tokens[index]):
        return tokens[index][position + 1]
    return next((run[0] for run in tokens[index + 1 :] if run), None)


def candidate_table_pages(act: ParsedAct) -> tuple[int, ...]:
    """Pages independently known, from our own text, to carry a Table.

    Fed to :func:`taxverity.corpus.tables.find_table_regions` so pdfplumber is
    only ever pointed at a page we already have textual evidence for — never a
    blind scan, which Step 1.1 measured to false-positive on ordinary prose.
    """
    pages: set[int] = set()
    for marker in act.markers:
        section = act.section(marker)
        lines = section.text.split("\n") if section.text else []
        section_pages = act.section_pages.get(marker, ())
        pages.update(
            section_pages[index]
            for index, line in enumerate(lines)
            if line.strip() == TABLE_LINE
        )
    return tuple(sorted(pages))


def parse_substructure(
    act: ParsedAct, table_regions: tuple[TableRegion, ...] = ()
) -> Substructure:
    sections: list[StatutoryNode] = []
    anomalies: list[Anomaly] = []
    clause_rooted: list[str] = []
    table_sections: list[str] = []

    for section in act.sections:
        pages = act.section_pages.get(section.marker, ())
        tops = act.section_line_tops.get(section.marker, ())
        in_table = tuple(
            in_any_region(table_regions, page, top)
            for page, top in zip(pages, tops, strict=True)
        )
        grown, found = build(section, pages, in_table=in_table)
        sections.append(grown)
        anomalies.extend(found)
        if is_clause_rooted(section.text):
            clause_rooted.append(section.marker)
        if any(line.strip() == TABLE_LINE for line in section.text.split("\n")):
            table_sections.append(section.marker)

    result = Substructure(
        sections=tuple(sections),
        anomalies=tuple(anomalies),
        clause_rooted=tuple(clause_rooted),
        table_sections=tuple(table_sections),
    )
    logger.info(
        "grew %d nodes under %d sections (%d clause-rooted, %d carrying a table)",
        sum(1 for section in result.sections for _ in section.walk()),
        len(result.sections),
        len(result.clause_rooted),
        len(result.table_sections),
    )
    if result.anomalies:
        logger.warning(
            "%d anomalies, %d of them invalidating %d sections whose shape is not "
            "trusted: %s",
            len(result.anomalies),
            sum(1 for anomaly in result.anomalies if anomaly.invalidates_structure),
            len(result.unreliable),
            list(result.unreliable),
        )
    return result


def round_trip_failures(
    grown: Iterable[StatutoryNode], original: Iterable[StatutoryNode]
) -> list[str]:
    """Markers whose tree does not reproduce the section text it was built from."""
    source = {node.marker: node.text for node in original}
    return [node.marker for node in grown if node.full_text() != source[node.marker]]
