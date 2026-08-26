from __future__ import annotations

import re
from collections.abc import Iterable
from enum import StrEnum

from pydantic import BaseModel, ConfigDict

from taxverity.corpus.loader import is_furniture, normalise
from taxverity.corpus.models import PageArtifact, Span
from taxverity.corpus.nodes import NodePath, NodeType, StatutoryNode

PARSE_STAGE_VERSION = 1

# Step 1.3 measured the first Schedule at page 622. Schedules restart their
# paragraph numbering at 1 with typography identical to sections, so a bare
# "<n>." cue cannot separate Schedule X paragraph 1 from section 1. Section
# parsing is therefore bounded to the pages before this one.
FIRST_SCHEDULE_PAGE = 622

BODY_FONT_PREFIX = "LiberationSerif"
BODY_SIZE_MIN = 8.0

# The number may carry a letter suffix (354A) and the period is sometimes set
# off from it by a space ("94 . (1)"), so the separator is not fixed width.
SECTION_LINE = re.compile(r"^(\d{1,3})([A-Z]{0,3})\s*\.")
# The trailing period is optional: on many pages the bold run stops at the
# digits and the period falls into the following roman span ("67" then ". (1)").
BOLD_MARKER = re.compile(r"^(\d{1,3})([A-Z]{0,3})\s*\.?$")
CHAPTER_LINE = re.compile(r"^CHAPTER\s+([IVXLCDM]+)$")
# A division heading ("3. —Representative assessees") shares this shape, but it
# always introduces a section on the same page, which is what separates the two.
DIVISION_LINE = re.compile(r"^(?:[A-Z]{1,2}|\d{1,3})\s*\.\s*—")
FOOTNOTE_LINE = re.compile(r"^\d{1,3}[a-z]?\.\s+\S")
AMENDMENT = re.compile(r"w\.e\.f\.|by (?:the )?Act No\.")

# The amendment apparatus sits below a rule that opens a ~63pt gap; ordinary
# line spacing is 11-13pt and paragraph spacing ~17pt. Without this the same
# "<n>. <text>" shape matches ordinary numbered table rows (pages 32, 455-457).
FOOTNOTE_GAP_MIN = 30.0

WHITESPACE = re.compile(r"\s+")
TITLE_EDGE_CHARS = "[]{}() "


class LineRole(StrEnum):
    FRONT_MATTER = "front_matter"
    CHAPTER_HEADING = "chapter_heading"
    DIVISION_HEADING = "division_heading"
    SECTION_TITLE = "section_title"
    SECTION_BODY = "section_body"
    FOOTNOTE = "footnote"
    FURNITURE = "furniture"


class LineAlignmentError(RuntimeError):
    pass


class Line(BaseModel):
    model_config = ConfigDict(frozen=True)

    page: int
    index: int
    text: str
    spans: tuple[Span, ...]

    @property
    def content(self) -> str:
        return normalise(self.text).strip()

    @property
    def is_bold(self) -> bool:
        printing = [span for span in self.spans if squash(span.text)]
        return bool(printing) and all(is_body_bold(span) for span in printing)

    @property
    def top(self) -> float | None:
        return min((span.bbox[1] for span in self.spans), default=None)

    @property
    def bottom(self) -> float | None:
        return max((span.bbox[3] for span in self.spans), default=None)


class Chapter(BaseModel):
    model_config = ConfigDict(frozen=True)

    numeral: str
    title: str | None
    page: int


class Footnote(BaseModel):
    model_config = ConfigDict(frozen=True)

    page: int
    text: str


class ParsedAct(BaseModel):
    model_config = ConfigDict(frozen=True)

    sections: tuple[StatutoryNode, ...]
    chapters: tuple[Chapter, ...]
    footnotes: tuple[Footnote, ...]
    front_matter: tuple[Line, ...]
    divisions: tuple[Line, ...]
    chapter_headings: tuple[Line, ...]
    titles: tuple[Line, ...]
    body_lines: int
    # Page of each body line, parallel to the lines of ``section.text``. Step 1.6
    # builds children out of those lines and needs their provenance for ADR-016.
    section_pages: dict[str, tuple[int, ...]] = {}
    # Vertical position of each body line, parallel to ``section_pages``. Step
    # 1.7 needs this to test a line against pdfplumber's table geometry — a
    # line's marker is real substructure only outside any measured table bbox.
    section_line_tops: dict[str, tuple[float | None, ...]] = {}

    def attributed_lines(self) -> int:
        """Every non-furniture line lands in exactly one of these buckets."""
        return (
            sum(len(node.text.split("\n")) for node in self.sections if node.text)
            + len(self.footnotes)
            + len(self.front_matter)
            + len(self.divisions)
            + len(self.chapter_headings)
            + len(self.titles)
        )

    def section(self, marker: str) -> StatutoryNode | None:
        for node in self.sections:
            if node.marker == marker:
                return node
        return None

    @property
    def markers(self) -> tuple[str, ...]:
        return tuple(node.marker for node in self.sections)

    def gaps(self) -> list[int]:
        seen = {int(SECTION_LINE.match(m + ".").group(1)) for m in self.markers}
        return [n for n in range(1, max(seen) + 1) if n not in seen]


def squash(text: str) -> str:
    return WHITESPACE.sub("", normalise(text))


def is_body_bold(span: Span) -> bool:
    return (
        span.bold
        and not span.furniture
        and span.size >= BODY_SIZE_MIN
        and span.font.startswith(BODY_FONT_PREFIX)
    )


def align_lines(artifact: PageArtifact) -> tuple[Line, ...]:
    """Map each text line onto the spans that produced it, in document order.

    ``get_text("text")`` inserts separators that the span texts do not carry, so
    the two only agree once whitespace is removed. Spans are consumed greedily
    in order, which is exact because both views come from the same layout pass.
    """
    spans = list(artifact.spans)
    cursor = 0
    lines: list[Line] = []
    for index, text in enumerate(
        line for line in artifact.text.split("\n") if line.strip()
    ):
        wanted = squash(text)
        seen = ""
        used: list[Span] = []
        while cursor < len(spans) and len(seen) < len(wanted):
            seen += squash(spans[cursor].text)
            used.append(spans[cursor])
            cursor += 1
        if seen != wanted:
            raise LineAlignmentError(
                f"page {artifact.page} line {index}: {seen!r} != {wanted!r}"
            )
        lines.append(
            Line(page=artifact.page, index=index, text=text, spans=tuple(used))
        )
    return tuple(lines)


def section_starts(lines: Iterable[Line]) -> dict[int, str]:
    """Line indices that open a section, keyed to the section's marker.

    Both cues must agree. The line-start regex alone matches amendment
    footnotes, which run their own 1, 2, 3 sequence; the bold marker alone
    matches the small footnote reference digits at the page foot. Requiring the
    line to *begin* with a body-bold span carrying the same number rejects both.
    """
    starts: dict[int, str] = {}
    for line in lines:
        match = SECTION_LINE.match(line.content)
        if not match or not line.spans:
            continue
        head = line.spans[0]
        if not is_body_bold(head):
            continue
        marker = BOLD_MARKER.match(normalise(head.text).strip())
        if marker and marker.group(0).rstrip(". ") == match.group(0).rstrip(". "):
            starts[line.index] = f"{match.group(1)}{match.group(2)}"
    return starts


def title_above(lines: tuple[Line, ...], start: int) -> tuple[str | None, int]:
    """The section title is the run of bold lines directly above the number."""
    parts: list[str] = []
    cursor = start - 1
    while cursor >= 0:
        line = lines[cursor]
        if not line.is_bold or CHAPTER_LINE.match(line.content):
            break
        if line.content.isupper() and len(line.content) > 3:
            break
        parts.append(line.content)
        cursor -= 1
    title = " ".join(reversed(parts)).strip().strip(TITLE_EDGE_CHARS).strip()
    return (title or None), cursor + 1


def footnote_start(
    lines: tuple[Line, ...], starts: dict[int, str], continuing: bool
) -> int | None:
    """Where the amendment apparatus begins, or None if this page carries none.

    The apparatus is always the trailing run of a page. A quoted repeal can spill
    onto the next page above that page's own first footnote marker, which is why
    a page following an apparatus block and opening no section is read as a
    continuation.
    """
    body = [line for line in lines if not is_furniture(line.text)]
    if not body:
        return None
    if continuing and not starts and AMENDMENT.search(page_text(body)):
        return body[0].index
    for position, line in enumerate(body):
        if not FOOTNOTE_LINE.match(line.content):
            continue
        if any(index >= line.index for index in starts):
            continue
        if not AMENDMENT.search(page_text(body[position:])):
            continue
        if line.index != 0 and not below_rule(body[:position], line):
            continue
        return line.index
    return None


def page_text(lines: Iterable[Line]) -> str:
    return "\n".join(normalise(line.text) for line in lines)


def below_rule(above: Iterable[Line], line: Line) -> bool:
    top = line.top
    if top is None:
        return False
    bottoms = [
        other.bottom
        for other in above
        if other.bottom is not None and other.bottom <= top - 1
    ]
    return bool(bottoms) and top - max(bottoms) > FOOTNOTE_GAP_MIN


def chapter_at(lines: tuple[Line, ...], index: int) -> tuple[str, str | None, int]:
    """The numeral, its title from the bold upper-case lines below it, and the span."""
    numeral = CHAPTER_LINE.match(lines[index].content).group(1)
    parts: list[str] = []
    cursor = index + 1
    while cursor < len(lines):
        line = lines[cursor]
        if not line.is_bold or not line.content.isupper():
            break
        parts.append(line.content)
        cursor += 1
    return numeral, " ".join(parts) or None, cursor - index


def classify_page(
    lines: tuple[Line, ...], continuing: bool
) -> tuple[dict[int, LineRole], dict[int, str], list[tuple[str, str | None]]]:
    starts = section_starts(lines)
    roles: dict[int, LineRole] = {}
    chapters: list[tuple[str, str | None]] = []

    footnotes_from = footnote_start(lines, starts, continuing)
    for line in lines:
        if is_furniture(line.text):
            roles[line.index] = LineRole.FURNITURE
        elif footnotes_from is not None and line.index >= footnotes_from:
            roles[line.index] = LineRole.FOOTNOTE

    for line in lines:
        if line.index in roles:
            continue
        if CHAPTER_LINE.match(line.content) and line.is_bold:
            numeral, title, span = chapter_at(lines, line.index)
            chapters.append((numeral, title))
            for cursor in range(line.index, line.index + span):
                roles[cursor] = LineRole.CHAPTER_HEADING
        elif DIVISION_LINE.match(line.content) and not line.is_bold:
            roles[line.index] = LineRole.DIVISION_HEADING

    for index in starts:
        _, title_from = title_above(lines, index)
        for cursor in range(title_from, index):
            if roles.get(cursor) in (None, LineRole.DIVISION_HEADING):
                roles[cursor] = LineRole.SECTION_TITLE

    return roles, starts, chapters


def parse(artifacts: Iterable[PageArtifact]) -> ParsedAct:
    sections: list[StatutoryNode] = []
    chapters: list[Chapter] = []
    footnotes: list[Footnote] = []
    section_pages: dict[str, tuple[int, ...]] = {}
    section_line_tops: dict[str, tuple[float | None, ...]] = {}
    front_matter: list[Line] = []
    divisions: list[Line] = []
    chapter_headings: list[Line] = []
    titles: list[Line] = []
    body_lines = 0

    open_marker: str | None = None
    open_title: str | None = None
    open_chapter: str | None = None
    body: list[str] = []
    pages: list[int] = []
    tops: list[float | None] = []
    continuing = False

    def close() -> None:
        if open_marker is None:
            return
        section_pages[open_marker] = tuple(pages)
        section_line_tops[open_marker] = tuple(tops)
        sections.append(
            StatutoryNode(
                type=NodeType.SECTION,
                marker=open_marker,
                path=NodePath.section(open_marker),
                title=open_title,
                text="\n".join(body),
                chapter=open_chapter,
                pages=tuple(dict.fromkeys(pages)),
            )
        )

    for artifact in artifacts:
        if artifact.page >= FIRST_SCHEDULE_PAGE:
            break
        lines = align_lines(artifact)
        roles, starts, page_chapters = classify_page(lines, continuing)
        continuing = LineRole.FOOTNOTE in roles.values()

        for numeral, title in page_chapters:
            chapters.append(Chapter(numeral=numeral, title=title, page=artifact.page))

        for line in lines:
            role = roles.get(line.index, LineRole.SECTION_BODY)
            if line.index in starts:
                close()
                open_marker = starts[line.index]
                open_title, _ = title_above(lines, line.index)
                open_chapter = chapters[-1].numeral if chapters else None
                body = []
                pages = []
                tops = []
                role = LineRole.SECTION_BODY
            if role is LineRole.FURNITURE:
                continue
            body_lines += 1
            if role is LineRole.FOOTNOTE:
                footnotes.append(Footnote(page=artifact.page, text=line.text))
                continue
            if role is LineRole.DIVISION_HEADING:
                divisions.append(line)
                continue
            if role is LineRole.CHAPTER_HEADING:
                chapter_headings.append(line)
                continue
            if role is LineRole.SECTION_TITLE:
                titles.append(line)
                continue
            if open_marker is None:
                front_matter.append(line)
                continue
            body.append(line.text)
            pages.append(line.page)
            tops.append(line.top)

    close()
    return ParsedAct(
        sections=tuple(sections),
        chapters=tuple(chapters),
        footnotes=tuple(footnotes),
        front_matter=tuple(front_matter),
        divisions=tuple(divisions),
        chapter_headings=tuple(chapter_headings),
        titles=tuple(titles),
        body_lines=body_lines,
        section_pages=section_pages,
        section_line_tops=section_line_tops,
    )
