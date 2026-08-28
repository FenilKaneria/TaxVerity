from __future__ import annotations

import re
from collections.abc import Iterable

from pydantic import BaseModel, ConfigDict

from taxverity.corpus.loader import is_furniture
from taxverity.corpus.models import PageArtifact
from taxverity.corpus.nodes import (
    CITATION_DEPTH_TYPES,
    NodePath,
    NodeType,
    StatutoryNode,
)
from taxverity.corpus.sections import (
    BODY_SIZE_MIN,
    Footnote,
    Line,
    align_lines,
    footnote_start,
)
from taxverity.corpus.substructure import Anomaly, build
from taxverity.corpus.tables import TableRegion, in_any_region
from taxverity.observability import get_logger

logger = get_logger(__name__)

SCHEDULE_STAGE_VERSION = 1

# Confirmed by Step 1.1/1.3: the first Schedule opens at page 622 and the
# corpus ends at page 665 (666 pages, 0-indexed) with no content after
# Schedule XVI.
FIRST_SCHEDULE_PAGE = 622

SCHEDULE_LINE = re.compile(r"^SCHEDULE\s+([IVXLCDM]+)$")
PART_LINE = re.compile(r"^PART\s+([A-Z])$")
REFERENCE_LINE = re.compile(r"^\[See section .+\]$")

# The trailing period is sometimes absent (Schedule II row 33, Schedule VII
# row 33), exactly as sections.py found for section numbers, so it is
# optional here too. A single uppercase-letter suffix is real (Schedule
# III's 38B-38D, Schedule IV's 13B/13C/13E, both genuine amendment
# insertions). The lookahead requires the marker to be immediately followed
# by whitespace or end-of-line, which is what rejects "1st April" and a bare
# "20 or 21)," on its own line without needing the position/size checks
# below.
PARAGRAPH_LINE = re.compile(r"^(\d{1,3})([A-Z]?)\.?(?=\s|$)")

# Genuine paragraph/row markers sit at the Act's left text margin -- measured
# 27.6-46.4pt across every prose and list Schedule sampled. A nested
# sub-table's own row numbers (Schedule II paragraph 2's inner "Sl. No."
# column, x0=239.6) and a wrapped body line that happens to open with a digit
# (Schedule XV's "20 or 21)," continuation, x0=55.2) both sit well beyond it.
# This one measured cutoff is what lets a single rule serve the bold "prose"
# family and the non-bold "list" family without knowing in advance which one
# a given Schedule is.
MARKER_LEFT_MARGIN_MAX = 50.0

# Paragraph numbering is contiguous almost everywhere, but Schedule XII Part A
# genuinely omits 28 -- confirmed by a direct search of the text, not a
# parsing artefact. Tolerating a gap of exactly one crosses that without
# accepting a wild jump: a wrapped "20 or 21)," read as a candidate is +19
# from paragraph 1 and must still be rejected.
MAX_NUMBERING_GAP = 2

_MARKER_KEY = re.compile(r"^(\d+)([A-Z]?)$")


def marker_key(marker: str) -> tuple[int, str]:
    match = _MARKER_KEY.match(marker)
    if not match:
        raise ValueError(f"not a schedule paragraph marker: {marker!r}")
    return int(match.group(1)), match.group(2)


def is_plausible_next(current: str | None, candidate: str) -> bool:
    """Whether ``candidate`` may open the next paragraph after ``current``.

    ``current is None`` means nothing has opened yet in this Schedule or Part,
    so only "1" may open. Otherwise the candidate must land within
    ``MAX_NUMBERING_GAP`` integers of the current one, or be a fresh
    letter-suffix branch off the same integer (38 -> 38B is real: Schedule
    III's 38A appears to have been omitted in the same amendment that
    inserted 38B-38D).
    """
    number, suffix = marker_key(candidate)
    if current is None:
        return number == 1 and not suffix
    current_number, current_suffix = marker_key(current)
    if number == current_number:
        return bool(suffix) and suffix > current_suffix
    return current_number < number <= current_number + MAX_NUMBERING_GAP


def paragraph_marker(line: Line) -> str | None:
    """The marker this line opens, or ``None`` if it is not a genuine one.

    Deliberately agnostic to boldness -- the "prose" and "list" Schedule
    families differ only in whether the opening line is bold, never in its
    position or size, so the marker's own left margin and font size are what
    generalise across both.
    """
    match = PARAGRAPH_LINE.match(line.content)
    if not match or not line.spans:
        return None
    head = line.spans[0]
    if head.furniture or head.size < BODY_SIZE_MIN:
        return None
    if head.bbox[0] > MARKER_LEFT_MARGIN_MAX:
        return None
    return f"{match.group(1)}{match.group(2)}"


def is_caption(line: Line) -> bool:
    """SCHEDULE and PART headings share one bold, all-uppercase caption style."""
    return bool(line.spans) and line.is_bold and line.content.isupper()


def schedule_preamble(
    lines: tuple[Line, ...], index: int
) -> tuple[str | None, list[str], int]:
    """The Schedule's caption and any reference line before paragraph 1.

    A "[See section ...]" cross-reference line, when present, sits between
    the heading and the caption. It is not discarded -- verbatim text is
    never dropped -- it is folded into the returned preamble instead.
    """
    cursor = index + 1
    preamble: list[str] = []
    if cursor < len(lines) and REFERENCE_LINE.match(lines[cursor].content):
        preamble.append(lines[cursor].text)
        cursor += 1
    parts: list[str] = []
    while (
        cursor < len(lines)
        and is_caption(lines[cursor])
        and not PART_LINE.match(lines[cursor].content)
    ):
        parts.append(lines[cursor].content)
        cursor += 1
    title = " ".join(parts) or None
    return title, preamble, cursor


def scan_boundaries(
    lines: tuple[Line, ...],
    numeral: str | None,
    local_marker: str | None,
) -> dict[int, str]:
    """Line indices that open a Schedule, Part, or paragraph on this page.

    A side-effect-free replay of the same sequential rules the real pass
    below applies, needed only to give ``footnote_start`` an honest
    ``starts`` dict. Without it, ``footnote_start``'s own "still inside the
    apparatus from the previous page" shortcut fires on *any* page that
    merely mentions amendment vocabulary somewhere -- which is nearly every
    Schedule page -- and swallows real content wholesale. A first version of
    this parser passed an empty dict here and lost the whole of Schedule XII
    to exactly that.
    """
    starts: dict[int, str] = {}
    for index, line in enumerate(lines):
        if is_furniture(line.text):
            continue
        heading = SCHEDULE_LINE.match(line.content)
        if heading and is_caption(line):
            numeral, local_marker = heading.group(1), None
            starts[index] = line.content
            continue
        if numeral is None:
            continue
        if PART_LINE.match(line.content) and is_caption(line):
            local_marker = None
            starts[index] = line.content
            continue
        marker = paragraph_marker(line)
        if marker is not None and is_plausible_next(local_marker, marker):
            local_marker = marker
            starts[index] = marker
    return starts


class ParsedSchedules(BaseModel):
    model_config = ConfigDict(frozen=True)

    schedules: tuple[StatutoryNode, ...]
    anomalies: tuple[Anomaly, ...]
    footnotes: tuple[Footnote, ...]
    # SCHEDULE and PART heading lines, verbatim -- their numeral is already
    # captured structurally (a Schedule's marker, a paragraph's part-prefixed
    # marker), but the raw caption line itself is not otherwise kept anywhere,
    # exactly the gap ParsedAct.chapter_headings closes for sections.
    headings: tuple[str, ...] = ()

    def schedule(self, numeral: str) -> StatutoryNode | None:
        for node in self.schedules:
            if node.marker == numeral:
                return node
        return None

    @property
    def unreliable(self) -> tuple[str, ...]:
        return tuple(sorted({anomaly.section for anomaly in self.anomalies}))


class _ScheduleState:
    """Everything reset when a new SCHEDULE heading is seen."""

    def __init__(self) -> None:
        self.numeral: str | None = None
        self.title: str | None = None
        self.preamble: list[str] = []
        self.part: str | None = None
        self.local_marker: str | None = None
        self.paragraphs: list[StatutoryNode] = []
        self.body: list[str] = []
        self.pages: list[int] = []
        self.tops: list[float | None] = []


def parse_schedules(
    artifacts: Iterable[PageArtifact], table_regions: tuple[TableRegion, ...] = ()
) -> ParsedSchedules:
    schedules: list[StatutoryNode] = []
    footnotes: list[Footnote] = []
    anomalies: list[Anomaly] = []
    headings: list[str] = []
    state = _ScheduleState()
    continuing = False

    def close_paragraph() -> None:
        if state.local_marker is None:
            return
        marker = f"{state.part or ''}{state.local_marker}"
        node = StatutoryNode(
            type=NodeType.SCHEDULE_PARAGRAPH,
            marker=marker,
            path=NodePath.schedule(state.numeral).child(NodeType.SCHEDULE_PARAGRAPH, marker),
            text="\n".join(state.body),
            pages=tuple(dict.fromkeys(state.pages)),
        )
        in_table = tuple(
            in_any_region(table_regions, page, top)
            for page, top in zip(state.pages, state.tops, strict=True)
        )
        grown, found = build(
            node,
            tuple(state.pages),
            level_types=CITATION_DEPTH_TYPES[NodeType.SCHEDULE][1:],
            in_table=in_table,
            depth_shift=0,
        )
        state.paragraphs.append(grown)
        anomalies.extend(found)
        state.local_marker = None
        state.body = []
        state.pages = []
        state.tops = []

    def close_schedule() -> None:
        if state.numeral is None:
            return
        close_paragraph()
        schedules.append(
            StatutoryNode(
                type=NodeType.SCHEDULE,
                marker=state.numeral,
                path=NodePath.schedule(state.numeral),
                title=state.title,
                text="\n".join(state.preamble),
                pages=tuple(
                    dict.fromkeys(page for para in state.paragraphs for page in para.pages)
                ),
                children=tuple(state.paragraphs),
            )
        )

    for artifact in artifacts:
        if artifact.page < FIRST_SCHEDULE_PAGE:
            continue
        lines = align_lines(artifact)
        starts = scan_boundaries(lines, state.numeral, state.local_marker)
        footnotes_from = footnote_start(lines, starts, continuing)
        continuing = footnotes_from is not None

        index = 0
        while index < len(lines):
            line = lines[index]

            if is_furniture(line.text):
                index += 1
                continue

            if footnotes_from is not None and index >= footnotes_from:
                footnotes.append(Footnote(page=artifact.page, text=line.text))
                index += 1
                continue

            heading = SCHEDULE_LINE.match(line.content)
            if heading and is_caption(line):
                close_schedule()
                state = _ScheduleState()
                state.numeral = heading.group(1)
                headings.append(line.text)
                state.title, preamble, index = schedule_preamble(lines, index)
                state.preamble.extend(preamble)
                continue

            if state.numeral is None:
                index += 1
                continue

            part = PART_LINE.match(line.content)
            if part and is_caption(line):
                state.part = part.group(1)
                state.local_marker = None
                headings.append(line.text)
                index += 1
                continue

            marker = paragraph_marker(line)
            if marker is not None and is_plausible_next(state.local_marker, marker):
                close_paragraph()
                state.local_marker = marker
                state.body.append(line.text)
                state.pages.append(line.page)
                state.tops.append(line.top)
                index += 1
                continue

            if state.local_marker is None:
                state.preamble.append(line.text)
            else:
                state.body.append(line.text)
                state.pages.append(line.page)
                state.tops.append(line.top)
            index += 1

    close_schedule()

    result = ParsedSchedules(
        schedules=tuple(schedules),
        anomalies=tuple(anomalies),
        footnotes=tuple(footnotes),
        headings=tuple(headings),
    )
    logger.info(
        "parsed %d schedules, %d nodes, %d footnote lines separated",
        len(result.schedules),
        sum(1 for schedule in result.schedules for _ in schedule.walk()),
        len(result.footnotes),
    )
    if result.anomalies:
        logger.warning(
            "%d unplaceable markers in %d paragraphs, whose shape is not trusted: %s",
            len(result.anomalies),
            len(result.unreliable),
            list(result.unreliable),
        )
    return result
