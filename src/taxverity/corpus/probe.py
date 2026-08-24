import re
from collections import defaultdict
from collections.abc import Iterable

from pydantic import BaseModel, ConfigDict

from taxverity.corpus.loader import normalise
from taxverity.corpus.models import PageArtifact, Span

BODY_FONT_PREFIX = "LiberationSerif"

SECTION_NUMBER_SPAN = re.compile(r"^(\d{1,3})\.$")
SECTION_LINE_START = re.compile(r"^\s*(\d{1,3})\.\s+\S", re.MULTILINE)
CHAPTER_SPAN = re.compile(r"^CHAPTER\s+([IVXLCDM]+)$")
SCHEDULE_LINE = re.compile(r"^\s*SCHEDULE\s+([IVXLCDM]+)\s*$", re.MULTILINE)

TITLE_EDGE_CHARS = "[]{}() "


class HeadingCandidate(BaseModel):
    model_config = ConfigDict(frozen=True)

    number: int
    page: int
    title: str | None
    cues: tuple[str, ...]


class StructureProbe(BaseModel):
    model_config = ConfigDict(frozen=True)

    candidates: tuple[HeadingCandidate, ...]
    chapters: tuple[tuple[str, int], ...]
    schedules: tuple[tuple[str, int], ...]
    regex_only: tuple[tuple[int, int], ...]

    def numbers_by_cue(self, cue: str) -> set[int]:
        return {c.number for c in self.candidates if cue in c.cues}

    @property
    def all_numbers(self) -> set[int]:
        return {c.number for c in self.candidates}

    def duplicates(self) -> dict[int, list[int]]:
        pages = defaultdict(list)
        for candidate in self.candidates:
            pages[candidate.number].append(candidate.page)
        return {n: p for n, p in sorted(pages.items()) if len(p) > 1}

    def gaps(self, expected_high: int) -> list[int]:
        return [n for n in range(1, expected_high + 1) if n not in self.all_numbers]


def is_body_bold(span: Span) -> bool:
    return span.bold and not span.furniture and span.font.startswith(BODY_FONT_PREFIX)


def title_before(spans: list[Span], index: int) -> str | None:
    """The section title sits on its own bold line above the number (Step 1.1)."""
    parts: list[str] = []
    cursor = index - 1
    while cursor >= 0:
        span = spans[cursor]
        text = normalise(span.text).strip()
        if not is_body_bold(span) or SECTION_NUMBER_SPAN.match(text):
            break
        if CHAPTER_SPAN.match(text) or text.isupper() and len(text) > 3:
            break
        parts.append(text)
        cursor -= 1
    if not parts:
        return None
    title = " ".join(reversed(parts)).strip()
    title = title.strip(TITLE_EDGE_CHARS).strip()
    return title or None


def probe_page(
    artifact: PageArtifact,
) -> tuple[list[HeadingCandidate], list[str], list[str], set[int]]:
    spans = list(artifact.spans)
    bold_hits: list[HeadingCandidate] = []
    chapters: list[str] = []

    for index, span in enumerate(spans):
        text = normalise(span.text).strip()
        if not is_body_bold(span):
            continue
        chapter = CHAPTER_SPAN.match(text)
        if chapter:
            chapters.append(chapter.group(1))
            continue
        number = SECTION_NUMBER_SPAN.match(text)
        if number:
            bold_hits.append(
                HeadingCandidate(
                    number=int(number.group(1)),
                    page=artifact.page,
                    title=title_before(spans, index),
                    cues=("bold",),
                )
            )

    page_text = normalise(artifact.text)
    regex_numbers = {int(m) for m in SECTION_LINE_START.findall(page_text)}
    schedules = SCHEDULE_LINE.findall(page_text)
    return bold_hits, chapters, schedules, regex_numbers


def probe(artifacts: Iterable[PageArtifact]) -> StructureProbe:
    candidates: list[HeadingCandidate] = []
    chapters: list[tuple[str, int]] = []
    schedules: list[tuple[str, int]] = []
    regex_only: list[tuple[int, int]] = []

    for artifact in artifacts:
        bold_hits, page_chapters, page_schedules, regex_numbers = probe_page(artifact)
        bold_numbers = {hit.number for hit in bold_hits}

        for hit in bold_hits:
            cues = ("bold", "regex") if hit.number in regex_numbers else ("bold",)
            candidates.append(hit.model_copy(update={"cues": cues}))

        for number in sorted(regex_numbers - bold_numbers):
            regex_only.append((number, artifact.page))

        chapters.extend((numeral, artifact.page) for numeral in page_chapters)
        schedules.extend((numeral, artifact.page) for numeral in page_schedules)

    # A regex-only number is a heading candidate too, but a weaker one: it may
    # equally be a numbered list item or a table row. Kept separate so the
    # report can show which sections rest on the weaker cue alone.
    known = {c.number for c in candidates}
    for number, page in regex_only:
        if number not in known:
            candidates.append(
                HeadingCandidate(number=number, page=page, title=None, cues=("regex",))
            )
            known.add(number)

    return StructureProbe(
        candidates=tuple(candidates),
        chapters=tuple(chapters),
        schedules=tuple(schedules),
        regex_only=tuple(regex_only),
    )
