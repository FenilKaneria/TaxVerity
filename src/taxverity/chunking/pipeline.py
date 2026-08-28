"""The corpus -> chunks pipeline, in one place.

Two scripts need it (Step 2.3's statistics report and Step 2.4's store build),
and a parser change that reached only one of them would give the two artifacts
different content under the same corpus_version.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import NamedTuple

from taxverity.chunking.chunker import build_chunks
from taxverity.chunking.models import Chunk
from taxverity.corpus.crossrefs import CrossReferenceIndex, extract_crossrefs
from taxverity.corpus.loader import read_pages_jsonl
from taxverity.corpus.schedules import FIRST_SCHEDULE_PAGE, parse_schedules
from taxverity.corpus.sections import parse as parse_sections
from taxverity.corpus.substructure import candidate_table_pages, parse_substructure
from taxverity.corpus.tables import find_table_regions
from taxverity.observability import get_logger

logger = get_logger(__name__)

UNVERSIONED = "0" * 64


class ChunkBuild(NamedTuple):
    chunks: tuple[Chunk, ...]
    untrusted: set[str]
    crossrefs: CrossReferenceIndex


def read_corpus_version(manifest: Path) -> str:
    if not manifest.exists():
        logger.warning("no %s — chunk ids in this run are not corpus-versioned", manifest)
        return UNVERSIONED
    return json.loads(manifest.read_text(encoding="utf-8"))["corpus_version"]


def build_from_corpus(pages_jsonl: Path, pdf: Path, corpus_version: str) -> ChunkBuild:
    pages = list(read_pages_jsonl(pages_jsonl))
    act = parse_sections(pages)
    sub = parse_substructure(
        act, table_regions=find_table_regions(pdf, candidate_table_pages(act))
    )
    schedules = parse_schedules(
        pages,
        table_regions=find_table_regions(pdf, range(FIRST_SCHEDULE_PAGE, len(pages))),
    )
    crossrefs = extract_crossrefs(sub.sections, schedules.schedules)
    untrusted = set(sub.unreliable) | set(schedules.unreliable)
    chunks = build_chunks(
        corpus_version,
        [*sub.sections, *schedules.schedules],
        chapters=act.chapters,
        crossrefs=crossrefs,
        untrusted=untrusted,
    )
    return ChunkBuild(chunks, untrusted, crossrefs)
