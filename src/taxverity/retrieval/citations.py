"""Step 3.5 -- a citation typed into a query is an address, not a bag of words.

BM25 splits ``22(2)`` into ``22`` and ``2`` (Step 3.4, note 7), so an exact
reference is the one query form lexical scoring is structurally bad at. This
routes it to a direct lookup instead.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence

from taxverity.chunking.models import Chunk
from taxverity.corpus.crossrefs import RefType, scan_references
from taxverity.corpus.nodes import NodePath
from taxverity.observability import get_logger
from taxverity.retrieval.base import ScoredChunk

logger = get_logger(__name__)

CITATION_STAGE_VERSION = 1

EXACT_SCORE = 1.0
ANCESTOR_SCORE = 0.5

# Users abbreviate where the Act never does. Each form requires a following
# digit, so an ordinary sentence ("Ms. 5 of the form", "sec of state") is not
# rewritten into a citation.
ABBREVIATIONS = re.compile(r"\b(?:u/s|s|ss|sec)\.?\s*(?=\d)", re.IGNORECASE)

# The corpus scanner routes "section 22 of the Income-tax Act" outside this
# corpus, and for statute prose that is right -- the 2025 Act quotes the 1961
# one by that name. In a *query* the same words mean the opposite: the user is
# naming the Act they are asking about. Own only when no other year is given.
OWN_ACT = re.compile(r"^(?:this act|income[-\s]?tax act)$", re.IGNORECASE)

# The scanner's captured name always stops at "Act", so the year that
# distinguishes this Act from the repealed 1961 one sits outside it. A query
# naming any other year of the Income-tax Act is asking about a corpus we do
# not hold, and the citation is dropped rather than answered from the wrong Act.
FOREIGN_YEAR = re.compile(r"income[-\s]?tax act,?\s*(?!2025)\d{4}", re.IGNORECASE)


def _is_own_act(name: str | None, query: str) -> bool:
    if name is None:
        return True
    if OWN_ACT.match(name.strip()) is None:
        return False
    return FOREIGN_YEAR.search(query) is None


def extract_query_citations(query: str) -> list[str]:
    """Citations the user typed, in order of appearance, no duplicates."""
    expanded = ABBREVIATIONS.sub("section ", query)
    citations: list[str] = []
    for scanned in scan_references(expanded):
        if scanned.ref_type not in (
            RefType.SECTION,
            RefType.SCHEDULE,
            RefType.SCHEDULE_PART,
            RefType.SCHEDULE_PARAGRAPH,
        ):
            continue
        if not _is_own_act(scanned.act_name, expanded):
            continue
        citations.extend(scanned.citations)
    return list(dict.fromkeys(citations))


class CitationRetriever:
    """Satisfies the Step 3.3 `Retriever` Protocol. Returns nothing at all for
    a query that names no provision -- it is a shortcut, not a search."""

    def __init__(self, chunks: Iterable[Chunk]) -> None:
        self._by_path = {chunk.node_path: chunk for chunk in chunks}

    def __len__(self) -> int:
        return len(self._by_path)

    def _lookup(self, citation: str) -> tuple[Chunk, float] | None:
        """Exact chunk, else the nearest ancestor that is one.

        ADR-056 prunes an untrusted subtree to its root, so a real citation
        inside section 206 names no chunk. Walking up returns the text that
        does contain it rather than nothing at all.
        """
        chunk = self._by_path.get(citation)
        if chunk is not None:
            return chunk, EXACT_SCORE
        try:
            path: NodePath | None = NodePath.parse(citation)
        except ValueError:
            return None
        while path is not None:
            chunk = self._by_path.get(path.render())
            if chunk is not None:
                return chunk, ANCESTOR_SCORE
            path = path.parent
        return None

    def search(self, query: str, k: int) -> Sequence[ScoredChunk]:
        if k < 1:
            raise ValueError(f"k must be at least 1, not {k}")
        results: list[ScoredChunk] = []
        seen: set[str] = set()
        for citation in extract_query_citations(query):
            found = self._lookup(citation)
            if found is None:
                continue
            chunk, score = found
            if chunk.chunk_id in seen:
                continue
            seen.add(chunk.chunk_id)
            results.append(ScoredChunk(chunk=chunk, score=score))
        results.sort(key=lambda result: -result.score)
        return results[:k]


class ShortcutRetriever:
    """A citation hit ahead of the lexical ranking, then the ranking itself.

    Its scores are ordinal only. A shortcut score and a BM25 score are not
    comparable (Step 3.3), so position -- not arithmetic on two scales -- is
    what decides the order here, and the emitted scores merely record it.
    """

    def __init__(self, shortcut: CitationRetriever, primary: object) -> None:
        self._shortcut = shortcut
        self._primary = primary

    def search(self, query: str, k: int) -> Sequence[ScoredChunk]:
        if k < 1:
            raise ValueError(f"k must be at least 1, not {k}")
        ordered: list[Chunk] = []
        seen: set[str] = set()
        for result in self._shortcut.search(query, k):
            ordered.append(result.chunk)
            seen.add(result.chunk.chunk_id)
        for result in self._primary.search(query, k):  # type: ignore[attr-defined]
            if result.chunk.chunk_id in seen:
                continue
            seen.add(result.chunk.chunk_id)
            ordered.append(result.chunk)
        return [
            ScoredChunk(chunk=chunk, score=float(len(ordered) - position))
            for position, chunk in enumerate(ordered[:k])
        ]
