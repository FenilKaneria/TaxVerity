from __future__ import annotations

from collections.abc import Collection, Iterable, Iterator, Sequence

from taxverity.chunking.models import Chunk
from taxverity.corpus.crossrefs import CrossReferenceIndex
from taxverity.corpus.nodes import NodeType, StatutoryNode
from taxverity.corpus.sections import Chapter
from taxverity.observability import get_logger

logger = get_logger(__name__)

DOC_ID = "income-tax-act-2025"

CHUNKER_STAGE_VERSION = 1


def layout(node: StatutoryNode) -> tuple[str, dict[str, tuple[int, int]]]:
    """A subtree's text, plus where each descendant sits inside it.

    Offsets are computed by construction rather than by searching for the
    child's text in the parent's: identical clause wording repeats across the
    Act, and a search would silently attach the second occurrence to the first.
    The assembly mirrors ``StatutoryNode.full_text()`` exactly.
    """
    parts: list[str] = []
    spans: dict[str, tuple[int, int]] = {}
    cursor = 0
    if node.text:
        parts.append(node.text)
        cursor = len(node.text)
    for child in node.children:
        text, child_spans = layout(child)
        if not text:
            continue
        start = cursor + 1 if parts else 0
        parts.append(text)
        cursor = start + len(text)
        for citation, (child_start, child_end) in child_spans.items():
            spans[citation] = (child_start + start, child_end + start)
    whole = "\n".join(parts)
    if node.citation:
        spans[node.citation] = (0, len(whole))
    return whole, spans


def walk_with_parents(
    root: StatutoryNode, pruned: Collection[str] = ()
) -> Iterator[tuple[StatutoryNode, StatutoryNode | None]]:
    """Document order, parents first, stopping under any pruned citation."""
    stack = [(root, None)]
    while stack:
        node, parent = stack.pop()
        yield node, parent
        if node.citation in pruned:
            continue
        stack.extend((child, node) for child in reversed(node.children))


def subtree_refs(node: StatutoryNode, by_source: dict[str, list[str]]) -> tuple[str, ...]:
    """Every resolved reference made anywhere in this chunk's own text."""
    seen: dict[str, None] = {}
    for descendant in node.walk():
        if descendant.citation:
            seen.update(dict.fromkeys(by_source.get(descendant.citation, ())))
    return tuple(seen)


def terms_in(text: str, terms: Sequence[str]) -> tuple[str, ...]:
    lowered = text.lower()
    return tuple(term for term in terms if term.lower() in lowered)


def references_by_source(crossrefs: CrossReferenceIndex | None) -> dict[str, list[str]]:
    by_source: dict[str, list[str]] = {}
    if crossrefs is None:
        return by_source
    for reference in crossrefs.references:
        if reference.resolved and reference.target_path:
            targets = by_source.setdefault(reference.from_path, [])
            if reference.target_path not in targets:
                targets.append(reference.target_path)
    return by_source


def chunk_root(
    root: StatutoryNode,
    *,
    corpus_version: str,
    chapter_titles: dict[str, str | None],
    by_source: dict[str, list[str]],
    glossary: Sequence[str],
    untrusted: Collection[str] = (),
) -> list[Chunk]:
    if root.path is None or root.citation is None:
        raise ValueError(f"a chunk root needs a citation path: {root.marker!r}")

    root_text, spans = layout(root)
    is_schedule = root.type is NodeType.SCHEDULE
    root_fields = {
        "doc_id": DOC_ID,
        "root_title": root.title,
        "schedule_number" if is_schedule else "section_number": root.marker,
        "chapter_numeral": root.chapter,
        "chapter_title": chapter_titles.get(root.chapter) if root.chapter else None,
    }

    chunks: list[Chunk] = []
    ids: dict[str, str] = {}
    for node, parent in walk_with_parents(root, untrusted):
        citation = node.citation
        if citation is None or citation not in spans:
            continue
        start, end = spans[citation]
        text = root_text[start:end]
        if not text.strip():
            continue
        chunk = Chunk.create(
            corpus_version,
            citation,
            text,
            parent_id=ids.get(parent.citation) if parent is not None else None,
            node_type=node.type,
            page_start=min(node.pages) if node.pages else min(root.pages),
            page_end=max(node.pages) if node.pages else max(root.pages),
            char_start=start,
            char_end=end,
            outgoing_refs=subtree_refs(node, by_source),
            defined_terms=terms_in(text, glossary),
            **root_fields,
        )
        ids[citation] = chunk.chunk_id
        chunks.append(chunk)
    return chunks


def build_chunks(
    corpus_version: str,
    roots: Iterable[StatutoryNode],
    *,
    chapters: Sequence[Chapter] = (),
    crossrefs: CrossReferenceIndex | None = None,
    untrusted: Collection[str] = (),
) -> tuple[Chunk, ...]:
    """One chunk per structural node, parents before their own children.

    A root whose tree shape Step 1.6/1.7 could not trust yields its own chunk
    and nothing below it: no text is lost (the root carries the whole subtree),
    and no chunk is emitted under a node_path the parser itself flagged --
    including a duplicate path, which would otherwise put two different chunks
    at one citation for the Phase 10 verifier to choose between.
    """
    chapter_titles = {chapter.numeral: chapter.title for chapter in chapters}
    by_source = references_by_source(crossrefs)
    glossary = tuple(term.term for term in crossrefs.glossary) if crossrefs else ()

    chunks: list[Chunk] = []
    root_count = 0
    for root in roots:
        root_count += 1
        chunks.extend(
            chunk_root(
                root,
                corpus_version=corpus_version,
                chapter_titles=chapter_titles,
                by_source=by_source,
                glossary=glossary,
                untrusted=untrusted,
            )
        )

    logger.info(
        "chunked corpus: %d chunks from %d roots, %d subtrees left unsplit as untrusted",
        len(chunks),
        root_count,
        len(untrusted),
    )
    return tuple(chunks)
