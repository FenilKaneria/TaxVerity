"""Step 5.8 — the statutory-term bridge (ADR-037, ADR-086): a question's
everyday words mapped to the Act's own, appended before retrieval.

The Act writes its own vocabulary down, in the section 2 glossary and in its
section titles, so the statutory side of the map is read off the corpus and
checked against it. The lay side exists nowhere in the Act and is hand-written
in `term_bridge_v1.json`. Deterministic, no model call.
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from taxverity.chunking.models import Chunk
from taxverity.corpus.crossrefs import defined_term
from taxverity.corpus.nodes import NodePath
from taxverity.observability import get_logger
from taxverity.retrieval.base import Retriever, ScoredChunk
from taxverity.retrieval.bm25 import tokenize
from taxverity.retrieval.citations import extract_query_citations

logger = get_logger(__name__)

BRIDGE_STAGE_VERSION = 1
BRIDGE_MAP_VERSION = 1
BRIDGE_MAP = Path(__file__).with_name("term_bridge_v1.json")

# A term found in more of the Act's roots than this cannot steer retrieval
# anywhere: "tax" is in 79% of them, "income" 77%, "interest" 22%. Set in the
# gap between 0.112 ("notification") and 0.146 ("resident") in the section 2
# glossary's own distribution, measured over the corpus before any gold run.
MAX_ROOT_SHARE = 0.125


class BridgeMapError(ValueError):
    """The map disagrees with the corpus it is meant to bridge into."""


class BridgeEntry(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    statutory: str
    # The chunk whose text carries `statutory`: a section 2 clause, or a root
    # whose title does. The evidence that the target is the Act's own wording.
    source: str
    lay: tuple[str, ...]


class BridgeMap(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    version: int
    entries: tuple[BridgeEntry, ...]


def load_bridge_map(path: Path = BRIDGE_MAP) -> tuple[BridgeEntry, ...]:
    loaded = BridgeMap.model_validate_json(path.read_text(encoding="utf-8"))
    if loaded.version != BRIDGE_MAP_VERSION:
        raise BridgeMapError(f"{path} is map version {loaded.version}, not {BRIDGE_MAP_VERSION}")
    return loaded.entries


def terms(text: str) -> tuple[str, ...]:
    """BM25's tokens with a trailing plural `s` folded off, so "instalments"
    meets "instalment". Both sides fold the same way, so a mangled stem
    ("taxe") still meets itself."""
    return tuple(
        token[:-1] if len(token) > 3 and token.endswith("s") and not token.endswith("ss") else token
        for token in tokenize(text)
    )


def find(words: Sequence[str], phrase: Sequence[str]) -> int | None:
    width = len(phrase)
    for start in range(len(words) - width + 1):
        if tuple(words[start : start + width]) == tuple(phrase):
            return start
    return None


class TermBridge:
    def __init__(self, entries: Iterable[BridgeEntry], chunks: Iterable[Chunk]) -> None:
        started = time.perf_counter()
        chunks = tuple(chunks)
        by_path = {chunk.node_path: chunk for chunk in chunks}
        self._roots = [terms(chunk.embed_text()) for chunk in chunks if chunk.parent_id is None]
        self._root_sets = [frozenset(root) for root in self._roots]

        self._lay: dict[tuple[str, ...], str] = {}
        self._targets: dict[str, tuple[str, ...]] = {}
        for entry in entries:
            target = terms(entry.statutory)
            source = by_path.get(entry.source)
            if not target or entry.statutory in self._targets:
                raise BridgeMapError(f"{entry.statutory!r}: empty or mapped twice")
            if source is None:
                raise BridgeMapError(f"{entry.statutory!r}: source {entry.source} names no chunk")
            if find(terms(source.embed_text()), target) is None:
                raise BridgeMapError(f"{entry.statutory!r} does not appear in {entry.source}")
            share = self.root_share(entry.statutory)
            if share > MAX_ROOT_SHARE:
                raise BridgeMapError(
                    f"{entry.statutory!r} is in {share:.3f} of roots, over {MAX_ROOT_SHARE}"
                )
            self._targets[entry.statutory] = target
            for phrase in entry.lay:
                key = terms(phrase)
                if not key or key in self._lay:
                    raise BridgeMapError(f"lay phrase {phrase!r}: empty or mapped twice")
                if find(key, target) is not None:
                    raise BridgeMapError(f"lay phrase {phrase!r} already says {entry.statutory!r}")
                self._lay[key] = entry.statutory

        # The definitions pull: section 2 clauses whose term is specific enough
        # to be worth delivering when a question uses it.
        self._glossary: list[tuple[tuple[str, ...], Chunk]] = []
        for chunk in chunks:
            path = NodePath.parse(chunk.node_path)
            if chunk.section_number != "2" or path.depth != 2:
                continue
            term = defined_term(chunk.text)
            if term is not None and self.root_share(term) <= MAX_ROOT_SHARE:
                self._glossary.append((terms(term), chunk))
        logger.info(
            "term bridge: %d statutory terms, %d lay phrases, %d pullable definitions, in %.2fs",
            len(self._targets),
            len(self._lay),
            len(self._glossary),
            time.perf_counter() - started,
        )

    @property
    def pullable(self) -> tuple[str, ...]:
        """The section 2 clauses the definitions pull may deliver."""
        return tuple(chunk.node_path for _, chunk in self._glossary)

    def root_share(self, phrase: str) -> float:
        want = terms(phrase)
        hits = sum(
            1
            for root, present in zip(self._roots, self._root_sets, strict=True)
            if present.issuperset(want) and find(root, want) is not None
        )
        return hits / len(self._roots)

    def expand(self, query: str) -> tuple[str, ...]:
        """The statutory terms the question's everyday words stand for, in the
        order the question uses them. An already-statutory question is left as
        it is: a term it already says is not added again, and a question that
        names a provision is not rewritten at all, since the citation shortcut
        answers it by citation (ADR-086)."""
        if extract_query_citations(query):
            return ()
        words = terms(query)
        found: dict[str, int] = {}
        for phrase, statutory in self._lay.items():
            at = find(words, phrase)
            if at is None or find(words, self._targets[statutory]) is not None:
                continue
            found[statutory] = min(at, found.get(statutory, at))
        return tuple(sorted(found, key=lambda statutory: found[statutory]))

    def rewrite(self, query: str) -> str:
        added = self.expand(query)
        return f"{query} ({'; '.join(added)})" if added else query

    def definitions(self, query: str) -> tuple[Chunk, ...]:
        """The section 2 clauses defining a specific term the question uses, in
        the order it uses them. Pass the rewritten question, so a term the
        bridge added is defined too."""
        words = terms(query)
        placed = [(find(words, term), chunk) for term, chunk in self._glossary]
        return tuple(chunk for at, chunk in sorted(
            ((at, chunk) for at, chunk in placed if at is not None), key=lambda pair: pair[0]
        ))


class BridgedRetriever:
    """Satisfies the Step 3.3 `Retriever` Protocol.

    Sits inside the citation shortcut: a cited provision is found by its
    citation, and the bridge rewrites only what fusion and the reranker see.
    """

    def __init__(self, bridge: TermBridge, inner: Retriever) -> None:
        self._bridge = bridge
        self._inner = inner

    def search(self, query: str, k: int) -> Sequence[ScoredChunk]:
        return self._inner.search(self._bridge.rewrite(query), k)
