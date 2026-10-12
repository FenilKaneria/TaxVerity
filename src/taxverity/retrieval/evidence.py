"""What the generator is handed: the fused ranking turned into distinct evidence within
a token budget, plus the provisions that evidence refers to, one hop out."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Sequence
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, model_validator

from taxverity.chunking.models import Chunk
from taxverity.chunking.stats import estimate_tokens
from taxverity.corpus.nodes import NodePath
from taxverity.retrieval.base import ScoredChunk
from taxverity.retrieval.bm25 import tokenize

EVIDENCE_STAGE_VERSION = 4

EVIDENCE_BUDGET = 4_000

# How deep delivery reads the ranking. Deduplication collapses a top 10 to about
# four distinct units, so 20 is what fills the budget, and it is the pool Step
# 5.6's reranker is budgeted for.
EVIDENCE_POOL = 20

# The k the hybrid ranking was adopted at. Hits above it are packed
# before any referenced text, hits below it after, so expansion can never
# displace evidence the adopted ranking itself delivered.
EXPANSION_HEAD = 10


# An excerpt never takes more than this share of the budget.
EXCERPT_SHARE = 0.5
EXCERPT_HEAD = 5
EXCERPT_RESERVE = 1 / 3
EXCERPT_WINDOW = 150
# Less room than this is not worth a partial reading of a section: it is the
# opening run and one more.
EXCERPT_MINIMUM = 300
# Marks lines left out. It is the only text in an excerpt that is not the
# chunk's own, and it carries no word or figure a claim could ground in.
ELISION = "[...]"


def windows(text: str, size: int = EXCERPT_WINDOW) -> list[str]:
    """Whole lines, in order, grouped into runs of about `size` tokens."""
    runs: list[str] = []
    current: list[str] = []
    held = 0
    for line in text.split("\n"):
        current.append(line)
        held += estimate_tokens(line)
        if held >= size:
            runs.append("\n".join(current))
            current, held = [], 0
    if current:
        runs.append("\n".join(current))
    return runs


def excerpt(text: str, query: str, room: int) -> str | None:
    """The opening run, then the runs sharing the most (and rarest) words with the
    query, in document order, within `room` tokens."""
    runs = windows(text)
    terms = set(tokenize(query))
    bags = [set(tokenize(run)) & terms for run in runs]
    spread = Counter(term for bag in bags for term in bag)
    bags = [{term for term in bag if spread[term] * 2 <= len(runs)} for bag in bags]

    def score(index: int) -> float:
        return sum(math.log(1 + len(runs) / spread[term]) for term in bags[index])

    # Each run is charged one elision mark: at most one precedes each chosen
    # run, and the opening run's pays for the mark that may close the excerpt.
    mark = estimate_tokens(ELISION)
    chosen = {0}
    spent = estimate_tokens(runs[0]) + mark
    for index in sorted(range(1, len(runs)), key=lambda i: (-score(i), i)):
        if not bags[index]:
            break
        cost = estimate_tokens(runs[index]) + mark
        if spent + cost <= room:
            chosen.add(index)
            spent += cost
    if len(chosen) == 1 or spent > room:
        return None
    parts: list[str] = []
    for index in sorted(chosen):
        if parts and index - 1 not in chosen:
            parts.append(ELISION)
        parts.append(runs[index])
    if len(runs) - 1 not in chosen:
        parts.append(ELISION)
    return "\n".join(parts)


class EvidenceRole(StrEnum):
    RETRIEVED = "retrieved"
    REFERENCED = "referenced"
    DEFINITION = "definition"


class ContextLine(BaseModel):
    """One ancestor's own lines, verbatim: the text before its first child."""

    model_config = ConfigDict(frozen=True)

    citation: str
    text: str


class EvidenceUnit(BaseModel):
    model_config = ConfigDict(frozen=True)

    chunk: Chunk
    # Root first. Siblings repeat their shared ancestors' lines, and each copy
    # is counted against the budget.
    context: tuple[ContextLine, ...]
    # A retrieved unit: the best rank, 1-based, among the hits it carries. A
    # referenced unit: the rank of the unit that cited it. A definition: one
    # past the pool, since nothing ranked it.
    rank: int
    tokens: int
    role: EvidenceRole
    # Why a referenced unit is present: the retrieved unit whose text cites it.
    cited_by: str | None = None
    # Set when the chunk is bigger than the whole budget. The model reads
    # this, and the verifier grounds against this, never the rest of the chunk.
    excerpt: str | None = None

    @model_validator(mode="after")
    def only_a_referenced_unit_is_cited_by_another(self) -> EvidenceUnit:
        if (self.role is EvidenceRole.REFERENCED) != (self.cited_by is not None):
            raise ValueError("cited_by is set for exactly the referenced units")
        return self

    @property
    def citation(self) -> str:
        return self.chunk.node_path

    @property
    def text(self) -> str:
        """What this unit delivers: the excerpt, or the whole chunk."""
        return self.chunk.text if self.excerpt is None else self.excerpt


class EvidencePack(BaseModel):
    model_config = ConfigDict(frozen=True)

    units: tuple[EvidenceUnit, ...]
    budget: int
    # Hits whose full text the pack does not carry, in rank order.
    skipped: tuple[str, ...]

    @property
    def tokens(self) -> int:
        return sum(unit.tokens for unit in self.units)


class EvidencePacker:
    def __init__(
        self, chunks: Iterable[Chunk], *, budget: int = EVIDENCE_BUDGET
    ) -> None:
        if budget < 1:
            raise ValueError(f"budget must be at least 1, not {budget}")
        self._budget = budget
        self._by_id = {chunk.chunk_id: chunk for chunk in chunks}
        self._by_path = {chunk.node_path: chunk for chunk in self._by_id.values()}
        self._own_end: dict[str, int] = {}
        for chunk in self._by_id.values():
            if chunk.parent_id is None:
                continue
            parent = self._by_id.get(chunk.parent_id)
            if parent is None:
                raise ValueError(
                    f"{chunk.node_path}: its parent is not in the chunk set"
                )
            if chunk.char_start is None or parent.char_start is None:
                raise ValueError(
                    f"{chunk.node_path}: a lead-in needs character offsets"
                )
            # Offsets index the root's text, so a parent's own lines
            # end where its first child begins.
            start = chunk.char_start - parent.char_start
            self._own_end[parent.chunk_id] = min(
                self._own_end.get(parent.chunk_id, start), start
            )
        self._lineage: dict[str, frozenset[str]] = {}

    def _ancestors(self, chunk: Chunk) -> list[Chunk]:
        chain = []
        while chunk.parent_id is not None:
            chunk = self._by_id[chunk.parent_id]
            chain.append(chunk)
        chain.reverse()
        return chain

    def _lineage_of(self, chunk: Chunk) -> frozenset[str]:
        """The chunk and every ancestor: whatever carries this chunk's text."""
        if chunk.chunk_id not in self._lineage:
            ids = {ancestor.chunk_id for ancestor in self._ancestors(chunk)}
            self._lineage[chunk.chunk_id] = frozenset({*ids, chunk.chunk_id})
        return self._lineage[chunk.chunk_id]

    def _resolve(self, citation: str) -> Chunk | None:
        """The chunk a reference names, else its nearest ancestor that is one."""
        path: NodePath | None = NodePath.parse(citation)
        while path is not None:
            chunk = self._by_path.get(path.render())
            if chunk is not None:
                return chunk
            path = path.parent
        return None

    def _targets(self, citer: Chunk) -> list[Chunk]:
        """What the citing chunk's text refers to, in document order."""
        targets: dict[str, Chunk] = {}
        for citation in citer.outgoing_refs:
            target = self._resolve(citation)
            if target is not None and target.chunk_id not in self._lineage_of(citer):
                targets.setdefault(target.chunk_id, target)
        return list(targets.values())

    def _unit(
        self,
        chunk: Chunk,
        rank: int,
        role: EvidenceRole,
        cited_by: str | None,
        excerpt: str | None = None,
    ) -> EvidenceUnit:
        context = []
        for ancestor in self._ancestors(chunk):
            own = ancestor.text[: self._own_end[ancestor.chunk_id]].rstrip("\n")
            if own.strip():
                context.append(ContextLine(citation=ancestor.node_path, text=own))
        tokens = estimate_tokens(chunk.text if excerpt is None else excerpt) + sum(
            estimate_tokens(line.text) for line in context
        )
        return EvidenceUnit(
            chunk=chunk,
            context=tuple(context),
            rank=rank,
            tokens=tokens,
            role=role,
            cited_by=cited_by,
            excerpt=excerpt,
        )

    def _excerpted(
        self, units: list[EvidenceUnit], chunk: Chunk, rank: int, query: str
    ) -> list[EvidenceUnit]:
        """An oversize hit as an excerpt, when enough budget is left for one."""
        left = self._budget - sum(unit.tokens for unit in units)
        lead_in = self._unit(chunk, rank, EvidenceRole.RETRIEVED, None, "").tokens
        room = min(left, int(self._budget * EXCERPT_SHARE)) - lead_in
        if room < EXCERPT_MINIMUM:
            return units
        text = excerpt(chunk.text, query, room)
        if text is None:
            return units
        return [*units, self._unit(chunk, rank, EvidenceRole.RETRIEVED, None, text)]

    def _place(
        self,
        units: list[EvidenceUnit],
        chunk: Chunk,
        rank: int,
        role: EvidenceRole,
        cited_by: str | None = None,
        *,
        limit: int | None = None,
    ) -> list[EvidenceUnit]:
        """One candidate against the pack so far."""
        lineage = self._lineage_of(chunk)
        if any(unit.chunk.chunk_id in lineage for unit in units):
            return units
        absorbed = [
            unit for unit in units if chunk.chunk_id in self._lineage_of(unit.chunk)
        ]
        retrieved = [
            unit.rank for unit in absorbed if unit.role is EvidenceRole.RETRIEVED
        ]
        if role is EvidenceRole.RETRIEVED:
            retrieved.append(rank)
        if retrieved:
            unit = self._unit(chunk, min(retrieved), EvidenceRole.RETRIEVED, None)
        else:
            unit = self._unit(chunk, rank, role, cited_by)
        kept = [u for u in units if u not in absorbed]
        budget = self._budget if limit is None else limit
        if sum(u.tokens for u in kept) + unit.tokens > budget:
            return units
        if not absorbed:
            return [*units, unit]
        first = units.index(absorbed[0])
        # Everything before the first absorbed unit is kept, so it keeps its
        # index in `kept`, and the pack's order holds.
        kept.insert(first, unit)
        return kept

    def _oversize(self, results: Sequence[ScoredChunk]) -> list[tuple[int, Chunk]]:
        """Hits in the top EXCERPT_HEAD bigger than the whole budget, with nothing
        beneath them in the ranking, and their ranks."""
        ranked = [result.chunk for result in results]
        return [
            (rank, chunk)
            for rank, chunk in enumerate(ranked[:EXCERPT_HEAD], start=1)
            if estimate_tokens(chunk.text) > self._budget
            and not any(
                other.chunk_id != chunk.chunk_id
                and chunk.chunk_id in self._lineage_of(other)
                for other in ranked
            )
        ]

    def pack(
        self,
        results: Sequence[ScoredChunk],
        *,
        expand: bool = False,
        fill_refs: bool = False,
        definitions: Sequence[Chunk] = (),
        query: str | None = None,
    ) -> EvidencePack:
        """Walk the ranking best first."""
        for chunk in (*(result.chunk for result in results), *definitions):
            if chunk.chunk_id not in self._by_id:
                raise ValueError(f"{chunk.node_path} is not in the packer's chunk set")
        head = EXPANSION_HEAD if expand else len(results)
        oversize = self._oversize(results) if query and query.strip() else []
        limit = None
        if oversize:
            limit = self._budget - int(self._budget * EXCERPT_RESERVE)

        units: list[EvidenceUnit] = []
        for rank, result in enumerate(results[:head], start=1):
            units = self._place(
                units, result.chunk, rank, EvidenceRole.RETRIEVED, limit=limit
            )
        if expand:
            for citer in list(units):
                for target in self._targets(citer.chunk):
                    units = self._place(
                        units,
                        target,
                        citer.rank,
                        EvidenceRole.REFERENCED,
                        citer.citation,
                        limit=limit,
                    )
        for rank, result in enumerate(results[head:], start=head + 1):
            units = self._place(
                units, result.chunk, rank, EvidenceRole.RETRIEVED, limit=limit
            )
        for rank, chunk in oversize:
            units = self._excerpted(units, chunk, rank, query or "")
        if fill_refs and not expand:
            # Unlike `expand`, this runs after every hit is placed,
            # so a reference only takes budget no hit could use.
            for citer in list(units):
                if citer.role is not EvidenceRole.RETRIEVED:
                    continue
                for target in self._targets(citer.chunk):
                    units = self._place(
                        units,
                        target,
                        citer.rank,
                        EvidenceRole.REFERENCED,
                        citer.citation,
                    )
        for chunk in definitions:
            units = self._place(units, chunk, len(results) + 1, EvidenceRole.DEFINITION)

        packed = {unit.chunk.chunk_id for unit in units}
        skipped = tuple(
            result.chunk.node_path
            for result in results
            if not packed & self._lineage_of(result.chunk)
        )
        return EvidencePack(units=tuple(units), budget=self._budget, skipped=skipped)
