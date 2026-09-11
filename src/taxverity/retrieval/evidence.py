"""Step 5.3 — what the generator is handed: the fused ranking turned into
distinct evidence within a token budget (ADR-082).

ADR-055 already gives a chunk its whole subtree, so a hit is delivered as
itself, not swapped for its section. Delivery handles the two problems that
remain. A hit whose ancestor is also a hit is the same text twice. A clause read
without the lines above it often says nothing on its own ("thirty per cent of
the annual value").
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from pydantic import BaseModel, ConfigDict

from taxverity.chunking.models import Chunk
from taxverity.chunking.stats import estimate_tokens
from taxverity.retrieval.base import ScoredChunk

EVIDENCE_STAGE_VERSION = 1

# Groq's free tier allows openai/gpt-oss-120b 8,000 tokens a minute (checked
# 2026-09-11; Step 7.1 re-verifies). Evidence gets about half, leaving the rest
# for the prompt, the user's facts and the answer. The count is the Step 2.3
# proxy, which runs 1.145x short of Jina's billed tokens over the corpus, so
# 4,000 here is about 4,600 real tokens.
EVIDENCE_BUDGET = 4_000

# How deep delivery reads the ranking. Deduplication collapses a top 10 to about
# four distinct units, so 20 is what fills the budget, and it is the pool Step
# 5.6's reranker is budgeted for.
EVIDENCE_POOL = 20


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
    # The best rank, 1-based, among the hits this unit carries.
    rank: int
    tokens: int

    @property
    def citation(self) -> str:
        return self.chunk.node_path


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
    def __init__(self, chunks: Iterable[Chunk], *, budget: int = EVIDENCE_BUDGET) -> None:
        if budget < 1:
            raise ValueError(f"budget must be at least 1, not {budget}")
        self._budget = budget
        self._by_id = {chunk.chunk_id: chunk for chunk in chunks}
        self._own_end: dict[str, int] = {}
        for chunk in self._by_id.values():
            if chunk.parent_id is None:
                continue
            parent = self._by_id.get(chunk.parent_id)
            if parent is None:
                raise ValueError(f"{chunk.node_path}: its parent is not in the chunk set")
            if chunk.char_start is None or parent.char_start is None:
                raise ValueError(f"{chunk.node_path}: a lead-in needs character offsets")
            # Offsets index the root's text (ADR-055), so a parent's own lines
            # end where its first child begins.
            start = chunk.char_start - parent.char_start
            self._own_end[parent.chunk_id] = min(self._own_end.get(parent.chunk_id, start), start)
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

    def _unit(self, chunk: Chunk, rank: int) -> EvidenceUnit:
        context = []
        for ancestor in self._ancestors(chunk):
            own = ancestor.text[: self._own_end[ancestor.chunk_id]].rstrip("\n")
            if own.strip():
                context.append(ContextLine(citation=ancestor.node_path, text=own))
        tokens = estimate_tokens(chunk.text) + sum(
            estimate_tokens(line.text) for line in context
        )
        return EvidenceUnit(chunk=chunk, context=tuple(context), rank=rank, tokens=tokens)

    def pack(self, results: Sequence[ScoredChunk]) -> EvidencePack:
        """Walk the ranking once, best first.

        A hit already carried by a packed unit, itself or an ancestor, adds
        nothing. A hit that is an ancestor of packed units replaces them, costs
        only the difference, and takes the best rank among them. A hit that does
        not fit is passed over and the walk goes on, so a giant root at rank 1
        does not crowd out everything below it.
        """
        units: list[EvidenceUnit] = []
        for rank, result in enumerate(results, start=1):
            chunk = result.chunk
            if chunk.chunk_id not in self._by_id:
                raise ValueError(f"{chunk.node_path} is not in the packer's chunk set")
            lineage = self._lineage_of(chunk)
            if any(unit.chunk.chunk_id in lineage for unit in units):
                continue
            absorbed = {
                unit.chunk.chunk_id
                for unit in units
                if chunk.chunk_id in self._lineage_of(unit.chunk)
            }
            kept = [unit for unit in units if unit.chunk.chunk_id not in absorbed]
            first = next(
                (i for i, unit in enumerate(units) if unit.chunk.chunk_id in absorbed), None
            )
            unit = self._unit(chunk, units[first].rank if first is not None else rank)
            if sum(u.tokens for u in kept) + unit.tokens > self._budget:
                continue
            if first is None:
                units.append(unit)
            else:
                # Everything before the first absorbed unit is kept, so it keeps
                # its index in `kept`, and rank order holds.
                kept.insert(first, unit)
                units = kept

        packed = {unit.chunk.chunk_id for unit in units}
        skipped = tuple(
            result.chunk.node_path
            for result in results
            if not packed & self._lineage_of(result.chunk)
        )
        return EvidencePack(units=tuple(units), budget=self._budget, skipped=skipped)
