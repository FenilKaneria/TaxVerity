"""Steps 5.3 and 5.4 — what the generator is handed: the fused ranking turned
into distinct evidence within a token budget (ADR-082), plus the provisions
that evidence refers to, one hop out (ADR-083).

ADR-055 already gives a chunk its whole subtree, so a hit is delivered as
itself, not swapped for its section. Delivery handles the two problems that
remain. A hit whose ancestor is also a hit is the same text twice. A clause read
without the lines above it often says nothing on its own ("thirty per cent of
the annual value").

Expansion handles a third: a clause that says "as determined under section 21"
is incomplete without section 21, and nothing in the ranking has to know that.
It is built and measured, and off by default: its registered rule rejected it,
because referenced text displaced labels from the pool's tail (ADR-083).

Step 5.8 adds definitions: the section 2 clause for a specific term the
question uses, placed after every retrieved hit so it only fills what the
budget leaves. Also off by default (ADR-086).

R27 adds excerpts: a hit bigger than the whole budget could never be packed,
so section 393 (the TDS table, unsplit under ADR-056) was retrieved first and
never read. Such a hit is now delivered as the runs of its own lines that
share words with the question, after every hit that fits has been placed.
"""

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

# The k the hybrid ranking was adopted at (ADR-081). Hits above it are packed
# before any referenced text, hits below it after, so expansion can never
# displace evidence the adopted ranking itself delivered.
EXPANSION_HEAD = 10

# R20 Step 20.2 measured a flat ancestor-promotion size-ratio guard (r009's
# finding) and REJECTED it (ADR-122): any ratio tight enough to block r009's
# real ~2.03x root/hit disparity (section 19 over 19(1)) also blocks several
# legitimate multi-child promotions in the gold v2 crossref slice (q020,
# q021, q062-q068) whose own incremental ratios during the walk run as low
# as ~2.0x-5x. No single threshold separates the two. Left unguarded; a
# pool-aware (not order-dependent) design is future work, not this pass.

# An excerpt never takes more than this share of the budget.
EXCERPT_SHARE = 0.5
# Only an oversize hit ranked this high (or named by the person, which the
# citation shortcut ranks first) is excerpted, with EXCERPT_RESERVE of the
# budget held back for it; the excerpt is placed after the whole ranking.
# Measured on gold v2: an excerpt placed in its own rank's turn cost s.194
# its place (answer-gold pack coverage 0.908 -> 0.888), and one for a hit
# ranked 7th or 12th was 1,000 to 1,500 tokens beside the point.
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
    """The opening run, then the runs sharing the most (and rarest) words with
    the query, in document order, within `room` tokens. None when no later
    run shares a telling word with the query: the opening lines alone answer
    nothing.

    A word found in more than half the runs ("the", "of", "resident" in a
    section about residents) tells no run from another and is not counted.
    """
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
    # R27: set when the chunk is bigger than the whole budget. The model reads
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
            # Offsets index the root's text (ADR-055), so a parent's own lines
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
        """The chunk a reference names, else its nearest ancestor that is one.

        ADR-056 prunes an untrusted subtree to its root, so a reference into
        section 393 names no chunk. The same walk as the citation shortcut's.
        """
        path: NodePath | None = NodePath.parse(citation)
        while path is not None:
            chunk = self._by_path.get(path.render())
            if chunk is not None:
                return chunk
            path = path.parent
        return None

    def _targets(self, citer: Chunk) -> list[Chunk]:
        """What the citing chunk's text refers to, in document order. A target
        that is the citer or an ancestor of it is dropped: the citer already
        carries it, or its lead-in does."""
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
        """One candidate against the pack so far.

        Already carried by a packed unit, itself or an ancestor: nothing
        changes. An ancestor of packed units: it replaces them, costs only the
        difference, and stays retrieved if any of them was, with their best
        rank. Too big for what is left: nothing changes. `limit` is a budget
        below the packer's own, while room is held back for an excerpt.
        """
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
        """Hits in the top EXCERPT_HEAD bigger than the whole budget, with
        nothing beneath them in the ranking, and their ranks. A hit with a
        ranked descendant is left alone: the ranking has already found the
        part of it that matters."""
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
        """Walk the ranking best first.

        Without `expand`, the Step 5.3 walk: one pass. With it, three passes
        over one budget (ADR-083, rejected, so not the default):

        1. Hits ranked 1 to EXPANSION_HEAD.
        2. What those units' text refers to, one hop, taken in the order of the
           unit citing it. A referenced unit's own references are not followed.
        3. The rest of the ranking.

        `fill_refs` (ADR-131) is the production alternative: after the whole
        ranking, what each retrieved unit refers to, one hop, in rank order,
        only where budget is left. It never costs a hit its place.

        `definitions` are placed last of all, so they never displace a hit
        (ADR-086, off by default). A hit that does not fit is passed over and
        the walk goes on, so a giant root at rank 1 does not crowd out
        everything below it.

        With `query` (R27), a hit bigger than the whole budget is delivered
        as an excerpt against that query, after the whole ranking and before
        any reference fill, with EXCERPT_RESERVE of the budget held back for
        it. Only a hit in the top EXCERPT_HEAD is excerpted. Without `query`,
        such a hit is passed over as before.
        """
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
            # ADR-131: unlike `expand`, this runs after every hit is placed,
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
