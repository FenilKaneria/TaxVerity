"""Step 5.7 — why a gold label is missing from the delivered evidence.

Each label the pack misses gets exactly one category, by the first rule that
fits, in `FailureCategory` order (ADR-085). The rules read the gold labels, so
this is an offline oracle: a query-time diagnoser (Phase 8.2) cannot use it.

"Genuinely absent" is not a category. Every answerable label names a chunk
(the gold-set suite pins it), so absence arises only for the negative slice,
which carries no labels and is reported apart.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from taxverity.chunking.models import Chunk
from taxverity.corpus.nodes import NodePath
from taxverity.evals.metrics import (
    CreditMode,
    UnresolvedCitationError,
    credits,
    normalise_citation,
)

TAXONOMY_STAGE_VERSION = 1
COHORTS_FILENAME = "failure_cohorts_v1.json"


class FailureCategory(StrEnum):
    """In precedence order: the first that fits wins."""

    # The pack carries a descendant of the label, not the label.
    LABEL_GRANULARITY = "label_granularity"
    # The ranking put the label in the pool; the budget left it out.
    BUDGET = "budget"
    # The part of the answer the pack does carry cites the missing part.
    DANGLING_FORWARD = "dangling_forward"
    # The missing part cites the part the pack carries; no outgoing edge leads to it.
    DANGLING_BACKWARD = "dangling_backward"
    # An independent part of a question with more than one label.
    MULTI_PART = "multi_part"
    # A one-label question whose wording reached nothing that leads to the answer.
    VOCABULARY = "vocabulary"


class LabelFailure(BaseModel):
    model_config = ConfigDict(frozen=True)

    query_id: str
    label: str
    category: FailureCategory
    # What decided the category: the delivered descendant, the pool hit, or the
    # delivered unit on the other end of the reference. Empty for the last two.
    via: tuple[str, ...]


def _related(a: str, b: str) -> bool:
    return credits(a, b, CreditMode.LENIENT) or credits(b, a, CreditMode.LENIENT)


class FailureClassifier:
    def __init__(self, chunks: Iterable[Chunk]) -> None:
        self._by_path = {normalise_citation(chunk.node_path): chunk for chunk in chunks}

    def _chunk(self, citation: str) -> Chunk:
        try:
            return self._by_path[normalise_citation(citation)]
        except KeyError:
            raise UnresolvedCitationError(f"{citation} names no chunk") from None

    def resolve(self, citation: str) -> Chunk | None:
        """Nearest existing ancestor: a reference into an untrusted, unsplit
        subtree names no chunk (ADR-056). The evidence packer's walk."""
        path: NodePath | None = NodePath.parse(citation)
        while path is not None:
            chunk = self._by_path.get(path.render())
            if chunk is not None:
                return chunk
            path = path.parent
        return None

    def cites(self, citer: str, target: str) -> bool:
        """Whether the citer's text refers to the target, any part of it or
        anything containing it. One hop, outgoing edges only."""
        for reference in self._chunk(citer).outgoing_refs:
            resolved = self.resolve(reference)
            if resolved is not None and _related(resolved.node_path, target):
                return True
        return False

    def classify(
        self,
        query_id: str,
        required: Sequence[str],
        pool: Sequence[str],
        packed: Sequence[str],
    ) -> tuple[LabelFailure, ...]:
        """One failure per required label that no packed citation credits
        leniently, in label order."""
        if not required:
            raise ValueError(f"{query_id} carries no label; negatives are not classified")
        for citation in (*required, *pool, *packed):
            self._chunk(citation)
        lenient = CreditMode.LENIENT
        failures = []
        for label in required:
            if any(credits(unit, label, lenient) for unit in packed):
                continue
            others = [other for other in required if other != label]
            # Units carrying some other part of the answer, whole or in part.
            support = [unit for unit in packed if any(_related(unit, o) for o in others)]
            descendants = tuple(unit for unit in packed if credits(label, unit, lenient))
            in_pool = tuple(hit for hit in pool if credits(hit, label, lenient))
            forward = tuple(unit for unit in support if self.cites(unit, label))
            backward = tuple(unit for unit in support if self.cites(label, unit))
            if descendants:
                category, via = FailureCategory.LABEL_GRANULARITY, descendants
            elif in_pool:
                category, via = FailureCategory.BUDGET, in_pool
            elif forward:
                category, via = FailureCategory.DANGLING_FORWARD, forward
            elif backward:
                category, via = FailureCategory.DANGLING_BACKWARD, backward
            elif others:
                category, via = FailureCategory.MULTI_PART, ()
            else:
                category, via = FailureCategory.VOCABULARY, ()
            failures.append(
                LabelFailure(query_id=query_id, label=label, category=category, via=via)
            )
        return tuple(failures)


class FailureCohorts(BaseModel):
    """The Step 5.7 classification, frozen as the ruler later steps are judged
    on. Step 5.8 measures its recovery on the vocabulary cohort (ADR-085)."""

    model_config = ConfigDict(frozen=True)

    stage_version: int
    # Citations, not chunk ids, so the cohort outlives a corpus_version (ADR-059).
    cohorts: dict[FailureCategory, tuple[tuple[str, str], ...]]


def cohorts_of(failures: Iterable[LabelFailure]) -> FailureCohorts:
    grouped: dict[FailureCategory, list[tuple[str, str]]] = {c: [] for c in FailureCategory}
    for failure in failures:
        grouped[failure.category].append((failure.query_id, failure.label))
    return FailureCohorts(
        stage_version=TAXONOMY_STAGE_VERSION,
        cohorts={category: tuple(sorted(pairs)) for category, pairs in grouped.items()},
    )


def write_cohorts(path: Path, cohorts: FailureCohorts) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(
        cohorts.model_dump(mode="json"), sort_keys=True, ensure_ascii=False, indent=2
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(text + "\n")


def load_cohorts(path: Path) -> FailureCohorts:
    return FailureCohorts.model_validate_json(path.read_text(encoding="utf-8"))
