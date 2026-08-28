from __future__ import annotations

import re
from collections import Counter
from collections.abc import Collection, Sequence

from pydantic import BaseModel, ConfigDict

from taxverity.chunking.models import Chunk
from taxverity.corpus.nodes import NodeType

STATS_STAGE_VERSION = 1

# A word/punctuation split, not a real subword tokenizer. No embedding model is
# chosen until the Phase 4.7 bake-off, and each candidate has its own
# vocabulary, so a real count taken now would be a count of the wrong model.
# This proxy is a floor: BPE splits long words further, never fewer. See
# SAFETY_FACTOR for how the floor is turned into a usable ceiling.
TOKEN = re.compile(r"\w+|[^\w\s]")

# Both Phase 4 candidates (jina-embeddings-v5-text-small, Qwen3-Embedding-0.6B)
# carry a 32K context window.
CONTEXT_TOKENS = 32_768

# A deliberately pessimistic multiplier on the proxy. Measured over this
# corpus the proxy averages 4.66 characters per token, close to the usual ~4
# chars/token BPE rule of thumb, so 2x is well past any plausible real
# tokenizer. It exists so the context check is an argument, not a hope.
SAFETY_FACTOR = 2

# Not a hard limit -- ADR-055 makes a large root chunk expected, and the
# precise unit is the child. This is the threshold above which a chunk gets
# named and explained in the report rather than left in an aggregate.
LARGE_CHUNK_TOKENS = 2_048

# Below this a chunk is too small to stand alone as retrieved evidence
# ("21. Tin."). Reported, not removed: it is still the text at that citation.
SMALL_CHUNK_TOKENS = 10

# The Act's own marker for text an amendment removed.
OMITTED = "[***]"


def estimate_tokens(text: str) -> int:
    return len(TOKEN.findall(text))


def percentile(values: Sequence[int], quantile: float) -> int:
    """Nearest-rank on an already sorted sequence -- no interpolation, so every
    reported figure is a value some real chunk actually has."""
    if not values:
        return 0
    index = min(len(values) - 1, int(quantile * len(values)))
    return values[index]


class Distribution(BaseModel):
    model_config = ConfigDict(frozen=True)

    count: int
    total: int
    minimum: int
    p50: int
    p90: int
    p95: int
    p99: int
    maximum: int

    @classmethod
    def of(cls, values: Collection[int]) -> Distribution:
        ordered = sorted(values)
        return cls(
            count=len(ordered),
            total=sum(ordered),
            minimum=ordered[0] if ordered else 0,
            p50=percentile(ordered, 0.50),
            p90=percentile(ordered, 0.90),
            p95=percentile(ordered, 0.95),
            p99=percentile(ordered, 0.99),
            maximum=ordered[-1] if ordered else 0,
        )


class Outlier(BaseModel):
    model_config = ConfigDict(frozen=True)

    node_path: str
    node_type: NodeType
    tokens: int
    characters: int
    reason: str


def outlier_reason(chunk: Chunk, untrusted: Collection[str]) -> str:
    if chunk.node_path in untrusted:
        return "untrusted subtree, unsplit (ADR-056)"
    if chunk.is_root:
        return "root carrying its whole subtree (ADR-055)"
    return "intermediate node carrying a large subtree (ADR-055)"


class ChunkStatistics(BaseModel):
    model_config = ConfigDict(frozen=True)

    chunks: int
    roots: int
    by_type: tuple[tuple[NodeType, int], ...]
    by_depth: tuple[tuple[int, int], ...]

    characters: Distribution
    tokens: Distribution

    over_context: int
    over_context_with_safety: int
    large: tuple[Outlier, ...]
    small: int
    omitted_only: int

    glossary_terms: int
    terms_per_chunk: Distribution
    chunks_without_terms: int
    term_frequency: tuple[tuple[str, int], ...]

    refs_per_chunk: Distribution
    chunks_with_refs: int

    @property
    def fits_context(self) -> bool:
        return self.over_context_with_safety == 0


def summarise(
    chunks: Sequence[Chunk],
    *,
    untrusted: Collection[str] = (),
    glossary_size: int = 0,
) -> ChunkStatistics:
    """Everything the Step 2.3 report prints, computed in one pass per measure.

    Token figures are over ``embed_text()`` -- the breadcrumb travels to the
    embedding model with the text, so the context check has to include it.
    Character figures are over ``text`` alone, which is what the store holds.
    """
    tokens = [estimate_tokens(chunk.embed_text()) for chunk in chunks]
    # Measured over the body alone: a chunk is too small to stand as evidence
    # because of what it says, and the breadcrumb is provenance, not content.
    body_tokens = [estimate_tokens(chunk.text) for chunk in chunks]
    term_counts = Counter(term for chunk in chunks for term in chunk.defined_terms)
    large = sorted(
        (
            Outlier(
                node_path=chunk.node_path,
                node_type=chunk.node_type,
                tokens=count,
                characters=len(chunk.text),
                reason=outlier_reason(chunk, untrusted),
            )
            for chunk, count in zip(chunks, tokens, strict=True)
            if count > LARGE_CHUNK_TOKENS
        ),
        key=lambda outlier: -outlier.tokens,
    )
    return ChunkStatistics(
        chunks=len(chunks),
        roots=sum(1 for chunk in chunks if chunk.is_root),
        by_type=tuple(Counter(chunk.node_type for chunk in chunks).most_common()),
        by_depth=tuple(
            sorted(Counter(chunk.node_path.count("(") for chunk in chunks).items())
        ),
        characters=Distribution.of([len(chunk.text) for chunk in chunks]),
        tokens=Distribution.of(tokens),
        over_context=sum(1 for count in tokens if count > CONTEXT_TOKENS),
        over_context_with_safety=sum(
            1 for count in tokens if count * SAFETY_FACTOR > CONTEXT_TOKENS
        ),
        large=tuple(large),
        small=sum(1 for count in body_tokens if count < SMALL_CHUNK_TOKENS),
        omitted_only=sum(1 for chunk in chunks if is_omitted_only(chunk.text)),
        glossary_terms=glossary_size,
        terms_per_chunk=Distribution.of([len(chunk.defined_terms) for chunk in chunks]),
        chunks_without_terms=sum(1 for chunk in chunks if not chunk.defined_terms),
        term_frequency=tuple(term_counts.most_common()),
        refs_per_chunk=Distribution.of([len(chunk.outgoing_refs) for chunk in chunks]),
        chunks_with_refs=sum(1 for chunk in chunks if chunk.outgoing_refs),
    )


_OMITTED_ONLY = re.compile(rf"^\S*\s*{re.escape(OMITTED)}$")


def is_omitted_only(text: str) -> bool:
    """A chunk whose entire body is its marker plus the Act's omission mark."""
    return bool(_OMITTED_ONLY.match(re.sub(r"\s+", " ", text).strip()))
