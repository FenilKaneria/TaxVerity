"""Step 4.1 — what an embedding backend is, and the stub the contract is
tested against while the model is still unchosen (Step 4.7)."""

from __future__ import annotations

import hashlib
import math
import struct
from collections.abc import Sequence
from enum import StrEnum
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from taxverity.corpus.loader import normalise

# The batch ceiling is part of the contract both halves must agree on, so it
# lives in the module neither of them can avoid importing. The service refuses
# a larger request; the client refuses to build one.
MAX_BATCH = 256


class EmbedKind(StrEnum):
    """Which side of a retrieval pair a text is being encoded as (ADR-069).

    Both Step 4.7 finalists encode the two sides differently, so a query
    embedded as a document is silently wrong rather than merely imprecise.
    Deliberately has no default at any layer it travels through: a default is
    exactly how that mistake happens quietly.
    """

    QUERY = "query"
    DOCUMENT = "document"


class ModelInfo(BaseModel):
    """The identity ADR-026 requires to detect version skew.

    The offline job (Step 4.4) stores it alongside the vectors and the query
    path refuses to serve on a mismatch. A vector carries no evidence of which
    weights produced it, so an index built with model A and queried with model
    B degrades recall silently and reports nothing.
    """

    model_config = ConfigDict(frozen=True)

    model_id: str = Field(min_length=1)
    dim: int = Field(gt=0)
    # A content revision of the weights (an upstream commit hash for a real
    # model), not a version of this codebase: the same model_id at two
    # revisions is two embedding spaces.
    revision: str = Field(min_length=1)
    # Execution stack — torch vs ONNX vs int8 change the numbers a model emits
    # for identical weights, which is the condition on the quantisation
    # experiment in plan §2.7.
    runtime: str = Field(min_length=1)
    # The prompt/pooling recipe, versioned. Without it the four fields above
    # would report a match after a query-prompt edit that moved every vector,
    # because the weights really are unchanged — the skew hole ADR-069 opens
    # and this field closes.
    encoding: str = Field(min_length=1)


@runtime_checkable
class Embedder(Protocol):
    def info(self) -> ModelInfo: ...

    def embed(self, texts: Sequence[str], kind: EmbedKind) -> list[list[float]]:
        """One unit-length vector per input text, in input order.

        `kind` is required, never defaulted: see ADR-069.
        """
        ...


STUB_MODEL_ID = "stub-hash-embedder"
STUB_DIM = 32
STUB_REVISION = "1"
STUB_RUNTIME = "stub"
# Not a claim to a real prompt recipe — the stub applies none. It only folds
# `kind` into its hash so the contract is testable end to end.
STUB_ENCODING = "stub"


class StubEmbedder:
    """Deterministic vectors from a text hash — no torch, no weights.

    Carries no semantics whatsoever, deliberately: it exists to exercise the
    HTTP contract, not retrieval quality. Any test asserting that a similarity
    score here means something is testing the stub, not the system.
    """

    def info(self) -> ModelInfo:
        return ModelInfo(
            model_id=STUB_MODEL_ID,
            dim=STUB_DIM,
            revision=STUB_REVISION,
            runtime=STUB_RUNTIME,
            encoding=STUB_ENCODING,
        )

    def embed(self, texts: Sequence[str], kind: EmbedKind) -> list[list[float]]:
        return [_hash_vector(text, kind) for text in texts]


def _hash_vector(text: str, kind: EmbedKind) -> list[float]:
    # normalise() so the stub agrees with the rest of the pipeline about what
    # two texts being "the same" means — the corpus carries soft hyphens and
    # NBSPs that differ byte-wise while reading identically (Step 1.1).
    # `kind` is in the seed so a test can prove it travelled the whole path;
    # a stub that ignored it would make the contract unobservable.
    seed = f"{kind.value}\x00{normalise(text)}".encode()
    raw = b""
    counter = 0
    while len(raw) < STUB_DIM * 4:
        raw += hashlib.sha256(seed + counter.to_bytes(4, "big")).digest()
        counter += 1
    values = [
        struct.unpack(">I", raw[i * 4 : i * 4 + 4])[0] / 2**31 - 1.0
        for i in range(STUB_DIM)
    ]
    norm = math.sqrt(sum(v * v for v in values))
    # An all-zero vector cannot be normalised; unreachable for a sha256 digest,
    # guarded because a division by zero here would surface as a 500.
    if norm == 0.0:
        return [1.0] + [0.0] * (STUB_DIM - 1)
    return [v / norm for v in values]
