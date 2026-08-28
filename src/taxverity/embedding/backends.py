"""Step 4.1 — what an embedding backend is, and the stub the contract is
tested against while the model is still unchosen (Step 4.7)."""

from __future__ import annotations

import hashlib
import math
import struct
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from taxverity.corpus.loader import normalise

# The batch ceiling is part of the contract both halves must agree on, so it
# lives in the module neither of them can avoid importing. The service refuses
# a larger request; the client refuses to build one.
MAX_BATCH = 256


class ModelInfo(BaseModel):
    """The four fields ADR-026 requires to detect version skew.

    The offline job (Step 4.4) stores all four alongside the vectors and the
    query path refuses to serve on a mismatch. A vector carries no evidence of
    which weights produced it, so an index built with model A and queried with
    model B degrades recall silently and reports nothing.
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


@runtime_checkable
class Embedder(Protocol):
    def info(self) -> ModelInfo: ...

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """One unit-length vector per input text, in input order."""
        ...


STUB_MODEL_ID = "stub-hash-embedder"
STUB_DIM = 32
STUB_REVISION = "1"
STUB_RUNTIME = "stub"


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
        )

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [_hash_vector(text) for text in texts]


def _hash_vector(text: str) -> list[float]:
    # normalise() so the stub agrees with the rest of the pipeline about what
    # two texts being "the same" means — the corpus carries soft hyphens and
    # NBSPs that differ byte-wise while reading identically (Step 1.1).
    seed = normalise(text).encode("utf-8")
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
