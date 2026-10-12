"""What an embedding backend is, the identity that guards against skew, and a stub for
testing the contract without a network or weights."""

from __future__ import annotations

import hashlib
import math
import struct
from collections.abc import Sequence
from enum import StrEnum
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from taxverity.corpus.loader import normalise


class EmbedKind(StrEnum):
    """Which side of a retrieval pair a text is being encoded as."""

    QUERY = "query"
    DOCUMENT = "document"


class ModelInfo(BaseModel):
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
    encoding: str = Field(min_length=1)


class ModelIdentityError(RuntimeError):
    """The embedder is not the one the index was built with."""


def describe(info: ModelInfo) -> str:
    return (
        f"{info.model_id}@{info.revision} dim={info.dim} "
        f"runtime={info.runtime} encoding={info.encoding}"
    )


@runtime_checkable
class Embedder(Protocol):
    def info(self) -> ModelInfo: ...

    def embed(self, texts: Sequence[str], kind: EmbedKind) -> list[list[float]]:
        """One unit-length vector per input text, in input order."""
        ...


STUB_MODEL_ID = "stub-hash-embedder"
STUB_DIM = 32
STUB_REVISION = "1"
STUB_RUNTIME = "stub"
# Not a claim to a real prompt recipe — the stub applies none. It only folds
# `kind` into its hash so the contract is testable end to end.
STUB_ENCODING = "stub"


class StubEmbedder:
    """Deterministic vectors from a text hash — no torch, no weights."""

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
    # NBSPs that differ byte-wise while reading identically.
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
