"""Step 4.4 — the local reference build of the served embedding model, pinned
to an exact upstream revision.

Not a serving path: embeddings are served by the Jina hosted API (ADR-075).
These weights exist to check the API's fidelity offline, and as a fallback if
the API budget runs out. Fetched by `scripts/download_models.py`.
"""

from __future__ import annotations

from dataclasses import dataclass

# Our application recipe, versioned independently of the weights (ADR-069): the
# EmbedKind value is passed straight through as the sentence-transformers
# prompt_name, the revision-pinned model card supplies the prompt text, and the
# output is unit-normalised. Bump this on any change to how encode() is called
# — a moved recipe shifts every vector while model_id and revision still match.
ST_ENCODING = "sentence-transformers/prompt_name+normalize/v1"

_SHA_LEN = 40


@dataclass(frozen=True)
class EmbedderSpec:
    key: str
    model_id: str
    revision: str
    dim: int
    encoding: str = ST_ENCODING

    def __post_init__(self) -> None:
        if len(self.revision) != _SHA_LEN or not all(
            c in "0123456789abcdef" for c in self.revision
        ):
            raise ValueError(
                f"revision must be a 40-char commit sha, got {self.revision!r}"
            )


# The retrieval-task build of jina-embeddings-v5-text-small, not the base repo:
# it ships as a plain sentence-transformers model (no custom_st.py, so no
# trust_remote_code), which is the packaging this project uses. See ADR-071.
JINA_V5 = EmbedderSpec(
    key="jina-v5",
    model_id="jinaai/jina-embeddings-v5-text-small-retrieval",
    revision="6856e76bb72982e58de0620458a4e8b3614da340",
    dim=1024,
)

# Qwen3-Embedding-0.6B was dropped with the Step 4.7 bake-off at R15 (ADR-075).
CANDIDATES: tuple[EmbedderSpec, ...] = (JINA_V5,)

# The base jina repo carries a 2.4 GB ONNX export we never load. allow_patterns
# keeps the download to what is actually read.
DOWNLOAD_ALLOW_PATTERNS: tuple[str, ...] = (
    "*.json",
    "*.txt",
    "*.model",
    "tokenizer.json",
    "model.safetensors",
    "1_Pooling/*",
    "2_Normalize/*",
)
