"""Step 4.4 — the real Embedder: a sentence-transformers model on torch.

Imports torch and sentence_transformers only inside `__init__`, so the module
itself stays cheap to import and free of the `embed` extra at load time
(ADR-065). Constructed by the offline job and, from Phase 15.2, the deployed
service — never on the API request path.
"""

from __future__ import annotations

from collections.abc import Sequence

from taxverity.embedding.backends import EmbedKind, ModelInfo
from taxverity.embedding.candidates import EmbedderSpec
from taxverity.observability import get_logger

logger = get_logger(__name__)

# Device (cuda vs cpu) is deliberately not folded in: the batch job builds
# vectors on the GPU and production queries on CPU by design, and fp32 torch
# differs by ~1e-6 between the two — noise for cosine retrieval. The runtime
# that does change the numbers is a quantised one (ONNX, int8), and that flips
# this string. See ADR-072.
RUNTIME = "torch"

_PROBE_TEXT = "probe"


class SentenceTransformerEmbedder:
    def __init__(
        self,
        spec: EmbedderSpec,
        *,
        device: str | None = None,
        cache_folder: str | None = None,
    ) -> None:
        import torch
        from sentence_transformers import SentenceTransformer

        resolved = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._spec = spec
        self._model = SentenceTransformer(
            spec.model_id,
            revision=spec.revision,
            device=resolved,
            cache_folder=cache_folder,
        )
        self.device = resolved
        # Measure the real output width rather than trust the model's
        # self-report (its API for that was renamed between the versions the
        # `embed` extra allows). This also does the first forward pass.
        width = len(self.embed([_PROBE_TEXT], EmbedKind.DOCUMENT)[0])
        if width != spec.dim:
            raise ValueError(
                f"{spec.model_id}@{spec.revision} embeds at dim={width}, "
                f"spec declares {spec.dim}"
            )
        logger.info(
            "loaded %s@%s on %s dim=%d",
            spec.model_id,
            spec.revision,
            resolved,
            spec.dim,
        )

    def info(self) -> ModelInfo:
        return ModelInfo(
            model_id=self._spec.model_id,
            dim=self._spec.dim,
            revision=self._spec.revision,
            runtime=RUNTIME,
            encoding=self._spec.encoding,
        )

    def embed(self, texts: Sequence[str], kind: EmbedKind) -> list[list[float]]:
        import numpy as np

        # prompt_name is exactly EmbedKind.value: both finalists ship a
        # `prompts` map keyed "query"/"document", verified against the pinned
        # revisions. An unknown name raises in sentence-transformers, which is
        # the failure we want if a future model does not follow the convention.
        vectors = self._model.encode(
            list(texts),
            prompt_name=kind.value,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        # Re-normalise in float64. The models carry a normalising layer but run
        # it in bf16, leaving the norm off unit by ~5e-4 — enough that a stored
        # dot product is not quite the cosine Step 4.5 will treat it as.
        vectors = np.asarray(vectors, dtype=np.float64)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors.tolist()
