"""Step 4.4 (R15) — the served Embedder: Jina's hosted embeddings API (ADR-075).

Called in-process by the offline job (DOCUMENT) and, from Step 4.5, the query
path (QUERY). No torch, no weights and no embedding service: the same endpoint
serves both sides of the index in every environment, which is what keeps them
in one embedding space (ADR-024 as amended).

Synchronous, for the same reason the deleted Step 4.3 client was: the offline
job is a batch loop and vector search is synchronous NumPy.
"""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from typing import Any

import httpx2

from taxverity.config import Settings
from taxverity.embedding.backends import EmbedKind, ModelInfo
from taxverity.observability import get_logger, redact

logger = get_logger(__name__)

API_URL = "https://api.jina.ai/v1/embeddings"
MODEL_ID = "jina-embeddings-v5-text-small"
DIM = 1024
# A hosted endpoint exposes no weights revision. What detects an upstream model
# change is the probe fingerprint in the vector manifest (ADR-075), not this.
REVISION = "hosted-unpinned"
RUNTIME = "jina-api"
# Bump on any change to the request recipe below — task names, normalisation,
# truncation, dimensions. A moved recipe shifts every vector while model_id
# still matches (ADR-069).
ENCODING = "jina-api/task+normalized/v1"

TASKS = {
    EmbedKind.QUERY: "retrieval.query",
    EmbedKind.DOCUMENT: "retrieval.passage",
}

# A section 2 request is ~14.6k tokens and takes a while to encode server-side.
DEFAULT_TIMEOUT = 120.0
DEFAULT_MAX_ATTEMPTS = 6
DEFAULT_BACKOFF_BASE = 1.0
MAX_RETRY_AFTER = 60.0

RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})


class EmbeddingAPIError(RuntimeError):
    pass


class JinaAPIEmbedder:
    def __init__(
        self,
        api_key: str,
        *,
        http_client: httpx2.Client | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        backoff_base: float = DEFAULT_BACKOFF_BASE,
    ) -> None:
        if not api_key:
            raise ValueError("api_key must be non-empty")
        if max_attempts < 1:
            raise ValueError(f"max_attempts must be at least 1, not {max_attempts}")
        self._headers = {"Authorization": f"Bearer {api_key}"}
        self._max_attempts = max_attempts
        self._backoff_base = backoff_base
        # A supplied client is borrowed, not owned.
        self._owns_client = http_client is None
        self._client = http_client or httpx2.Client(timeout=timeout)
        # Tokens billed so far, as the API reports them. The offline job paces
        # itself on this rather than on an estimate.
        self.tokens_used = 0

    @classmethod
    def from_settings(cls, settings: Settings, **kwargs: Any) -> JinaAPIEmbedder:
        return cls(settings.require("jina_api_key"), **kwargs)

    def info(self) -> ModelInfo:
        return ModelInfo(
            model_id=MODEL_ID,
            dim=DIM,
            revision=REVISION,
            runtime=RUNTIME,
            encoding=ENCODING,
        )

    def embed(self, texts: Sequence[str], kind: EmbedKind) -> list[list[float]]:
        """One unit-length vector per input text, in input order.

        `kind` is required, never defaulted (ADR-069).
        """
        if not texts:
            return []
        # Query text crosses a trust boundary here, so it is redacted like every
        # other egress path (rule 03, path 4). Document text is public statute,
        # and redacting it would silently move a corpus vector wherever a
        # provision happens to print a nine-digit figure.
        inputs = [redact(text) for text in texts] if kind is EmbedKind.QUERY else list(texts)
        body = self._post(
            {
                "model": MODEL_ID,
                "task": TASKS[kind],
                "input": inputs,
                "dimensions": DIM,
                "normalized": True,
                "embedding_type": "float",
                # An over-length input must fail, not be cut: truncating a
                # provision drops exactly the qualifier that changes the answer.
                "truncate": False,
            }
        )
        # A malformed body is an API failure like any other, so it surfaces as
        # EmbeddingAPIError: Step 4.5 degrades on that type and must not see a
        # bare KeyError it cannot tell apart from a bug of ours.
        try:
            data = sorted(body["data"], key=lambda item: item["index"])
            indices = [item["index"] for item in data]
            vectors = [item["embedding"] for item in data]
        except (KeyError, TypeError) as error:
            raise EmbeddingAPIError(f"malformed embeddings response: {error!r}") from error
        if indices != list(range(len(texts))):
            raise EmbeddingAPIError(
                f"asked for {len(texts)} vectors, received indices {indices}"
            )
        widths = {len(vector) for vector in vectors}
        if widths != {DIM}:
            raise EmbeddingAPIError(f"expected width {DIM}, received {sorted(widths)}")
        self.tokens_used += int(body.get("usage", {}).get("total_tokens", 0))
        return [_unit(vector) for vector in vectors]

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        # The payload carries user query text and is never logged — not on
        # success, not on retry, not on failure (ADR-067's surviving half).
        last: Exception | None = None
        for attempt in range(1, self._max_attempts + 1):
            delay = self._backoff_base * 2 ** (attempt - 1)
            try:
                response = self._client.post(API_URL, json=payload, headers=self._headers)
            except httpx2.RequestError as error:
                last = error
            else:
                if response.status_code not in RETRYABLE_STATUS:
                    if response.status_code >= 400:
                        raise EmbeddingAPIError(
                            f"Jina embeddings API returned {response.status_code}: "
                            f"{response.text[:200]}"
                        )
                    try:
                        return response.json()
                    except ValueError as error:
                        raise EmbeddingAPIError(
                            "Jina embeddings API returned a body that is not JSON"
                        ) from error
                last = EmbeddingAPIError(
                    f"Jina embeddings API returned {response.status_code}"
                )
                delay = _retry_after(response) or delay
            if attempt < self._max_attempts:
                logger.warning(
                    "embedding API call failed (attempt %d/%d): %s; retrying in %.2fs",
                    attempt,
                    self._max_attempts,
                    last,
                    delay,
                )
                time.sleep(delay)
        raise EmbeddingAPIError(
            f"embedding API failed after {self._max_attempts} attempts: {last}"
        ) from last

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> JinaAPIEmbedder:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _retry_after(response: httpx2.Response) -> float | None:
    value = response.headers.get("retry-after")
    if value is None:
        return None
    try:
        return min(max(float(value), 0.0), MAX_RETRY_AFTER)
    except ValueError:
        return None


def _unit(vector: list[float]) -> list[float]:
    # Re-normalised in float64 even though the request asks for normalized
    # output: Step 4.5 treats a stored dot product as the cosine, so the norm
    # must be exact rather than whatever precision the server ran in.
    norm = math.sqrt(math.fsum(v * v for v in vector))
    if norm == 0.0:
        raise EmbeddingAPIError("received an all-zero vector")
    return [v / norm for v in vector]
