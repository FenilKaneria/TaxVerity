"""Step 4.3 — the caller's half of the embedding contract: batching, bounded
retry, and the model-identity assertion ADR-026 requires.

Synchronous by choice. The offline job (Step 4.4) is a batch loop, vector
search (Step 4.5) is synchronous NumPy, and the Phase 14 API runs a sync
dependency in a threadpool; an async client would be structure built ahead of
a caller that wants it.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

import httpx2

from taxverity.embedding.backends import MAX_BATCH, EmbedKind, ModelInfo
from taxverity.observability import get_logger

logger = get_logger(__name__)

DEFAULT_TIMEOUT = 30.0
DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_BACKOFF_BASE = 0.5

# 503 is the readiness gate and the failed-warm-up refusal (Step 4.2); the
# first is transient by design and the second resolves when the task is
# replaced. A 4xx is the client's own fault and repeating it changes nothing.
RETRYABLE_STATUS = frozenset({429, 502, 503, 504})


class EmbeddingClientError(RuntimeError):
    pass


class ModelIdentityError(EmbeddingClientError):
    """The service is not serving the weights this client was told to expect.

    Deliberately outside the retry path: a second call to the same service
    returns the same wrong model, and its vectors are silently incomparable
    with the index they would be searched against.
    """


class EmbedKindError(EmbeddingClientError):
    """The service encoded the text as the other side of the retrieval pair.

    Outside the retry path for the same reason as ModelIdentityError: the same
    service applies the same recipe on a repeat call, and a query encoded as a
    document is silently comparable-looking against the index (ADR-069).
    """


class EmbeddingServiceError(EmbeddingClientError):
    pass


def describe(info: ModelInfo) -> str:
    return (
        f"{info.model_id}@{info.revision} dim={info.dim} "
        f"runtime={info.runtime} encoding={info.encoding}"
    )


class EmbeddingClient:
    def __init__(
        self,
        base_url: str | None = None,
        *,
        http_client: httpx2.Client | None = None,
        expected: ModelInfo | None = None,
        batch_size: int = MAX_BATCH,
        timeout: float = DEFAULT_TIMEOUT,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        backoff_base: float = DEFAULT_BACKOFF_BASE,
    ) -> None:
        if (base_url is None) == (http_client is None):
            raise ValueError("pass exactly one of base_url or http_client")
        if not 1 <= batch_size <= MAX_BATCH:
            raise ValueError(f"batch_size must be in 1..{MAX_BATCH}, not {batch_size}")
        if max_attempts < 1:
            raise ValueError(f"max_attempts must be at least 1, not {max_attempts}")
        self._batch_size = batch_size
        self._max_attempts = max_attempts
        self._backoff_base = backoff_base
        # A supplied client is borrowed, not owned: a caller passing its own
        # pooled client does not expect us to close it underneath.
        self._owns_client = http_client is None
        self._client = http_client or httpx2.Client(base_url=base_url, timeout=timeout)
        # Pinned eagerly so a mismatch surfaces at startup rather than at the
        # first query. `/model-info` is ungated (Steps 4.1, 4.2), so this
        # succeeds while the service is still warming up.
        self.model = self._pin_identity(expected)

    def _pin_identity(self, expected: ModelInfo | None) -> ModelInfo:
        served = ModelInfo.model_validate(self._request("GET", "/model-info"))
        if expected is not None and served != expected:
            raise ModelIdentityError(
                f"embedding service at {self._client.base_url} serves "
                f"{describe(served)}, expected {describe(expected)}"
            )
        logger.info(
            "embedding client pinned to %s at %s",
            describe(served),
            self._client.base_url,
        )
        return served

    def embed(self, texts: Sequence[str], kind: EmbedKind) -> list[list[float]]:
        """One unit-length vector per input text, in input order.

        `kind` is required and never defaulted (ADR-069): ingestion passes
        DOCUMENT, the query path passes QUERY, and there is no third caller.
        """
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            vectors.extend(
                self._embed_batch(list(texts[start : start + self._batch_size]), kind)
            )
        return vectors

    def _embed_batch(self, batch: list[str], kind: EmbedKind) -> list[list[float]]:
        payload = self._request(
            "POST", "/embed", json={"texts": batch, "kind": kind.value}
        )
        served_kind = payload["kind"]
        if served_kind != kind.value:
            raise EmbedKindError(
                f"asked the embedding service for kind={kind.value}, "
                f"it applied kind={served_kind}"
            )
        served = ModelInfo.model_validate(payload["model"])
        # Checked per response, not only at construction: a rolling deploy can
        # replace the service under a long-lived client, which is precisely
        # when skew appears and precisely what the per-response echo is for.
        if served != self.model:
            raise ModelIdentityError(
                f"embedding service changed identity mid-session: pinned to "
                f"{describe(self.model)}, now serving {describe(served)}"
            )
        vectors = payload["embeddings"]
        if len(vectors) != len(batch):
            raise EmbeddingServiceError(
                f"asked for {len(batch)} vectors, received {len(vectors)}"
            )
        widths = {len(vector) for vector in vectors}
        if widths != {self.model.dim}:
            raise EmbeddingServiceError(
                f"service declares dim={self.model.dim} but returned widths "
                f"{sorted(widths)}"
            )
        return vectors

    def _request(self, method: str, path: str, *, json: Any = None) -> Any:
        # The request body carries user query text and is never logged, on the
        # same rule as the service's own /embed handler (Step 4.1 note 4). It
        # is also not passed through redact(): redaction would change the text
        # and therefore change the vector, and this service is inside our own
        # trust boundary rather than a vendor's. See ADR-067.
        last: Exception | None = None
        for attempt in range(1, self._max_attempts + 1):
            try:
                response = self._client.request(method, path, json=json)
                if response.status_code in RETRYABLE_STATUS:
                    raise EmbeddingServiceError(
                        f"{method} {path} returned {response.status_code}"
                    )
                response.raise_for_status()
                return response.json()
            except httpx2.HTTPStatusError as error:
                raise EmbeddingServiceError(
                    f"{method} {path} failed: {error}"
                ) from error
            except (httpx2.RequestError, EmbeddingServiceError) as error:
                last = error
            if attempt < self._max_attempts:
                delay = self._backoff_base * 2 ** (attempt - 1)
                logger.warning(
                    "embed %s %s failed (attempt %d/%d): %s; retrying in %.2fs",
                    method,
                    path,
                    attempt,
                    self._max_attempts,
                    last,
                    delay,
                )
                time.sleep(delay)
        raise EmbeddingServiceError(
            f"{method} {path} failed after {self._max_attempts} attempts: {last}"
        ) from last

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> EmbeddingClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
