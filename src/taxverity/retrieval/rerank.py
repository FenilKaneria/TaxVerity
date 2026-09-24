"""Step 5.6 — cross-encoder reranking of the fused pool through Jina's hosted
rerank API (ADR-051, ADR-052, ADR-084).

Quality-additive, never correctness-critical: any vendor failure leaves the
fusion order in place and the query is answered normally (ADR-052). That
degradation is what justifies the vendor call at all, the same footing as the
dense leg's fallback to BM25 (ADR-075).
"""

from __future__ import annotations

import hashlib
import math
import threading
import time
from collections import OrderedDict
from collections.abc import Sequence
from typing import Any, Protocol

import httpx2

from taxverity.chunking.models import Chunk
from taxverity.config import Settings
from taxverity.corpus.loader import normalise
from taxverity.observability import get_logger, redact
from taxverity.retrieval.base import Retriever, ScoredChunk

logger = get_logger(__name__)

RERANK_STAGE_VERSION = 1

API_URL = "https://api.jina.ai/v1/rerank"
MODEL_ID = "jina-reranker-v3.5"

# The pool evidence delivery reads (EVIDENCE_POOL), so the reranker decides the
# whole of it, tail included (ADR-084).
RERANK_DEPTH = 20

# Equal to the registered p95 budget (ADR-084): on the query path a call slower
# than the budget is abandoned for the fusion order rather than waited on.
RERANK_TIMEOUT = 1.5

# One attempt on the query path: a retry only adds latency before the same
# fallback. The offline measurement raises this to ride out rate limits.
DEFAULT_MAX_ATTEMPTS = 1
DEFAULT_BACKOFF_BASE = 1.0
MAX_RETRY_AFTER = 60.0
RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})

RERANK_CACHE_SIZE = 256


class RerankError(RuntimeError):
    """Every vendor failure, whatever its cause. `RerankRetriever` degrades on
    exactly this type."""


class Reranker(Protocol):
    def score(self, query: str, chunks: Sequence[Chunk]) -> dict[str, float]:
        """A relevance score per chunk id, for every chunk given."""
        ...


class JinaReranker:
    def __init__(
        self,
        api_key: str,
        *,
        http_client: httpx2.Client | None = None,
        timeout: float = RERANK_TIMEOUT,
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
        self.tokens_used = 0
        # Duration of each successful request, backoff excluded: what the p95
        # budget is measured over.
        self.latencies_ms: list[float] = []

    @classmethod
    def from_settings(cls, settings: Settings, **kwargs: Any) -> JinaReranker:
        return cls(settings.require("jina_api_key"), **kwargs)

    def score(self, query: str, chunks: Sequence[Chunk]) -> dict[str, float]:
        if not chunks:
            return {}
        ids = [chunk.chunk_id for chunk in chunks]
        if len(set(ids)) != len(ids):
            raise ValueError("a chunk was given twice")
        body = self._post(
            {
                "model": MODEL_ID,
                # User text crossing a trust boundary (rule 03). The documents
                # are public statute and go verbatim.
                "query": redact(query),
                "documents": [chunk.embed_text() for chunk in chunks],
                "top_n": len(chunks),
                "return_documents": False,
            }
        )
        try:
            pairs = [
                (int(item["index"]), float(item["relevance_score"])) for item in body["results"]
            ]
        except (KeyError, TypeError, ValueError) as error:
            raise RerankError(f"malformed rerank response: {error!r}") from error
        if sorted(index for index, _ in pairs) != list(range(len(chunks))):
            raise RerankError(f"asked to score {len(chunks)} documents, received {len(pairs)}")
        if not all(math.isfinite(score) for _, score in pairs):
            raise RerankError("received a non-finite relevance score")
        self.tokens_used += int(body.get("usage", {}).get("total_tokens", 0))
        return {ids[index]: score for index, score in pairs}

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        # The payload carries user query text and is never logged.
        last: Exception | None = None
        for attempt in range(1, self._max_attempts + 1):
            delay = self._backoff_base * 2 ** (attempt - 1)
            started = time.perf_counter()
            try:
                response = self._client.post(API_URL, json=payload, headers=self._headers)
            except httpx2.RequestError as error:
                last = error
            else:
                if response.status_code not in RETRYABLE_STATUS:
                    if response.status_code >= 400:
                        raise RerankError(
                            f"Jina rerank API returned {response.status_code}: "
                            f"{response.text[:200]}"
                        )
                    try:
                        body = response.json()
                    except ValueError as error:
                        raise RerankError("Jina rerank API returned a body that is not JSON") from error
                    elapsed = (time.perf_counter() - started) * 1000
                    self.latencies_ms.append(elapsed)
                    logger.debug(
                        "reranked %d documents in %.0f ms",
                        len(payload["documents"]),
                        elapsed,
                    )
                    return body
                last = RerankError(f"Jina rerank API returned {response.status_code}")
                delay = _retry_after(response) or delay
            if attempt < self._max_attempts:
                logger.warning(
                    "rerank API call failed (attempt %d/%d): %s; retrying in %.2fs",
                    attempt,
                    self._max_attempts,
                    last,
                    delay,
                )
                time.sleep(delay)
        raise RerankError(f"rerank API failed after {self._max_attempts} attempt(s): {last}") from last

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> JinaReranker:
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


class CachedReranker:
    """An in-process LRU over another reranker (ADR-052, ADR-084).

    Keyed by the query as it is sent, normalised and redacted, and by the
    candidate set rather than its order: the same question over the same pool
    is the same request. A failure is never cached, so the next call retries.
    """

    def __init__(self, inner: Reranker, *, maxsize: int = RERANK_CACHE_SIZE) -> None:
        if maxsize < 1:
            raise ValueError(f"maxsize must be at least 1, not {maxsize}")
        self._inner = inner
        self._maxsize = maxsize
        self._entries: OrderedDict[tuple[str, frozenset[str]], dict[str, float]] = OrderedDict()
        # R21 Part B: sub-query searches run on two threads and share this
        # cache; the lock guards the OrderedDict, never the vendor call.
        self._lock = threading.Lock()

    def __len__(self) -> int:
        return len(self._entries)

    def score(self, query: str, chunks: Sequence[Chunk]) -> dict[str, float]:
        digest = hashlib.sha256(redact(normalise(query)).encode("utf-8")).hexdigest()
        key = (digest, frozenset(chunk.chunk_id for chunk in chunks))
        with self._lock:
            if key in self._entries:
                self._entries.move_to_end(key)
                return dict(self._entries[key])
        scores = self._inner.score(query, chunks)
        with self._lock:
            self._entries[key] = dict(scores)
            if len(self._entries) > self._maxsize:
                self._entries.popitem(last=False)
        return scores


class RerankRetriever:
    """Satisfies the Step 3.3 `Retriever` Protocol.

    Reorders the first `depth` results of the primary ranking by relevance and
    keeps the rest behind them in their own order. Sits inside the citation
    shortcut, so an exact citation hit is never reranked (ADR-062).

    Its scores are ordinal: a relevance score and a fusion score share no scale,
    so they rank this retriever's own output and say nothing else (Step 3.3).
    """

    def __init__(self, primary: Retriever, reranker: Reranker, *, depth: int = RERANK_DEPTH) -> None:
        if depth < 1:
            raise ValueError(f"depth must be at least 1, not {depth}")
        self._primary = primary
        self._reranker = reranker
        self._depth = depth

    def search(self, query: str, k: int) -> Sequence[ScoredChunk]:
        if k < 1:
            raise ValueError(f"k must be at least 1, not {k}")
        results = list(self._primary.search(query, max(k, self._depth)))
        head, tail = results[: self._depth], results[self._depth :]
        if not head:
            return []
        try:
            scores = self._reranker.score(query, [result.chunk for result in head])
        except RerankError as error:
            # The query text is never logged: it is user input (rule 03).
            logger.warning("rerank unavailable, keeping the fusion order: %s", error)
            return results[:k]
        order = sorted(range(len(head)), key=lambda i: (-scores[head[i].chunk.chunk_id], i))
        ranked = [head[i].chunk for i in order] + [result.chunk for result in tail]
        return [
            ScoredChunk(chunk=chunk, score=float(len(ranked) - position))
            for position, chunk in enumerate(ranked[:k])
        ]
