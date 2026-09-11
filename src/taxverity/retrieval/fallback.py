"""Step 5.2 — the dense leg's required degradation (ADR-075): when the vendor
call fails, the query is answered from the lexical path instead of failing."""

from __future__ import annotations

from collections.abc import Sequence

from taxverity.observability import get_logger
from taxverity.retrieval.base import Retriever, ScoredChunk
from taxverity.retrieval.dense import DenseRetrievalError

logger = get_logger(__name__)


class FallbackRetriever:
    """Satisfies the Step 3.3 `Retriever` Protocol.

    Catches exactly `DenseRetrievalError`, the one type the dense path converts
    vendor failure into (ADR-078). Anything else is a bug and propagates, since
    degrading on it would hide the bug behind a working-looking answer.
    """

    def __init__(self, primary: Retriever, fallback: Retriever) -> None:
        self._primary = primary
        self._fallback = fallback

    def search(self, query: str, k: int) -> Sequence[ScoredChunk]:
        if k < 1:
            raise ValueError(f"k must be at least 1, not {k}")
        try:
            return self._primary.search(query, k)
        except DenseRetrievalError as error:
            # The query text is never logged: it is user input (Rule 03).
            logger.warning("dense retrieval unavailable, answering from BM25: %s", error)
            return self._fallback.search(query, k)
