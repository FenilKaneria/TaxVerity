"""Step 3.4 — in-process BM25 over the Step 2.4 chunk store: the lexical
baseline every later retrieval component is measured against."""

from __future__ import annotations

import math
import re
import time
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence

from taxverity.chunking.models import Chunk
from taxverity.corpus.loader import normalise
from taxverity.observability import get_logger
from taxverity.retrieval.base import ScoredChunk

logger = get_logger(__name__)

BM25_STAGE_VERSION = 2

# Chosen at Step 4.4b by two-fold cross-validation on the gold set (ADR-077).
# b fell from Okapi's 0.75 to 0.3: chunks here run from a `[***]` mark to a
# 59,000-character root, and the full length penalty buried the long provisions
# that carry whole answers. 0.3 is the edge of the pre-registered grid, not a
# measured optimum; widening the grid after seeing it would fit the ruler.
# k1 moved nothing measurable and stays at Okapi's 1.5.
K1 = 1.5
B = 0.3

# An alphanumeric run, so a letter-suffixed number survives whole: 354A and 80C
# are single statutory tokens, and splitting them loses the only thing that
# distinguishes them from 354 and 80.
TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return TOKEN.findall(normalise(text).casefold())


class BM25Retriever:
    """Satisfies the Step 3.3 `Retriever` Protocol."""

    def __init__(
        self,
        chunks: Iterable[Chunk],
        *,
        k1: float = K1,
        b: float = B,
        tokenizer: Callable[[str], list[str]] = tokenize,
    ) -> None:
        if k1 < 0 or not 0 <= b <= 1:
            raise ValueError(f"need k1 >= 0 and 0 <= b <= 1, not k1={k1}, b={b}")
        started = time.perf_counter()
        self._chunks = tuple(chunks)
        self._k1 = k1
        self._b = b
        self.tokenize = tokenizer
        # The breadcrumb is indexed along with the text: it is the surface Phase 4
        # embeds, so the two retrievers stay comparable, and a section title is a
        # real lexical signal for a question that does not quote the statute.
        self._documents = [
            Counter(tokenizer(chunk.embed_text())) for chunk in self._chunks
        ]
        self._lengths = [document.total() for document in self._documents]
        self._position = {chunk.chunk_id: i for i, chunk in enumerate(self._chunks)}
        total = sum(self._lengths)
        self._average_length = total / len(self._documents) if self._documents else 0.0

        self._postings: dict[str, dict[int, int]] = {}
        for index, document in enumerate(self._documents):
            for token, frequency in document.items():
                self._postings.setdefault(token, {})[index] = frequency

        count = len(self._documents)
        # The positive-idf variant. Robertson's original goes negative once a term
        # appears in more than half the corpus, and "income" is close enough to
        # that here that a common term could subtract from a document's score.
        self._idf = {
            token: math.log(1 + (count - len(postings) + 0.5) / (len(postings) + 0.5))
            for token, postings in self._postings.items()
        }
        logger.info(
            "indexed %d chunks, %d tokens (%d distinct), avg length %.1f, "
            "k1=%.2f b=%.2f, in %.2fs",
            count,
            total,
            len(self._postings),
            self._average_length,
            k1,
            b,
            time.perf_counter() - started,
        )

    def __len__(self) -> int:
        return len(self._chunks)

    def idf(self, token: str) -> float:
        return self._idf.get(token, 0.0)

    def term_frequencies(self, chunk: Chunk) -> Mapping[str, int]:
        return self._documents[self._position[chunk.chunk_id]]

    def search(self, query: str, k: int) -> Sequence[ScoredChunk]:
        # Deduplicated, so a query repeating a word does not weight it twice: the
        # user's phrasing is not evidence about the corpus.
        return self.search_weighted(dict.fromkeys(self.tokenize(query), 1.0), k)

    def search_weighted(
        self, term_weights: Mapping[str, float], k: int
    ) -> Sequence[ScoredChunk]:
        if k < 1:
            raise ValueError(f"k must be at least 1, not {k}")
        scores: dict[int, float] = {}
        for token, query_weight in term_weights.items():
            postings = self._postings.get(token)
            if postings is None:
                continue
            idf = self._idf[token]
            for index, frequency in postings.items():
                norm = 1 - self._b + self._b * self._lengths[index] / self._average_length
                weight = frequency * (self._k1 + 1) / (frequency + self._k1 * norm)
                scores[index] = scores.get(index, 0.0) + query_weight * idf * weight

        # Corpus order breaks a tie, so a run is reproducible: the chunk store is
        # byte-identical across builds, and ranking must be too.
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:k]
        return [
            ScoredChunk(chunk=self._chunks[index], score=score)
            for index, score in ranked
        ]
