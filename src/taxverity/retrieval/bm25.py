"""Step 3.4 — in-process BM25 over the Step 2.4 chunk store: the lexical
baseline every later retrieval component is measured against."""

from __future__ import annotations

import math
import re
import time
from collections import Counter
from collections.abc import Iterable, Sequence

from taxverity.chunking.models import Chunk
from taxverity.corpus.loader import normalise
from taxverity.observability import get_logger
from taxverity.retrieval.base import ScoredChunk

logger = get_logger(__name__)

BM25_STAGE_VERSION = 1

# Okapi's usual defaults. Untuned on purpose: Step 3.6 produces the first
# measurement, and tuning against the gold set before that number exists would
# leave nothing to compare a tuned run to.
K1 = 1.5
B = 0.75

# An alphanumeric run, so a letter-suffixed number survives whole: 354A and 80C
# are single statutory tokens, and splitting them loses the only thing that
# distinguishes them from 354 and 80.
TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return TOKEN.findall(normalise(text).casefold())


class BM25Retriever:
    """Satisfies the Step 3.3 `Retriever` Protocol."""

    def __init__(self, chunks: Iterable[Chunk]) -> None:
        started = time.perf_counter()
        self._chunks = tuple(chunks)
        # The breadcrumb is indexed along with the text: it is the surface Phase 4
        # embeds, so the two retrievers stay comparable, and a section title is a
        # real lexical signal for a question that does not quote the statute.
        documents = [tokenize(chunk.embed_text()) for chunk in self._chunks]
        self._lengths = [len(document) for document in documents]
        total = sum(self._lengths)
        self._average_length = total / len(documents) if documents else 0.0

        self._postings: dict[str, dict[int, int]] = {}
        for index, document in enumerate(documents):
            for token, frequency in Counter(document).items():
                self._postings.setdefault(token, {})[index] = frequency

        count = len(documents)
        # The positive-idf variant. Robertson's original goes negative once a term
        # appears in more than half the corpus, and "income" is close enough to
        # that here that a common term could subtract from a document's score.
        self._idf = {
            token: math.log(1 + (count - len(postings) + 0.5) / (len(postings) + 0.5))
            for token, postings in self._postings.items()
        }
        logger.info(
            "indexed %d chunks, %d tokens (%d distinct), avg length %.1f, in %.2fs",
            count,
            total,
            len(self._postings),
            self._average_length,
            time.perf_counter() - started,
        )

    def __len__(self) -> int:
        return len(self._chunks)

    def search(self, query: str, k: int) -> Sequence[ScoredChunk]:
        if k < 1:
            raise ValueError(f"k must be at least 1, not {k}")
        scores: dict[int, float] = {}
        # Deduplicated, so a query repeating a word does not weight it twice: the
        # user's phrasing is not evidence about the corpus.
        for token in dict.fromkeys(tokenize(query)):
            postings = self._postings.get(token)
            if postings is None:
                continue
            idf = self._idf[token]
            for index, frequency in postings.items():
                norm = 1 - B + B * self._lengths[index] / self._average_length
                weight = frequency * (K1 + 1) / (frequency + K1 * norm)
                scores[index] = scores.get(index, 0.0) + idf * weight

        # Corpus order breaks a tie, so a run is reproducible: the chunk store is
        # byte-identical across builds, and ranking must be too.
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:k]
        return [
            ScoredChunk(chunk=self._chunks[index], score=score)
            for index, score in ranked
        ]
