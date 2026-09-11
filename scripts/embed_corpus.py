"""Step 4.4 — offline: encode every chunk in the store to a document vector
through the Jina hosted embeddings API (ADR-075).

Never on a request path. Produces `data/vectors/jina-api/`; Step 4.5 searches it
and Step 4.6 measures it. Requires `TAXVERITY_JINA_API_KEY`.

    uv run python scripts/embed_corpus.py
    uv run python scripts/embed_corpus.py --tpm 1800000   # paid tier
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from taxverity.chunking.pipeline import read_corpus_version
from taxverity.chunking.store import load_chunks
from taxverity.config import MissingSettingError, Settings
from taxverity.embedding.backends import EmbedKind
from taxverity.embedding.jina_api import RUNTIME, JinaAPIEmbedder
from taxverity.embedding.store import (
    embed_probes,
    load_vector_store,
    verify_fingerprint,
    write_vector_store,
)
from taxverity.observability import configure_logging, get_logger

logger = get_logger(__name__)

VECTORS_KEY = "jina-api"

# Requests are packed by characters, not a fixed count. ~60k characters is ~15k
# tokens at this corpus's measured ~4.1 characters per token, which keeps each
# request inside the timeout, and lets section 2 (59k characters) travel as a
# request of its own.
MAX_BATCH = 64
BUDGET_CHARS = 60_000
# Under the free tier's 100K tokens per minute, so the job paces itself instead
# of living on 429s.
DEFAULT_TPM = 90_000
LOG_EVERY_BATCHES = 20


def _pack(texts: list[str]) -> list[list[str]]:
    batches: list[list[str]] = []
    current: list[str] = []
    chars = 0
    for text in texts:
        if current and (len(current) >= MAX_BATCH or chars + len(text) > BUDGET_CHARS):
            batches.append(current)
            current, chars = [], 0
        current.append(text)
        chars += len(text)
    if current:
        batches.append(current)
    return batches


def _encode_all(embedder: JinaAPIEmbedder, texts: list[str], tpm: int) -> np.ndarray:
    batches = _pack(texts)
    out: list[np.ndarray] = []
    started = time.perf_counter()
    spent_before = embedder.tokens_used
    for done, batch in enumerate(batches, start=1):
        # DOCUMENT is a literal here and the only document-side call site in the
        # system (ADR-069). It is never taken from an argument.
        vectors = embedder.embed(batch, EmbedKind.DOCUMENT)
        out.append(np.asarray(vectors, dtype=np.float32))
        # Paced on the tokens the API actually billed, not on an estimate.
        earliest = started + (embedder.tokens_used - spent_before) * 60.0 / tpm
        wait = earliest - time.perf_counter()
        if wait > 0:
            time.sleep(wait)
        if done % LOG_EVERY_BATCHES == 0 or done == len(batches):
            logger.info(
                "encoded %d/%d batches, %d chunks, %d tokens",
                done,
                len(batches),
                sum(len(b) for b in out),
                embedder.tokens_used - spent_before,
            )
    return np.vstack(out)


def main() -> int:
    configure_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--tpm", type=int, default=DEFAULT_TPM)
    args = parser.parse_args()

    settings = Settings()
    try:
        embedder = JinaAPIEmbedder.from_settings(settings)
    except MissingSettingError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    corpus_version = read_corpus_version(settings.interim_dir / "corpus_manifest.json")
    chunks, chunk_manifest = load_chunks(
        settings.interim_dir, corpus_version=corpus_version
    )
    texts = [chunk.embed_text() for chunk in chunks]
    chunk_ids = tuple(chunk.chunk_id for chunk in chunks)
    logger.info("loaded %d chunks for corpus_version %s", len(texts), corpus_version)

    destination = settings.vectors_dir / VECTORS_KEY
    with embedder:
        probes = embed_probes(embedder)
        started = time.perf_counter()
        vectors = _encode_all(embedder, texts, args.tpm)
        elapsed = time.perf_counter() - started
        manifest = write_vector_store(
            destination,
            vectors=vectors,
            chunk_ids=chunk_ids,
            corpus_version=corpus_version,
            doc_id=chunk_manifest.doc_id,
            model=embedder.info(),
            kind=EmbedKind.DOCUMENT,
            device=RUNTIME,
            probes=probes,
        )
        # Re-checked at the end, against what was written: an upstream model
        # change during a ~15-minute job would otherwise leave an index built
        # by two models.
        _, _, loaded = load_vector_store(destination, corpus_version=corpus_version)
        lowest = verify_fingerprint(embedder, loaded)
        tokens = embedder.tokens_used

    info = manifest.model
    print(f"model           {info.model_id} ({info.runtime}, {info.encoding})")
    print(f"chunks          {manifest.chunk_count}  dim {manifest.dim}")
    print(f"corpus_version  {corpus_version}")
    print(f"tokens billed   {tokens}")
    print(f"encode          {elapsed:.1f}s")
    print(f"fingerprint     lowest probe cosine {lowest:.6f}")
    print(f"vectors         {destination / 'corpus_vectors.npy'}")
    print(f"vectors_sha256  {manifest.vectors_sha256}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
