"""Step 4.4 — offline: encode every chunk in the store to a document vector.

Runs on the development GPU (ADR-070), never on a request path. Produces one
vector store per candidate model under `data/vectors/<key>/`; Step 4.5 searches
it and Step 4.6 measures it. Requires `taxverity[embed]` and the model snapshot
(`scripts/download_models.py`).

    uv run python scripts/embed_corpus.py            # jina-v5 (leading candidate)
    uv run python scripts/embed_corpus.py --model qwen3-0.6b
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import NamedTuple

import numpy as np

from taxverity.chunking.pipeline import read_corpus_version
from taxverity.chunking.store import load_chunks
from taxverity.config import Settings
from taxverity.embedding.backends import EmbedKind, ModelInfo
from taxverity.embedding.candidates import CANDIDATES, EmbedderSpec
from taxverity.embedding.sentence_transformer import SentenceTransformerEmbedder
from taxverity.embedding.store import write_vector_store
from taxverity.observability import configure_logging, get_logger

logger = get_logger(__name__)

# Batches are packed by a character budget, not a fixed count: self-attention
# memory is quadratic in sequence length and this build has only the O(n^2)
# math kernel, so a fixed count OOMs the 6 GB card on the ~10 chunks that run
# past a few thousand tokens (section 2 is 14.6k tokens, unsplit — ADR-056).
# The budget keeps a normal GPU batch well inside memory; an over-budget chunk
# is encoded alone and, if the GPU still cannot hold it, on the CPU.
MAX_BATCH = 32
BUDGET_CHARS = 10000
LOG_EVERY_BATCHES = 50

_SPECS = {spec.key: spec for spec in CANDIDATES}


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


class _Result(NamedTuple):
    vectors: np.ndarray
    info: ModelInfo
    device: str
    cpu_fallback_chunks: int


def _encode_all(spec: EmbedderSpec, texts: list[str]) -> _Result:
    import torch

    primary = SentenceTransformerEmbedder(spec)
    # This Windows torch build has no flash / GQA-aware memory-efficient SDPA
    # kernel, so a chunk past ~6k tokens forces the O(n^2) math path and OOMs
    # the 6 GB card. ~20 chunks in this corpus are that large (section 2 is
    # 14.6k tokens, unsplit — ADR-056). They are re-encoded on the CPU rather
    # than truncated: the plan chose a 32K context precisely so nothing is cut.
    fallback: SentenceTransformerEmbedder | None = None
    batches = _pack(texts)
    out: list[np.ndarray] = []
    fell_back = 0
    started = time.perf_counter()
    for done, batch in enumerate(batches, start=1):
        # DOCUMENT is a literal here and the only document-side call site in the
        # system (ADR-069). It is never taken from an argument.
        try:
            vecs = primary.embed(batch, EmbedKind.DOCUMENT)
        except torch.OutOfMemoryError:
            if primary.device == "cpu":
                raise
            torch.cuda.empty_cache()
            if fallback is None:
                logger.warning("CUDA OOM on a batch — bringing up a CPU embedder")
                fallback = SentenceTransformerEmbedder(spec, device="cpu")
            vecs = fallback.embed(batch, EmbedKind.DOCUMENT)
            fell_back += len(batch)
        out.append(np.asarray(vecs, dtype=np.float32))
        if done % LOG_EVERY_BATCHES == 0 or done == len(batches):
            encoded = sum(len(b) for b in out)
            rate = encoded / (time.perf_counter() - started)
            logger.info(
                "encoded %d/%d batches, %d chunks (%.0f chunks/s)",
                done,
                len(batches),
                encoded,
                rate,
            )
    device = primary.device if not fell_back else f"{primary.device}+cpu"
    return _Result(np.vstack(out), primary.info(), device, fell_back)


def main() -> int:
    configure_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=sorted(_SPECS), default="jina-v5")
    args = parser.parse_args()
    spec = _SPECS[args.model]

    # Let the allocator hand memory back between the large-chunk batches instead
    # of fragmenting. Must be set before torch is first imported. Determinism is
    # deliberately not forced (ADR-073): torch's deterministic mode disables the
    # O(n) flash-attention kernel, and the O(n^2) fallback OOMs this card on the
    # corpus's long chunks.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    settings = Settings()
    corpus_version = read_corpus_version(settings.interim_dir / "corpus_manifest.json")
    chunks, chunk_manifest = load_chunks(
        settings.interim_dir, corpus_version=corpus_version
    )
    texts = [chunk.embed_text() for chunk in chunks]
    chunk_ids = tuple(chunk.chunk_id for chunk in chunks)
    logger.info("loaded %d chunks for corpus_version %s", len(texts), corpus_version)

    started = time.perf_counter()
    result = _encode_all(spec, texts)
    elapsed = time.perf_counter() - started

    destination = settings.vectors_dir / spec.key
    manifest = write_vector_store(
        destination,
        vectors=result.vectors,
        chunk_ids=chunk_ids,
        corpus_version=corpus_version,
        doc_id=chunk_manifest.doc_id,
        model=result.info,
        kind=EmbedKind.DOCUMENT,
        device=result.device,
    )

    print(f"model           {spec.key}  {spec.model_id}@{spec.revision[:12]}")
    print(
        f"device          {result.device}  (cpu fallback: {result.cpu_fallback_chunks} chunks)"
    )
    print(f"chunks          {manifest.chunk_count}  dim {manifest.dim}")
    print(f"corpus_version  {corpus_version}")
    print(
        f"encode          {elapsed:.1f}s ({manifest.chunk_count / elapsed:.0f} chunks/s)"
    )
    print(f"vectors         {destination / 'corpus_vectors.npy'}")
    print(f"vectors_sha256  {manifest.vectors_sha256}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
