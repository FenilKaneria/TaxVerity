import hashlib
import json
from collections.abc import Iterable, Iterator
from pathlib import Path

import pymupdf

from taxverity.corpus.models import (
    BBOX_PRECISION,
    SIZE_PRECISION,
    CorpusManifest,
    PageArtifact,
    Span,
)
from taxverity.observability import get_logger

logger = get_logger(__name__)

EXTRACT_STAGE_VERSION = 1

BOLD_FLAG = 1 << 4
ITALIC_FLAG = 1 << 1

# Identified in the Step 1.1 profile. Matched by string, never by font:
# NotoSans carries the running footer but also renders some clause markers,
# so a font-based filter would delete real statutory content.
FURNITURE_EXACT = frozenset(
    {
        "Income Tax Department",
        "Ministry of Finance, Government of India",
    }
)
FURNITURE_PREFIXES = ("Downloaded/Printed on",)

HASH_CHUNK_BYTES = 1 << 20


def normalise(text: str) -> str:
    """Strip the soft hyphen and NBSP that defeat otherwise-clean matches (Step 1.1)."""
    return text.replace("\xad", "").replace("\xa0", " ")


def is_furniture(text: str) -> bool:
    stripped = normalise(text).strip()
    if stripped in FURNITURE_EXACT:
        return True
    return any(stripped.startswith(prefix) for prefix in FURNITURE_PREFIXES)


def span_from_dict(raw: dict) -> Span:
    return Span(
        text=raw["text"],
        font=raw["font"],
        size=round(float(raw["size"]), SIZE_PRECISION),
        bold=bool(raw["flags"] & BOLD_FLAG),
        italic=bool(raw["flags"] & ITALIC_FLAG),
        bbox=tuple(round(float(value), BBOX_PRECISION) for value in raw["bbox"]),
        furniture=is_furniture(raw["text"]),
    )


def page_artifact_from_dict(
    page_number: int, text: str, blocks: Iterable[dict]
) -> PageArtifact:
    spans = [
        span_from_dict(span)
        for block in blocks
        for line in block.get("lines", [])
        for span in line["spans"]
        if span["text"].strip()
    ]
    return PageArtifact(page=page_number, text=text, spans=tuple(spans))


def extract_pages(pdf_path: Path) -> Iterator[PageArtifact]:
    with pymupdf.open(pdf_path) as doc:
        logger.info("extracting %d pages from %s", doc.page_count, pdf_path.name)
        for index, page in enumerate(doc):
            yield page_artifact_from_dict(
                index, page.get_text("text"), page.get_text("dict")["blocks"]
            )


def to_json_line(artifact: PageArtifact) -> str:
    return json.dumps(
        artifact.model_dump(mode="json"),
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def write_pages_jsonl(
    artifacts: Iterable[PageArtifact], destination: Path
) -> tuple[int, str]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    count = 0
    # newline="" so Python does not translate LF to CRLF on Windows, which
    # would make the artifact and its hash platform-dependent.
    with destination.open("w", encoding="utf-8", newline="") as handle:
        for artifact in artifacts:
            line = to_json_line(artifact) + "\n"
            handle.write(line)
            digest.update(line.encode("utf-8"))
            count += 1
    artifact_sha256 = digest.hexdigest()
    logger.info(
        "wrote %d pages to %s (sha256 %s)", count, destination, artifact_sha256
    )
    return count, artifact_sha256


def read_pages_jsonl(source: Path) -> Iterator[PageArtifact]:
    # Logged on entry, not on exhaustion: every caller that stops at the first
    # Schedule page abandons this generator, so a completion log would be
    # missing precisely where the read did happen.
    logger.info("reading pages from %s", source)
    with source.open(encoding="utf-8", newline="") as handle:
        for line in handle:
            if line.strip():
                yield PageArtifact.model_validate_json(line)


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def compute_corpus_version(
    source_sha256: str, stage_versions: dict[str, int], artifact_sha256: str
) -> str:
    """Hash every input a downstream consumer must agree on to trust a chunk id."""
    payload = json.dumps(
        {
            "source_sha256": source_sha256,
            "stage_versions": stage_versions,
            "artifact_sha256": artifact_sha256,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_manifest(
    pdf_path: Path,
    page_count: int,
    artifact_sha256: str,
    stage_versions: dict[str, int],
) -> CorpusManifest:
    source_sha256 = hash_file(pdf_path)
    return CorpusManifest(
        source_name=pdf_path.name,
        source_sha256=source_sha256,
        source_bytes=pdf_path.stat().st_size,
        page_count=page_count,
        extractor="pymupdf",
        extractor_version=pymupdf.__version__,
        stage_versions=stage_versions,
        artifact_sha256=artifact_sha256,
        corpus_version=compute_corpus_version(
            source_sha256, stage_versions, artifact_sha256
        ),
    )


def write_manifest(manifest: CorpusManifest, destination: Path) -> None:
    logger.info("corpus_version %s", manifest.corpus_version)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        manifest.model_dump(mode="json"), sort_keys=True, ensure_ascii=False, indent=2
    )
    with destination.open("w", encoding="utf-8", newline="") as handle:
        handle.write(payload + "\n")


def read_manifest(source: Path) -> CorpusManifest:
    return CorpusManifest.model_validate_json(source.read_text(encoding="utf-8"))
