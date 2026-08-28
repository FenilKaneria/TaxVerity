"""Step 1.2 driver — extract the corpus to a per-page JSONL artifact plus a manifest."""

from taxverity.config import Settings
from taxverity.corpus.crossrefs import CROSSREF_STAGE_VERSION
from taxverity.corpus.loader import (
    EXTRACT_STAGE_VERSION,
    build_manifest,
    extract_pages,
    write_manifest,
    write_pages_jsonl,
)
from taxverity.corpus.schedules import SCHEDULE_STAGE_VERSION
from taxverity.corpus.sections import PARSE_STAGE_VERSION
from taxverity.corpus.substructure import SUBSTRUCTURE_STAGE_VERSION
from taxverity.corpus.tables import TABLE_STAGE_VERSION
from taxverity.observability import configure_logging, get_logger

logger = get_logger(__name__)

# corpus_version must move whenever any stage's parsing logic changes, not just
# extraction — the later stages re-derive from pages.jsonl at run time rather
# than writing their own artifact (Step 1.9), so their version is the only
# signal a downstream consumer has that their output would differ.
STAGE_VERSIONS = {
    "extract": EXTRACT_STAGE_VERSION,
    "parse": PARSE_STAGE_VERSION,
    "substructure": SUBSTRUCTURE_STAGE_VERSION,
    "table": TABLE_STAGE_VERSION,
    "schedule": SCHEDULE_STAGE_VERSION,
    "crossref": CROSSREF_STAGE_VERSION,
}


def main():
    settings = Settings()
    configure_logging(settings)
    pdf_path = settings.resolve_corpus_pdf()
    out_dir = settings.interim_dir
    pages_path = out_dir / "pages.jsonl"
    manifest_path = out_dir / "corpus_manifest.json"

    page_count, artifact_sha256 = write_pages_jsonl(extract_pages(pdf_path), pages_path)

    manifest = build_manifest(pdf_path, page_count, artifact_sha256, STAGE_VERSIONS)
    write_manifest(manifest, manifest_path)

    logger.info(
        "artifact is %.1f MiB; source sha256 %s",
        pages_path.stat().st_size / 1024**2,
        manifest.source_sha256,
    )
    logger.info("wrote manifest %s", manifest_path)


if __name__ == "__main__":
    main()
