"""Step 1.2 driver — extract the corpus to a per-page JSONL artifact plus a manifest."""

import sys

from taxverity.config import Settings
from taxverity.corpus.loader import (
    EXTRACT_STAGE_VERSION,
    build_manifest,
    extract_pages,
    write_manifest,
    write_pages_jsonl,
)


def main():
    settings = Settings()
    pdf_path = settings.resolve_corpus_pdf()
    out_dir = settings.interim_dir
    pages_path = out_dir / "pages.jsonl"
    manifest_path = out_dir / "corpus_manifest.json"

    print(f"Extracting {pdf_path}", file=sys.stderr)
    page_count, artifact_sha256 = write_pages_jsonl(extract_pages(pdf_path), pages_path)

    manifest = build_manifest(
        pdf_path, page_count, artifact_sha256, {"extract": EXTRACT_STAGE_VERSION}
    )
    write_manifest(manifest, manifest_path)

    size_mb = pages_path.stat().st_size / 1024**2
    print(
        f"Wrote {page_count} pages to {pages_path} ({size_mb:.1f} MiB)", file=sys.stderr
    )
    print(f"  artifact sha256 {artifact_sha256}", file=sys.stderr)
    print(f"  source   sha256 {manifest.source_sha256}", file=sys.stderr)
    print(f"Wrote {manifest_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
