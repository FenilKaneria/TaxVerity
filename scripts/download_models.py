"""Step 4.4 — fetch both Step 4.7 embedding finalists at their pinned revisions.

One-time setup. Writes into the Hugging Face hub cache, not the repo. Re-running
is a no-op once the snapshots are present. Requires `taxverity[embed]`.
"""

from __future__ import annotations

import sys
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

from taxverity.embedding.candidates import (
    CANDIDATES,
    DOWNLOAD_ALLOW_PATTERNS,
    EmbedderSpec,
)
from taxverity.observability import configure_logging, get_logger

logger = get_logger(__name__)


def _dir_bytes(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def fetch(spec: EmbedderSpec) -> tuple[Path, int]:
    resolved = HfApi().model_info(spec.model_id, revision=spec.revision).sha
    if resolved != spec.revision:
        raise RuntimeError(
            f"{spec.model_id}: pinned revision {spec.revision} resolved to "
            f"{resolved} on the hub"
        )
    logger.info("downloading %s@%s", spec.model_id, spec.revision)
    local = snapshot_download(
        spec.model_id,
        revision=spec.revision,
        allow_patterns=list(DOWNLOAD_ALLOW_PATTERNS),
    )
    path = Path(local)
    size = _dir_bytes(path)
    logger.info("have %s (%.0f MiB) at %s", spec.key, size / 1024 / 1024, path)
    return path, size


def main() -> int:
    configure_logging()
    for spec in CANDIDATES:
        path, size = fetch(spec)
        print(f"{spec.key:12s} {spec.model_id}@{spec.revision[:12]}")
        print(f"{'':12s} {size / 1024 / 1024:.0f} MiB  {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
