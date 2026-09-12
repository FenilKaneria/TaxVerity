"""Step 6.6 — run the startup guard against TAXVERITY_DATABASE_URL.

The same call Phase 14's API makes before it serves, so a deploy can be checked
without starting one. Exit 1 on refusal.
"""

from __future__ import annotations

import sys

import psycopg

from taxverity.config import Settings
from taxverity.db.serving import ServingRefusal, resolve_serving, serving_target
from taxverity.embedding.backends import describe
from taxverity.observability import configure_logging


def main() -> int:
    configure_logging()
    settings = Settings()
    corpus_version, embedding_set_id = serving_target(settings)
    with psycopg.connect(settings.require("database_url")) as conn:
        try:
            served = resolve_serving(
                conn,
                corpus_version=corpus_version,
                embedding_set_id=embedding_set_id,
            )
        except ServingRefusal as refusal:
            print("REFUSED")
            for reason in refusal.reasons:
                print(f"  - {reason}")
            return 1

    print("SERVABLE")
    print(f"corpus       {served.corpus_version}")
    print(f"document     {served.doc_id}")
    print(f"chunks       {served.chunk_count}")
    print(f"set          {served.embedding_set_id} ({served.vector_count} vectors)")
    print(f"model        {describe(served.model)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
