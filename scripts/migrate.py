"""Step 6.2 — apply pending schema migrations to TAXVERITY_DATABASE_URL."""

from __future__ import annotations

import sys

import psycopg

from taxverity.config import Settings
from taxverity.db.migrate import discover_migrations, migrate
from taxverity.observability import configure_logging


def main() -> int:
    configure_logging()
    url = Settings().require("database_url")
    with psycopg.connect(url, autocommit=True) as conn:
        applied = migrate(conn)
    head = discover_migrations()[-1]
    print(f"applied  {', '.join(f'{v:04d}' for v in applied) or 'none'}")
    print(f"head     {head.version:04d}_{head.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
