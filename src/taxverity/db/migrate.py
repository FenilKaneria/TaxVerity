"""Step 6.2 — plain-SQL schema migrations (ADR-088).

A migration is a file `NNNN_name.sql` in `migrations/`. Each is applied once,
in order, inside its own transaction, and recorded with its sha256. An applied
migration whose file has since changed is refused rather than skipped: the
database would otherwise claim a schema the code no longer describes.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import psycopg

from taxverity.observability import get_logger

logger = get_logger(__name__)

MIGRATIONS_DIR = Path(__file__).parent / "migrations"
MIGRATION_FILE = re.compile(r"^(\d{4})_([a-z0-9_]+)\.sql$")

SCHEMA_MIGRATIONS_DDL = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    version    integer PRIMARY KEY,
    name       text NOT NULL,
    sha256     text NOT NULL,
    applied_at timestamptz NOT NULL DEFAULT now()
)
"""


class MigrationError(RuntimeError):
    pass


@dataclass(frozen=True)
class Migration:
    version: int
    name: str
    sql: str

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.sql.encode("utf-8")).hexdigest()


def discover_migrations(directory: Path = MIGRATIONS_DIR) -> tuple[Migration, ...]:
    migrations: list[Migration] = []
    for path in sorted(directory.glob("*.sql")):
        match = MIGRATION_FILE.match(path.name)
        if match is None:
            raise MigrationError(f"{path.name} is not named NNNN_name.sql")
        # read_text translates CRLF, so a checkout with autocrlf hashes the same
        # as the LF file the migration was applied from.
        migrations.append(
            Migration(int(match.group(1)), match.group(2), path.read_text("utf-8"))
        )
    versions = [m.version for m in migrations]
    if versions != list(range(1, len(migrations) + 1)):
        raise MigrationError(
            f"migrations in {directory} are numbered {versions}; "
            f"they must run 1..n with no gap or duplicate"
        )
    return tuple(migrations)


def migrate(
    conn: psycopg.Connection, directory: Path = MIGRATIONS_DIR
) -> tuple[int, ...]:
    """Apply every pending migration; return the versions applied."""
    if not conn.autocommit:
        raise ValueError(
            "migrate() needs an autocommit connection: it commits each "
            "migration in its own transaction"
        )
    migrations = discover_migrations(directory)
    conn.execute(SCHEMA_MIGRATIONS_DDL)
    applied = {
        version: sha256
        for version, sha256 in conn.execute(
            "SELECT version, sha256 FROM schema_migrations"
        ).fetchall()
    }

    known = {m.version: m for m in migrations}
    for version, sha256 in sorted(applied.items()):
        if version not in known:
            raise MigrationError(
                f"database has migration {version:04d} applied, but this code "
                f"only knows up to {len(migrations):04d}"
            )
        if known[version].sha256 != sha256:
            raise MigrationError(
                f"migration {version:04d}_{known[version].name} changed after it "
                f"was applied; write a new migration instead of editing it"
            )

    pending = [m for m in migrations if m.version not in applied]
    if applied and pending and pending[0].version < max(applied):
        raise MigrationError(
            f"migration {pending[0].version:04d} is unapplied but "
            f"{max(applied):04d} is applied; migrations must run in order"
        )

    for migration in pending:
        with conn.transaction():
            conn.execute(migration.sql)
            conn.execute(
                "INSERT INTO schema_migrations (version, name, sha256) "
                "VALUES (%s, %s, %s)",
                (migration.version, migration.name, migration.sha256),
            )
        logger.info("applied migration %04d_%s", migration.version, migration.name)

    logger.info(
        "schema at migration %04d (%d applied now)", len(migrations), len(pending)
    )
    return tuple(m.version for m in pending)
