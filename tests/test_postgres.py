import re

import psycopg
import pytest
from psycopg.conninfo import conninfo_to_dict

from taxverity.config import REPO_ROOT, Settings

PGVECTOR_VERSION = "0.8.6"
POSTGRES_MAJOR = 17
IMAGE = f"pgvector/pgvector:{PGVECTOR_VERSION}-pg{POSTGRES_MAJOR}-bookworm"


def _compose() -> str:
    return (REPO_ROOT / "compose.yaml").read_text(encoding="utf-8")


def _compose_env(name: str) -> str:
    match = re.search(rf"^\s+{name}:\s*(\S+)\s*$", _compose(), flags=re.MULTILINE)
    assert match, f"{name} not set in compose.yaml"
    return match.group(1)


def _service_block(name: str) -> str:
    # `compose.yaml` gained an `api` service at Step 15.3; a regex over the
    # whole file would see both services' `ports:` entries. Top-level service
    # keys are two-space indented, so the block runs to the next one (or to
    # `volumes:`, which sits at zero indent after the last service).
    match = re.search(
        rf"^  {name}:\n(.*?)(?=^  \S|^\S)", _compose(), flags=re.MULTILINE | re.DOTALL
    )
    assert match, f"{name} service not found in compose.yaml"
    return match.group(1)


def test_image_is_pinned_to_one_pgvector_release_and_postgres_major():
    images = re.findall(
        r"^\s+image:\s*(\S+)\s*$", _service_block("postgres"), flags=re.MULTILINE
    )
    assert images == [IMAGE]


def test_postgres_port_is_published_on_loopback_only():
    published = re.findall(
        r'^\s+-\s*"([^"]+)"\s*$', _service_block("postgres"), flags=re.MULTILINE
    )
    assert published == ["127.0.0.1:5432:5432"]


def test_volume_name_carries_the_postgres_major():
    assert f"pgdata-pg{POSTGRES_MAJOR}:/var/lib/postgresql/data" in _compose()


# RDS and Supabase never run Docker init scripts, so an extension created by one
# would exist only in dev. The first migration (Step 6.2) owns it everywhere.
def test_no_init_script_is_mounted():
    assert "docker-entrypoint-initdb.d" not in _compose()


def test_env_example_url_points_at_the_compose_database():
    text = (REPO_ROOT / ".env.example").read_text(encoding="utf-8")
    match = re.search(r"^#\s*TAXVERITY_DATABASE_URL=(\S+)$", text, flags=re.MULTILINE)
    assert match, "TAXVERITY_DATABASE_URL missing from .env.example"
    url = conninfo_to_dict(match.group(1))
    assert url == {
        "user": _compose_env("POSTGRES_USER"),
        "password": _compose_env("POSTGRES_PASSWORD"),
        "dbname": _compose_env("POSTGRES_DB"),
        # An IP, not "localhost": on Windows "localhost" can resolve to ::1
        # first, and the port is published on IPv4 loopback only.
        "host": "127.0.0.1",
        "port": "5432",
    }


# Skipped only when no database is configured. A configured database that
# cannot be reached fails: it is a broken dev environment, not an absent one.
@pytest.fixture(scope="module")
def conn():
    url = Settings().database_url
    if url is None:
        pytest.skip("TAXVERITY_DATABASE_URL not set; run `docker compose up -d`")
    with psycopg.connect(
        url.get_secret_value(), autocommit=True, connect_timeout=5
    ) as connection:
        yield connection


def test_server_runs_the_pinned_postgres_major(conn):
    version_num = int(conn.execute("SHOW server_version_num").fetchone()[0])
    assert version_num // 10000 == POSTGRES_MAJOR


def test_pinned_pgvector_release_is_available(conn):
    """`PGVECTOR_VERSION` pins `compose.yaml`'s local Docker image (Step 6.1,
    ADR-087) — it says nothing about a managed Postgres. `conn` resolves
    whatever `TAXVERITY_DATABASE_URL` is currently configured to, and since
    Step 17.1 that is routinely Supabase (ADR-111), not the local compose
    instance; Supabase ships its own pgvector release on its own schedule
    and this repo has no lever over it. The strict pin applies only when
    talking to loopback."""
    row = conn.execute(
        "SELECT default_version FROM pg_available_extensions WHERE name = 'vector'"
    ).fetchone()
    if conn.info.host not in ("127.0.0.1", "localhost"):
        assert row is not None
        return
    assert row == (PGVECTOR_VERSION,)


def test_vector_extension_computes_cosine_distance(conn):
    # Rolled back so this test leaves no schema behind; CREATE EXTENSION is
    # transactional in Postgres.
    with conn.transaction(force_rollback=True):
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        orthogonal, identical = conn.execute(
            "SELECT '[1,0,0]'::vector <=> '[0,1,0]'::vector,"
            " '[0.6,0.8]'::vector <=> '[0.6,0.8]'::vector"
        ).fetchone()
    assert orthogonal == pytest.approx(1.0)
    assert identical == pytest.approx(0.0)
