import os
from pathlib import Path

import pytest

from taxverity.chunking.chunker import build_chunks
from taxverity.config import ENV_PREFIX, MissingSettingError, Settings
from taxverity.corpus.crossrefs import extract_crossrefs
from taxverity.corpus.loader import read_pages_jsonl
from taxverity.corpus.schedules import FIRST_SCHEDULE_PAGE, parse_schedules
from taxverity.corpus.sections import parse
from taxverity.corpus.substructure import candidate_table_pages, parse_substructure
from taxverity.corpus.tables import find_table_regions
from taxverity.evals.gold import GOLD_V2_FILENAME, load_gold_set


# Session-scoped, and autouse so it is ordered ahead of every other fixture in
# this scope: the session-scoped corpus extraction resolves its path during
# setup, which is too early for a function-scoped patch to have cleaned up.
@pytest.fixture(scope="session", autouse=True)
def isolated_environment():
    with pytest.MonkeyPatch.context() as patcher:
        for name in [key for key in os.environ if key.startswith(ENV_PREFIX)]:
            patcher.delenv(name, raising=False)
        yield


# --- shared corpus fixtures ---------------------------------------------------
# Session scope is per-file in pytest only for the *value*, not the definition:
# a fixture defined in a test module is re-evaluated for every module that
# defines its own copy. Four modules were each running their own pdfplumber
# geometry scan of the same pages; defined here, one scan serves all of them.

INTERIM = Path(__file__).resolve().parents[1] / "data" / "interim" / "pages.jsonl"


@pytest.fixture(scope="session")
def act_pages():
    if not INTERIM.exists():
        pytest.skip("run scripts/extract_corpus.py to build data/interim/pages.jsonl")
    return list(read_pages_jsonl(INTERIM))


@pytest.fixture(scope="session")
def act(act_pages):
    return parse(act_pages)


@pytest.fixture(scope="session")
def pdf_path():
    try:
        return Settings().resolve_corpus_pdf()
    except MissingSettingError:
        pytest.skip("corpus PDF not found — table geometry needs the real PDF")


@pytest.fixture(scope="session")
def section_table_regions(act, pdf_path):
    return find_table_regions(pdf_path, candidate_table_pages(act))


@pytest.fixture(scope="session")
def schedule_table_regions(act_pages, pdf_path):
    return find_table_regions(pdf_path, range(FIRST_SCHEDULE_PAGE, len(act_pages)))


@pytest.fixture(scope="session")
def sub(act, section_table_regions):
    return parse_substructure(act, table_regions=section_table_regions)


@pytest.fixture(scope="session")
def parsed_schedules(act_pages, schedule_table_regions):
    return parse_schedules(act_pages, table_regions=schedule_table_regions)


@pytest.fixture(scope="session")
def crossrefs(sub, parsed_schedules):
    return extract_crossrefs(sub.sections, parsed_schedules.schedules)


# Step 2.2's chunk set, shared by the chunker and store corpus suites so the
# whole parse pipeline runs once per session rather than once per module.
CHUNK_TEST_VERSION = "c" * 64


@pytest.fixture(scope="session")
def untrusted(sub, parsed_schedules):
    return set(sub.unreliable) | set(parsed_schedules.unreliable)


@pytest.fixture(scope="session")
def chunks(act, sub, parsed_schedules, crossrefs, untrusted):
    return build_chunks(
        CHUNK_TEST_VERSION,
        [*sub.sections, *parsed_schedules.schedules],
        chapters=act.chapters,
        crossrefs=crossrefs,
        untrusted=untrusted,
    )


# The Step 3.7 gold set, shared by its own suite and the Step 3.2 metrics suite.
@pytest.fixture(scope="session")
def gold():
    return load_gold_set(Settings().evals_dir / "datasets" / GOLD_V2_FILENAME)
