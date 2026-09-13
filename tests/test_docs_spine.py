import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

EXPECTED_ADR_COUNT = 111
EXPECTED_PHASE_COUNT = 18


# PLAN.md and DECISIONS.md are gitignored, so a fresh clone does not have them.
# These checks guard the working copy where they do exist rather than failing
# on their absence.
def _read(name: str) -> str:
    path = REPO_ROOT / name
    if not path.exists():
        pytest.skip(f"{name} not present in this working copy")
    return path.read_text(encoding="utf-8")


def test_adr_ids_are_present_and_contiguous():
    ids = re.findall(r"^## ADR-(\d{3}) — ", _read("DECISIONS.md"), flags=re.MULTILINE)
    assert [int(i) for i in ids] == list(range(1, EXPECTED_ADR_COUNT + 1))


def test_every_adr_declares_a_status():
    decisions = _read("DECISIONS.md")
    bodies = re.split(r"^## ADR-\d{3} — ", decisions, flags=re.MULTILINE)[1:]
    missing = [b.splitlines()[0] for b in bodies if "**Status:**" not in b]
    assert missing == []


def test_plan_documents_every_phase():
    headings = re.findall(
        r"^## Phase (\d{1,2}) — ", _read("PLAN.md"), flags=re.MULTILINE
    )
    assert [int(h) for h in headings] == list(range(0, EXPECTED_PHASE_COUNT + 1))


def test_safety_policy_destination_is_recorded():
    assert "SAFETY_POLICY.md" in _read("docs/README.md")
