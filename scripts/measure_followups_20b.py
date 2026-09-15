"""R18 gate 3 (part) — the 5 follow-up cases re-measured with QueryContextualizer
on Groq's 20b model instead of the production default.

Reuses `answer_smoke.py`'s `run_followups()` and `build_followup_retrieval()`
directly (same live Jina embedding/rerank calls it always makes — a
follow-up's rewritten query only exists once the LLM has produced it, so
there is nothing to pre-embed). Only the LLM client passed to
`QueryContextualizer` differs: Groq 20b primary, Gemini fallback, cache off.

Writes `data/answers/followup_smoke_20b.json` and
`reports/followup_smoke_20b.md`.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from answer_smoke import render_followups, run_followups  # noqa: E402

from taxverity.chunking.pipeline import read_corpus_version  # noqa: E402
from taxverity.chunking.store import load_chunks  # noqa: E402
from taxverity.config import MissingSettingError, Settings  # noqa: E402
from taxverity.evals.followups import (  # noqa: E402
    distinct_models,
    store_followup_run,
)
from taxverity.evals.gold import GOLD_V2_FILENAME, load_gold_set  # noqa: E402
from taxverity.llm.client import GEMINI, GROQ_20B, LLMClient, LLMError  # noqa: E402
from taxverity.observability import configure_logging, get_logger  # noqa: E402

logger = get_logger(__name__)

REPORT = Path("reports/followup_smoke_20b.md")
RUN = Path("data/answers/followup_smoke_20b.json")


def main() -> int:
    configure_logging()
    settings = Settings()
    try:
        llm = LLMClient.from_settings(settings, primary=GROQ_20B, fallback=GEMINI)
    except MissingSettingError as error:
        print(f"cannot run: {error}")
        return 1

    gold = {
        q.query_id: q
        for q in load_gold_set(settings.evals_dir / "datasets" / GOLD_V2_FILENAME)
    }
    corpus_version = read_corpus_version(settings.interim_dir / "corpus_manifest.json")
    chunks, _ = load_chunks(settings.interim_dir, corpus_version=corpus_version)

    try:
        run = run_followups(settings, gold, llm, chunks)
    except LLMError as error:
        print(f"the provider failed mid-run: {error}")
        return 1

    models = distinct_models(run)
    RUN.parent.mkdir(parents=True, exist_ok=True)
    store_followup_run(RUN, run)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(
        f"Measured against {sorted(models)}.\n\n" + render_followups(run),
        encoding="utf-8",
        newline="",
    )
    hits = sum(1 for record in run.records if record.hit)
    print(f"follow-up cases: {hits}/{len(run.records)} hit expected citation")
    print(f"models used: {sorted(models)}")
    print(f"wrote {RUN}")
    print(f"wrote {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
