"""R18 gate 3 (part) — the loss-sign held-out turns re-measured against Groq's 20b.

`measure_loss_sign.py`'s `--stage` machinery is pinned to the 120b model's
own before/after fix history (`SIGN_FAILURES_BEFORE`, `CONTEXT_FAILURES_BEFORE`
in `evals/loss_sign.py`) — reusing its `judge_loss_fix`/`judge_context_guard`
for a 120b-vs-20b comparison would conflate "does the fix generalize" with
"does 20b reproduce 120b's exact known residue", which is not the question
here. This script asks the plain question instead: **does the 20b model ever
get a loss sign wrong** on the two held-out sets (12 + 12 turns) — the same
bar `test_the_fix_still_holds_on_the_held_out_turns` holds the 120b node to.

The 42-turn gold set is not re-run here: `measure_extraction.py --model 20b`
already covers it.

Bills Groq's 20b bucket, paced like `measure_extraction.py`. Cache off, same
reason `--stage context-after` turns it off: the system prompt did not
change, so a cached pass would replay old samples instead of drawing new
ones from the 20b model.

Writes `data/extraction/loss_holdout_run_v1_20b.json`,
`data/extraction/loss_holdout_run_v2_20b.json`, and
`reports/loss_sign_20b.md`.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from measure_extraction import DEFAULT_PAUSE, measure  # noqa: E402

from taxverity.config import MissingSettingError, Settings  # noqa: E402
from taxverity.evals.extraction import (  # noqa: E402
    LOSS_HOLDOUT_FILENAME,
    LOSS_HOLDOUT_V2_FILENAME,
    ExtractionScore,
    LabelledTurn,
    judge_extraction,
    load_extraction_gold,
    store_run,
)
from taxverity.evals.loss_sign import unclean  # noqa: E402
from taxverity.llm.client import GEMINI, GROQ_20B, LLMError  # noqa: E402
from taxverity.llm.extract import FactExtractor  # noqa: E402
from taxverity.observability import configure_logging, get_logger  # noqa: E402

logger = get_logger(__name__)

REPORT = Path("reports/loss_sign_20b.md")


def run_set(
    turns: list[LabelledTurn], extractor: FactExtractor, pause: float, path: Path
) -> ExtractionScore:
    run = measure(turns, extractor, pause)
    store_run(path, run)
    logger.info("stored %d turns at %s", len(run), path)
    return judge_extraction(turns, run)


def render(v1: ExtractionScore, v2: ExtractionScore, elapsed: float) -> str:
    v1_bad = [t for t in v1.turns if unclean(t)]
    v2_bad = [t for t in v2.turns if unclean(t)]
    any_wrong_sign = any(t.value_wrong for t in v1.turns) or any(t.value_wrong for t in v2.turns)
    lines = [
        "# R18 gate 3 (part) — loss-sign held-out turns against Groq 20b",
        "",
        f"24 held-out turns (12 + 12), {v1.tokens + v2.tokens:,} tokens, {elapsed:.0f}s.",
        "",
        "## Verdict",
        "",
    ]
    if any_wrong_sign:
        lines.append(
            "**FAIL — at least one held-out turn's value is wrong on 20b.** "
            "This node should not move off Groq 120b."
        )
    else:
        lines.append(
            f"**PASS — no wrong value on either held-out set.** "
            f"v1 clean {v1.clean_turns}/{len(v1.turns)}, "
            f"v2 clean {v2.clean_turns}/{len(v2.turns)}."
        )
    lines += ["", "## v1 (12 turns)", ""]
    if not v1_bad:
        lines.append("All clean.")
    for t in v1_bad:
        lines.append(f"- **{t.turn_id}**: wrong {t.value_wrong}, missed {t.missed}, invented {t.spurious}")
    lines += ["", "## v2 (12 turns)", ""]
    if not v2_bad:
        lines.append("All clean.")
    for t in v2_bad:
        lines.append(f"- **{t.turn_id}**: wrong {t.value_wrong}, missed {t.missed}, invented {t.spurious}")
    return "\n".join(lines) + "\n"


def main() -> int:
    configure_logging()
    settings = Settings()
    datasets = settings.evals_dir / "datasets"
    runs = settings.data_dir / "extraction"
    holdout = list(load_extraction_gold(datasets, LOSS_HOLDOUT_FILENAME))
    holdout_v2 = list(load_extraction_gold(datasets, LOSS_HOLDOUT_V2_FILENAME))

    try:
        extractor = FactExtractor.from_settings(
            settings, cache=False, primary=GROQ_20B, fallback=GEMINI
        )
    except MissingSettingError as error:
        print(f"cannot run: {error}")
        return 1

    started = time.perf_counter()
    try:
        v1 = run_set(holdout, extractor, DEFAULT_PAUSE, runs / "loss_holdout_run_v1_20b.json")
        v2 = run_set(holdout_v2, extractor, DEFAULT_PAUSE, runs / "loss_holdout_run_v2_20b.json")
    except LLMError as error:
        print(f"the provider failed mid-run: {error}")
        return 1
    elapsed = time.perf_counter() - started

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(render(v1, v2, elapsed), encoding="utf-8", newline="")
    print(f"v1 clean {v1.clean_turns}/{len(v1.turns)}, v2 clean {v2.clean_turns}/{len(v2.turns)}")
    print(f"wrote {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
