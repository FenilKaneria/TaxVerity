"""Step 7.7 — the extraction node measured against 42 hand-labelled turns.

The first run bills Groq once per turn, paced under the free tier's 8,000
tokens a minute (Step 7.1), and stores every result under
`data/extraction/`. Later runs read that file and bill nothing, which is also
what makes a scoring change free.

The Step 7.3 cache is on by default, so a re-run after a crash pays only for
the turns it had not reached. Its hazard applies here as it does everywhere: a
run that is entirely cache hits did not exercise the provider, and a genuine
re-measurement means bumping `LLM_CACHE_VERSION` or emptying `data/llm/`.

Writes `reports/extraction_eval.md`.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from taxverity.config import MissingSettingError, Settings
from taxverity.evals.extraction import (
    Counts,
    ExtractionScore,
    LabelledTurn,
    TurnSlice,
    judge_extraction,
    load_extraction_gold,
    load_run,
    store_run,
)
from taxverity.facts import FactField
from taxverity.llm.client import LLMError
from taxverity.llm.extract import FactExtractor
from taxverity.observability import configure_logging, get_logger

logger = get_logger(__name__)

REPORT = Path("reports/extraction_eval.md")
RUN_FILENAME = "extraction_run_v1.json"

# Step 7.1 measured 8,000 tokens a minute on the free tier. One extraction is
# roughly a thousand, so a short pause keeps a 42-turn run inside the limit
# without relying on the client's 429 retry to do the pacing.
DEFAULT_PAUSE = 8.0


def measure(
    turns: list[LabelledTurn], extractor: FactExtractor, pause: float
) -> dict:
    run = {}
    for index, labelled in enumerate(turns, start=1):
        started = time.perf_counter()
        run[labelled.turn_id] = extractor.extract(labelled.turn)
        elapsed = time.perf_counter() - started
        logger.info(
            "%s (%d/%d) in %.1fs, %d tokens",
            labelled.turn_id,
            index,
            len(turns),
            elapsed,
            run[labelled.turn_id].tokens,
        )
        if index < len(turns):
            time.sleep(pause)
    return run


def table(header: str, rows: list[tuple[str, Counts]]) -> list[str]:
    lines = [
        f"| {header} | labelled | found | spurious | missed | precision | recall |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, counts in rows:
        labelled = counts.true_positives + counts.false_negatives
        lines.append(
            f"| {name} | {labelled} | {counts.true_positives} | "
            f"{counts.false_positives} | {counts.false_negatives} | "
            f"{counts.precision:.3f} | {counts.recall:.3f} |"
        )
    return lines


def render(score: ExtractionScore, turns: list[LabelledTurn], elapsed: float) -> str:
    labelled = score.counts.true_positives + score.counts.false_negatives
    lines = [
        "# Extraction eval — Step 7.7",
        "",
        f"{len(turns)} labelled turns, {labelled} labelled facts, "
        f"{score.tokens:,} tokens, {elapsed:.0f}s.",
        "",
        "## Headline",
        "",
        "Detection is whether the node found the field at all; value and status "
        "accuracy are measured only over the fields it did find, because a "
        "prompt change fixes the first and `normalise_value` fixes the second.",
        "",
        f"- Field precision **{score.counts.precision:.3f}**, "
        f"recall **{score.counts.recall:.3f}**, f1 **{score.counts.f1:.3f}**",
        f"- Value accuracy **{score.value_accuracy:.3f}**",
        f"- Status accuracy **{score.status_accuracy:.3f}**",
        f"- Strict (field, value and status all right) **{score.strict_rate:.3f}**",
        f"- Turns with nothing wrong: **{score.clean_turns}/{len(turns)}**",
        "",
        "## Source spans",
        "",
        "`parse_facts()` refuses a span the turn does not contain, so no "
        "surviving fact can carry a fabricated one. The rate below is therefore "
        "of *attempts*, which is the only honest denominator.",
        "",
        f"- Fabricated-span rate **{score.fabricated_span_rate:.3f}** "
        f"({sum(t.fabricated_spans for t in score.turns)} of "
        f"{sum(t.stated_attempts for t in score.turns)} stated attempts)",
        f"- Stated facts quoting nothing: "
        f"{sum(t.unquoted for t in score.turns)}",
        f"- Turns that needed a repair: **{score.repaired_turns}**, "
        f"of which the repair helped **{score.repairs_that_helped}**",
        "",
        "## By slice",
        "",
    ]
    lines += table(
        "slice",
        [
            (slice_.value, score.per_slice[slice_])
            for slice_ in TurnSlice
            if slice_ in score.per_slice
        ],
    )
    lines += ["", "## By field", ""]
    lines += table(
        "field",
        [
            (field.value, score.per_field[field])
            for field in FactField
            if field in score.per_field
        ],
    )

    lines += ["", "## Every turn that was not clean", ""]
    labels = {turn.turn_id: turn for turn in turns}
    imperfect = [
        judgement
        for judgement in score.turns
        if judgement.missed or judgement.spurious or judgement.value_wrong
    ]
    if not imperfect:
        lines.append("None.")
    for judgement in imperfect:
        lines.append(
            f"- **{judgement.turn_id}** ({judgement.slice.value}) "
            f"`{labels[judgement.turn_id].turn}`"
        )
        if judgement.missed:
            lines.append(
                f"  - missed: {', '.join(f.value for f in judgement.missed)}"
            )
        if judgement.spurious:
            lines.append(
                f"  - invented: {', '.join(f.value for f in judgement.spurious)}"
            )
        for field, expected, actual in judgement.value_wrong:
            lines.append(f"  - {field.value}: expected {expected}, got {actual}")
        for field, expected, actual in judgement.status_wrong:
            lines.append(
                f"  - {field.value}: expected status {expected}, got {actual}"
            )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pause", type=float, default=DEFAULT_PAUSE)
    parser.add_argument(
        "--stored",
        action="store_true",
        help="judge the stored run without calling the model",
    )
    parser.add_argument(
        "--no-repair", action="store_true", help="measure the node without its repair"
    )
    args = parser.parse_args()
    configure_logging()

    settings = Settings()
    turns = list(load_extraction_gold(settings.evals_dir / "datasets"))
    stored = settings.data_dir / "extraction" / RUN_FILENAME

    started = time.perf_counter()
    if args.stored:
        run = load_run(stored)
        logger.info("judging the stored run at %s", stored)
    else:
        try:
            extractor = FactExtractor.from_settings(
                settings, repair=not args.no_repair
            )
        except MissingSettingError as error:
            print(f"cannot run: {error}")
            return 1
        try:
            run = measure(turns, extractor, args.pause)
        except LLMError as error:
            print(f"the provider failed mid-run: {error}")
            return 1
        store_run(stored, run)
    elapsed = time.perf_counter() - started

    score = judge_extraction(turns, run)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(render(score, turns, elapsed), encoding="utf-8", newline="")

    print(
        f"field precision {score.counts.precision:.3f}, "
        f"recall {score.counts.recall:.3f}"
    )
    print(f"value accuracy {score.value_accuracy:.3f}")
    print(f"status accuracy {score.status_accuracy:.3f}")
    print(f"strict {score.strict_rate:.3f}")
    print(f"fabricated-span rate {score.fabricated_span_rate:.3f}")
    print(f"clean turns {score.clean_turns}/{len(turns)}")
    print(f"tokens {score.tokens:,}")
    if not args.stored:
        print(f"wrote {stored}")
    print(f"wrote {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
