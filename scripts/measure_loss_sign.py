"""The loss-sign fix measured by the rule in `taxverity.evals.loss_sign`,
registered before the fix was written (ADR-099).

Two stages, run on either side of the change:

- `--stage before`: the held-out loss turns through the node as it stands, so
  the fix's effect is measured rather than assumed. The 7.7 gold set's "before"
  is Step 7.7's own stored run.
- `--stage after`: the held-out turns again, and the 42-turn 7.7 gold set, both
  through the fixed node. Then the rule is judged.
- `--stage recheck`: the held-out turns through the node as it stands now. Added
  when ADR-100 renamed `assessment_year` to `tax_year`, which left the stored
  runs above unloadable; `reports/loss_sign_fix.md` is their frozen record, and
  the 7.7 gold half of the recheck is `scripts/measure_extraction.py`'s run.

Bills Groq once per turn not already stored or cached, paced as Step 7.7 was.
`--stored` judges what is on disk without calling the model.

Writes runs under `data/extraction/` and `reports/loss_sign_fix.md`.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from measure_extraction import DEFAULT_PAUSE, measure

from taxverity.config import MissingSettingError, Settings
from taxverity.evals.extraction import (
    LOSS_HOLDOUT_FILENAME,
    ExtractionScore,
    LabelledTurn,
    judge_extraction,
    load_extraction_gold,
    load_run,
    store_run,
)
from taxverity.evals.loss_sign import judge_loss_fix, unclean
from taxverity.llm.client import LLMError
from taxverity.llm.extract import EXTRACTION_STAGE_VERSION, FactExtractor
from taxverity.observability import configure_logging, get_logger

logger = get_logger(__name__)

REPORT = Path("reports/loss_sign_fix.md")
HOLDOUT_BEFORE = "loss_holdout_run_before.json"
HOLDOUT_AFTER = "loss_holdout_run_after.json"
GOLD_BEFORE = "extraction_run_v1.json"
GOLD_AFTER = "extraction_run_v2.json"
HOLDOUT_RECHECK = "loss_holdout_run_v3.json"


def rows(score: ExtractionScore, labels: dict[str, LabelledTurn]) -> list[str]:
    lines = ["| turn | clean | wrong values | missed | invented | repaired | turn text |", "|---|---|---|---|---|---|---|"]
    for t in score.turns:
        wrong = "; ".join(f"{f.value}: expected {e}, got {a}" for f, e, a in t.value_wrong) or "—"
        lines.append(
            f"| {t.turn_id} | {'no' if unclean(t) else 'yes'} | {wrong} "
            f"| {', '.join(f.value for f in t.missed) or '—'} "
            f"| {', '.join(f.value for f in t.spurious) or '—'} "
            f"| {'yes' if t.repaired else ''} | {labels[t.turn_id].turn} |"
        )
    return lines


def headline(name: str, score: ExtractionScore, count: int) -> str:
    return (
        f"| {name} | {score.value_accuracy:.3f} | {score.strict_rate:.3f} "
        f"| {score.clean_turns}/{count} | {score.repaired_turns} | {score.tokens:,} |"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("before", "after", "recheck"), required=True)
    parser.add_argument("--stored", action="store_true", help="judge stored runs only")
    parser.add_argument("--pause", type=float, default=DEFAULT_PAUSE)
    args = parser.parse_args()
    configure_logging()

    settings = Settings()
    datasets = settings.evals_dir / "datasets"
    runs = settings.data_dir / "extraction"
    holdout = list(load_extraction_gold(datasets, LOSS_HOLDOUT_FILENAME))
    gold = list(load_extraction_gold(datasets))

    def run_or_load(turns: list[LabelledTurn], filename: str) -> dict:
        path = runs / filename
        if args.stored:
            return load_run(path)
        run = measure(turns, extractor, args.pause)
        store_run(path, run)
        logger.info("stored %d turns at %s", len(run), path)
        return run

    extractor = None
    if not args.stored:
        try:
            extractor = FactExtractor.from_settings(settings)
        except MissingSettingError as error:
            print(f"cannot run: {error}", file=sys.stderr)
            return 1

    started = time.perf_counter()
    try:
        if args.stage == "recheck":
            recheck = judge_extraction(holdout, run_or_load(holdout, HOLDOUT_RECHECK))
            print(f"held-out recheck: {recheck.clean_turns}/{len(holdout)} clean, {recheck.tokens:,} tokens")
            for t in recheck.turns:
                if unclean(t):
                    print(f"  {t.turn_id}: wrong {t.value_wrong}, missed {t.missed}, invented {t.spurious}")
            return 0
        if args.stage == "before":
            before = judge_extraction(holdout, run_or_load(holdout, HOLDOUT_BEFORE))
            elapsed = time.perf_counter() - started
            print(f"held-out before the fix: {before.clean_turns}/{len(holdout)} clean, {before.tokens:,} tokens, {elapsed:.0f}s")
            for t in before.turns:
                if unclean(t):
                    print(f"  {t.turn_id}: wrong {t.value_wrong}, missed {t.missed}, invented {t.spurious}")
            return 0
        holdout_after = judge_extraction(holdout, run_or_load(holdout, HOLDOUT_AFTER))
        gold_after = judge_extraction(gold, run_or_load(gold, GOLD_AFTER))
    except LLMError as error:
        print(f"the provider failed mid-run: {error}", file=sys.stderr)
        return 1
    elapsed = time.perf_counter() - started

    holdout_before = judge_extraction(holdout, load_run(runs / HOLDOUT_BEFORE))
    gold_before = judge_extraction(gold, load_run(runs / GOLD_BEFORE))
    verdict = judge_loss_fix(holdout_after, gold_before, gold_after)

    labels = {t.turn_id: t for t in holdout}
    gold_labels = {t.turn_id: t for t in gold}
    lines = [
        "# Loss-sign fix — extraction",
        "",
        "Auto-generated by `scripts/measure_loss_sign.py` (ADR-099).",
        f"Extraction stage version after the fix: {EXTRACTION_STAGE_VERSION}. Run time {elapsed:.0f}s.",
        "",
        "## The rule (registered before the fix was written)",
        "",
        "1. Every held-out turn is clean: 8 loss turns and 4 controls where",
        "   loss vocabulary sits beside a positive amount.",
        "2. t013 and t017 of the Step 7.7 gold set carry the right value.",
        "3. No 7.7 turn that was clean before the fix is unclean after it.",
        "",
        f"**Verdict: {'ADOPTED' if verdict.adopted else 'REJECTED'}.**",
        "",
        *([f"- {r}" for r in verdict.reasons] or ["- every clause held"]),
        "",
        "## Headline",
        "",
        "| run | value accuracy | strict | clean turns | repaired | tokens |",
        "|---|---|---|---|---|---|",
        headline("held-out, before", holdout_before, len(holdout)),
        headline("held-out, after", holdout_after, len(holdout)),
        headline("7.7 gold, before (Step 7.7's run)", gold_before, len(gold)),
        headline("7.7 gold, after", gold_after, len(gold)),
        "",
        "## Held-out turns, before",
        "",
        *rows(holdout_before, labels),
        "",
        "## Held-out turns, after",
        "",
        *rows(holdout_after, labels),
        "",
        "## 7.7 gold turns that are not clean, after",
        "",
        *rows(
            ExtractionScore(
                turns=tuple(t for t in gold_after.turns if unclean(t)),
                counts=gold_after.counts,
                per_slice={},
                per_field={},
            ),
            gold_labels,
        ),
        "",
    ]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="")
    print(f"held-out after: {holdout_after.clean_turns}/{len(holdout)} clean")
    print(f"7.7 gold after: value {gold_after.value_accuracy:.3f}, strict {gold_after.strict_rate:.3f}, clean {gold_after.clean_turns}/{len(gold)}")
    print(f"fix: {'ADOPTED' if verdict.adopted else 'REJECTED'} {list(verdict.reasons)}")
    print(f"wrote {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
