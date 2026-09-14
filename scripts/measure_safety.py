"""Step 12.4 — the Step 12.2 classifier measured against 30 hand-labelled
safety cases.

Bills Groq once per case, paced under the free tier's 8,000 tokens a minute
(Step 7.1), and stores every result under `data/safety/`. Later runs read that
file and bill nothing, which is also what makes a scoring change free.

The Step 7.3 cache is on by default, so a re-run after a crash pays only for
the cases it had not reached. A run that is entirely cache hits did not
exercise the provider — the same hazard ADR-094 records.

Writes `reports/safety_eval.md`.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from taxverity.config import MissingSettingError, Settings
from taxverity.evals.extraction import Counts
from taxverity.evals.safety import (
    Prediction,
    SafetyCase,
    SafetyScore,
    judge_safety,
    load_run,
    load_safety_cases,
    store_run,
)
from taxverity.llm.client import LLMError
from taxverity.observability import configure_logging, get_logger
from taxverity.safety.classifier import IntentClassifier, ScopeCategory

logger = get_logger(__name__)

REPORT = Path("reports/safety_eval.md")
RUN_FILENAME = "safety_run_v2.json"

# Step 7.1 measured 8,000 tokens a minute on the free tier. A classification is
# small (~250 tokens for the system prompt plus reasoning), but the pause stays
# conservative rather than relying on the client's 429 retry to do the pacing.
DEFAULT_PAUSE = 4.0


def measure(
    cases: list[SafetyCase], classifier: IntentClassifier, pause: float
) -> dict[str, Prediction]:
    run: dict[str, Prediction] = {}
    for index, case in enumerate(cases, start=1):
        started = time.perf_counter()
        result = classifier.classify(case.question)
        elapsed = time.perf_counter() - started
        run[case.case_id] = Prediction(result.category, result.tokens)
        logger.info(
            "%s (%d/%d) -> %s in %.1fs, %d tokens",
            case.case_id,
            index,
            len(cases),
            result.category.value,
            elapsed,
            result.tokens,
        )
        if index < len(cases):
            time.sleep(pause)
    return run


def table(rows: list[tuple[str, Counts]]) -> list[str]:
    lines = [
        "| category | labelled | correct | false positives | false negatives | precision | recall |",
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


def render(score: SafetyScore, cases: list[SafetyCase], elapsed: float) -> str:
    lines = [
        "# Safety eval — Step 12.4",
        "",
        f"{len(cases)} labelled cases, {score.tokens:,} tokens, {elapsed:.0f}s.",
        "",
        "## Headline",
        "",
        "Refusal precision/recall treat `prohibited` as the positive class — "
        "rule 03 requires both directions: low recall means an evasion "
        "question got through, low precision means a lawful planning "
        "question was refused. Both are weighed equally.",
        "",
        f"- Four-category accuracy **{score.accuracy:.3f}** "
        f"({len(cases) - len(score.mistakes)}/{len(cases)})",
        f"- Refusal precision **{score.refusal.precision:.3f}**, "
        f"recall **{score.refusal.recall:.3f}**",
        "",
        "## By category",
        "",
    ]
    lines += table(
        [
            (category.value, score.per_category[category])
            for category in ScopeCategory
        ]
    )
    lines += ["", "## Misclassifications", ""]
    if not score.mistakes:
        lines.append("None.")
    for case in score.mistakes:
        lines.append(
            f"- **{case.case_id}** expected `{case.expected.value}`, got "
            f"`{case.predicted.value}` - `{case.question}`"
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
    args = parser.parse_args()
    configure_logging()

    settings = Settings()
    cases = list(load_safety_cases(settings.evals_dir / "datasets"))
    stored = settings.data_dir / "safety" / RUN_FILENAME

    started = time.perf_counter()
    if args.stored:
        run = load_run(stored)
        logger.info("judging the stored run at %s", stored)
    else:
        try:
            classifier = IntentClassifier.from_settings(settings)
        except MissingSettingError as error:
            print(f"cannot run: {error}")
            return 1
        try:
            run = measure(cases, classifier, args.pause)
        except LLMError as error:
            print(f"the provider failed mid-run: {error}")
            return 1
        store_run(stored, run)
    elapsed = time.perf_counter() - started

    score = judge_safety(cases, run)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(render(score, cases, elapsed), encoding="utf-8", newline="")

    print(f"accuracy {score.accuracy:.3f}")
    print(
        f"refusal precision {score.refusal.precision:.3f}, "
        f"recall {score.refusal.recall:.3f}"
    )
    print(f"tokens {score.tokens:,}")
    if not args.stored:
        print(f"wrote {stored}")
    print(f"wrote {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
