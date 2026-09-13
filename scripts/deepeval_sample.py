"""Step 10.10 — a DeepEval sample over the Step 10.7 smoke-set answers (ADR-053).

Report only, no gate. The deterministic verifier is the grounding control;
this is an offline second opinion on answer quality from an LLM judge, which is
exactly why it never runs on a request path and is never imported from `src/`.

The judge is our own `LLMClient` behind the Step 7.3 cache, so redaction holds
on its calls too and a re-run bills nothing. DeepEval's telemetry is opted out
before it is imported.

Each answer is judged against only the evidence units its served claims cite,
not the whole pack: faithfulness asks whether the answer follows from its
sources, and the full pack would cost about 4,000 tokens a judgement for text
the answer never used.

Writes `reports/deepeval_sample.md`.
"""

from __future__ import annotations

import os

os.environ["DEEPEVAL_TELEMETRY_OPT_OUT"] = "YES"

import argparse  # noqa: E402
import asyncio  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

from deepeval.metrics import FaithfulnessMetric, GEval  # noqa: E402
from deepeval.models import DeepEvalBaseLLM  # noqa: E402
from deepeval.test_case import LLMTestCase, SingleTurnParams  # noqa: E402

from taxverity.config import MissingSettingError, Settings  # noqa: E402
from taxverity.evals.answers import (  # noqa: E402
    ANSWER_RUN_FILENAME,
    AnswerRecord,
    load_answer_run,
)
from taxverity.llm.cache import CachedLLMClient  # noqa: E402
from taxverity.llm.client import LLMClient, Message  # noqa: E402
from taxverity.observability import configure_logging, get_logger  # noqa: E402

logger = get_logger(__name__)

REPORT = Path("reports") / "deepeval_sample.md"
DEFAULT_LIMIT = 5
# A judgement is a few thousand tokens against 8,000 a minute (Step 7.1).
DEFAULT_PAUSE = 30.0
JUDGE_MAX_COMPLETION_TOKENS = 4_096

RUBRIC = (
    "Judge the answer to a question about the Income-tax Act, 2025 (India). "
    "A good answer addresses what was asked directly, states only what the "
    "retrieval context supports, judged against that context and never against "
    "memory of any earlier Act, does not overstate certainty where the Act is conditional, "
    "and is not padded with provisions the question did not ask about. An answer "
    "that says too little to be useful scores low even if every sentence is true."
)


class Judge(DeepEvalBaseLLM):
    """DeepEval's model interface over our own client."""

    def __init__(self, llm, name: str) -> None:
        self._llm = llm
        self._name = name
        super().__init__(model=name)

    def load_model(self):
        return self._llm

    def generate(self, prompt: str, schema=None):
        completion = self._llm.complete(
            [Message(role="user", content=prompt)],
            max_completion_tokens=JUDGE_MAX_COMPLETION_TOKENS,
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        if schema is None:
            return completion.text
        return schema.model_validate_json(completion.text)

    async def a_generate(self, prompt: str, schema=None):
        return await asyncio.to_thread(self.generate, prompt, schema)

    def get_model_name(self) -> str:
        return self._name


def cited_context(record: AnswerRecord) -> list[str]:
    paths = {c.path for claim in record.claims for c in claim.citations}

    def related(unit: str, path: str) -> bool:
        return (
            unit == path or path.startswith(unit + "(") or unit.startswith(path + "(")
        )

    return [
        f"{unit.citation}:\n{unit.text}"
        for unit in record.evidence
        if any(related(unit.citation, path) for path in paths)
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument("--pause", type=float, default=DEFAULT_PAUSE)
    args = parser.parse_args()
    configure_logging()
    settings = Settings()

    try:
        run = load_answer_run(settings.data_dir / "answers" / ANSWER_RUN_FILENAME)
        base = LLMClient.from_settings(settings)
    except (FileNotFoundError, MissingSettingError) as error:
        print(f"error: {error} (run scripts/answer_smoke.py first)", file=sys.stderr)
        return 1
    judge = Judge(CachedLLMClient(base, settings.llm_cache_dir), base.primary.model)

    # Only answers with a served claim can be judged; a negative's honest
    # silence has no output to score.
    sample = [r for r in run.records if r.claims][: args.limit]
    rows = []
    for index, record in enumerate(sample, start=1):
        case = LLMTestCase(
            input=record.question,
            actual_output=record.answer_text(),
            retrieval_context=cited_context(record),
        )
        faithfulness = FaithfulnessMetric(model=judge, async_mode=False)
        quality = GEval(
            name="Answer quality",
            criteria=RUBRIC,
            # Without the evidence the judge grades against its own memory of
            # the 1961 Act, and calls verbatim 2025 text unsupported.
            evaluation_params=[
                SingleTurnParams.INPUT,
                SingleTurnParams.ACTUAL_OUTPUT,
                SingleTurnParams.RETRIEVAL_CONTEXT,
            ],
            model=judge,
            async_mode=False,
        )
        before = sum(base.tokens_used.values())
        for metric in (faithfulness, quality):
            try:
                metric.measure(case)
            except Exception as error:  # a judge failure is reported, never fatal
                logger.warning(
                    "%s: %s failed: %s", record.query_id, type(metric).__name__, error
                )
        spent = sum(base.tokens_used.values()) - before
        rows.append((record, faithfulness, quality))
        logger.info(
            "%s (%d/%d) judged, %d tokens", record.query_id, index, len(sample), spent
        )
        if spent and index < len(sample):
            time.sleep(args.pause)

    lines = [
        "# DeepEval sample — Step 10.10",
        "",
        "Auto-generated by `scripts/deepeval_sample.py`. Report only, no gate",
        "(ADR-053). Judge: our `LLMClient` "
        f"(`{base.primary.model}`), the same model that wrote the answers, so",
        "these scores are a self-consistency signal, not an independent one.",
        "Both metrics see only the evidence units the answer cites.",
        "",
        f"{len(sample)} answers judged, {sum(base.tokens_used.values()):,} judge tokens this run.",
        "",
        "| query | claims | faithfulness | answer quality | question |",
        "|---|---|---|---|---|",
    ]
    for record, faithfulness, quality in rows:
        lines.append(
            f"| {record.query_id} | {len(record.claims)} | {_score(faithfulness)} "
            f"| {_score(quality)} | {record.question} |"
        )
    lines += ["", "## Judge reasons", ""]
    for record, faithfulness, quality in rows:
        lines += [
            f"- **{record.query_id}** faithfulness: {faithfulness.reason or '—'}",
            f"  answer quality: {quality.reason or '—'}",
        ]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="")
    for record, faithfulness, quality in rows:
        print(
            f"{record.query_id}: faithfulness {_score(faithfulness)}, quality {_score(quality)}"
        )
    print(f"wrote {REPORT}")
    return 0


def _score(metric) -> str:
    return "error" if metric.score is None else f"{metric.score:.2f}"


if __name__ == "__main__":
    raise SystemExit(main())
