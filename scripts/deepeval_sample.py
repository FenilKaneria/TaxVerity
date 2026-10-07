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

`--source v1` instead scores every frozen generation eval v1 answer with a
grounded served claim (not a sample) on Faithfulness, Answer Relevancy and
Contextual Relevancy, judged by gpt-5-mini, and compares faithfulness with
our own calibrated claim judge. Writes `reports/deepeval_generation_v1.md`
and `data/answers/deepeval_v1.json` (resumable).
"""

from __future__ import annotations

import os

os.environ["DEEPEVAL_TELEMETRY_OPT_OUT"] = "YES"

import argparse  # noqa: E402
import asyncio  # noqa: E402
import json  # noqa: E402
import statistics  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

from deepeval.metrics import (  # noqa: E402
    AnswerRelevancyMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric,
    GEval,
)
from deepeval.models import DeepEvalBaseLLM  # noqa: E402
from deepeval.test_case import LLMTestCase, SingleTurnParams  # noqa: E402

from taxverity.config import MissingSettingError, Settings  # noqa: E402
from taxverity.evals.answers import (  # noqa: E402
    ANSWER_RUN_FILENAME,
    AnswerRecord,
    load_answer_run,
)
from taxverity.evals.generation import (  # noqa: E402
    DeepEvalInputs,
    cohen_kappa,
    deepeval_inputs,
)
from taxverity.llm.cache import CachedLLMClient  # noqa: E402
from taxverity.llm.client import OPENAI_MINI, LLMClient, Message  # noqa: E402
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
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--pause", type=float, default=DEFAULT_PAUSE)
    parser.add_argument("--source", choices=("smoke", "v1"), default="smoke")
    parser.add_argument("--run", type=Path, default=V1_RUN)
    args = parser.parse_args()
    configure_logging()
    settings = Settings()
    if args.source == "v1":
        return run_v1(settings, args.run, args.limit)

    try:
        run = load_answer_run(settings.data_dir / "answers" / ANSWER_RUN_FILENAME)
        base = LLMClient.from_settings(settings)
    except (FileNotFoundError, MissingSettingError) as error:
        print(f"error: {error} (run scripts/answer_smoke.py first)", file=sys.stderr)
        return 1
    judge = Judge(CachedLLMClient(base, settings.llm_cache_dir), base.primary.model)

    # Only answers with a served claim can be judged; a negative's honest
    # silence has no output to score.
    sample = [r for r in run.records if r.claims][: args.limit or DEFAULT_LIMIT]
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


V1_RUN = Path("data") / "answers" / "generation_eval_v1strict.json"
V1_REPORT = Path("reports") / "deepeval_generation_v1.md"
V1_SCORES = Path("data") / "answers" / "deepeval_v1.json"
# DeepEval's own default pass mark; used only to binarise for kappa.
FAITHFUL = 1.0
METRICS = ("faithfulness", "answer_relevancy", "contextual_relevancy")


def _measure(metric, case, item_id: str) -> dict:
    try:
        metric.measure(case)
    except Exception as error:  # a judge failure is reported, never fatal
        logger.warning("%s: %s failed: %s", item_id, type(metric).__name__, error)
        return {"score": None, "reason": f"error: {error}"}
    return {"score": metric.score, "reason": metric.reason}


def run_v1(settings: Settings, run_path: Path, limit: int | None) -> int:
    try:
        run = json.loads(run_path.read_text(encoding="utf-8"))
        # No fallback: a judge that switched vendors midway would make its
        # scores incomparable.
        base = LLMClient(OPENAI_MINI, settings.require(OPENAI_MINI.settings_key))
    except (FileNotFoundError, MissingSettingError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    judge = Judge(
        CachedLLMClient(base, settings.llm_cache_dir / "deepeval"), OPENAI_MINI.model
    )
    inputs = [
        x
        for item_id, record in sorted(run["records"].items())
        if (x := deepeval_inputs(item_id, record)) is not None
    ][: limit or None]

    stored = (
        json.loads(V1_SCORES.read_text(encoding="utf-8")) if V1_SCORES.exists() else {}
    )
    for index, x in enumerate(inputs, start=1):
        done = stored.get(x.item_id)
        if done and all(done[m]["score"] is not None for m in METRICS):
            continue
        grounded_case = LLMTestCase(
            input=x.question,
            actual_output=x.grounded_output,
            retrieval_context=list(x.cited_context),
        )
        answer_case = LLMTestCase(
            input=x.question,
            actual_output=x.answer_output,
            retrieval_context=list(x.full_context),
        )
        stored[x.item_id] = {
            "faithfulness": _measure(
                FaithfulnessMetric(model=judge, async_mode=False),
                grounded_case,
                x.item_id,
            ),
            "answer_relevancy": _measure(
                AnswerRelevancyMetric(model=judge, async_mode=False),
                answer_case,
                x.item_id,
            ),
            "contextual_relevancy": _measure(
                ContextualRelevancyMetric(model=judge, async_mode=False),
                answer_case,
                x.item_id,
            ),
        }
        V1_SCORES.write_text(
            json.dumps(stored, indent=1, ensure_ascii=False), encoding="utf-8"
        )
        logger.info("%s (%d/%d) judged", x.item_id, index, len(inputs))

    lines = _v1_report(
        inputs, stored, OPENAI_MINI.model, sum(base.tokens_used.values())
    )
    V1_REPORT.parent.mkdir(parents=True, exist_ok=True)
    V1_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="")
    print("\n".join(lines[:26]))
    print(f"wrote {V1_REPORT}")
    return 0


def _score_of(stored: dict, item_id: str, name: str) -> float | None:
    return stored.get(item_id, {}).get(name, {}).get("score")


def _v1_report(
    inputs: list[DeepEvalInputs], stored: dict, model: str, tokens: int
) -> list[str]:
    def scores(name: str) -> list[float]:
        return [
            s for x in inputs if (s := _score_of(stored, x.item_id, name)) is not None
        ]

    def mean(values: list[float]) -> str:
        return f"{statistics.fmean(values):.3f}" if values else "—"

    paired = [
        (x.our_supported_rate, x.our_all_supported, f)
        for x in inputs
        if x.our_supported_rate is not None
        and (f := _score_of(stored, x.item_id, "faithfulness")) is not None
    ]
    rho = agree = kappa = "—"
    if len(paired) >= 3:
        try:
            rho = "{:.3f}".format(
                statistics.correlation(
                    [p[0] for p in paired], [p[2] for p in paired], method="ranked"
                )
            )
        except statistics.StatisticsError:
            rho = "undefined (a constant series)"
        ours = [bool(p[1]) for p in paired]
        theirs = [p[2] >= FAITHFUL for p in paired]
        matches = sum(a == b for a, b in zip(ours, theirs, strict=True))
        agree = f"{matches / len(paired):.3f}"
        kappa = f"{cohen_kappa(ours, theirs):.3f}"

    faith = scores("faithfulness")
    relevancy = scores("answer_relevancy")
    context = scores("contextual_relevancy")
    lines = [
        "# DeepEval cross-check — generation eval v1",
        "",
        "Auto-generated by `scripts/deepeval_sample.py --source v1`. Report only,",
        f"no gate (ADR-053). Judge: `{model}`, the model that wrote the answers,",
        "so these scores are partly a self-consistency signal. Our own claim",
        "judge is the one calibrated against human labels (kappa 0.634).",
        "",
        f"{len(inputs)} answers with a grounded served claim; "
        f"{tokens:,} judge tokens this run (0 on a cached re-run).",
        "",
        "## Standard metric names",
        "",
        "| metric | DeepEval mean | n | our equivalent |",
        "|---|---|---|---|",
        f"| Faithfulness | {mean(faith)} | {len(faith)} | claim judge: supported "
        "0.746, unsupported-served 0.020; live verifier gate on every line |",
        f"| Answer relevancy | {mean(relevancy)} | {len(relevancy)} | none before "
        "this run (key-point recall 0.830 measures completeness) |",
        f"| Contextual relevancy | {mean(context)} | {len(context)} | retrieval "
        "gold v2 metrics; answer-gold pack coverage 0.898 |",
        "| Citation accuracy | no DeepEval metric | — | deterministic `cites_gold`"
        " / `gold_citation_coverage`; the verifier resolves every `[n]` |",
        "",
        "## Faithfulness agreement with our claim judge",
        "",
        f"- paired answers: {len(paired)}",
        f"- Spearman, our supported share vs DeepEval faithfulness: {rho}",
        f"- binary, all claims supported vs faithfulness >= {FAITHFUL}: "
        f"agreement {agree}, kappa {kappa}",
        "",
        "## Per item",
        "",
        "| item | ours supported | faithfulness | answer rel. | context rel. |",
        "|---|---|---|---|---|",
    ]
    for x in inputs:
        cells = [
            "error" if (s := _score_of(stored, x.item_id, m)) is None else f"{s:.2f}"
            for m in METRICS
        ]
        ours = "—" if x.our_supported_rate is None else f"{x.our_supported_rate:.2f}"
        lines.append(f"| {x.item_id} | {ours} | " + " | ".join(cells) + " |")
    lines += ["", "## Lowest faithfulness — judge reasons", ""]
    low = sorted(
        (x for x in inputs if _score_of(stored, x.item_id, "faithfulness") is not None),
        key=lambda x: _score_of(stored, x.item_id, "faithfulness"),
    )[:10]
    for x in low:
        f = stored[x.item_id]["faithfulness"]
        lines.append(f"- **{x.item_id}** ({f['score']:.2f}): {f['reason']}")
    return lines


def _score(metric) -> str:
    return "error" if metric.score is None else f"{metric.score:.2f}"


if __name__ == "__main__":
    raise SystemExit(main())
