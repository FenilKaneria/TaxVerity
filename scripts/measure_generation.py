"""Generation eval — answer quality over `evals/datasets/answer_gold_v1.jsonl`.

Four steps, each re-runnable on its own because each reads and writes the
stored run `data/answers/generation_eval_<label>.json`:

1. `run`: every item through `build_deps` + `build_graph` as the API runs a
   turn (same routing, models, verifier, evidence gate) against the LOCAL
   compose database, one throwaway user and thread per item. Records the
   served events, the verifier's outcome on every line before and after the
   repair (`AnswerGenerator.observer`), the evidence pack as shown, latency
   and tokens.
2. `judge`: an LLM from a different vendor than the generator labels each
   served statute claim against the passages it cites, each first-pass line
   the same way (the ungated ablation), and each answer's key-point coverage.
   Cached on disk, so re-judging a stored run costs nothing.
3. `export-labels` / `agreement`: a seeded sample of judged claims for a
   human to label blind, then Cohen's kappa between human and judge. The
   judge's numbers are quoted only if kappa clears the floor below.
4. `report`: `reports/generation_eval.md`.

    uv run python scripts/measure_generation.py run --label v1
    uv run python scripts/measure_generation.py judge --label v1
    uv run python scripts/measure_generation.py export-labels --label v1
    uv run python scripts/measure_generation.py agreement --label v1
    uv run python scripts/measure_generation.py report --label v1

Adoption floors, fixed before the first run (rule 01's measurement habit):
served unsupported-claim rate <= 2%, abstention recall >= 0.90, calculator
exact match 100%, and judge-human kappa >= 0.60 before any judged number is
quoted.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import secrets
import sys
import time
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import psycopg

from taxverity.auth.passwords import hash_password
from taxverity.config import MissingSettingError, Settings
from taxverity.evals.generation import (
    CLAIM_JUDGE_PROMPT,
    GENERATION_EVAL_VERSION,
    JUDGED_TYPES,
    KEY_POINT_JUDGE_PROMPT,
    ClaimLabel,
    ItemKind,
    JudgeParseError,
    claim_judge_input,
    cohen_kappa,
    key_point_judge_input,
    load_answer_gold,
    parse_claim_label,
    parse_coverage,
    passages_for,
    summarise,
)
from taxverity.generation.generate import DraftOutcome
from taxverity.graph.build import build_deps, build_graph
from taxverity.llm.cache import CachedLLMClient
from taxverity.llm.client import GEMINI, GROQ, OPENAI_MINI, LLMClient, Message
from taxverity.observability import configure_logging, get_logger
from taxverity.threads.store import create_thread

logger = get_logger(__name__)

RESULT_DIR = Path("data") / "answers"
REPORT_PATH = Path("reports") / "generation_eval.md"
LOCAL_DATABASE_URL = "postgresql://taxverity:taxverity@127.0.0.1:5432/taxverity"
# Same reasoning as advisor_smoke.py: lets free-tier token buckets refill so
# each turn's latency is one person's turn, not the previous turn's debt.
DEFAULT_PAUSE = 20.0
JUDGES = {"gemini": GEMINI, "groq": GROQ, "openai": OPENAI_MINI}
DEFAULT_JUDGE = "gemini"  # generation runs on OpenAI gpt-5-mini (ADR-129)
JUDGE_MAX_TOKENS = 600
LABEL_SAMPLE = 30
LABEL_SEED = 20261004

FLOORS = {
    "unsupported_rate_served": ("<=", 0.02),
    "abstention_recall": (">=", 0.90),
    "calculator_exact": (">=", 1.00),
}
KAPPA_FLOOR = 0.60


def _result_path(label: str) -> Path:
    return RESULT_DIR / f"generation_eval_{label}.json"


def _load(label: str) -> dict[str, Any]:
    run = json.loads(_result_path(label).read_text(encoding="utf-8"))
    if run.get("version") != GENERATION_EVAL_VERSION:
        raise ValueError(f"{label} was written by eval version {run.get('version')}")
    return run


def _store(label: str, run: dict[str, Any]) -> None:
    path = _result_path(label)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(run, ensure_ascii=False, indent=1, default=str) + "\n",
        encoding="utf-8",
        newline="",
    )


# --- 1. run ----------------------------------------------------------------


class TokenRecorder(logging.Handler):
    """Sums `LLMClient`'s own per-call boundary records (advisor_smoke.py's
    format) into one turn's call count and tokens."""

    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self.calls = 0
        self.tokens = 0

    def emit(self, record: logging.LogRecord) -> None:
        args = record.args if isinstance(record.args, tuple) else ()
        if str(record.msg).startswith("llm %s/%s answered") and len(args) >= 5:
            self.calls += 1
            self.tokens += int(args[3]) + int(args[4])

    def take(self) -> tuple[int, int]:
        taken = (self.calls, self.tokens)
        self.calls, self.tokens = 0, 0
        return taken


def _require_local(url: str) -> None:
    # Creates users, threads and messages: never against production.
    host = urlparse(url).hostname
    if host not in ("127.0.0.1", "localhost"):
        raise RuntimeError(f"refusing to run against a non-local database host: {host}")


def _create_user(conn: psycopg.Connection) -> uuid.UUID:
    email = f"generation-eval-{uuid.uuid4().hex[:12]}@example.com"
    (user_id,) = conn.execute(
        "INSERT INTO users (email, password_hash, email_verified_at) "
        "VALUES (%s, %s, now()) RETURNING user_id",
        (email, hash_password(secrets.token_urlsafe(24))),
    ).fetchone()
    return user_id


def run_item(
    graph: Any, generator: Any, user_id: uuid.UUID, thread_id: uuid.UUID, question: str
) -> dict:
    # A corrective retry calls the generator twice; the last pass is the one
    # whose pack and lines the served answer came from.
    passes: list[dict[str, list[DraftOutcome]]] = []

    def observe(stage: str, outcomes: list[DraftOutcome]) -> None:
        if stage == "first_pass":
            passes.append({})
        passes[-1][stage] = outcomes

    generator.observer = observe
    started = time.perf_counter()
    emitted: list[dict[str, Any]] = []
    final_state: dict[str, Any] = {}
    try:
        for mode, chunk in graph.stream(
            {"user_id": user_id, "thread_id": thread_id, "question": question},
            stream_mode=["custom", "values"],
        ):
            if mode == "custom":
                emitted.append(chunk)
            else:
                final_state = chunk
    finally:
        generator.observer = None
    seconds = round(time.perf_counter() - started, 2)

    pack = final_state.get("pack")
    evidence = (
        [
            {
                "marker": marker,
                "citation": unit.citation,
                "text": "\n".join(
                    [*(line.text for line in unit.context), unit.chunk.text]
                ),
            }
            for marker, unit in enumerate(pack.units, start=1)
        ]
        if pack is not None
        else []
    )
    last = passes[-1] if passes else {}
    category = final_state.get("category")
    final = final_state.get("final")
    return {
        "question": question,
        "seconds": seconds,
        "category": category.value if category is not None else None,
        "route": final.route if final is not None else None,
        "retried": bool(final_state.get("retried")),
        "served": [e for e in emitted if "verified" in e],
        "withheld": [e for e in emitted if "reason" in e and "verified" not in e],
        "fixed_text": final_state.get("answer_text"),
        "first_pass": [vars(o) for o in last.get("first_pass", [])],
        "final": [vars(o) for o in last.get("final", [])],
        "evidence": evidence,
    }


def cmd_run(args: argparse.Namespace) -> int:
    settings = Settings(database_url=args.database_url)
    url = settings.require("database_url")
    _require_local(url)
    items = [i for i in load_answer_gold() if not args.only or i.item_id in args.only]
    if args.kind:
        items = [i for i in items if i.kind.value in args.kind]
    path = _result_path(args.label)
    run = (
        _load(args.label)
        if path.exists()
        else {"version": GENERATION_EVAL_VERSION, "label": args.label, "records": {}}
    )
    recorder = TokenRecorder()
    llm_logger = logging.getLogger("taxverity.llm.client")
    llm_logger.addHandler(recorder)
    llm_logger.setLevel(logging.INFO)
    try:
        with psycopg.connect(url, autocommit=True) as conn:
            deps, served = build_deps(settings, conn)
            graph = build_graph(deps)
            run["corpus_version"] = served.corpus_version
            run["run_at"] = time.strftime("%Y-%m-%d %H:%M")
            recorder.take()
            first = True
            for item in items:
                if item.item_id in run["records"] and not args.rerun:
                    continue  # resumable: a stopped run picks up where it left off
                if not first:
                    time.sleep(args.pause)
                first = False
                user_id = _create_user(conn)
                thread = create_thread(conn, user_id, f"generation eval {item.item_id}")
                try:
                    record = run_item(
                        graph, deps.generator, user_id, thread.thread_id, item.question
                    )
                except Exception as error:  # noqa: BLE001 — record, keep going
                    logger.exception("item failed: %s", item.item_id)
                    record = {"question": item.question, "error": repr(error)}
                finally:
                    conn.execute("DELETE FROM users WHERE user_id = %s", (user_id,))
                record["llm_calls"], record["tokens"] = recorder.take()
                run["records"][item.item_id] = record
                _store(args.label, run)  # after every item, so nothing is lost
                logger.info(
                    "%s: %s served, %s withheld, %ss",
                    item.item_id,
                    len(record.get("served", [])),
                    len(record.get("withheld", [])),
                    record.get("seconds"),
                )
    finally:
        llm_logger.removeHandler(recorder)
    print(f"wrote {_result_path(args.label)}")
    return 0


# --- 2. judge --------------------------------------------------------------


def _judge_client(settings: Settings, name: str) -> CachedLLMClient:
    provider = JUDGES[name]
    # No fallback on purpose: a judge that silently switched vendors halfway
    # through would make its labels incomparable.
    client = LLMClient(provider, settings.require(provider.settings_key))
    return CachedLLMClient(client, settings.llm_cache_dir / "judge")


def _ask(client: CachedLLMClient, system: str, user: str) -> str:
    completion = client.complete(
        [Message(role="system", content=system), Message(role="user", content=user)],
        max_completion_tokens=JUDGE_MAX_TOKENS,
        response_format={"type": "json_object"},
        temperature=0.0,
    )
    return completion.text


def _label_claim(client: CachedLLMClient, text: str, passages: list) -> str:
    if not passages:
        return ClaimLabel.UNCITED.value
    reply = _ask(client, CLAIM_JUDGE_PROMPT, claim_judge_input(text, passages))
    try:
        return parse_claim_label(reply).value
    except JudgeParseError:
        # Left out of every rate rather than guessed; the count of judged
        # claims in the report shows how many were dropped.
        logger.warning("judge reply unusable, recorded as unjudged: %r", reply[:120])
        return "unjudged"


def cmd_judge(args: argparse.Namespace) -> int:
    settings = Settings()
    try:
        client = _judge_client(settings, args.judge)
    except MissingSettingError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    items = {i.item_id: i for i in load_answer_gold()}
    run = _load(args.label)
    run["judge_model"] = JUDGES[args.judge].model
    for item_id, record in run["records"].items():
        if "error" in record:
            continue
        evidence = {e["marker"]: (e["citation"], e["text"]) for e in record["evidence"]}
        served = {
            str(claim["id"]): _label_claim(
                client, claim["text"], passages_for(claim["text"], evidence)
            )
            for claim in record["served"]
            if claim["type"] in JUDGED_TYPES and claim.get("citations")
        }
        first_pass = {
            str(line["claim_id"]): _label_claim(
                client, line["line"], passages_for(line["line"], evidence)
            )
            for line in record["first_pass"]
            if line["type"] in JUDGED_TYPES
        }
        judge: dict[str, Any] = {
            "served": {k: v for k, v in served.items() if v != "unjudged"},
            "first_pass": {k: v for k, v in first_pass.items() if v != "unjudged"},
        }
        item = items[item_id]
        if item.kind is ItemKind.ANSWERABLE:
            answer = "\n".join(c["text"] for c in record["served"]) or (
                record.get("fixed_text") or ""
            )
            reply = _ask(
                client,
                KEY_POINT_JUDGE_PROMPT,
                key_point_judge_input(answer, item.key_points),
            )
            try:
                judge["key_points"] = list(parse_coverage(reply, len(item.key_points)))
            except JudgeParseError:
                logger.warning("%s: key-point reply unusable", item_id)
        record["judge"] = judge
        logger.info("%s judged", item_id)
    _store(args.label, run)
    print(
        f"judged with {run['judge_model']}; cache {client.hits} hits, {client.misses} misses"
    )
    return 0


# --- 3. human agreement ----------------------------------------------------


def _labels_path(label: str) -> Path:
    return RESULT_DIR / f"generation_eval_{label}_human_labels.csv"


def cmd_export_labels(args: argparse.Namespace) -> int:
    run = _load(args.label)
    rows = []
    for item_id, record in sorted(run["records"].items()):
        judge = record.get("judge")
        if not judge:
            continue
        evidence = {e["marker"]: (e["citation"], e["text"]) for e in record["evidence"]}
        lines = {str(c["id"]): c["text"] for c in record["served"]}
        raw = {str(o["claim_id"]): o["line"] for o in record["first_pass"]}
        for stage, texts in (("served", lines), ("first_pass", raw)):
            for claim_id, label in judge.get(stage, {}).items():
                if label == ClaimLabel.UNCITED.value:
                    continue  # assigned by rule, nothing for a human to check
                text = texts[claim_id]
                passages = passages_for(text, evidence)
                rows.append(
                    {
                        "item_id": item_id,
                        "stage": stage,
                        "claim_id": claim_id,
                        "sentence": text,
                        "passages": "\n\n".join(f"[{c}] {t}" for c, t in passages),
                        "human_label": "",
                    }
                )
    sample = random.Random(LABEL_SEED).sample(rows, min(args.n, len(rows)))
    path = _labels_path(args.label)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(sample[0]) if sample else [])
        writer.writeheader()
        writer.writerows(sample)
    print(
        f"wrote {len(sample)} claims to {path}. Fill human_label with one of: "
        + ", ".join(
            label.value for label in ClaimLabel if label is not ClaimLabel.UNCITED
        )
        + ". Do not look at the judge's labels first."
    )
    return 0


def cmd_agreement(args: argparse.Namespace) -> int:
    run = _load(args.label)
    with _labels_path(args.label).open(encoding="utf-8") as handle:
        rows = [r for r in csv.DictReader(handle) if r["human_label"].strip()]
    human, judge = [], []
    for row in rows:
        judged = run["records"][row["item_id"]]["judge"][row["stage"]][row["claim_id"]]
        human.append(ClaimLabel(row["human_label"].strip()) is ClaimLabel.SUPPORTED)
        judge.append(ClaimLabel(judged) is ClaimLabel.SUPPORTED)
    kappa = cohen_kappa(human, judge)
    agreement = sum(h == j for h, j in zip(human, judge, strict=True)) / len(human)
    run["agreement"] = {"n": len(human), "agreement": agreement, "kappa": kappa}
    _store(args.label, run)
    print(json.dumps(run["agreement"], indent=1))
    return 0


# --- 4. report -------------------------------------------------------------


def _fmt(value: Any) -> str:
    if value is None:
        return "not measured"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _verdict(summary: dict[str, Any], key: str) -> str:
    op, floor = FLOORS[key]
    value = summary.get(key)
    if value is None:
        return "not measured"
    ok = value <= floor if op == "<=" else value >= floor
    return ("PASS" if ok else "FAIL") + f" ({op} {floor})"


def render(run: dict[str, Any]) -> str:
    items = load_answer_gold()
    summary = summarise(items, run["records"])
    agreement = run.get("agreement")
    trusted = agreement is not None and agreement["kappa"] >= KAPPA_FLOOR
    rows = [
        ("Items run / errors", f"{summary['items_run']} / {summary['errors']}"),
        ("Lines on first pass", summary["lines_first_pass"]),
        ("First-pass verifier pass rate", summary["first_pass_pass_rate"]),
        ("Repair success rate", summary["repair_success_rate"]),
        ("Withheld-line rate (final)", summary["withheld_line_rate"]),
        ("Answer rate (answerable with a grounded claim)", summary["answer_rate"]),
        ("Citation hit rate (cites a gold provision)", summary["citation_hit_rate"]),
        ("Gold citation coverage", summary["gold_citation_coverage"]),
        ("Abstention recall (negatives)", summary["abstention_recall"]),
        ("False abstention rate (answerable)", summary["false_abstention_rate"]),
        ("Calculator exact match", summary["calculator_exact"]),
        ("Safety cases correct", summary["safety_correct"]),
        ("Claims judged (served)", summary["claims_judged"]),
        ("Supported rate (served)", summary["supported_rate"]),
        ("Partially supported rate (served)", summary["partial_rate"]),
        ("Unsupported-claim rate, served (gated)", summary["unsupported_rate_served"]),
        (
            "Unsupported-claim rate, first pass (ungated)",
            summary["unsupported_rate_ungated"],
        ),
        ("Key-point recall", summary["key_point_recall"]),
        (
            "Latency p50 / p95 (s)",
            f"{summary['latency_p50_s']:.1f} / {summary['latency_p95_s']:.1f}",
        ),
        ("Tokens per turn", f"{summary['tokens_per_turn']:,.0f}"),
    ]
    lines = [
        "# Generation eval",
        "",
        "Auto-generated by `scripts/measure_generation.py` over",
        "`evals/datasets/answer_gold_v1.jsonl` (49 answerable, 16 negative,",
        "8 calculation, 8 safety items), through the compiled graph.",
        "",
        f"Run `{run['label']}` at {run.get('run_at', '?')}; "
        f"judge `{run.get('judge_model', 'not run')}`.",
        "",
        "| measure | value |",
        "|---|---|",
        *(f"| {name} | {_fmt(value)} |" for name, value in rows),
        "",
        "## Floors (fixed before the first run)",
        "",
        *(f"- `{key}`: {_verdict(summary, key)}" for key in FLOORS),
        "",
        "## Judge agreement",
        "",
    ]
    if agreement is None:
        lines.append("Not measured: judged rates above are not yet quotable.")
    else:
        lines.append(
            f"{agreement['n']} claims labelled blind by a human: agreement "
            f"{agreement['agreement']:.3f}, Cohen's kappa {agreement['kappa']:.3f} "
            f"({'meets' if trusted else 'below'} the {KAPPA_FLOOR} floor)."
        )
    return "\n".join(lines) + "\n"


def cmd_report(args: argparse.Namespace) -> int:
    run = _load(args.label)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(render(run), encoding="utf-8", newline="")
    print(json.dumps(summarise(load_answer_gold(), run["records"]), indent=1))
    print(f"wrote {REPORT_PATH}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run the items through the graph")
    run.add_argument("--label", required=True)
    run.add_argument("--only", nargs="*", default=[], help="item ids, e.g. g001 g002")
    run.add_argument(
        "--kind", nargs="*", default=[], choices=[k.value for k in ItemKind]
    )
    run.add_argument("--pause", type=float, default=DEFAULT_PAUSE)
    run.add_argument("--rerun", action="store_true", help="redo items already stored")
    run.add_argument("--database-url", default=LOCAL_DATABASE_URL)
    run.set_defaults(func=cmd_run)

    judge = sub.add_parser("judge", help="label a stored run with the LLM judge")
    judge.add_argument("--label", required=True)
    judge.add_argument("--judge", choices=sorted(JUDGES), default=DEFAULT_JUDGE)
    judge.set_defaults(func=cmd_judge)

    export = sub.add_parser("export-labels", help="sample claims for a human")
    export.add_argument("--label", required=True)
    export.add_argument("--n", type=int, default=LABEL_SAMPLE)
    export.set_defaults(func=cmd_export_labels)

    agree = sub.add_parser("agreement", help="kappa between human and judge")
    agree.add_argument("--label", required=True)
    agree.set_defaults(func=cmd_agreement)

    report = sub.add_parser("report", help="write reports/generation_eval.md")
    report.add_argument("--label", required=True)
    report.set_defaults(func=cmd_report)

    args = parser.parse_args()
    configure_logging()
    try:
        return args.func(args)
    except (MissingSettingError, RuntimeError, FileNotFoundError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
