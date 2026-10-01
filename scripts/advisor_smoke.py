"""R22 Part A — the advisor smoke set, driven through the compiled graph.

Report-only (ADR-110): no new eval framework, no score gate on answer quality.
Each script in `evals/datasets/advisor_smoke_v1.jsonl` is a short multi-turn
conversation run through `build_deps` + `build_graph` exactly as the API runs a
turn — same routing, same per-node models, same verifier — against the LOCAL
compose database only, with a throwaway user and thread per script.

What it records per turn, for a human to read and for before/after comparison
across runs (`--label`):
- the route, the intent, and the served/withheld counts;
- every LLM call the turn made (provider, model, tokens, seconds), plus
  retries and cross-vendor fallbacks, captured from `taxverity.llm.client`'s
  own boundary log records rather than by instrumenting each node;
- per-node wall-clock time (the graph's own trace) and the answer text.

Two checks stay hard, because they belong to the rigorous list (rule 01):
a `guard` script's turns serve no claim at all, and no served text in a
`guard` script carries a forbidden figure.

Writes `data/answers/advisor_smoke_<label>.json` and `reports/advisor_smoke.md`.
"""

from __future__ import annotations

import argparse
import json
import logging
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
from taxverity.generation.claims import ClaimEvent
from taxverity.generation.verifier import numbers_in
from taxverity.graph.build import build_deps, build_graph
from taxverity.observability import configure_logging, get_logger
from taxverity.threads.store import create_thread

logger = get_logger(__name__)

DATASET_PATH = Path("evals") / "datasets" / "advisor_smoke_v1.jsonl"
RESULT_DIR = Path("data") / "answers"
REPORT_PATH = Path("reports") / "advisor_smoke.md"
LOCAL_DATABASE_URL = "postgresql://taxverity:taxverity@127.0.0.1:5432/taxverity"
# A pause between turns lets the per-minute token bucket partly refill, so a
# turn's latency is measured as one person's turn would be, not as the tail of
# the previous turn's rate-limit debt. Same pause on every run being compared.
DEFAULT_PAUSE = 30.0
ADVISOR_SMOKE_VERSION = 1


class LLMCallRecorder(logging.Handler):
    """Collects `LLMClient`'s own boundary records: one per answered call, one
    per retry, one per fallback. The format strings are `llm/client.py`'s."""

    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self.calls: list[dict[str, Any]] = []
        self.retries: list[str] = []
        self.fallbacks: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        message = str(record.msg)
        args = record.args if isinstance(record.args, tuple) else ()
        if message.startswith("llm %s/%s answered") and len(args) >= 8:
            provider, model, seconds, prompt, completion, reasoning, finish, _ = args[
                :8
            ]
            self.calls.append(
                {
                    "provider": provider,
                    "model": model,
                    "seconds": round(float(seconds), 2),
                    "prompt_tokens": int(prompt),
                    "completion_tokens": int(completion),
                    "reasoning_tokens": int(reasoning),
                    "finish_reason": finish,
                }
            )
        elif "call failed (attempt" in message and args:
            self.retries.append(f"{args[0]}: {args[3]} (waited {float(args[4]):.1f}s)")
        elif "unavailable" in message and "falling back" in message and args:
            self.fallbacks.append(f"{args[0]} -> {args[2]}")

    def take(self) -> dict[str, Any]:
        taken = {
            "calls": self.calls,
            "retries": self.retries,
            "fallbacks": self.fallbacks,
        }
        self.calls, self.retries, self.fallbacks = [], [], []
        return taken


def load_scripts(path: Path = DATASET_PATH) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _require_local(url: str) -> None:
    # This script creates users, threads and messages. It must never write
    # them into the production database `.env` points at.
    host = urlparse(url).hostname
    if host not in ("127.0.0.1", "localhost"):
        raise RuntimeError(f"refusing to run against a non-local database host: {host}")


def _create_user(conn: psycopg.Connection) -> uuid.UUID:
    email = f"advisor-smoke-{uuid.uuid4().hex[:12]}@example.com"
    (user_id,) = conn.execute(
        "INSERT INTO users (email, password_hash, email_verified_at) "
        "VALUES (%s, %s, now()) RETURNING user_id",
        (email, hash_password(secrets.token_urlsafe(24))),
    ).fetchone()
    return user_id


def run_turn(
    graph: Any, user_id: uuid.UUID, thread_id: uuid.UUID, question: str
) -> dict:
    started = time.perf_counter()
    events: list[dict[str, Any]] = []
    final_state: dict[str, Any] = {}
    for mode, chunk in graph.stream(
        {"user_id": user_id, "thread_id": thread_id, "question": question},
        stream_mode=["custom", "values"],
    ):
        if mode == "custom":
            events.append(chunk)
        else:
            final_state = chunk
    elapsed = round(time.perf_counter() - started, 1)

    claim_events = [
        e for e in final_state.get("events", []) if isinstance(e, ClaimEvent)
    ]
    final = final_state.get("final")
    answer_text = final_state.get("answer_text")
    served_text = (
        answer_text
        if answer_text is not None
        else "\n".join(event.text for event in claim_events)
    )
    category = final_state.get("category")
    intent = final_state.get("intent")
    return {
        "question": question,
        "seconds": elapsed,
        "category": category.value if category is not None else None,
        "intent": intent.value if intent is not None else None,
        "route": final.route if final is not None else None,
        "rewritten_query": final_state.get("query"),
        "served": sum(1 for e in events if "verified" in e),
        "withheld": sum(1 for e in events if "reason" in e and "verified" not in e),
        "emitted_claims": [e for e in events if "verified" in e or "reason" in e],
        "served_types": [event.type.value for event in claim_events],
        "clarify": list(final_state.get("clarify_questions", ())),
        "retried": bool(final_state.get("retried")),
        "fixed_text": answer_text,
        "served_text": served_text,
        "stages": [e.get("stage") for e in events if "stage" in e],
        "trace": [entry.model_dump() for entry in final.trace] if final else [],
    }


def run_live(settings: Settings, label: str, pause: float, only: set[str]) -> dict:
    url = settings.require("database_url")
    _require_local(url)
    recorder = LLMCallRecorder()
    llm_logger = logging.getLogger("taxverity.llm.client")
    llm_logger.addHandler(recorder)
    llm_logger.setLevel(logging.INFO)

    records: list[dict[str, Any]] = []
    with psycopg.connect(url, autocommit=True) as conn:
        deps, served = build_deps(settings, conn)
        graph = build_graph(deps)
        recorder.take()  # drop anything logged while composing
        scripts = [s for s in load_scripts() if not only or s["id"] in only]
        first = True
        for script in scripts:
            user_id = _create_user(conn)
            thread = create_thread(conn, user_id, f"advisor smoke {script['id']}")
            turns = []
            try:
                for question in script["turns"]:
                    if not first:
                        time.sleep(pause)
                    first = False
                    try:
                        turn = run_turn(graph, user_id, thread.thread_id, question)
                    except Exception as error:  # noqa: BLE001 — record, keep going
                        logger.exception("turn failed: %s", script["id"])
                        turn = {"question": question, "error": repr(error)}
                    turn["llm"] = recorder.take()
                    turns.append(turn)
                    logger.info(
                        "%s turn %d: %s in %ss",
                        script["id"],
                        len(turns),
                        turn.get("route"),
                        turn.get("seconds"),
                    )
            finally:
                conn.execute("DELETE FROM users WHERE user_id = %s", (user_id,))
            records.append({**script, "results": turns})
    llm_logger.removeHandler(recorder)
    return {
        "version": ADVISOR_SMOKE_VERSION,
        "label": label,
        "run_at": time.strftime("%Y-%m-%d %H:%M"),
        "corpus_version": served.corpus_version,
        "pause": pause,
        "records": records,
    }


def guard_failures(run: dict) -> list[str]:
    failures: list[str] = []
    for record in run["records"]:
        if record.get("expect") != "guard":
            continue
        forbidden = {
            value
            for raw in record.get("forbidden_numbers", [])
            for value in numbers_in(raw)
        }
        for i, turn in enumerate(record["results"], start=1):
            if turn.get("served"):
                failures.append(
                    f"{record['id']} turn {i}: served {turn['served']} claim(s)"
                )
            leaked = numbers_in(turn.get("served_text") or "") & forbidden
            if leaked:
                failures.append(
                    f"{record['id']} turn {i}: served forbidden figure(s) {sorted(leaked)}"
                )
    return failures


def _turn_totals(turn: dict) -> dict[str, Any]:
    calls = turn.get("llm", {}).get("calls", [])
    by_model: dict[str, int] = {}
    for call in calls:
        key = f"{call['provider']}/{call['model']}"
        by_model[key] = (
            by_model.get(key, 0) + call["prompt_tokens"] + call["completion_tokens"]
        )
    return {
        "calls": len(calls),
        "tokens": sum(by_model.values()),
        "by_model": by_model,
        "retries": len(turn.get("llm", {}).get("retries", [])),
        "fallbacks": len(turn.get("llm", {}).get("fallbacks", [])),
    }


def summarise(run: dict) -> dict[str, Any]:
    turns = [t for r in run["records"] for t in r["results"] if "error" not in t]
    seconds = sorted(t["seconds"] for t in turns)
    totals = [_turn_totals(t) for t in turns]
    by_model: dict[str, int] = {}
    for total in totals:
        for key, value in total["by_model"].items():
            by_model[key] = by_model.get(key, 0) + value
    return {
        "turns": len(turns),
        "errors": sum(1 for r in run["records"] for t in r["results"] if "error" in t),
        "median_seconds": seconds[len(seconds) // 2] if seconds else 0,
        "max_seconds": seconds[-1] if seconds else 0,
        "total_seconds": round(sum(seconds), 1),
        "llm_calls": sum(t["calls"] for t in totals),
        "tokens_by_model": by_model,
        "retries": sum(t["retries"] for t in totals),
        "fallbacks": sum(t["fallbacks"] for t in totals),
        "served": sum(t["served"] for t in turns),
        "withheld": sum(t["withheld"] for t in turns),
        "corrective_retries": sum(1 for t in turns if t.get("retried")),
        "insufficient_evidence": sum(
            1
            for t in turns
            if (t.get("fixed_text") or "").startswith("I couldn't find")
        ),
    }


def render(run: dict, baseline: dict | None) -> str:
    summary = summarise(run)
    lines = [
        "# Advisor smoke set — R22",
        "",
        "Auto-generated by `scripts/advisor_smoke.py`. Report-only (ADR-110):",
        "multi-turn conversations through the compiled graph against the local",
        "database. Token counts come from `LLMClient`'s own log records; a Gemini",
        "call's count undercounts its hidden reasoning (rule 05).",
        "",
        f"Run `{run['label']}` at {run['run_at']}, pause {run['pause']}s between turns.",
        "",
    ]
    columns = [("this run", summary)]
    if baseline is not None:
        columns.insert(0, (f"`{baseline['label']}`", summarise(baseline)))
    header = "| measure | " + " | ".join(name for name, _ in columns) + " |"
    lines += [header, "|---|" + "---|" * len(columns)]
    for key in (
        "turns", "errors", "median_seconds", "max_seconds", "total_seconds",
        "llm_calls", "retries", "fallbacks", "served", "withheld",
        "corrective_retries", "insufficient_evidence",
    ):  # fmt: skip
        lines.append(f"| {key} | " + " | ".join(str(s[key]) for _, s in columns) + " |")
    models = sorted({m for _, s in columns for m in s["tokens_by_model"]})
    for model in models:
        lines.append(
            f"| tokens {model} | "
            + " | ".join(f"{s['tokens_by_model'].get(model, 0):,}" for _, s in columns)
            + " |"
        )

    failures = guard_failures(run)
    lines += [
        "",
        "## Guard checks (hard)",
        "",
        "**PASS.**" if not failures else "**FAIL.** " + "; ".join(failures),
        "",
        "## Every turn",
    ]
    for record in run["records"]:
        lines += ["", f"### {record['id']} — {record['topic']}"]
        for i, turn in enumerate(record["results"], start=1):
            lines += ["", f"**Turn {i}:** {turn['question']}", ""]
            if "error" in turn:
                lines.append(f"Error: `{turn['error']}`")
                continue
            totals = _turn_totals(turn)
            lines.append(
                f"{turn['seconds']}s · {turn['category']}/{turn['intent']} · route "
                f"`{turn['route']}` · {totals['calls']} LLM calls, {totals['tokens']:,} "
                f"tokens · served {turn['served']}, withheld {turn['withheld']}"
                + (" · corrective retry" if turn.get("retried") else "")
            )
            if (
                turn.get("rewritten_query")
                and turn["rewritten_query"] != turn["question"]
            ):
                lines.append(f"Rewritten: {turn['rewritten_query']}")
            nodes = ", ".join(
                f"{e['node']} {e['ms'] / 1000:.1f}s" for e in turn["trace"]
            )
            lines.append(f"Nodes: {nodes}")
            calls = "; ".join(
                f"{c['model'].split('/')[-1]} {c['prompt_tokens']}+{c['completion_tokens']} "
                f"{c['seconds']}s"
                for c in turn["llm"]["calls"]
            )
            lines.append(f"Calls: {calls}")
            if turn["llm"]["retries"] or turn["llm"]["fallbacks"]:
                lines.append(
                    "Waits: "
                    + "; ".join(turn["llm"]["retries"] + turn["llm"]["fallbacks"])
                )
            if turn["clarify"]:
                lines.append("Clarify: " + " | ".join(turn["clarify"]))
            lines += ["", "```text", turn["served_text"] or "(nothing served)", "```"]
            withheld = [e for e in turn["emitted_claims"] if "verified" not in e]
            if withheld:
                lines.append(
                    "Withheld: "
                    + ", ".join(f"#{e['id']} {e['reason']}" for e in withheld)
                )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True, help="run name, e.g. baseline")
    parser.add_argument("--compare", help="label of a stored run to compare against")
    parser.add_argument("--pause", type=float, default=DEFAULT_PAUSE)
    parser.add_argument("--only", nargs="*", default=[], help="script ids to run")
    parser.add_argument("--database-url", default=LOCAL_DATABASE_URL)
    parser.add_argument("--stored", action="store_true", help="re-report a stored run")
    args = parser.parse_args()
    configure_logging()
    result_path = RESULT_DIR / f"advisor_smoke_{args.label}.json"

    if args.stored:
        run = json.loads(result_path.read_text(encoding="utf-8"))
    else:
        settings = Settings(database_url=args.database_url)
        try:
            run = run_live(settings, args.label, args.pause, set(args.only))
        except (MissingSettingError, RuntimeError) as error:
            print(f"error: {error}", file=sys.stderr)
            return 1
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(
            json.dumps(run, ensure_ascii=False, indent=1, default=str) + "\n",
            encoding="utf-8",
            newline="",
        )

    baseline = None
    if args.compare:
        baseline_path = RESULT_DIR / f"advisor_smoke_{args.compare}.json"
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(render(run, baseline), encoding="utf-8", newline="")
    failures = guard_failures(run)
    print(json.dumps(summarise(run), indent=1))
    print("guards:", "PASS" if not failures else "FAIL " + "; ".join(failures))
    print(f"wrote {result_path}")
    print(f"wrote {REPORT_PATH}")
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
