"""Step 13.8 — the 10.7 smoke set, driven through the graph's own retrieval
and generation nodes (`retrieve`, `generate_verify`, and Step 13.5's
`retrieve_retry`) rather than calling the generator directly.

SIMPLIFIED per ADR-110: no new eval framework. This reuses `answer_smoke.py`'s
production retrieval composition and gold loading unchanged, and reuses the
Step 7.3 LLM cache, so re-running this after `answer_smoke.py` bills nothing
for a question its first pass already answered — new cost is limited to
questions whose first pass served zero statute claims and that therefore
exercise Step 13.5's retry with a wider, previously-unrequested pack.

Classification, extraction, the calculator and the DB-backed nodes
(`load_thread`, `merge_facts`, `finalize`) are deliberately not exercised
here: these are pure statutory questions with no facts, matching 10.7's own
design, and those nodes already have dedicated coverage (Steps 13.2, 13.6) on
fakes. What this script adds beyond 10.7 is proof that the graph's own
`retrieve -> generate_verify -> retrieve_retry -> generate_verify` wiring
behaves the same way over real production retrieval and a real (cached) LLM —
in particular, that the corrective loop's guard holds here too: no negative
question is ever rescued into a served claim.

Writes `data/answers/graph_smoke_v1.json` and `reports/graph_smoke.md`.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from answer_smoke import DEFAULT_PAUSE, build_retrieval  # noqa: E402

from taxverity.config import MissingSettingError, Settings
from taxverity.evals.answers import SMOKE_QUERY_IDS
from taxverity.evals.gold import GOLD_V2_FILENAME, QuerySlice, load_gold_set
from taxverity.generation.claims import ClaimEvent, WithheldEvent
from taxverity.generation.generate import AnswerGenerator
from taxverity.graph import nodes
from taxverity.graph.state import GraphDeps
from taxverity.llm.cache import CachedLLMClient
from taxverity.llm.client import LLMClient, LLMError
from taxverity.llm.tracing import LangfuseTracer, TracedLLMClient
from taxverity.memory.fact_state import ThreadFactState
from taxverity.observability import configure_logging, get_logger
from taxverity.retrieval.evidence import EvidencePacker
from taxverity.safety.evidence_gate import served_statute_claims

logger = get_logger(__name__)

RESULT_PATH = Path("data") / "answers" / "graph_smoke_v1.json"
REPORT_PATH = Path("reports") / "graph_smoke.md"
GRAPH_SMOKE_EVAL_VERSION = 1


def _noop(_event: object) -> None:
    pass


def run_query(deps: GraphDeps, question: str) -> dict[str, Any]:
    state: dict[str, Any] = {
        "query": question,
        "fact_state": ThreadFactState(),
        "computation": None,
    }
    state.update(nodes.retrieve(state, deps, writer=_noop))
    first_pack = tuple(unit.citation for unit in state["pack"].units)
    error: str | None = None
    try:
        state.update(nodes.generate_verify(state, deps, writer=_noop))
    except LLMError as failure:
        error = str(failure)
        logger.warning("first pass: provider failed: %s", failure)
        state["events"] = []

    retried = False
    if error is None and served_statute_claims(state["events"]) == 0:
        state.update(nodes.retrieve_retry(state, deps, writer=_noop))
        retried = True
        try:
            state.update(nodes.generate_verify(state, deps, writer=_noop))
        except LLMError as failure:
            error = str(failure)
            logger.warning("retry pass: provider failed: %s", failure)

    events = state.get("events", [])
    return {
        "first_pass_pack": first_pack,
        "retried": retried,
        "final_pack": tuple(unit.citation for unit in state["pack"].units),
        "served": served_statute_claims(events),
        "withheld": sum(1 for e in events if isinstance(e, WithheldEvent)),
        "citations": tuple(
            citation.path
            for event in events
            if isinstance(event, ClaimEvent)
            for citation in event.citations
        ),
        "error": error,
    }


def run_live(settings: Settings, pause: float) -> dict[str, Any]:
    gold = {
        q.query_id: q
        for q in load_gold_set(settings.evals_dir / "datasets" / GOLD_V2_FILENAME)
    }
    questions = [gold[i].question for i in SMOKE_QUERY_IDS]
    chunks, retriever = build_retrieval(settings, questions)
    packer = EvidencePacker(chunks)
    base = LLMClient.from_settings(settings)
    llm = TracedLLMClient(
        CachedLLMClient(base, settings.llm_cache_dir),
        LangfuseTracer.from_settings(settings),
    )
    generator = AnswerGenerator(llm, {chunk.node_path: chunk for chunk in chunks})
    deps = GraphDeps(
        conn=None,  # type: ignore[arg-type]
        chunks={chunk.node_path: chunk for chunk in chunks},
        retriever=retriever,
        packer=packer,
        classifier=None,  # type: ignore[arg-type]
        contextualizer=None,  # type: ignore[arg-type]
        extractor=None,  # type: ignore[arg-type]
        generator=generator,
    )

    records = []
    for index, query_id in enumerate(SMOKE_QUERY_IDS, start=1):
        query = gold[query_id]
        before = sum(base.tokens_used.values())
        result = run_query(deps, query.question)
        spent = sum(base.tokens_used.values()) - before
        records.append({"query_id": query_id, "slice": query.slice.value, **result})
        logger.info(
            "%s (%d/%d): retried=%s served=%d withheld=%d %d tokens",
            query_id, index, len(SMOKE_QUERY_IDS),
            result["retried"], result["served"], result["withheld"], spent,
        )  # fmt: skip
        if spent and index < len(SMOKE_QUERY_IDS):
            time.sleep(pause)
    return {
        "eval_version": GRAPH_SMOKE_EVAL_VERSION,
        "model": base.primary.model,
        "tokens": sum(base.tokens_used.values()),
        "records": records,
    }


def render(run: dict[str, Any]) -> str:
    served_on_negative = [
        r["query_id"] for r in run["records"] if r["slice"] == QuerySlice.NEGATIVE.value and r["served"]
    ]
    retried = [r["query_id"] for r in run["records"] if r["retried"]]
    rescued = [r["query_id"] for r in run["records"] if r["retried"] and r["served"]]
    gate = "PASS" if not served_on_negative else "FAIL"
    lines = [
        "# Graph smoke set — Step 13.8",
        "",
        "The 10.7 smoke set, driven through `retrieve` -> `generate_verify` ->",
        "`retrieve_retry` -> `generate_verify` (Step 13.5's corrective loop),",
        "not through the generator directly. No facts, no computation, no",
        "classification, no DB — see `scripts/graph_smoke.py`'s own docstring.",
        "",
        "| Measure | Value |",
        "|---|---|",
        f"| Model | `{run['model']}` |",
        f"| Questions | {len(run['records'])} |",
        f"| Retried (13.5 fired) | {len(retried)}: {', '.join(retried) or 'none'} |",
        f"| Rescued by the retry | {len(rescued)}: {', '.join(rescued) or 'none'} |",
        f"| Groq tokens this run | {run['tokens']:,} |",
        "",
        "## Guard: no negative question is ever rescued into a served claim",
        "",
        f"**{gate}.** Served on a negative: {', '.join(served_on_negative) or 'none'}.",
        "",
        "## Every question",
        "",
    ]
    for record in run["records"]:
        lines.append(
            f"- **{record['query_id']}** ({record['slice']}): first pass "
            f"{', '.join(record['first_pass_pack']) or 'nothing'}; "
            f"{'retried, final pass ' + (', '.join(record['final_pack']) or 'nothing') if record['retried'] else 'not retried'}; "
            f"served {record['served']}, withheld {record['withheld']}"
            f"{'; error: ' + record['error'] if record['error'] else ''}."
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pause", type=float, default=DEFAULT_PAUSE)
    parser.add_argument("--stored", action="store_true", help="re-report the stored run")
    args = parser.parse_args()
    configure_logging()
    settings = Settings()

    if args.stored:
        run = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    else:
        try:
            run = run_live(settings, args.pause)
        except (MissingSettingError, FileNotFoundError, RuntimeError) as error:
            print(f"error: {error}", file=sys.stderr)
            return 1
        RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
        RESULT_PATH.write_text(
            json.dumps(run, sort_keys=True, ensure_ascii=False, indent=1) + "\n",
            encoding="utf-8",
            newline="",
        )

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(render(run), encoding="utf-8", newline="")
    served_on_negative = [
        r["query_id"] for r in run["records"] if r["slice"] == QuerySlice.NEGATIVE.value and r["served"]
    ]
    print(f"served on negative: {', '.join(served_on_negative) or 'none'}")
    print(f"tokens {run['tokens']:,}")
    print(f"wrote {RESULT_PATH}")
    print(f"wrote {REPORT_PATH}")
    return 0 if not served_on_negative else 2


if __name__ == "__main__":
    raise SystemExit(main())
