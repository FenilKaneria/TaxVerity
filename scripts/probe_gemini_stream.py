"""R18 gate 1 — the Gemini streaming spike (PLAN R18, ADR-118).

Step 7.1's Groq probe found "a newline always ends a content delta and never
lands mid-delta" and PLAN R18 registered that same property as this gate's
question for Gemini. **A first run against real Gemini traffic showed the
opposite for Gemini** (a newline lands mid-delta routinely — its deltas batch
several complete NDJSON lines together, unlike Groq's near-token-granular
ones) — but re-reading `generation/claims.py`'s `LineBuffer.feed()` shows the
"never mid-delta" property was never actually load-bearing: `feed()` does
`(self._partial + delta).split("\n")`, which is correct for *any* number of
embedded newlines in *any* delta. The real gate is whether `iter_lines()` +
`parse_claim()` extract the right claims from Gemini's stream, not where the
newlines happen to fall — so this script checks that directly, and treats the
delta-batching pattern itself only as a latency characteristic worth
recording (claims may arrive in bursts, not a smooth trickle).

Runs the real production system prompt (`generation.generate.SYSTEM_PROMPT`)
and a real evidence pack (BM25 + citation shortcut, no Jina tokens billed)
against Gemini `N_RUNS` times, feeds each run's captured deltas through the
real `iter_lines()`/`parse_claim()`, and records:
  - how many lines parsed as valid claims vs malformed lines,
  - the delta-batching shape (deltas containing a newline, mid-delta or not)
    as a latency note, not a gate,
  - whether the stream's own last delta ends with a trailing newline (Groq's
    answer was no — the parser must flush at end-of-stream regardless, and
    `iter_lines()` already does).

Every run records its own failure rather than raising: the report is the
product, same discipline as Step 7.1's `probe_groq.py`.

Writes `reports/gemini_streaming_spike.md`.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from probe_groq import SPIKE_QUERY  # noqa: E402

from taxverity.chunking.pipeline import read_corpus_version  # noqa: E402
from taxverity.chunking.store import load_chunks  # noqa: E402
from taxverity.config import MissingSettingError, Settings  # noqa: E402
from taxverity.facts import UserFacts  # noqa: E402
from taxverity.generation.claims import (  # noqa: E402
    MalformedClaim,
    iter_lines,
    parse_claim,
)
from taxverity.generation.generate import SYSTEM_PROMPT, render_context  # noqa: E402
from taxverity.llm.client import GEMINI, LLMClient, LLMError, Message  # noqa: E402
from taxverity.observability import configure_logging, get_logger  # noqa: E402
from taxverity.retrieval.bm25 import BM25Retriever  # noqa: E402
from taxverity.retrieval.citations import (  # noqa: E402
    CitationRetriever,
    ShortcutRetriever,
)
from taxverity.retrieval.evidence import EVIDENCE_POOL, EvidencePacker  # noqa: E402

logger = get_logger(__name__)

REPORT = Path("reports") / "gemini_streaming_spike.md"
N_RUNS = 2
# Gemini's hidden reasoning is billed against this cap (R18's live probe,
# even with reasoning_effort="low") and Step 7.1's 2,048 (sized for Groq)
# truncated every run here before a single NDJSON line completed.
MAX_COMPLETION_TOKENS = 4_096


@dataclass
class RunResult:
    index: int
    ok: bool
    error: str | None = None
    delta_count: int = 0
    deltas_containing_newline: int = 0
    deltas_ending_on_newline: int = 0
    newline_ever_mid_delta: bool = False
    mid_delta_examples: list[str] = field(default_factory=list)
    trailing_newline_on_last_delta: bool = False
    # The actual gate: what iter_lines()/parse_claim() extracted.
    lines_extracted: int = 0
    claims_parsed: int = 0
    malformed_lines: list[str] = field(default_factory=list)
    text: str = ""
    elapsed: float = 0.0


def probe(client: LLMClient, messages: list[Message], index: int) -> RunResult:
    started = time.perf_counter()
    deltas: list[str] = []
    try:
        stream = client.stream(messages, max_completion_tokens=MAX_COMPLETION_TOKENS)
        for delta in stream:
            deltas.append(delta)
    except LLMError as error:
        return RunResult(index=index, ok=False, error=str(error), elapsed=time.perf_counter() - started)

    mid_delta = [d for d in deltas if "\n" in d and not d.endswith("\n")]

    claims_parsed = 0
    malformed: list[str] = []
    for line in iter_lines(iter(deltas)):
        try:
            parse_claim(line)
            claims_parsed += 1
        except MalformedClaim as error:
            malformed.append(f"{line!r} ({error})")

    return RunResult(
        index=index,
        ok=True,
        delta_count=len(deltas),
        deltas_containing_newline=sum(1 for d in deltas if "\n" in d),
        deltas_ending_on_newline=sum(1 for d in deltas if d.endswith("\n")),
        newline_ever_mid_delta=bool(mid_delta),
        mid_delta_examples=[repr(d) for d in mid_delta[:5]],
        trailing_newline_on_last_delta=bool(deltas) and deltas[-1].endswith("\n"),
        lines_extracted=claims_parsed + len(malformed),
        claims_parsed=claims_parsed,
        malformed_lines=malformed,
        text="".join(deltas),
        elapsed=time.perf_counter() - started,
    )


def render(results: list[RunResult], citations: list[str]) -> str:
    lines = [
        "# R18 gate 1 — Gemini streaming spike",
        "",
        f"Model: `{GEMINI.model}`. {len(results)} runs against a real evidence pack "
        f"({len(citations)} units: {', '.join(citations)}), the real "
        "`generation.generate.SYSTEM_PROMPT`.",
        "",
        "## Verdict",
        "",
    ]
    ok_runs = [r for r in results if r.ok]
    any_malformed = any(r.malformed_lines for r in ok_runs)
    any_mid_delta = any(r.newline_ever_mid_delta for r in ok_runs)
    if not ok_runs:
        lines.append("**Inconclusive — every run failed before streaming any text.**")
    elif any_malformed:
        lines.append(
            "**FAIL — `iter_lines()`/`parse_claim()` produced a malformed line at "
            "least once.** See Malformed lines below. R18 cannot move "
            "`generate_verify` to Gemini until this is understood."
        )
    else:
        lines.append(
            f"**PASS on {len(ok_runs)}/{len(results)} completed runs** — every line "
            "`iter_lines()` extracted from Gemini's stream parsed as a valid claim, "
            "regardless of where newlines fell across delta boundaries "
            f"({'some deltas batched multiple newlines' if any_mid_delta else 'no delta ever batched a newline mid-content'}"
            ", `LineBuffer.feed()` handles both). Gate 1 clears on correctness; "
            "gates 2 and 3 still gate the routing decision, and the run-failure "
            "rate below is a separate availability concern worth carrying forward."
        )
    lines += ["", "## Per-run detail", ""]
    lines += [
        "| run | ok | deltas | contain \\n | mid-delta \\n | claims parsed | malformed | elapsed |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        if not r.ok:
            lines.append(f"| {r.index} | FAIL | - | - | - | - | - | {r.elapsed:.1f}s ({r.error}) |")
            continue
        lines.append(
            f"| {r.index} | ok | {r.delta_count} | {r.deltas_containing_newline} | "
            f"{r.newline_ever_mid_delta} | {r.claims_parsed} | {len(r.malformed_lines)} | "
            f"{r.elapsed:.1f}s |"
        )
    if any_malformed:
        lines += ["", "## Malformed lines", ""]
        for r in ok_runs:
            for line in r.malformed_lines:
                lines.append(f"- run {r.index}: `{line}`")
    if any(r.mid_delta_examples for r in ok_runs):
        lines += ["", "## Delta-batching examples (latency note, not a failure)", ""]
        for r in ok_runs:
            for example in r.mid_delta_examples:
                lines.append(f"- run {r.index}: `{example}`")
    fail_count = sum(1 for r in results if not r.ok)
    if fail_count:
        lines += [
            "",
            "## Availability note",
            "",
            f"{fail_count}/{len(results)} runs failed before their first token "
            "(503/429/timeout, all logged). This is a Gemini free-tier "
            "availability characteristic, separate from the correctness gate "
            "above — carry it into gate 2 (the two-arm benchmark) and gate 3.",
        ]
    lines += ["", "## One run's raw text (sanity check)", ""]
    if ok_runs:
        lines.append("```")
        lines.append(ok_runs[0].text[:2000])
        lines.append("```")
    return "\n".join(lines) + "\n"


def main() -> int:
    configure_logging()
    settings = Settings()
    try:
        key = settings.require("gemini_api_key")
    except MissingSettingError as error:
        print(f"cannot run: {error}")
        return 1

    logger.info("building an evidence pack for the streaming probe")
    corpus_version = read_corpus_version(settings.interim_dir / "corpus_manifest.json")
    chunks, _ = load_chunks(settings.interim_dir, corpus_version=corpus_version)
    retriever = ShortcutRetriever(CitationRetriever(chunks), BM25Retriever(chunks))
    packer = EvidencePacker(chunks)
    pack = packer.pack(retriever.search(SPIKE_QUERY, EVIDENCE_POOL))

    messages = [
        Message(role="system", content=SYSTEM_PROMPT),
        Message(
            role="user",
            content=render_context(SPIKE_QUERY, pack, UserFacts(facts=(), unmapped=()), None),
        ),
    ]

    client = LLMClient(GEMINI, key)
    logger.info("running %d streaming probes against %s", N_RUNS, GEMINI.model)
    results = [probe(client, messages, i) for i in range(1, N_RUNS + 1)]
    client.close()

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(render(results, [u.citation for u in pack.units]), encoding="utf-8", newline="\n")
    ok = sum(1 for r in results if r.ok)
    mid = sum(1 for r in results if r.ok and r.newline_ever_mid_delta)
    print(f"{ok}/{len(results)} runs completed, {mid} showed a mid-delta newline")
    print(f"report: {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
