"""Step 7.1 — provider spike: what Groq's `openai/gpt-oss-120b` actually does.

A spike, like Step 1.1: no `src/` code and no tests. Steps 7.2, 7.5, 7.6, 9.x
and 10.3/10.4 are all specified against assumptions about this endpoint that
nobody has checked, and ADR-082's evidence budget rests on two of them — an
8,000 tokens-a-minute free-tier limit, and a proxy token count calibrated
against Jina's tokenizer rather than gpt-oss's.

Every probe records its own failure rather than raising: the report is the
product, and a provider that refuses something is a finding, not a crash.

Writes `reports/groq_spike.md`.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import httpx2

from taxverity.chunking.pipeline import read_corpus_version
from taxverity.chunking.store import load_chunks
from taxverity.config import Settings
from taxverity.observability import configure_logging, get_logger
from taxverity.retrieval.bm25 import BM25Retriever
from taxverity.retrieval.citations import CitationRetriever, ShortcutRetriever
from taxverity.retrieval.evidence import EVIDENCE_POOL, EvidencePacker

logger = get_logger(__name__)

API = "https://api.groq.com/openai/v1"
MODEL = "openai/gpt-oss-120b"
REPORT = Path("reports") / "groq_spike.md"
TIMEOUT = 120.0

# A question with a real answer in the Act, used for the evidence-pack probes.
SPIKE_QUERY = "How much standard deduction can I claim on my rented-out flat?"

RATE_HEADERS = (
    "x-ratelimit-limit-requests",
    "x-ratelimit-remaining-requests",
    "x-ratelimit-reset-requests",
    "x-ratelimit-limit-tokens",
    "x-ratelimit-remaining-tokens",
    "x-ratelimit-reset-tokens",
    "retry-after",
)


class Probe:
    """One question put to the provider, with whatever it answered."""

    def __init__(self, name: str, question: str) -> None:
        self.name = name
        self.question = question
        self.findings: dict[str, Any] = {}
        self.error: str | None = None

    def record(self, **findings: Any) -> None:
        self.findings.update(findings)


def rate_limits(response: httpx2.Response, *, pace: bool = True) -> dict[str, str]:
    limits = {h: response.headers[h] for h in RATE_HEADERS if h in response.headers}
    if pace:
        _pace(limits)
    return limits


def _parse_reset(value: str) -> float:
    """Groq states a reset as `46.335s`, `1m26.4s` or `832ms`."""
    seconds, rest = 0.0, value
    if "m" in rest and "ms" not in rest:
        minutes, rest = rest.split("m", 1)
        seconds += float(minutes) * 60
    if rest.endswith("ms"):
        return seconds + float(rest[:-2]) / 1000
    return seconds + float(rest.rstrip("s") or 0)


def _pace(limits: dict[str, str]) -> None:
    """Wait out the token window rather than spending the spike's findings on a
    429. The stream probe alone costs ~4,800 of the 8,000-a-minute allowance.
    """
    remaining = limits.get("x-ratelimit-remaining-tokens")
    reset = limits.get("x-ratelimit-reset-tokens")
    if remaining is None or reset is None or int(remaining) >= 6_000:
        return
    wait = min(_parse_reset(reset) + 1.0, 70.0)
    logger.info("%s tokens left this minute; waiting %.1fs", remaining, wait)
    time.sleep(wait)


def chat(
    client: httpx2.Client, body: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, str], float]:
    started = time.perf_counter()
    response = client.post("/chat/completions", json={"model": MODEL, **body})
    elapsed = time.perf_counter() - started
    limits = rate_limits(response)
    if response.status_code != 200:
        raise RuntimeError(f"HTTP {response.status_code}: {response.text[:400]}")
    return response.json(), limits, elapsed


def usage_of(payload: dict[str, Any]) -> dict[str, Any]:
    usage = payload.get("usage") or {}
    keep = (
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "completion_time",
        "queue_time",
    )
    found = {k: usage[k] for k in keep if k in usage}
    details = usage.get("completion_tokens_details") or {}
    if "reasoning_tokens" in details:
        found["reasoning_tokens"] = details["reasoning_tokens"]
    return found


def _parses(text: Any) -> Any:
    try:
        return json.loads(text)
    except (ValueError, TypeError):
        return None


def probe_models(client: httpx2.Client) -> Probe:
    probe = Probe("models", "Is the model served, and what are its stated limits?")
    try:
        response = client.get("/models")
        response.raise_for_status()
        served = {m["id"]: m for m in response.json()["data"]}
        entry = served.get(MODEL)
        probe.record(
            present=entry is not None,
            context_window=entry and entry.get("context_window"),
            max_completion_tokens=entry and entry.get("max_completion_tokens"),
            owned_by=entry and entry.get("owned_by"),
            fallback_candidates=sorted(
                i for i in served if "gpt-oss" in i or "llama" in i
            )[:8],
        )
    except Exception as exc:
        probe.error = f"{type(exc).__name__}: {exc}"
    return probe


def probe_plain(client: httpx2.Client) -> Probe:
    probe = Probe("plain", "Latency, usage and finish_reason on an ordinary call.")
    try:
        payload, limits, elapsed = chat(
            client,
            {
                "messages": [
                    {"role": "user", "content": "Reply with the single word: ready."}
                ],
                "max_completion_tokens": 32,
            },
        )
        choice = payload["choices"][0]
        probe.record(
            seconds=round(elapsed, 3),
            finish_reason=choice["finish_reason"],
            content=(choice["message"].get("content") or "")[:120],
            reasoning_field_present="reasoning" in choice["message"],
            usage=usage_of(payload),
            rate_limits=limits,
        )
    except Exception as exc:
        probe.error = f"{type(exc).__name__}: {exc}"
    return probe


def probe_reasoning(client: httpx2.Client) -> Probe:
    probe = Probe(
        "reasoning_effort", "What does reasoning_effort cost on the hot path?"
    )
    question = "A resident individual has salary income of 12,00,000. Name the two tax regimes."
    for effort in ("low", "medium", "high"):
        try:
            payload, _, elapsed = chat(
                client,
                {
                    "messages": [{"role": "user", "content": question}],
                    "reasoning_effort": effort,
                    "max_completion_tokens": 400,
                },
            )
            probe.findings[effort] = {
                "seconds": round(elapsed, 3),
                "usage": usage_of(payload),
                "finish_reason": payload["choices"][0]["finish_reason"],
            }
        except Exception as exc:
            probe.findings[effort] = {"error": f"{type(exc).__name__}: {exc}"}
    return probe


def probe_json_object(client: httpx2.Client) -> Probe:
    probe = Probe("json_object", "Does the loose JSON mode work, as a 7.6 fallback?")
    try:
        payload, _, elapsed = chat(
            client,
            {
                "messages": [
                    {
                        "role": "system",
                        "content": "Reply in JSON with keys `salary` and `age`.",
                    },
                    {"role": "user", "content": "I earn 14,00,000 a year and I am 41."},
                ],
                "response_format": {"type": "json_object"},
                "max_completion_tokens": 200,
            },
        )
        content = payload["choices"][0]["message"].get("content") or ""
        probe.record(
            seconds=round(elapsed, 3),
            raw=content[:300],
            parses=_parses(content),
            usage=usage_of(payload),
        )
    except Exception as exc:
        probe.error = f"{type(exc).__name__}: {exc}"
    return probe


FACTS_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["fields"],
    "properties": {
        "fields": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["name", "value", "status", "source_span"],
                "properties": {
                    "name": {"type": "string"},
                    "value": {"type": "string"},
                    "status": {
                        "type": "string",
                        "enum": ["stated", "inferred", "missing"],
                    },
                    "source_span": {"type": "string"},
                },
            },
        }
    },
}

FACTS_TURN = "I'm 41, salaried, earned 14,00,000 last year and paid 1,50,000 into PPF."


def probe_json_schema(client: httpx2.Client) -> Probe:
    probe = Probe(
        "json_schema", "Does strict schema mode hold the Step 7.5 UserFacts shape?"
    )
    try:
        payload, _, elapsed = chat(
            client,
            {
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "Extract tax-relevant facts. `source_span` must be copied verbatim "
                            "from the user's text. Use status `inferred` when you did not read "
                            "it directly."
                        ),
                    },
                    {"role": "user", "content": FACTS_TURN},
                ],
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "user_facts",
                        "strict": True,
                        "schema": FACTS_SCHEMA,
                    },
                },
                "max_completion_tokens": 800,
            },
        )
        content = payload["choices"][0]["message"].get("content") or ""
        parsed = _parses(content)
        spans_verbatim = None
        if isinstance(parsed, dict):
            spans = [f.get("source_span", "") for f in parsed.get("fields", [])]
            spans_verbatim = {s: (s in FACTS_TURN) for s in spans}
        probe.record(
            seconds=round(elapsed, 3),
            raw=content[:600],
            parsed=parsed if isinstance(parsed, dict) else None,
            source_spans_verbatim=spans_verbatim,
            usage=usage_of(payload),
        )
    except Exception as exc:
        probe.error = f"{type(exc).__name__}: {exc}"
    return probe


TAX_TOOL = {
    "type": "function",
    "function": {
        "name": "compute_tax",
        "description": "Compute tax under one regime. The only source of arithmetic.",
        "parameters": {
            "type": "object",
            "additionalProperties": False,
            "required": ["total_income", "regime"],
            "properties": {
                "total_income": {"type": "number"},
                "regime": {"type": "string", "enum": ["old", "new"]},
            },
        },
    },
}


def probe_tools(client: httpx2.Client) -> Probe:
    probe = Probe(
        "tools", "Tool-call shape, and whether tool_choice=required is honoured."
    )
    # Twice each: the first spike run saw `required` fail with a 400
    # `tool_use_failed` and the second saw it succeed, so a single attempt
    # cannot tell "unsupported" from "intermittent".
    for attempt in (1, 2):
        for base in ("auto", "required"):
            choice = f"{base}#{attempt}"
            try:
                payload, _, elapsed = chat(
                    client,
                    {
                        "messages": [
                            {
                                "role": "user",
                                "content": "My total income is 14,00,000. New regime tax?",
                            }
                        ],
                        "tools": [TAX_TOOL],
                        "tool_choice": base,
                        "max_completion_tokens": 400,
                    },
                )
                message = payload["choices"][0]["message"]
                calls = message.get("tool_calls") or []
                probe.findings[choice] = {
                    "seconds": round(elapsed, 3),
                    "finish_reason": payload["choices"][0]["finish_reason"],
                    "call_count": len(calls),
                    "names": [c["function"]["name"] for c in calls],
                    "arguments": [_parses(c["function"]["arguments"]) for c in calls],
                    "content_alongside": (message.get("content") or "")[:160],
                    "usage": usage_of(payload),
                }
            except Exception as exc:
                probe.findings[choice] = {"error": f"{type(exc).__name__}: {exc}"}
    return probe


CLAIM_SYSTEM = (
    "You answer questions about the Income-tax Act, 2025 using ONLY the evidence below.\n"
    "Emit NDJSON: one JSON object per line, no prose, no markdown fence.\n"
    'Each line: {"text": "<one sentence>", "citations": ["<citation exactly as labelled>"]}\n'
    "Cite only citations that appear in the evidence."
)


def build_pack(settings: Settings) -> tuple[str, int, list[str]]:
    """A real evidence pack for the streaming and calibration probes.

    BM25 plus the citation shortcut, so this bills nothing on the Jina side and
    still produces a genuine prompt rather than a toy.
    """
    corpus_version = read_corpus_version(settings.interim_dir / "corpus_manifest.json")
    chunks, _ = load_chunks(settings.interim_dir, corpus_version=corpus_version)
    retriever = ShortcutRetriever(CitationRetriever(chunks), BM25Retriever(chunks))
    packer = EvidencePacker(chunks)
    pack = packer.pack(retriever.search(SPIKE_QUERY, EVIDENCE_POOL))
    blocks = []
    for unit in pack.units:
        lead = "\n".join(line.text for line in unit.context)
        blocks.append(f"[{unit.citation}]\n{lead}\n{unit.chunk.text}".strip())
    return "\n\n".join(blocks), pack.tokens, [u.citation for u in pack.units]


def probe_stream(client: httpx2.Client, evidence: str, proxy_tokens: int) -> Probe:
    probe = Probe(
        "stream_ndjson", "Does claim-per-line streaming behave as Step 10.4 assumes?"
    )
    body = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": CLAIM_SYSTEM},
            {
                "role": "user",
                "content": f"EVIDENCE:\n{evidence}\n\nQUESTION: {SPIKE_QUERY}",
            },
        ],
        "stream": True,
        "stream_options": {"include_usage": True},
        "max_completion_tokens": 700,
    }
    try:
        started = time.perf_counter()
        first_token: float | None = None
        deltas: list[str] = []
        usage: dict[str, Any] = {}
        limits: dict[str, str] = {}
        with client.stream("POST", "/chat/completions", json=body) as response:
            # Pacing here would be counted as time-to-first-token.
            limits = rate_limits(response, pace=False)
            if response.status_code != 200:
                raise RuntimeError(
                    f"HTTP {response.status_code}: {response.read()[:400]!r}"
                )
            for line in response.iter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                event = json.loads(data)
                if event.get("usage"):
                    usage = event["usage"]
                for choice in event.get("choices", []):
                    piece = (choice.get("delta") or {}).get("content")
                    if piece:
                        if first_token is None:
                            first_token = time.perf_counter() - started
                        deltas.append(piece)
        text = "".join(deltas)
        lines = [ln for ln in text.splitlines() if ln.strip()]
        billed = usage.get("prompt_tokens")
        probe.record(
            ttft_seconds=round(first_token, 3) if first_token else None,
            total_seconds=round(time.perf_counter() - started, 3),
            delta_count=len(deltas),
            deltas_ending_on_newline=sum(1 for d in deltas if d.endswith("\n")),
            deltas_containing_newline=sum(1 for d in deltas if "\n" in d),
            newline_ever_mid_delta=any(
                "\n" in d and not d.endswith("\n") for d in deltas
            ),
            line_count=len(lines),
            lines_parsing_as_json=sum(
                1 for ln in lines if isinstance(_parses(ln), dict)
            ),
            first_line=lines[0][:200] if lines else None,
            proxy_prompt_tokens=proxy_tokens,
            billed_prompt_tokens=billed,
            proxy_ratio=round(billed / proxy_tokens, 4)
            if billed and proxy_tokens
            else None,
            usage=usage_of({"usage": usage}),
            rate_limits=limits,
        )
        _pace(limits)
    except Exception as exc:
        probe.error = f"{type(exc).__name__}: {exc}"
    return probe


MINI_EVIDENCE = """[21(1)]
The annual value of a house property shall be determined under this section.

[22(1)(a)]
thirty per cent of the annual value as determined under section 21;

[22(1)(b)]
where the property is acquired with borrowed capital, the interest payable on
such capital."""


def probe_multiclaim(client: httpx2.Client) -> Probe:
    """The Step 10.4 question the first stream probe could not answer.

    That one produced a single claim line, so no newline was ever emitted and
    the parser contract went untested. A small hand-written pack is enough —
    what is being measured is delta framing, not retrieval.
    """
    probe = Probe(
        "stream_multiclaim", "Where do newlines fall relative to stream deltas?"
    )
    body = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": CLAIM_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"EVIDENCE:\n{MINI_EVIDENCE}\n\nQUESTION: What deductions are "
                    "allowed against income from house property? Emit one line per "
                    "deduction, at least three lines."
                ),
            },
        ],
        "stream": True,
        "reasoning_effort": "low",
        "max_completion_tokens": 700,
    }
    try:
        deltas: list[str] = []
        with client.stream("POST", "/chat/completions", json=body) as response:
            if response.status_code != 200:
                raise RuntimeError(
                    f"HTTP {response.status_code}: {response.read()[:400]!r}"
                )
            for line in response.iter_lines():
                if not line.startswith("data: ") or line[6:] == "[DONE]":
                    continue
                for choice in json.loads(line[6:]).get("choices", []):
                    piece = (choice.get("delta") or {}).get("content")
                    if piece:
                        deltas.append(piece)
        text = "".join(deltas)
        lines = [ln for ln in text.splitlines() if ln.strip()]
        newline_deltas = [d for d in deltas if "\n" in d]
        probe.record(
            delta_count=len(deltas),
            line_count=len(lines),
            lines_parsing_as_json=sum(
                1 for ln in lines if isinstance(_parses(ln), dict)
            ),
            newline_delta_count=len(newline_deltas),
            newline_deltas=[repr(d) for d in newline_deltas[:8]],
            newline_ever_mid_delta=any(
                "\n" in d and not d.endswith("\n") for d in deltas
            ),
            delta_is_bare_newline=sum(1 for d in newline_deltas if d == "\n"),
            trailing_newline_on_last_line=text.endswith("\n"),
            first_line=lines[0][:160] if lines else None,
        )
    except Exception as exc:
        probe.error = f"{type(exc).__name__}: {exc}"
    return probe


def probe_reasoning_off(client: httpx2.Client) -> Probe:
    """Reasoning tokens are billed against max_completion_tokens — the `plain`
    probe returned empty content at finish_reason=length because 30 of its 32
    allowed tokens went to reasoning. Whether reasoning can be turned off is
    therefore a hot-path cost question, not a tuning nicety.
    """
    probe = Probe("reasoning_off", "Can reasoning be disabled, and what does it save?")
    question = "Reply with exactly: ready."
    for label, extra in (
        ("effort_none", {"reasoning_effort": "none"}),
        ("format_hidden", {"reasoning_format": "hidden", "reasoning_effort": "low"}),
        ("no_cap_low_effort", {"reasoning_effort": "low"}),
    ):
        try:
            payload, _, elapsed = chat(
                client,
                {
                    "messages": [{"role": "user", "content": question}],
                    "max_completion_tokens": 64,
                    **extra,
                },
            )
            choice = payload["choices"][0]
            probe.findings[label] = {
                "seconds": round(elapsed, 3),
                "finish_reason": choice["finish_reason"],
                "content": (choice["message"].get("content") or "")[:80],
                "usage": usage_of(payload),
            }
        except Exception as exc:
            probe.findings[label] = {"error": f"{type(exc).__name__}: {exc}"}
    return probe


def probe_stream_with_schema(client: httpx2.Client) -> Probe:
    probe = Probe("stream_plus_json_schema", "Do streaming and strict schema compose?")
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": FACTS_TURN}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "user_facts",
                "strict": True,
                "schema": FACTS_SCHEMA,
            },
        },
        "stream": True,
        "max_completion_tokens": 400,
    }
    try:
        with client.stream("POST", "/chat/completions", json=body) as response:
            status = response.status_code
            body_text = response.read().decode("utf-8", "replace")
        probe.record(
            status=status,
            accepted=status == 200,
            detail=body_text[:400] if status != 200 else "streamed without error",
        )
    except Exception as exc:
        probe.error = f"{type(exc).__name__}: {exc}"
    return probe


FINDINGS = """## Findings

Interpretation of the run below. Each claim names the probe it rests on.

1. **The free tier is 8,000 tokens a minute, and ADR-082's assumption was
   right** (`plain.rate_limits`). Requests are capped at 1,000 with a rolling
   reset that lengthens as they are spent — 12m, 27m, 49m across three runs of
   this script — so requests are not the binding constraint; tokens are.
2. **One evidence-pack query costs 4,641 total tokens** (`stream_ndjson`), so
   the free tier serves **about 1.7 queries a minute**, single-user. That is a
   real capacity ceiling on the served product, not a dev-loop nuisance, and
   Step 7.2 has to decide whether it is answered by paid tier, by a smaller
   `EVIDENCE_BUDGET`, or by accepting a queue.
3. **The Step 2.3 proxy counter runs 9.85% short of gpt-oss's tokenizer**
   (`stream_ndjson.proxy_ratio = 1.0985`, stable across runs on an identical
   prompt), against 14.5% short of Jina's. So `EVIDENCE_BUDGET = 4,000` proxy
   tokens is ~4,394 real ones — closer than ADR-082 assumed, and its stated
   "about 4,600" was pessimistic in the safe direction.
4. **Strict `json_schema` holds the Step 7.5 `UserFacts` shape, and copies
   `source_span` verbatim** (`json_schema`): 4 of 4 spans were exact
   substrings of the user's turn. The looser `json_object` mode also works
   (`json_object`), so 7.6 has a fallback. Value formatting is *not* stable —
   one run returned `"1400000"` and another `"14,00,000"` for the same input,
   so 7.5's schema must carry the normalisation, not the prompt.
5. **`tool_choice: "required"` is unreliable: HTTP 400 `tool_use_failed` on
   3 of 6 attempts across four runs of this script**, while `auto` called the
   tool correctly on 6 of 6 with identical arguments (`tools`). The failures
   and successes came in whole runs, not alternating within one, and the 400
   carries an empty `failed_generation`, so there is nothing to repair. The
   report below shows one such run; `required#1`/`required#2` carried the 400
   on two earlier ones. Phase 9 must not depend on forced tool calls.
6. **Reasoning cannot be switched off, and its tokens are billed against
   `max_completion_tokens`** (`reasoning_off`): `reasoning_effort: "none"` is
   rejected outright, and the `plain` probe spent 22 of its 32 allowed tokens
   reasoning. A cap sized for the visible answer alone truncates it. `low` is
   the cheapest setting that exists and is what the hot path should use —
   `high` burned 398 reasoning tokens on a trivial question.
7. **Streaming and strict schema compose** (`stream_plus_json_schema`), and
   streaming NDJSON works, with two caveats for Step 10.4
   (`stream_multiclaim`): a newline always *ends* a content delta and never
   lands mid-delta (6 observations over 3 runs — too few to build on, so the
   parser must still buffer), and **the last line carries no trailing
   newline**, so the parser has to flush at end-of-stream or silently drop the
   final claim.
8. **Time to first token is 1.7 s on a 4,372-token prompt** (`stream_ndjson`),
   with queue time 0.3-0.6 s throughout. The dead-air problem Rule 04 names is
   real but small at this size.
9. Context window 131,072 and max completion 65,536 (`models`) — neither is
   anywhere near binding. `openai/gpt-oss-20b` is served on the same key, which
   is a same-vendor degradation option the plan does not currently use.

"""


def render(probes: list[Probe], pack_citations: list[str]) -> str:
    out = [
        "# Step 7.1 — Groq provider spike",
        "",
        f"Model: `{MODEL}`. Endpoint: `{API}`.",
        "",
        "A spike: no `src/` code, no tests. Findings shape Steps 7.2, 7.5, 7.6,",
        "9.x and 10.3/10.4, and re-check ADR-082's evidence budget.",
        "",
        f"Evidence pack used for the streaming probes: {len(pack_citations)} units — "
        + ", ".join(f"`{c}`" for c in pack_citations),
        "",
        FINDINGS,
    ]
    for probe in probes:
        out += [f"## {probe.name}", "", f"*{probe.question}*", ""]
        if probe.error:
            out += [f"**FAILED** — `{probe.error}`", ""]
        out += [
            "```json",
            json.dumps(probe.findings, indent=2, ensure_ascii=False),
            "```",
            "",
        ]
    return "\n".join(out)


def main() -> None:
    configure_logging()
    settings = Settings()
    key = settings.require("groq_api_key")

    logger.info("building an evidence pack for the streaming probes")
    evidence, proxy_tokens, citations = build_pack(settings)
    logger.info(
        "evidence pack built: %d units, %d proxy tokens", len(citations), proxy_tokens
    )

    with httpx2.Client(
        base_url=API,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        timeout=TIMEOUT,
    ) as client:
        probes = [
            probe_models(client),
            probe_plain(client),
            probe_reasoning(client),
            probe_json_object(client),
            probe_json_schema(client),
            probe_tools(client),
            probe_stream(client, evidence, proxy_tokens),
            probe_multiclaim(client),
            probe_reasoning_off(client),
            probe_stream_with_schema(client),
        ]

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(render(probes, citations), encoding="utf-8", newline="\n")
    for probe in probes:
        status = "FAILED" if probe.error else "ok"
        print(f"{probe.name:<24} {status}")
    print(f"\nwrote {REPORT}")


if __name__ == "__main__":
    main()
