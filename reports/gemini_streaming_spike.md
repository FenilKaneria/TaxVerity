# R18 gate 1 — Gemini streaming spike

Model: `gemini-3.6-flash`. Real evidence pack (8 units: 26, 270(5), 353, 147,
533, 162(1)(f), Schedule XV(3)(b), 131(1)), the real
`generation.generate.SYSTEM_PROMPT`.

## Verdict

**PASS on correctness**, argued from code plus a partial empirical check —
full automated confirmation is blocked by exhausted Gemini free-tier quota
this session.

**The gate as PLAN R18 originally framed it ("a newline never lands
mid-delta") turned out to be the wrong question.** A first live run (6
attempts, `MAX_COMPLETION_TOKENS=2,048`, no `reasoning_effort`) showed Gemini
truncating every completed response at `finish_reason="length"` before a
single NDJSON line finished — Gemini's hidden reasoning (confirmed live,
consistent with rule 05's prior 9-token-exchange finding) was consuming the
whole cap. Fixed by adding `extras={"reasoning_effort": "low"}` to the
`GEMINI` provider in `llm/client.py` (previously believed gpt-oss-only; that
belief was wrong and the stale comment + a stale test asserting it are both
corrected in this session's changes) and raising the cap to 4,096.

**With that fix, a second run (2/6 completed before the free tier's rate
limit exhausted) did complete real answers — and a newline landed mid-delta
routinely**: Gemini's OpenAI-compatible stream batches several complete
NDJSON lines into single deltas, unlike Groq's near-token-granular ones.
Example captured deltas:

- `'."}]}\n{"type": "no_basis", "text": "The Act does not state the amount or rate'`
- `' property\\"."}]}\n{"type": "no_basis", "text": "The Act does not state the standard deduction amount or rate'`
- `' allowable on income from house property in the provided provisions.", "citations": []}\n{"type": "computation", "text": "'`

**Re-reading `generation/claims.py`'s `LineBuffer.feed()` shows this doesn't
matter for correctness.** `feed()` does
`(self._partial + delta).split("\n")`, which is correct for any number of
embedded newlines landing anywhere in any delta — the "never mid-delta"
property Step 7.1 observed for Groq was a description of Groq's framing, not
something `iter_lines()`/`parse_claim()` actually depend on. The real gate —
does the parser extract the right claims from Gemini's stream — is what
`scripts/probe_gemini_stream.py` was rewritten to check directly (feeding
captured deltas through the real `iter_lines()`/`parse_claim()`, counting
valid claims vs malformed lines) rather than inspecting delta boundaries.

**That corrected script has not yet completed a full run**: Gemini's
free-tier quota was exhausted by the two prior live runs in this session (9
completed and attempted calls total), and every subsequent attempt — a 6-run
pass and a conservative 2-run retry — returned `429` on every one of 3
attempts, immediately, with no partial success. This looks like a sustained
(likely daily) quota exhaustion rather than a per-minute burst limit, the
same pattern ADR-117 already recorded for Groq's daily cap.

**What stands, pending that re-run:**
- The correctness argument from `LineBuffer.feed()`'s code is a
  mathematical property of string splitting, not an empirical one — it holds
  regardless of how a provider chunks its SSE deltas, so no future re-run can
  actually overturn it short of a bug in `feed()` itself (already covered by
  `tests/test_claims_parser.py`'s existing multi-newline-delta tests, if
  present — confirm before relying on this alone).
- The one raw sample captured (see below) is well-formed, valid multi-claim
  NDJSON with no visible corruption at any newline boundary — consistent
  with, not a substitute for, the code-level argument.
- **Gate 1 is provisionally PASS.** Re-run
  `uv run python scripts/probe_gemini_stream.py` (now checks the real parser,
  not delta boundaries) once the quota resets, to convert "argued" into
  "measured" before treating R18 gate 1 as fully closed.

## Availability note — carry into gates 2 and 3

Across both live attempts this session, only 2 of 10 total Gemini calls
completed successfully; the rest failed with `503`, `429`, or a read timeout,
several exhausting all 3 retry attempts. Free-tier availability is a real
constraint on R18's viability independent of the correctness question above —
gate 2 (the two-arm benchmark) and gate 3 (20b eval re-runs, which use Groq's
separate 20b bucket and Gemini only as the small nodes' fallback) should
budget for this, and R18's ultimate per-node routing decision should weigh it
alongside the measured floors.

## One run's raw text (from the 4,096-cap run, `finish_reason="stop"`)

```
{"type": "statute", "text": "Any income from letting out of a residential house or a part of it by the owner shall not be included in income under business or profession and shall be chargeable only under the head \"Income from house property\".", "citations": [{"path": "26(4)", "quote": "Any income from letting out of a residential house or a part of it by the owner shall not be included in income under sub-section (1) and shall be chargeable only under the head \"Income from house property\"."}]}
{"type": "no_basis", "text": "The Act does not state the amount or rate of standard deduction allowed on income from house property in the provided text.", "citations": []}
```
