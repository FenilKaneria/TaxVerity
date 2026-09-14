# Step 17.8 — deploy smoke test + latency pass

Target: `https://5iiiww6f4ghk7lo6mrzkw7cdzm0sxlze.lambda-url.ap-south-1.on.aws`
Frontend: `https://frontend-swart-xi-44.vercel.app`

## Method

`scripts/smoke_deploy.py`: register -> mint a verify token directly against
the deployed database (mirrors the interactive shortcut used earlier this
project, since the script cannot click a mailed link) -> verify -> login ->
create a thread -> one turn per question over real SSE, asserting rule 04's
contract (no `claim` event ever carries `verified:false`, every turn ends
with `final` carrying the disclaimer). Reuses the Step 10.7 smoke set's 15
questions (`SMOKE_QUERY_IDS`) rather than authoring a second curated list, per
ADR-110's "reuse gold v2 where possible."

## Run 1 — 15 questions, Lambda timeout at its original 120s

14/15 questions completed the full event contract cleanly: verified claims,
`final`, disclaimer present, nothing unverified ever served. `q003` ("What
does section 6(5) substitute for the sixty-day period?") hit the minimal
corrective retry (ADR-033/110) and needed 120.36s server-side — **Lambda's
function timeout, set to exactly 120s at Step 17.2, killed the invocation
mid-stream**: the connection closed with no `final` event and no disclaimer,
after a `withheld` and a second "refining search" stage. Nothing unverified
was shown (the grounding invariant held), but the turn simply never finished.

| query | claims | withheld | total_s | first_claim_s |
|---|---|---|---|---|
| q001 | 3 | 1 | 17.07 | 3.91 |
| q002 | 1 | 0 | 28.62 | 28.61 |
| q003 | 0 | 1 | 120.36 (truncated, no `final`) | 90.46 |
| q009 | 1 | 0 | 12.23 | 12.23 |
| q010 | 1 | 1 | 83.08 | 42.81 |
| q011 | 2 | 0 | 57.51 | 57.21 |
| q012 | 1 | 0 | 30.36 | 30.28 |
| q019 | 12 | 0 | 77.79 | 77.57 |
| q020 | 1 | 0 | 23.53 | 23.53 |
| q021 | 1 | 2 | 109.36 | 48.97 |
| q025-q029 (negative slice) | 0 | 0 | 0.3-5.1 | n/a |

**Fix applied**: `aws lambda update-function-configuration --timeout 180`
(also updated in `infra/main.tf`), reasoned to be safe against every other
component in the path — no API Gateway (direct Function URL, no 29s cap), no
client-side `fetch` timeout in the frontend (rule 04: only the user's own Stop
button aborts), Lambda bills actual duration not the configured ceiling.

## Run 2+ — Groq free-tier throttling under repeated smoke-test load

Attempting to re-verify `q003` cleanly under the new 180s ceiling, and
separately a full paced 15-question re-run (50s between questions, matching
`answer_smoke.py`'s `DEFAULT_PAUSE`, plus a re-login before each turn since 15
paced questions can outlast the 15-minute access-token TTL), both runs hit
sustained `groq returned 429` — **a known, already-documented capacity ceiling
(Step 7.1: ~1.7 evidence-pack queries a minute on Groq's free tier), not a
regression from anything built at 17.6-17.8.** One turn issues 3-4 sequential
Groq calls (classify, extract, generate, and the corrective retry's second
generate), so even a single isolated turn can exhaust the per-minute budget on
its own after the earlier runs had already spent it — the throttling is
per-API-key, not per-script-invocation, and does not reset just because a
script run was stopped.

**A genuine robustness gap this surfaced, not previously covered by any
test:** `graph/nodes.py`'s `extract_facts` node has no error handling around
`deps.extractor.extract()`. When Groq returns 429 three times in a row,
`LLMUnavailable` propagates uncaught out of the LangGraph node, killing the
whole turn — the SSE stream ends with no `final` event and no disclaimer, the
same silent-truncation symptom as the timeout issue above, but from a
different cause. **No Gemini fallback is configured** (rule 05, Step 7.2:
"Gemini has no live key configured"), so there is currently no degradation
path for this at all. Logged as residue for Phase 18 hardening — not fixed
here, since a production rate limiter and a real cross-vendor fallback are
both explicitly out of 17.8's SIMPLIFIED scope (ADR-110) and chasing a
perfectly clean re-run against a free-tier LLM to prove it is exactly the
"budget study" rule 01 says to avoid.

## Cold start

First `/health` response after a Lambda config change (forces a fresh
execution environment): **7.07s**, including real network RTT (this machine
to `ap-south-1`) and full corpus hydration from Supabase (chunk load, BM25
build, pgvector index, term bridge, LLM/reranker client construction). Not
directly comparable to Step 15.5's 3.8s local-docker figure (different network
path, no Supabase round trip there), but the same order of magnitude.

## Bug found and fixed by this pass, unrelated to the two findings above

Every verification and password-reset email linked to `/verify?token=...` and
`/reset?token=...` — the frontend's actual routes are `/verify-email` and
`/reset-password` (confirmed against `useSearchParams().get("token")` in both
page components). Every real verification/reset link ever sent 404'd.
Fixed in `src/taxverity/auth/accounts.py` (4 call sites); regression tests
added to `tests/test_email_flows.py` pinning the exact link path, since no
existing test checked more than "a token is present somewhere in the body" —
exactly how this shipped and stayed broken. Rebuilt and redeployed.

## Summary

- Deploy pipeline (Vercel <-> Lambda Function URL, RESPONSE_STREAM, real
  claim-level SSE) is proven correct end to end for the common case: 14/15
  smoke questions, register/verify/login/account-deletion, guest path (17.6).
- Two real production gaps found and one fixed live: the email link path bug
  (fixed, deployed, regression-tested); the 120s Lambda timeout truncating a
  corrective-retry turn (fixed: 180s, applied and documented in Terraform).
- One gap found and **not** fixed, logged as Phase 18 residue: `extract_facts`
  has no degradation path for a Groq 429, and there is no configured
  cross-vendor fallback to degrade to.
