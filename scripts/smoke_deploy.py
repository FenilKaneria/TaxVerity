"""Step 17.8 — deploy smoke test + latency pass. SIMPLIFIED per ADR-110: no
new eval framework, no budget study. Drives the real deployed HTTP API
(register -> verify -> login -> create a thread -> one turn per question over
SSE), reusing the Step 10.7 smoke set's 15 questions rather than authoring a
second curated list.

Email verification cannot be completed by clicking a link in this script, so
it mints the verify token directly against the database the deployed API
itself uses (`auth.email_tokens.issue_email_token`) — the same shortcut used
interactively earlier in this project, not a new mechanism. Everything else
goes over the wire exactly as a browser would: `httpx2` streaming the SSE
response, no test client, no in-process shortcut.

Writes `data/answers/deploy_smoke_v1.json` and `reports/deploy_smoke.md`.
"""

from __future__ import annotations

import argparse
import json
import secrets
import statistics
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import httpx2
import psycopg

sys.path.insert(0, str(Path(__file__).resolve().parent))

from taxverity.auth.email_tokens import issue_email_token
from taxverity.config import Settings
from taxverity.evals.answers import SMOKE_QUERY_IDS
from taxverity.evals.gold import GOLD_V2_FILENAME, load_gold_set
from taxverity.observability import configure_logging, get_logger

logger = get_logger(__name__)

RESULT_PATH = Path("data") / "answers" / "deploy_smoke_v1.json"
REPORT_PATH = Path("reports") / "deploy_smoke.md"


def _login(client: httpx2.Client, email: str, password: str) -> str:
    resp = client.post("/v1/auth/login", json={"email": email, "password": password})
    resp.raise_for_status()
    return resp.json()["access_token"]


def _register_and_verify(
    client: httpx2.Client, conn: psycopg.Connection, email: str, password: str
) -> uuid.UUID:
    resp = client.post("/v1/auth/register", json={"email": email, "password": password})
    resp.raise_for_status()

    row = conn.execute(
        "SELECT user_id FROM users WHERE email = %s", (email,)
    ).fetchone()
    if row is None:
        raise RuntimeError(f"registration did not create a user row for {email}")
    user_id = row[0]
    token = issue_email_token(conn, user_id, "verify")

    resp = client.post("/v1/auth/verify-email", json={"token": token})
    resp.raise_for_status()
    return user_id


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True, help="the live API base URL")
    parser.add_argument(
        "--email",
        default=f"smoke+{secrets.token_hex(4)}@example.invalid",
        help="throwaway account email; defaults to a random one",
    )
    parser.add_argument("--password", default="Sm0ke-Test-Not-Real-9")
    parser.add_argument("--keep-account", action="store_true", help="skip DB cleanup")
    parser.add_argument(
        "--pause",
        type=float,
        default=50.0,
        help="seconds between questions, matching answer_smoke.py's DEFAULT_PAUSE "
        "(Groq free tier holds ~1.7 evidence-pack queries a minute)",
    )
    args = parser.parse_args()

    settings = Settings()
    gold_path = settings.evals_dir / "datasets" / GOLD_V2_FILENAME
    gold = {q.query_id: q for q in load_gold_set(gold_path)}
    questions = [(qid, gold[qid].question) for qid in SMOKE_QUERY_IDS]

    with psycopg.connect(
        settings.database_url.get_secret_value(), autocommit=True
    ) as conn, httpx2.Client(base_url=args.base_url, timeout=200.0) as client:
        health = client.get("/health")
        health.raise_for_status()
        logger.info("target reachable: %s", args.base_url)

        user_id = _register_and_verify(client, conn, args.email, args.password)
        client.headers["Authorization"] = f"Bearer {_login(client, args.email, args.password)}"
        logger.info("registered, verified and logged in as %s", args.email)

        thread = client.post("/v1/threads", json={"title": "17.8 smoke"})
        thread.raise_for_status()
        thread_id = thread.json()["thread_id"]

        records: list[dict[str, Any]] = []
        try:
            for i, (query_id, question) in enumerate(questions):
                if i > 0:
                    # Groq's free tier holds ~1.7 evidence-pack queries a
                    # minute (Step 7.1); firing these back-to-back throttles
                    # every question after the first two into a full-timeout
                    # stall with nothing served. Re-login each time too — the
                    # access JWT is short-lived (rule 04) and 15 paced
                    # questions can outlast it.
                    time.sleep(args.pause)
                    client.headers["Authorization"] = (
                        f"Bearer {_login(client, args.email, args.password)}"
                    )
                started = time.perf_counter()
                first_claim_s: float | None = None
                with client.stream(
                    "POST",
                    f"/v1/threads/{thread_id}/turns",
                    json={"question": question},
                ) as response:
                    response.raise_for_status()
                    events = []
                    name: str | None = None
                    for raw_line in response.iter_lines():
                        line = raw_line.rstrip("\r")
                        if line.startswith("event:"):
                            name = line.removeprefix("event:").strip()
                        elif line.startswith("data:") and name is not None:
                            payload = json.loads(line.removeprefix("data:").strip())
                            events.append({"event": name, **payload})
                            if name in ("claim", "withheld") and first_claim_s is None:
                                first_claim_s = time.perf_counter() - started
                            name = None
                total_s = time.perf_counter() - started

                claim_events = [e for e in events if e["event"] == "claim"]
                unverified = [e for e in claim_events if e.get("verified") is not True]
                final_events = [e for e in events if e["event"] == "final"]

                record = {
                    "query_id": query_id,
                    "question": question,
                    "event_types": [e["event"] for e in events],
                    "claims_served": len(claim_events),
                    "withheld": len([e for e in events if e["event"] == "withheld"]),
                    "has_final": len(final_events) == 1,
                    "has_disclaimer": bool(
                        final_events and final_events[0].get("disclaimer")
                    ),
                    "any_unverified_claim": len(unverified) > 0,
                    "total_s": round(total_s, 2),
                    "time_to_first_claim_s": (
                        round(first_claim_s, 2) if first_claim_s is not None else None
                    ),
                }
                records.append(record)
                logger.info(
                    "%s: %d claims, %d withheld, %.1fs",
                    query_id,
                    record["claims_served"],
                    record["withheld"],
                    total_s,
                )
        finally:
            if not args.keep_account:
                conn.execute("DELETE FROM users WHERE user_id = %s", (user_id,))
                logger.info("cleaned up smoke account %s", args.email)

    _write_report(records, args.base_url)


def _write_report(records: list[dict[str, Any]], base_url: str) -> None:
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(
        json.dumps({"base_url": base_url, "records": records}, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    contract_failures = [
        r["query_id"]
        for r in records
        if r["any_unverified_claim"] or not r["has_final"] or not r["has_disclaimer"]
    ]
    totals = [r["total_s"] for r in records]
    first_claims = [r["time_to_first_claim_s"] for r in records if r["time_to_first_claim_s"] is not None]

    def pct(values: list[float], p: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        idx = min(len(ordered) - 1, int(len(ordered) * p))
        return ordered[idx]

    lines = [
        "# Step 17.8 — deploy smoke test + latency pass",
        "",
        f"Target: `{base_url}`",
        f"Questions: {len(records)}",
        f"Contract failures: {len(contract_failures)}"
        + (f" ({', '.join(contract_failures)})" if contract_failures else ""),
        "",
        "## Latency (warm, sequential over one thread)",
        "",
        f"- total turn: p50 {statistics.median(totals):.1f}s, "
        f"p95 {pct(totals, 0.95):.1f}s, max {max(totals):.1f}s",
    ]
    if first_claims:
        lines.append(
            f"- time to first claim/withheld: p50 {statistics.median(first_claims):.1f}s, "
            f"p95 {pct(first_claims, 0.95):.1f}s"
        )
    lines += [
        "",
        "## Per-question",
        "",
        "| query | claims | withheld | total_s | first_claim_s |",
        "|---|---|---|---|---|",
    ]
    for r in records:
        lines.append(
            f"| {r['query_id']} | {r['claims_served']} | {r['withheld']} | "
            f"{r['total_s']} | {r['time_to_first_claim_s']} |"
        )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
