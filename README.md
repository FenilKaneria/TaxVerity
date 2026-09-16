# TaxVerity

A grounded tax **advisor** for the **Income-tax Act, 2025 (India)**.

Most people don't know which lawful deductions, exemptions or regime choices
actually apply to them, and a generic LLM will advise confidently without
being checkable against the law. TaxVerity does the opposite: a user
describes a financial or tax situation in natural language, and the system
recommends what the Act actually supports — from the actual statutory text,
not the model's memorised knowledge — with traceable citations and
deterministic arithmetic. Where the Act doesn't address something, it says so
plainly instead of guessing.

The governing rule, from which most of the architecture follows:

> An unsupported claim is a correctness failure, not a quality issue.

## Status

Phases 0–17 built (corpus, retrieval, calculator, generation with grounding
verification, accounts/threads, safety/scope, the LangGraph query graph, API,
frontend, AWS deployment). Phase 18 (hardening) is in progress; see the
in-progress R19 work below.

The step-level roadmap and architecture decision records (`PLAN.md`,
`DECISIONS.md`) are maintained outside version control and are not part of
this repository. [`docs/`](docs/) holds written policy, including
`SAFETY_POLICY.md`.

**Known limit:** the calculator is corpus-only (ADR-100) — it computes only
figures the 2025 Act itself prints, and routes to a text-only answer for
anything outside that (old-regime slabs, surcharge, marginal relief, cess).

## Corpus

`Income-tax-Act-2025.pdf` is expected at the repository root. It is **not**
tracked in git (101.8 MiB, over GitHub's per-file limit); obtain it from the
India Code portal and place it there. From Step 1.2 onward its content hash is
recorded in the corpus manifest, which is what actually pins the version.

## Development

Requires [uv](https://docs.astral.sh/uv/) and Python 3.13.

```bash
uv sync                  # create .venv, install deps and the package
uv run pytest            # tests
uv run ruff check        # lint
uv run ruff format       # format
```

Postgres + pgvector for local development runs in Docker:

```bash
docker compose up -d --wait   # pgvector 0.8.6 on Postgres 17, 127.0.0.1:5432
```

Set `TAXVERITY_DATABASE_URL` in `.env` as shown in `.env.example`. Without it,
the database tests skip. Then create the schema and load the corpus:

```bash
uv run python scripts/migrate.py         # apply pending SQL migrations
uv run python scripts/ingest_corpus.py   # load chunks and vectors, idempotent
```

## Running the API in a container

`Dockerfile` builds the same image ECR/Lambda serves in production (ADR-076)
and the `api` service `compose.yaml` runs locally (Step 15.3) — no separate
dev image. It needs `TAXVERITY_JINA_API_KEY`, `TAXVERITY_GROQ_API_KEY` and
`TAXVERITY_JWT_SECRET` set in `.env`, and the database migrated and loaded
per the steps above.

```bash
docker compose up -d --wait   # postgres, then the api image (builds if needed)
curl http://127.0.0.1:8080/health
```

**Measured once (2026-09-14), against the real corpus** — 8,351 chunks, one
embedding set — not asserted in CI (Step 15.5): image size **1.01 GB** (no
torch — ADR-075 moved embeddings and reranking off this image); cold
container start to a passing `/health` (chunk hydration, BM25 build, pgvector
index, term bridge, LLM/reranker client construction) **~3.8 s**, well inside
Lambda's 10 s INIT budget (ADR-076).
