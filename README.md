# TaxVerity

Evidence-grounded question answering over the **Income-tax Act, 2025 (India)**.

A user describes a financial or tax situation in natural language; the system
answers from the actual statutory text — not the model's memorised knowledge —
with traceable citations and deterministic arithmetic.

The governing rule, from which most of the architecture follows:

> An unsupported claim is a correctness failure, not a quality issue.

## Status

Phase 0 — foundations. Steps 0.1 (repo skeleton) and 0.2 (documentation
spine) complete; Step 0.3 (config module) is next.

The roadmap and the architecture decision records are maintained outside
version control and are not part of this repository. [`docs/`](docs/) holds
written policy; `SAFETY_POLICY.md` arrives in Step 12.1.

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
