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
