# docs/

Written policy and reference material — the documents a reader consults to
understand how the system is *meant* to behave, as distinct from:

- `PLAN.md` — what each step is and whether it is done
- `DECISIONS.md` — why an architectural choice was made
- `reports/` — findings produced by a step (corpus profile, structure probe,
  failure taxonomy)
- `evals/` — hand-labelled datasets and the measured results run against them
  (`evals/datasets/` is authored by hand and tracked in git; `evals/reports/`
  is generated per `corpus_version` from Step 3.6)

## Contents

This directory exists because Step 0.2 established the documentation spine;
files arrive when a step needs them.

| File | Arrived in | Purpose |
|---|---|---|
| `SAFETY_POLICY.md` | Step 12.1 (2026-09-13) | The scope taxonomy and the avoidance/evasion boundary with worked examples, implementing ADR-023. The safety eval set (12.4) is written against it, and it is the reference to consult rather than re-deriving the taxonomy ad hoc. |
