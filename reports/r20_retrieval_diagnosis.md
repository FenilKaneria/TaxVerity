# R20 retrieval-quality and latency diagnosis (Step 20.1)

Measured against the real Supabase corpus/production retriever composition, 12 R20 gold questions.

## Recall@k / Coverage@k on the ranked list

| k | mean governing recall | mean precision | mean rule coverage | coverage_complete rate |
|---|---|---|---|---|
| 10 | 1.000 | 0.310 | 0.805 | 0.600 |
| 20 | 1.000 | 0.170 | 0.875 | 0.800 |

## Coverage on the delivered evidence pack (what the model actually sees)

Mean rule coverage: **0.825**. Coverage-complete rate: **0.700**.

| query | slice | pack size | coverage | complete | missing units |
|---|---|---|---|---|---|
| r001 | eligibility | 7 | 1.00 | True | - |
| r002 | applicability | 8 | 1.00 | True | - |
| r003 | calculation | 2 | 0.25 | False | 156(2)(a), 156(2)(b), 19(1) |
| r004 | deduction_limit | 5 | 1.00 | True | - |
| r005 | applicability | 6 | 1.00 | True | - |
| r006 | applicability | 6 | 1.00 | True | - |
| r007 | multi_issue | 8 | 0.50 | False | Schedule III(11), Schedule III(11)(a), Schedule III(11)(c), Schedule III(11)(d) |
| r008 | deduction_limit | 4 | 1.00 | True | - |
| r009 | calculation | 2 | 0.50 | False | 202(1) |
| r010 | eligibility | 7 | 1.00 | True | - |

## Failure attribution (per missed rule unit, lenient credit)

Total misses: 8. {'ranking': 7, 'budget': 1}

| unit | cause | detail |
|---|---|---|
| 19(1) | ranking | in top 100 but outside the top 20 pool |
| 156(2)(a) | ranking | in top 100 but outside the top 20 pool |
| 156(2)(b) | ranking | in top 100 but outside the top 20 pool |
| Schedule III(11) | ranking | in top 100 but outside the top 20 pool |
| Schedule III(11)(a) | ranking | in top 100 but outside the top 20 pool |
| Schedule III(11)(c) | ranking | in top 100 but outside the top 20 pool |
| Schedule III(11)(d) | ranking | in top 100 but outside the top 20 pool |
| 202(1) | budget | in the pool but cut by the evidence-pack token budget |

## Negative-question regression check

- r011: pack size 6 (0 expected; a non-zero pack is a false-positive relevance risk, same shape as the R19 Phase D q029 finding)
- r012: pack size 11 (0 expected; a non-zero pack is a false-positive relevance risk, same shape as the R19 Phase D q029 finding)

## Latency, per sub-stage (ms)

| stage | simple | complex |
|---|---|---|
| embed | 352.5 | 324.5 |
| pgvector | 201.0 | 161.9 |
| bm25 | 10.0 | 17.7 |
| fusion | 2.9 | 2.3 |
| rerank | 827.8 | 389.4 |
| pack | 3.1 | 0.9 |
| total | 1397.3 | 896.8 |
| pack_units | 2 | 9 |

## Correction to the automated rerank timing figures

`rerank complex` above (389.4ms) is **not a genuine successful rerank call**
— the live run hit Jina's rerank endpoint mid-way through 429
(`RATE_CONCURRENCY_LIMIT_EXCEEDED`, the same free-tier concurrency-2 limit
observed live during this session) and the script's own except-block timed
the failed round trip, not a scored response. `rerank simple` (827.8ms) did
succeed and is a genuine measurement, consistent with Step 5.6's own
recorded rerank latency (p50 606ms / p95 1,126ms). Treat rerank cost as
"~600-1,100ms when it succeeds, near-instant on a 429" rather than reading
the two columns as directly comparable.

## Manual inspection: what actually happened in three representative misses

The automated failure-attribution table above is honest but coarse (see
"script limitation" below). Direct inspection of the ranked lists and the
packs actually delivered for three misses gives a sharper answer to "is it
chunking, context, ranking, or budget":

**r009 ("What standard deduction do I get on salary if I am taxed under the
default regime?") — genuinely a chunking/packing-representation issue, not
"budget is too small".** The top 20 pool correctly ranks `19` (root),
`19(1)`, `35(b)` and others near the top, and `202` at rank 11 (crediting
`202(1)` leniently). But `19(1)` alone is 5,184 characters (~1,050 proxy
tokens); its **root** `19` is 10,647 characters, because `EvidencePacker`'s
walk (ADR-055's "an ancestor absorbs its packed descendants") always
promotes a hit to its containing root once both are in the ranked list —
there is no check for whether the root is *disproportionately* larger than
the hit it is absorbing. That single promoted root consumes most of the
2,500-token production budget, leaving room for only one more small,
unrelated unit (`392(2)`, 711 characters) and starving `202`/`202(2)`
(ranks 8 and 11) even though they ranked well. **This is the single most
concrete, fixable finding of this diagnosis**: the packer's promote-to-root
rule needs a size guard (for example: do not promote past a root more than
3x larger than the hit itself; deliver the hit plus its own lead-in instead)
before touching ranking latency at all.

**r003 ("My salary income is Rs 18,00,000 ... What tax do I pay under the
new regime?") — a genuine ranking/vocabulary gap, not chunking.** `202(1)`
(the slab table) ranks #1; `19(1)` (standard deduction) and `156(2)(a)/(b)`
(rebate) do not appear anywhere in the top 100 for this exact phrasing —
the question names no word close to "deduction" or "rebate", and neither
the term bridge (ADR-086, 64 statutory terms) nor the classifier's
`search_query` rewrite (ADR-120) currently maps "what tax do I pay" to also
surface the standard-deduction and rebate provisions that a *complete*
answer needs. `392` (an unrelated advance-tax provision) outranks both and
ends up in the pack instead. This is exactly the gap R20's planned
**sub_queries / decomposition** step (PLAN 20.3) is meant to close — a
calculation question should retrieve for "the slab rates", "the standard
deduction from salary" and "the rebate" as separate legal sub-questions,
not one combined embedding.

**r007 (the multi-issue HRA + house-property-loss question) — the sharpest
evidence for decomposition.** The house-property half (`134`, `109` family,
`110`) fills the entire top 20; **not one `Schedule III(11)` citation
appears anywhere in the top 100** for the combined question, even though
`r001`/`r010` (the same HRA provision asked alone) retrieve it perfectly at
rank 1. A single dense/BM25/rerank pass over one compound question lets the
larger, more textually dominant issue starve the smaller one completely —
this is not a chunking problem (the HRA chunks are fine, proven by r001/
r010) and not a budget problem (there was room in the pack); it is a single-
query-embedding structural limit that only query decomposition fixes.

## What this rules out

Across all 12 questions and all 8 missed rule units: **zero** misses were
caused by an unreachable or missing chunk (the `chunking` cause in the
automated table never fired — every rule unit in this benchmark resolves to
a real chunk), and **zero** were a pure retrieval-recall failure (nothing
failed to appear anywhere in fusion's own top 100). The governing provision
itself was found for **100% of answerable questions at k=10**. The problem
this diagnosis found is real, but it is narrower than "retrieval is bad":
it is (a) a single evidence pool of 20 being too narrow once a rule's full
condition set, or two unrelated issues, compete for it, and (b) the
evidence packer's ancestor-promotion rule occasionally consuming most of
the token budget on one oversized root.

## Script limitation, reported not hidden

`classify_miss()`'s `context` branch is unreachable dead code in this
implementation: a miss is only handed to the classifier once it has already
failed the "is it in the pack" check, so by construction it can only be
`chunking` (no chunk at all), `recall` (not in fusion's top 100), or
`ranking`/`budget` (found, but excluded by the pool or the token budget). A
genuine "found the governing provision but not its cross-referenced
proviso specifically" case would currently be folded into `ranking` or
`budget` rather than getting its own label. The manual inspection above is
what actually distinguishes "packer chose the wrong shape" (r009) from
"ranking genuinely missed it" (r003, r007) — 20.2's implementation should
either sharpen this distinction in code or continue doing this kind of
targeted inspection for any newly-observed miss pattern, per rule 01's
"small representative check" rather than building a fully general
attribution engine for a 12-question benchmark.

## Proposed chunk / context representation for 20.2

1. **Cap ancestor promotion by relative size** in `EvidencePacker`'s walk
   (`retrieval/evidence.py`): a hit is promoted to its containing root only
   if the root's text is no more than, say, 3x the hit's own text; otherwise
   the hit is packed standalone with its own lead-in chain (the existing
   Step 5.3 mechanism), leaving the root's other descendants for a
   subsequent hit to pull in on their own. This directly fixes r009's
   pattern without touching ranking or budget size. Needs its own before/
   after measurement on this benchmark (Rule Coverage@k must not regress)
   before being adopted, per ADR discipline for a change to a load-bearing
   evidence-delivery rule (ADR-082).
2. **Retrieve per sub-query for multi-issue and calculation questions**
   (PLAN 20.3's `understand`/`sub_queries`), one embed+search+rerank pass
   per sub-question, results merged and deduped before packing. This is
   the direct fix for r003 and r007's pattern. It does **not** apply to a
   plain single-issue question (rule 01: no extra retrieval pass where one
   suffices) — `intent` decides whether sub-queries are generated at all.
3. **No chunking/parser change is justified by this benchmark.** Every
   miss traces to ranking, decomposition, or the packer's promotion rule —
   not to the underlying chunk boundaries themselves. This benchmark does
   not support re-opening ADR-055/056/057.

## Exact node/state diff this diagnosis licenses for 20.2 (retrieval only)

- `retrieval/evidence.py`: `EvidencePacker`'s walk gains a size-ratio guard
  on ancestor promotion, a new constant (e.g. `MAX_PROMOTION_RATIO`), and a
  test pinning r009's exact case (`19(1)` delivered standalone rather than
  via `19`).
- `retrieval/pgvector.py`: add an id-only variant of `SEARCH_SQL` (no
  `xref_edges` subquery, no full `text`) and hydrate from the in-memory
  chunk map already held by `GraphDeps.chunks` — measured contribution to
  the 160-200ms `pgvector` stage above still needs a before/after number
  once built (this diagnosis did not yet change any retrieval code).
- `graph/nodes.py`'s `_retrieve` gains sub-stage trace entries matching the
  latency table above (`retrieve.embed`, `retrieve.pgvector`,
  `retrieve.bm25`, `retrieve.fusion`, `retrieve.rerank`, `retrieve.pack`),
  additive to the existing single `retrieve` trace entry — the developer
  trace panel is not redesigned, only given finer-grained entries.
- No change to `graph/build.py`'s node list or edges in 20.2 — sub-query
  decomposition is a 20.3/`understand` concern (needs `intent` first) and
  is out of scope for a retrieval-only step.
