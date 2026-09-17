# R20 Step 20.2 — sub-query decomposition benchmark

Live re-run against the real Supabase corpus, production retriever composition and the real `IntentClassifier`, comparing the plain single-query baseline against sub-query decomposition (ADR-123) on the same 10 answerable R20 gold questions.

## Sub-queries the live classifier actually emitted

| query | sub_queries |
|---|---|
| r001 | 0 |
| r002 | 0 |
| r003 | 0 |
| r004 | 0 |
| r005 | 2 |
| r006 | 2 |
| r007 | 2 |
| r008 | 2 |
| r009 | 0 |
| r010 | 0 |

## Recall@k / Coverage@k on the ranked list

| variant | k | mean governing recall | mean precision | mean rule coverage | coverage_complete rate |
|---|---|---|---|---|---|
| baseline | 10 | 1.000 | 0.320 | 0.825 | 0.700 |
| baseline | 20 | 1.000 | 0.170 | 0.875 | 0.800 |
| decomposed | 10 | 1.000 | 0.350 | 0.875 | 0.800 |
| decomposed | 20 | 1.000 | 0.195 | 0.925 | 0.900 |

## Coverage on the delivered evidence pack

- baseline: mean rule coverage **0.825**, coverage-complete rate **0.700**
- decomposed: mean rule coverage **0.875**, coverage-complete rate **0.800**

| query | slice | baseline pack coverage | decomposed pack coverage | baseline missing | decomposed missing |
|---|---|---|---|---|---|
| r001 | eligibility | 1.00 | 1.00 | - | - |
| r002 | applicability | 1.00 | 1.00 | - | - |
| r003 | calculation | 0.25 | 0.25 | 156(2)(a), 156(2)(b), 19(1) | 156(2)(a), 156(2)(b), 19(1) |
| r004 | deduction_limit | 1.00 | 1.00 | - | - |
| r005 | applicability | 1.00 | 1.00 | - | - |
| r006 | applicability | 1.00 | 1.00 | - | - |
| r007 | multi_issue | 0.50 | 1.00 | Schedule III(11), Schedule III(11)(a), Schedule III(11)(c), Schedule III(11)(d) | - |
| r008 | deduction_limit | 1.00 | 1.00 | - | - |
| r009 | calculation | 0.50 | 0.50 | 202(1) | 202(1) |
| r010 | eligibility | 1.00 | 1.00 | - | - |
