Measured against [('groq', 'openai/gpt-oss-20b')].

## Follow-up cases — Step 11.8

Multi-turn retrieval, simplified (ADR-110): no recall study, no gate.
Each case is a prior gold question plus a follow-up phrasing that only
makes sense after it; Step 11.7's contextualizer rewrites it before
retrieval sees it. `expected` is for a human to eyeball, not a score.

Rewritten and hit expected citation: 4/5.

### f001 — What if my income is below that threshold instead?

Prior turn: What does section 6(5) substitute for the sixty-day period?

Rewritten (yes): What if my income is below the threshold?

Expected: 6(4), 6(5). Retrieved: 341(6)(b), 332(5), 285, Schedule III(17), 439, 62(2), 341(8), 156(2)(b), 175(3), 263(1), 45(9). miss.

### f002 — Can I pay it to my mother instead?

Prior turn: I pay rent but my employer gives me no house rent allowance. Can I deduct the rent I pay?

Rewritten (yes): Can I pay my rent to my mother instead?

Expected: 134(1), 134(2). Retrieved: 134, Schedule III(11), 533(2)(f), 402(29), 397(2)(e), 28(1), 2(5)(c), 17(1), 23, 21, 2(5)(a), 2(5)(b)(ii). HIT.

### f003 — What if I buy it two years later instead?

Prior turn: I sold my house and bought another one with the money. Do I still pay capital gains tax?

Rewritten (yes): If I buy a new house two years after selling my old one, will I still have to pay capital gains tax?

Expected: 82(1). Retrieved: 86, 82, 83, 84. HIT.

### f004 — What about after that period ends?

Prior turn: Can I deduct the interest on my education loan, and for how many years?

Rewritten (yes): What happens after the deduction period for education loan interest ends?

Expected: 129(2). Retrieved: 129, 130, 131, 132. HIT.

### f005 — Does the same limit apply to a plug-in hybrid instead?

Prior turn: What deduction does section 132 give for a loan taken to buy an electric vehicle?

Rewritten (yes): Does the same deduction limit under section 132 apply to a plug‑in hybrid vehicle?

Expected: 132. Retrieved: 132, 32(i)(B), 122, 58, 133(3), 156, 19(2)(a), 131. HIT.
