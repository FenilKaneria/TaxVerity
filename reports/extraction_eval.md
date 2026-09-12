# Extraction eval — Step 7.7

42 labelled turns, 49 labelled facts, 48,809 tokens, 415s.

## Headline

Detection is whether the node found the field at all; value and status accuracy are measured only over the fields it did find, because a prompt change fixes the first and `normalise_value` fixes the second.

- Field precision **1.000**, recall **1.000**, f1 **1.000**
- Value accuracy **0.959**
- Status accuracy **0.980**
- Strict (field, value and status all right) **0.939**
- Turns with nothing wrong: **40/42**

## Source spans

`parse_facts()` refuses a span the turn does not contain, so no surviving fact can carry a fabricated one. The rate below is therefore of *attempts*, which is the only honest denominator.

- Fabricated-span rate **0.000** (0 of 46 stated attempts)
- Stated facts quoting nothing: 0
- Turns that needed a repair: **1**, of which the repair helped **1**

## By slice

| slice | labelled | found | spurious | missed | precision | recall |
|---|---|---|---|---|---|---|
| simple | 16 | 16 | 0 | 0 | 1.000 | 1.000 |
| formatting | 8 | 8 | 0 | 0 | 1.000 | 1.000 |
| inferred | 4 | 4 | 0 | 0 | 1.000 | 1.000 |
| loss | 3 | 3 | 0 | 0 | 1.000 | 1.000 |
| multi | 15 | 15 | 0 | 0 | 1.000 | 1.000 |
| none | 0 | 0 | 0 | 0 | 0.000 | 0.000 |
| out_of_vocabulary | 0 | 0 | 0 | 0 | 0.000 | 0.000 |
| pii | 3 | 3 | 0 | 0 | 1.000 | 1.000 |

## By field

| field | labelled | found | spurious | missed | precision | recall |
|---|---|---|---|---|---|---|
| assessment_year | 2 | 2 | 0 | 0 | 1.000 | 1.000 |
| regime | 3 | 3 | 0 | 0 | 1.000 | 1.000 |
| age | 6 | 6 | 0 | 0 | 1.000 | 1.000 |
| residential_status | 4 | 4 | 0 | 0 | 1.000 | 1.000 |
| salary_income | 11 | 11 | 0 | 0 | 1.000 | 1.000 |
| house_property_income | 3 | 3 | 0 | 0 | 1.000 | 1.000 |
| business_income | 2 | 2 | 0 | 0 | 1.000 | 1.000 |
| capital_gains_short_term | 2 | 2 | 0 | 0 | 1.000 | 1.000 |
| capital_gains_long_term | 3 | 3 | 0 | 0 | 1.000 | 1.000 |
| other_sources_income | 2 | 2 | 0 | 0 | 1.000 | 1.000 |
| deduction_savings_insurance | 2 | 2 | 0 | 0 | 1.000 | 1.000 |
| deduction_health_insurance | 3 | 3 | 0 | 0 | 1.000 | 1.000 |
| deduction_other | 1 | 1 | 0 | 0 | 1.000 | 1.000 |
| tds_paid | 3 | 3 | 0 | 0 | 1.000 | 1.000 |
| advance_tax_paid | 2 | 2 | 0 | 0 | 1.000 | 1.000 |

## Every turn that was not clean

- **t013** (loss) `I have a loss of 250000 from my rented house after the interest deduction.`
  - house_property_income: expected -250000, got 250000
- **t017** (loss) `I booked a short-term capital loss of 40,000.`
  - capital_gains_short_term: expected -40000, got 40000
