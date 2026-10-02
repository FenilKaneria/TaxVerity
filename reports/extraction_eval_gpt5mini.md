# Extraction eval — Step 7.7

42 labelled turns, 49 labelled facts, 65,133 tokens, 102s.

## Headline

Detection is whether the node found the field at all; value and status accuracy are measured only over the fields it did find, because a prompt change fixes the first and `normalise_value` fixes the second.

- Field precision **1.000**, recall **0.959**, f1 **0.979**
- Value accuracy **0.979**
- Status accuracy **0.957**
- Strict (field, value and status all right) **0.918**
- Turns with nothing wrong: **39/42**

## Source spans

`parse_facts()` refuses a span the turn does not contain, so no surviving fact can carry a fabricated one. The rate below is therefore of *attempts*, which is the only honest denominator.

- Fabricated-span rate **0.000** (0 of 47 stated attempts)
- Stated facts quoting nothing: 0
- Turns that needed a repair: **3**, of which the repair helped **3**

## By slice

| slice | labelled | found | spurious | missed | precision | recall |
|---|---|---|---|---|---|---|
| simple | 16 | 16 | 0 | 0 | 1.000 | 1.000 |
| formatting | 8 | 8 | 0 | 0 | 1.000 | 1.000 |
| inferred | 4 | 2 | 0 | 2 | 1.000 | 0.500 |
| loss | 3 | 3 | 0 | 0 | 1.000 | 1.000 |
| multi | 15 | 15 | 0 | 0 | 1.000 | 1.000 |
| none | 0 | 0 | 0 | 0 | 0.000 | 0.000 |
| out_of_vocabulary | 0 | 0 | 0 | 0 | 0.000 | 0.000 |
| pii | 3 | 3 | 0 | 0 | 1.000 | 1.000 |

## By field

| field | labelled | found | spurious | missed | precision | recall |
|---|---|---|---|---|---|---|
| tax_year | 2 | 2 | 0 | 0 | 1.000 | 1.000 |
| regime | 3 | 3 | 0 | 0 | 1.000 | 1.000 |
| age | 6 | 6 | 0 | 0 | 1.000 | 1.000 |
| residential_status | 4 | 2 | 0 | 2 | 1.000 | 0.500 |
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

- **t008** (inferred) `I retired at 58 and that was two years ago.`
  - age: expected 60, got 58
  - age: expected status inferred, got stated
- **t011** (inferred) `I have been living in Dubai for the last four years and only visit India for a week each year.`
  - missed: residential_status
- **t038** (inferred) `I have been in India for the whole of the last ten years.`
  - missed: residential_status
