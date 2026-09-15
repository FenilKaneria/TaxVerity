# R18 gate 3 (part) — loss-sign held-out turns against Groq 20b

24 held-out turns (12 + 12), 22,953 tokens, 227s.

## Verdict

**PASS — no wrong value on either held-out set.** v1 clean 10/12, v2 clean 10/12.

## v1 (12 turns)

- **t002**: wrong (), missed (<FactField.CAPITAL_GAINS_LONG_TERM: 'capital_gains_long_term'>,), invented ()
- **t004**: wrong (), missed (<FactField.HOUSE_PROPERTY_INCOME: 'house_property_income'>,), invented ()

## v2 (12 turns)

- **t002**: wrong (), missed (<FactField.HOUSE_PROPERTY_INCOME: 'house_property_income'>,), invented ()
- **t008**: wrong (), missed (<FactField.HOUSE_PROPERTY_INCOME: 'house_property_income'>,), invented ()
