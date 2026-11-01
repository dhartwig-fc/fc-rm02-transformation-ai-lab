# Synthetic AML Data Generator (`synthgen`)

Creates a fake bank, with customers, accounts and transactions. It hides known money-laundering patterns in the transactions, runs simple monitoring rules to produce alerts, and keeps an **answer key** so you can measure how well an AI triages.

```bash
python3 -m synthgen --seed 42 --customers 200 --days 90 --out data/synthetic/run-42
```

Same seed = identical data every time. Output is git-ignored. Refuses to write into `data/real/`, `data/restricted/` or `cases/`.

| File | Contents |
|---|---|
| `customers.csv`, `accounts.csv`, `transactions.csv` | The fake bank |
| `alerts.csv` | Alerts from 4 simple rules, labelled TRUE/FALSE_POSITIVE |
| `ground_truth.csv` | The hidden cases (**answer key**: hide from any model you are testing) |
| `rules.csv`, `manifest.json` | Rule descriptions; seed, counts, synthetic notice |

**Hidden typologies:** structuring, mule pass-through, high-risk jurisdiction, funnel account.
**Legitimate look-alikes (false positives):** cash business (market trader), group holiday collection.

**Provably fake:** every IBAN uses the bank code `TEST`, which doesn't exist (check digits are valid). These IBANs still trip the sensitivity gate on purpose. Prefix messages with `[synthetic-data]`.

**v0.1 simplifications:** one account per customer, no balances, GBP only, illustrative high-risk country list.

Tests: `tests/test_synthgen.py` (143 tests passing across the suite).
