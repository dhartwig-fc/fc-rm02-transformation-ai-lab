<!--
  Reference copy of the Week 10 capstone output, committed so the expected
  deliverable can be read without running anything. It is GENERATED -- do not
  edit it by hand. Regenerate with:

      python weeks/week10_capstone.py

  Every figure comes from a synthetic population; the numbers illustrate the
  shape of the argument, not any real portfolio.
-->

# Tuning Paper: TM-014 High Value Outbound Wires

**Date:** 2026-09-14  
**Author:** Dan Hartwig  
**Status:** Draft for independent validation

## 1. Recommendation

Move the threshold from **£50,000** to **£58,780**.

- Alert volume: 7,219 -> 6,271 (-948, a 13.1% reduction)
- True cases detected: 583 -> 526 (-57)
- Precision: 8.08% -> 8.39% (+0.31pp)
- Recall: 47.83% -> 43.15% (-4.68pp)
- Investigator effort per true case: 12.4 -> 11.9 alerts

## 2. Rule under review

```
ALERT IF monthly_outbound_wire_value > 50,000
  proposed: monthly_outbound_wire_value > 58,780
  scope: all active customers | frequency: monthly
```

## 3. Data and sample

- Observations: 44,984 scored records
- True cases in sample: 1,219 (2.71% base rate)
- Label basis: Confirmed SAR/STR submission within 90 days of the alert period

- Period covered: 2024-01 to 2025-06 (18 periods)

## 4. Methodology

Retrospective backtest over the sample above. The rule was re-executed at each candidate threshold and scored against the label set, producing the sweep in section 5. Threshold selection was constrained by operational alert capacity rather than chosen to maximise a single composite metric, so that the detection/effort trade-off is made explicitly rather than implied by a scoring function.

## 5. Threshold sweep

| Threshold (£) | Alerts | True cases found | Missed | Precision | Recall | Alerts per case |
| --- | --- | --- | --- | --- | --- | --- |
| 1,628 | 42,734 | 1,207 | 12 | 2.82% | 99.02% | 35.4 |
| 2,589 | 40,173 | 1,198 | 21 | 2.98% | 98.28% | 33.5 |
| 3,485 | 37,612 | 1,172 | 47 | 3.12% | 96.14% | 32.1 |
| 4,330 | 35,050 | 1,150 | 69 | 3.28% | 94.34% | 30.5 |
| 5,199 | 32,489 | 1,131 | 88 | 3.48% | 92.78% | 28.7 |
| 6,114 | 29,928 | 1,114 | 105 | 3.72% | 91.39% | 26.9 |
| 7,140 | 27,366 | 1,094 | 125 | 4.00% | 89.75% | 25.0 |
| 8,297 | 24,805 | 1,074 | 145 | 4.33% | 88.11% | 23.1 |
| 9,662 | 22,244 | 1,051 | 168 | 4.72% | 86.22% | 21.2 |
| 11,461 | 19,611 | 1,016 | 203 | 5.18% | 83.35% | 19.3 |
| 13,909 | 16,943 | 974 | 245 | 5.75% | 79.90% | 17.4 |
| 18,045 | 14,275 | 924 | 295 | 6.47% | 75.80% | 15.4 |
| 25,308 | 11,607 | 829 | 390 | 7.14% | 68.01% | 14.0 |
| 37,536 | 8,939 | 687 | 532 | 7.69% | 56.36% | 13.0 |
| 58,780 | 6,271 | 526 | 693 | 8.39% | 43.15% | 11.9 |
| 98,939 | 3,603 | 340 | 879 | 9.44% | 27.89% | 10.6 |
| 231,236 | 935 | 89 | 1,130 | 9.52% | 7.30% | 10.5 |

## 6. Below-the-line testing

A simple random sample of 753 records was drawn from the 38,713 records the rule does not alert on. 18 were true cases.

- Observed below-the-line productive rate: **2.39%**
- 95% Wilson confidence interval: 1.52% to 3.75%
- Estimated cases missed across the full below-the-line population: **925** (upper bound 1,451)

The upper bound is the figure to test against risk appetite: it is the worst case the sample is consistent with, not the best guess.

## 7. Segment calibration

| Segment | Global £ | Alerts | Cases | Calibrated £ | Alerts | Cases | Uplift |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CORPORATE | 58,522 | 3,666 | 285 | 27,066 | 4,260 | 331 | +46 |
| PRIVATE | 58,522 | 852 | 140 | 18,451 | 1,196 | 196 | +56 |
| RETAIL | 58,522 | 0 | 0 | none | 0 | 0 | +0 |
| SME | 58,522 | 1,780 | 102 | 76,484 | 817 | 51 | -51 |
| TOTAL | 58,522 | 6,298 | 527 | n/a | 6,273 | 578 | +51 |

Both options are costed at the same total alert budget, so the uplift column is additional detection for no additional investigator effort.

## 8. Stability and drift

Performance was recomputed independently in each period. 21 of 24 periods breached at least one control limit.


| Period | Alerts | Cases | Precision | Recall | PSI | Stability | Breach |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2024-01 | 265 | 22 | 8.30% | 46.81% | 0.000 | stable | 1 |
| 2024-02 | 267 | 13 | 4.87% | 30.95% | 0.012 | stable | 1 |
| 2024-03 | 322 | 18 | 5.59% | 37.50% | 0.014 | stable | 0 |
| 2024-04 | 323 | 27 | 8.36% | 42.19% | 0.021 | stable | 0 |
| 2024-05 | 324 | 27 | 8.33% | 45.76% | 0.025 | stable | 0 |
| 2024-06 | 364 | 28 | 7.69% | 46.67% | 0.044 | stable | 1 |
| 2024-07 | 368 | 32 | 8.70% | 49.23% | 0.057 | stable | 1 |
| 2024-08 | 389 | 35 | 9.00% | 42.68% | 0.090 | stable | 1 |
| 2024-09 | 356 | 30 | 8.43% | 40.54% | 0.067 | stable | 1 |
| 2024-10 | 412 | 28 | 6.80% | 45.16% | 0.078 | stable | 1 |
| 2024-11 | 420 | 32 | 7.62% | 47.06% | 0.114 | moderate shift | 1 |
| 2024-12 | 471 | 43 | 9.13% | 49.43% | 0.131 | moderate shift | 1 |
| 2025-01 | 465 | 47 | 10.11% | 62.67% | 0.135 | moderate shift | 1 |
| 2025-02 | 475 | 34 | 7.16% | 53.97% | 0.172 | moderate shift | 1 |
| 2025-03 | 480 | 46 | 9.58% | 56.10% | 0.197 | moderate shift | 1 |
| 2025-04 | 497 | 47 | 9.46% | 48.45% | 0.174 | moderate shift | 1 |
| 2025-05 | 509 | 35 | 6.88% | 52.24% | 0.204 | moderate shift | 1 |
| 2025-06 | 512 | 39 | 7.62% | 50.65% | 0.244 | moderate shift | 1 |
| 2025-07 | 538 | 48 | 8.92% | 56.47% | 0.258 | significant shift | 1 |
| 2025-08 | 580 | 46 | 7.93% | 54.12% | 0.293 | significant shift | 1 |
| 2025-09 | 552 | 48 | 8.70% | 53.33% | 0.281 | significant shift | 1 |
| 2025-10 | 624 | 56 | 8.97% | 57.14% | 0.326 | significant shift | 1 |
| 2025-11 | 614 | 58 | 9.45% | 58.00% | 0.333 | significant shift | 1 |
| 2025-12 | 627 | 66 | 10.53% | 63.46% | 0.355 | significant shift | 1 |

## 9. Champion vs challenger

| rule | alerts | tp | fn | precision | recall |
| --- | --- | --- | --- | --- | --- |
| TM-014 incumbent | 7,219 | 583 | 636 | 8.08% | 47.83% |
| TM-014 re-tuned | 6,271 | 526 | 693 | 8.39% | 43.15% |
| TM-014 + risk overlay | 6,699 | 613 | 606 | 9.15% | 50.29% |


Netted metrics conceal reallocation, so the overlap is broken out below. The 'Champion only' row is risk the challenger would newly miss.


| Cell | Records | True cases | Share of all cases | Precision |
| --- | --- | --- | --- | --- |
| Both alert | 6,271 | 526 | 43.15% | 8.39% |
| Champion only | 0 | 0 | 0.00% | n/a |
| Challenger only | 428 | 87 | 7.14% | 20.33% |
| Neither alerts | 38,285 | 606 | 49.71% | 1.58% |

## 10. Limitations and assumptions

- Labels are a proxy for true financial crime. A record with no SAR/STR is not proven clean -- it may be undetected risk, which biases measured recall upward. Every figure in this paper is conditional on the label set.

- Backtesting assumes historical behaviour is representative of the forward period. Any known upcoming change in product, customer mix or typology invalidates that assumption and should be raised before implementation.

- The population is drifting (PSI 0.35 in the final period). The recommended threshold fits capacity in the tuning window but is projected to exceed it within the holdout, so it should be treated as valid for two quarters and re-assessed, not set annually.

- Segment data quality has not been assessed. Segment-specific thresholds make segment assignment an AML control, and that dependency must be confirmed before recommendation 2 is implemented.

- The risk-overlay challenger was not recommended because field coverage for jurisdiction and PEP flags is unverified, not because it underperformed.


## 11. Approval

| Role | Name | Date | Decision |
| --- | --- | --- | --- |
| Tuning analyst | | | |
| Financial crime risk owner | | | |
| Independent model validation | | | |
