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
- True cases detected: 518 -> 462 (-56)
- Precision: 7.18% -> 7.37% (+0.19pp)
- Recall: 40.72% -> 36.32% (-4.40pp)
- Investigator effort per true case: 13.9 -> 13.6 alerts

## 2. Rule under review

```
ALERT IF monthly_outbound_wire_value > 50,000
  proposed: monthly_outbound_wire_value > 58,780
  scope: all active customers | frequency: monthly
```

## 3. Data and sample

- Observations: 44,984 scored records
- True cases in sample: 1,272 (2.83% base rate)
- Label basis: Confirmed SAR/STR submission within 90 days of the alert period

- Period covered: 2024-01 to 2025-06 (18 periods)

## 4. Methodology

Retrospective backtest over the sample above. The rule was re-executed at each candidate threshold and scored against the label set, producing the sweep in section 5. Threshold selection was constrained by operational alert capacity rather than chosen to maximise a single composite metric, so that the detection/effort trade-off is made explicitly rather than implied by a scoring function.

## 5. Threshold sweep

| Threshold (£) | Alerts | True cases found | Missed | Precision | Recall | Alerts per case |
| --- | --- | --- | --- | --- | --- | --- |
| 1,628 | 42,734 | 1,253 | 19 | 2.93% | 98.51% | 34.1 |
| 2,589 | 40,173 | 1,223 | 49 | 3.04% | 96.15% | 32.8 |
| 3,485 | 37,612 | 1,186 | 86 | 3.15% | 93.24% | 31.7 |
| 4,330 | 35,050 | 1,163 | 109 | 3.32% | 91.43% | 30.1 |
| 5,199 | 32,489 | 1,142 | 130 | 3.52% | 89.78% | 28.4 |
| 6,114 | 29,928 | 1,107 | 165 | 3.70% | 87.03% | 27.0 |
| 7,140 | 27,366 | 1,067 | 205 | 3.90% | 83.88% | 25.6 |
| 8,297 | 24,805 | 1,037 | 235 | 4.18% | 81.53% | 23.9 |
| 9,662 | 22,244 | 1,000 | 272 | 4.50% | 78.62% | 22.2 |
| 11,461 | 19,611 | 965 | 307 | 4.92% | 75.86% | 20.3 |
| 13,909 | 16,943 | 922 | 350 | 5.44% | 72.48% | 18.4 |
| 18,045 | 14,275 | 861 | 411 | 6.03% | 67.69% | 16.6 |
| 25,308 | 11,607 | 777 | 495 | 6.69% | 61.08% | 14.9 |
| 37,536 | 8,939 | 629 | 643 | 7.04% | 49.45% | 14.2 |
| 58,780 | 6,271 | 462 | 810 | 7.37% | 36.32% | 13.6 |
| 98,939 | 3,603 | 299 | 973 | 8.30% | 23.51% | 12.1 |
| 231,236 | 935 | 73 | 1,199 | 7.81% | 5.74% | 12.8 |

## 6. Below-the-line testing

A simple random sample of 753 records was drawn from the 38,713 records the rule does not alert on. 14 were true cases.

- Observed below-the-line productive rate: **1.86%**
- 95% Wilson confidence interval: 1.11% to 3.10%
- Estimated cases missed across the full below-the-line population: **720** (upper bound 1,199)

The upper bound is the figure to test against risk appetite: it is the worst case the sample is consistent with, not the best guess.

## 7. Segment calibration

| Segment | Global £ | Alerts | Cases | Calibrated £ | Alerts | Cases | Uplift |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CORPORATE | 58,522 | 3,666 | 213 | 106,233 | 2,629 | 178 | -35 |
| PRIVATE | 58,522 | 852 | 159 | 18,451 | 1,196 | 225 | +66 |
| RETAIL | 58,522 | 0 | 0 | 36,450 | 30 | 1 | +1 |
| SME | 58,522 | 1,780 | 92 | 50,245 | 2,430 | 124 | +32 |
| TOTAL | 58,522 | 6,298 | 464 | n/a | 6,285 | 528 | +64 |

Both options are costed at the same total alert budget, so the uplift column is additional detection for no additional investigator effort.

## 8. Stability and drift

Performance was recomputed independently in each period. 21 of 24 periods breached at least one control limit.


| Period | Alerts | Cases | Precision | Recall | PSI | Stability | Breach |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2024-01 | 265 | 17 | 6.42% | 33.33% | 0.000 | stable | 1 |
| 2024-02 | 267 | 13 | 4.87% | 22.81% | 0.012 | stable | 1 |
| 2024-03 | 322 | 22 | 6.83% | 37.29% | 0.014 | stable | 0 |
| 2024-04 | 323 | 21 | 6.50% | 37.50% | 0.021 | stable | 0 |
| 2024-05 | 324 | 23 | 7.10% | 42.59% | 0.025 | stable | 0 |
| 2024-06 | 364 | 14 | 3.85% | 26.92% | 0.044 | stable | 1 |
| 2024-07 | 368 | 20 | 5.43% | 32.79% | 0.057 | stable | 1 |
| 2024-08 | 389 | 28 | 7.20% | 45.16% | 0.090 | stable | 1 |
| 2024-09 | 356 | 33 | 9.27% | 38.37% | 0.067 | stable | 1 |
| 2024-10 | 412 | 21 | 5.10% | 27.27% | 0.078 | stable | 1 |
| 2024-11 | 420 | 26 | 6.19% | 44.07% | 0.114 | moderate shift | 1 |
| 2024-12 | 471 | 39 | 8.28% | 47.56% | 0.131 | moderate shift | 1 |
| 2025-01 | 465 | 42 | 9.03% | 43.30% | 0.135 | moderate shift | 1 |
| 2025-02 | 475 | 33 | 6.95% | 41.77% | 0.172 | moderate shift | 1 |
| 2025-03 | 480 | 31 | 6.46% | 41.89% | 0.197 | moderate shift | 1 |
| 2025-04 | 497 | 52 | 10.46% | 49.52% | 0.174 | moderate shift | 1 |
| 2025-05 | 509 | 40 | 7.86% | 48.78% | 0.204 | moderate shift | 1 |
| 2025-06 | 512 | 43 | 8.40% | 54.43% | 0.244 | moderate shift | 1 |
| 2025-07 | 538 | 44 | 8.18% | 54.32% | 0.258 | significant shift | 1 |
| 2025-08 | 580 | 47 | 8.10% | 52.81% | 0.293 | significant shift | 1 |
| 2025-09 | 552 | 40 | 7.25% | 47.06% | 0.281 | significant shift | 1 |
| 2025-10 | 624 | 64 | 10.26% | 55.17% | 0.326 | significant shift | 1 |
| 2025-11 | 614 | 53 | 8.63% | 49.53% | 0.333 | significant shift | 1 |
| 2025-12 | 627 | 40 | 6.38% | 45.98% | 0.355 | significant shift | 1 |

## 9. Champion vs challenger

| rule | alerts | tp | fn | precision | recall |
| --- | --- | --- | --- | --- | --- |
| TM-014 incumbent | 7,219 | 518 | 754 | 7.18% | 40.72% |
| TM-014 re-tuned | 6,271 | 462 | 810 | 7.37% | 36.32% |
| TM-014 + risk overlay | 6,764 | 545 | 727 | 8.06% | 42.85% |


Netted metrics conceal reallocation, so the overlap is broken out below. The 'Champion only' row is risk the challenger would newly miss.


| Cell | Records | True cases | Share of all cases | Precision |
| --- | --- | --- | --- | --- |
| Both alert | 6,271 | 462 | 36.32% | 7.37% |
| Champion only | 0 | 0 | 0.00% | n/a |
| Challenger only | 493 | 83 | 6.53% | 16.84% |
| Neither alerts | 38,220 | 727 | 57.15% | 1.90% |

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
