<!--
  Reference copy of the Week 10 capstone output, committed so the expected
  deliverable can be read without running anything. It is GENERATED -- do not
  edit it by hand. Regenerate with:

      python weeks/week10_capstone.py

  Every figure comes from a synthetic population; the numbers illustrate the
  shape of the argument, not any real portfolio.
-->

# Tuning Paper: TM-021 Cash Deposit Structuring

**Date:** 2026-09-14  
**Author:** Dan Hartwig  
**Status:** Draft for independent validation

## 1. Recommendation

Move the threshold from **£10,000** to **£18,503**.

- Alert volume: 11,537 -> 6,271 (-5,266, a 45.6% reduction)
- True cases detected: 857 -> 593 (-264)
- Precision: 7.43% -> 9.46% (+2.03pp)
- Recall: 68.40% -> 47.33% (-21.07pp)
- Investigator effort per true case: 13.5 -> 10.6 alerts

## 2. Rule under review

```
ALERT IF cash_deposits_30d > 10,000
  proposed: cash_deposits_30d > 18,503
  scope: all active customers | frequency: monthly, rolling 30 days
```

## 3. Data and sample

- Observations: 44,984 scored records
- True cases in sample: 1,253 (2.79% base rate)
- Label basis: Confirmed SAR/STR submission within 90 days of the alert period

- Period covered: 2024-01 to 2025-06 (18 periods)

## 4. Methodology

Retrospective backtest over the sample above. The rule was re-executed at each candidate threshold and scored against the label set, producing the sweep in section 5. Threshold selection was constrained by operational alert capacity rather than chosen to maximise a single composite metric, so that the detection/effort trade-off is made explicitly rather than implied by a scoring function.

## 5. Threshold sweep

| Threshold (£) | Alerts | True cases found | Missed | Precision | Recall | Alerts per case |
| --- | --- | --- | --- | --- | --- | --- |
| 791 | 42,734 | 1,245 | 8 | 2.91% | 99.36% | 34.3 |
| 1,274 | 40,173 | 1,237 | 16 | 3.08% | 98.72% | 32.5 |
| 1,709 | 37,612 | 1,221 | 32 | 3.25% | 97.45% | 30.8 |
| 2,115 | 35,050 | 1,210 | 43 | 3.45% | 96.57% | 29.0 |
| 2,536 | 32,489 | 1,193 | 60 | 3.67% | 95.21% | 27.2 |
| 2,991 | 29,928 | 1,168 | 85 | 3.90% | 93.22% | 25.6 |
| 3,488 | 27,366 | 1,143 | 110 | 4.18% | 91.22% | 23.9 |
| 4,009 | 24,805 | 1,121 | 132 | 4.52% | 89.47% | 22.1 |
| 4,621 | 22,244 | 1,087 | 166 | 4.89% | 86.75% | 20.5 |
| 5,431 | 19,611 | 1,056 | 197 | 5.38% | 84.28% | 18.6 |
| 6,390 | 16,943 | 1,002 | 251 | 5.91% | 79.97% | 16.9 |
| 7,802 | 14,275 | 944 | 309 | 6.61% | 75.34% | 15.1 |
| 9,942 | 11,607 | 858 | 395 | 7.39% | 68.48% | 13.5 |
| 13,344 | 8,939 | 746 | 507 | 8.35% | 59.54% | 12.0 |
| 18,503 | 6,271 | 593 | 660 | 9.46% | 47.33% | 10.6 |
| 26,905 | 3,603 | 404 | 849 | 11.21% | 32.24% | 8.9 |
| 50,884 | 935 | 151 | 1,102 | 16.15% | 12.05% | 6.2 |

## 6. Below-the-line testing

A simple random sample of 753 records was drawn from the 38,713 records the rule does not alert on. 11 were true cases.

- Observed below-the-line productive rate: **1.46%**
- 95% Wilson confidence interval: 0.82% to 2.60%
- Estimated cases missed across the full below-the-line population: **566** (upper bound 1,005)

The upper bound is the figure to test against risk appetite: it is the worst case the sample is consistent with, not the best guess.

## 7. Segment calibration

| Segment | Global £ | Alerts | Cases | Calibrated £ | Alerts | Cases | Uplift |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CORPORATE | 18,449 | 922 | 117 | 6,812 | 3,126 | 299 | +182 |
| PRIVATE | 18,449 | 1,069 | 179 | 9,770 | 1,196 | 196 | +17 |
| RETAIL | 18,449 | 21 | 0 | 17,498 | 30 | 2 | +2 |
| SME | 18,449 | 4,286 | 299 | 29,828 | 1,946 | 169 | -130 |
| TOTAL | 18,449 | 6,298 | 595 | n/a | 6,298 | 666 | +71 |

Both options are costed at the same total alert budget, so the uplift column is additional detection for no additional investigator effort.

## 8. Stability and drift

Performance was recomputed independently in each period. 20 of 24 periods breached at least one control limit.


| Period | Alerts | Cases | Precision | Recall | PSI | Stability | Breach |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2024-01 | 465 | 35 | 7.53% | 63.64% | 0.000 | stable | 1 |
| 2024-02 | 480 | 34 | 7.08% | 72.34% | 0.003 | stable | 0 |
| 2024-03 | 519 | 39 | 7.51% | 70.91% | 0.010 | stable | 0 |
| 2024-04 | 546 | 49 | 8.97% | 72.06% | 0.013 | stable | 0 |
| 2024-05 | 529 | 38 | 7.18% | 61.29% | 0.017 | stable | 0 |
| 2024-06 | 576 | 46 | 7.99% | 71.88% | 0.036 | stable | 1 |
| 2024-07 | 598 | 42 | 7.02% | 66.67% | 0.056 | stable | 1 |
| 2024-08 | 583 | 41 | 7.03% | 56.94% | 0.049 | stable | 1 |
| 2024-09 | 601 | 48 | 7.99% | 66.67% | 0.051 | stable | 1 |
| 2024-10 | 624 | 48 | 7.69% | 72.73% | 0.081 | stable | 1 |
| 2024-11 | 674 | 43 | 6.38% | 70.49% | 0.110 | moderate shift | 1 |
| 2024-12 | 720 | 71 | 9.86% | 78.02% | 0.110 | moderate shift | 1 |
| 2025-01 | 743 | 45 | 6.06% | 65.22% | 0.126 | moderate shift | 1 |
| 2025-02 | 731 | 44 | 6.02% | 64.71% | 0.153 | moderate shift | 1 |
| 2025-03 | 793 | 57 | 7.19% | 70.37% | 0.170 | moderate shift | 1 |
| 2025-04 | 779 | 65 | 8.34% | 68.42% | 0.211 | moderate shift | 1 |
| 2025-05 | 774 | 51 | 6.59% | 68.00% | 0.190 | moderate shift | 1 |
| 2025-06 | 802 | 61 | 7.61% | 68.54% | 0.199 | moderate shift | 1 |
| 2025-07 | 806 | 61 | 7.57% | 71.76% | 0.200 | moderate shift | 1 |
| 2025-08 | 834 | 54 | 6.47% | 70.13% | 0.246 | moderate shift | 1 |
| 2025-09 | 861 | 69 | 8.01% | 79.31% | 0.266 | significant shift | 1 |
| 2025-10 | 883 | 63 | 7.13% | 67.74% | 0.315 | significant shift | 1 |
| 2025-11 | 965 | 75 | 7.77% | 75.00% | 0.350 | significant shift | 1 |
| 2025-12 | 925 | 90 | 9.73% | 79.65% | 0.338 | significant shift | 1 |

## 9. Champion vs challenger

| rule | alerts | tp | fn | precision | recall |
| --- | --- | --- | --- | --- | --- |
| TM-021 incumbent | 11,537 | 857 | 396 | 7.43% | 68.40% |
| TM-021 re-tuned | 6,271 | 593 | 660 | 9.46% | 47.33% |
| TM-021 + risk overlay | 6,875 | 681 | 572 | 9.91% | 54.35% |


Netted metrics conceal reallocation, so the overlap is broken out below. The 'Champion only' row is risk the challenger would newly miss.


| Cell | Records | True cases | Share of all cases | Precision |
| --- | --- | --- | --- | --- |
| Both alert | 6,271 | 593 | 47.33% | 9.46% |
| Champion only | 0 | 0 | 0.00% | n/a |
| Challenger only | 604 | 88 | 7.02% | 14.57% |
| Neither alerts | 38,109 | 572 | 45.65% | 1.50% |

## 10. Limitations and assumptions

- Labels are a proxy for true financial crime. A record with no SAR/STR is not proven clean -- it may be undetected risk, which biases measured recall upward. Every figure in this paper is conditional on the label set.

- Backtesting assumes historical behaviour is representative of the forward period. Any known upcoming change in product, customer mix or typology invalidates that assumption and should be raised before implementation.

- The population is drifting (PSI 0.34 in the final period). The recommended threshold fits capacity in the tuning window but is projected to exceed it within the holdout, so it should be treated as valid for two quarters and re-assessed, not set annually.

- Segment data quality has not been assessed. Segment-specific thresholds make segment assignment an AML control, and that dependency must be confirmed before recommendation 2 is implemented.

- The risk-overlay challenger was not recommended because field coverage for jurisdiction and PEP flags is unverified, not because it underperformed.


## 11. Approval

| Role | Name | Date | Decision |
| --- | --- | --- | --- |
| Tuning analyst | | | |
| Financial crime risk owner | | | |
| Independent model validation | | | |
