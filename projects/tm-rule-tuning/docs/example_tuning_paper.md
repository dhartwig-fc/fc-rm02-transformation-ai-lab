<!--
  Reference copy of the Week 10 capstone tuning paper.
  GENERATED -- do not edit by hand. Regenerate by running the week script.

  Figures come from a synthetic population; they illustrate the shape of the
  argument, not any real portfolio.
-->

# Mini Tuning Paper: TM-021 Cash Deposit Structuring

**Date:** 2026-09-14  
**Author:** Dan Hartwig  
**Status:** Draft for independent challenge

## Background

Current rule:

```
ALERT IF cash_deposits_30d > 10,000
  proposed: Cash > £20,000 (all segments) AND velocity > 0
```

The rule currently generates 26,897 alerts over the sample period at 7.71% precision and 69.93% recall, requiring 13.0 alerts of investigator effort per true case identified.

## Method

Retrospective backtest over 100,000 scored records containing 2,966 true cases (2.97% base rate).

- Label basis: Confirmed SAR/STR submission within 90 days of the alert period

- Period covered: 2025-01 to 2025-12 (12 periods)

- Candidate thresholds: 5 values from £10,000 to £30,000
- The rule was re-executed at each candidate and scored against the label set.

- Fixed investigation capacity: 1,500 alerts/month allocated to this rule.

- Objective: maximise recall subject to that constraint, with a required volume reduction.

- Stability tested independently across Q1-Q4.


## Results

| Threshold (£) | Alerts | True cases found | Missed | Precision | Recall |
| --- | --- | --- | --- | --- | --- |
| 10,000 | 26,897 | 2,074 | 892 | 7.71% | 69.93% |
| 15,000 | 18,726 | 1,685 | 1,281 | 9.00% | 56.81% |
| 20,000 | 13,523 | 1,344 | 1,622 | 9.94% | 45.31% |
| 25,000 | 9,822 | 1,087 | 1,879 | 11.07% | 36.65% |
| 30,000 | 7,251 | 859 | 2,107 | 11.85% | 28.96% |


## Recommendation

Move the threshold from **£10,000** to **£20,000**.

- Alert volume: 26,897 -> 12,766 (-14,131, a 52.5% reduction)
- True cases detected: 2,074 -> 1,303 (-771)
- Precision: 7.71% -> 10.21% (+2.50pp)
- Recall: 69.93% -> 43.93% (-25.99pp)
- Investigator effort per true case: 13.0 -> 9.8 alerts


## Limitations

- Labels are a proxy for true financial crime. A record with no SAR/STR is not proven clean -- it may be undetected risk. Measured recall is therefore optimistic, and every figure above is conditional on the label set.

- SAR labels are systematically absent below the line, because a SAR can only be filed on a case someone investigated, and someone only investigates what alerted. The bias has a known direction: it overstates the rule's coverage.

- Backtesting assumes historical behaviour is representative of the forward period. Any known upcoming change in product, customer mix or typology invalidates that assumption and should be raised before implementation.

- A null velocity value fails the AND condition closed, silently removing that customer from monitoring. Field coverage must be confirmed before implementation.

- Segment-specific thresholds make segment assignment an AML control.

- Volume grew across all four quarters; the calibration is sized against Q4 and should be re-assessed within two quarters rather than annually.
