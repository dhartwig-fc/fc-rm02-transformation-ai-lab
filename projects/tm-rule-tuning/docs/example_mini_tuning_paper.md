<!--
  Reference copy of the Week 9 mini tuning paper -- the five-section form the
  specification asks for. GENERATED; regenerate with:

      python weeks/week09_tuning_papers.py
-->

# Mini Tuning Paper: TM-014 High Value Outbound Wires

**Date:** 2026-09-14  
**Author:** Dan Hartwig  
**Status:** Draft for independent challenge

## Background

Current rule:

```
ALERT IF monthly_outbound_wire_value > 50,000
  scope: all active customers
  frequency: monthly, run on the 1st for the preceding calendar month
```

The rule currently generates 4,309 alerts over the sample period at 11.77% precision and 42.04% recall, requiring 8.5 alerts of investigator effort per true case identified.

## Method

Retrospective backtest over 40,000 scored records containing 1,206 true cases (3.02% base rate).

- Label basis: Confirmed SAR/STR submission within 90 days of the alert period

- Period covered: 2025-01 to 2025-12 (12 periods)

- Candidate thresholds: 40 values from £1,293 to £380,705
- The rule was re-executed at each candidate and scored against the label set.

- Operational constraint: 2,500 alerts per period.

- Objective: maximise recall subject to that constraint.

- Out-of-time validation on periods held back from tuning.


## Results

| Threshold (£) | Alerts | True cases found | Missed | Precision | Recall |
| --- | --- | --- | --- | --- | --- |
| 1,293 | 37,999 | 1,192 | 14 | 3.14% | 98.84% |
| 2,476 | 34,204 | 1,163 | 43 | 3.40% | 96.43% |
| 3,608 | 30,408 | 1,130 | 76 | 3.72% | 93.70% |
| 4,731 | 26,612 | 1,098 | 108 | 4.13% | 91.04% |
| 6,040 | 22,815 | 1,070 | 136 | 4.69% | 88.72% |
| 7,645 | 19,020 | 1,042 | 164 | 5.48% | 86.40% |
| 10,067 | 15,024 | 993 | 213 | 6.61% | 82.34% |
| 14,788 | 11,029 | 911 | 295 | 8.26% | 75.54% |
| 28,672 | 7,033 | 705 | 501 | 10.02% | 58.46% |
| 68,187 | 3,037 | 397 | 809 | 13.07% | 32.92% |


## Recommendation

Move the threshold from **£50,000** to **£94,771**.

- Alert volume: 4,309 -> 2,038 (-2,271, a 52.7% reduction)
- True cases detected: 507 -> 277 (-230)
- Precision: 11.77% -> 13.59% (+1.83pp)
- Recall: 42.04% -> 22.97% (-19.07pp)
- Investigator effort per true case: 8.5 -> 7.4 alerts


## Limitations

- Labels are a proxy for true financial crime. A record with no SAR/STR is not proven clean -- it may be undetected risk. Measured recall is therefore optimistic, and every figure above is conditional on the label set.

- SAR labels are systematically absent below the line, because a SAR can only be filed on a case someone investigated, and someone only investigates what alerted. The bias has a known direction: it overstates the rule's coverage.

- Backtesting assumes historical behaviour is representative of the forward period. Any known upcoming change in product, customer mix or typology invalidates that assumption and should be raised before implementation.

- Coverage overlap with TM-009 (rapid movement of funds) has not been assessed; some cases counted here may also be detected by that rule.
