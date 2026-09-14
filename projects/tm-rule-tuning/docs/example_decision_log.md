<!--
  Reference copy of the Week 10 tuning decision log -- the specification's recommended habit.
  GENERATED -- do not edit by hand. Regenerate by running the week script.

  Figures come from a synthetic population; they illustrate the shape of the
  argument, not any real portfolio.
-->

# Tuning Decision Log: TM-021 Cash Deposit Structuring

**Date:** 2026-09-14  
**Author:** Dan Hartwig  
**Options tested:** 16

Every option considered during this exercise, including those rejected.
Recorded as the work was done, not reconstructed afterwards.


## 1. Baseline

| Option tested | Alerts | Precision | Recall | FPR | Outcome |
| --- | --- | --- | --- | --- | --- |
| cash > £10,000 (incumbent) | 26,897 | 7.71% | 69.93% | 25.58% | for information |


- **cash > £10,000 (incumbent)**
  - _Operational implication:_ 2,241 alerts/month against a 1,500 allocation -- 149% of capacity.
  - _Assumptions:_ Case label = SAR/STR filed within 90 days of the alert period.
  - _Rationale:_ Baseline as currently running. Breaches capacity; volume reduction required.


## 2. Threshold sweep

| Option tested | Alerts | Precision | Recall | FPR | Outcome |
| --- | --- | --- | --- | --- | --- |
| cash > £10,000 | 26,897 | 7.71% | 69.93% | 25.58% | rejected |
| cash > £15,000 | 18,726 | 9.00% | 56.81% | 17.56% | rejected |
| cash > £20,000 | 13,523 | 9.94% | 45.31% | 12.55% | carried forward |
| cash > £25,000 | 9,822 | 11.07% | 36.65% | 9.00% | carried forward |
| cash > £30,000 | 7,251 | 11.85% | 28.96% | 6.59% | carried forward |


- **cash > £10,000**
  - _Operational implication:_ 2,241 alerts/month (149% of allocation).
  - _Rationale:_ Exceeds the fixed investigation capacity.

- **cash > £15,000**
  - _Operational implication:_ 1,560 alerts/month (104% of allocation).
  - _Rationale:_ Exceeds the fixed investigation capacity.

- **cash > £20,000**
  - _Operational implication:_ 1,127 alerts/month (75% of allocation).
  - _Rationale:_ Fits the allocation.

- **cash > £25,000**
  - _Operational implication:_ 818 alerts/month (55% of allocation).
  - _Rationale:_ Fits the allocation.

- **cash > £30,000**
  - _Operational implication:_ 604 alerts/month (40% of allocation).
  - _Rationale:_ Fits the allocation.


## 3. Segment calibration

| Option tested | Alerts | Precision | Recall | FPR | Outcome |
| --- | --- | --- | --- | --- | --- |
| Global £20,000 | 13,523 | 9.94% | 45.31% | 12.55% | carried forward |
| Retail £10,000 / Corporate £25,000 | 12,001 | 9.27% | 37.49% | 11.22% | carried forward |
| Retail £15,000 / Corporate £35,000 | 5,640 | 12.71% | 24.17% | 5.07% | carried forward |
| Retail £20,000 / Corporate £40,000 | 4,069 | 14.77% | 20.26% | 3.57% | carried forward |


- **Global £20,000**
  - _Operational implication:_ 1,127 alerts/month.
  - _Assumptions:_ Segment assignment is accurate and stable; it becomes an AML control.
  - _Rationale:_ Within capacity.

- **Retail £10,000 / Corporate £25,000**
  - _Operational implication:_ 1,000 alerts/month.
  - _Assumptions:_ Segment assignment is accurate and stable; it becomes an AML control.
  - _Rationale:_ Within capacity.

- **Retail £15,000 / Corporate £35,000**
  - _Operational implication:_ 470 alerts/month.
  - _Assumptions:_ Segment assignment is accurate and stable; it becomes an AML control.
  - _Rationale:_ Within capacity.

- **Retail £20,000 / Corporate £40,000**
  - _Operational implication:_ 339 alerts/month.
  - _Assumptions:_ Segment assignment is accurate and stable; it becomes an AML control.
  - _Rationale:_ Within capacity.


## 4. Challenger design

| Option tested | Alerts | Precision | Recall | FPR | Outcome |
| --- | --- | --- | --- | --- | --- |
| Retail £15,000 / Corporate £35,000 (no velocity condition) | 5,640 | 12.71% | 24.17% | 5.07% | carried forward |
| Retail £15,000 / Corporate £35,000 AND velocity > 0 | 5,239 | 13.17% | 23.26% | 4.69% | carried forward |
| Retail £15,000 / Corporate £35,000 AND velocity > 2 | 2,907 | 15.55% | 15.24% | 2.53% | carried forward |
| Retail £15,000 / Corporate £35,000 AND velocity > 3 | 1,720 | 18.14% | 10.52% | 1.45% | carried forward |


- **Retail £15,000 / Corporate £35,000 (no velocity condition)**
  - _Operational implication:_ 470 alerts/month.
  - _Assumptions:_ Velocity is populated for every customer in the production engine.
  - _Rationale:_ Within capacity.

- **Retail £15,000 / Corporate £35,000 AND velocity > 0**
  - _Operational implication:_ 437 alerts/month.
  - _Assumptions:_ Velocity is populated for every customer in the production engine.
  - _Rationale:_ Within capacity.

- **Retail £15,000 / Corporate £35,000 AND velocity > 2**
  - _Operational implication:_ 242 alerts/month.
  - _Assumptions:_ Velocity is populated for every customer in the production engine.
  - _Rationale:_ Within capacity.

- **Retail £15,000 / Corporate £35,000 AND velocity > 3**
  - _Operational implication:_ 143 alerts/month.
  - _Assumptions:_ Velocity is populated for every customer in the production engine.
  - _Rationale:_ Within capacity.


## 6. Recommendation

| Option tested | Alerts | Precision | Recall | FPR | Outcome |
| --- | --- | --- | --- | --- | --- |
| SPEC EXAMPLE: Retail £15,000 / Corporate £35,000 AND velocity > 3 | 1,720 | 18.14% | 10.52% | 1.45% | rejected |
| ADOPTED: Cash > £20,000 (all segments) AND velocity > 0 | 12,766 | 10.21% | 43.93% | 11.81% | adopted |


- **SPEC EXAMPLE: Retail £15,000 / Corporate £35,000 AND velocity > 3**
  - _Operational implication:_ 143 alerts/month -- only 10% of the allocation, leaving 1,357/month of paid capacity idle.
  - _Assumptions:_ Velocity populated for all customers; segment assignment stable.
  - _Rationale:_ Over-corrects. Delivers the volume reduction but gives up 59pp of recall and leaves most of the investigation capacity unused. Fails the missed-risk constraint.

- **ADOPTED: Cash > £20,000 (all segments) AND velocity > 0**
  - _Operational implication:_ 1,064 alerts/month (71% of allocation); Q4 run-rate 1,449/month.
  - _Assumptions:_ Velocity populated for all customers; SAR-based labels; population continues drifting at the observed rate.
  - _Rationale:_ Delivers the required volume reduction while spending the allocated capacity rather than undershooting it. Recovers 33pp of recall against the example calibration. Recall still falls against the incumbent; that cost is quantified and put to the risk owner explicitly.
