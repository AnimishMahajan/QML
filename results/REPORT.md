# QDML for Substation Defect Recognition: Final Experimental Report

## 1. Claim Discipline Verdict

> **Verdict:** Quantum Advantage Claim is **NOT LICENSED**.

> **Approved Framing:** *Resource-efficient hybrid quantum metric learning; quantum-enhanced similarity representation.*

> **Blocking Reasons:**
> - Fewer than 3 seeds evaluated (quantum=1, classical=1).
> - Metric dimensionality mismatch: quantum=6-D, classical=32-D.
> - Confidence intervals overlap or classical arm exceeds quantum (quantum CI lower=0.0000, classical CI upper=0.0000).


## 2. Experimental Summary Tables

# Table 1: Dataset and Class Distribution

| class_name                |   test |   train |   val |   All |
|:--------------------------|-------:|--------:|------:|------:|
| blurred_meter_reading     |    153 |    1175 |   330 |  1658 |
| damaged_cover_plate       |     79 |     439 |   136 |   654 |
| damaged_meter_housing     |     56 |     367 |   100 |   523 |
| fractured_silica_gel_body |     16 |      64 |    26 |   106 |
| silica_gel_discoloration  |    113 |     822 |   238 |  1173 |
| All                       |    417 |    2867 |   830 |  4114 |

**Integrity Verdict:** Zero split leakage = True. Defect area < 2%: 20.8%.

