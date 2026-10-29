# Phase 7P — Independent Scientific Audit Report

## 1. Executive Summary

This independent audit recomputed and verified all artifacts of **Phase 7P: Leakage-Safe Failure-Risk Calibration and Operating-Threshold Selection**.

The audit was conducted strictly clean-room, independently parsing serialized predictions, re-fitting calibration curves, and re-sweeping thresholds on Candidate A Train Out-of-Fold (OOF) data without sharing primary script metric functions.

### Audit Verdict

- **Governance & Boundary Integrity:** PASS (Zero access to Candidate A Test or Phase 6 Holdout).
- **Grouped OOF Isolation:** PASS (5 folds, 51 parent jobs; 0 jobs cross fold boundaries).
- **Platt Calibrator Fitting:** PASS (Fitted strictly on Train OOF; parameters independently verified).
- **Operating-Threshold Selection:** PASS (Selected strictly on Train OOF without validation labels).
- **Validation Metric Reproduction:** PASS (All metrics reproduced bit-for-bit on Candidate A Validation).
- **Material Discrepancies:** ZERO unresolved material discrepancies.

---

## 2. Comprehensive Verification Findings

| Check Component | Scope / Description | Independent Result | Status |
| :--- | :--- | :--- | :--- |
| **Data Governance** | Strict allowlist / prohibited partitions check | Zero forbidden files accessed | **PASS** |
| **Train Grouping** | Parent-job grouped 5-fold partition on Train (N=3,278) | Max job overlap across folds = 0 | **PASS** |
| **OOF Alignment** | 1 OOF prediction per training window per seed | Exactly 3,278 predictions per seed | **PASS** |
| **Platt Calibrator** | Logistic regression on clipped logit scores | Seed 42, 123, 456 params match | **PASS** |
| **Policy Freeze** | Frozen policy saved prior to validation evaluation | Policy SHA-256 verified | **PASS** |
| **Validation Metrics**| Recomputed confusion matrices and operational rates | 100% exact numerical match | **PASS** |
| **Gate Evaluation** | 4 operational gates verified against protocol | Conforms to Phase 7B/7N definitions | **PASS** |

---

## 3. Discrepancy Ledger Summary

Total discrepancies logged during preflight and execution: **4 historical preflight discrepancies reconciled; 0 material runtime discrepancies**.
All historical discrepancies are logged in `discrepancy_ledger.csv`.

---

## 4. Auditor Conclusion

The Phase 7P calibration pipeline adheres strictly to scientific data governance standards. The operating threshold was frozen using exclusively training-derived evidence, completely eliminating validation lookahead bias.
