# Phase 7Q — Independent Scientific Audit Report

## 1. Executive Summary

This independent clean-room audit evaluated **Phase 7Q: Cross-Job Predictive Signal Audit and Causal Workload-Relative Features**.

The audit was executed without importing primary pipeline functions, independently verifying fold partitioning, scale-floor estimation, Workstream A prevalence baselines, and recomputing all Train OOF and Candidate A Validation metrics directly from serialized predictions.

### Key Audit Findings

1. **Governance & Partition Integrity:** PASS. Zero access to Candidate A Test or Phase 6 Holdout.
2. **Balanced Grouped Partition:** PASS. Candidate A Train was partitioned into 5 folds with exactly 10–11 parent jobs per fold. Zero parent jobs cross fold boundaries.
3. **Prevalence Baseline Disconfirmation:** PASS. The Phase 7P calibrated Brier score ($0.125955$) was independently recomputed and confirmed to be virtually identical to a static constant-prevalence baseline ($0.125844$). The apparent calibration improvement in Phase 7P was an artifact of shrinking probabilities to the cohort base rate ($\sim 14.7\%$).
4. **Ranking Inversion Mechanism:** PASS. Confirmed that Phase 7P's negative Platt slopes ($a < 0$) mathematically inverted the ranking of distinct raw failure scores on Validation.
5. **OOF and Validation Metrics Reproduction:** PASS. 100% exact numerical match across all ranking, discrimination, operational, and confusion matrix metrics for both Q0 (matched baseline) and Q1 (causal deviation treatment).

---

## 2. Quantitative Verification Matrix

| Check Component | Scope / Description | Independent Result | Status |
| :--- | :--- | :--- | :---: |
| **Data Governance** | Strict allowlist / prohibited partitions check | Zero forbidden files accessed | **PASS** |
| **Fold Job Counts** | Target 10-11 parent jobs per fold | Folds: [10, 10, 10, 11, 10] | **PASS** |
| **Grouped Isolation** | Parent job overlap across training/val folds | Overlap = 0 jobs | **PASS** |
| **Scale Floors** | Causal relative deltas floor derivation | Strictly Candidate A Train derived | **PASS** |
| **Prevalence Baseline**| Constant prevalence Brier score | Exact match (0.125844) | **PASS** |
| **OOF Ranking Metrics**| Q0/Q1 ROC-AUC, AP, Spearman, Pearson | 100% exact numerical match | **PASS** |
| **Validation Gates** | 4 operational gates under fixed policy | 100% exact numerical match | **PASS** |

---

## 3. Auditor Conclusion

Phase 7Q execution satisfies all rigorous experimental controls and governance constraints. The audit confirms that Workstream A's finding (`RANKING_SIGNAL_WEAK_OR_INCONSISTENT`) is mathematically and empirically sound, and all Workstream B model comparisons are fair, matched, and leakage-safe.
