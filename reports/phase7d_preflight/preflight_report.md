# Phase 7D Preflight — Scaler Methodology and Diagnostic Protocol Report

**Date:** October 28, 2026  
**Analyst / Role:** Senior ML Research Engineer & Data-Pipeline Auditor  
**Governing Frozen Split:** Candidate A ([results/phase7a_4/candidate_a_job_grouped_split.csv](file:///c:/Users/hp/Desktop/Research%20project/results/phase7a_4/candidate_a_job_grouped_split.csv))  
**Governing Evaluation Protocol:** Protocol v2.0 ([results/phase7b_1/phase7b_evaluation_protocol_v2.json](file:///c:/Users/hp/Desktop/Research%20project/results/phase7b_1/phase7b_evaluation_protocol_v2.json))  
**Audited Input Dataset:** [`data/processed/phase7c/`](file:///c:/Users/hp/Desktop/Research%20project/data/processed/phase7c/)  
**Audited Preprocessing Scaler:** [`configs/phase7c/fitted_scaler_phase7c.json`](file:///c:/Users/hp/Desktop/Research%20project/configs/phase7c/fitted_scaler_phase7c.json)  
**Machine-Readable Preflight Ledger:** [`results/phase7d_preflight/preflight_ledger.json`](file:///c:/Users/hp/Desktop/Research%20project/results/phase7d_preflight/preflight_ledger.json)  

---

## 1. Executive Verdict & Readiness Decision

### **Verdict:** **`READY FOR TRAINING APPROVAL`**

A rigorous preflight investigation was conducted to prepare for the Phase 7D diagnostic experiment. The investigation resolved the scaler weighting question, locked the diagnostic hypothesis and experimental protocol, and verified the implementation readiness of all dataset tensors, labels, and script dependencies.

### Summary of Preflight Findings:
1. **Scaler Weighting Clarified:** Overlapping sliding windows weight interior physical observations by up to $5\times$ (mean multiplicity $3.713\times$). This weighting is **strictly train-internal with zero data leakage**. It aligns with sequence tensor optimization objectives and induces only minor parameter shifts (mean z-score shift of $0.032\sigma$). Retaining the frozen Phase 7C scaler is recommended for Phase 7D to preserve controlled comparability with Phase 7B.
2. **Diagnostic Protocol Locked:** Phase 7D is formally framed to test whether removing inherited preprocessing contamination resolves the operational failure of the baseline Random Forest model on Candidate A, or whether parent-job distribution shift is the primary culprit.
3. **Implementation Validated:** All 31 features, tensor dimensions, class distributions, and isolation boundaries are verified. Zero code paths touch the quarantined Phase 6 holdout.

---

## 2. Task 1 — Scaler Weighting and Population Analysis

### 2.1 Fitting Population Mechanics
In [scripts/prepare_phase7c_leakage_free_data.py](file:///c:/Users/hp/Desktop/Research%20project/scripts/prepare_phase7c_leakage_free_data.py) (lines 94–103), the scaler fitting population is formed by reshaping the raw training tensor:
$$\mathbf{X}_{\text{train\_raw}} \in \mathbb{R}^{3,278 \times 5 \times 31} \implies \mathbf{X}_{\text{flat}} \in \mathbb{R}^{16,390 \times 31}$$

Because sliding windows have length $L=5$ ($1,500\text{s}$) and advance by $1$ step ($300\text{s}$), physical telemetry steps are sampled repeatedly across consecutive windows:
- **Total window-step observations:** $16,390$ ($3,278$ windows $\times 5$ steps)
- **Unique physical `(entity_id, timestamp)` steps:** $4,414$
- **Mean step multiplicity:** **$3.713\times$**

#### Multiplicity Breakdown across Physical Steps:
| Multiplicity in Windows | Unique Physical Steps | Percentage of Steps | Observation Description |
| :---: | :---: | :---: | :--- |
| **Appears 1x** | 622 | 14.09% | Boundary steps (task start or end), or tasks with only 1 window |
| **Appears 2x** | 552 | 12.51% | Near-boundary steps (second from start/end) |
| **Appears 3x** | 528 | 11.96% | Intermediate steps (third from start/end) |
| **Appears 4x** | 480 | 10.87% | Intermediate steps (fourth from start/end) |
| **Appears 5x** | 2,232 | **50.57%** | Fully interior steps in stable, long-running execution |
| **Total** | **4,414** | **100.00%** | All Candidate A physical training telemetry steps |

### 2.2 Parameter Comparison: Window-Weighted vs. Unique-Physical
Using Candidate A training data exclusively, scaler parameters were computed across both populations:

| Feature Name | Window Mean ($\mu_w$) | Unique Mean ($\mu_u$) | Mean Diff ($|\Delta \mu|$) | Window Std ($\sigma_w$) | Unique Std ($\sigma_u$) | Std Diff ($|\Delta \sigma|$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `cpu_utilization` | 0.02875 | 0.02828 | 0.00048 | 0.02352 | 0.02399 | 0.00047 |
| `memory_utilization` | 0.00980 | 0.00974 | 0.00006 | 0.01186 | 0.01186 | 0.00000 |
| `assigned_memory` | 0.01143 | 0.01139 | 0.00004 | 0.01302 | 0.01290 | 0.00012 |
| `disk_io_time` | 0.00119 | 0.00115 | 0.00004 | 0.00640 | 0.00617 | 0.00023 |
| `cycles_per_instruction` | 2.86359 | 2.82398 | 0.03961 | 2.38366 | 2.45926 | 0.07560 |
| `memory_accesses_per_instruction` | 0.00660 | 0.00676 | 0.00017 | 0.01012 | 0.01067 | 0.00055 |
| `cpu_utilization_lag_1` | 0.02863 | 0.02835 | 0.00028 | 0.02338 | 0.02399 | 0.00061 |
| `cpu_diff_1` | 0.00012 | -0.00007 | 0.00019 | 0.01120 | 0.01158 | 0.00038 |
| `cpu_rolling_mean_6` | 0.02817 | 0.02846 | 0.00029 | 0.02199 | 0.02190 | 0.00009 |
| `cpu_trend_slope_6` | 0.00028 | -0.00006 | 0.00034 | 0.00347 | 0.00405 | 0.00058 |
| `memory_to_assigned_ratio` | 0.79414 | 0.79148 | 0.00266 | 0.18610 | 0.18406 | 0.00203 |
| `task_age_minutes` | 101.28123 | 98.63573 | 2.64551 | 28.52665 | 32.19901 | 3.67236 |

#### Summary of Parameter Differences:
- **Mean Absolute Shift in Mean:** $0.10501$ (driven primarily by `task_age_minutes` which shifts by $2.65\text{ min}$).
- **Mean Absolute Shift in Std:** $0.71027$ (driven by high-ratio spikes in `cpu_rel_change_1`).
- **Mean Absolute Shift in Median:** $0.03569$.
- **Mean Standardized Z-Score Discrepancy:** $|z_w - z_u|$ across all 31 features is **$0.0321\sigma$**.

### 2.3 Methodological Assessment & Recommendation
1. **Absence of Leakage:** Both populations are derived exclusively from Candidate A Training data. Neither validation nor test data is touched.
2. **Alignment with Loss Function:** Neural sequence models (LSTM, Transformer) optimize cross-entropy loss over mini-batches of window slices. Flattened scaling guarantees that the inputs received during gradient updates have empirical mean 0.0 and unit variance. If unique-step scaling were used, window slices would have non-zero empirical means during training.
3. **Recommendation:** **Retain the frozen Phase 7C scaler for the primary Phase 7D diagnostic experiment.**
   - *Scientific Justification:* Phase 7D's goal is to diagnose the effect of removing legacy Phase 2 contamination. Retaining the sequence-tensor scaler isolates the impact of clean data reconstruction without introducing an unannounced preprocessing shift.
   - If an unweighted unique-physical scaler is investigated, it should be structured as an explicit ablation experiment in a subsequent phase.

---

## 3. Task 2 — Diagnostic Experiment Lock

### 3.1 Experimental Hypotheses
Phase 7D is locked to evaluate two competing hypotheses:

$$\begin{aligned}
\mathbf{H_1} \text{ (Contamination Causality)} &: \text{The operational failure of Random Forest in Phase 7B was primarily caused} \\
&\quad \text{by the legacy Phase 2 imputation constant (2.5) and overlapping parent-job scaling.} \\
&\quad \implies \text{Random Forest will pass or improve substantially toward operational gates on clean data.} \\[1ex]
\mathbf{H_0} \text{ (Distribution Shift Culprit)} &: \text{Random Forest fails operational gates due to parent-job workload distribution shift} \\
&\quad \text{across unseen cloud workloads, independent of preprocessing contamination.} \\
&\quad \implies \text{Random Forest will continue to fail operational gates on clean data.}
\end{aligned}$$

### 3.2 Locked Baseline Architecture & Preprocessing
- **Model Architecture:** Scikit-Learn `RandomForestClassifier`:
  ```python
  RandomForestClassifier(
      n_estimators=100,
      max_depth=15,
      class_weight="balanced",
      random_state=42,
      n_jobs=-1
  )
  ```
- **Input Dimension:** Tabular feature vector $x_{\text{last}} \in \mathbb{R}^{31}$ (the final step of the 5-step sequence tensor, $X[:, -1, :]$).
- **Allowed Preprocessing:** Strictly the clean, frozen Phase 7C dataset ([`data/processed/phase7c/`](file:///c:/Users/hp/Desktop/Research%20project/data/processed/phase7c/)) standardized by [`configs/phase7c/fitted_scaler_phase7c.json`](file:///c:/Users/hp/Desktop/Research%20project/configs/phase7c/fitted_scaler_phase7c.json).

### 3.3 Governing Protocol & Frozen Splits
- **Protocol:** Protocol v2.0 (`results/phase7b_1/phase7b_evaluation_protocol_v2.json`).
- **Frozen Split:** Candidate A (`results/phase7a_4/candidate_a_job_grouped_split.csv` | SHA-256: `f84bc4fe...`):
  - **Train:** 3,278 windows, 284 entities, 51 parent jobs
  - **Validation:** 657 windows, 69 entities, 12 parent jobs
  - **Test:** 1,025 windows, 59 entities, 11 parent jobs

### 3.4 Operational Gates & Candidate Selection Rules
- **Validation Gates (Stage 2):**
  - **Gate 1 (Recall):** Validation Episode Recall $\ge 70.0\%$ AND Validation Critical Window Recall $\ge 50.0\%$.
  - **Gate 2 (False Alarms):** Validation Pure-Normal Entity FAR $\le 10.0\%$ AND Validation Normal-Window FPR $\le 5.0\%$.
- **Selection & Freezing Rule:**
  - Candidate $M^*$ is chosen as the model satisfying both Validation Gate 1 and Gate 2 with the highest Validation Macro-F1.
  - If zero models satisfy both gates, the model with the highest Validation Macro-F1 is selected and permanently frozen as $M^*$ before Test set evaluation.
- **Test Gate Evaluation (Stage 3):**
  - Confirmatory evaluation of $M^*$ on Candidate A Test split against Test Gate 1 and Test Gate 2.
  - Official verdict: `OPERATIONAL_WINNER` or `NO_OPERATIONAL_WINNER`.
  - Non-selected models are reported strictly as descriptive benchmarks.

### 3.5 Diagnostic Decision Matrix

| Empirical Outcome in Phase 7D | Test Metrics Profile | Inferred Cause of Phase 7B Failure | Action / Next Step |
| :--- | :--- | :--- | :--- |
| **Outcome A: Complete Operational Gate Pass** | Episode Recall $\ge 70\%$, Critical Recall $\ge 50\%$, Entity FAR $\le 10\%$, Window FPR $\le 5\%$ | **Contamination Was Culprit ($H_1$ Supported):** Purging legacy $2.5$ and training-only scaler restored model calibration. | Declare Random Forest `OPERATIONAL_WINNER`. |
| **Outcome B: Persistent Operational Failure** | Entity FAR $\approx 40\text{--}50\%$, Critical Recall $\approx 15\text{--}25\%$, Episode Recall $< 50\%$ | **Distribution Shift is Culprit ($H_0$ Supported):** Preprocessing contamination was benign; Random Forest cannot generalize across parent jobs. | Confirm `NO_OPERATIONAL_WINNER`. Proceed to sequence architecture redesign. |
| **Outcome C: Partial Calibration Shift** | Entity FAR drops noticeably (e.g. $20\text{--}30\%$), Macro-F1 increases, but fails one gate | **Dual Contribution:** Preprocessing contamination degraded calibration, but parent-job shift remains significant. | Document exact margin of contamination effect; evaluate sequence models. |

### 3.6 Permitted Comparisons vs. Required Protocol Amendments
- **Permitted Without Amendment:**
  - Direct metric comparison between Phase 7B RF and Phase 7D RF on the exact same Candidate A split under identical seeds and gates.
  - Evaluation of the 3 descriptive baseline models (LSTM, Transformer, Hybrid) alongside RF.
- **Requires Formal Protocol Amendment:**
  - Relaxing Gate 1 or Gate 2 thresholds.
  - Modifying Random Forest hyperparameters (`max_depth`, `n_estimators`, `class_weight`).
  - Calibrating decision thresholds on Test data.
  - Switching to an unapproved split.

---

## 4. Task 3 — Implementation Readiness Checklist

Every operational dependency for Phase 7D was inspected and verified:

```
========================================================================================
                          PHASE 7D IMPLEMENTATION READINESS AUDIT
========================================================================================
Input Dataset Location              : data/processed/phase7c/ (Frozen)
Dataset Tensors Finite Integrity    : 100.0% Finite (0 NaNs, 0 Infs across all splits)
Input Tensor Dimensions             : Train: (3278, 5, 31) | Val: (657, 5, 31) | Test: (1025, 5, 31)
Target Vector Dimensions            : Train: (3278,)       | Val: (657,)       | Test: (1025,)
Aligned Metadata Rows               : Train: 3,278 rows    | Val: 657 rows     | Test: 1,025 rows
Class Distribution (Normal/Warn/Crit):
  - Train                           : 2,797 (85.33%) / 244 (7.44%) / 237 (7.23%)
  - Validation                      : 560 (85.24%) / 48 (7.31%) / 49 (7.46%)
  - Test                            : 921 (89.85%) / 49 (4.78%) / 55 (5.37%)
Active Preprocessing Scaler         : configs/phase7c/fitted_scaler_phase7c.json (Verified)
Quarantined Phase 6 Holdout Access  : ZERO ACCESS (Verified sealed)
Output Isolation Paths              : results/phase7d/ | reports/phase7d/ | models/checkpoints_phase7d/
========================================================================================
```

### Operational Plan for Phase 7D Execution:
To satisfy Research Safeguard 6 (preserving historical scripts unchanged), Phase 7D model training will be launched via a newly created, dedicated script:
`scripts/train_and_evaluate_phase7d.py`

This script will:
1. Ingest `data/processed/phase7c/` and verify `configs/phase7c/fitted_scaler_phase7c.json`.
2. Train all 4 models from scratch under seed 42.
3. Execute Stage 2 validation qualification and freeze $M^*$.
4. Execute Stage 3 one-shot confirmatory test evaluation.
5. Compute dual cluster bootstrap confidence intervals.
6. Serialize outputs to `results/phase7d/` and `reports/phase7d/`.

---

## 5. Final Decision & Gate Summary

### **Verdict:** **`READY FOR TRAINING APPROVAL`**

All preflight checks have concluded successfully:
- Scaler weighting mechanics have been rigorously quantified and documented.
- The diagnostic hypothesis and evaluation gates are locked.
- Clean, leakage-free dataset arrays in `data/processed/phase7c/` are verified and finite.
- Research safeguards remain 100% enforced.

The project is fully prepared for Phase 7D execution upon receiving your explicit authorization to begin model training.
