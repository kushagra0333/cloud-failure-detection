# Phase 3A.1 — Baseline Integrity Audit Before LSTM

**Project:** Temporal LSTM/Transformer Framework for Early Failure Prediction in Cloud Computing  
**Phase:** 3A.1 — Baseline Integrity Audit & Pre-LSTM Verification Gate  
**Execution Date:** 2026-10-26  
**Auditor:** Senior ML Engineer & Independent Experimental Auditor  
**Status:** **ALL CHECKS PASSED — APPROVED AS PHASE 3B BENCHMARK**

---

## Executive Summary

Phase 3A.1 independently audits the code, metadata, saved prediction files, and metrics from Phase 3A before training any deep learning models. 

Every claim, metric, and artifact was verified through programmatic recomputation from raw files and prediction CSVs. **No models were trained during this audit.**

### Summary of Audit Verdicts
- **Experimental Protocol & Isolation:** **PASS**. Model representation and hyperparameters were tuned and selected strictly on training and validation sets. The test set was evaluated exactly once after freezing.
- **Metric Recomputation:** **PASS**. All window-level metrics (Accuracy, Macro-F1, per-class P/R/F1, Confusion Matrices, and PR-AUC) recomputed from [`results/phase3a/test_predictions_*.csv`](file:///c:/Users/hp/Desktop/Research%20project/results/phase3a/) match published metrics with **0.00e+00 difference**.
- **Prediction Alignment:** **PASS**. All 1,003 test prediction rows match the primary test metadata row-for-row, timestamp-for-timestamp, and entity-for-entity.
- **Episode-Level Semantics:** **PASS**. Reconstructed failure timestamps match raw `FAIL` events with 0.000000s error. Lead time strictly measures the earliest pre-failure alert.
- **Denominator Transparency:** **PASS**. The pure-normal denominator ($N=48$) is explicitly documented, and supplementary window-level and all-entity false positive metrics are reported.
- **Phase 3B Gate Decision:** **GO FOR PHASE 3B (TEMPORAL LSTM)**.

---

## Part A: Verification of Baseline Experiment Integrity

### Criterion A.1: Training and Validation Isolation for Representation Selection
- **Audit Check:** Inspect [`scripts/phase3a_baselines.py`](file:///c:/Users/hp/Desktop/Research%20project/scripts/phase3a_baselines.py) to confirm where sequence-to-vector representations were chosen.
- **Finding:** In Step 3, the three representations (`final_step` $D=31$, `summary` $D=93$, `flattened` $D=465$) were fitted on `X_tr, y_tr` and evaluated exclusively on `X_val, y_val`. Random Forest achieved Val Macro-F1 of $0.7026$ on `final_step` vs $0.6870$ on `flattened`. The decision to freeze `final_step` was finalized in Step 3/4 before touching the test partition.
- **Verdict:** **PASS**

### Criterion A.2: Strict Test Partition Quarantine
- **Audit Check:** Confirm that test data was not used for tuning, threshold selection, or model selection.
- **Finding:** Step 5 evaluated validation data and froze all model instances. Step 6 loaded `X_te, y_te` and executed a single `predict` and `predict_proba` call on the frozen classifiers. No hyperparameter searches or threshold adjustments were executed on the test partition.
- **Verdict:** **PASS**

### Criterion A.3: Independent Recomputation of Published Metrics
- **Audit Check:** Recompute all window-level metrics directly from the raw test predictions ([`results/phase3a/test_predictions_majority.csv`](file:///c:/Users/hp/Desktop/Research%20project/results/phase3a/test_predictions_majority.csv), [`test_predictions_logistic_regression.csv`](file:///c:/Users/hp/Desktop/Research%20project/results/phase3a/test_predictions_logistic_regression.csv), [`test_predictions_random_forest.csv`](file:///c:/Users/hp/Desktop/Research%20project/results/phase3a/test_predictions_random_forest.csv)) and compare against [`results/phase3a/baseline_metrics.json`](file:///c:/Users/hp/Desktop/Research%20project/results/phase3a/baseline_metrics.json) and [`reports/phase3a_baseline_results.md`](file:///c:/Users/hp/Desktop/Research%20project/reports/phase3a_baseline_results.md).

| Model | Metric | Published Metric | Recomputed Metric | Absolute Difference | Match Status |
|:---|:---|:---:|:---:|:---:|:---:|
| **Majority Class** | Macro-F1 | 0.308516 | 0.308516 | 0.00e+00 | **MATCH** |
| | Accuracy | 0.861416 | 0.861416 | 0.00e+00 | **MATCH** |
| | Warning Recall | 0.000000 | 0.000000 | 0.00e+00 | **MATCH** |
| | Critical Recall | 0.000000 | 0.000000 | 0.00e+00 | **MATCH** |
| | Warning PR-AUC | 0.067797 | 0.067797 | 0.00e+00 | **MATCH** |
| | Critical PR-AUC | 0.070788 | 0.070788 | 0.00e+00 | **MATCH** |
| **Logistic Regression** | Macro-F1 | 0.414253 | 0.414253 | 0.00e+00 | **MATCH** |
| | Accuracy | 0.635095 | 0.635095 | 0.00e+00 | **MATCH** |
| | Warning Recall | 0.470588 | 0.470588 | 0.00e+00 | **MATCH** |
| | Critical Recall | 0.352113 | 0.352113 | 0.00e+00 | **MATCH** |
| | Warning PR-AUC | 0.137063 | 0.137063 | 0.00e+00 | **MATCH** |
| | Critical PR-AUC | 0.231568 | 0.231568 | 0.00e+00 | **MATCH** |
| **Random Forest** | **Macro-F1** | **0.571824** | **0.571824** | **0.00e+00** | **MATCH** |
| | Accuracy | 0.865404 | 0.865404 | 0.00e+00 | **MATCH** |
| | Warning Recall | 0.250000 | 0.250000 | 0.00e+00 | **MATCH** |
| | Critical Recall | 0.450704 | 0.450704 | 0.00e+00 | **MATCH** |
| | Warning PR-AUC | 0.370060 | 0.370060 | 0.00e+00 | **MATCH** |
| | Critical PR-AUC | 0.563728 | 0.563728 | 0.00e+00 | **MATCH** |

- **Verdict:** **PASS**

### Criterion A.4: Metadata and Prediction Row Alignment
- **Audit Check:** Verify that the saved prediction CSVs match [`data/processed/test_metadata.csv`](file:///c:/Users/hp/Desktop/Research%20project/data/processed/test_metadata.csv) row-by-row.
- **Finding:**
  - Length: Exactly 1,003 rows across all files.
  - Entity IDs: 100% exact match (`(df_pred["entity_id"] == test_meta["entity_id"]).all() == True`).
  - Prediction Timestamps: 100% exact match (`(df_pred["prediction_timestamp"] == test_meta["prediction_timestamp"]).all() == True`).
  - True Labels: 100% exact match (`(df_pred["label"] == test_meta["label"]).all() == True`).
- **Verdict:** **PASS**

### Criterion A.5: Class Order Consistency
- **Audit Check:** Verify that classes are consistently indexed across all modules.
- **Finding:**
  - `0`: `NORMAL`
  - `1`: `WARNING`
  - `2`: `CRITICAL`
  - Consistently defined in `src/ingestion/schema.py`, `src/labeling/target_generator.py`, `src/models/baselines.py`, `src/evaluation/metrics.py`, and all test assertion files.
- **Verdict:** **PASS**

### Criterion A.6: PR-AUC / Average Precision Mathematical Correctness
- **Audit Check:** Verify how One-vs-Rest Average Precision is calculated for WARNING and CRITICAL.
- **Finding:**
  - For Logistic Regression and Random Forest, `probs[:, c]` contains continuous predicted probabilities. Scikit-learn's `average_precision_score(y_true == c, probs[:, c])` computes the area under the interpolated precision-recall curve.
  - For Majority Class, predicted probabilities are constant ($0.0$ for minority classes). In [`src/evaluation/metrics.py`](file:///c:/Users/hp/Desktop/Research%20project/src/evaluation/metrics.py) lines 66–74, constant probabilities are detected and assigned the mathematically exact prevalence $\frac{N_c}{N}$ ($0.0678$ for Warning, $0.0708$ for Critical), setting flag `pr_auc_is_prevalence = True`.
- **Verdict:** **PASS**

### Criterion A.7: Undefined Metric Handling
- **Audit Check:** Verify that zero-division cases in precision/recall are not silently masked or corrupted.
- **Finding:** Scikit-learn functions `precision_recall_fscore_support` and `f1_score` explicitly pass `zero_division=0.0`. When the Majority Class predicts zero Warning or Critical windows, precision, recall, and F1 evaluate cleanly to $0.0$, correctly giving a Macro-F1 of $\frac{0.9255 + 0.0 + 0.0}{3} = 0.3085$.
- **Verdict:** **PASS**

### Criterion A.8: Experiment Configuration Consistency
- **Audit Check:** Verify that prediction files, reports, and JSON artifacts reference the identical experiment.
- **Finding:** Library versions (`scikit-learn 1.9.1`, `numpy 2.4.6`, `pandas 2.3.3`, Python `3.11.9`), seed (`42`), and hyperparameter states match identically across `results/phase3a/baseline_metrics.json` and `reports/phase3a_baseline_results.md`.
- **Verdict:** **PASS**

---

## Part B: Comprehensive Audit of Episode-Level Metrics

### 1. Timestamp Matching to Raw Source Events
- **Audit Check:** Verify whether `fail_timestamp = round(prediction_timestamp + time_to_failure, 1)` identifies true source event timestamps.
- **Empirical Proof:** All 139 pre-failure windows in the test set were compared against the raw Borg 2011 event shards (`data/raw/sample_task_events.csv.gz`). The maximum absolute discrepancy is **$0.000000\text{s}$**. Reconstructed failure timestamps are physically exact.

### 2. Window Grouping Mechanics
- **Grouping Rule:** Pre-failure windows (`label > 0`) are grouped by the composite key `(entity_id, fail_timestamp)`.
- **Finding:** In the Test set, 139 pre-failure windows group into **36 distinct failure episodes** across 32 unique tasks (average multiplicity = $3.86$ windows per episode).

### 3. Multi-Alert and Lead Time Semantics
- **Multi-Alert Handling:** A single failure episode spans multiple 5-minute sliding windows (typically 1–2 Warning windows followed by 1–2 Critical windows). If a model triggers alerts on multiple consecutive windows, the episode is counted as detected **at most once** (`detected_episodes += 1`). Duplicate alerts are treated as ongoing alert confirmations.
- **Earliest Lead Time:** Lead time is calculated from the earliest triggered alert:
  $$\text{Lead Time} = T_{\text{fail}} - t_{\text{earliest\_alert}}$$
  Because windows are sorted ascending by `prediction_timestamp`, `.iloc[0]` strictly selects the earliest alert.
- **Alerts After Failure Verification:** Can an alert occurring after failure count as detection?
  **Mathematically NO.** In `TargetLabelGenerator`, pre-failure windows are labeled `1` or `2` strictly when lookahead $\tau = T_{\text{fail}} - t_{\text{pred}} > 0$. Any observation after failure has $\tau \le 0$ and cannot point forward to that failure. An assertion check in our audit confirmed that $t_{\text{pred}} < T_{\text{fail}}$ for $100\%$ of pre-failure windows (min lead time $= 0.28\text{s} > 0$).

### 4. Disaggregated Warning vs. Critical Phase Detections
In the Phase 3A report, detections were reported as combined alerts ($\hat{y} \in \{1, 2\}$). The audit disaggregated detections into specific alert phases:

| Model | Total Episodes | Combined Detection ($\hat{y} \ge 1$) | Detected by Warning Alert ($\hat{y} = 1$) | Detected by Critical Alert ($\hat{y} = 2$) | Mean Lead Time | Median Lead Time |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Majority Class** | 36 | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | N/A | N/A |
| **Logistic Regression** | 36 | **32 (88.9%)** | **24 (66.7%)** | 13 (36.1%) | **1,138.8s** (19.0 min) | **1,353.0s** (22.6 min) |
| **Random Forest** | 36 | **23 (63.9%)** | 14 (38.9%) | **17 (47.2%)** | 964.7s (16.1 min) | 878.9s (14.6 min) |

- **Missed Episodes Denominator:** In all cases, missed episodes remain in the denominator ($N=36$). Detections plus misses sum exactly to $36$.

### 5. False Alarm Denominator Audit & Reporting

The Phase 3A report stated a False Alarm Rate of $10.4\%$ for Random Forest based on a denominator of $48$ "pure-normal" entities. The audit investigated this denominator:

1. **Composition of Test Entities ($N=80$):**
   - **Pure Normal Entities ($N=48$):** Tasks that complete their runs without any failure event in the observation window.
   - **Failed Entities ($N=32$):** Tasks that experience a failure event during the observation window.
2. **Limitation of the $N=48$ Denominator:**
   Restricting the false alarm denominator to $48$ pure-normal entities ignores normal operating windows that occur on failed tasks prior to their 30-minute failure horizon (i.e. during the healthy earlier phases of their runs).
3. **Multi-Level False Positive Reporting:**
   To provide full transparency without inventing exposure metrics, the audit reports three complementary false-positive measures:

| False Alarm Level | Denominator Description | Total Denominator Size | Majority Class False Alarms | Logistic Regression False Alarms | Random Forest False Alarms |
|:---|:---|:---:|:---:|:---:|:---:|
| **Entity-Level (Pure Normal)** | Entities with zero failure events throughout trace | **48 entities** | 0 / 48 (**0.0%**) | 21 / 48 (**43.8%**) | **5 / 48 (10.4%)** |
| **Entity-Level (All Entities)** | All entities evaluated during their normal windows | **80 entities** | 0 / 80 (**0.0%**) | 37 / 80 (**46.2%**) | **12 / 80 (15.0%)** |
| **Window-Level False Positive Rate** | Ground-truth NORMAL windows ($y=0$) | **864 windows** | 0 / 864 (**0.0%**) | 284 / 864 (**32.87%**) | **45 / 864 (5.21%)** |

> **Scientific Caveat on Exposure Durations:** In accordance with the prompt guidance, **task-hour failure rates are NOT reported**. Raw Borg trace shards in `data/raw/` cover ~3 hours of cluster activity with discrete 5-minute sampling; extrapolating to task-hours or MTBF without full 29-day exposure records would be statistically ungrounded.

---

## Part C: Master Audit Scorecard

| Check # | Audit Item | Verification Method | Status |
|:---|:---|:---|:---:|
| **Check 1** | Model & Representation Selection on Train/Val Only | Code inspection of `scripts/phase3a_baselines.py` Step 3 | **PASS** |
| **Check 2** | Test Partition Quarantine | Code inspection; single test evaluation call on frozen models | **PASS** |
| **Check 3** | Exact Recomputation of Published Test Metrics | Programmatic calculation from `test_predictions_*.csv` | **PASS** |
| **Check 4** | Prediction & Metadata Row Alignment | Assertion check of entity IDs, timestamps, and true labels | **PASS** |
| **Check 5** | Consistent Class Indexing (0=Norm, 1=Warn, 2=Crit) | Cross-module verification against schema and models | **PASS** |
| **Check 6** | PR-AUC / Average Precision Calculation | One-vs-Rest calculation and constant prevalence handling | **PASS** |
| **Check 7** | Majority Baseline & Undefined Metric Handling | Explicit zero-division handling and prevalence assignment | **PASS** |
| **Check 8** | Saved Artifact & Report Configuration Consistency | Exact match of hyperparameter bundles and environment | **PASS** |
| **Check 9** | Episode Failure Timestamp Fidelity | Match against raw Borg 2011 event shards (0.000000s diff) | **PASS** |
| **Check 10**| Causal Lead-Time & Alert Directionality | Earliest pre-failure alert; alerts after failure cannot count | **PASS** |
| **Check 11**| Disaggregated Alert Phase Reporting | Warning-phase vs Critical-phase detection rates reported | **PASS** |
| **Check 12**| False Alarm Denominator Transparency | Multi-level reporting (pure-normal, all-entity, window-FPR) | **PASS** |
| **Suite** | Complete Unit Test Suite | 25 / 25 unit tests passing (`pytest tests/`) | **PASS** |

### **Audit Conclusion & Recommendation:**
$$\mathbf{AUDIT \; RESULT: \; ALL \; 12 \; INTEGRITY \; CHECKS \; PASSED}$$

The Phase 3A baselines are scientifically sound, leakage-free, and mathematically verified. **The Random Forest test Macro-F1 of 0.5718 is confirmed as the official benchmark for Phase 3B.**

Proceed to **Phase 3B — Temporal LSTM Framework**.
