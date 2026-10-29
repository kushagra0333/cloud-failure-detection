# Phase 7B Pre-Training Pipeline Verification and Evaluation Protocol Report

**Author:** Antigravity (AI Research Assistant)  
**Date:** October 27, 2026  
**Local Time:** 23:50:00+05:30  
**Phase:** Phase 7B (Pre-Training Checks & Protocol Freeze)  
**Status:** Pre-Training Audit Complete — Awaiting Final Protocol Review Before Retraining  
**Official Pre-Training Verdict:** **`ALL PRE-TRAINING CHECKS PASSED — PROTOCOL FROZEN`**

---

## 1. Executive Summary

In accordance with user authorization for Phase 7B, the **Candidate A development split** has been officially frozen and partitioned into dedicated Phase 7B dataset arrays in [data/processed/phase7b/](file:///c:/Users/hp/Desktop/Research%20project/data/processed/phase7b/). 

Before launching any model retraining, this audit independently verified:
1. **The Frozen Split Integrity:** Verified that [results/phase7a_4/candidate_a_job_grouped_split.csv](file:///c:/Users/hp/Desktop/Research%20project/results/phase7a_4/candidate_a_job_grouped_split.csv) matches its verified SHA-256 checksum (`f84bc4fe54f8e0583e2ee52f49b2fbe8e6fcafb4f020c656985a1acc31bb4946`).
2. **Zero Parent-Job Overlap:** Confirmed that Train ($N=51$ jobs), Validation ($N=12$ jobs), and Test ($N=11$ jobs) share **0 overlapping parent jobs** at both the string and raw cluster integer database level.
3. **Data Pipeline Correctness:** Confirmed **0 post-failure window endpoints**, **0 cross-failure history violations**, **0 NaNs**, **0 infinite values**, and **0 duplicate windows**.
4. **Feature Engineering Audit:** Verified that feature calculation uses strictly trailing observations with zero target or future telemetry leakage.
5. **Predeclared Evaluation Protocol:** Formally documented and frozen all primary metrics, secondary metrics, operational acceptance gates, and parent-job cluster bootstrap uncertainty estimates **prior to any model training**.
6. **Quarantine of Historical Holdout:** Reaffirmed that the spent Phase 6 confirmatory holdout is quarantined and will not be accessed during model retraining or used for fresh confirmatory claims.

> [!IMPORTANT]
> **Stop Condition Strictly Observed:** In accordance with user instructions (*"First complete the pre-training checks and show me their results. Do not begin training until those checks pass and the evaluation protocol is presented for review"*), **no models have been trained or tuned**. Retraining will commence only after user review and approval of this protocol.

---

## 2. Frozen Split Verification and Phase 7B Dataset Partitioning

### 2.1 Split Assignment Integrity
The Candidate A assignment file was verified against its cryptographic checksum:
- **Assignment File:** [results/phase7a_4/candidate_a_job_grouped_split.csv](file:///c:/Users/hp/Desktop/Research%20project/results/phase7a_4/candidate_a_job_grouped_split.csv)
- **Verified SHA-256:** `f84bc4fe54f8e0583e2ee52f49b2fbe8e6fcafb4f020c656985a1acc31bb4946`
- **Integrity Status:** **`IMMUTABLE & UNMODIFIED`**

### 2.2 Phase 7B Dedicated Dataset Artifacts
The 4,960 clean development windows were partitioned from the repaired Phase 7A diagnostic data into versioned Phase 7B files in [data/processed/phase7b/](file:///c:/Users/hp/Desktop/Research%20project/data/processed/phase7b/):

| File Name | Path | Dimensions / Rows | Size (Bytes) | SHA-256 Checksum |
| :--- | :--- | :---: | :---: | :--- |
| `X_train.npy` | [X_train.npy](file:///c:/Users/hp/Desktop/Research%20project/data/processed/phase7b/X_train.npy) | `(3278, 5, 31)` | 2,032,488 | `b78a2c4187f55b1eb79803bf377f075d9e57bfd6945a557bfa16a75b22b7a9aa` |
| `y_train.npy` | [y_train.npy](file:///c:/Users/hp/Desktop/Research%20project/data/processed/phase7b/y_train.npy) | `(3278,)` | 26,352 | `ff9e93bfd29037fe63ec04bc99f9c7a233b8110b95543c7b80540d6cfaea4960` |
| `train_metadata.csv` | [train_metadata.csv](file:///c:/Users/hp/Desktop/Research%20project/data/processed/phase7b/train_metadata.csv) | 3,278 rows | 158,582 | `55899986fc0a9e7f867491cf3b123ce3e2f5ff579c3dd3f3d7904e9fb7e96bb9` |
| `X_val.npy` | [X_val.npy](file:///c:/Users/hp/Desktop/Research%20project/data/processed/phase7b/X_val.npy) | `(657, 5, 31)` | 407,468 | `26639c09191e4534ef034ca27b8aa075594b9ffbe8142cb06fa6b1e6e0fb8f2c` |
| `y_val.npy` | [y_val.npy](file:///c:/Users/hp/Desktop/Research%20project/data/processed/phase7b/y_val.npy) | `(657,)` | 5,384 | `fc59c5d013e8e24ddc89e83ae41088d8b9d3e86c0e5a6f2b5fa4b47ebc7b415a` |
| `val_metadata.csv` | [val_metadata.csv](file:///c:/Users/hp/Desktop/Research%20project/data/processed/phase7b/val_metadata.csv) | 657 rows | 31,348 | `ad22c222ffda876a40a5a40b3cb19be00c0250d03b30e06240d4239fe248e83f` |
| `X_test.npy` | [X_test.npy](file:///c:/Users/hp/Desktop/Research%20project/data/processed/phase7b/X_test.npy) | `(1025, 5, 31)` | 635,628 | `99e69c8e8d89e50e93231495c026ca2b0a3f6567116f15b6329c49d885627720` |
| `y_test.npy` | [y_test.npy](file:///c:/Users/hp/Desktop/Research%20project/data/processed/phase7b/y_test.npy) | `(1025,)` | 8,328 | `2a84391216d9a1f2b604f86d63ef2fe59187e148e6ef921aa2133c69c6f24021` |
| `test_metadata.csv` | [test_metadata.csv](file:///c:/Users/hp/Desktop/Research%20project/data/processed/phase7b/test_metadata.csv) | 1,025 rows | 48,229 | `30a2fef42cbb12270928929e083c2705ff76bcfa4bf49122589ff2ef6bcf012c` |

### 2.3 Partitioned Split Summary

| Split | Parent Jobs | Entities | Windows | Windows % | Normal ($y=0$) | Normal % | Warning ($y=1$) | Warning % | Critical ($y=2$) | Critical % |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **TRAIN** | 51 | 284 | 3,278 | 66.09% | 2,797 | 65.38% | 244 | 71.55% | 237 | 69.50% |
| **VAL** | 12 | 69 | 657 | 13.25% | 560 | 13.09% | 48 | 14.08% | 49 | 14.37% |
| **TEST** | 11 | 59 | 1,025 | 20.67% | 921 | 21.53% | 49 | 14.37% | 55 | 16.13% |
| **TOTAL** | **74** | **412** | **4,960** | **100.0%** | **4,278** | **100.0%** | **341** | **100.0%** | **341** | **100.0%** |

---

## 3. Pre-Training Pipeline Verification Checks

All automated data integrity checks passed with **zero defects**:

| Integrity Verification Check | Diagnostic Test | Observed Result | Status |
| :--- | :--- | :---: | :---: |
| **Parent-Job Overlap** | Train $\cap$ Val, Train $\cap$ Test, Val $\cap$ Test | **0 jobs (0.0%)** | **`PASS`** |
| **Raw `job_id` Set Disjointness** | Set intersection of cluster 64-bit integer IDs | **0 jobs (0.0%)** | **`PASS`** |
| **Window Uniqueness** | `dev_window_idx` duplicate check | **0 duplicates** | **`PASS`** |
| **Entity Accounting** | Total unique entities | **412 / 412** | **`PASS`** |
| **Post-Failure Endpoints** | $\text{prediction\_timestamp} \ge t_{\text{first\_fail}}$ | **0 violations** | **`PASS`** |
| **Cross-Failure History** | Sequence history $[t-1200\text{s}, t]$ crossing $t_{\text{first\_fail}}$ | **0 violations** | **`PASS`** |
| **Tensor Numerical Sanity** | Check for NaNs or $\pm\infty$ in $X_{\text{train}}, X_{\text{val}}, X_{\text{test}}$ | **0 NaNs / 0 Infs** | **`PASS`** |
| **Label Array Sanity** | Check for NaNs or labels outside $\{0, 1, 2\}$ in $y$ | **0 NaNs / 0 Invalid** | **`PASS`** |
| **Label-Feature Alignment** | Row count match between $X, y,$ and metadata | **Exact match (4,960)** | **`PASS`** |
| **Feature Column Leakage** | Check for target, failure, or future telemetry in features | **0 forbidden columns** | **`PASS`** |

---

## 4. Feature Engineering and Lookback Analysis

We audited the feature construction implementation in [src/features/engineer.py](file:///c:/Users/hp/Desktop/Research%20project/src/features/engineer.py) and [src/features/scaler.py](file:///c:/Users/hp/Desktop/Research%20project/src/features/scaler.py):

1. **Lag Feature Initialization:**
   `FeatureEngineer` computes lag features up to lag 6 (`shift(1)`, `shift(2)`, `shift(3)`, `shift(6)`). For the first $k$ observations of a container, `shift(k)` yields `NaN`.
   - In `LeakageSafeScaler.transform()`, missing values are imputed using the **median of that feature fitted strictly on the Training split**.
   - No validation or test data statistics leak into training.
2. **Rolling Feature Initialization:**
   Rolling statistics (mean, standard deviation, range, trend slope) utilize `min_periods=1` or `min_periods=3`. At task start, rolling statistics are computed over whatever valid preceding observations exist, preventing initialization NaNs.
3. **Telemetry Bin Boundaries:**
   Raw telemetry measurements are integrated into discrete 300-second bins from `start_time_sec` to `end_time_sec`.
4. **Historical Telemetry Duration:**
   - Sequence length $L = 5$ time steps covers discrete timestamps $[t - 1200\text{s}, t]$ (endpoint delta $= 1,200\text{s} = 20$ minutes).
   - The 5 underlying bins integrate a continuous duration of $5 \times 300\text{s} = 1,500\text{s}$ (25 minutes) over $[t - 1500\text{s}, t]$.
   - Extended rolling features at step 0 look back an additional 6 steps ($1,800\text{s}$), meaning the earliest raw telemetry point required by the feature extractor is at $t - 3,000\text{s}$ (50 minutes before prediction anchor $t$).
   - Combined with the 30-minute warning horizon ($W_{\text{warn}} = 1,800\text{s}$), the total informational footprint is **4,800 seconds (80 minutes)**.

---

## 5. Predeclared Evaluation Protocol (Frozen)

This protocol is permanently frozen in [results/phase7b/phase7b_evaluation_protocol.json](file:///c:/Users/hp/Desktop/Research%20project/results/phase7b/phase7b_evaluation_protocol.json) prior to model training:

### 5.1 Primary and Secondary Metrics Hierarchy
- **Primary Metric:** **Macro-F1 Score** across all three classes $\{0, 1, 2\}$ on the Candidate A Test split:
  $$\text{Macro-F1} = \frac{\text{F1}_{\text{Normal}} + \text{F1}_{\text{Warning}} + \text{F1}_{\text{Critical}}}{3}$$
  - Checkpoints during training are selected exclusively by **Validation split Macro-F1**.
- **Secondary Window-Level Metrics:**
  - Critical Class ($y=2$): Precision, Recall, F1 (urgent 15-minute failure horizon).
  - Warning Class ($y=1$): Precision, Recall, F1 (early 30-minute warning horizon).
  - Normal Class ($y=0$): Precision, Recall, F1.
  - Overall Accuracy, PR-AUC per class, and 3x3 Confusion Matrix.
- **Episode-Level Early Failure Detection Metrics:**
  - An episode is uniquely identified by `(entity_id, fail_timestamp)`.
  - Detection occurs if any pre-failure window ($y \in \{1, 2\}$) yields $y_{\text{pred}} > 0$.
  - **Episode Recall:** Fraction of eligible failure episodes detected before failure.
  - **Detection Lead Time:** Time difference $t_{\text{fail}} - t_{\text{first\_alert}}$ (median, IQR, min, max).
- **False-Alarm and Operational Containment Metrics:**
  - **Pure-Normal Entity FAR:** Percentage of pure-normal entities triggering $\ge 1$ false alarm (target $\le 10.0\%$).
  - **Pure-Normal Parent-Job FAR:** Percentage of pure-normal parent jobs triggering $\ge 1$ false alarm (target $\le 15.0\%$).
  - **Normal-Window FPR:** Percentage of normal windows falsely predicted as failure (target $\le 5.0\%$).

### 5.2 Parent-Job Cluster Bootstrap Uncertainty Estimation
Because container tasks from the same parent job share execution characteristics, window-level i.i.d. bootstrap underestimates variance.
- **Cluster Bootstrap Procedure:** Resamples $J_{\text{test}} = 11$ parent jobs with replacement ($B = 1,000$ iterations).
- All windows of sampled jobs are drawn together, preserving intra-cluster dependencies.
- Reports empirical 95% confidence intervals $[2.5\%, 97.5\%]$ for Macro-F1, Episode Recall, Entity FAR, and Window FPR.

### 5.3 Predefined Operational Acceptance Criteria
A model is declared an **`OPERATIONAL_WINNER`** if and only if it satisfies both gates simultaneously on the Test split:
- **Gate 1 (Early Failure Detection):**
  $$\text{Episode Detection Recall} \ge 70.0\% \quad \text{AND} \quad \text{Critical Window Recall} \ge 50.0\%$$
- **Gate 2 (False-Alarm Containment):**
  $$\text{Pure-Normal Entity FAR} \le 10.0\% \quad \text{AND} \quad \text{Normal-Window FPR} \le 5.0\%$$
- **Winner Determination:** If multiple models pass both gates, the model with the highest Test Macro-F1 wins. If zero models pass both gates, the verdict is **`NO_OPERATIONAL_WINNER`**.
- **Immutability Clause:** These thresholds cannot be changed after viewing test results.

---

## 6. Model Training Specification (For Phase 7B Retraining)

When retraining is initiated, the four existing project architectures will be trained strictly on the Candidate A Training split (`data/processed/phase7b/X_train.npy`, `y_train.npy`):

1. **Random Forest Baseline:**
   - Architecture: `RandomForestClassifier(n_estimators=100, max_depth=12, min_samples_split=5, class_weight='balanced', random_state=42)`
   - Input: 31-dimensional tabular feature vector $x_{\text{last}}$ (endpoint step of each sequence).
2. **Temporal LSTM:**
   - Architecture: `TemporalLSTMClassifier(input_dim=31, hidden_dim=64, num_layers=2, dropout=0.2, num_classes=3)`
   - Training: Adam optimizer ($\text{lr}=10^{-3}$), weighted cross-entropy (`weights=[1.0, 1.5, 2.0]`), batch size 64, early stopping on validation Macro-F1.
   - Input: 3D sequence tensor `(batch, 5, 31)`.
3. **Compact Temporal Transformer:**
   - Architecture: `CompactTemporalTransformer(input_dim=31, d_model=32, nhead=4, num_layers=2, dim_feedforward=64, dropout=0.1, num_classes=3)`
   - Training: AdamW optimizer ($\text{lr}=5 \times 10^{-4}$, weight decay $10^{-4}$), weighted cross-entropy (`weights=[1.0, 1.5, 2.0]`), batch size 64, early stopping on validation Macro-F1.
   - Input: 3D sequence tensor `(batch, 5, 31)`.
4. **Representation Hybrid:**
   - Architecture: Frozen Transformer representation encoder (32-dim pooled embedding) concatenated with the 31-dim tabular vector ($63$ dimensions total) $\to$ `RandomForestClassifier`.

---

## 7. Deliverables Summary

| Deliverable Path | Type | Status | Description |
| :--- | :---: | :---: | :--- |
| [results/phase7b/pretraining_pipeline_checks.json](file:///c:/Users/hp/Desktop/Research%20project/results/phase7b/pretraining_pipeline_checks.json) | JSON | **Verified** | Machine-readable data pipeline validation results |
| [results/phase7b/phase7b_evaluation_protocol.json](file:///c:/Users/hp/Desktop/Research%20project/results/phase7b/phase7b_evaluation_protocol.json) | JSON | **Frozen** | Machine-readable predeclared evaluation protocol |
| [reports/phase7b/phase7b_pretraining_audit_report.md](file:///c:/Users/hp/Desktop/Research%20project/reports/phase7b/phase7b_pretraining_audit_report.md) | Markdown | **Complete** | Comprehensive pre-training verification report |
| [data/processed/phase7b/](file:///c:/Users/hp/Desktop/Research%20project/data/processed/phase7b/) | Directory | **Created** | Dedicated Phase 7B arrays (`X_train.npy`, `y_train.npy`, etc.) |

---

## 8. Official Verdict and Next Step

### Official Phase 7B Pre-Training Verdict
# **`ALL PRE-TRAINING CHECKS PASSED — PROTOCOL FROZEN`**

All data integrity checks, split isolation checks, and feature pipeline audits have passed with zero defects. The evaluation protocol and operational acceptance gates are locked.

**I have paused execution in accordance with your instructions. Please review the pre-training check results and the evaluation protocol above. Once you confirm, I will initiate model retraining.**
