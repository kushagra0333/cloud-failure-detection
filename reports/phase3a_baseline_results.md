# Phase 3A — Leakage-Safe Baseline Modeling Report

**Project:** Temporal LSTM/Transformer Framework for Early Failure Prediction in Cloud Computing  
**Phase:** 3A — Leakage-Safe Baseline Modeling (Tabular & Naive Baselines)  
**Execution Date:** 2026-10-26  
**Auditor & Lead Researcher:** Senior ML Engineer & Independent Experimental Auditor  
**Status:** **COMPLETE & SCIENTIFICALLY VALIDATED** (All Baselines Evaluated; Zero Deep Models Trained)

---

## Executive Summary

Phase 3A establishes the empirical performance benchmark for the research project. In strict accordance with experimental isolation protocols, three baseline models were trained exclusively on the training partition and evaluated on both the primary entity-grouped validation and held-out test partitions:
1. **Majority-Class Naive Classifier** (Predicts dominant class `NORMAL = 0` always)
2. **Class-Weighted Logistic Regression** (Multinomial $L_2$-penalized linear model)
3. **Class-Weighted Random Forest** (Ensemble of 100 balanced decision trees, `max_depth = 12`)

### Key Empirical Findings
- **Primary Model Comparison (Macro-F1):** On the held-out primary test set ($N=1,003$), **Class-Weighted Random Forest** achieved the highest overall performance with a **Macro-F1 of 0.5718** and **Accuracy of 0.8654**, substantially outperforming **Class-Weighted Logistic Regression** (Macro-F1 = 0.4143, Accuracy = 0.6351) and the **Majority-Class floor** (Macro-F1 = 0.3085, Accuracy = 0.8614).
- **Sequence-to-Vector Representation:** An empirical validation comparison revealed that extracting the **final observed feature vector** ($x_{t=15} \in \mathbb{R}^{31}$) outperformed the full flattened sequence ($x \in \mathbb{R}^{465}$, Macro-F1 = 0.6870 vs. 0.7026 on validation). Because the Phase 2 feature pipeline already engineered trailing rolling statistics, lags, and OLS slopes into each timestep, the 31-dimensional final step preserves parsimony and causal ordering without $15\times$ over-parameterization.
- **Episode-Level Failure Detection vs. False Alarms:** 
  - Logistic Regression achieved high episode sensitivity (detecting **88.9%** of failure episodes with a mean advance warning lead time of **19.0 minutes**), but suffered an unacceptable **43.8% False Alarm Rate** on pure normal tasks (21 out of 48 normal tasks triggered false alerts).
  - In contrast, Random Forest achieved **63.9% Episode Detection** (23 out of 36 test episodes) with a mean lead time of **16.1 minutes** (964.7s), while maintaining a restrained **10.4% False Alarm Rate** (only 5 out of 48 normal tasks flagged).
- **Benchmarking Benchmark for Phase 3B:** These baselines establish a rigorous, leakage-free comparison point. The primary target for the Phase 3B Temporal LSTM is to surpass the Random Forest Macro-F1 benchmark ($0.5718$) by exploiting non-linear temporal sequence dynamics without manual feature aggregation.

---

## 1. Pre-Training Schema and Trace Version Confirmation

Prior to model fitting, the source lifecycle event schema was verified against actual downloaded raw files in `data/raw/`:
- **Selected Trace:** Google Cluster Trace 2011-2 (Borg 2011 schema, Reiss et al. 2011).
- **Schema Mapping Verification:**
  - `ClusterEventTypeV2.FAIL == 3` (abnormal execution failure)
  - `ClusterEventTypeV2.FINISH == 4` (normal successful completion)
  - `ClusterEventTypeV2.KILL == 5` (user/job cancellation, censored)
  - `ClusterEventTypeV2.EVICT == 2` (preemption/resource overcommit, censored)
- **Empirical Shard Verification:** Chunked scans across `sample_task_events.csv.gz` (parts 0, 1, and 2) verified the presence of 6,624 total `FAIL` rows (`event_type == 3`) and 79,778 `FINISH` rows (`event_type == 4`).
- **Trace Version Contrast:** Borg 2019 (v3, Wilkes 2020) maps `FAIL == 5` and `FINISH == 6`. The pipeline correctly uses `get_event_classifier("2011")`, ensuring exact semantic fidelity.

**Verdict: Source Event Schema is Scientifically Validated [PASS].**

---

## 2. Primary Dataset Verification (Entity-Grouped)

The primary evaluation uses entity-grouped splitting, ensuring that all windows and failure episodes for any given task instance reside strictly within one partition.

### 2.1 Partition Counts, Tensor Shapes, and Class Distributions

The actual saved files in `data/processed/` were loaded and verified:

| Partition | Window Count ($N$) | Input Tensor Shape | Class 0 (NORMAL) | Class 1 (WARNING) | Class 2 (CRITICAL) | Total Pre-Failure Windows | Unique Entities | Unique Failure Episodes |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Train** | 4,798 | `(4798, 15, 31)` | 4,188 (87.3%) | 247 (5.1%) | 363 (7.6%) | 610 (12.7%) | 367 | 207 |
| **Validation** | 1,015 | `(1015, 15, 31)` | 840 (82.8%) | 72 (7.1%) | 103 (10.1%) | 175 (17.2%) | 78 | 61 |
| **Test** | 1,003 | `(1003, 15, 31)` | 864 (86.1%) | 68 (6.8%) | 71 (7.1%) | 139 (13.9%) | 80 | 36 |
| **Total** | **6,816** | — | **5,892** (86.4%) | **387** (5.7%) | **537** (7.9%) | **924** (13.6%) | **525** | **304** |

- Input tensors contain strictly zero NaNs.
- Partition sizes and shapes exactly match Phase 2.4 specifications.
- Pairwise entity intersections are strictly empty:
  $$\text{Entities}(D_{\text{train}}) \cap \text{Entities}(D_{\text{val}}) = \emptyset, \quad \text{Entities}(D_{\text{train}}) \cap \text{Entities}(D_{\text{test}}) = \emptyset, \quad \text{Entities}(D_{\text{val}}) \cap \text{Entities}(D_{\text{test}}) = \emptyset$$

---

## 3. Sequence-to-Vector Representation Rationale

Traditional tabular baselines (Logistic Regression and Random Forest) operate on fixed-dimensional vectors $\mathbb{R}^D$ rather than 3D sequential tensors $\mathbb{R}^{L \times F}$. A principled sequence conversion strategy is required.

### 3.1 Comparison of Conversion Strategies

Three sequence conversion strategies were evaluated on the **Validation Set** using models fitted strictly on the Training set (with fixed seed 42):

```
Sequence Conversion Options (L=15 timesteps, F=31 features):
Option 1: Final Step (D=31)       -> [ x_{t=15} ] (Preserves trailing rolling features, parsimonious)
Option 2: Trajectory Summary (D=93)-> [ x_{t=15} | mean(x_{1..15}) | std(x_{1..15}) ]
Option 3: Full Flattened (D=465)   -> [ x_{1} | x_{2} | ... | x_{15} ] (High dimensional, collinear)
```

| Strategy | Feature Dim ($D$) | Logistic Regression Val Macro-F1 | Random Forest Val Macro-F1 | Empirical Assessment |
|:---|:---:|:---:|:---:|:---|
| **Final Step ($x_{t=15}$)** | **31** | 0.4817 | **0.7026** | **Selected:** Parsimonious; optimal RF performance; trailing rolling features already captured |
| **Trajectory Summary** | 93 | **0.5424** | 0.6734 | Effective for linear models, but degrades Random Forest tree partition efficiency |
| **Full Flattened** | 465 | 0.5162 | 0.6870 | High collinearity; severe over-parameterization on minority classes (only 247 warning samples) |

### 3.2 Scientific Justification for Selecting Final Step ($D=31$)

1. **Information Sufficiency:** Timestep $L=15$ corresponds to prediction timestamp $t_{\text{pred}}$. In Phase 2, [`src/features/engineer.py`](file:///c:/Users/hp/Desktop/Research%20project/src/features/engineer.py) computed trailing causal dynamics directly into each row:
   - 1-, 2-, 3-, and 6-step lags (`cpu_utilization_lag_k`, `memory_utilization_lag_k`)
   - 3- and 6-step rolling means, standard deviations, and ranges
   - 6-step analytical OLS trend slopes (`cpu_trend_slope_6`)
   - 6-step threshold exceedance fractions (`cpu_exceed_fraction_6`)
   - Task age and missingness tracking
   Therefore, timestep 15 is not a single point observation; it is a compact summary of the preceding 75 minutes of execution.
2. **Consequences of Sequence Flattening:** Flattening 15 steps expands dimensionality to $465$ features. With only $247$ warning samples in training, flattening yields a feature-to-sample ratio exceeding $1.88$, inducing collinearity between adjacent lagged steps and degrading Random Forest validation Macro-F1 from $0.7026$ down to $0.6870$.
3. **Decision:** The **Final Step ($D=31$)** was selected as the frozen feature representation for all primary baseline evaluations.

---

## 4. Model Training & Validation-Driven Freeze Protocol

### 4.1 Training Safeguards
- **Learned Transformations:** Standard Scaler parameters and median imputers were fitted exclusively on training data during Phase 2 ([`configs/fitted_scaler.json`](file:///c:/Users/hp/Desktop/Research%20project/configs/fitted_scaler.json)).
- **Class-Weight Calculation:** In both Logistic Regression and Random Forest, `class_weight="balanced"` computes inverse weights strictly from `y_train`:
  $$w_0 = \frac{4798}{3 \times 4188} = 0.3819, \quad w_1 = \frac{4798}{3 \times 247} = 6.4750, \quad w_2 = \frac{4798}{3 \times 363} = 4.4059$$
- **Random Seed:** Fixed seed `42` for all model initializations, tree building, and shuffling.

### 4.2 Validation Set Performance & Model Selection

Validation metrics were computed to select the top-performing baseline and freeze all hyperparameter configurations prior to touching the test set:

| Model | Val Macro-F1 (Primary) | Val Accuracy (Secondary) | Warning Recall | Critical Recall | Warning PR-AUC | Critical PR-AUC |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Majority Class** | 0.3019 | 0.8276 | 0.0000 | 0.0000 | 0.0709* | 0.1015* |
| **Class-Weighted Logistic Regression** | 0.4817 | 0.6384 | **0.6667** | 0.5728 | 0.1576 | 0.6512 |
| **Class-Weighted Random Forest** | **0.7026** | **0.8739** | 0.2917 | **0.6408** | **0.4672** | **0.7834** |

*\*Note: For Majority Class, PR-AUC equals the empirical class prevalence due to constant predicted probabilities.*

**Freezing Protocol Decision:** **Class-Weighted Random Forest** achieved the superior validation Macro-F1 ($0.7026$). All hyperparameter settings (`n_estimators=100`, `max_depth=12`, `C=1.0`, `strategy="final_step"`) were **frozen**. The held-out test set was evaluated exactly once.

---

## 5. Held-Out Test Set Results (Primary Benchmark)

The held-out test partition ($N=1,003$ windows; 864 Normal, 68 Warning, 71 Critical) represents tasks completely unobserved during training and validation.

### 5.1 Comprehensive Window-Level Metrics

| Metric | Majority Class Baseline | Class-Weighted Logistic Regression | Class-Weighted Random Forest |
|:---|:---:|:---:|:---:|
| **Macro-F1 (Primary)** | **0.3085** | **0.4143** | **0.5718** |
| **Accuracy (Secondary)** | 0.8614 | 0.6351 | 0.8654 |
| **NORMAL Precision** | 0.8614 | 0.9355 | 0.9151 |
| **NORMAL Recall** | 1.0000 | 0.6713 | 0.9479 |
| **NORMAL F1-Score** | 0.9255 | 0.7817 | 0.9312 |
| **WARNING Precision** | 0.0000 | 0.1185 | 0.3400 |
| **WARNING Recall** | 0.0000 | 0.4706 | 0.2500 |
| **WARNING F1-Score** | 0.0000 | 0.1893 | 0.2881 |
| **CRITICAL Precision** | 0.0000 | 0.2212 | 0.5517 |
| **CRITICAL Recall** | 0.0000 | 0.3521 | 0.4507 |
| **CRITICAL F1-Score** | 0.0000 | 0.2717 | 0.4961 |
| **Warning Recall** | 0.0000 | 0.4706 | 0.2500 |
| **Critical Recall** | 0.0000 | 0.3521 | 0.4507 |
| **Warning PR-AUC** | 0.0678* | 0.1371 | **0.3701** |
| **Critical PR-AUC** | 0.0708* | 0.2316 | **0.5637** |

### 5.2 Confusion Matrices (Test Set, N = 1,003)

```
Majority Class Confusion Matrix:
Predicted ->    NORMAL    WARNING    CRITICAL
Actual NORMAL    864         0          0
Actual WARNING    68         0          0
Actual CRITICAL   71         0          0

Logistic Regression Confusion Matrix:
Predicted ->    NORMAL    WARNING    CRITICAL
Actual NORMAL    580       206         78
Actual WARNING    26        32         10
Actual CRITICAL   14        32         25

Random Forest Confusion Matrix:
Predicted ->    NORMAL    WARNING    CRITICAL
Actual NORMAL    819        25         20
Actual WARNING    45        17          6
Actual CRITICAL   31         8         32
```

### 5.3 Diagnostic Discussion of Test Results
1. **The Majority-Class Baseline:** Achieves an Accuracy of $86.14\%$ simply by predicting Normal for all samples, yet yields a Macro-F1 of only $0.3085$ and zero failure detection ($0.0\%$ recall on warning and critical). This demonstrates why Accuracy is misleading in imbalanced telemetry monitoring and establishes Macro-F1 as the mandatory primary metric.
2. **Logistic Regression Trade-Off:** Linear decision boundaries with balanced class weights succeed in flagging warning samples (Recall = $47.06\%$), but suffer from severe false positive inflation: 284 normal windows were misclassified as Warning or Critical (Normal Precision = $0.9355$, Normal Recall = $0.6713$), dragging its Macro-F1 down to $0.4143$.
3. **Random Forest Superiority:** Random Forest non-linear decision splits achieve a balanced operating point. Normal recall remains high at $94.79\%$ (only 45 false alerts across 864 normal samples), while achieving $0.5517$ precision and $0.4507$ recall on Critical failures, yielding the benchmark Macro-F1 of **0.5718**.

---

## 6. Failure-Episode Level Evaluation

### 6.1 Empirical Validation of Episode Timestamps

The audit confirmed that `fail_timestamp = round(prediction_timestamp + time_to_failure, 1)` corresponds exactly to the true source failure event in raw trace shards:
- Across all 139 pre-failure windows in the test set, the maximum difference between reconstructed failure time and raw `FAIL` event timestamp is **$0.000000\text{s}$**.
- Reconstructed failure timestamps represent true ground-truth physical failure points.

### 6.2 Episode Grouping and Evaluation Rules

1. **Episode Definition:** All sliding windows for a task entity $E$ associated with the same future failure event timestamp $T_{\text{fail}}$ belong to episode $(E, T_{\text{fail}})$.
2. **Detection Rule:** An episode is defined as **DETECTED** if at least one pre-failure window ($0 < \tau \le 30\text{m}$) predicts an alert ($\hat{y} \in \{1: \text{WARNING}, 2: \text{CRITICAL}\}$).
3. **Lead Time Calculation:**
   $$\text{Lead Time} = T_{\text{fail}} - t_{\text{earliest\_alert}}$$
   The advance warning time before actual task termination.
4. **Duplicate Alert Policy:** Once an alert is triggered, subsequent alert windows are treated as confirmations; they do not increment the detected episode count.
5. **False Alarm Entity Rate:** Evaluated on pure normal entities (tasks that complete without any failure). A false alarm is flagged if any window for a pure normal entity predicts $\hat{y} \in \{1, 2\}$.
6. **Caveat on Independent Real-World Failures:** In cluster computing, a single misconfigured task can fail, be restarted by Borg, and fail again (a crash-loop). The 36 test episodes represent **36 distinct failure occurrences** across 32 failed entities, not necessarily 36 independent root causes.

### 6.3 Episode-Level Test Set Results

| Metric | Majority Class | Class-Weighted Logistic Regression | Class-Weighted Random Forest |
|:---|:---:|:---:|:---:|
| **Total Test Failure Episodes** | 36 | 36 | 36 |
| **Detected Episodes** | 0 | **32** | 23 |
| **Episode Detection Rate** | 0.0% | **88.9%** | **63.9%** |
| **Mean Advance Lead Time** | 0.0s (0.0 min) | **1,138.8s** (19.0 min) | **964.7s** (16.1 min) |
| **Median Advance Lead Time** | 0.0s (0.0 min) | **1,353.0s** (22.6 min) | **878.9s** (14.6 min) |
| **Pure Normal Entities ($N$)** | 48 | 48 | 48 |
| **False Alarm Entities** | 0 | 21 | **5** |
| **False Alarm Entity Rate** | 0.0% | **43.8%** | **10.4%** |

```
Operational Lead Time Trade-Off:
Logistic Regression: [======================== 88.9% Detection ] -> BUT 43.8% Normal Tasks False Alarmed
Random Forest:       [================= 63.9% Detection ]        -> AND ONLY 10.4% Normal Tasks False Alarmed
```

**Operational Interpretation:** Logistic Regression alerts cluster operators to nearly 90% of failures with a 19-minute lead time, but generates alert fatigue by falsely flagging over 40% of healthy tasks. Random Forest provides a clinically practical balance: detecting nearly two-thirds of failures 16 minutes in advance while false-alerting on only 1 in 10 healthy tasks.

---

## 7. Supplementary Chronological Experiment

The chronological partition (`data/processed/chronological/`) tests forward temporal forecasting. Models were trained on the early period ($t \le 6000\text{s}$, $N=1,603$) and evaluated on the late test period ($t \ge 10500\text{s}$, $N=460$).

### 7.1 Historical Overlap Limitation

> [!WARNING]
> **Scientific Qualification on Chronological Evaluation:**  
> The 35-minute embargo ($2100.0\text{s}$) strictly prevents target-label leakage ($T_{\text{train\_target\_max}} = 7800.0\text{s} \le 8100.0\text{s}$). However, because the trace duration is 95 minutes and sequence length is $L=15$ (75 minutes), long-running tasks active across epochs share historical observations. Chronological results represent a supplementary proof-of-concept temporal test, not an entity-disjoint benchmark.

### 7.2 Supplementary Chronological Test Results ($N=460$)

| Model | Chrono Macro-F1 | Chrono Accuracy | Warning Recall | Critical Recall | Episode Detection Rate ($N=98$) | Mean Lead Time | False Alarm Rate ($N=144$) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Majority Class** | 0.2599 | 0.6391 | 0.0000 | 0.0000 | 0.0% | 0.0 min | 0.0% |
| **Logistic Regression** | 0.3728 | 0.6087 | 0.0366 | 0.2024 | 28.6% (28/98) | 7.1 min | 16.0% (23/144) |
| **Random Forest** | **0.3978** | **0.6674** | 0.0244 | **0.2381** | 26.5% (26/98) | 7.8 min | **4.9%** (7/144) |

**Analysis:** Chronological performance is lower than entity-grouped performance across all models. This occurs because the early training epoch ($t \le 6000\text{s}$) contains very few failure events (only 42 warning samples out of 1,603), whereas the test epoch ($t \ge 10500\text{s}$) has high failure activity (82 warning and 84 critical samples out of 460). The temporal shift in cluster workload makes forward generalization challenging for static baselines, creating a strong motivation for sequential temporal architectures in Phase 3B.

---

## 8. Artifact Manifest and Reproducibility Record

All predictions and metrics have been generated and serialized with exact reproducibility.

### 8.1 Machine-Readable Artifact Manifest

| Artifact Path | Format | Description |
|:---|:---:|:---|
| [`results/phase3a/baseline_metrics.json`](file:///c:/Users/hp/Desktop/Research%20project/results/phase3a/baseline_metrics.json) | JSON | Complete numerical metrics bundle (window & episode metrics for all models) |
| [`reports/artifacts/phase3a/baseline_metrics.json`](file:///c:/Users/hp/Desktop/Research%20project/reports/artifacts/phase3a/baseline_metrics.json) | JSON | Identical copy preserved in reports artifacts directory |
| [`results/phase3a/test_predictions_majority.csv`](file:///c:/Users/hp/Desktop/Research%20project/results/phase3a/test_predictions_majority.csv) | CSV | Test window predictions, entity IDs, timestamps, probabilities ($N=1,003$) |
| [`results/phase3a/test_predictions_logistic_regression.csv`](file:///c:/Users/hp/Desktop/Research%20project/results/phase3a/test_predictions_logistic_regression.csv) | CSV | Test window predictions for Logistic Regression ($N=1,003$) |
| [`results/phase3a/test_predictions_random_forest.csv`](file:///c:/Users/hp/Desktop/Research%20project/results/phase3a/test_predictions_random_forest.csv) | CSV | Test window predictions for Random Forest ($N=1,003$) |
| [`reports/artifacts/phase3a/test_predictions_*.csv`](file:///c:/Users/hp/Desktop/Research%20project/reports/artifacts/phase3a/) | CSV | Complete mirror of test prediction CSVs in reports artifacts |

### 8.2 Execution Environment Specifications
- **Python Version:** `3.11.9 (tags/v3.11.9:de54cf5, Apr 2 2024)`
- **Operating System:** Windows 11 (AMD64)
- **Scikit-Learn Version:** `1.9.1`
- **NumPy Version:** `2.4.6`
- **Pandas Version:** `2.3.3`
- **Random Seed:** `42` across all models and test assertions

### 8.3 Automated Test Suite Status
All **25 unit tests** in the project test suite passed cleanly (`pytest tests/`):
- 5 feature engineering and schema tests: **PASSED**
- 3 data splitting and embargo tests: **PASSED**
- 2 scaling and imputation tests: **PASSED**
- 3 target labeling horizon tests: **PASSED**
- 2 window generation history tests: **PASSED**
- 4 schema error handling tests: **PASSED**
- 6 new Phase 3A baseline and metric computation tests ([`tests/test_phase3a_baselines.py`](file:///c:/Users/hp/Desktop/Research%20project/tests/test_phase3a_baselines.py)): **PASSED**

---

## 9. Readiness Assessment for Phase 3B (Temporal LSTM)

Phase 3A is complete. The baseline results establish the following comparative foundation for **Phase 3B — Temporal LSTM Framework**:

1. **Primary Target Benchmark:**
   - **Target Macro-F1 to beat:** **0.5718** (established by Class-Weighted Random Forest on held-out test data).
   - **Target Episode Detection:** $> 64\%$ with false alarm rate $\le 10\%$.
2. **Key Challenge Identified:**
   - The primary weakness of tabular baselines is early warning detection (Warning Recall = $25.0\%$ in Random Forest, F1 = $0.2881$). Tabular models struggle to differentiate subtle early degradation from background resource noise.
   - The Temporal LSTM will leverage full multi-step temporal transitions $(x_1, \dots, x_{15})$ through recurrent gates to recognize subtle, sequential failure trajectories earlier in the task lifecycle.
3. **No-Deep-Model Verification:**
   - Exactly zero PyTorch neural networks (LSTM or Transformer) were instantiated or trained during Phase 3A.

$$\mathbf{PHASE \; 3A \; COMPLETE: \; READY \; FOR \; PHASE \; 3B \; (LSTM \; MODELING)}$$
