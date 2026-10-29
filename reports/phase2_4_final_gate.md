# Phase 2.4 — Final Go/No-Go Gate for Phase 3

**Project:** Temporal LSTM/Transformer Framework for Early Failure Prediction in Cloud Computing  
**Phase:** 2.4 — Final Gate & Verification Checkpoint (Pre-Training)  
**Execution Date:** 2026-10-26  
**Auditor:** Senior ML Engineer & Independent Experimental Auditor  
**Status:** **GO FOR PHASE 3** (All Invariants & Criteria Verified; Zero Models Fitted)

---

## Executive Summary

Phase 2.4 serves as the formal gate before model training. An independent mathematical and forensic audit was executed against the raw data shards, preprocessing pipeline, processed array tensors, and metadata artifacts. 

**Core Gate Invariants Verified:**
1. **Chronological Embargo:** Mathematically verified on the discrete 300s temporal grid. With a future lookahead horizon of $H_{\text{warning}} = 1800.0\text{s}$ (30 min), the purged interval between partitions is $2100.0\text{s}$ (35 min), providing an exact remaining safety buffer of $300.0\text{s}$ (5 min) with zero target-label leakage.
2. **Entity-Grouped Isolation:** Recomputed unique entity IDs independently across partitions ($N_{\text{train}} = 367$, $N_{\text{val}} = 78$, $N_{\text{test}} = 80$, total 525). Pairwise entity intersections are strictly empty. All 304 verified failure episodes are completely contained within their respective partitions with zero episode crossing.
3. **Preprocessing Causality & Scaler Isolation:** All 31 engineered features are computed strictly grouped by `entity_id` using trailing operators (no global statistics). Scalers and median imputers were fitted strictly on training partitions (`train_flat_df`), verified by disjoint parameter values across `configs/fitted_scaler.json` and `configs/fitted_scaler_chronological.json`.
4. **Failure-Event & Label Semantics:** Borg 2011 lifecycle event codes (`FAIL == 3`, `FINISH == 4`, `KILL == 5`, `EVICT == 2`) are correctly mapped. Future labels evaluate events strictly occurring at $t > t_{\text{pred}}$. The 924 pre-failure windows in the dataset represent 304 distinct failure episodes ($3.04\times$ ratio), establishing the invariant that sliding windows are not independent failures.
5. **Phase 3 Evaluation Plan:** Formulated completely without fitting any predictive models.
6. **No-Training Rule:** Strictly observed. No baseline, LSTM, or Transformer models were fitted during Phase 2.4.

---

## 1. Mathematical Verification of Chronological Embargo

The supplementary chronological split (`data/processed/chronological/`) tests temporal forward forecasting. In accordance with Master Engineering Prompt Section 2.8, partitions must be separated by an embargo interval at least equal to the maximum future-label horizon ($H_{\text{warning}} = 1800.0\text{s}$) to prevent future target outcomes from leaking into earlier training or validation labels.

### 1.1 Partition Boundaries and Inclusivity

Telemetry observations in the processed dataset reside on a 300-second grid spanning $[5100.0\text{s}, 10800.0\text{s}]$ (total prediction window span: 5,700 seconds = 95.0 minutes). On this discrete grid, each partition is an **inclusive closed interval** $[t_{\min}, t_{\max}]$:

| Partition | Prediction Timestamp Range ($t_{\text{pred}}$) | Boundary Policy | Discrete Grid Timestamps | Window Count ($N$) |
|:---|:---:|:---:|:---|:---:|
| **Train** | $[5100.0\text{s}, 6000.0\text{s}]$ | Inclusive | $\{5100.0, 5400.0, 5700.0, 6000.0\}$ (4 steps) | 1,603 |
| **Purged Embargo 1** | $(6000.0\text{s}, 8100.0\text{s})$ | Excluded | $\{6300.0, 6600.0, 6900.0, 7200.0, 7500.0, 7800.0\}$ (6 steps) | 2,390 |
| **Validation** | $[8100.0\text{s}, 8400.0\text{s}]$ | Inclusive | $\{8100.0, 8400.0\}$ (2 steps) | 724 |
| **Purged Embargo 2** | $(8400.0\text{s}, 10500.0\text{s})$ | Excluded | $\{8700.0, 9000.0, 9300.0, 9600.0, 9900.0, 10200.0\}$ (6 steps) | 1,639 |
| **Test** | $[10500.0\text{s}, 10800.0\text{s}]$ | Inclusive | $\{10500.0, 10800.0\}$ (2 steps) | 460 |
| **Total** | $[5100.0\text{s}, 10800.0\text{s}]$ | — | 20 discrete 300s steps | **6,816** |

### 1.2 Target Lookahead Horizon Calculation

For any window with prediction timestamp $t_{\text{pred}}$, target label generation inspects future lifecycle events occurring strictly in the interval $(t_{\text{pred}}, t_{\text{pred}} + H_{\text{warning}}]$ where $H_{\text{warning}} = 1800.0\text{s}$ (30.0 minutes).

For each partition, the maximum future-label horizon is given by:
$$T_{\text{target\_max}} = t_{\text{partition\_max}} + H_{\text{warning}}$$

1. **Training Partition Target Horizon:**
   $$t_{\text{train\_max}} = 6000.0\text{s}$$
   $$T_{\text{train\_target\_max}} = 6000.0\text{s} + 1800.0\text{s} = 7800.0\text{s}$$
   All training labels are evaluated on events occurring at $t \le 7800.0\text{s}$.

2. **Validation Partition Target Horizon:**
   $$t_{\text{val\_max}} = 8400.0\text{s}$$
   $$T_{\text{val\_target\_max}} = 8400.0\text{s} + 1800.0\text{s} = 10200.0\text{s}$$
   All validation labels are evaluated on events occurring at $t \le 10200.0\text{s}$.

3. **Test Partition Target Horizon:**
   $$t_{\text{test\_max}} = 10800.0\text{s}$$
   $$T_{\text{test\_target\_max}} = 10800.0\text{s} + 1800.0\text{s} = 12600.0\text{s}$$

### 1.3 Mathematical Embargo Accounting

The audit separates the purged interval duration, the future-label horizon duration, and the remaining safety buffer:

$$\Delta t_{\text{purged}} = t_{\text{next\_min}} - t_{\text{prev\_max}}$$
$$\text{Remaining Safety Gap} = \Delta t_{\text{purged}} - H_{\text{warning}} = t_{\text{next\_min}} - T_{\text{prev\_target\_max}}$$

| Split Boundary | Purged Interval ($\Delta t_{\text{purged}}$) | Future-Label Horizon ($H_{\text{warning}}$) | Remaining Safety Gap ($\text{Gap}$) | Zero-Overlap Status |
|:---|:---:|:---:|:---:|:---:|
| **Train $\to$ Val** | $8100.0 - 6000.0 = \mathbf{2100.0\text{s}}$ (35.0 min) | $\mathbf{1800.0\text{s}}$ (30.0 min) | $8100.0 - 7800.0 = \mathbf{300.0\text{s}}$ (5.0 min) | **PASS** ($\text{Gap} > 0$) |
| **Val $\to$ Test** | $10500.0 - 8400.0 = \mathbf{2100.0\text{s}}$ (35.0 min) | $\mathbf{1800.0\text{s}}$ (30.0 min) | $10500.0 - 10200.0 = \mathbf{300.0\text{s}}$ (5.0 min) | **PASS** ($\text{Gap} > 0$) |

### 1.4 Clarification of Prior Descriptions

In earlier Phase 2.2 and 2.3 summaries, the embargo was colloquially described as a "30-minute embargo". While the required lookahead buffer is 30 minutes ($1800.0\text{s}$), the discrete 300s grid enforces partition start times at the next grid step ($8100.0\text{s}$ and $10500.0\text{s}$). Consequently:
- Exactly **6 grid steps** (35.0 minutes / $2100.0\text{s}$) are purged at each boundary.
- An additional **5.0-minute ($300.0\text{s}$) safety margin** exists between the latest possible training event ($7800.0\text{s}$) and the earliest validation observation ($8100.0\text{s}$).
- Mathematically, **zero target-label leakage exists across chronological partitions**.

---

## 2. Independent Verification of Entity-Grouped Isolation

The primary evaluation dataset (`data/processed/`) uses entity-grouped splitting to test generalization to unseen task instances.

### 2.1 Independent Recomputation of Entity IDs

Entity IDs (`job_id_task_index`) were extracted independently from `train_metadata.csv`, `val_metadata.csv`, and `test_metadata.csv`:

```
Train unique entities: 367
Val unique entities  :  78
Test unique entities :  80
Total unique entities: 525
```

Pairwise set intersections were computed:
$$\text{Entities}(D_{\text{train}}) \cap \text{Entities}(D_{\text{val}}) = \emptyset \quad (|0|)$$
$$\text{Entities}(D_{\text{train}}) \cap \text{Entities}(D_{\text{test}}) = \emptyset \quad (|0|)$$
$$\text{Entities}(D_{\text{val}}) \cap \text{Entities}(D_{\text{test}}) = \emptyset \quad (|0|)$$

**Verdict: Pairwise entity intersections are strictly empty.**

### 2.2 Failure Episode Containment

A failure episode is defined by the unique tuple $(entity\_id, T_{\text{fail}})$. In the entity-grouped split:
- **Train failure episodes:** 207 distinct episodes
- **Val failure episodes:** 61 distinct episodes
- **Test failure episodes:** 36 distinct episodes
- **Total failure episodes:** 304 distinct episodes

Cross-split episode overlaps:
$$\text{Episodes}(D_{\text{train}}) \cap \text{Episodes}(D_{\text{val}}) = \emptyset \quad (|0|)$$
$$\text{Episodes}(D_{\text{train}}) \cap \text{Episodes}(D_{\text{test}}) = \emptyset \quad (|0|)$$
$$\text{Episodes}(D_{\text{val}}) \cap \text{Episodes}(D_{\text{test}}) = \emptyset \quad (|0|)$$

**Verdict: No failure episode crosses partitions. Every failed task's complete history, alerts, and terminal failure reside strictly within one partition.**

### 2.3 Feature Engineering and Preprocessing Audit

Code inspection of [`src/features/engineer.py`](file:///c:/Users/hp/Desktop/Research%20project/src/features/engineer.py) verified:
1. All feature transforms are executed on `df.groupby("entity_id", sort=False)`.
2. All mathematical operators are strictly causal and trailing:
   - Group B lags: `shift(1)`, `shift(2)`, `shift(3)`, `shift(6)`.
   - Group C differences: $x_t - x_{t-1}$.
   - Group D rolling statistics: `rolling(w).mean()`, `rolling(w).std(ddof=0)`, `rolling(w).max() - rolling(w).min()`.
   - Group E trend: analytical ordinary least squares slope over trailing window of 6 steps ($w=6$).
   - Group F exceedance: trailing fraction of steps exceeding threshold.
   - Group H age: $t_{\text{end}} - \min(t_{\text{start}})$ per entity.
3. **Global Statistics Check:** Exactly **zero** global dataset statistics (no global means, variances, percentiles, or clustering centers) were computed prior to splitting.

### 2.4 Scaler and Imputation Fitting Isolation

The state files [`configs/fitted_scaler.json`](file:///c:/Users/hp/Desktop/Research%20project/configs/fitted_scaler.json) and [`configs/fitted_scaler_chronological.json`](file:///c:/Users/hp/Desktop/Research%20project/configs/fitted_scaler_chronological.json) were inspected:

```json
{
  "Entity-Grouped Scaler": {
    "is_fitted": true,
    "feature_count": 31,
    "cpu_utilization_mean": 0.027444683015346527,
    "cpu_utilization_std": 0.03623340278863907
  },
  "Chronological Scaler": {
    "is_fitted": true,
    "feature_count": 31,
    "cpu_utilization_mean": 0.02753026969730854,
    "cpu_utilization_std": 0.03842762112617493
  }
}
```

- In `scripts/run_phase2_pipeline.py`, the scaler is fitted strictly on `train_flat_df = X[train_idx].reshape(-1, 31)`. Validation and test tensors are transformed using the training parameters.
- In `scripts/build_chronological_split.py`, the chronological scaler is fitted strictly on the chronological training subset ($t \le 6000.0\text{s}$).
- The distinct parameter values ($0.027445$ vs $0.027530$) confirm that each scaler reflects only its training subset.

### 2.5 Leakage Distinction Matrix

To prevent misleading blanket claims, the audit categorizes five distinct types of leakage and specifies their exact status in both splits:

| Leakage Category | Description | Primary Split (Entity-Grouped) | Supplementary Split (Chronological) |
|:---|:---|:---:|:---:|
| **Target-Label Leakage** | Future failure events leaking into earlier training labels | **ELIMINATED** (Disjoint task lifetimes) | **ELIMINATED** (35-min embargo $> 30$-min horizon) |
| **Task Identity Leakage** | Same task instance present in multiple partitions | **ELIMINATED** ($|Train \cap Val \cap Test| = 0$) | **PRESENT** (Long-running tasks persist across epochs) |
| **Episode Leakage** | Pre-failure windows for one failure split across sets | **ELIMINATED** (Zero episode overlap) | **ELIMINATED** (Separated by 35-min embargo) |
| **Scaler / Imputer Leakage** | Normalization parameters fitted on validation or test sets | **ELIMINATED** (Fitted strictly on train) | **ELIMINATED** (Fitted strictly on train) |
| **Historical Timeline Overlap** | Trailing telemetry history ($L=15$, 75 min) spans across split boundary | **ELIMINATED** (Disjoint tasks) | **PRESENT** (Tasks active at 8100s share history from $< 6000$s) |
| **Global Concurrency Effects** | Concurrent cluster load / co-location during same trace epoch | **PRESENT** (Tasks co-existed in Borg cluster) | **PRESENT** (Continuous cluster trace snippet) |

**Scientific Conclusion:** Verified entity isolation strictly eliminates task identity, failure episode, and scaler leakage, ensuring an unbiased test of *unseen entity generalization*. It does not eliminate cluster-level co-location effects, nor should it be conflated with chronological isolation.

---

## 3. Failure-Event and Label Semantics Verification

### 3.1 Trace Version and Event-Code Mapping

The raw dataset in `data/raw/` is Google Cluster Trace 2011-2 (ClusterData 2011-2, Reiss et al. 2011). The mapping in [`src/ingestion/schema.py`](file:///c:/Users/hp/Desktop/Research%20project/src/ingestion/schema.py) was cross-checked against Borg documentation:

| Lifecycle Event | Borg 2011 (v2) Code (`ClusterEventTypeV2`) | Borg 2019 (v3) Code (`ClusterEventTypeV3`) | Semantics in Project Pipeline |
|:---|:---:|:---:|:---|
| **SUBMIT** | 0 | 0 | Task eligible for scheduling |
| **SCHEDULE** | 1 | 3 | Task scheduled to a machine |
| **EVICT** | 2 | 4 | Descheduled due to priority/preemption (Censored: `label = -1`) |
| **FAIL** | **3** | **5** | **Abnormal execution termination** (Target failure: `label = 1` or `2`) |
| **FINISH** | **4** | **6** | **Normal execution completion** (Normal run: `label = 0`) |
| **KILL** | 5 | 7 | Cancelled by user or dependency (Censored: `label = -1`) |
| **LOST** | 6 | 8 | Disappeared from trace |

The project uses `get_event_classifier("2011")`, which binds `ClusterEventTypeV2`.
- `ClusterEventTypeV2.FAIL == 3` (verified abnormal execution failure).
- `ClusterEventTypeV2.FINISH == 4` (verified successful execution completion).

### 3.2 Prediction-Time Horizon Semantics

In [`src/labeling/target_generator.py`](file:///c:/Users/hp/Desktop/Research%20project/src/labeling/target_generator.py), the prediction timestamp $t_{\text{pred}}$ is the end timestamp of the 15th observation step.
1. Only events with `time_sec > t_pred` are evaluated.
2. Let $\tau = T_{\text{fail}} - t_{\text{pred}}$:
   - **CRITICAL (`label = 2`):** $0.0\text{s} < \tau \le 900.0\text{s}$ (0 to 15 minutes before failure).
   - **WARNING (`label = 1`):** $900.0\text{s} < \tau \le 1800.0\text{s}$ (15 to 30 minutes before failure).
   - **NORMAL (`label = 0`):** No qualifying failure in $(t_{\text{pred}}, t_{\text{pred}} + 1800.0\text{s}]$, and verified non-failure (telemetry extends $\ge 1800.0\text{s}$ or normal `FINISH` event observed).
   - **UNKNOWN (`label = -1`):** Censored termination (`KILL`, `EVICT`) within 30 minutes, or telemetry ends prematurely at shard boundary. All `UNKNOWN` windows are strictly excluded from sliding window tensor construction.

### 3.3 Episode Grouping and Window-to-Episode Ratio

In telemetry monitoring, sliding windows advance incrementally (step size = 300s). Consequently, a single approaching failure produces a succession of warning and critical alert windows:

```
Timeline of a Failed Task:
[--- Step 1 ---][--- Step 2 ---][--- Step 3 ---][--- Step 4 ---] -> [FAIL Event at T_fail]
                 t_pred = T-1500s  t_pred = T-1200s  t_pred = T-900s   t_pred = T-600s
                 (tau = 1500s)     (tau = 1200s)     (tau = 900s)      (tau = 600s)
                 [WARNING (1)]     [WARNING (1)]     [CRITICAL (2)]    [CRITICAL (2)]
```

Empirical accounting across the 6,816 primary windows:
- **Unique Failure Episodes:** 304 distinct episodes
- **Total Pre-Failure Windows:** 924 windows
  - Warning windows ($15\text{m} < \tau \le 30\text{m}$): 387 windows
  - Critical windows ($0\text{m} < \tau \le 15\text{m}$): 537 windows
- **Normal Windows:** 5,892 windows
- **Average Windows per Failure Episode:** $\frac{924}{304} = \mathbf{3.04\times}$

**Invariant for Evaluation:** Pre-failure windows are sequentially correlated samples of an underlying failure episode. Reporting 924 windows as "924 failures" would overestimate statistical sample size by $3.04\times$. Therefore, Phase 3 must report both **window-level metrics** (per-step alert accuracy) and **episode-level metrics** (fraction of distinct failure episodes detected by at least one alert prior to failure).

---

## 4. Phase 3 Evaluation Plan

The Phase 3 evaluation plan specifies all models, metrics, protocols, and safeguards. **No models are fitted in Phase 2.4.**

```
Phase 3 Evaluation Architecture
+-----------------------------------------------------------------------------------+
| 1. DATASETS                                                                       |
|    - Primary Benchmark: Entity-Grouped Split (N=6,816; Train=4798, Val=1015, Test=1003) |
|    - Supplementary Test: Chronological Split (N=2,787; Train=1603, Val=724, Test=460)  |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
| 2. MODEL SUITE                                                                    |
|    - Baseline 1: Majority-Class Classifier (Predicts class 0 always)              |
|    - Baseline 2: Class-Weighted Logistic Regression (Flattened 15x31 = 465-dim)   |
|    - Baseline 3: Random Forest Classifier (100 estimators, max_depth=12)          |
|    - Deep Model: Bidirectional / Unidirectional Temporal LSTM (Hidden=64, Layers=2)|
|    - Conditional: Temporal Transformer (Only if LSTM macro-F1 & data warrant it)  |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
| 3. TRAINING SAFEGUARDS                                                            |
|    - Class-imbalance weights computed strictly from Train partition               |
|    - Fixed random seed = 42 for all initializations, batches, and splits          |
|    - Early stopping: Monitored on Validation Loss / Macro-F1 (patience=10 epochs) |
|    - Test Set: Held out completely; evaluated exactly ONCE on final selected model|
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
| 4. DUAL EVALUATION METRICS                                                         |
|    - Window-Level: Macro-F1, Per-Class P/R/F1, Confusion Matrix, PR-AUC          |
|    - Episode-Level: Episode Detection Rate, Earliest Lead Time, False Episode Rate|
+-----------------------------------------------------------------------------------+
```

### 4.1 Primary vs. Supplementary Split Architecture

1. **Primary Benchmark: Entity-Grouped Partitioning (`data/processed/`)**
   - **Role:** The primary scientific benchmark evaluating generalization to unseen tasks, jobs, and containers.
   - **Composition:** Train ($N=4,798$), Validation ($N=1,015$), Test ($N=1,003$). Total $N=6,816$.
   - **Integrity:** Zero entity overlap ($|Train \cap Val \cap Test| = 0$), zero failure episode crossing, scalers fitted on train only.
2. **Supplementary Benchmark: Chronological Partitioning (`data/processed/chronological/`)**
   - **Role:** A secondary proof-of-concept test evaluating forward temporal forecasting.
   - **Composition:** Train ($N=1,603$), Validation ($N=724$), Test ($N=460$). Total $N=2,787$.
   - **Explicit Qualification:** The short trace duration (95 min prediction span) allows a 35-minute target-label embargo ($2100.0\text{s}$), completely preventing future label leakage. However, long-running tasks active across epochs share trailing historical observations ($L=15$, 75 min). All chronological results must be explicitly reported with this qualification.

### 4.2 Candidate Models

1. **Baseline 1 — Majority-Class Classifier:**
   - Always predicts the dominant class (`NORMAL = 0`).
   - Serves as the naive floor for accuracy, precision, and macro-F1.
2. **Baseline 2 — Class-Weighted Logistic Regression:**
   - Input: Flattened feature tensor $x_i \in \mathbb{R}^{15 \times 31} \to \mathbb{R}^{465}$.
   - Regularization: $L_2$ penalty, multinomial loss, balanced inverse-class weighting.
3. **Baseline 3 — Random Forest Classifier:**
   - Input: Flattened feature tensor $x_i \in \mathbb{R}^{465}$ or step-aggregated summary features (mean, min, max, slope across 15 steps).
   - Hyperparameters: $n_{\text{estimators}} = 100$, $\text{max\_depth} = 12$, $\text{class\_weight} = \text{"balanced"}$, fixed random seed = 42.
4. **Deep Model — Temporal LSTM:**
   - Architecture: 2-layer LSTM, hidden dimension $H \in \{32, 64\}$, dropout $= 0.2$, linear classification head $\mathbb{R}^H \to \mathbb{R}^3$.
   - Sequential processing: Consumes 15 sequential timesteps $(x_1, \dots, x_{15})$ preserving temporal autocorrelation without flattening.
   - Optimization: AdamW ($\text{lr} = 10^{-3}$, weight decay $= 10^{-4}$), batch size = 64.
5. **Conditional Deep Model — Temporal Transformer:**
   - **Trigger Condition:** Implemented only if the LSTM achieves validation macro-F1 $\ge 0.50$ and training curves show stable gradient propagation on the available dataset size. If the dataset size ($N_{\text{train}} = 4,798$) leads to severe Transformer overfitting, this limitation will be documented and the LSTM will remain the primary deep model.

### 4.3 Evaluation Metrics Suite

For 3-class classification ($\mathcal{C} \in \{0: \text{NORMAL}, 1: \text{WARNING}, 2: \text{CRITICAL}\}$):

1. **Per-Class Metrics:**
   - Precision: $P_c = \frac{TP_c}{TP_c + FP_c}$
   - Recall: $R_c = \frac{TP_c}{TP_c + FN_c}$
   - F1-Score: $F1_c = \frac{2 P_c R_c}{P_c + R_c}$
2. **Global Classification Performance:**
   - **Macro-F1:** $\text{Macro-F1} = \frac{1}{3} \sum_{c=0}^2 F1_c$ (Primary model selection metric, giving equal weight to minority failure classes).
   - **Confusion Matrix:** $3 \times 3$ normalized confusion matrix showing misclassification flow (e.g. Critical mispredicted as Warning vs. Normal).
   - **PR-AUC (Precision-Recall Area Under Curve):** Computed for Warning and Critical classes using continuous class probabilities.
   - **Failure Alert Recall:** Total recall of approaching failures:
     $$\text{Alert Recall} = \frac{TP_{\text{warning}} + TP_{\text{critical}}}{N_{\text{actual\_failures}}}$$
3. **Episode-Level Failure Detection Metrics:**
   - **Episode Detection Rate:** Percentage of distinct failure episodes ($N=36$ in test) where at least one pre-failure window ($0 < \tau \le 30\text{m}$) generated a correct Warning or Critical alert.
   - **Earliest Lead Time:** Time difference $\tau$ between the first triggered alert and the actual failure timestamp $T_{\text{fail}}$ (ideal target: $15\text{m} < \tau \le 30\text{m}$).
   - **False Episode Alert Rate:** Rate of false alarm episodes triggered on tasks that finished normally.

### 4.4 Imbalance Handling and Training Safeguards

1. **Class-Imbalance Handling:**
   - Class distribution in training data: Normal: 4,188 (87.3%), Critical: 363 (7.6%), Warning: 247 (5.1%).
   - Class weights: Computed strictly from the training partition:
     $$w_c = \frac{N_{\text{train}}}{3 \cdot N_{\text{train}, c}}$$
   - Loss Function: Weighted Cross-Entropy Loss:
     $$\mathcal{L} = -\sum_{c=0}^2 w_c \, y_c \log(\hat{p}_c)$$
     or multi-class Focal Loss ($\gamma = 2.0$) if cross-entropy suffers from over-confident majority predictions.
2. **Reproducibility:**
   - Fixed random seeds: `seed = 42` for PyTorch (`torch.manual_seed`), NumPy (`np.random.seed`), and Python (`random.seed`).
   - Deterministic CuDNN/CPU flags enabled.
3. **Validation and Early Stopping:**
   - Validation set is used solely for hyperparameter tuning and early stopping.
   - Early stopping criterion: Validation loss or Validation Macro-F1 with patience $= 10$ epochs.
4. **Test Set Isolation:**
   - Test partition ($N=1,003$) is held out completely during model exploration and hyperparameter search.
   - Evaluated exactly **once** on the best checkpoint of each model family to generate the final reported metrics.

### 4.5 Dataset Limitations and Scientific Qualifications

The Phase 3 report must explicitly document the following empirical constraints:
1. **Sample Size:** $N=6,816$ total windows ($4,798$ training windows) derived from 525 unique task instances and 304 failure episodes. While sufficient for statistical baselines and compact LSTMs, this is a constrained sample size for large multi-head Transformers.
2. **Observation Horizon:** Raw trace snippets cover ~3 hours of Borg cell operation. Long-term seasonal effects (multi-day or weekly cycles) cannot be observed.
3. **Chronological Generalization:** As demonstrated in Section 1, the chronological split successfully isolates target labels via a 35-minute embargo, but retains historical overlap for long-lived tasks. Chronological results will be reported as supplementary proof-of-concept findings.

---

## 5. Master Scorecard and Gate Decision

The 7 formal verification criteria were audited using `scripts/phase2_4_final_gate.py` and the 19 pytest test cases:

| Criterion | Audit Requirement | Verification Method | Status |
|:---|:---|:---|:---:|
| **Criterion 1** | Chronological Embargo & Boundary Accounting | Mathematical calculation of $t_{\text{pred}}$, 35m purged intervals, and 5m remaining safety buffers | **PASS** |
| **Criterion 2** | Entity-Grouped Isolation & Episode Containment | Independent recomputation of entity sets ($|A \cap B| = 0$) and failure episode containment | **PASS** |
| **Criterion 3** | Preprocessing & Scaler Training-Only Fitting | Source code audit of `FeatureEngineer` and distinct fitted parameters in scaler JSONs | **PASS** |
| **Criterion 4** | Borg 2011 vs 2019 Lifecycle Event Mapping | Verification of `ClusterEventTypeV2.FAIL == 3` and `FINISH == 4` | **PASS** |
| **Criterion 5** | Label Horizon Definitions (0-15m Crit, 15-30m Warn) | Mathematical verification of $\tau = T_{\text{fail}} - t_{\text{pred}}$ strictly for future events ($t > t_{\text{pred}}$) | **PASS** |
| **Criterion 6** | Episode vs Sliding-Window Semantic Distinction | Empirical verification of 304 episodes producing 924 windows ($3.04\times$ ratio) | **PASS** |
| **Criterion 7** | Dataset Shapes, NaNs, and Artifact Consistency | Exact tensor dimension checks ($(N, 15, 31)$), 0 NaNs, full metadata alignment | **PASS** |
| **Test Suite** | Comprehensive Unit Tests (`pytest tests/`) | 19 unit tests covering schema, features, splits, scalers, and horizons | **PASS** (19/19) |
| **No-Training Rule** | Zero models trained during Phase 2.4 | System-wide process check; no weights or model artifacts created | **PASS** |

### Gate Verdict

$$\mathbf{VERDICT: \; ALL \; CRITERIA \; PASSED \; \longrightarrow \; GO \; FOR \; PHASE \; 3}$$

All empirical data, scripts, and artifacts required for Phase 3 model training are verified, causal, leakage-safe, and ready for baseline and LSTM implementation.
