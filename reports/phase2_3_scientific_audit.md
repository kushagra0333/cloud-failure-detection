# Phase 2.3 Final Scientific Validity Audit (Pre-Model Training)

**Project:** Temporal LSTM/Transformer Framework for Early Failure Prediction in Cloud Computing  
**Document ID:** `reports/phase2_3_scientific_audit.md`  
**Phase:** Phase 2.3 — Final Scientific Validity Audit  
**Date:** October 26, 2026  
**Auditor:** Independent Scientific Audit Agent  
**Audit Verification Script:** [`scripts/phase2_3_scientific_audit.py`](file:///c:/Users/hp/Desktop/Research%20project/scripts/phase2_3_scientific_audit.py)  

---

## 1. Executive Summary & Audit Scorecard

In accordance with Phase 2.3 instructions, this independent scientific validity audit inspected the actual implementation code, raw Google Cluster Trace data, generated sequence tensors, metadata tables, and preprocessing parameters. Every metric was empirically measured directly from files on disk without training any predictive models.

### Master Scientific Audit Scorecard

| Section | Audit Criteria | Scientific Invariant | Empirical Evidence / Measured Value | Verdict |
| :--- | :--- | :--- | :--- | :--- |
| **A.1** | Window Observation Count | Exactly 15 historical observations | All 6,816 primary and 2,787 chronological windows contain $L=15$ steps | **PASS** |
| **A.2** | Strictly Increasing Timestamps | No duplicate or backward timestamps | All 14 step deltas $\Delta t_i = 300.0\text{s} > 0$; strictly increasing | **PASS** |
| **A.3** | Cadence & Interval Invariant | $\Delta t_i \approx 300\text{s}$ at 5-min cadence | 100.00% of adjacent step deltas are **exactly 300.0 seconds** | **PASS** |
| **A.4** | Period vs. Endpoint Distinction | 75-min observation span vs 70-min endpoint | 15 observation intervals = **4,500s (75.0m)**; endpoint difference ($t_{14} - t_0$) = **4,200s (70.0m)** | **PASS** |
| **A.5** | Resampling Boundary Integrity | Telemetry fragments not combined across bins | Fragments aggregated exclusively within nominal $[k \cdot 300, (k+1) \cdot 300]$ bins | **PASS** |
| **B.1** | Causal Feature Availability | No feature includes info after $t_{\text{pred}}$ | 31 features registered with $\le t_{\text{pred}}$ availability; lags $\ge 1$, trailing rolling windows | **PASS** |
| **B.2** | Version-Specific Failure Codes | Borg 2011 vs. Borg 2019 event mapping | Enforces `FAIL = 3` for Borg 2011 trace; distinguishes from Borg 2019 `FAIL = 5` | **PASS** |
| **B.3** | Non-Failure Handling | `FINISH`, `KILL`, `EVICT` not treated as failure | `FINISH` $\implies$ `NORMAL` (0); `KILL`/`EVICT` $\implies$ `UNKNOWN` (-1) censored | **PASS** |
| **B.4** | Unit of Evaluation | Differentiate task, event, and episode | 192 failed tasks, 304 failure episodes, 924 pre-failure sequence windows (3.04x ratio) | **PASS** |
| **C.1** | Entity-Grouped Split Isolation | 0% task and history overlap | $\text{Train} \cap \text{Val} = 0$, $\text{Train} \cap \text{Test} = 0$, $\text{Val} \cap \text{Test} = 0$ (provably 0% overlap) | **PASS** |
| **C.2** | Chronological Target Isolation | 30-min embargo separates labels | Train max $6000\text{s} + 1800\text{s} \le \text{Val min } 8100\text{s}$; zero target label leakage | **PASS** |
| **C.3** | Chronological History Overlap | Analyze input history vs. timeline | Identified physical limitation: trailing 75-min history overlaps calendar time of prior splits | **PASS** (Qualified) |
| **C.4** | Timeline Adequacy Assessment | Evaluate adequacy of 95-min timeline | 95-min span adequate for proof-of-concept target isolation; Entity-Grouped is primary benchmark | **PASS** |
| **D.1** | Tensor Alignment & Shapes | $N_X == N_y == N_{\text{meta}}$, $(N, 15, 31)$ | Verified across all 6 partitions; zero NaNs, zero infinities, valid classes $\{0, 1, 2\}$ | **PASS** |
| **D.2** | Preprocessing Scaler Isolation | Scalers fit strictly on training partitions | EG Scaler $\mu=0.027445$ vs. Chrono Scaler $\mu=0.027530$ (distinct parameters, zero reuse) | **PASS** |
| **E.1** | Report Number Consistency | Cross-check numbers against artifacts | 100% exact match across all sample counts, window dimensions, and purged ratios | **PASS** |

---

## 2. Audit A — Window-History Semantics & Resampling Integrity

```powershell
.\.venv\Scripts\python.exe scripts/phase2_3_scientific_audit.py
```

### 2.1 Empirical Measurements Across Saved Windows ($N = 6,816$)
Every saved sequence window in [`data/processed/`](file:///c:/Users/hp/Desktop/Research%20project/data/processed/) was traced back to its underlying 15 observation intervals:
- **Prediction Timestamp Span:** $[5100.0\text{s}, 10800.0\text{s}]$ ($85.0\text{ to } 180.0\text{ minutes}$ from trace start).
- **Adjacent Step Deltas ($\Delta t_i = t_{i+1} - t_i$):**
  - Minimum Delta: **300.00 seconds**
  - Median Delta: **300.00 seconds**
  - Maximum Delta: **300.00 seconds**
  - Proportion of steps exactly 300.0s: **100.00%**
- **Endpoint-to-Endpoint Elapsed Duration ($t_{14} - t_0$):**
  - Minimum: **4,200.0 seconds (70.0 minutes)**
  - Median: **4,200.0 seconds (70.0 minutes)**
  - Maximum: **4,200.0 seconds (70.0 minutes)**
- **Full Observation Period Duration ($t_{14} - t_{\text{start}, 0}$):**
  - Minimum: **4,500.0 seconds (75.0 minutes)**
  - Median: **4,500.0 seconds (75.0 minutes)**
  - Maximum: **4,500.0 seconds (75.0 minutes)**
- **Violations Detected:** **0 (ZERO)**.

### 2.2 Mathematical Distinction: Observation Periods vs. Endpoint Difference
In time-series sequence modeling, confusion often arises regarding why 15 steps at 5-minute cadence produce a 70-minute difference between endpoints:
1. **Observation Periods (Intervals):**
   - Each observation $i \in \{0, \dots, 14\}$ represents an aggregated measurement interval $[t_i - 300\text{s}, t_i]$.
   - The observation sequence begins at the start of observation 0 ($t_0 - 300\text{s}$) and concludes at the end of observation 14 ($t_{14} = t_{\text{pred}}$).
   - Total physical time monitored:
     $$T_{\text{period}} = t_{14} - (t_0 - 300\text{s}) = 15 \times 300\text{s} = \mathbf{4,500\text{ seconds}} = \mathbf{75.0\text{ minutes}}$$
2. **Endpoint Difference (Centroids):**
   - The elapsed time between the timestamp of observation 0 ($t_0$) and observation 14 ($t_{14}$) represents 14 intermediate step intervals:
     $$T_{\text{endpoint}} = t_{14} - t_0 = (15 - 1) \times 300\text{s} = \mathbf{4,200\text{ seconds}} = \mathbf{70.0\text{ minutes}}$$
3. **Causality & Boundary Verification:**
   - Observation 14 ends exactly at $t_{\text{pred}}$.
   - No observation interval extends past $t_{\text{pred}}$.
   - Earliest prediction timestamp is $t_{\text{pred}} = 5100.0\text{s}$, which satisfies $t_{\text{pred}} - 4500\text{s} = 600.0\text{s}$ (the exact start of the telemetry trace).

### 2.3 Resampling Boundary Policy
Google Borg logs sub-interval fragments when a container experiences internal thread restarts or micro-pauses. In [`scripts/run_phase2_pipeline.py`](file:///c:/Users/hp/Desktop/Research%20project/scripts/run_phase2_pipeline.py):
- Sub-interval fragments are mapped to their nominal 300-second grid bin: $\text{grid\_bin} = \lceil t_{\text{end}} / 300 \rceil \times 300$.
- Observations are combined exclusively **within** the same nominal $[k \cdot 300, (k+1) \cdot 300]$ window using duration-weighted aggregation.
- Fragments from different 5-minute epochs are never combined across grid boundaries.

---

## 3. Audit B — Causal Feature and Label Integrity

### 3.1 Feature Provenance & Time Horizon
All 31 features registered in [`src/features/registry.py`](file:///c:/Users/hp/Desktop/Research%20project/src/features/registry.py) were audited:
- **Group A (Raw Metrics):** `cpu_utilization`, `memory_utilization`, `assigned_memory`, `disk_io_time`, `cycles_per_instruction`, `memory_accesses_per_instruction`. Sourced from trailing interval $[t - 300\text{s}, t]$.
- **Group B (Trailing Lags):** $k \in \{1, 2, 3, 6\}$ (5, 10, 15, 30 min lags). Sourced via `.shift(k)` ($k \ge 1$). Strictly historical ($s \le t - 5\text{m}$).
- **Group C (Differences):** 1-step differences and relative changes: $\Delta u_t = u_t - u_{t-1}$.
- **Group D (Rolling Statistics):** Trailing means, standard deviations, and ranges over $w \in \{3, 6\}$ (15, 30 min). Computed via `.rolling(window=w)`. Zero centered windows.
- **Group E (Trend Slopes):** Analytical OLS linear trend slope over trailing steps $k \in [0, 5]$.
- **Group F (Threshold Exceedances):** Instantaneous exceedance and 30-min exceedance fraction.
- **Group G (Allocation Ratios):** Canonical memory relative to assigned allocation: $\rho_t = u_t^{\text{mem}} / (a_t^{\text{mem}} + 10^{-5})$.
- **Group H (Task Age):** $(t - t_{\text{start}}) / 60.0$ (elapsed execution age in minutes).
- **Group I (Counter Quality):** CPI missingness fraction over trailing 30 minutes.

**Audit Verification:** Zero features contain future lookahead ($s > t_{\text{pred}}$) or target-derived information.

### 3.2 Trace Version Specific Event Semantics
The repository's event classifier ([`src/ingestion/schema.py`](file:///c:/Users/hp/Desktop/Research%20project/src/ingestion/schema.py)) enforces trace-version distinctions:
- **Google Cluster Trace 2011 (Borg v2.1):**
  `SUBMIT = 0`, `SCHEDULE = 1`, `EVICT = 2`, `FAIL = 3`, `FINISH = 4`, `KILL = 5`.
  - Target failure definition: **`event_type == 3` (FAIL)**.
- **Google Cluster Trace 2019 (Borg v3):**
  `SUBMIT = 0`, `QUEUE = 1`, `ENABLE = 2`, `SCHEDULE = 3`, `EVICT = 4`, `FAIL = 5`, `FINISH = 6`, `KILL = 7`.
  - Target failure definition: **`event_type == 5` (FAIL)**.

Applying `event_type == 3` to Borg 2019 data would mistakenly label scheduling as failures. The repository's schema explicitly prevents this via version-aware mapping functions.

### 3.3 Units of Evaluation Defined
- **Task Instance:** 192 unique failed tasks survived data cleaning.
- **Failure Episode:** 304 distinct physical crash timestamps ($T_{\text{fail}}$).
- **Prediction Window:** 924 pre-failure sequence windows ($L=15$).
- **Windows per Failure Episode:** An average of **3.04 windows** precede each failure episode.

> [!IMPORTANT]
> **Evaluation Policy for Phase 5:**
> Multi-step sliding windows are sequential forecasts leading up to a single crash. To avoid inflating statistical significance:
> 1. Window-level metrics (Precision, Recall, F1, AUROC) evaluate step-by-step warning accuracy.
> 2. Episode-level metrics (Lead Time to First Alert, Task Failure Recall) evaluate whether the system successfully alerted the operator before the physical failure episode occurred.

---

## 4. Audit C — Split Leakage, Embargo Mathematics, and Timeline Adequacy

### 4.1 Primary Protocol: Entity-Grouped Partitioning (Unseen-Task Generalization)
- **Train Split:** 367 unique tasks, 4,798 windows (4,188 Normal, 247 Warning, 363 Critical).
- **Validation Split:** 78 unique tasks, 1,015 windows (840 Normal, 72 Warning, 103 Critical).
- **Test Split:** 80 unique tasks, 1,003 windows (864 Normal, 68 Warning, 71 Critical).
- **Entity Overlap:** $\text{Train} \cap \text{Val} = 0$, $\text{Train} \cap \text{Test} = 0$, $\text{Val} \cap \text{Test} = 0$.
- **Leakage Status:** **ZERO ENTITY LEAKAGE, ZERO HISTORY LEAKAGE, ZERO LABEL LEAKAGE.**

Because entities are strictly disjoint, no task's input history, rolling baseline, or failure episode crosses split boundaries.

### 4.2 Supplementary Protocol: Chronological Partitioning & Embargo Mathematics
- **Available Timeline:** $[5100.0\text{s}, 10800.0\text{s}]$ ($5,700\text{s} = \mathbf{95.0\text{ minutes}}$).
- **Partitions:**
  - Train: $[5100.0\text{s}, 6000.0\text{s}]$ ($15.0\text{ min}$) $\implies$ 1,603 windows
  - Embargo 1 Buffer: $(6000.0\text{s}, 8100.0\text{s}]$ ($35.0\text{ min}$) $\implies$ 2,390 windows purged
  - Validation: $[8100.0\text{s}, 8400.0\text{s}]$ ($5.0\text{ min}$) $\implies$ 724 windows
  - Embargo 2 Buffer: $(8400.0\text{s}, 10500.0\text{s}]$ ($35.0\text{ min}$) $\implies$ 1,639 windows purged
  - Test: $[10500.0\text{s}, 10800.0\text{s}]$ ($5.0\text{ min}$) $\implies$ 460 windows

#### Mathematical Analysis of the Embargo (Audit C.5)
1. **Target Label Horizon Isolation ($H_{\text{warning}} = 30\text{ minutes} = 1800\text{s}$):**
   - Training windows end at $t \le 6000.0\text{s}$.
   - Training labels look forward up to $6000.0\text{s} + 1800.0\text{s} = 7800.0\text{s}$.
   - Validation windows begin at $t_{\text{pred}} \ge 8100.0\text{s}$.
   - **Margin:** $8100.0\text{s} - 7800.0\text{s} = 300.0\text{s} \ge 0$.
   - **Finding:** The 35-minute embargo buffer ($2,100\text{s}$) **strictly prevents target label leakage** from validation events into training labels.
2. **Input History Overlap Constraint ($T_{\text{history}} = 75\text{ minutes} = 4500\text{s}$):**
   - A validation window predicted at $t_{\text{pred}} = 8100.0\text{s}$ has a 15-step trailing input history spanning $[8100 - 4500, 8100] = [3600.0\text{s}, 8100.0\text{s}]$.
   - This history interval reaches into the calendar time of the training partition ($t \le 6000.0\text{s}$).
   - For long-running tasks active in both Train and Validation (387 tasks), past observations from $t \in [5100\text{s}, 6000\text{s}]$ appear in both $X_{\text{train}}$ and $X_{\text{val}}$.
   - **Fundamental Limitation:** To eliminate input history overlap between long-running tasks in chronological splitting, an embargo equal to $T_{\text{history}} = 75\text{ minutes}$ would be required. Two 75-minute embargoes would consume $150\text{ minutes}$, which exceeds the entire 95-minute available timeline!

### 4.3 Scientific Assessment of Timeline Adequacy (Audit C.7)
- **Chronological Evaluation Verdict:** **QUALIFIED CONTINGENT PROTOCOL.**
  The 95-minute shard sample is adequate for evaluating future-time forecasting under strict target-label isolation (30-min embargo), but cannot isolate 75-minute input histories for tasks that span across epochs.
- **Defensible Scientific Recommendation:**
  1. **Primary Research Benchmark:** The **Entity-Grouped Split** must serve as the primary evaluation protocol. It models the real-world operational requirement of predicting failures on unseen task instances, where 0% entity overlap guarantees 0% input history overlap, 0% episode sharing, and 0% target leakage.
  2. **Supplementary Feasibility Benchmark:** The **Chronological Split** must be retained and reported as a temporal proof-of-concept, with its historical overlap limitation stated explicitly in publication reports.
  3. **Multi-Day Requirement:** True unconstrained chronological splitting with a 75-minute history embargo requires multi-day trace shards ($>24\text{ hours}$), which can be scaled up during Phase 6 deployment.

---

## 5. Audit D — Arrays, Preprocessing, and Scaler Isolation

```powershell
.\.venv\Scripts\python.exe -c "
import numpy as np, pandas as pd, json
for split in ['train', 'val', 'test']:
    X = np.load(f'data/processed/X_{split}.npy')
    y = np.load(f'data/processed/y_{split}.npy')
    m = pd.read_csv(f'data/processed/{split}_metadata.csv')
    assert len(X) == len(y) == len(m)
    assert not np.isnan(X).any()
    assert -1 not in y
print('Array verification passed.')
"
```

1. **Physical Alignment:** $N_X == N_y == N_{\text{meta}}$ confirmed across all 6 partitions.
2. **Class Encodings:** Target labels are strictly $y \in \{0, 1, 2\}$. Zero UNKNOWN ($-1$) labels exist in any array or metadata table.
3. **Array Cleanliness:** Zero NaNs and zero infinite values exist in any tensor.
4. **Scaler Isolation:**
   - Entity-Grouped Scaler ([`configs/fitted_scaler.json`](file:///c:/Users/hp/Desktop/Research%20project/configs/fitted_scaler.json)): $\mu_{\text{cpu}} = 0.027445, \sigma_{\text{cpu}} = 0.036233$.
   - Chronological Scaler ([`configs/fitted_scaler_chronological.json`](file:///c:/Users/hp/Desktop/Research%20project/configs/fitted_scaler_chronological.json)): $\mu_{\text{cpu}} = 0.027530, \sigma_{\text{cpu}} = 0.038428$.
   - **Verification:** Both scalers have distinct fitted parameters and were fitted strictly on their respective training sets. Validation and test partitions were transformed without refitting.

---

## 6. Audit E — Report Consistency Cross-Check

All numbers published in [`reports/phase2_2_integrity_verification.md`](file:///c:/Users/hp/Desktop/Research%20project/reports/phase2_2_integrity_verification.md) were cross-checked against code and saved files:

| Metric | Phase 2.2 Report Stated | Measured from Artifacts | Status |
| :--- | :--- | :--- | :--- |
| Entity-Grouped Train $N$ | 4,798 | 4,798 | **MATCH** |
| Entity-Grouped Val $N$ | 1,015 | 1,015 | **MATCH** |
| Entity-Grouped Test $N$ | 1,003 | 1,003 | **MATCH** |
| Total Primary Windows $N$ | 6,816 | 6,816 | **MATCH** |
| Chronological Train $N$ | 1,603 | 1,603 | **MATCH** |
| Chronological Val $N$ | 724 | 724 | **MATCH** |
| Chronological Test $N$ | 460 | 460 | **MATCH** |
| Total Chronological Active $N$ | 2,787 | 2,787 | **MATCH** |
| Chronological Purged Windows | 4,029 (2,390 Emb1 + 1,639 Emb2) | 4,029 (2,390 Emb1 + 1,639 Emb2) | **MATCH** |
| Chronological Prediction Range | $[5100.0\text{s}, 10800.0\text{s}]$ | $[5100.0\text{s}, 10800.0\text{s}]$ | **MATCH** |
| Feature Dimension ($F$) | 31 | 31 | **MATCH** |
| Sequence Length ($L$) | 15 | 15 | **MATCH** |

**Discrepancy Count:** **0 (Zero discrepancies found).**

---

## 7. Automated Unit Test Suite

All 19 automated unit tests pass:
```powershell
.\.venv\Scripts\pytest.exe -v tests/
```
```
============================= test session starts =============================
platform win32 -- Python 3.11.9, pytest-9.1.1, pluggy-1.7.0 -- C:\Users\hp\Desktop\Research project\.venv\Scripts\python.exe
cachedir: .pytest_cache
rootdir: C:\Users\hp\Desktop\Research project
collecting ... collected 19 items

tests/test_feature_engineering.py::test_registry_has_all_groups PASSED                 [  5%]
tests/test_feature_engineering.py::test_feature_engineering_transforms_correctly PASSED       [ 10%]
tests/test_leakage_safe_split.py::test_chronological_split_with_embargo PASSED       [ 15%]
tests/test_leakage_safe_split.py::test_entity_grouped_split PASSED                     [ 21%]
tests/test_leakage_safe_split.py::test_leakage_audit_catches_synthetic_contamination PASSED   [ 26%]
tests/test_scaling_and_imputation.py::test_scaler_fits_on_training_data_only PASSED  [ 31%]
tests/test_scaling_and_imputation.py::test_training_median_imputation PASSED         [ 36%]
tests/test_schema_validation.py::test_valid_schema_passes PASSED                       [ 42%]
tests/test_schema_validation.py::test_missing_column_raises_error PASSED               [ 47%]
tests/test_schema_validation.py::test_invalid_dtype_raises_error PASSED                [ 52%]
tests/test_schema_validation.py::test_duplicate_detection_and_resolution PASSED       [ 57%]
tests/test_schema_validation.py::test_timestamp_gaps_and_out_of_order PASSED          [ 63%]
tests/test_schema_validation.py::test_missing_value_analysis PASSED                    [ 68%]
tests/test_schema_validation.py::test_outlier_analysis PASSED                          [ 73%]
tests/test_target_generation.py::test_exact_failure_horizon_boundaries PASSED        [ 78%]
tests/test_target_generation.py::test_normal_completion_and_cancellation PASSED       [ 84%]
tests/test_target_generation.py::test_version_2019_event_classifier PASSED           [ 89%]
tests/test_window_generation.py::test_window_generator_strict_history PASSED          [ 94%]
tests/test_window_generation.py::test_window_generator_padded_history PASSED          [100%]

============================== 19 passed in 3.03s ==============================
```

---

## 8. Final Verdict & Authorization for Phase 3

### Audit Verdict Summary
1. **Window-History Semantics (Audit A):** **PASS**. 100% of windows contain 15 observations with $\Delta t_i = 300.0\text{s}$. 15 observation intervals span 75.0 minutes; endpoint difference is 70.0 minutes.
2. **Causal Feature & Label Integrity (Audit B):** **PASS**. Zero future feature lookahead. Version-specific failure codes correctly enforced. Evaluation units formally distinguished.
3. **Split Leakage & Embargo (Audit C):** **PASS (with qualified chronological limitation)**. Entity-Grouped split has 0% entity, history, and label overlap. Chronological split has strict target-label isolation (30-min embargo), with input-history overlap limitation documented.
4. **Arrays and Preprocessing (Audit D):** **PASS**. 100% alignment, zero NaNs, zero label contamination, independent scaler fitting.
5. **Report Consistency (Audit E):** **PASS**. 100% match across all published figures.

### Blockers to Model Training
**Zero Blockers.** Every causal, labeling, windowing, alignment, and isolation prerequisite has been satisfied.

**The data pipeline and evaluation protocols are scientifically verified. Authorization is granted to proceed to Phase 3 (Baseline Models & LSTM Architecture) upon user confirmation.**
