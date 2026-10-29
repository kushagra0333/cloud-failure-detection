# Phase 1 Completion Report: Dataset Audit and Project Foundation

**Project:** Temporal LSTM/Transformer Framework for Early Failure Prediction in Cloud Computing  
**Phase:** 1 — Dataset Audit and Project Foundation  
**Auditor / Engineer:** Senior Machine Learning Engineer & Independent Experimental Auditor  
**Date:** October 2026  
**Status:** **GATE PASSED — PHASE 2 UNBLOCKED**

---

## 1. Files Created or Modified

| File Path | Description |
| :--- | :--- |
| `reports/dataset_audit.md` | Comprehensive forensic audit of candidate cloud trace datasets (Google Cluster Trace 2019/2011, Alibaba Trace, Azure VM Traces) analyzing schemas, event semantics, and failure definitions. |
| `configs/default_config.yaml` | Master YAML configuration file controlling all paths, sequence lengths ($L=15$), sampling intervals ($\Delta t=5$ min), prediction horizons ($H_{\mathrm{critical}}=15$ min, $H_{\mathrm{warning}}=30$ min), feature options, model architectures, and training hyperparameters. |
| `src/__init__.py` | Package root initialization. |
| `src/ingestion/__init__.py` | Data ingestion subpackage initialization. |
| `src/ingestion/schema.py` | Official event enumeration (`ClusterEventType`: `SUBMIT`, `SCHEDULE`, `EVICT`, `FAIL`, `FINISH`, `KILL`, `LOST`) and target label definitions (`FailureLabel`: `NORMAL=0`, `WARNING=1`, `CRITICAL=2`, `UNKNOWN=-1`). |
| `src/ingestion/validator.py` | Data validation algorithms for schema integrity, duplicate resolution, timestamp sorting, gap reporting, missingness rates, and Tukey's IQR/physical domain outlier profiling. |
| `tests/test_schema_validation.py` | Automated pytest suite validating schema rules, type constraints, duplicate handling, irregular gaps, missingness analysis, and outlier profiling. |
| `scripts/verify_phase1.py` | End-to-end verification script for checking environment dependencies, directory layout, config loading, schema mappings, and audit documentation. |

---

## 2. Algorithms and Mathematical Formulas Implemented

1. **Lifecycle Event Disambiguation:**
   - Abnormal Failure: $\text{event\_type} == 3\ (\text{FAIL})$
   - Normal Completion: $\text{event\_type} == 4\ (\text{FINISH})$
   - Cancellation: $\text{event\_type} == 5\ (\text{KILL})$
   - Eviction/Preemption: $\text{event\_type} == 2\ (\text{EVICT})$

2. **Temporal Grid & History Formulation:**
   - Sampling step: $\Delta t = 300\text{ seconds} = 5\text{ minutes}$
   - History window: $L = 15$ steps $\implies T_{\mathrm{history}} = L \Delta t = 75\text{ minutes}$
   - Input matrix representation: $X_{i,t} \in \mathbb{R}^{L \times F}$

3. **Target Horizon Formulation:**
   $$\tau_{i,t} = T_i^{\mathrm{FAIL}} - t$$
   $$y_{i,t} = \begin{cases}
   \text{CRITICAL } (2), & 0 < \tau_{i,t} \le 15\text{ min} \\
   \text{WARNING } (1), & 15 < \tau_{i,t} \le 30\text{ min} \\
   \text{NORMAL } (0), & \text{no failure in } (t, t + 30\text{ min}] \\
   \text{UNKNOWN } (-1), & \text{future observation } < 30\text{ min and no terminal event}
   \end{cases}$$

4. **Timestamp Gap and Interval Validation:**
   - Chronological sort: $t_1 \le t_2 \le \dots \le t_n$
   - Interval step: $\delta_t = t_t - t_{t-1}$
   - Gap flag condition: $\delta_t > 1.5 \cdot \Delta t$

5. **Missing-Value Ratio:**
   $$r_j = \frac{\sum_{k=1}^N \mathbf{1}(x_{k,j} \text{ is missing})}{N}$$

6. **Tukey's Exploratory Outlier Bounds:**
   $$\mathrm{IQR} = Q_3 - Q_1$$
   $$\text{Lower} = Q_1 - 1.5 \cdot \mathrm{IQR}, \qquad \text{Upper} = Q_3 + 1.5 \cdot \mathrm{IQR}$$

---

## 3. Assumptions Made

1. **Google Borg Traces** (`clusterdata-2019` / `clusterdata-2011`) are adopted as the authoritative ground-truth reference schema because they are the only public cloud datasets that unambiguously distinguish verified execution failures (`FAIL`) from normal completions (`FINISH`), preemptions (`EVICT`), and cancellations (`KILL`).
2. **Microsoft Azure VM Traces** are explicitly disqualified from failure prediction due to the absence of failure cause codes; deallocation (`vm_deleted`) cannot be treated as abnormal failure without committing target definition fraud.
3. **Censoring Horizon Rule:** If an entity's telemetry terminates before the 30-minute forward horizon without an explicit terminal lifecycle event (`FINISH`, `FAIL`, or `KILL`), the target label is censored as `UNKNOWN (-1)` and excluded from supervised loss calculation to prevent false negative contamination.

---

## 4. Tests Executed and Actual Outcomes

Executed via pytest test runner (`.\.venv\Scripts\pytest.exe -v`):
- `tests/test_schema_validation.py::test_valid_schema_passes`: **PASSED**
- `tests/test_schema_validation.py::test_missing_column_raises_error`: **PASSED**
- `tests/test_schema_validation.py::test_invalid_dtype_raises_error`: **PASSED**
- `tests/test_schema_validation.py::test_duplicate_detection_and_resolution`: **PASSED**
- `tests/test_schema_validation.py::test_timestamp_gaps_and_out_of_order`: **PASSED**
- `tests/test_schema_validation.py::test_missing_value_analysis`: **PASSED**
- `tests/test_schema_validation.py::test_outlier_analysis`: **PASSED**

**Overall Test Suite Result:** 7 passed in 0.57s (100% pass rate).

---

## 5. Dataset Limitations

1. **Volume of Google Cluster Trace 2019:** Full May 2019 traces exceed 2.4 TiB across 8 Borg cells. Running end-to-end training on local hardware or free Google Colab/Kaggle instances requires extracting verified representative cell slices (e.g. 24–48 hour continuous windows of Cell 'a') or using the verified 2011 partition subsets.
2. **Missing Sensor Readings:** CPI and MAI hardware performance counters are intermittently null for short-lived tasks that terminated before hardware counter sampling rounds completed.

---

## 6. Remaining Issues

None for Phase 1. All environment dependencies (Python 3.11, PyTorch CPU 2.14.1, NumPy 2.4, Pandas 3.1, Scikit-Learn 1.9, PyYAML 6.0, Pytest 9.1), schemas, config files, and verification scripts are operational.

---

## 7. Exact Command to Run Next

To transition into Phase 2 (Data cleaning, mathematical feature engineering, label generation, windowing, and leakage-safe splitting):

```powershell
.\.venv\Scripts\python.exe -m pytest tests/
```

Followed by running Phase 2 feature engineering and windowing pipelines.

---

## 8. Clear Statement of Gate Status

**PHASE 1 IS COMPLETE AND FULLY VERIFIED.**  
**PHASE 2 IS UNBLOCKED.**
