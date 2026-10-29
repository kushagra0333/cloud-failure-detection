# Phase 7A.2 — Authorized Labeling Repair and Diagnostic Regeneration Report

**Project:** Temporal LSTM/Transformer Framework for Early Failure Prediction in Cloud Computing  
**Phase:** Phase 7A.2 — Authorized Source-Code Repair & Diagnostic Data Regeneration  
**Author:** Independent Machine Learning Scientific Auditor & Pair Programmer  
**Date:** October 27, 2026  
**Governing Policy:** **Policy A: Terminal Entity Lifecycle** (Protocol Amendment 2.1 line 201)  
**Historical Evaluation Status:** Spent, Unblinded, and Permanently Preserved (`NO_OPERATIONAL_WINNER` locked)  
**Audit Verdict:** **`REPAIR PASSED`**

---

## 1. Executive Summary & Policy Adoption

### 1.1 Policy Adoption
In accordance with your explicit authorization, we adopted **Policy A: Terminal Entity Lifecycle** for this labeling repair:
1. **Strict Post-First-Failure Censoring:** If a task has already experienced its first abnormal failure event (`FAIL = 3` in 2011 trace, `5` in 2019 trace) at or before prediction timestamp $t$ ($T_{\text{fail}}^{(1)} \le t$), the prediction point is strictly labeled `UNKNOWN` (-1).
2. **Strictly Pre-Failure Observations Only:** Only observations occurring strictly before the first failure ($t < T_{\text{fail}}^{(1)}$) are eligible for prediction.
3. **Exact Collision Censoring:** An observation occurring exactly at the failure timestamp ($t == T_{\text{fail}}$) is concurrent (lead time $\tau = 0.0\text{s}$) and is labeled `UNKNOWN` (-1).
4. **Permanent Termination:** Repeated failures do not re-qualify post-first-failure telemetry. Any observation following $T_{\text{fail}}^{(1)}$ remains `UNKNOWN` (-1).
5. **Preservation of Pre-Failure Semantics:** All valid pre-failure observations retain their exact Critical ($\tau \le 900\text{s}$), Warning ($900\text{s} < \tau \le 1800\text{s}$), Normal ($\tau > 1800\text{s}$ or `FINISH`), `KILL`, and `EVICT` classifications.
6. **Robust Input Guards:** Missing (`None`), `NaN`, or non-finite (`Inf`) timestamps immediately return `FailureLabel.UNKNOWN, None`.

### 1.2 Protocol Evidence Supporting This Policy
- [Protocol Amendment 2.1 Line 201](file:///c:/Users/hp/Desktop/Research%20project/results/phase6/protocol_l5_amendment_2_1.json#L201): *"censored lookahead (< 1800s without FINISH) or **post-failure step -> label -1 (strictly excluded)**"*
- [`reports/phase6_1_cohort_verification.md` Line 93](file:///c:/Users/hp/Desktop/Research%20project/reports/phase6_1_cohort_verification.md#L93): *"Any telemetry timestamp $t \ge T_{\text{fail}}$ is strictly dropped prior to window generation. Post-failure steps are classified as censored (`label = -1`) and discarded."*
- [`scripts/verify_cohort_phase6_1.py` Line 311](file:///c:/Users/hp/Desktop/Research%20project/scripts/verify_cohort_phase6_1.py#L311): *"In Borg Trace 2011, retry attempts obtain distinct task_index values or execute sequentially. Each entity_id represents an isolated container execution. Post-failure steps are censored and excluded to prevent retrospective alerts from counting as early detection."*

---

## 2. Source-Code Patch Applied

We applied the minimal necessary patch to [`src/labeling/target_generator.py`](file:///c:/Users/hp/Desktop/Research%20project/src/labeling/target_generator.py).

### 2.1 File Hashes
- **Original File:** [`src/labeling/target_generator.py`](file:///c:/Users/hp/Desktop/Research%20project/src/labeling/target_generator.py)
  - Pre-patch SHA-256: `d04f074581eec9ee64b02178133be23fceb9943263372d134ebf1c952f39c648`
  - Backup Path: [`results/phase7a/target_generator_original.py.bak`](file:///c:/Users/hp/Desktop/Research%20project/results/phase7a/target_generator_original.py.bak)
- **Patched File:** [`src/labeling/target_generator.py`](file:///c:/Users/hp/Desktop/Research%20project/src/labeling/target_generator.py)
  - Post-patch SHA-256: `22e8c14a9f9730857c901707558adc13449486442808364ab928bfabafa2fcfe`

### 2.2 Complete Code Diff

```diff
--- a/src/labeling/target_generator.py
+++ b/src/labeling/target_generator.py
@@ -53,10 +53,27 @@ class TargetLabelGenerator:
         t = prediction_time_sec
 
+        # Handle missing, null, or non-finite prediction timestamps safely
+        if pd.isna(t) or not np.isfinite(t):
+            return FailureLabel.UNKNOWN, None
+
+        # 0. Strict Prior/Concurrent Failure Check (Protocol Amendment 2.1 line 201)
+        # Any observation at or after the entity's first failure event (time_sec <= t)
+        # is post-failure telemetry and must be labeled UNKNOWN (-1) and strictly excluded.
+        if not task_events.empty and "event_type" in task_events.columns and "time_sec" in task_events.columns:
+            prior_fails = task_events[
+                (task_events["event_type"] == self.event_cls.FAIL) &
+                (task_events["time_sec"] <= t)
+            ]
+            if not prior_fails.empty:
+                return FailureLabel.UNKNOWN, None
+
         # 1. Locate any abnormal failure event occurring strictly after prediction time t
-        future_fails = task_events[
-            (task_events["event_type"] == self.event_cls.FAIL) &
-            (task_events["time_sec"] > t)
-        ]
+        future_fails = pd.DataFrame()
+        if not task_events.empty and "event_type" in task_events.columns and "time_sec" in task_events.columns:
+            future_fails = task_events[
+                (task_events["event_type"] == self.event_cls.FAIL) &
+                (task_events["time_sec"] > t)
+            ]
 
         if not future_fails.empty:
             earliest_fail_sec = future_fails["time_sec"].min()
```

### 2.3 Plain-Language Explanation of Changes
1. **Timestamp Guard:** Before doing any calculations, the function checks if the timestamp is missing (`NaN` or `None`). If so, it immediately returns `UNKNOWN` (-1) instead of crashing with a Python type error.
2. **Prior Failure Check:** The function checks if the task already suffered an abnormal `FAIL` event at or before time $t$. If it has, the observation is marked `UNKNOWN` (-1). Because `WindowGenerator(exclude_unknown=True)` ignores `UNKNOWN` rows, this permanently prevents post-crash telemetry from becoming training windows or evaluation questions.
3. **Safe DataFrame Access:** We added guards (`if not task_events.empty and ...`) so that missing or empty event tables do not cause `KeyError` crashes.

---

## 3. Regression and Integration Test Results

All tests were executed directly against the production codebase:

```bash
.\.venv\Scripts\pytest.exe tests/test_phase7a_labeling_repair.py
```
**Result: 18 passed in 2.79s (100% pass rate).**

```bash
.\.venv\Scripts\pytest.exe -q
```
**Full Repository Test Suite Result: 177 passed in 15.14s (100% pass rate, 0 failures, 0 errors).**

### 3.1 Test Coverage Summary

| Scenario ID | Test Name | Behavior Verified | Status |
| :---: | :--- | :--- | :---: |
| **01** | `test_scenario_01_valid_normal_sufficient_followup` | Normal observation with $\ge 30\text{m}$ follow-up retains label 0 | **PASS** |
| **02** | `test_scenario_02_valid_normal_finish_event` | Normal observation with `FINISH` event retains label 0 | **PASS** |
| **03** | `test_scenario_03_failure_within_critical_horizon` | Lead time $\tau \le 900\text{s}$ retains label 2 (Critical) | **PASS** |
| **04** | `test_scenario_04_failure_exactly_at_critical_boundary` | Exact boundary $\tau = 900.0\text{s}$ retains label 2 (Critical) | **PASS** |
| **05** | `test_scenario_05_failure_in_warning_interval` | Lead time $900\text{s} < \tau \le 1800\text{s}$ retains label 1 (Warning) | **PASS** |
| **06** | `test_scenario_06_failure_exactly_at_warning_boundary` | Exact boundary $\tau = 1800.0\text{s}$ retains label 1 (Warning) | **PASS** |
| **07** | `test_scenario_07_failure_just_beyond_warning_horizon` | Lead time $\tau = 1800.1\text{s} > 1800\text{s}$ retains label 0 (Normal) | **PASS** |
| **08** | `test_scenario_08_insufficient_future_telemetry` | $< 30\text{m}$ follow-up without terminal event is `UNKNOWN` (-1) | **PASS** |
| **09** | `test_scenario_09_telemetry_after_failure_strictly_censored` | Telemetry after failure is strictly `UNKNOWN` (-1) | **PASS** |
| **10** | `test_scenario_10_repeated_failures_step_after_prior_failure` | Repeated failure task step after failure 1 is `UNKNOWN` (-1) | **PASS** |
| **11** | `test_scenario_11_window_between_two_failures_strictly_censored`| Step between two failures is strictly `UNKNOWN` (-1) | **PASS** |
| **12** | `test_scenario_12_failure_exactly_at_prediction_timestamp` | Concurrent collision $t == T_{\text{fail}}$ is `UNKNOWN` (-1) | **PASS** |
| **13** | `test_scenario_13_user_cancellation_within_30m` | User `KILL` cancellation within 30m is `UNKNOWN` (-1) | **PASS** |
| **14** | `test_scenario_14_missing_or_nan_timestamp` | `NaN` prediction timestamp safely returns `UNKNOWN` (-1) | **PASS** |
| **15** | `test_scenario_15_clean_prefailure_retains_label` | Clean pre-failure case retains original Critical label | **PASS** |
| **16** | `test_scenario_16_window_generator_excludes_unknown_endpoints` | End-to-end: `WindowGenerator` excludes all post-failure endpoints | **PASS** |
| **17** | `test_scenario_17_no_sequence_contains_post_failure_telemetry` | End-to-end: 100% of history timestamps in tensors are $< T_{\text{fail}}$ | **PASS** |
| **18** | `test_scenario_18_feature_and_label_arrays_remain_aligned` | End-to-end: Tensor $X$ and label vector $y$ remain perfectly aligned | **PASS** |

---

## 4. Dataset Reconciliation & Row Counts (Before vs. After Repair)

Diagnostic datasets were generated into [`data/processed/phase7a_diagnostic/`](file:///c:/Users/hp/Desktop/Research%20project/data/processed/phase7a_diagnostic/) and summarized in [`results/phase7a/diagnostic_data_summary.csv`](file:///c:/Users/hp/Desktop/Research%20project/results/phase7a/diagnostic_data_summary.csv) and [`results/phase7a/diagnostic_split_reconciliation.json`](file:///c:/Users/hp/Desktop/Research%20project/results/phase7a/diagnostic_split_reconciliation.json).

### 4.1 Grand Window Count Reconciliation

| Split | Original Total Windows | Excluded Post-Failure Windows | Retained Clean Windows | Retained Normal ($y=0$) | Retained Warning ($y=1$) | Retained Critical ($y=2$) | Removed Normal | Removed Warning | Removed Critical | Retained Entities |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Train** | 4,798 | **1,403** | **3,395** | 2,959 | 215 | 221 | 1,229 | 32 | 142 | 344 |
| **Val** | 1,015 | **237** | **778** | 643 | 72 | 63 | 197 | 0 | 40 | 73 |
| **Test** | 1,003 | **216** | **787** | 676 | 54 | 57 | 188 | 14 | 14 | 74 |
| **Confirmatory** | 5,511 | **196** | **5,315** | 4,655 | 286 | 374 | 156 | 14 | 26 | 335 |
| **TOTALS** | **12,327** | **2,052** | **10,275** | **8,933** | **627** | **715** | **1,770** | **60** | **222** | **826** |

### 4.2 Why Affected Labels Differ from Removed Windows
1. **Observation-Level vs. Window-Level Units:** `TargetLabelGenerator` operates on individual 300-second telemetry steps (rows in raw tables), whereas `WindowGenerator` operates on 5-step rolling sequences.
2. **History Contiguity Gate:** A telemetry step must have 4 preceding contiguous 300-second observations to form a window. If an entity suffered a crash and had only 1 or 2 lingering telemetry steps before shutting down, those steps were relabeled `UNKNOWN` (-1) in the table, but they never formed a valid window in the original dataset anyway.
3. **Endpoint Purging:** In the confirmatory holdout, exactly **156 Normal windows** were removed ($149$ post-all-fail normal + $7$ cross-history normal). Under Policy A, the 40 Warning/Critical windows belonging to subsequent retry episodes of multi-failure tasks were also purged, resulting in exactly **196 total excluded windows** and leaving **5,315 strictly clean pre-failure windows**.

### 4.3 Post-Failure and Cross-Failure Windows Remaining
- Post-failure Normal windows remaining across all clean diagnostic datasets: **0 (0.00%)**.
- Windows whose 5-step histories cross a failure event remaining: **0 (0.00%)**.
- Sequence tensor history verification: **100% of observations in all retained windows have timestamp $t < T_{\text{fail}}^{(1)}$**.

---

## 5. Artifact Directory Inventory & Hashes

### 5.1 Newly Generated Diagnostic Artifacts

All files were written to fresh locations without touching historical files:

| Artifact File | Size | SHA-256 Hash | Role |
| :--- | :---: | :--- | :--- |
| `data/processed/phase7a_diagnostic/train_metadata_repaired.csv` | 108.7 KB | `4f826359eb3b1f1f99c273898f8b80b7d5eb45e776ea2d8f993f10111f1816bc` | Clean Training Metadata |
| `data/processed/phase7a_diagnostic/X_train_repaired.npy` | 2.1 MB | `f044aa81b1d8350dbefee3364f0b2f34cb5e09ddaa8c29215886ec163f587e59` | Clean Training Input Tensors $[3395, 5, 31]$ |
| `data/processed/phase7a_diagnostic/y_train_repaired.npy` | 27.2 KB | `04bf07d2f447f551b9d10787e9eb7d1d2834b1713d789e924ba95f76bfe4e954` | Clean Training Labels $[3395]$ |
| `data/processed/phase7a_diagnostic/val_metadata_repaired.csv` | 26.2 KB | `6da9a7fe91b2c5890885145b2bb5a5c68f1c84138b0e77d3aaae46ce4672e128` | Clean Validation Metadata |
| `data/processed/phase7a_diagnostic/X_val_repaired.npy` | 482.4 KB | `0b6a987d609a6331a31d9237df6d288d65a88cbf39b8bc7e25fb78536ec6fba0` | Clean Validation Input Tensors $[778, 5, 31]$ |
| `data/processed/phase7a_diagnostic/y_val_repaired.npy` | 6.3 KB | `4147774e6f477e6822fc943ebdf9ae8e3483861274a4cf7f5c76db1797c234b9` | Clean Validation Labels $[778]$ |
| `data/processed/phase7a_diagnostic/test_metadata_repaired.csv` | 26.5 KB | `a9069d2f25603c4f74d08ee7183eec18596f264ae41c2c2f8150d03b0d46f36d` | Clean Test Metadata |
| `data/processed/phase7a_diagnostic/X_test_repaired.npy` | 488.0 KB | `adbb2c24ca0c8f5f4b5952d7083049b1ff53bc1be1ff5c00e12dcfdfc633a69a` | Clean Test Input Tensors $[787, 5, 31]$ |
| `data/processed/phase7a_diagnostic/y_test_repaired.npy` | 6.4 KB | `0cf9b33a01099688469ad01ffad3d43d837691d604ec2829fc438e8cb28dfd05` | Clean Test Labels $[787]$ |
| `data/processed/phase7a_diagnostic/confirmatory_metadata_repaired.csv` | 151.1 KB | `04a11f2a336ec88f117df8c06cb44331dfaa4ae879008bc591c015b6fe811fa1` | Clean Confirmatory Metadata |
| `data/processed/phase7a_diagnostic/X_confirmatory_repaired.npy` | 3.3 MB | `f044bb71cfb2ec4c54f5ea0bc8cb6cb6e72c050d24db13f1791a84f509c6ebfa` | Clean Confirmatory Input Tensors $[5315, 5, 31]$ |
| `data/processed/phase7a_diagnostic/y_confirmatory_repaired.npy` | 42.6 KB | `6f2c0352ef200c59828d546059c258d415ebc2c8f8ab3412574047a4627d337c` | Clean Confirmatory Labels $[5315]$ |
| `results/phase7a/diagnostic_split_reconciliation.json` | 3.8 KB | `8ce5a0bd0ab7c4adcb8fa66bbdbf2e463a8a37f59458df4c7ea762bf63d08599` | Full JSON Split Summary |
| `results/phase7a/diagnostic_data_summary.csv` | 682 B | `fc75f7dfa74d283cb7e2524beec848a609d5be31b017f9b8c0fb2ef0a20a4060` | Tabular Split Comparison |

### 5.2 Confirmation of Historical Artifact Immutability
All historical Phase 6 artifacts were verified as untouched. Their SHA-256 hashes match historical preflight logs:
- `results/phase6/confirmatory_predictions_l5.csv`: `d1d798314218ba56...` (**UNTOUCHED**)
- `results/phase6/confirmatory_metrics_l5.json`: `121828c4bfcd3871...` (**UNTOUCHED**)
- `results/phase6/protocol_l5_amendment_2_1.json`: `f354cd9f02343e83...` (**UNTOUCHED**)
- `data/processed/confirmatory_l5/X_confirmatory_l5.npy`: `f5401db048685057...` (**UNTOUCHED**)
- `data/processed/l5/X_train.npy`: `ffe168bb058f49c6...` (**UNTOUCHED**)

---

## 6. Prohibited Actions Enforced

1. **Zero Retraining:** No machine learning models were retrained in this phase.
2. **Zero Metric Optimization:** No decision thresholds were tuned to artificially improve F1 or false-alarm rates.
3. **No Rescoring of Confirmatory Holdout:** The Phase 6 confirmatory holdout remains spent and unblinded. The clean confirmatory tensors generated in `data/processed/phase7a_diagnostic/` are strictly for diagnostic sensitivity analyses.
4. **Permanent Operational Verdict:** The official operational verdict of Phase 6 remains immutable:
   $$\mathbf{NO\_OPERATIONAL\_WINNER}$$

---

## 7. Remaining Risks and Unresolved Questions

1. **Parent-Job Leakage in Development Splits:** While post-failure telemetry is now completely eradicated, 75%–80% of parent jobs still overlap between `train` and `val` in the historical splitting setup. When retraining is authorized in future phases, `job_grouped` splitting must be applied to prevent batch-workload memorization.
2. **Uncalibrated Argmax Thresholds:** The high false-alarm rates on pure-normal tasks ($17.4\% - 33.9\%$) were driven by uncalibrated 3-class argmax decision rules, not by post-failure data. Solving the operational gate failure will require cost-sensitive threshold selection on the validation split.

---

## 8. Final Audit Verdict

$$\mathbf{REPAIR\ PASSED}$$

The defect in `src/labeling/target_generator.py` is safely, cleanly, and minimally repaired. All 18 regression and integration tests pass, all 177 repository tests pass, all diagnostic datasets are regenerated into isolated directories, and zero post-failure or cross-failure windows remain. All historical records remain intact. We stand ready for your direction before any model retraining or experiment execution.
