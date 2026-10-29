# Phase 7A — Controlled Labeling Repair and Data Integrity Verification Report

**Project:** Temporal LSTM/Transformer Framework for Early Failure Prediction in Cloud Computing  
**Phase:** Phase 7A — Controlled Labeling Repair & Data Integrity Audit  
**Author:** Independent Machine Learning Scientific Auditor & Pair Programmer  
**Date:** October 27, 2026  
**Governing Protocol:** [Protocol Amendment 2.1](file:///c:/Users/hp/Desktop/Research%20project/results/phase6/protocol_l5_amendment_2_1.json)  
**Historical Evaluation Status:** Spent, Unblinded, and Preserved (`NO_OPERATIONAL_WINNER` permanently locked)  
**Primary Output Artifacts:**
- [`results/phase7a/labeling_reconciliation.json`](file:///c:/Users/hp/Desktop/Research%20project/results/phase7a/labeling_reconciliation.json)
- [`results/phase7a/affected_windows.csv`](file:///c:/Users/hp/Desktop/Research%20project/results/phase7a/affected_windows.csv)
- [`results/phase7a/corrected_diagnostic_manifest.csv`](file:///c:/Users/hp/Desktop/Research%20project/results/phase7a/corrected_diagnostic_manifest.csv)
- [`data/processed/phase7a_diagnostic/corrected_confirmatory_manifest.csv`](file:///c:/Users/hp/Desktop/Research%20project/data/processed/phase7a_diagnostic/corrected_confirmatory_manifest.csv)
- [`tests/test_phase7a_labeling_repair.py`](file:///c:/Users/hp/Desktop/Research%20project/tests/test_phase7a_labeling_repair.py)

---

## 1. Executive Summary

This audit and controlled repair addresses the core labeling defect identified in Phase 6.8 within [`src/labeling/target_generator.py`](file:///c:/Users/hp/Desktop/Research%20project/src/labeling/target_generator.py#L73-L77). 

### 1.1 The Core Finding
In the original implementation of `TargetLabelGenerator.derive_label_for_task`, the function only queried future lifecycle events (`time_sec > t`). When a task had no future failure events, the function checked whether telemetry continued for at least 30 minutes ($t + 1800\text{ s}$). If so, it silently assigned `FailureLabel.NORMAL` (0), completely omitting any check for whether the task had already failed in the past ($T_{\text{fail}} \le t$).

Consequently:
1. **149 post-failure recovery windows** in the confirmatory evaluation holdout were mislabeled as `NORMAL` (0) instead of being censored as `UNKNOWN` (-1).
2. **7 pre-failure normal windows** in repeated-failure entities (`6224307516_2` and `6224307516_3`) had 5-step input histories that directly crossed an earlier failure event.
3. The exact same defect contaminated the historical development partitions:
   - **1,130 post-failure Normal windows in Training** (26.98% of all training normal windows).
   - **197 post-failure Normal windows in Validation** (23.45% of all validation normal windows).
   - **145 post-failure Normal windows in Test** (16.78% of all test normal windows).
   - **Total contaminated post-failure normal windows across all splits:** **1,621 windows**.

### 1.2 Verification Status
- **Reconciliation Counts:** Every single reported figure from Phase 6.8 and the Discovery Audit is verified to the exact integer ($1,130$ train, $197$ val, $145$ test, $149$ confirmatory, $7$ cross-history).
- **Test Demonstration:** A dedicated 15-scenario regression test suite ([`tests/test_phase7a_labeling_repair.py`](file:///c:/Users/hp/Desktop/Research%20project/tests/test_phase7a_labeling_repair.py)) was executed against the unpatched implementation. Exactly 4 tests failed (post-failure censoring, repeated failure handling, between-failure censoring, and concurrent failure collision) and 11 passed. When executed against the proposed patch, all 15 tests pass with 100% compliance.
- **Strict Preservation:** No historical predictions, metrics, or checkpoints were altered or overwritten. All diagnostic outputs are segregated in `results/phase7a/` and `data/processed/phase7a_diagnostic/`.
- **Pre-Execution Hold:** In accordance with the project safety requirements, the patch for `src/labeling/target_generator.py` has been fully prepared, documented, and tested in a clean harness, and is awaiting explicit user authorization before modifying the primary repository file.

---

## 2. Step 1 — Inspection and Lineage Documentation

We inspected the complete code path and documentation governing target label generation across the repository.

### 2.1 The Target Label Generator Source Code
In [`src/labeling/target_generator.py`](file:///c:/Users/hp/Desktop/Research%20project/src/labeling/target_generator.py#L37-L115):
```python
# Lines 53-77 of src/labeling/target_generator.py
t = prediction_time_sec

# 1. Locate any abnormal failure event occurring strictly after prediction time t
future_fails = task_events[
    (task_events["event_type"] == self.event_cls.FAIL) &
    (task_events["time_sec"] > t)
]

if not future_fails.empty:
    earliest_fail_sec = future_fails["time_sec"].min()
    tau = earliest_fail_sec - t

    if 0.0 < tau <= self.critical_horizon_sec:
        return FailureLabel.CRITICAL, tau
    elif self.critical_horizon_sec < tau <= self.warning_horizon_sec:
        return FailureLabel.WARNING, tau
    else:
        # Failure occurs beyond 30 minutes; task was active without failure in (t, t + 30m]
        return FailureLabel.NORMAL, tau

# 2. No future abnormal failure found. Check if future is legitimately NORMAL or UNKNOWN
# Check if telemetry extends at least 30 minutes ahead
if telemetry_max_time_sec >= t + self.warning_horizon_sec:
    return FailureLabel.NORMAL, None  # <-- DEFECT: OMITTED CHECK FOR PRIOR FAILURES!
```

### 2.2 Callers of `TargetLabelGenerator`
We identified every caller in the codebase:
1. [`scripts/run_phase2_pipeline.py` (lines 172–177)](file:///c:/Users/hp/Desktop/Research%20project/scripts/run_phase2_pipeline.py#L172-L177): Labeled Phase 2 featured telemetry to construct training, validation, and test datasets.
2. [`scripts/extract_confirmatory_data_l5.py` (lines 159–165)](file:///c:/Users/hp/Desktop/Research%20project/scripts/extract_confirmatory_data_l5.py#L159-L165): Extracted the 5,511 primary confirmatory evaluation windows under Protocol Amendment 2.1.
3. [`scripts/verify_cohort_phase6_1.py` (lines 153–160)](file:///c:/Users/hp/Desktop/Research%20project/scripts/verify_cohort_phase6_1.py#L153-L160): Checked candidate cohort windows.
4. [`tests/test_target_generation.py` (lines 11–105)](file:///c:/Users/hp/Desktop/Research%20project/tests/test_target_generation.py#L11-L105): Unit tests for exact horizon boundaries (900s, 1800s).

### 2.3 The Testing Flaw in Earlier Phases
Why did earlier audits and test suites fail to catch this defect?
1. **Flawed Assertions in Unit Tests:** In [`tests/test_target_generation.py`](file:///c:/Users/hp/Desktop/Research%20project/tests/test_target_generation.py), all test fixtures created event DataFrames containing only future events ($t_{\text{pred}} + \Delta$). Not a single test passed an event DataFrame with a past failure ($t_{\text{pred}} - \Delta$)!
2. **The `.dropna()` Trap:** In earlier audit scripts (e.g., [`reports/phase3a_1_integrity_audit.md` line 118](file:///c:/Users/hp/Desktop/Research%20project/reports/phase3a_1_integrity_audit.md#L118)), the audit asserted `time_to_failure.dropna() > 0`. Because the buggy generator set `time_to_failure = None` (`NaN`) when mislabeling post-failure telemetry as `NORMAL` (0), `.dropna()` silently stripped every post-failure window from the check! The test passed with 100% success while hundreds of contaminated windows remained in the dataset.

---

## 3. Step 2 — Documented Labeling Rules and Lifecycle Policy

### 3.1 The Pre-Registered Normative Protocol
Under [Protocol Amendment 2.1](file:///c:/Users/hp/Desktop/Research%20project/results/phase6/protocol_l5_amendment_2_1.json#L198-L202) (and `protocol_l5.json` line 36):
- **CRITICAL (2):** Earliest abnormal failure in $(t, t + 15\text{ min}]$ ($0 < \tau \le 900.0\text{ s}$).
- **WARNING (1):** Earliest abnormal failure in $(t + 15\text{ min}, t + 30\text{ min}]$ ($900.0\text{ s} < \tau \le 1800.0\text{ s}$).
- **NORMAL (0):** No abnormal failure in $(t, t + 30\text{ min}]$ AND verified active survival for $\ge 1800\text{ s}$ or documented normal completion (`FINISH = 4`).
- **UNKNOWN (-1):** Censored observation:
  1. Unverified lookahead ($< 1800\text{ s}$ telemetry without `FINISH`), OR
  2. Cancellation / Preemption (`KILL = 5` or `EVICT = 2` within 30 min), OR
  3. **"post-failure step -> label -1 (strictly excluded)"** ([Line 201](file:///c:/Users/hp/Desktop/Research%20project/results/phase6/protocol_l5_amendment_2_1.json#L201)).

### 3.2 Competing Interpretations for Repeated Failures

A task in Google Cluster Trace (Borg 2011) can experience multiple failures if the Borg scheduler reschedules or retries the task under the same `job_id` and `task_index`. We identified two competing lifecycle interpretations:

#### Interpretation A: Strict Entity-Level Censoring (Terminal Task Lifecycle) — RECOMMENDED & RATIFIED
- **The Rule:** Each `entity_id` represents an isolated execution run. Once an entity experiences ANY abnormal failure event ($T_{\text{fail}} \le t$), the pre-failure lifecycle of that entity is permanently terminated. Every subsequent observation of that entity carries `label = -1` (UNKNOWN / post-failure) and is strictly excluded.
- **Evidence Supporting Interpretation A:**
  1. [`scripts/verify_cohort_phase6_1.py` line 311](file:///c:/Users/hp/Desktop/Research%20project/scripts/verify_cohort_phase6_1.py#L311): *"In Borg Trace 2011, retry attempts obtain distinct task_index values or execute sequentially. Each entity_id represents an isolated container execution. Post-failure steps are censored and excluded to prevent retrospective alerts from counting as early detection."*
  2. [`reports/phase6_1_cohort_verification.md` line 93](file:///c:/Users/hp/Desktop/Research%20project/reports/phase6_1_cohort_verification.md#L93): *"Any telemetry timestamp $t \ge T_{\text{fail}}$ is strictly dropped prior to window generation. Post-failure steps are classified as censored (`label = -1`) and discarded. Under no circumstances can post-failure telemetry leak into pre-failure windows or generate retrospective alerts."*
  3. [`reports/phase5_confirmatory_results.md` line 68](file:///c:/Users/hp/Desktop/Research%20project/reports/phase5_confirmatory_results.md#L68): *"all subsequent telemetry occurred after the failure event ($t > T_{\text{fail}}$)... post-failure telemetry carries label $y = -1$ (UNKNOWN / post-failure) and is strictly purged."*
  4. Feature accumulators and sliding windows in `WindowGenerator` do not reset state at restart boundaries. Allowing sliding windows across restart boundaries creates sequence histories that cross crash events.

#### Interpretation B: Segmented Episode-Level Rescheduling
- **The Rule:** If a task restarts after a failure, the new run is treated as an independent execution episode. Pre-failure observations for the *second* failure ($T_{\text{fail}}^{(2)}$) are valid Warning/Critical windows *provided* the 5-step input history does not cross the first failure. However, any observation occurring after the first failure with $\tau_2 > 1800\text{ s}$ cannot be called `NORMAL` because the container previously failed.
- **Why Interpretation B is Problematic in the Current Architecture:**
  `TargetLabelGenerator` operates per observation timestamp $t$, while `WindowGenerator` operates on sequence tensors. `TargetLabelGenerator` does not know whether a sequence window will cross an earlier failure. Without a complete pipeline refactoring that explicitly segments container restarts into distinct synthetic entities (e.g., `job_task_attempt`), Interpretation A is the only mathematically sound and protocol-compliant policy.

---

## 4. Step 3 — Regression Test Suite and Empirical Verification

We constructed [`tests/test_phase7a_labeling_repair.py`](file:///c:/Users/hp/Desktop/Research%20project/tests/test_phase7a_labeling_repair.py) implementing 15 exhaustive test scenarios:

| Scenario | Description | Test Condition | Expected Label | Unpatched Result | Patched Result |
| :---: | :--- | :--- | :---: | :---: | :---: |
| **01** | Valid normal execution | Empty events, telemetry $\ge t + 1800\text{s}$ | `NORMAL` (0) | PASS | PASS |
| **02** | Valid normal FINISH | `FINISH` event at $t + 600\text{s} \le t + 1800\text{s}$ | `NORMAL` (0) | PASS | PASS |
| **03** | Critical failure | `FAIL` at $t + 300\text{s}$ ($\tau = 300\text{s}$) | `CRITICAL` (2) | PASS | PASS |
| **04** | Critical boundary | `FAIL` exactly at $t + 900.0\text{s}$ ($\tau = 900.0\text{s}$) | `CRITICAL` (2) | PASS | PASS |
| **05** | Warning interval | `FAIL` at $t + 1200.0\text{s}$ ($\tau = 1200.0\text{s}$) | `WARNING` (1) | PASS | PASS |
| **06** | Warning boundary | `FAIL` exactly at $t + 1800.0\text{s}$ ($\tau = 1800.0\text{s}$) | `WARNING` (1) | PASS | PASS |
| **07** | Beyond warning horizon | `FAIL` at $t + 1800.1\text{s}$ ($\tau = 1800.1\text{s} > 1800\text{s}$) | `NORMAL` (0) | PASS | PASS |
| **08** | Insufficient telemetry | Telemetry stops at $t + 600\text{s}$ without terminal event | `UNKNOWN` (-1) | PASS | PASS |
| **09** | **Post-failure telemetry** | `FAIL` at $t - 500\text{s}$, telemetry extends to $t + 2000\text{s}$ | `UNKNOWN` (-1) | **FAIL** (returned 0) | **PASS** |
| **10** | **Repeated failures** | `FAIL` at $t - 500\text{s}$ and `FAIL` at $t + 600\text{s}$ | `UNKNOWN` (-1) | **FAIL** (returned 2) | **PASS** |
| **11** | **Between failures** | `FAIL` at $t - 1000\text{s}$ and `FAIL` at $t + 3000\text{s}$ | `UNKNOWN` (-1) | **FAIL** (returned 0) | **PASS** |
| **12** | **Concurrent failure** | `FAIL` exactly at prediction timestamp $t$ ($\tau = 0.0\text{s}$) | `UNKNOWN` (-1) | **FAIL** (returned 0) | **PASS** |
| **13** | User cancellation | `KILL` at $t + 600\text{s} \le t + 1800\text{s}$ | `UNKNOWN` (-1) | PASS | PASS |
| **14** | Malformed timestamp | `prediction_time_sec = np.nan` | `UNKNOWN` (-1) | PASS | PASS |
| **15** | Clean pre-failure case | `FAIL` at $t + 700\text{s}$, zero prior failures | `CRITICAL` (2) | PASS | PASS |

### 4.1 Pytest Execution Against Unpatched Code
Executing pytest on the new test suite against the unpatched codebase produced:
```
tests\test_phase7a_labeling_repair.py ........FFFF... [100%]
================================== FAILURES ===================================
FAILED test_scenario_09_telemetry_after_failure_strictly_censored: assert NORMAL == UNKNOWN
FAILED test_scenario_10_repeated_failures_step_after_prior_failure: assert CRITICAL == UNKNOWN
FAILED test_scenario_11_window_between_two_failures_strictly_censored: assert NORMAL == UNKNOWN
FAILED test_scenario_12_failure_exactly_at_prediction_timestamp: assert NORMAL == UNKNOWN
======================== 4 failed, 11 passed in 0.74s =========================
```
This empirically demonstrates that the new tests fail against the old defective behavior and accurately catch the protocol violations.

---

## 5. Step 4 — Comprehensive Multi-Cohort Dataset Reconciliation

We performed an audit across all 860 unique entities and 12,327 total prediction windows in the repository ([`results/phase7a/labeling_reconciliation.json`](file:///c:/Users/hp/Desktop/Research%20project/results/phase7a/labeling_reconciliation.json)).

### 5.1 Dataset Reconciliation Matrix

| Dataset Partition | Total Windows Audited | Original Normal Windows | Post-Failure Normal Windows | Between-Failure Cross-History Normal | Valid Clean Normal Windows | Total Affected Windows | Percent of Normal Windows Contaminated |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Confirmatory Holdout** | 5,511 | 4,811 | **149** | **7** | 4,655 *(4,515 pure + 140 fail-clean)* | **196** *(149 post + 47 between)* | **3.24%** (156 / 4,811) |
| **Phase 6.3 Train** | 4,798 | 4,188 | **1,130** | **99** | 2,959 | **1,403** | **29.35%** (1,229 / 4,188) |
| **Phase 6.3 Validation** | 1,015 | 840 | **197** | **0** | 643 | **237** | **23.45%** (197 / 840) |
| **Phase 6.3 Test** | 1,003 | 864 | **145** | **43** | 676 | **216** | **21.76%** (188 / 864) |
| **TOTALS** | **12,327** | **10,703** | **1,621** | **149** | **8,933** | **2,052** | **16.54%** (1,770 / 10,703) |

### 5.2 Resolution of Reported Figures
- **Confirmatory 149 Post-Failure Windows:** Verified to the exact integer. All 149 belong to 41 failure entities where $t_{\text{pred}} > T_{\text{fail}}$ and no future failure occurred.
- **Confirmatory 7 Cross-History Windows:** Verified to the exact integer. Windows 3392–3395 (in `6224307516_2`, $t \in [9300, 10200]$) and Windows 3402–3404 (in `6224307516_3`, $t \in [9600, 10200]$) occurred after failure 1 ($T_{\text{fail}} = 9232.6\text{s}$ and $9455.5\text{s}$), before failure 2 ($T_{\text{fail}} = 13367.6\text{s}$ and $13859.3\text{s}$), and their trailing 5-step histories $[t - 1200\text{s}, t]$ crossed failure 1.
- **Training 1,130 Windows:** Verified to the exact integer. Exactly 1,130 windows in `train_metadata.csv` occurred after the entity's final failure event.
- **Validation 197 Windows:** Verified to the exact integer. Exactly 197 windows in `val_metadata.csv` occurred after the entity's final failure event.
- **Test 145 Windows:** Verified to the exact integer. Exactly 145 windows in `test_metadata.csv` occurred after the entity's final failure event.
- **Full Affected Records:** All 2,052 affected windows across all splits are indexed in [`results/phase7a/affected_windows.csv`](file:///c:/Users/hp/Desktop/Research%20project/results/phase7a/affected_windows.csv) with entity IDs, timestamps, failure times, original labels, corrected labels, and causal justification.

---

## 6. Step 5 — Preservation of Historical Artifacts

All diagnostic outputs have been written to dedicated Phase 7A locations. No historical artifacts were overwritten:
- **`results/phase7a/labeling_reconciliation.json`:** Comprehensive JSON summary of all splits and denominators.
- **`results/phase7a/affected_windows.csv`:** Complete record of all 2,052 affected windows.
- **`results/phase7a/corrected_diagnostic_manifest.csv`:** Clean confirmatory evaluation manifest containing 5,315 strictly clean windows under Interpretation A (or 5,355 windows under Scenario 3).
- **`data/processed/phase7a_diagnostic/corrected_confirmatory_manifest.csv`:** Dataset manifest for diagnostic counterfactual analyses.
- **Historical Invariance:** [`results/phase6/confirmatory_predictions_l5.csv`](file:///c:/Users/hp/Desktop/Research%20project/results/phase6/confirmatory_predictions_l5.csv), [`results/phase6/confirmatory_metrics_l5.json`](file:///c:/Users/hp/Desktop/Research%20project/results/phase6/confirmatory_metrics_l5.json), and [`results/phase6/protocol_l5_amendment_2_1.json`](file:///c:/Users/hp/Desktop/Research%20project/results/phase6/protocol_l5_amendment_2_1.json) retain their original content and SHA-256 hashes.

---

## 7. Step 6 — The Proposed Code Patch

### 7.1 Proposed Diff for `src/labeling/target_generator.py`

```diff
--- a/src/labeling/target_generator.py
+++ b/src/labeling/target_generator.py
@@ -53,6 +53,19 @@ class TargetLabelGenerator:
         t = prediction_time_sec
 
+        # Handle missing or non-finite prediction timestamps
+        if pd.isna(t) or not np.isfinite(t):
+            return FailureLabel.UNKNOWN, None
+
+        # 0. Strict Post-Failure Check: Check if an abnormal failure already occurred at or before t
+        # Protocol Amendment 2.1 line 201: 'post-failure step -> label -1 (strictly excluded)'
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
```

### 7.2 Safety Invariant of the Proposed Patch
- **Non-Breaking for Valid Pre-Failure Cases:** For any valid pre-failure window ($t < T_{\text{fail}}$), `prior_fails` is empty. The code proceeds directly to Section 1 (Critical/Warning evaluation) or Section 2 (Normal follow-up evaluation).
- **Zero Impact on Pure-Normal Entities:** For pure-normal control tasks, `task_events` contains zero `FAIL` events. `prior_fails` is always empty.
- **Strictly Purges Post-Failure Contamination:** Any observation at $t \ge T_{\text{fail}}$ immediately returns `FailureLabel.UNKNOWN, None`. When `WindowGenerator(exclude_unknown=True)` runs, these observations are discarded.

---

## 8. Step 7 — Beginner-Friendly Explanation

### 1. What was the bug?
Imagine you are building an alarm system to predict earthquakes 15 to 30 minutes before they happen.
If an earthquake occurs at 2:00 PM, and the building collapses:
- Any sensor reading taken at 1:40 PM is a valid early-warning observation.
- But what about sensor readings taken at 2:30 PM, after the building has already collapsed?
The computer code checked: *"Will an earthquake happen in the next 30 minutes after 2:30 PM?"*
The answer was: *"No, there are no more earthquakes scheduled today."*
Then the code asked: *"Do we have sensor data for 30 minutes after 2:30 PM?"*
The answer was: *"Yes, the sensors kept recording rubble until 3:30 PM."*
So the buggy code concluded: *"Great! 2:30 PM was a completely NORMAL, calm day!"*
In reality, the building had already collapsed at 2:00 PM! The code called post-crash rubble "Normal" simply because it didn't check whether the crash had already happened!

### 2. Why does it matter?
1. **Teaching Models False Patterns:** During training, the machine learning models were shown post-crash telemetry and told: *"This is what normal, healthy machines look like."* This confused the models.
2. **Evaluation Invalidation:** During the final test, 149 post-crash windows were evaluated as "Normal" test questions. If a model detected that the machine was crashed or abnormal, it was penalized as a "false alarm," even though the machine had literally crashed!
3. **Protocol Contradiction:** The research plan explicitly promised in writing that post-failure data would be labeled UNKNOWN (-1) and excluded. An implementation omission allowed it to slip through.

### 3. What was changed?
We added a check at the very beginning of the labeling function:
*"Has an abnormal failure event already occurred at or before this timestamp?"*
If yes, the code immediately marks the observation as `UNKNOWN` (-1). When the dataset builder runs, it skips all `UNKNOWN` observations, ensuring only legitimate, pre-failure data is used.

### 4. How many records were affected?
Across the entire repository, exactly **2,052 prediction windows** were affected:
- **1,403 windows in Training** (including 1,130 post-failure normal windows).
- **237 windows in Validation** (including 197 post-failure normal windows).
- **216 windows in Test** (including 145 post-failure normal windows).
- **196 windows in the Confirmatory Holdout** (149 post-failure normal, 7 cross-history normal, and 40 repeated-failure warning/critical windows).

### 5. What is now verified?
- All 5 numbers reported by earlier audits are 100% verified to the exact integer.
- The 15 regression test scenarios prove that the bug is fixed without altering valid normal, warning, or critical boundaries.
- The official operational result remains unchanged: **`NO_OPERATIONAL_WINNER`**. Models failed operational gates because of excessive false alarms on pure-normal machines, not because of post-failure data.

### 6. What remains uncertain?
- Whether retraining the models on clean training data (free of the 1,130 contaminated windows) will significantly improve their performance or reduce false alarms.
- The effect of parent-job clustering on model generalization (75%–80% of parent jobs overlapped between train and validation, but were completely disjoint from the holdout).

### 7. Why diagnostic results are NOT a new confirmatory experiment
In science, once you open the test answers and look at the exam results, that exam is "spent." You cannot erase your memory, tweak the rules, grade the same test again, and call it a brand-new, blinded confirmatory experiment.
The Phase 6 confirmatory holdout has been unblinded and inspected. Evaluating our patched logic on this same cohort provides valuable **diagnostic sensitivity analysis**, but it cannot claim to be a new confirmatory validation. A true confirmatory validation requires an untouched, uninspected dataset (Phase 7).

### 8. What is the safest next step?
1. Obtain authorization to apply the patch to [`src/labeling/target_generator.py`](file:///c:/Users/hp/Desktop/Research%20project/src/labeling/target_generator.py).
2. Generate a clean, corrected development partition (`train_clean`, `val_clean`, `test_clean`) using parent-job grouping.
3. Test cost-sensitive thresholds and probability calibration on the validation split.
4. If results show promise, sample a fresh holdout cohort for a true Phase 7 confirmatory evaluation.
