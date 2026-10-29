# Phase 7A.1 — Patch Safety Review & Methodological Reconciliation

**Project:** Temporal LSTM/Transformer Framework for Early Failure Prediction in Cloud Computing  
**Phase:** Phase 7A.1 — Controlled Patch Safety Audit & Architectural Verification  
**Author:** Independent Machine Learning Scientific Auditor & Pair Programmer  
**Date:** October 27, 2026  
**Governing Protocol:** [Protocol Amendment 2.1](file:///c:/Users/hp/Desktop/Research%20project/results/phase6/protocol_l5_amendment_2_1.json)  
**Primary Output Artifact:** [`results/phase7a/patch_safety_review.json`](file:///c:/Users/hp/Desktop/Research%20project/results/phase7a/patch_safety_review.json)  
**Final Audit Recommendation:** **`SAFE AFTER SPECIFIED CHANGES`**

---

## 1. Executive Summary & Audit Verdict

This audit resolves all five safety and architectural questions raised prior to authorizing modifications to [`src/labeling/target_generator.py`](file:///c:/Users/hp/Desktop/Research%20project/src/labeling/target_generator.py). 

Our final verdict is:
$$\mathbf{SAFE\ AFTER\ SPECIFIED\ CHANGES}$$

### Why it is NOT "Safe as Originally Proposed" Without Changes:
While the proposed patch to `TargetLabelGenerator.derive_label_for_task` is mathematically correct and fixes the post-failure labeling defect, our investigation of the wider pipeline revealed a critical architectural insight:
1. **The Window Generator Invariant:** [`src/labeling/window_generator.py`](file:///c:/Users/hp/Desktop/Research%20project/src/labeling/window_generator.py#L80) checks `target_label == FailureLabel.UNKNOWN` **only at the prediction endpoint** ($t_{\text{pred}}$). It **never checks** whether any intermediate steps in the 5-step input history $[t_{\text{pred}} - 4 \times 300\text{s}, t_{\text{pred}}]$ are labeled `UNKNOWN`.
2. **The Policy Dependence:**
   - Under **Policy A (Terminal Entity Lifecycle)**: Any observation at or after the entity's first failure ($t \ge T_{\text{fail}}^{(1)}$) is permanently labeled `UNKNOWN` (-1). Because time flows forward, no observation can ever have a post-failure step in its past without its endpoint also being at or after $T_{\text{fail}}^{(1)}$. Therefore, under Policy A, **zero sequence histories can ever cross a failure event**, and the single patch to `target_generator.py` is 100% mathematically sufficient.
   - Under **Policy B (Segmented Episode-Level Rescheduling)**: If a restarted task is permitted to generate pre-failure alerts for a *second* failure ($T_{\text{fail}}^{(2)}$), `target_generator.py` alone is **NOT sufficient**, because `window_generator.py` will happily construct a 5-step sequence that crosses over the first failure event. Supporting Policy B would require modifying `window_generator.py`.
3. **The Specification Requirement:** To ensure 100% safety, Policy A must be formally ratified as the governing policy for Phase 7A diagnostic data, and `tests/test_phase7a_labeling_repair.py` must include tensor-level assertions verifying that no generated window spans across a failure event.

---

## 2. Reconcile the Counts

The audit report presented several figures: **1,621**, **149**, **7**, **2,052**, and split-level counts **1,403 / 237 / 216 / 196**. We audited all 12,327 prediction rows across the repository to establish their exact mathematical definitions and confirm mutual exclusivity.

### 2.1 The Grand Reconciliation Table

Every prediction record belongs to exactly one dataset split, has exactly one prediction timestamp, and falls into exactly one mutually exclusive category:

| Dataset Split | Source Artifact | Total Rows ($N$) | Category A: Post-All-Failures Normal ($t > T_{\text{last\_fail}}$) | Category B: Between-Failures Cross-History Normal | Category C: Between-Failures Clean Normal | Category D: Between-Failures Warning ($y=1$) | Category E: Between-Failures Critical ($y=2$) | Category F: Clean Pre-Failure Normal ($y=0$) | Category G: Pure-Normal Controls ($y=0$) | Clean Pre-Failure Warning/Critical ($y \in \{1,2\}$) | Total Affected Windows ($A+B+C+D+E$) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Phase 6.3 Train** | `data/processed/l5/train_metadata.csv` | 4,798 | **1,130** | **40** | **59** | 32 | 142 | 2,959 | 0 | 436 *(215 warn, 221 crit)* | **1,403** |
| **Phase 6.3 Val** | `data/processed/l5/val_metadata.csv` | 1,015 | **197** | **0** | **0** | 0 | 40 | 643 | 0 | 135 *(72 warn, 63 crit)* | **237** |
| **Phase 6.3 Test** | `data/processed/l5/test_metadata.csv` | 1,003 | **145** | **8** | **35** | 14 | 14 | 676 | 0 | 111 *(54 warn, 57 crit)* | **216** |
| **Confirmatory** | `results/phase6/confirmatory_predictions_l5.csv` | 5,511 | **149** | **7** | **0** | 14 | 26 | 140 | 4,515 | 660 *(286 warn, 374 crit)* | **196** |
| **TOTALS** | *All Cohorts Combined* | **12,327** | **1,621** | **55** | **94** | **60** | **222** | **4,418** | **4,515** | **1,342** | **2,052** |

### 2.2 Precise Definition of Every Historical Figure

1. **`1,621 Post-Failure Normal Windows`:**
   - **Exact Definition:** The total number of windows across all four repository partitions labeled `NORMAL` (0) that occur strictly after all abnormal failure events have completed for that entity ($t_{\text{pred}} > T_{\text{last\_fail}}$).
   - **Exact Arithmetic:** $1,130\text{ (train)} + 197\text{ (val)} + 145\text{ (test)} + 149\text{ (confirmatory)} = \mathbf{1,621}$.
   - **Mutual Exclusivity:** Mutually exclusive from all between-failure windows and all pre-failure windows.

2. **`149 Post-Failure Normal Confirmatory Windows`:**
   - **Exact Definition:** The subset of Category A belonging strictly to the Phase 6.4 confirmatory holdout ([`results/phase6/confirmatory_predictions_l5.csv`](file:///c:/Users/hp/Desktop/Research%20project/results/phase6/confirmatory_predictions_l5.csv)). These windows belong to 41 failure entities where $t_{\text{pred}} > T_{\text{fail}}$ and no subsequent failure exists in the trace.

3. **`7 Cross-Failure-History Normal Confirmatory Windows`:**
   - **Exact Definition:** Windows in the confirmatory holdout labeled `NORMAL` (0) that occurred between two failure events in entities `6224307516_2` (4 windows: indices 3392–3395) and `6224307516_3` (3 windows: indices 3402–3404). Their lookahead to failure 2 was $> 30\text{ min}$, but their trailing 5-step input history $[t_{\text{pred}} - 1200\text{s}, t_{\text{pred}}]$ crossed directly over failure 1 ($T_{\text{fail}} = 9232.6\text{s}$ and $9455.5\text{s}$).
   - **Sum of Contaminated Normal Windows in Holdout:** $149\text{ (post-all-fail)} + 7\text{ (cross-history)} = \mathbf{156}$.
   - **Remaining Clean Failure-Cohort Normal Windows in Holdout:** $296 - 156 = \mathbf{140}$.
   - **Total Clean Normal Windows in Holdout:** $4,515\text{ (pure-normal)} + 140 = \mathbf{4,655}$.

4. **`2,052 Total Affected Windows`:**
   - **Exact Definition:** Every window across the entire repository that occurred after at least one abnormal failure event in the entity's history ($t_{\text{pred}} \ge T_{\text{fail}}^{(1)}$). This includes post-all-failure normal windows (1,621), between-failure normal windows (149), between-failure warning windows (60), and between-failure critical windows (222).
   - **Exact Arithmetic:** $1,403\text{ (train)} + 237\text{ (val)} + 216\text{ (test)} + 196\text{ (confirmatory)} = \mathbf{2,052}$.
   - **Overlap Status:** Zero overlap. The sum of the four split counts equals 2,052 exactly.

---

## 3. Verify the Proposed Patch Against Actual Code

We audited [`src/labeling/target_generator.py`](file:///c:/Users/hp/Desktop/Research%20project/src/labeling/target_generator.py) against all project schemas, edge cases, and callers.

```python
# PROPOSED PATCH: src/labeling/target_generator.py (lines 53-73)
        t = prediction_time_sec

        # 1. Handle missing, null, or non-finite timestamps safely
        if pd.isna(t) or not np.isfinite(t):
            return FailureLabel.UNKNOWN, None

        # 2. Strict Prior/Concurrent Failure Check (Protocol Amendment 2.1 line 201)
        if not task_events.empty and "event_type" in task_events.columns and "time_sec" in task_events.columns:
            prior_fails = task_events[
                (task_events["event_type"] == self.event_cls.FAIL) &
                (task_events["time_sec"] <= t)
            ]
            if not prior_fails.empty:
                return FailureLabel.UNKNOWN, None

        # 3. Locate abnormal failure occurring strictly after prediction time t
        future_fails = pd.DataFrame()
        if not task_events.empty and "event_type" in task_events.columns and "time_sec" in task_events.columns:
            future_fails = task_events[
                (task_events["event_type"] == self.event_cls.FAIL) &
                (task_events["time_sec"] > t)
            ]
```

### 3.1 Verification Checklist

1. **Correct Event Type Enumeration:**
   - The code accesses `self.event_cls.FAIL`.
   - In [`src/ingestion/schema.py`](file:///c:/Users/hp/Desktop/Research%20project/src/ingestion/schema.py#L13-L49), `ClusterEventTypeV2.FAIL == 3` (Google Cluster Trace 2011) and `ClusterEventTypeV3.FAIL == 5` (2019 trace).
   - The factory `get_event_classifier(self.trace_version)` returns `ClusterEventTypeV2` for version `"2011"` and `ClusterEventTypeV3` for `"2019"`. Verified correct.

2. **DataFrame and Timestamp Safety Guards:**
   - If `task_events` is empty or missing `"event_type"` or `"time_sec"`, the patch avoids `KeyError` or index errors via safe boolean guards.
   - If `prediction_time_sec` is `None`, `NaN`, or `Inf`, the function immediately returns `FailureLabel.UNKNOWN, None`.

3. **Exact Failure Timestamp Collision ($t == T_{\text{fail}}$):**
   - In early failure prediction, an alert at the exact second of a crash ($\tau = 0.0\text{s}$) is retrospective detection, not early prediction.
   - The patch evaluates `task_events["time_sec"] <= t`. Because $T_{\text{fail}} \le t$, it triggers `prior_fails` and returns `FailureLabel.UNKNOWN, None`. Verified correct.

4. **Preservation of Pre-Failure Semantics:**
   - For any observation before a failure ($t < T_{\text{fail}}$), `prior_fails` is empty. The code executes identical downstream checks:
     - $0 < \tau \le 900.0\text{s} \to \text{CRITICAL}$ (2).
     - $900.0\text{s} < \tau \le 1800.0\text{s} \to \text{WARNING}$ (1).
     - $\tau > 1800.0\text{s} \to \text{NORMAL}$ (0).
     - Future `FINISH` event $\le t + 1800\text{s} \to \text{NORMAL}$ (0).
     - Future `KILL` or `EVICT` $\le t + 1800\text{s} \to \text{UNKNOWN}$ (-1).
     - Telemetry $< t + 1800\text{s}$ without terminal event $\to \text{UNKNOWN}$ (-1).

5. **Exclusion from Tensors and Datasets:**
   - In [`src/labeling/window_generator.py` line 80](file:///c:/Users/hp/Desktop/Research%20project/src/labeling/window_generator.py#L80):
     ```python
     target_label = labels[pred_idx]
     if self.exclude_unknown and target_label == FailureLabel.UNKNOWN:
         continue
     ```
   - Returning `UNKNOWN` (-1) guarantees that `WindowGenerator` skips that endpoint. The window is **never added to the window list, never enters tensor $X$, never enters label vector $y$, and never enters metadata CSVs**.

---

## 4. Investigate Repeated Failures Carefully

We reconstructed the concrete lifecycle timelines for the primary repeated-failure entities in the confirmatory holdout using raw trace events:

### 4.1 Timeline Case 1: Entity `6224307516_2`
- **Raw FAIL Events:** 4 failures at $t = 7101.8\text{s}$, $9232.6\text{s}$, $13367.6\text{s}$, and $13651.1\text{s}$.
- **Lifecycle Sequence:**
  - $t = 6494.8\text{s}$: Container scheduled.
  - $t = 7101.8\text{s}$: **FAIL (Event 3)**.
  - $t = 7103.2\text{s}$: Rescheduled by Borg.
  - $t = 7800\text{s} - 9000\text{s}$: 5 prediction windows generated with labels $y \in \{1, 2\}$ pointing to failure 2 at $9232.6\text{s}$.
  - $t = 9232.6\text{s}$: **FAIL (Event 3)**.
  - $t = 9234.0\text{s}$: Rescheduled by Borg.
  - $t = 9300\text{s} - 10200\text{s}$: 4 prediction windows generated with label $y = 0$ (`NORMAL`) pointing to failure 3 at $13367.6\text{s}$ ($\tau > 3000\text{s}$).

| Prediction Time ($t_{\text{pred}}$) | 5-Step History Interval $[t_{\text{pred}} - 1200\text{s}, t_{\text{pred}}]$ | Relation to Failures | Original Label | Policy A Label & Reason | Policy B Label & Reason |
| :---: | :---: | :--- | :---: | :--- | :--- |
| **7800.0s** | $[6600.0\text{s}, 7800.0\text{s}]$ | Crosses Fail 1 ($7101.8\text{s}$) | `WARNING` (1) | **UNKNOWN (-1)**: Post-Fail 1 | **UNKNOWN (-1)**: History crosses Fail 1 |
| **8100.0s** | $[6900.0\text{s}, 8100.0\text{s}]$ | Crosses Fail 1 ($7101.8\text{s}$) | `WARNING` (1) | **UNKNOWN (-1)**: Post-Fail 1 | **UNKNOWN (-1)**: History crosses Fail 1 |
| **8400.0s** | $[7200.0\text{s}, 8400.0\text{s}]$ | Clean after Fail 1, before Fail 2 | `CRITICAL` (2) | **UNKNOWN (-1)**: Post-Fail 1 | **CRITICAL (2)**: Pre-Fail 2 ($\tau = 832.6\text{s}$) |
| **8700.0s** | $[7500.0\text{s}, 8700.0\text{s}]$ | Clean after Fail 1, before Fail 2 | `CRITICAL` (2) | **UNKNOWN (-1)**: Post-Fail 1 | **CRITICAL (2)**: Pre-Fail 2 ($\tau = 532.6\text{s}$) |
| **9000.0s** | $[7800.0\text{s}, 9000.0\text{s}]$ | Clean after Fail 1, before Fail 2 | `CRITICAL` (2) | **UNKNOWN (-1)**: Post-Fail 1 | **CRITICAL (2)**: Pre-Fail 2 ($\tau = 232.6\text{s}$) |
| **9300.0s** | $[8100.0\text{s}, 9300.0\text{s}]$ | Crosses Fail 2 ($9232.6\text{s}$) | `NORMAL` (0) | **UNKNOWN (-1)**: Post-Fail 2 | **UNKNOWN (-1)**: History crosses Fail 2 |
| **9600.0s** | $[8400.0\text{s}, 9600.0\text{s}]$ | Crosses Fail 2 ($9232.6\text{s}$) | `NORMAL` (0) | **UNKNOWN (-1)**: Post-Fail 2 | **UNKNOWN (-1)**: History crosses Fail 2 |
| **9900.0s** | $[8700.0\text{s}, 9900.0\text{s}]$ | Crosses Fail 2 ($9232.6\text{s}$) | `NORMAL` (0) | **UNKNOWN (-1)**: Post-Fail 2 | **UNKNOWN (-1)**: History crosses Fail 2 |
| **10200.0s** | $[9000.0\text{s}, 10200.0\text{s}]$ | Crosses Fail 2 ($9232.6\text{s}$) | `NORMAL` (0) | **UNKNOWN (-1)**: Post-Fail 2 | **UNKNOWN (-1)**: History crosses Fail 2 |

### 4.2 Timeline Case 2: Entity `6250804554_0`
- **Raw FAIL Events:** 3 failures at $t = 1108.9\text{s}$, $2939.1\text{s}$, and $4776.2\text{s}$.
- At $t = 2100.0\text{s}$: History is $[900.0\text{s}, 2100.0\text{s}]$. This interval spans across failure 1 at $1108.9\text{s}$.
- At $t = 2400.0\text{s}$ and $2700.0\text{s}$: Histories $[1200.0\text{s}, 2400.0\text{s}]$ and $[1500.0\text{s}, 2700.0\text{s}]$ occur cleanly after failure 1 and before failure 2.
- At $t = 3000.0\text{s} - 3900.0\text{s}$: Windows cross failure 2 at $2939.1\text{s}$.

### 4.3 Policy Decision: Policy A vs. Policy B

```
+-------------------------------------------------------------------------------------------------------+
| COMPARISON OF REPEATED-FAILURE LIFECYCLE POLICIES                                                     |
+------------------------------------+------------------------------------------------------------------+
| POLICY A: Terminal Entity Lifecycle| POLICY B: Segmented Episode-Level Rescheduling                  |
+------------------------------------+------------------------------------------------------------------+
| - Once an entity fails, all future | - Permits pre-failure alerts for subsequent failure episodes if |
|   telemetry is censored (-1).      |   the trailing history does not cross the prior failure.        |
| - Strictly complies with Protocol  | - Contradicts Protocol line 201: ("post-failure step -> label -1|
|   line 201 ("post-failure step ->  |   strictly excluded").                                          |
|   label -1 strictly excluded").    | - Requires changing WindowGenerator to inspect history spans.    |
| - Mathematically guarantees zero   | - Leaves edge-case ambiguity when container restarts retain      |
|   cross-failure histories.         |   un-cleared memory leaks or machine cache effects.             |
| - Result: 5,315 clean holdout wins.| - Result: 5,355 clean holdout wins (Phase 6.8 Scenario 3).       |
+------------------------------------+------------------------------------------------------------------+
```

**Auditor Finding:** Policy A is the only policy that is fully consistent with the ratified protocol text, the historical cohort verification documents, and the existing architecture of `window_generator.py`.

---

## 5. Validate the Regression Tests

We evaluated [`tests/test_phase7a_labeling_repair.py`](file:///c:/Users/hp/Desktop/Research%20project/tests/test_phase7a_labeling_repair.py).

### 5.1 Confirmation of Production Function Execution
- The test suite imports the production class directly:
  ```python
  from src.labeling.target_generator import TargetLabelGenerator
  ```
- It does **not** test a mock or scratch harness.
- When executed against the current, unpatched repository code, pytest returned:
  `tests\test_phase7a_labeling_repair.py ........FFFF... [100%]`
  `4 failed, 11 passed in 0.74s`

### 5.2 Forensic Explanation of the Four Test Failures
1. **`test_scenario_09_telemetry_after_failure_strictly_censored`:**
   - Condition: Task fails at $t - 500\text{s}$; telemetry continues to $t + 2000\text{s}$.
   - Failure: `assert <FailureLabel.NORMAL: 0> == <FailureLabel.UNKNOWN: -1>`.
   - Reason: Unpatched code found no future failures, saw telemetry $> t + 1800\text{s}$, and returned `NORMAL` (0).
2. **`test_scenario_10_repeated_failures_step_after_prior_failure`:**
   - Condition: Task fails at $t - 500\text{s}$ and again at $t + 600\text{s}$.
   - Failure: `assert <FailureLabel.CRITICAL: 2> == <FailureLabel.UNKNOWN: -1>`.
   - Reason: Unpatched code looked forward to failure 2, ignored the prior crash, and returned `CRITICAL` (2).
3. **`test_scenario_11_window_between_two_failures_strictly_censored`:**
   - Condition: Task fails at $t - 1000\text{s}$ and again at $t + 3000\text{s}$.
   - Failure: `assert <FailureLabel.NORMAL: 0> == <FailureLabel.UNKNOWN: -1>`.
   - Reason: Unpatched code saw lookahead $\tau = 3000\text{s} > 1800\text{s}$ and returned `NORMAL` (0).
4. **`test_scenario_12_failure_exactly_at_prediction_timestamp`:**
   - Condition: Failure occurs at exact second $t$ ($\tau = 0.0\text{s}$).
   - Failure: `assert <FailureLabel.NORMAL: 0> == <FailureLabel.UNKNOWN: -1>`.
   - Reason: Unpatched code queried `time_sec > t` (which excluded $t$), found zero future failures, and returned `NORMAL` (0).

---

## 6. Check the Wider Pipeline

We audited the relationship between `TargetLabelGenerator`, `FeatureEngineer`, and `WindowGenerator`:

### 6.1 Does `window_generator.py` Exclude UNKNOWN Windows?
**YES.** Line 80 of `src/labeling/window_generator.py` explicitly executes:
```python
if self.exclude_unknown and target_label == FailureLabel.UNKNOWN:
    continue
```
Because all extraction and training pipelines run with `exclude_unknown=True`, returning `FailureLabel.UNKNOWN` (-1) guarantees that the prediction endpoint is dropped and no window is created.

### 6.2 Can a 5-Step Sequence Cross a Failure if its Endpoint is Labeled UNKNOWN?
**NO.** If the endpoint is labeled `UNKNOWN`, the entire window is skipped. No tensor is built.

### 6.3 Can a Sequence Cross a Failure if its Endpoint is Labeled NORMAL, WARNING, or CRITICAL?
- **Under Policy B:** YES. Because `window_generator.py` does not check the history steps $[t_{\text{pred}} - 1200\text{s}, t_{\text{pred}}]$, if an observation after a restart is labeled Warning or Critical, its trailing history can span across the crash.
- **Under Policy A:** NO. Any observation at $t \ge T_{\text{fail}}^{(1)}$ has `prior_fails.empty == False` and receives label `UNKNOWN` (-1). Because pre-failure observations ($t < T_{\text{fail}}^{(1)}$) can never have a future failure in their past, **it is mathematically impossible for any window to cross a failure under Policy A**.

### 6.4 Are Features Calculated Using Post-Failure Telemetry?
**NO for Pre-Failure Windows.** We audited [`src/features/engineer.py`](file:///c:/Users/hp/Desktop/Research%20project/src/features/engineer.py#L48-L100). All feature operators are **strictly trailing**:
- Lags: `shift(1), shift(2), shift(3), shift(6)`
- Differences: `df["cpu_utilization"] - shift(1)`
- Rolling statistics: `rolling(w, min_periods=1).mean()`
- Slope: OLS slope over trailing 6 bins.

Because there are **zero forward-looking operators**, the feature values for all pre-failure observations are mathematically identical whether post-failure telemetry exists in the dataframe or is purged beforehand.

### 6.5 Would a Corrected Dataset Require Regenerating Features or Only Labels?
- **Feature Math:** The mathematical formulas and 31 feature columns on pre-failure rows are 100% correct and do not need re-engineering.
- **Dataset Tensors:** Because 1,130 training rows (and 149 confirmatory rows) will change from label 0 to label -1, `WindowGenerator` will skip them, reducing the window count $N$. Therefore, the output arrays $X \in \mathbb{R}^{N \times 5 \times 31}$ and $y \in \mathbb{R}^N$ **must be regenerated** by re-running `WindowGenerator` on the correctly labeled dataframe.

---

## 7. Audit Recommendations & Authorization Conditions

### Verdict: **`SAFE AFTER SPECIFIED CHANGES`**

We recommend authorizing the repair subject to the following three mandatory specifications:

1. **Ratify Policy A as Governing:** Formally adopt Policy A (Terminal Entity Lifecycle: any observation at $t \ge T_{\text{fail}}^{(1)}$ is labeled `UNKNOWN`). This guarantees that no sequence history can ever cross a crash without requiring intrusive refactoring of `window_generator.py`.
2. **Mandate Segregated Diagnostic Outputs:** The patch must only be used to generate new diagnostic datasets in `data/processed/phase7a_diagnostic/` and `results/phase7a/`. Historical artifacts in `data/processed/l5/` and `results/phase6/` must remain permanently untouched.
3. **Add Tensor-Level Window Generator Tests:** Expand [`tests/test_phase7a_labeling_repair.py`](file:///c:/Users/hp/Desktop/Research%20project/tests/test_phase7a_labeling_repair.py) to include an end-to-end integration test asserting that `WindowGenerator(exclude_unknown=True)` generates zero windows with post-failure endpoints and zero windows with cross-failure histories.
