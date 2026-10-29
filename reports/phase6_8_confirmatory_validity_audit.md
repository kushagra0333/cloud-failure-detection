# Phase 6.8 — Confirmatory Cohort Validity, Temporal Censoring, and Metric Integrity Audit

**Project:** Temporal LSTM/Transformer Framework for Early Failure Prediction in Cloud Computing  
**Auditor:** Independent Machine Learning Scientific Auditor  
**Date of Audit:** October 27, 2026  
**Governing Protocol:** [Protocol Amendment 2.1](file:///c:/Users/hp/Desktop/Research%20project/results/phase6/protocol_l5_amendment_2_1.json)  
**Evaluated Cohort:** Frozen $L=5$ Confirmatory Holdout ($N = 5,511$ primary windows across 335 evaluable tasks)  
**Primary Audit Artifact:** [`results/phase6/phase6_8_confirmatory_validity_audit.json`](file:///c:/Users/hp/Desktop/Research%20project/results/phase6/phase6_8_confirmatory_validity_audit.json)  
**Official Operational Verdict:** **`NO_OPERATIONAL_WINNER`** (Reaffirmed as immutable)  
**Final Scientific Validity Verdict:** **`MATERIAL PROTOCOL VIOLATION FOUND`** (Verdict C)

---

## 1. Executive Summary & Preservation Invariants

### 1.1 Objective and Central Scientific Question
Phase 6.7 independently reconstructed raw failure event timestamps from the Google Cluster Trace and resolved the Phase 6.6 temporal classification of the 296 failure-cohort normal windows from 140 pre-failure / 156 post-failure to **147 pre-failure / 149 post-failure**.

This Phase 6.8 audit investigates the fundamental methodological question:
> **Were any windows included in the confirmatory evaluation metrics that should have been censored, labeled UNKNOWN (-1), or excluded under the frozen protocol?**

This inquiry is strictly read-only and post hoc. In accordance with rigorous scientific integrity standards:
- No models were retrained or tuned.
- No decision thresholds, predictions, or ground-truth labels were modified.
- No historical artifacts from Phase 6.4, 6.5, 6.6, or 6.7 were modified or overwritten.
- The official operational verdict remains permanently locked as **`NO_OPERATIONAL_WINNER`**.
- Inconvenient windows are not removed to alter model pass/fail outcomes. Counterfactual evaluations are presented strictly as post-hoc diagnostic sensitivity analyses.

### 1.2 Summary of Primary Findings
1. **Material Censoring Violation Identified:** Exactly **149 post-failure recovery windows** belonging to failure-cohort entities were labeled `NORMAL` (0) and included in the confirmatory evaluation, directly violating the pre-registered protocol mandate:  
   *`"censored lookahead (< 1800s without FINISH) or post-failure step -> label -1 (strictly excluded)"`* ([`protocol_l5_amendment_2_1.json` line 201](file:///c:/Users/hp/Desktop/Research%20project/results/phase6/protocol_l5_amendment_2_1.json#L201)).
2. **Root Cause in Implementation:** The defect was traced to [`src/labeling/target_generator.py`](file:///c:/Users/hp/Desktop/Research%20project/src/labeling/target_generator.py#L73-L77). When a task had no *future* failure events (`future_fails.empty`), the target generator evaluated whether telemetry extended at least 30 minutes ahead. If true, it silently returned `FailureLabel.NORMAL` (0), completely omitting any check for whether a failure had already occurred in the *past*.
3. **Contaminated Sequence Histories:** In addition to the 149 post-failure windows, **7 pre-failure normal windows** (in tasks `6224307516_2` and `6224307516_3`) possessed valid pre-failure lookahead ($\tau > 3000\text{ s} > 30\text{ min}$ before failure 2), but their trailing 5-step input history spanned across an earlier failure event.
4. **Valid Prediction Windows:** The remaining **5,355 windows** (4,515 pure-normal windows, 400 Critical windows, 300 Warning windows, and 140 pre-failure clean normal windows) strictly satisfy all causal, temporal, and history invariants.
5. **Operational Invariance:** Diagnostic counterfactual re-evaluations prove that the operational failure of the $L=5$ framework is **100% robust and invariant** across all censoring scenarios. When all 149 post-failure windows are censored, all four models still fail both the Window FPR gate (19.31%–33.91% vs. $\le 5.0\%$) and the Entity FAR gate (31.82%–64.14% vs. $\le 12.0\%$).

---

## 2. Authoritative Frozen Temporal Rules & Source Traceability

We extracted the authoritative temporal rules governing failure prediction and censoring from Protocol Amendment 2.1, the original Phase 6.2 protocol, and the core implementation modules:

| Rule Category | Exact Protocol Definition | Source Location | Operational Enforcement |
| :--- | :--- | :--- | :--- |
| **Critical Horizon** | `0 < time_to_failure <= 900.0s (0 < tau <= 15m) -> y=2` | `protocol_l5_amendment_2_1.json:198` | Strict retrospective sliding window; earliest abnormal failure in $(t, t + 15\text{m}]$. |
| **Warning Horizon** | `900.0s < time_to_failure <= 1800.0s (15m < tau <= 30m) -> y=1` | `protocol_l5_amendment_2_1.json:199` | Strict retrospective sliding window; earliest abnormal failure in $(t + 15\text{m}, t + 30\text{m}]$. |
| **Normal Horizon** | `time_to_failure > 1800.0s with >= 1800.0s follow-up or FINISH event -> y=0` | `protocol_l5_amendment_2_1.json:200` | No failure in $(t, t + 30\text{m}]$ AND verified active survival for $\ge 30\text{m}$ or normal `FINISH`. |
| **Unknown / Censored** | `censored lookahead (< 1800s without FINISH) or post-failure step -> label -1 (strictly excluded)` | `protocol_l5_amendment_2_1.json:201`; `protocol_l5.json:36`; `cohort_verification.json:92` | Steps with $< 30\text{m}$ unverified lookahead or occurring post-failure must be labeled -1 and excluded. |
| **Primary Cohort Denominators** | 5,511 total windows (400 Critical, 300 Warning, 4,811 Normal: 4,515 pure-normal, 296 failure-cohort normal) across 335 evaluable tasks (137 fail, 198 pure-normal). | `protocol_l5_amendment_2_1.json` Tenets 1–3 | Formally ratified confirmatory population manifest. |
| **Operational Gates** | Window FPR $\le 5.0\%$ (max 240/4,811 FPs); Entity FAR $\le 12.0\%$ (max 23/198 alert entities). | `protocol_l5_amendment_2_1.json` Tenets 5–6 | Dual operational gate criteria for deployability. |
| **Disaggregated Reporting** | Mandates reporting pure-normal FPR (FP/4,515) and failure-cohort normal FPR (FP/296) alongside pooled FPR. | `protocol_l5_amendment_2_1.json` Tenet 7 | Distinguishes non-failing workload behavior from early pre-failure dynamics. |

### The Core Protocol Contradiction
A fundamental contradiction exists within Protocol Amendment 2.1:
- **Normative Causal Text:** Line 201 states that any *"post-failure step -> label -1 (strictly excluded)"*.
- **Empirical Manifest Tenet:** Tenet 3 ratifies and locks exactly *"4,515 pure-normal windows + 296 failure-cohort normal windows = 4,811 total normal windows"*.
- **Historical Assumption:** Tenet 7 explicitly described these 296 windows as *"early pre-failure telemetry dynamics"*, revealing that the protocol authors assumed all 296 windows were pre-failure observations. The presence of 149 post-failure windows was an undetected implementation bug embedded into the ratified manifest.

---

## 3. Reconciliation of Every Confirmatory Window Against Raw Events

Every single row of the 5,511 confirmatory prediction records in [`results/phase6/confirmatory_predictions_l5.csv`](file:///c:/Users/hp/Desktop/Research%20project/results/phase6/confirmatory_predictions_l5.csv) was independently matched against raw Google Cluster Trace lifecycle events (`data/raw/sample_task_events*.csv.gz`, filtering for event type 3 = `FAIL` in Borg 2011 schema).

```
Total Confirmatory Windows Audited: 5,511
├── Pure-Normal Control Windows (N = 4,515 across 198 entities)
│   ├── Valid Normal Windows (tau > 1800s or FINISH, zero FAIL events in trace): 4,515 (100.0%)
│   └── Temporal / Censoring Violations: 0 (0.0%)
├── Pre-Failure Critical Windows (N = 400 across 137 entities)
│   ├── Valid Critical Windows (0.1906s <= tau <= 892.8226s): 400 (100.0%)
│   └── Temporal / Censoring Violations: 0 (0.0%)
├── Pre-Failure Warning Windows (N = 300 across 137 entities)
│   ├── Valid Warning Windows (901.6868s <= tau <= 1792.8226s): 300 (100.0%)
│   └── Temporal / Censoring Violations: 0 (0.0%)
└── Failure-Cohort Normal Windows (N = 296 across 109 entities)
    ├── Valid Clean Pre-Failure Normal Windows (tau > 1800s, zero prior fail): 140 (47.30%)
    ├── Pre-Failure Normal with Cross-Failure History (tau > 1800s, history crosses prior fail): 7 (2.36%)
    └── Post-Failure Recovery Normal Windows (t_pred > t_prev_fail, no future fail): 149 (50.34%)
```

### 3.1 Horizon Boundary Verification
- **Critical Horizon:** All 400 Critical windows strictly satisfy $0 < \tau \le 900.0\text{ s}$. The empirical range is $[0.1906\text{ s}, 892.8226\text{ s}]$. Exactly zero windows exceed 900 seconds or occur post-failure.
- **Warning Horizon:** All 300 Warning windows strictly satisfy $900.0\text{ s} < \tau \le 1800.0\text{ s}$. The empirical range is $[901.6868\text{ s}, 1792.8226\text{ s}]$. Exactly zero windows violate the Warning boundary.
- **Pure-Normal Controls:** Across all 198 pure-normal control tasks, exactly zero abnormal failure events (`FAIL == 3`) exist in the trace. All 4,515 windows represent legitimate non-failing execution.

---

## 4. Specifically Investigating the 296 Failure-Cohort Normal Windows

### 4.1 Root Cause in `TargetLabelGenerator`
We audited the source code of [`src/labeling/target_generator.py`](file:///c:/Users/hp/Desktop/Research%20project/src/labeling/target_generator.py):

```python
# Lines 55-77 of src/labeling/target_generator.py:
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
        return FailureLabel.NORMAL, tau

# 2. No future abnormal failure found. Check if future is legitimately NORMAL or UNKNOWN
if telemetry_max_time_sec >= t + self.warning_horizon_sec:
    return FailureLabel.NORMAL, None  # <-- BUG: OMITTED CHECK FOR PRIOR FAILURES!
```

When a task fails at $t_1$, and telemetry continues past $t_1$ (e.g., container cleanup, Borg retry delay, or restart within the same task ID):
1. For any evaluation window at $t > t_1$ where no subsequent failure occurs, `future_fails` is empty.
2. The generator checks `telemetry_max_time_sec >= t + 1800.0`.
3. If true, it returns `FailureLabel.NORMAL` (0) instead of checking whether `task_events["time_sec"] <= t` contained a prior `FAIL` event!
4. Consequently, 149 post-failure observations were labeled `NORMAL` (0) instead of `UNKNOWN` (-1).
5. When `WindowGenerator(exclude_unknown=True)` ran, it excluded label -1, but included these 149 windows because they had been mislabeled as label 0!

### 4.2 Comprehensive 296-Window Reconciliation Table

| Window Index Range | Entity ID | Window Count | Prediction Time Span | Temporal Relation to Failures | History Invariant | Frozen Protocol Permitted? | Scientific Status |
| :--- | :--- | :---: | :---: | :--- | :--- | :---: | :--- |
| **Various (Clean Pre-Fail)** | 107 entities | **140** | $1200\text{ s} - 8700\text{ s}$ | $1801.7\text{ s} \le \tau \le 4259.3\text{ s}$ before first failure. | Clean; zero prior failure in task lifecycle. | **YES** | **VALID PRE-FAILURE NORMAL** |
| **Indices 469–472** | `6224307516_2` | **4** | $9300\text{ s} - 10200\text{ s}$ | $3167.6\text{ s} \le \tau \le 4067.6\text{ s}$ before failure 2 (at $13367.6\text{ s}$). | Contaminated; history $[8100, 9300]$ crosses failure 1 at $9232.6\text{ s}$. | Ambiguous | **PRE-FAIL CROSS-HISTORY** |
| **Indices 479–481** | `6224307516_3` | **3** | $9600\text{ s} - 10200\text{ s}$ | $3659.3\text{ s} \le \tau \le 4259.3\text{ s}$ before failure 2 (at $13859.3\text{ s}$). | Contaminated; history $[8400, 9600]$ crosses failure 1 at $9455.5\text{ s}$. | Ambiguous | **PRE-FAIL CROSS-HISTORY** |
| **Indices 482–504, etc.** | 41 entities | **149** | $2700\text{ s} - 10800\text{ s}$ | $t_{\text{pred}} > t_{\text{fail}}$; no subsequent failure in task lifecycle. | 131 cross prior failure; 18 entirely post-failure. | **NO** | **POST-FAILURE CENSORING VIOLATION** |

### 4.3 Detailed Audit of the 149 Post-Failure Windows
- **Temporal Position:** All 149 windows have $t_{\text{pred}} > T_{\text{fail}}$. The task had already failed abnormally.
- **Input History Contamination:** For 131 of these 149 windows (87.9%), the trailing 5-step history span $[t_{\text{pred}} - 1200\text{ s}, t_{\text{pred}}]$ directly encompassed the failure timestamp ($t_{\text{start}} \le T_{\text{fail}} \le t_{\text{pred}}$).
- **Physical Interpretation:** These windows represent post-mortem container telemetry or scheduler retries executing within the same task ID. Evaluating these windows as "Normal" rewards or penalizes models based on post-crash telemetry, directly contradicting the causal premise of early failure prediction.

---

## 5. Confirmatory Denominators & Diagnostic Counterfactual Analysis

To evaluate the mathematical impact of this protocol violation, we computed diagnostic counterfactual metrics across four clearly defined evaluation populations.

> [!WARNING]
> **Methodological Disclaimer:** These counterfactual evaluations are post-hoc diagnostic analyses intended solely to assess the sensitivity of the scientific conclusions to cohort censoring. They are **not** replacement confirmatory results, and must never be used for threshold tuning or model selection.

### 5.1 Counterfactual Scenario Definitions
- **Scenario 0 (As Frozen):** The exact ratified evaluation manifest ($N = 5,511$ windows, 4,811 normal windows).
- **Scenario 1 (Censoring 149 Post-Failure Windows):** Removes the 149 post-failure recovery windows that violated protocol line 201 ($N = 5,362$ windows, 4,662 normal windows).
- **Scenario 2 (Censoring All 296 Failure-Cohort Normal Windows):** Restricts the Normal evaluation population strictly to the 4,515 pure-normal control windows ($N = 5,215$ windows, 4,515 normal windows).
- **Scenario 3 (Censoring 149 Post-Failure + 7 Cross-History Windows):** Retains only strictly clean observations ($N = 5,355$ windows, 4,655 normal windows).

### 5.2 Comparative Counterfactual Performance Table

| Scenario & Evaluation Population | Model Name | Normal Windows ($N$) | False Positives ($N$) | Window FPR (%) | FPR Gate ($\le 5.0\%$) | Pure-Normal Entities ($N$) | Alert Entities ($N$) | Entity FAR (%) | FAR Gate ($\le 12.0\%$) | Macro-F1 | Overall Gate Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Scenario 0: As Frozen**  <br>*(Ratified Confirmatory)*  <br>Total $N = 5,511$ | Random Forest  <br>Temporal LSTM  <br>Compact Transformer  <br>Representation Hybrid | 4,811  <br>4,811  <br>4,811  <br>4,811 | 1,029  <br>1,293  <br>975  <br>1,667 | 21.39%  <br>26.88%  <br>20.27%  <br>34.65% | FAIL  <br>FAIL  <br>FAIL  <br>FAIL | 198  <br>198  <br>198  <br>198 | 70  <br>83  <br>63  <br>127 | 35.35%  <br>41.92%  <br>31.82%  <br>64.14% | FAIL  <br>FAIL  <br>FAIL  <br>FAIL | 0.6111  <br>0.4425  <br>0.4479  <br>0.3924 | **FAIL**  <br>**FAIL**  <br>**FAIL**  <br>**FAIL** |
| **Scenario 1: Censoring 149 Post-Fail**  <br>*(Strict Protocol Rule)*  <br>Total $N = 5,362$ | Random Forest  <br>Temporal LSTM  <br>Compact Transformer  <br>Representation Hybrid | 4,662  <br>4,662  <br>4,662  <br>4,662 | 990  <br>1,193  <br>900  <br>1,581 | 21.24%  <br>25.59%  <br>19.31%  <br>33.91% | FAIL  <br>FAIL  <br>FAIL  <br>FAIL | 198  <br>198  <br>198  <br>198 | 70  <br>83  <br>63  <br>127 | 35.35%  <br>41.92%  <br>31.82%  <br>64.14% | FAIL  <br>FAIL  <br>FAIL  <br>FAIL | 0.6144  <br>0.4515  <br>0.4559  <br>0.3985 | **FAIL**  <br>**FAIL**  <br>**FAIL**  <br>**FAIL** |
| **Scenario 2: Pure-Normal Only**  <br>*(Zero Failure Tasks in Normal)*  <br>Total $N = 5,215$ | Random Forest  <br>Temporal LSTM  <br>Compact Transformer  <br>Representation Hybrid | 4,515  <br>4,515  <br>4,515  <br>4,515 | 978  <br>1,055  <br>787  <br>1,475 | 21.66%  <br>23.37%  <br>17.43%  <br>32.67% | FAIL  <br>FAIL  <br>FAIL  <br>FAIL | 198  <br>198  <br>198  <br>198 | 70  <br>83  <br>63  <br>127 | 35.35%  <br>41.92%  <br>31.82%  <br>64.14% | FAIL  <br>FAIL  <br>FAIL  <br>FAIL | 0.6168  <br>0.4646  <br>0.4679  <br>0.4073 | **FAIL**  <br>**FAIL**  <br>**FAIL**  <br>**FAIL** |
| **Scenario 3: Strictly Clean History**  <br>*(Censoring 149 Post + 7 Cross)*  <br>Total $N = 5,355$ | Random Forest  <br>Temporal LSTM  <br>Compact Transformer  <br>Representation Hybrid | 4,655  <br>4,655  <br>4,655  <br>4,655 | 990  <br>1,193  <br>900  <br>1,581 | 21.27%  <br>25.63%  <br>19.33%  <br>33.96% | FAIL  <br>FAIL  <br>FAIL  <br>FAIL | 198  <br>198  <br>198  <br>198 | 70  <br>83  <br>63  <br>127 | 35.35%  <br>41.92%  <br>31.82%  <br>64.14% | FAIL  <br>FAIL  <br>FAIL  <br>FAIL | 0.6143  <br>0.4515  <br>0.4558  <br>0.3984 | **FAIL**  <br>**FAIL**  <br>**FAIL**  <br>**FAIL** |

### 5.3 Mathematical Proof of Verdict Invariance
The counterfactual analysis proves three crucial mathematical truths:
1. **Entity FAR is Structurally Invariant:** The Entity False Alarm Rate gate is evaluated strictly across the 198 pure-normal control tasks. Because failure-cohort normal windows belong entirely to failure tasks, censoring or uncensoring them changes Entity FAR by **0.00%**. All models fail the FAR gate by a factor of $2.6\times$ to $5.3\times$.
2. **Window FPR Gate Failure is Robust:** Even under the most favorable censoring scenario (Scenario 2, Compact Transformer), Window FPR drops only to $17.43\%$, which is still **$3.5\times$ higher** than the allowable $5.0\%$ threshold.
3. **Operational Verdict Invariance:** Under every conceivable interpretation of the protocol, **zero models pass the operational gates**. The historical verdict **`NO_OPERATIONAL_WINNER`** is robust, immutable, and mathematically decoupled from the post-failure censoring violation.

---

## 6. Audit of the Phase 6.7 Implementation

We audited [`scripts/run_phase6_7_independent_scientific_audit.py`](file:///c:/Users/hp/Desktop/Research%20project/scripts/run_phase6_7_independent_scientific_audit.py) to assess the accuracy of its event lineage and statistical calculations:

### 6.1 Verified Correct Components
- **Event Type Enumeration:** Phase 6.7 correctly mapped event type 3 to `FAIL` under the Google Cluster Trace 2011 schema (`fail_events_raw = events_raw[events_raw['event_type'] == 3]`).
- **Timestamp Scaling:** Timestamp microsecond division (`timestamp / 1e6`) was verified as exact, correctly matching the 300-second grid cadence.
- **Episode Timestamp Matching:** All 141 failure episodes were verified to match raw trace events within 0.1s.
- **Timing Reconciliation:** The discovery of the 7-window discrepancy in tasks `6224307516_2` and `6224307516_3` was verified as mathematically exact. The correct partition is indeed **147 pre-failure** and **149 post-failure** normal windows.
- **Bootstrap Uncertainty:** The entity-cluster and parent-job cluster bootstrap calculations accurately resampled multi-level hierarchical clusters with replacement.

### 6.2 Implementation Defect Identified in Phase 6.7
- **Omission of Causal Protocol Rule Audit:** Phase 6.7 successfully audited the *arithmetic partition* of the 296 windows (147 vs. 149), but it treated the 149 post-failure windows as legitimate evaluable members of the confirmatory cohort. Phase 6.7 failed to cross-reference these 149 windows against the normative text of Protocol Amendment 2.1 line 201 (*"post-failure step -> label -1 (strictly excluded)"*). It did not identify that these 149 windows represented an implementation-level censoring violation.

---

## 7. Final Scientific Verdict and Research Recommendations

### 7.1 Formal Verdict Declaration

```
========================================================================================
FINAL SCIENTIFIC VALIDITY VERDICT:
[C] MATERIAL PROTOCOL VIOLATION FOUND
========================================================================================
```

**Justification:**
1. The confirmatory evaluation dataset contained **149 post-failure recovery windows** that violated the explicit protocol specification requiring post-failure steps to be labeled UNKNOWN (-1) and strictly excluded.
2. An additional **7 pre-failure windows** possessed input sequence histories that crossed an earlier failure boundary.
3. These violations resulted from a concrete implementation omission in `TargetLabelGenerator.derive_label_for_task`, which failed to check for prior failure events when evaluating windows without future failures.
4. While this violation represents a material protocol inconsistency in the dataset construction, our diagnostic counterfactual analysis conclusively demonstrates that it **did not alter the scientific conclusion of the research project**. All four models fail both operational gates by large margins across all censoring interpretations.

### 7.2 Required Revisions for the Research Paper
1. **Full Transparent Disclosure:** The research paper must explicitly report that the confirmatory manifest contained 149 post-failure recovery windows due to an implementation omission in the labeling generator.
2. **Present Counterfactual Sensitivity:** The paper should include the diagnostic counterfactual table (Scenario 0 vs. Scenario 1 vs. Scenario 2), demonstrating that the high false-alarm rates ($17.4\% - 34.6\%$) and operational gate failures are fundamental properties of the model representations, not artifacts of post-failure data contamination.
3. **Retire Confirmatory Cohort:** The Phase 6 confirmatory holdout is permanently unblinded, audited, and retired. It must never be used for threshold tuning or hyperparameter adjustment.
4. **Codebase Patch Required for Future Studies:** Prior to any future study (e.g., Phase 7), [`src/labeling/target_generator.py`](file:///c:/Users/hp/Desktop/Research%20project/src/labeling/target_generator.py) must be updated to explicitly check for past failures and assign `FailureLabel.UNKNOWN` (-1) to any observation following a failure event.

---

### Reaffirmation of Operational Verdict
The official operational conclusion of the research project remains:
$$\mathbf{NO\_OPERATIONAL\_WINNER}$$
No model passed the operational gates. No winning model is declared.
