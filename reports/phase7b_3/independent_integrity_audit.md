# Phase 7B.3 — Independent Results, Preprocessing, and Metric Integrity Audit

**Audit Date:** October 28, 2026  
**Auditor:** Independent Scientific Audit Agent (DeepMind Antigravity)  
**Target Evaluation Report:** [reports/phase7b/phase7b_retraining_and_evaluation_report.md](file:///c:/Users/hp/Desktop/Research%20project/reports/phase7b/phase7b_retraining_and_evaluation_report.md)  
**Target Results Directory:** [results/phase7b/](file:///c:/Users/hp/Desktop/Research%20project/results/phase7b/)  
**Primary Configuration:** [results/phase7b_1/phase7b_evaluation_protocol_v2.json](file:///c:/Users/hp/Desktop/Research%20project/results/phase7b_1/phase7b_evaluation_protocol_v2.json)  
**Target Candidate Split:** [results/phase7a_4/candidate_a_job_grouped_split.csv](file:///c:/Users/hp/Desktop/Research%20project/results/phase7a_4/candidate_a_job_grouped_split.csv)  

---

## Executive Summary

This independent forensic audit was executed under strict read-only restrictions following the completed Phase 7B retraining and evaluation run. All underlying model outputs, predictions, metadata, logs, checkpoints, and scalers were analyzed without retraining models, modifying existing files, or accessing the quarantined historical Phase 6 confirmatory holdout.

### Core Audit Findings
1. **Artifact Provenance (Audit A) — `VERIFIED`:**
   Execution logs, process lifecycle records, and filesystem timestamps establish that the final files in [results/phase7b/](file:///c:/Users/hp/Desktop/Research%20project/results/phase7b/) were generated exclusively by execution `task-3577` (duration: 58.50s, exit code: 0). All four model checkpoint SHA-256 hashes on disk match the values recorded in [results/phase7b/training_manifest.json](file:///c:/Users/hp/Desktop/Research%20project/results/phase7b/training_manifest.json) bit-for-bit.
2. **Preprocessing Contamination Risk (Audit B) — `ISSUE FOUND`:**
   A serious data contamination flaw was confirmed in the data preprocessing chain. While [scripts/prepare_phase7b_scaler_and_data.py](file:///c:/Users/hp/Desktop/Research%20project/scripts/prepare_phase7b_scaler_and_data.py) inverted the standardization linearly, it recovered already-imputed scalar values ($a_{\text{old}} = 2.500000$ for CPI and early lag steps) rather than `NaN`. These legacy Phase 2 imputation medians were fitted on the Phase 2 training split, which contained **100.0% (12 of 12) of Candidate A validation parent jobs** and **100.0% (11 of 11) of Candidate A test parent jobs** (representing $75.4\%$ of validation entities and $62.7\%$ of test entities). True preprocessing isolation was therefore not maintained.
3. **Metric Calculations & Reproducibility (Audit C) — `VERIFIED WITH REPORTING DISCREPANCY`:**
   All evaluation metrics for the frozen candidate Random Forest ($M^*$) were recomputed from raw test predictions ([results/phase7b/test_predictions.csv](file:///c:/Users/hp/Desktop/Research%20project/results/phase7b/test_predictions.csv)) and verified against [results/phase7b/test_evaluation_results.json](file:///c:/Users/hp/Desktop/Research%20project/results/phase7b/test_evaluation_results.json) to 16 decimal places: Macro-F1 ($0.3965$), Accuracy ($0.8390$), Warning Recall ($6.12\%$), Critical Recall ($20.00\%$), Episode Recall ($38.10\%$, 8 of 21), Window FPR ($8.14\%$, 75 of 921), and Pure-Normal Entity FAR ($44.74\%$, 17 of 38).
   - **Investigation of Parent-Job FAR (66.67% vs [100%, 100%]):** The investigation established that in `test_predictions.csv` and `test_evaluation_results.json`, Random Forest alerted on **3 of 3 pure-normal test parent jobs**, giving a true FAR of **$100.00\%$**, which matches the bootstrap interval $[100.0\%, 100.0\%]$. The figure **$66.67\%$** in Table 5.1 and Table 8.1 of the Phase 7B report is a **reporting transcription error** copied from the adjacent Representation Hybrid row ($2/3$ jobs), rather than a coding error or conditional resampling artifact.
4. **Protocol Fidelity (Audit D) — `VERIFIED`:**
   Candidate selection was performed exclusively using validation results before test metrics were accessed. Random Forest failed both Test Gate 1 and Test Gate 2, and the final verdict of **`NO_OPERATIONAL_WINNER`** follows strictly and unavoidably from the pre-registered Protocol v2.0.

---

## Audit A — Exact Run and Artifact Provenance

### 1. Process Timeline and Execution Trace
The repository's execution logs and background task records were analyzed to reconstruct the exact provenance of the artifacts in [results/phase7b/](file:///c:/Users/hp/Desktop/Research%20project/results/phase7b/):

| Task / Process ID | Launch Time (UTC) | End Time (UTC) | Duration | Status | Role & Behavior |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `task-3491` | 2026-10-28 00:05:15 | 2026-10-28 00:16:08 | ~10m 53s | **INTERRUPTED** | Trained all 4 models (RF in 0.24s, LSTM in 7.13s, Transformer in 14.1s, Hybrid in 0.38s). Halted during slow unvectorized cluster bootstrap. **Zero result files were serialized.** |
| `task-3567` | 2026-10-28 00:24:13 | 2026-10-28 00:25:20 | 1m 07s | **CRASHED** | Vectorized bootstrap ran fast, but crashed on line 896 (`KeyError: 'far_pct'` instead of `entity_far_pct`). **Zero result files were serialized.** |
| `task-3577` | 2026-10-28 00:26:04 | 2026-10-28 00:27:11 | **58.50s** | **COMPLETED (Code 0)** | Clean end-to-end execution. Trained models deterministically (seed 42), qualified candidate on validation, scored test split, ran dual 1,000-iteration bootstrap, and serialized all 6 files in [results/phase7b/](file:///c:/Users/hp/Desktop/Research%20project/results/phase7b/). |

**Finding:** The final files in [results/phase7b/](file:///c:/Users/hp/Desktop/Research%20project/results/phase7b/) were generated **exclusively by execution `task-3577`**.

### 2. Cryptographic Checkpoint and Artifact Hashes
The table below records the SHA-256 hashes of all relevant execution scripts, checkpoints, inputs, and results. Checksum verification confirms that checkpoints on disk match the manifest identically.

| Artifact Category | File Path | File Size | SHA-256 Checksum | Manifest Match |
| :--- | :--- | :---: | :--- | :---: |
| **Execution Script** | [scripts/train_and_evaluate_phase7b.py](file:///c:/Users/hp/Desktop/Research%20project/scripts/train_and_evaluate_phase7b.py) | 39,521 B | `3a9895101d87f24e18e6194fdb42d2b9125650ac545d005fd5686ac5025bc331` | `N/A` |
| **Model Checkpoint** | [models/checkpoints_phase7b/random_forest_phase7b.joblib](file:///c:/Users/hp/Desktop/Research%20project/models/checkpoints_phase7b/random_forest_phase7b.joblib) | 3,800,977 B | `3dcd7704bab47790c7e360902767d41458f7c976b77f55b7d4f96556a67510b3` | **`MATCH`** |
| **Model Checkpoint** | [models/checkpoints_phase7b/best_lstm_checkpoint_phase7b.pt](file:///c:/Users/hp/Desktop/Research%20project/models/checkpoints_phase7b/best_lstm_checkpoint_phase7b.pt) | 255,801 B | `bdb5083b2f9efac08170966fc61c8d62e5278b954156e884870e9c513f51118e` | **`MATCH`** |
| **Model Checkpoint** | [models/checkpoints_phase7b/best_transformer_checkpoint_phase7b.pt](file:///c:/Users/hp/Desktop/Research%20project/models/checkpoints_phase7b/best_transformer_checkpoint_phase7b.pt) | 92,484 B | `54267c7b9daaedf167a9990f11895e6dcb116baafa86304b8e7aa88262b973db` | **`MATCH`** |
| **Model Checkpoint** | [models/checkpoints_phase7b/representation_hybrid_phase7b.joblib](file:///c:/Users/hp/Desktop/Research%20project/models/checkpoints_phase7b/representation_hybrid_phase7b.joblib) | 3,324,897 B | `e3ae8bcbb92a643c0f6adf668f11aa8cc7b5be96def25fad07dd3b7f49df7f84` | **`MATCH`** |
| **Dataset Tensor** | [data/processed/phase7b/X_train.npy](file:///c:/Users/hp/Desktop/Research%20project/data/processed/phase7b/X_train.npy) | 2,032,488 B | `e3572ad41c49622d106da62ebf0360a7e0892095f9cbf1a7114b306b328122a6` | `MATCH` |
| **Dataset Tensor** | [data/processed/phase7b/X_val.npy](file:///c:/Users/hp/Desktop/Research%20project/data/processed/phase7b/X_val.npy) | 407,468 B | `07c2a715a3df8b83e37fe5fa108a7061f1c24388ffc7ea9eb7366f065fcba03a` | `MATCH` |
| **Dataset Tensor** | [data/processed/phase7b/X_test.npy](file:///c:/Users/hp/Desktop/Research%20project/data/processed/phase7b/X_test.npy) | 635,628 B | `f04494c8e7e17cb584f22c1bcbbba7fb1972b22bbbb3ce4ee055a434d284f183` | `MATCH` |
| **Scaler Artifact** | [configs/fitted_scaler_phase7b.json](file:///c:/Users/hp/Desktop/Research%20project/configs/fitted_scaler_phase7b.json) | 3,842 B | `05ffbbff68936ef3eb13aee04505f5cecf1808b8e0586e2467d01d4a89fb36fc` | `MATCH` |
| **Result File** | [results/phase7b/training_manifest.json](file:///c:/Users/hp/Desktop/Research%20project/results/phase7b/training_manifest.json) | 2,905 B | `a25217c4bfd35167dc7f40b54394871f6bdf3d41aa47890abd2f9523572ba3ee` | `N/A` |
| **Result File** | [results/phase7b/validation_metrics.json](file:///c:/Users/hp/Desktop/Research%20project/results/phase7b/validation_metrics.json) | 26,212 B | `c1162a3432650170b112ea005c72ecef3c3f3101d7cba4aba5c412397aab9f4a` | `N/A` |
| **Result File** | [results/phase7b/test_evaluation_results.json](file:///c:/Users/hp/Desktop/Research%20project/results/phase7b/test_evaluation_results.json) | 26,525 B | `484b5ff28ca6ce23d3f0514475b9239dc6031328426ab8fbbf35f11c2214dc8c` | `N/A` |
| **Result File** | [results/phase7b/bootstrap_ci.json](file:///c:/Users/hp/Desktop/Research%20project/results/phase7b/bootstrap_ci.json) | 19,099 B | `42fe753408a5cbfb80008985405b356d07318854d6f8eb953a7bd90efa69e677` | `N/A` |
| **Result File** | [results/phase7b/test_predictions.csv](file:///c:/Users/hp/Desktop/Research%20project/results/phase7b/test_predictions.csv) | 240,102 B | `d344c8eed680d46bc9e311f136a0d8ad61c8f6d3a1596f632de7fd8b13319fc3` | `N/A` |
| **Result File** | [results/phase7b/test_episode_metrics.csv](file:///c:/Users/hp/Desktop/Research%20project/results/phase7b/test_episode_metrics.csv) | 858 B | `6423570bfca9d6cec96fe723f6c081016d69b0ee09e9d3ae770143f82d765850` | `N/A` |

*Note on Hashes:* Cryptographic hashes establish identity, traceability, and bit-level immutability across executions; they do not by themselves establish scientific validity or absence of methodological leakage.

---

## Audit B — Preprocessing Contamination Risk

### 1. Complete Trace of Feature Provenance
The lifecycle of telemetry features from Borg trace records to Phase 7B arrays was traced:
1. **Raw Trace Extraction:** Telemetry in `data/raw/sample_task_usage.csv.gz` and `sample_task_events.csv.gz` contains resource metrics for 5,744 windows across 525 entities.
2. **Feature Generation:** In `src/features/engineer.py`, features were extracted trailing per entity. Missing metrics (such as `cycles_per_instruction`, which has $81.3\%$ missingness, and early lag steps) were left as `np.nan`.
3. **Phase 2 Legacy Fitting:** In `scripts/run_phase2_pipeline.py` (lines 225–233), `LeakageSafeScaler` was fitted strictly on the legacy Phase 2 training split (4,798 windows, 367 entities, 81 parent jobs). It computed:
   - Median imputer: $a_{\text{old}} = 2.500000$ for CPI.
   - Standardizer: $\mu_{\text{old}} = 3.577577$, $\sigma_{\text{old}} = 6.452630$.
4. **Phase 2 Imputation Application:** `scaler.transform()` replaced every `np.nan` with $2.500000$ and standardized to create `X_train.npy`, `X_val.npy`, `X_test.npy`.
5. **Phase 7A Diagnostic Slicing:** Phase 7A diagnostic data sliced these pre-scaled arrays.
6. **Phase 7B Scaler Preparation:** In `scripts/prepare_phase7b_scaler_and_data.py`, scaling was inverted linearly:
   $$x_{\text{raw}} = z_{\text{old}} \cdot (\sigma_{\text{old}} + \epsilon_{\text{old}}) + \mu_{\text{old}}$$
   Because the input array already contained $(2.500000 - \mu_{\text{old}}) / \sigma_{\text{old}}$ in missing positions, linear inversion recovered the scalar $2.500000$, **NOT `NaN`**.
7. **Phase 7B Refitting:** `scaler_phase7b.fit()` was fitted on Candidate A Train ($N=3,278 \times 5 = 16,390$ steps). However, its missing values were already pre-filled with the scalar $2.500000$.

### 2. Parent-Job and Entity Overlap with Legacy Imputation Pool
The critical question is: **Did the legacy Phase 2 training pool include entities or parent jobs that are now in Candidate A Validation or Test?**

An exact set intersection was computed between the Phase 2 training split and Candidate A partitions:

| Candidate A Partition | Total Windows | Total Entities | Entities in Phase 2 Train | Entity Contamination % | Total Parent Jobs | Jobs in Phase 2 Train | Parent-Job Contamination % |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Validation Split** | 657 | 69 | 52 | **75.36%** | 12 | 12 | **100.00%** |
| **Test Split** | 1,025 | 59 | 37 | **62.71%** | 11 | 11 | **100.00%** |

### 3. Forensic Evaluation of Contamination
- **Finding:** The value $a_{\text{old}} = 2.500000$ used to impute missing CPI and lag values was fitted using data from **100% of Candidate A validation parent jobs and 100% of Candidate A test parent jobs**.
- **Assessment:** When `prepare_phase7b_scaler_and_data.py` unscaled the arrays, it did not recover the original missingness; it recovered an array pre-imputed with statistics derived from validation and test parent jobs. Although Candidate A Train subsequently computed its own scaling parameters ($\mu_{\text{new}} = 2.863589$, $\sigma_{\text{new}} = 2.383657$), the underlying numbers being scaled in validation and test were already contaminated by legacy Phase 2 median imputation.
- **Scientific Conclusion:** **Preprocessing isolation cannot be established for the current `data/processed/phase7b/` arrays.**

### 4. Feasibility of Corrective Isolation
- **Raw Data Availability:** Complete raw telemetry files (`sample_task_usage.csv.gz` and `sample_task_events.csv.gz`) exist in `data/raw/`.
- **Holdout Isolation:** Candidate A development windows ($N=4,960$) are completely isolated from the quarantined Phase 6 confirmatory holdout ($N=784$).
- **Mandated Corrective Procedure:**
  1. Re-run `src/features/engineer.py` on raw telemetry for development tasks, leaving unobserved CPI and early lags strictly as `np.nan`.
  2. Partition the unscaled feature matrix by Candidate A assignments.
  3. Fit `LeakageSafeScaler` strictly on Candidate A Training data, calculating medians and means omitting `np.nan`.
  4. Transform Candidate A Train, Val, and Test using these training-only medians and scaling parameters.

---

## Audit C — Independent Metric Recomputation & Investigation

### 1. Recomputation of Primary Metrics on Frozen Candidate ($M^*$)
Using preserved test predictions ([results/phase7b/test_predictions.csv](file:///c:/Users/hp/Desktop/Research%20project/results/phase7b/test_predictions.csv)) and ground truth labels, all metrics for the frozen Random Forest were independently recomputed:

| Metric | Target / Rule | Independently Recomputed | Saved JSON ([test_evaluation_results.json](file:///c:/Users/hp/Desktop/Research%20project/results/phase7b/test_evaluation_results.json)) | Absolute Discrepancy | Audit Status |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Macro-F1** | Unweighted mean of 3 classes | **0.396451** | **0.396451** | $0.00 \times 10^0$ | **`VERIFIED`** |
| **Accuracy** | Fraction correct ($N=1,025$) | **0.839024** | **0.839024** | $0.00 \times 10^0$ | **`VERIFIED`** |
| **Warning Recall** | Class 1 recall ($N=49$) | **0.061224** (3 / 49) | **0.061224** | $0.00 \times 10^0$ | **`VERIFIED`** |
| **Critical Recall** | Class 2 recall ($N=55$) | **0.200000** (11 / 55) | **0.200000** | $0.00 \times 10^0$ | **`VERIFIED`** |
| **Episode Recall %** | $\ge 1$ alert in $[0, 30]$ min TTF ($N=21$) | **38.0952%** (8 / 21) | **38.0952%** | $0.00 \times 10^0$ | **`VERIFIED`** |
| **Normal Window FPR %** | False alarms on Class 0 ($N=921$) | **8.1433%** (75 / 921) | **8.1433%** | $0.00 \times 10^0$ | **`VERIFIED`** |
| **Pure-Normal Entity FAR %** | False alarm tasks ($N=38$) | **44.7368%** (17 / 38) | **44.7368%** | $0.00 \times 10^0$ | **`VERIFIED`** |
| **Pure-Normal Parent-Job FAR %** | False alarm jobs ($N=3$) | **100.0000%** (3 / 3) | **100.0000%** | $0.00 \times 10^0$ | **`VERIFIED`** |

### 2. Failure Episode Invariants & Lead Time Verification
- **Total Failure Episodes:** 21 distinct task failure episodes.
- **Pre-Failure Time Window:** Evaluated strictly over windows where $0.0 < \text{TTF} \le 1,800.0\text{s}$ (30 minutes).
  - Minimum TTF: $143.8\text{s}$
  - Maximum TTF: $1,797.5\text{s}$
- **Lead Time Statistics for $M^*$:**
  - Mean lead time: $1,187.9\text{s}$ ($19.8$ minutes).
  - Median lead time: $1,391.6\text{s}$ ($23.2$ minutes).
- **Post-Failure Exclusion:** Zero post-failure windows were included ($0$ windows with $\text{TTF} \le 0$).
- **No Duplicate Credit:** Each episode was credited at most once upon its first true-positive alert.

### 3. Investigation of the Parent-Job FAR Discrepancy (66.67% vs [100%, 100%])
The user requested an investigation into why Table 5.1 and Table 8.1 of the Phase 7B report state that Pure-Normal Parent-Job FAR is **$66.67\%$**, while its 95% cluster-bootstrap interval is **$[100.0\%, 100.0\%]$**.

#### Intermediate Counts & Forensic Trace
In `results/phase7b/test_predictions.csv`, there are exactly 3 pure-normal parent jobs in the Test set (jobs having zero failure episodes and zero warning/critical true labels):
1. **Job `5633011295`** ($20$ windows): Random Forest triggered $2$ false alerts ($1$ Warning, $1$ Critical). Alerted = **True**.
2. **Job `6127642779`** ($660$ windows): Random Forest triggered $21$ false alerts ($10$ Warning, $11$ Critical). Alerted = **True**.
3. **Job `6251639598`** ($40$ windows): Random Forest triggered $11$ false alerts ($10$ Warning, $1$ Critical). Alerted = **True**.

#### Root Cause Determination
- **Calculated Metric:** Number of pure-normal jobs with alerts = $3$. Total pure-normal jobs = $3$.  
  $$\text{Parent-Job FAR}_{\text{RF}} = \frac{3}{3} = \mathbf{100.00\%}$$
- **Recorded JSON Metric:** In `results/phase7b/test_evaluation_results.json` (line 152), the value is recorded as:
  ```json
  "parent_job_far": {
    "false_alarm_jobs": 3,
    "denominator_jobs": 3,
    "far_pct": 100.0,
    "status": "SECONDARY_DIAGNOSTIC_METRIC"
  }
  ```
- **Bootstrap Behavior:** Because all 3 pure-normal jobs triggered alerts, any resample containing at least one pure-normal job will yield $k / k = 100.0\%$ false alarm rate. Thus, 100% of valid bootstrap samples yielded an FAR of $100.0\%$, giving $95\% \text{ CI} = [100.0\%, 100.0\%]$.
- **Origin of 66.67%:** In the same test set, the **Representation Hybrid** model alerted on Job `6127642779` ($46$ alerts) and Job `6251639598` ($16$ alerts), but generated $0$ alerts on Job `5633011295`. Thus, Representation Hybrid had an FAR of $2/3 = \mathbf{66.67\%}$.
- **Conclusion:** The appearance of $66.67\%$ in Table 5.1 and Table 8.1 of the report is definitively a **markdown report transcription error** where the author accidentally copied the figure from the Representation Hybrid row into the Random Forest row. There is **no error** in the underlying Python scripts, the JSON deliverables, or the bootstrap resampling distributions.

### 4. Model Selection Isolation Before Test Evaluation
Code inspection of [scripts/train_and_evaluate_phase7b.py](file:///c:/Users/hp/Desktop/Research%20project/scripts/train_and_evaluate_phase7b.py) (lines 659–753) and execution log analysis of `task-3577.log` (lines 45–60) confirm:
- Validation metrics for all four models were computed and evaluated against Gates 1 and 2.
- Since no model passed both gates, Selection Rule 2 selected Random Forest based strictly on highest **Validation Macro-F1** ($0.3527$).
- Candidate $M^*$ was permanently locked into `validation_results["selected_candidate"]` and serialized before test scoring was invoked.
- **Evidence:** Verified by code structure, log sequence, and serialized timestamps.

---

## Audit D — Protocol Consistency

### 1. Protocol Verification against `phase7b_evaluation_protocol_v2.json`
The evaluation execution was checked against the pre-registered Protocol v2.0:

| Protocol Requirement | Implementation in Script | Verification Outcome |
| :--- | :--- | :---: |
| **Validation-Only Candidate Selection** | Lines 738–753 select candidate $M^*$ strictly on validation metrics | **`VERIFIED`** |
| **Test Gate 1 (Recall)** | Requires Episode Recall $\ge 70.0\%$ and Critical Recall $\ge 50.0\%$ | **`VERIFIED`** |
| **Test Gate 2 (False Alarms)** | Requires Pure-Normal Entity FAR $\le 10.0\%$ and Normal Window FPR $\le 5.0\%$ | **`VERIFIED`** |
| **Mandatory Operational Verdict** | If $M^*$ fails Gate 1 or Gate 2, output `NO_OPERATIONAL_WINNER` | **`VERIFIED`** |
| **Descriptive Containment** | Non-selected models firewalled from retroactive promotion | **`VERIFIED`** |

### 2. Confirmatory Gate Decision Audit
Candidate $M^*$ (Random Forest) was scored against the pre-registered operational gates:
1. **Gate 1 (Failure Recall):**
   - Episode Recall: $38.10\%$ (Required: $\ge 70.0\%$) $\to$ **FAIL**
   - Critical Recall: $20.00\%$ (Required: $\ge 50.0\%$) $\to$ **FAIL**
2. **Gate 2 (False-Alarm Containment):**
   - Entity FAR: $44.74\%$ (Required: $\le 10.0\%$) $\to$ **FAIL**
   - Window FPR: $8.14\%$ (Required: $\le 5.0\%$) $\to$ **FAIL**
3. **Official Final Operational Verdict:**
   # **`NO_OPERATIONAL_WINNER`**

The verdict `NO_OPERATIONAL_WINNER` follows strictly, objectively, and unavoidably from the experimental data and the frozen protocol.

---

## Deliverables Summary Table

| Finding / Audit Check | Component | Status | Evidence / Source |
| :--- | :---: | :---: | :--- |
| **Run Provenance** | Audit A | **`VERIFIED`** | `task-3577` log, execution duration 58.50s, exit code 0 |
| **Checkpoint Integrity** | Audit A | **`VERIFIED`** | Checkpoints match manifest SHA-256 hashes 100% bit-for-bit |
| **Preprocessing Isolation** | Audit B | **`ISSUE FOUND`** | Pre-imputed CPI ($2.50$) fitted on 100% of test/val parent jobs |
| **Metric Reproducibility** | Audit C | **`VERIFIED`** | All test metrics reproduced to 16 decimal places from raw predictions |
| **Job FAR 66.67% Discrepancy**| Audit C | **`ISSUE FOUND`** | Reporting transcription error in Table 5.1/8.1; JSON and code have 100.0% |
| **Candidate Freezing** | Audit C | **`VERIFIED`** | $M^*$ selected on validation data prior to test evaluation |
| **Protocol Compliance** | Audit D | **`VERIFIED`** | Adheres to Protocol v2.0; gates failed; `NO_OPERATIONAL_WINNER` confirmed |

---

## Final Forensic Verdict

Based on the forensic evidence gathered across Audits A, B, C, and D:
1. All test predictions, confusion matrices, numerators, denominators, and bootstrap confidence intervals are mathematically verified and reproducible from the saved raw prediction files.
2. The pre-registered evaluation protocol was strictly followed, candidate selection was properly isolated to validation data, and the final verdict of `NO_OPERATIONAL_WINNER` is methodologically sound.
3. However, the input tensors in `data/processed/phase7b/` carry forward legacy Phase 2 median imputation values derived from training entities and parent jobs that now comprise 100% of the validation and test parent jobs.

Therefore, this audit concludes with the official verdict:

# **`METRICS REPRODUCED — PREPROCESSING CONTAMINATION RISK`**

*(With the explicit finding that the Parent-Job FAR of 66.67% reported in Table 5.1 of the Phase 7B report is a documented markdown transcription error, whereas the true calculated, JSON-recorded, and bootstrap-consistent metric is 100.00%.)*

---
*Independent audit complete. All audit deliverables are permanently serialized in [results/phase7b_3/independent_integrity_audit.json](file:///c:/Users/hp/Desktop/Research%20project/results/phase7b_3/independent_integrity_audit.json) and this report.*
