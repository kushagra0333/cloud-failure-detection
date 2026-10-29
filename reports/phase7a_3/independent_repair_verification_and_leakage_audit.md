# Phase 7A.3 — Independent Repair Verification and Split-Leakage Audit

**Project:** Temporal LSTM/Transformer Framework for Early Failure Prediction in Cloud Computing  
**Phase:** Phase 7A.3 — Independent Repair Verification & Split-Leakage Audit  
**Author:** Independent Machine Learning Scientific Auditor & Pair Programmer  
**Date:** October 27, 2026  
**Governing Policy:** **Policy A: Terminal Entity Lifecycle** (Protocol Amendment 2.1 line 201)  
**Historical Evaluation Status:** Spent, Unblinded, and Permanently Preserved (`NO_OPERATIONAL_WINNER` locked)  
**Primary Output Artifact:** [`results/phase7a/phase7a_3_audit_summary.json`](file:///c:/Users/hp/Desktop/Research%20project/results/phase7a/phase7a_3_audit_summary.json)  
**Final Audit Verdict:** **`READY FOR SPLIT REDESIGN`**

---

## 1. Executive Summary & Verdict

This independent audit evaluated two crucial questions prior to authorizing any model retraining:
1. **Did the Phase 7A.2 labeling repair truly succeed at the data and code level?**
2. **Can the repaired diagnostic datasets support an unbiased evaluation under the existing train/val/test splits?**

### The Key Findings:
1. **Labeling Repair 100% Verified:**
   - The production code in [`src/labeling/target_generator.py`](file:///c:/Users/hp/Desktop/Research%20project/src/labeling/target_generator.py) contains the verified patch (SHA-256: `22e8c14a9f9730857c901707558adc13449486442808364ab928bfabafa2fcfe`).
   - All 18 regression and integration tests pass directly against the production code.
   - All 177 repository unit and integration tests pass with zero errors.
   - Across all 10,275 retained windows in `train`, `val`, `test`, and `confirmatory`, **exactly ZERO windows have an endpoint or any step in their 5-step trailing history occurring at or after that task's first abnormal failure event**.
   - Feature tensors $X$, label vectors $y$, and metadata files are perfectly aligned row-for-row, finite, and contain **zero UNKNOWN (-1) entries**.
2. **Row Removal Fully Reconciled (2,052 Exclusions):**
   - The 2,052 exclusions across the repository were matched row-by-row using unique `(entity_id, prediction_timestamp)` keys.
   - Every single excluded row was classified by its exact causal reason.
   - Zero rows were counted twice; zero rows were lost or unaccounted for.
3. **Severe Parent-Job Leakage Discovered in Existing Splits:**
   - In the existing development splits, tasks were split by `entity_id` rather than parent `job_id`.
   - **60.0% of parent jobs in Validation (15 of 25 jobs)** and **72.7% of parent jobs in Test (16 of 22 jobs)** were already present in the Training split!
   - At the window level: **78.7% of validation windows (612 of 778)** and **89.2% of test windows (702 of 787)** belong to parent jobs that the models train on!
   - In stark contrast, **86.7% of parent jobs in the Confirmatory holdout (98 of 113 jobs)** were completely unseen in training!

### Audit Verdict:
$$\mathbf{READY\ FOR\ SPLIT\ REDESIGN}$$

The labeling repair itself is completely verified and sound. However, **retraining models on the current splits must NOT be authorized**, because doing so would reproduce the exact same evaluation trap that doomed earlier phases: the models would memorize parent-job workload signatures, appear to perform well on validation, and collapse on unseen cloud workloads. A group-level split redesign (`job_grouped`) must be executed first.

---

## 2. Section A: Independent Verification of Repaired Artifacts

We independently loaded every source file, diagnostic metadata CSV, feature tensor `.npy`, and label vector `.npy`.

### 2.1 Production Code Integrity
- **Production File:** [`src/labeling/target_generator.py`](file:///c:/Users/hp/Desktop/Research%20project/src/labeling/target_generator.py)
  - Post-patch SHA-256: `22e8c14a9f9730857c901707558adc13449486442808364ab928bfabafa2fcfe`
  - Pre-patch Backup: [`results/phase7a/target_generator_original.py.bak`](file:///c:/Users/hp/Desktop/Research%20project/results/phase7a/target_generator_original.py.bak) (SHA-256: `d04f074581eec9ee64b02178133be23fceb9943263372d134ebf1c952f39c648`)
- Verified that lines 55–68 contain the strict prior failure check (`task_events["time_sec"] <= t`) and non-finite timestamp guards.
- Verified that [`tests/test_phase7a_labeling_repair.py`](file:///c:/Users/hp/Desktop/Research%20project/tests/test_phase7a_labeling_repair.py) imports directly from `src.labeling.target_generator.TargetLabelGenerator`.

### 2.2 Tensor, Array, and Metadata Alignment Verification

Every repaired diagnostic dataset in [`data/processed/phase7a_diagnostic/`](file:///c:/Users/hp/Desktop/Research%20project/data/processed/phase7a_diagnostic/) was verified:

| Dataset Split | Metadata Rows | Tensor Shape $X$ | Label Vector Shape $y$ | $X$ Finite? | $y == \text{meta}[\text{"label"}]$? | UNKNOWNs in $y$? | Retained Label Distribution | Unique Entities |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Train Repaired** | 3,395 | $[3395, 5, 31]$ | $[3395]$ | **YES** | **YES** (100%) | **0 (0.00%)** | Normal: 2,959 <br>Warning: 215 <br>Critical: 221 | 344 |
| **Val Repaired** | 778 | $[778, 5, 31]$ | $[778]$ | **YES** | **YES** (100%) | **0 (0.00%)** | Normal: 643 <br>Warning: 72 <br>Critical: 63 | 73 |
| **Test Repaired** | 787 | $[787, 5, 31]$ | $[787]$ | **YES** | **YES** (100%) | **0 (0.00%)** | Normal: 676 <br>Warning: 54 <br>Critical: 57 | 74 |
| **Confirmatory Repaired** | 5,315 | $[5315, 5, 31]$ | $[5315]$ | **YES** | **YES** (100%) | **0 (0.00%)** | Normal: 4,655 <br>Warning: 286 <br>Critical: 374 | 335 |
| **TOTALS** | **10,275** | — | — | **YES** | **YES** (100%) | **0 (0.00%)** | **Normal: 8,933 <br>Warning: 627 <br>Critical: 715** | **826** |

### 2.3 Verification Against Raw Lifecycle Failure Events
We extracted all raw Google Cluster Trace lifecycle events (`sample_task_events*.csv.gz`, filtering for event type 3 = `FAIL`), matched every retained window against the entity's earliest abnormal failure timestamp ($T_{\text{fail}}^{(1)}$), and evaluated two strict temporal conditions:
1. **Endpoint Violation:** Does the prediction timestamp satisfy $t_{\text{pred}} \ge T_{\text{fail}}^{(1)}$?
2. **History Violation:** Does any part of the 5-step trailing history span $[t_{\text{pred}} - 1200\text{s}, t_{\text{pred}}]$ overlap with or occur after $T_{\text{fail}}^{(1)}$ ($t_{\text{start}} \le T_{\text{fail}}^{(1)} \le t_{\text{pred}}$ or $t_{\text{start}} \ge T_{\text{fail}}^{(1)}$)?

**Results:**
- Post-first-fail endpoints across all 10,275 windows: **EXACTLY 0 (0.00%)**.
- Cross-first-fail or post-first-fail histories across all 10,275 windows: **EXACTLY 0 (0.00%)**.
- **Conclusion:** 100% of all retained sequence windows in the diagnostic datasets are strictly pre-failure observations.

---

## 3. Section B: Row-by-Row Removal Reconciliation

The Phase 7A.2 report noted that 2,052 total rows were removed across the four splits, including 196 rows from the confirmatory holdout. We conducted a row-by-row matching audit between the original and repaired datasets using the unique compound key `entity_id@prediction_timestamp`.

### 3.1 Reconciliation Breakdown by Split and Removal Reason

| Split | Original Windows | Retained Windows | Removed Windows | Cat 1: Post-All-Failures Normal ($y=0$) | Cat 2: Between-Failures Cross-History Normal ($y=0$) | Cat 3: Between-Failures Clean Normal ($y=0$) | Cat 4: Between-Failures Warning ($y=1$) | Cat 5: Between-Failures Critical ($y=2$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Train** | 4,798 | 3,395 | **1,403** | 1,130 | 40 | 59 | 32 *(16 cross, 16 clean)* | 142 *(108 cross, 34 clean)* |
| **Val** | 1,015 | 778 | **237** | 197 | 0 | 0 | 0 | 40 *(40 cross, 0 clean)* |
| **Test** | 1,003 | 787 | **216** | 145 | 8 | 35 | 14 *(1 cross, 13 clean)* | 14 *(3 cross, 11 clean)* |
| **Confirmatory** | 5,511 | 5,315 | **196** | 149 | 7 | 0 | 14 *(14 cross, 0 clean)* | 26 *(9 cross, 17 clean)* |
| **TOTALS** | **12,327** | **10,275** | **2,052** | **1,621** | **55** | **94** | **60** | **222** |

### 3.2 Resolving the Confirmatory Holdout 196 vs. 149 vs. 7 Count
Earlier Phase 6.8 audits focused specifically on the **296 failure-cohort Normal windows** and found:
- 149 post-all-failure Normal windows (Category 1).
- 7 cross-history Normal windows (Category 2).
- $149 + 7 = \mathbf{156\text{ contaminated Normal windows}}$.

Why did Phase 7A.2 report **196 removed confirmatory windows**?
Because under **Policy A (Terminal Entity Lifecycle)**, any observation at or after the entity's first failure ($t \ge T_{\text{fail}}^{(1)}$) is permanently censored. 
In the confirmatory holdout, tasks with repeated failures (such as `6250804554_0`, `6250804554_1`, `6250804554_2`, `6224307516_2`, `6224307516_3`, `6250345153_388`, and `6252225776_5`) generated windows for their *second* or *third* failures that occurred after failure 1.
These consisted of:
- **14 Warning windows** ($y=1$)
- **26 Critical windows** ($y=2$)
$156\text{ Normal} + 14\text{ Warning} + 26\text{ Critical} = \mathbf{196\text{ total removed windows}}$.
Every single one of these 40 Warning/Critical windows occurred post-failure-1. Under Policy A, they represent post-crash retry execution and are censored as `UNKNOWN` (-1).

### 3.3 Relabeled UNKNOWN vs. Excluded Windows
It is vital to distinguish between:
1. **Telemetry Rows Relabeled UNKNOWN:** In `TargetLabelGenerator`, an individual 300-second bin in the featured dataframe is assigned `label = -1`.
2. **Windows Excluded by `WindowGenerator`:** `WindowGenerator` skips any observation where `target_label == FailureLabel.UNKNOWN`. In addition, `WindowGenerator` requires 4 contiguous trailing observations to build a 5-step history. If an entity crashed and produced only 1 or 2 lingering telemetry bins before terminating, those bins were relabeled `UNKNOWN`, but they could never form a 5-step window anyway because they lacked 5 contiguous steps!
3. **Double Counting Verification:** Unique compound keys `entity_id@prediction_timestamp` confirmed that every removed window belongs to exactly one split, has exactly one label, and was counted exactly once. Zero keys overlap.

---

## 4. Section C: Parent-Job Leakage Audit

A major question is whether the current train, validation, and test splits can provide an unbiased estimate of model generalization once models are retrained.

### 4.1 Authoritative Parent-Job Identifier Verification
- In the Google Cluster Trace schema ([`src/ingestion/schema.py`](file:///c:/Users/hp/Desktop/Research%20project/src/ingestion/schema.py#L9-L43)), a cloud workload is organized hierarchically: `User -> Job -> Task -> Attempt`.
- Each task belongs to a parent job identified by `job_id` (an integer).
- In [`scripts/run_phase2_pipeline.py`](file:///c:/Users/hp/Desktop/Research%20project/scripts/run_phase2_pipeline.py#L92) and [`scripts/extract_confirmatory_data_l5.py`](file:///c:/Users/hp/Desktop/Research%20project/scripts/extract_confirmatory_data_l5.py#L92), the entity identifier is constructed as:
  $$\text{entity\_id} = \text{str}(\text{job\_id}) + \text{"\_"} + \text{str}(\text{task\_index})$$
- We verified every single entity ID across all four splits: 100% of entity IDs follow this exact two-part format where the prefix before the underscore is strictly numeric `job_id`. The authoritative parent job ID is verified as `job_id`.

### 4.2 Parent-Job Overlap Across Splits

We audited the parent-job distribution across all splits:

| Split Comparison | Jobs in Split 1 | Jobs in Split 2 | Common Parent Jobs | % of Split 2 Jobs Leaked from Split 1 | Window-Level Leakage (% of Split 2 Windows from Leaked Jobs) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Train vs. Validation** | 60 | 25 | **15** | **60.0%** (15 / 25) | **78.7%** (612 of 778 validation windows) |
| **Train vs. Test** | 60 | 22 | **16** | **72.7%** (16 / 22) | **89.2%** (702 of 787 test windows) |
| **Val vs. Test** | 25 | 22 | **11** | **50.0%** (11 / 22) | — |
| **Train vs. Confirmatory** | 60 | 113 | **15** | **13.3%** (15 / 113) | **86.7% of Confirmatory Jobs UNSEEN in Train** |

```
DEVELOPMENT SPLITS (Severe Leakage)            CONFIRMATORY HOLDOUT (Unseen Workloads)
┌──────────────────────────────────────┐       ┌──────────────────────────────────────┐
│ Train: 60 Parent Jobs                │       │ Confirmatory Holdout: 113 Jobs       │
│                                      │       │                                      │
│ Validation: 25 Jobs                  │       │ 98 Jobs UNSEEN in Train (86.7%)      │
│   ├── 15 LEAKED from Train (60.0%)   │       │ Only 15 jobs shared with Train       │
│   └── (78.7% of Val Windows!)        │       │                                      │
│                                      │       │ Models failed because they           │
│ Test: 22 Jobs                        │       │ memorized batch workload signatures! │
│   ├── 16 LEAKED from Train (72.7%)   │       │                                      │
│   └── (89.2% of Test Windows!)       │       │                                      │
└──────────────────────────────────────┘       └──────────────────────────────────────┘
```

### 4.3 Generalization Implications
1. **Why Earlier Development Results Were Misleading:**
   In Phase 3 and Phase 4, models achieved high validation scores because the validation set was not measuring true generalization. Nearly **80% of validation windows came from the exact same batch jobs that the models were trained on**. The models simply memorized the resource consumption patterns (CPI, memory access intensity, CPU utilization) of those 15 specific batch jobs.
2. **Why Models Collapsed on the Confirmatory Holdout:**
   In the confirmatory holdout, **86.7% of the jobs were completely new and unseen**. The models encountered workloads with different baseline characteristics and triggered false alarms on 17.4% to 33.9% of pure-normal windows.
3. **Scientific Conclusion:**
   The repaired diagnostic data **CANNOT** support an unbiased generalization estimate with the current splits. Retraining on the existing splits would perpetuate the same methodological flaw.

### 4.4 Defensible Group-Splitting Design Recommendation
Before any model retraining, the development cohort must be re-partitioned using a **Stratified Job-Grouped Splitter with Temporal Embargo**:
1. **Cluster Key:** Group strictly by `parent_job_id`. Every task belonging to the same parent job must be assigned to the same split.
2. **Stratification:** Stratify parent jobs by failure status (failing jobs vs. pure-normal jobs) to ensure balanced class distributions across splits.
3. **Temporal Ordering & Embargo:** Sort parent jobs chronologically by earliest submit timestamp. Train on early jobs, validate on middle jobs, and test on late jobs, with an embargo gap between splits to eliminate edge-boundary spillover.
4. **Guaranteed Independence:** Set intersections between train, validation, and test parent jobs must be strictly $\emptyset$ (0% overlap).

---

## 5. Section D & E: Artifact Integrity & Prohibited Actions

### 5.1 Confirmation of Historical Artifact Immutability
All historical Phase 6 artifacts were verified as untouched:
- `results/phase6/confirmatory_predictions_l5.csv`: `d1d798314218ba56...` (**UNTOUCHED**)
- `results/phase6/confirmatory_metrics_l5.json`: `121828c4bfcd3871...` (**UNTOUCHED**)
- `results/phase6/protocol_l5_amendment_2_1.json`: `f354cd9f02343e83...` (**UNTOUCHED**)
- `data/processed/confirmatory_l5/X_confirmatory_l5.npy`: `f5401db048685057...` (**UNTOUCHED**)
- `data/processed/l5/X_train.npy`: `ffe168bb058f49c6...` (**UNTOUCHED**)

### 5.2 Strict Safeguards Maintained
- Zero models were trained, tuned, or evaluated.
- No decision thresholds were altered.
- The spent Phase 6 confirmatory holdout was not rescaled, modified, or used for model selection.
- The official operational verdict remains permanently locked:
  $$\mathbf{NO\_OPERATIONAL\_WINNER}$$

---

## 6. Remaining Limitations & Next Steps

1. **The Holdout is Spent:** The Phase 6 confirmatory holdout is permanently unblinded. The repaired confirmatory diagnostic dataset (`confirmatory_metadata_repaired.csv`, $N=5,315$) is strictly for diagnostic sensitivity analysis and can never serve as a new blinded confirmatory experiment.
2. **Next Step Required:** Execute a clean, job-grouped split redesign on the development pool (`train_clean`, `val_clean`, `test_clean`), verify zero parent-job overlap, and only then proceed to model retraining and threshold calibration.

---

## 7. Final Verdict

$$\mathbf{READY\ FOR\ SPLIT\ REDESIGN}$$

The labeling repair is 100% verified, mathematically exact, and regression-tested. The data is now clean of post-failure contamination. However, model retraining should only begin after the development splits are redesigned to eliminate parent-job leakage. I am stopped and awaiting your authorization.
