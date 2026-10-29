# Phase 7A.5 — Candidate A Split Audit Before Approval

**Author:** Antigravity (AI Research Assistant)  
**Date:** October 27, 2026  
**Local Time:** 23:45:00+05:30  
**Phase:** Phase 7A.5 (Pre-Approval Audit of Candidate A Development Split)  
**Status:** Complete — Independent Read-Only Audit  
**Official Verdict:** **`CANDIDATE A VERIFIED — READY FOR APPROVAL`**

---

## 1. Executive Summary

This independent audit verifies the design, reproducibility, mathematical properties, and scientific claims of **Candidate A (Stratified Group Split by Parent Job, Seed 48)** before any project activation or model retraining in Phase 7B.

### What Was Verified
1. **100% Bit-for-Bit Reproducibility:** Re-running the split generation procedure in an isolated diagnostic directory reproduced [candidate_a_job_grouped_split.csv](file:///c:/Users/hp/Desktop/Research%20project/results/phase7a_4/candidate_a_job_grouped_split.csv) with zero discrepancies (identical SHA-256: `f84bc4fe54f8e0583e2ee52f49b2fbe8e6fcafb4f020c656985a1acc31bb4946`).
2. **Authoritative Raw Job Isolation:** Zero parent-job overlap was verified not only on derived string IDs, but directly on the **authoritative 64-bit integer `job_id`** from raw trace files (`sample_task_events*.csv.gz`).
3. **No Window or Entity Loss:** Exactly 4,960 development windows across 412 entities and 74 parent jobs are accounted for. No windows were dropped or duplicated.
4. **Stratification & Seed Selection Disclosed:** The search across 2,000 seeds is fully disclosed. Stratification operated strictly at the parent-job level, with zero exposure to or leakage from the spent confirmatory holdout.
5. **Feature-History Duration Resolved:** The difference between the previously reported $[t - 1200\text{s}, t]$ interval and the 1,500s history statement is reconciled using code evidence from [src/labeling/window_generator.py](file:///c:/Users/hp/Desktop/Research%20project/src/labeling/window_generator.py) and [src/features/engineer.py](file:///c:/Users/hp/Desktop/Research%20project/src/features/engineer.py).
6. **Generalization Claims Clarified:** Candidate A establishes **workload generalization to unseen parent jobs** within the cluster trace. It **does not** establish future-time generalization or cross-cluster generalization.
7. **Confirmatory Holdout Reality:** The historical Phase 6 confirmatory holdout shares **20 parent jobs (1,727 windows, 32.5%)** with the development pool, confirming that it was never parent-job isolated and remains a spent diagnostic artifact.

---

## 2. Independent Reproduction and Assignment Verification

### 2.1 Isolated Diagnostic Re-Run
We re-executed the Candidate A assignment logic with Seed 48 in an isolated diagnostic directory:
- **Reproduced Output:** [results/phase7a_5/diagnostic_reproduction/reproduced_candidate_a.csv](file:///c:/Users/hp/Desktop/Research%20project/results/phase7a_5/diagnostic_reproduction/reproduced_candidate_a.csv)
- **Saved Output:** [results/phase7a_4/candidate_a_job_grouped_split.csv](file:///c:/Users/hp/Desktop/Research%20project/results/phase7a_4/candidate_a_job_grouped_split.csv)

```
Saved Assignment SHA-256:      f84bc4fe54f8e0583e2ee52f49b2fbe8e6fcafb4f020c656985a1acc31bb4946
Reproduced Assignment SHA-256: f84bc4fe54f8e0583e2ee52f49b2fbe8e6fcafb4f020c656985a1acc31bb4946
Exact Bit-for-Bit Match:       TRUE
Row-by-Row Value Comparison:   100% IDENTICAL across all 4,960 rows and 7 columns
Discrepancies Detected:        0
```

### 2.2 Accounting and Structural Invariants
Direct analysis of the saved assignments confirmed:
- **Every Parent Job in Exactly One Split:** For all 74 parent jobs, the number of distinct splits assigned is exactly 1 (`(job_split_map == 1).all() == True`).
- **Every Window Appears Exactly Once:** 4,960 rows, with unique `dev_window_idx` spanning `[0, 4959]`.
- **No Silently Dropped or Duplicated Entities:** All 412 development entities are present and accounted for.

### 2.3 Authoritative Raw Integer `job_id` Overlap Check
To ensure that string formatting did not mask job collisions, we mapped every entity back to the underlying Google Cluster Trace raw event logs (`data/raw/sample_task_events*.csv.gz`) and extracted the raw 64-bit integer `job_id`:

- **Train Raw Integer `job_id`s ($N=51$):** `{3998246322, 4861489847, ..., 6252065564}`
- **Val Raw Integer `job_id`s ($N=12$):** `{4665617026, 6127571078, ..., 698645235}`
- **Test Raw Integer `job_id`s ($N=11$):** `{1650628364, 5633010199, ..., 6251639598}`

```python
# Raw integer set intersections:
assert set(train_raw_job_ids) & set(val_raw_job_ids) == set()  # 0 overlap
assert set(train_raw_job_ids) & set(test_raw_job_ids) == set() # 0 overlap
assert set(val_raw_job_ids) & set(test_raw_job_ids) == set()   # 0 overlap
```
**Finding:** Pairwise parent-job overlap is **strictly 0.0%** at the raw cluster database level.

### 2.4 Recalculated Counts Directly from Saved Assignments

| Split | Parent Jobs | Entities | Windows | Windows % | Normal ($y=0$) | Normal % | Warning ($y=1$) | Warning % | Critical ($y=2$) | Critical % |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **TRAIN** | 51 | 284 | 3,278 | 66.09% | 2,797 | 65.38% | 244 | 71.55% | 237 | 69.50% |
| **VAL** | 12 | 69 | 657 | 13.25% | 560 | 13.09% | 48 | 14.08% | 49 | 14.37% |
| **TEST** | 11 | 59 | 1,025 | 20.67% | 921 | 21.53% | 49 | 14.37% | 55 | 16.13% |
| **TOTAL** | **74** | **412** | **4,960** | **100.0%** | **4,278** | **100.0%** | **341** | **100.0%** | **341** | **100.0%** |

All recalculated counts match the Phase 7A.4 report with **zero discrepancy**.

---

## 3. Stratification Procedure Verification and Disclosure

### 3.1 Parent-Job Representation and Counting
1. **Cluster Aggregation:** Parent jobs were represented as aggregate clusters containing total windows and class counts:
   $$\text{n\_crit}_j = \sum_{w \in \text{Job}_j} \mathbb{I}(y_w = 2), \quad \text{n\_warn}_j = \sum_{w \in \text{Job}_j} \mathbb{I}(y_w = 1)$$
2. **Strata Definition:** Failing jobs ($N=50$) and pure-normal jobs ($N=24$) were grouped into four strata:
   - High-Critical Density (`n_crit >= 10`): 8 jobs
   - Low-Critical Density (`0 < n_crit < 10`): 36 jobs
   - Warning-Only (`n_crit == 0` and `n_warn > 0`): 6 jobs
   - Pure-Normal (`n_crit == 0` and `n_warn == 0`): 24 jobs
3. **Parent-Job Level Partitioning:** Stratification occurred **strictly at the parent-job level**. The permutation and slicing operated on lists of parent job identifiers. When a job ID was assigned to a split, all of its tasks and windows were assigned to that split. No individual windows were ever split across boundaries.
4. **No Confirmatory Holdout Influence:** The spent confirmatory holdout ($N=5,315$) was completely excluded from the development pool and had zero influence on strata definition or job assignment.

### 3.2 Full Disclosure: Seed Search and Selection Bias Risk
- **Search Disclosure:** Seed 48 was selected following an exploratory search across 2,000 random seeds (seeds 0 through 1,999) in `scratch/search_balanced_candidate_a.py`.
- **Search Objective:** The search sought to minimize deviation from the target 70% Train / 15% Val / 15% Test proportions across total windows, Critical windows, and Warning windows, subject to minimum safety thresholds (at least 35 Critical and 35 Warning windows in both Val and Test).
- **Selection Bias Risk:**
  - In statistical methodology, searching seeds to optimize validation class balance is a form of **hyper-split selection**.
  - A validation set found through a seed search has a more favorable class balance than an arbitrary random draw.
  - **Crucial Distinction:** Because **zero models were trained or evaluated** during the seed search, the search did not optimize model accuracy or exploit feature patterns. It only balanced marginal class counts.
  - Transparently disclosing this search ensures the research record remains honest and defensible.

---

## 4. Resolution of Feature-History Duration

A technical discrepancy was identified between two statements in historical documentation:
1. Historical reports stated that sequence windows span $[t - 1200\text{s}, t]$.
2. The Phase 7A.4 report referred to a 1,500-second feature history ($5 \times 300\text{s}$).

We inspected [src/labeling/window_generator.py](file:///c:/Users/hp/Desktop/Research%20project/src/labeling/window_generator.py) and [src/features/engineer.py](file:///c:/Users/hp/Desktop/Research%20project/src/features/engineer.py) to resolve this mathematically:

### 4.1 The Sequence Window ($L=5$ Time Steps)
In [src/labeling/window_generator.py:L48-L49](file:///c:/Users/hp/Desktop/Research%20project/src/labeling/window_generator.py#L48-L49):
```python
total_duration = timestamps[-1] - timestamps[0]
nominal_duration = (self.sequence_length - 1) * self.expected_interval_sec
```
For sequence length $L = 5$ sampled at $\Delta t = 300\text{s}$:
- There are **5 discrete timestamps** in the sequence tensor: $[t - 1200\text{s}, t - 900\text{s}, t - 600\text{s}, t - 300\text{s}, t]$.
- The difference between the first timestamp and the last timestamp is:
  $$\Delta T_{\text{endpoints}} = t - (t - 1200\text{s}) = \mathbf{1,200\text{ seconds (20 minutes)}}$$
- However, each discrete timestamp is the `end_time_sec` of a 300-second aggregation bin. The first bin at $t - 1200\text{s}$ aggregates telemetry over $[t - 1500\text{s}, t - 1200\text{s}]$.
- Therefore, the underlying telemetry observations span:
  $$\text{Telemetry Bin Coverage} = 5 \times 300\text{s} = \mathbf{1,500\text{ seconds (25 minutes)}}$$
- **Reconciliation:** Both statements are correct from different reference frames:
  - $[t - 1200\text{s}, t]$ describes the **span of discrete sequence timestamps**.
  - $1,500\text{s}$ describes the **total telemetry duration integrated across the 5 bins**.

### 4.2 Extended Rolling Feature Lookback
Furthermore, in [src/features/engineer.py:L70-L102](file:///c:/Users/hp/Desktop/Research%20project/src/features/engineer.py#L70-L102), `FeatureEngineer` calculates:
- Lag features up to lag 6 (`shift(6) = 6 * 300s = 1,800s`).
- Rolling statistics (mean, std, range, slope) over windows of $w=6$ (`6 * 300s = 1,800s`).

Because the earliest sequence step $t_0$ is at $t - 1200\text{s}$, its lag-6 feature requires historical telemetry extending back:
$$t - 1,200\text{s} - 1,800\text{s} = \mathbf{t - 3,000\text{ seconds (50 minutes before prediction time } t)}$$

### 4.3 Total Informational Footprint
Combining feature lookback and target lookahead:
- Deepest telemetry lookback: $3,000\text{ seconds (50 minutes backward)}$
- Furthest warning lookahead: $W_{\text{warn}} = 1,800\text{ seconds (30 minutes forward)}$
- Total window footprint:
  $$\text{Total Window Footprint} = 3,000\text{s} + 1,800\text{s} = \mathbf{4,800\text{ seconds (80 minutes)}}$$
- Because the entire trace observation period is only **5,700 seconds (95 minutes)**, a single window's reach encompasses **84.2% of the trace**. This mathematically confirms why chronological splitting with embargoes cannot function on this dataset without eliminating the validation split.

---

## 5. Scope of Generalization and Holdout Reality

### 5.1 What Candidate A Measures
- **Workload / Job-Level Generalization:**
  Candidate A evaluates whether models trained on 51 parent jobs can predict early failures on **12 unseen validation parent jobs** and **11 unseen test parent jobs** executing within the same cluster environment.
  This directly answers the operational question: *"Can the model detect failures on newly deployed batch pipelines and applications it was not trained on?"*

### 5.2 What Candidate A Does NOT Measure
- **Future-Time Generalization (Temporal Drift):**
  Candidate A **does not** measure generalization across weeks, months, or years. Because 91.9% of parent jobs were submitted at $t=0.0\text{s}$, Train, Val, and Test jobs execute **concurrently in the same 95-minute observation window**.
- **Cross-Cluster / Cross-Environment Generalization:**
  Candidate A **does not** evaluate whether models generalize to different datacenters, architectures, or orchestrators. The dataset comes strictly from one single Google Borg cell in May 2011.

### 5.3 Audit of the Historical Confirmatory Holdout
We conducted a cross-cohort parent-job audit between the development pool ($N=4,960$ windows, 74 jobs) and the historical confirmatory holdout ($N=5,315$ windows, 113 jobs):

```
Development Unique Parent Jobs:               74 jobs
Confirmatory Holdout Unique Parent Jobs:      113 jobs
Shared Parent Jobs:                           20 jobs
Confirmatory Windows from Shared Parent Jobs: 1,727 of 5,315 windows (32.5%)
```

**Key Finding:**
The Phase 6 confirmatory holdout was isolated by `entity_id` (task instance), but **was never isolated by parent job**. Nearly one-third (32.5%) of confirmatory holdout windows come from parent jobs that appear in the development data.
Combined with the fact that its ground truth labels and model metrics have been unblinded and analyzed across multiple previous audit phases, **the confirmatory holdout is a spent diagnostic resource**. It cannot be used as an independent holdout test set for future model evaluation.

---

## 6. Official Verdict and Recommendation

### Official Verdict
# **`CANDIDATE A VERIFIED — READY FOR APPROVAL`**

### Summary of Justification:
1. **100% Bit-for-Bit Reproducible:** Saved assignments in [results/phase7a_4/candidate_a_job_grouped_split.csv](file:///c:/Users/hp/Desktop/Research%20project/results/phase7a_4/candidate_a_job_grouped_split.csv) are confirmed identical to independent re-runs.
2. **Zero Parent-Job Overlap:** Verified on authoritative raw cluster integer `job_id`s across Train, Validation, and Test.
3. **No Dropped or Duplicated Windows:** Exactly 4,960 windows across 412 entities and 74 jobs.
4. **Balanced Minority Classes:** Train (237 Crit, 244 Warn), Val (49 Crit, 48 Warn), Test (55 Crit, 49 Warn) provide robust statistical power.
5. **Honest Methodological Boundaries:** The seed search, feature-history duration (1,200s endpoints vs 1,500s bins vs 3,000s rolling features), and generalization limits (workload generalization, not temporal drift) are fully documented.
6. **Strict Safeguards Observed:** No models were trained, tuned, or evaluated; historical files remain untouched.

---

## 7. Deliverables Summary

| Artifact Path | Format | Description |
| :--- | :--- | :--- |
| [results/phase7a_5/candidate_a_preapproval_audit.json](file:///c:/Users/hp/Desktop/Research%20project/results/phase7a_5/candidate_a_preapproval_audit.json) | JSON | Machine-readable verification data, raw job IDs, and holdout overlap metrics |
| [reports/phase7a_5/candidate_a_preapproval_audit.md](file:///c:/Users/hp/Desktop/Research%20project/reports/phase7a_5/candidate_a_preapproval_audit.md) | Markdown | Comprehensive pre-approval audit report |
| [results/phase7a_5/diagnostic_reproduction/reproduced_candidate_a.csv](file:///c:/Users/hp/Desktop/Research%20project/results/phase7a_5/diagnostic_reproduction/reproduced_candidate_a.csv) | CSV | Independently reproduced assignment file |

---

## 8. Next Step (Awaiting User Approval)

The audit is complete. To proceed to **Phase 7B (Clean Model Retraining and Evaluation)**, please confirm:
- Do you approve activating **Candidate A** as the project's development split configuration for retraining?
