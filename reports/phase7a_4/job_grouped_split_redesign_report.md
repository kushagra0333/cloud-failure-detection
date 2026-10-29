# Phase 7A.4 — Job-Grouped Split Redesign and Leakage Verification Report

**Author:** Antigravity (AI Research Assistant)  
**Date:** October 27, 2026  
**Local Time:** 23:25:00+05:30  
**Phase:** Phase 7A.4 (Development Split Redesign & Group Leakage Verification)  
**Status:** Complete — Read-Only Design & Verification  
**Official Verdict:** **`SPLIT DESIGN VERIFIED`**

---

## 1. Executive Summary

This report establishes a defensible, mathematically verified development split design for the research project:
> **Temporal LSTM/Transformer Framework for Early Failure Prediction in Cloud Computing**

### The Core Problem Solved
In prior phases (Phases 1 through 6.8), the dataset was partitioned using **entity-level splitting** (`entity_id` = `f"{job_id}_{task_index}"`). As revealed during the Phase 7A.3 audit, while distinct tasks were kept in separate splits, tasks belonging to the **same parent job** were scattered across training, validation, and test:
- **60.0% of validation jobs** and **72.7% of test jobs** were present in the training set.
- **78.7% of validation windows** and **89.2% of test windows** came from parent jobs already seen during training.
- Because tasks within a parent job share identical binaries, user priorities, resource allocations, and cluster runtime conditions, models could memorize job-level behavioral signatures rather than learning true early failure precursors.

### Key Deliverables of Phase 7A.4
1. **Source Data Verification:** Confirmed the integrity and SHA-256 hashes of all 10 diagnostic data files (4,960 development windows).
2. **Authoritative Parent-Job Mapping:** Verified from Google Cluster Trace schema definitions and raw event files that `entity_id.split('_')[0]` is a 100% bijective mapping to the cluster `job_id` across all 412 development entities.
3. **Temporal Concurrency Finding:** Proved empirically that 91.9% of parent jobs (68 of 74) were submitted at $t=0.0$s and 77.9% of all windows belong to jobs that continue running until the very end of the trace ($t=10,800$s). A pure chronological split is mathematically incompatible with parent-job isolation and leads to catastrophic data starvation (~78% discarded).
4. **Candidate Split Designs Evaluated:**
   - **Candidate A (Stratified Group Split by Parent Job):** Groups all tasks of each parent job into exactly one split with 0% overlap, while stratifying by failure severity to maintain balanced Critical (69.5% Tr, 14.4% Va, 16.1% Te) and Warning (71.6% Tr, 14.1% Va, 14.4% Te) representations.
   - **Candidate B (Temporal Group Split Constrained by Jobs):** Attempts temporal ordering, but suffers severe class starvation in validation (only 3 critical windows, 0.88%) due to job concurrency.
5. **Recommendation:** Recommend **Candidate A (Seed 48)** as the scientifically defensible split configuration for Phase 7B retraining.
6. **Strict Safeguards Observed:** Zero models were trained, tuned, or evaluated. The spent Phase 6 confirmatory holdout ($N=5,315$) remained completely untouched and was not used in any split decisions.

---

## 2. Source Data Integrity and Verification

### 2.1 File-Level Verification and SHA-256 Hashes
All analyses were conducted starting from the clean, repaired Phase 7A diagnostic data generated in [results/phase7a/](file:///c:/Users/hp/Desktop/Research%20project/results/phase7a/) and stored in [data/processed/phase7a_diagnostic/](file:///c:/Users/hp/Desktop/Research%20project/data/processed/phase7a_diagnostic/).

| File Name | Path | Size (Bytes) | SHA-256 Checksum | Verification Role |
| :--- | :--- | :--- | :--- | :--- |
| `train_metadata_repaired.csv` | [train_metadata_repaired.csv](file:///c:/Users/hp/Desktop/Research%20project/data/processed/phase7a_diagnostic/train_metadata_repaired.csv) | 115,217 | `e012ab201aa4525b78dae855d124d2785a70c3061cd3c1006d1ba83aa1430897` | Dev Pool (Original Train) |
| `val_metadata_repaired.csv` | [val_metadata_repaired.csv](file:///c:/Users/hp/Desktop/Research%20project/data/processed/phase7a_diagnostic/val_metadata_repaired.csv) | 27,795 | `833ed347f715de8026b0ac6b978fe5afbf1d543a23d7641981343b83095e288d` | Dev Pool (Original Val) |
| `test_metadata_repaired.csv` | [test_metadata_repaired.csv](file:///c:/Users/hp/Desktop/Research%20project/data/processed/phase7a_diagnostic/test_metadata_repaired.csv) | 27,345 | `abeb4ca0cf8642ba1d0a16fe449ce3f1e2949776a0cdd55a5fafca5e5331aa09` | Dev Pool (Original Test) |
| `X_train_repaired.npy` | [X_train_repaired.npy](file:///c:/Users/hp/Desktop/Research%20project/data/processed/phase7a_diagnostic/X_train_repaired.npy) | 2,105,028 | `ef3bc8e7ed6dc1cefd2e03c3ce581ceb1eed37814bfc27bf95590eed8ee293cf` | Features (3395, 5, 31) |
| `X_val_repaired.npy` | [X_val_repaired.npy](file:///c:/Users/hp/Desktop/Research%20project/data/processed/phase7a_diagnostic/X_val_repaired.npy) | 482,488 | `2774b50a4e3ff840b3a0c9b71c34e2224049add9284f94a9512c944f82564140` | Features (778, 5, 31) |
| `X_test_repaired.npy` | [X_test_repaired.npy](file:///c:/Users/hp/Desktop/Research%20project/data/processed/phase7a_diagnostic/X_test_repaired.npy) | 488,068 | `b1497a7034d24bd2ad5ac0db2ec08dbb389ff75dbc06edc2160824d988467aa4` | Features (787, 5, 31) |
| `y_train_repaired.npy` | [y_train_repaired.npy](file:///c:/Users/hp/Desktop/Research%20project/data/processed/phase7a_diagnostic/y_train_repaired.npy) | 27,288 | `2f09005b6a56f3e8ea237984aca474c75b91dc61fd5cfb1891beba92c3155972` | Labels (3395,) |
| `y_val_repaired.npy` | [y_val_repaired.npy](file:///c:/Users/hp/Desktop/Research%20project/data/processed/phase7a_diagnostic/y_val_repaired.npy) | 6,352 | `df32bba10edc2d086955239c1625f77de50f4f213f03ed1fa6e64348e0bd20db` | Labels (778,) |
| `y_test_repaired.npy` | [y_test_repaired.npy](file:///c:/Users/hp/Desktop/Research%20project/data/processed/phase7a_diagnostic/y_test_repaired.npy) | 6,424 | `7695bb1ed4a383f4c2c000adcad859fbd6c5098dcd6aab6bfdcbd4c007cc3986` | Labels (787,) |
| `confirmatory_metadata_repaired.csv` | [confirmatory_metadata_repaired.csv](file:///c:/Users/hp/Desktop/Research%20project/data/processed/phase7a_diagnostic/confirmatory_metadata_repaired.csv) | 150,883 | `964e26a5a4ee2aef4f8cf37fd1fb516f7bd823efa4b935d181347fc078e75625` | **FROZEN HOLDOUT — UNTOUCHED** |
| `X_confirmatory_repaired.npy` | [X_confirmatory_repaired.npy](file:///c:/Users/hp/Desktop/Research%20project/data/processed/phase7a_diagnostic/X_confirmatory_repaired.npy) | 3,295,428 | `cc4c9ea01430df879390d0dbd2c52e374b442e6f77a2568d85d10e75328df3b7` | **FROZEN HOLDOUT — UNTOUCHED** |
| `y_confirmatory_repaired.npy` | [y_confirmatory_repaired.npy](file:///c:/Users/hp/Desktop/Research%20project/data/processed/phase7a_diagnostic/y_confirmatory_repaired.npy) | 42,648 | `73942367133bb6748a38b36f759b1ea8a396d5cbeedef1d711674f90a12d3fde` | **FROZEN HOLDOUT — UNTOUCHED** |

> [!IMPORTANT]
> **Confirmatory Holdout Preservation:** The spent Phase 6 confirmatory holdout ($N=5,315$ windows, 335 entities) was loaded solely to verify its cryptographic hash and confirm that it is **strictly isolated**. It was **not** included in the development pool, was **not** used to tune split parameters, and was **not** consulted in candidate split design.

---

## 3. Authoritative Parent-Job Identifier Verification

### 3.1 Verification Against Schema and Raw Traces
A critical requirement of this audit is not to assume that splitting `entity_id` on an underscore (`_`) is valid without empirical proof from the raw data schema.

1. **Schema Definition:** In Google Cluster Trace 2011 (v2.1; Reiss et al., 2011; Wilkes, 2020), every task belongs to a job. The schema defines:
   - `job_id`: 64-bit integer identifier assigned by Borg upon job submission.
   - `task_index`: Non-negative integer denoting the specific container replica within that job.
2. **Construction in Pipeline:** In [src/ingestion/schema.py](file:///c:/Users/hp/Desktop/Research%20project/src/ingestion/schema.py#L98-L120) and `scripts/verify_cohort_phase6_1.py:L97`, `entity_id` was constructed as:
   $$\text{entity\_id} = f"\{\text{job\_id}\}\_\{\text{task\_index}\}"$$
3. **Forensic Verification Against Raw Files:**
   We inspected the raw trace files in `data/raw/` (`sample_task_events.csv.gz`, `sample_task_events_part1.csv.gz`, `sample_task_events_part2.csv.gz`). Across all 412 development entities:
   - Neither `job_id` nor `task_index` ever contains an underscore; both are strictly numeric integers.
   - For **100% of the 412 entities (412/412)**, the prefix before the underscore strictly matches the integer `job_id` in the raw event table.
   - Splitting on `_` at index 0 is **mathematically exact and bijective**.

```python
# Verified invariant across 100% of 412 development entities:
parts = entity_id.split("_")
assert len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit()
assert parts[0] == str(raw_event_row["job_id"])
assert parts[1] == str(raw_event_row["task_index"])
```

### 3.2 Aggregated Development Pool Counts

Combining `train_metadata_repaired.csv` (3,395 windows), `val_metadata_repaired.csv` (778 windows), and `test_metadata_repaired.csv` (787 windows) yields the complete development pool:

- **Total Development Windows ($N$):** 4,960
- **Total Unique Entities:** 412
- **Total Unique Parent Jobs:** 74
- **Window-Level Class Distribution:**
  - **Normal ($y=0$):** 4,278 windows (86.25%)
  - **Warning ($y=1$):** 341 windows (6.88%)
  - **Critical ($y=2$):** 341 windows (6.88%)
  - Total failure-associated windows ($y \in \{1, 2\}$): 682 windows (13.75%)

### 3.3 Parent-Job Level Structure and Failure Distribution

Grouping by the 74 parent jobs reveals significant structural patterns:

- **Failing Jobs ($N=50$):** Jobs that contain at least one task experiencing failure, generating minority class windows:
  - Jobs generating both Critical and Warning windows: **42 jobs**
  - Jobs generating Critical windows only: **2 jobs** (Jobs `5633011441`, `6238841354`)
  - Jobs generating Warning windows only: **6 jobs** (Jobs `5633010278`, `609260421`, `6238843128`, `6238843875`, `6252065564`, `698645235`)
- **Pure-Normal Jobs ($N=24$):** Jobs where zero tasks ever fail, generating 100% Normal windows (2,304 windows total).
- **Multi-Task Jobs:**
  - **36 of 74 jobs (48.6%)** have multiple tasks present within the development pool (e.g., Job `6251995937` has 75 tasks; Job `6251414911` has 59 tasks; Job `6251277720` has 47 tasks).
  - **61 of 74 jobs (82.4%)** have multiple tasks in the raw cluster event tables.
- **Multi-Failure Jobs:**
  - **44 of 74 jobs (59.5%)** have multiple raw failure events.
  - **39 of 74 jobs (52.7%)** have multiple distinct tasks that fail.
  - **21 of 74 jobs (28.4%)** have exactly 1 raw failure event.
  - **9 of 74 jobs (12.2%)** have zero failure events in the raw event log.

---

## 4. Temporal Concurrency and Embargo Analysis

A central question in machine learning on time-series is whether an **embargo** (a buffer zone purged around split boundaries) is required, and whether a **chronological split** is feasible.

### 4.1 What an Embargo Protects Against
In time-series forecasting, an embargo prevents information leakage across a temporal boundary $T_{\text{split}}$:
1. **Feature Lookback Overlap:** When predicting at time $t$, features look back over a history of length $L \times \Delta t$. If validation begins at $T_{\text{split}}$, its earliest prediction points look backward into the training interval $[T_{\text{split}} - L\Delta t, T_{\text{split}}]$.
2. **Label Lookahead Overlap:** A training point at $T_{\text{split}} - \epsilon$ evaluates whether a failure occurs within the warning horizon $W_{\text{warn}}$. This looks forward into the validation interval $[T_{\text{split}}, T_{\text{split}} + W_{\text{warn}}]$.
3. **Embargo Duration:** To prevent both leakages on a continuous time series, an embargo must purge all observations within $W_{\text{warn}}$ before $T_{\text{split}}$ and potentially $L\Delta t$ after $T_{\text{split}}$:
   $$\text{Embargo Duration} = W_{\text{warn}} = 1,800\text{ seconds (30 minutes)}$$
   $$\text{Feature History} = L \times \Delta t = 5 \times 300\text{s} = 1,500\text{ seconds (25 minutes)}$$

### 4.2 Empirical Discovery: High Job Concurrency Prevents Chronological Splitting
We investigated whether the 74 development parent jobs follow a sequential global timeline. The empirical findings demonstrate that they do not:

```
Distribution of Job Minimum Prediction Timestamps (t_min):
  t_min = 5100.0s : 68 jobs (91.9%)
  t_min > 5100.0s :  6 jobs ( 8.1%)

Distribution of Job Maximum Prediction Timestamps (t_max):
  t_max = 10800.0s: 42 jobs (56.8% of jobs, 77.9% of windows, 77.1% of Critical windows)
  t_max < 10800.0s: 32 jobs (43.2% of jobs, 22.1% of windows, 22.9% of Critical windows)

Raw Event Submission Times:
  job_submit_time == 0.0s: 68 jobs (91.9%)
```

#### Why Chronological Splitting Fails on This Dataset:
1. **Trace Window Duration vs. Prediction Horizon:**
   The entire development telemetry span is from $5,100$s to $10,800$s — a duration of only **5,700 seconds (95 minutes, or 1.58 hours)**.
   Under a nominal 70% / 15% / 15% chronological split:
   - The 15% validation interval spans only $0.15 \times 5,700\text{s} = \mathbf{855\text{ seconds (14.25 minutes)}}$.
   - However, the required 30-minute warning embargo is **1,800 seconds**.
   - An 1,800s embargo is **longer than the entire validation interval**. Applying the embargo would erase 100% of the validation split.
2. **Boundary Crossing of Concurrent Jobs:**
   Because 68 of 74 jobs start at 5,100s and 42 jobs run until 10,800s, any chronological boundary cuts right through the active execution of **50 to 64 jobs (over 80% of all windows)**.
   - If jobs spanning the boundary are split across boundaries, **parent-job leakage is re-introduced**.
   - If jobs spanning the boundary are discarded to preserve 0% job overlap, **78% of all windows and 77% of failures are lost**, leaving the dataset statistically starved.

### 4.3 Why Parent-Job Group Separation Provides Complete Isolation
In cloud cluster operations, each parent job runs in isolated containers (Borg allocs / cgroups). 
As verified in [src/features/engineer.py:L48](file:///c:/Users/hp/Desktop/Research%20project/src/features/engineer.py#L48):
- Features are computed strictly using a `df.groupby("entity_id")` transform. No feature for Job B ever inspects data from Job A.
- Labels are computed strictly from that entity's own future failure events.
- Therefore, when Parent Job A is assigned to Train and Parent Job B is assigned to Validation/Test, **they share zero features, zero labels, and zero state**, even though they execute concurrently on the cluster.
- **Conclusion:** Group separation by parent job provides rigorous statistical and informational isolation without discarding 78% of the data.

---

## 5. Candidate Split Designs Comparison

We formulated and evaluated two candidate split designs on the 4,960 development windows:

### 5.1 Candidate A: Stratified Group Split by Parent Job (Recommended)
- **Methodology:** Parent jobs ($N=74$) are partitioned into three strata: high-critical-density failing jobs, low-critical-density failing jobs, and pure-normal jobs. Jobs are allocated deterministically (Seed 48) to achieve ~70% Train, ~15% Val, and ~15% Test proportions across both total windows and minority failure classes.
- **Guarantee:** 100% of tasks belonging to any parent job remain strictly in one split.

### 5.2 Candidate B: Temporal Group Split Constrained by Parent Jobs
- **Methodology:** Parent jobs are sorted by their mean prediction timestamp and assigned sequentially to Train, Val, and Test to approximate chronological progression while keeping parent jobs grouped.
- **Limitation:** Due to concurrent job lifespans, all three splits still span from 5,100s to 10,800s, while large job sizes cause severe class starvation in validation.

---

### 5.3 Comparative Metrics Table

| Metric | Candidate A: Train | Candidate A: Val | Candidate A: Test | Candidate B: Train | Candidate B: Val | Candidate B: Test |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Parent Jobs Count** | **51** (68.9%) | **12** (16.2%) | **11** (14.9%) | **62** (83.8%) | **3** (4.1%) | **9** (12.2%) |
| **Unique Entities Count** | **284** (68.9%) | **69** (16.7%) | **59** (14.3%) | **233** (56.6%) | **9** (2.2%) | **170** (41.3%) |
| **Total Windows Count** | **3,278** (66.1%) | **657** (13.2%) | **1,025** (20.7%) | **3,261** (65.7%) | **180** (3.6%) | **1,519** (30.6%) |
| **Normal Windows ($y=0$)** | **2,797** (65.4%) | **560** (13.1%) | **921** (21.5%) | **2,819** (65.9%) | **173** (4.0%) | **1,286** (30.1%) |
| **Warning Windows ($y=1$)** | **244** (71.6%) | **48** (14.1%) | **49** (14.4%) | **219** (64.2%) | **4** (1.2%) | **118** (34.6%) |
| **Critical Windows ($y=2$)** | **237** (69.5%) | **49** (14.4%) | **55** (16.1%) | **223** (65.4%) | **3** (0.9%) | **115** (33.7%) |
| **Prediction Timestamp Span** | [5100s, 10800s] | [5100s, 10800s] | [5100s, 10800s] | [5100s, 10800s] | [5100s, 10800s] | [5100s, 10800s] |
| **Train $\cap$ Val Job Overlap** | **0 (0.0%)** | — | — | **0 (0.0%)** | — | — |
| **Train $\cap$ Test Job Overlap**| **0 (0.0%)** | — | — | **0 (0.0%)** | — | — |
| **Val $\cap$ Test Job Overlap**  | — | **0 (0.0%)** | — | — | **0 (0.0%)** | — |

---

## 6. Pairwise Parent-Job Overlap Verification

For both Candidate A and Candidate B, pairwise set intersection tests were executed across all parent job identifiers:

```python
# Pairwise Overlap Audit Code Executed:
train_jobs = set(candidate_splits["train"])
val_jobs = set(candidate_splits["val"])
test_jobs = set(candidate_splits["test"])

assert len(train_jobs & val_jobs) == 0  # 0 overlapping parent jobs
assert len(train_jobs & test_jobs) == 0 # 0 overlapping parent jobs
assert len(val_jobs & test_jobs) == 0   # 0 overlapping parent jobs
assert len(train_jobs | val_jobs | test_jobs) == 74 # All 74 jobs accounted for
```

### Contrast with Historical Split
In the historical entity-split configuration:
- Train $\cap$ Val Parent Jobs: **12 overlapping jobs (60.0% of Val jobs leaked)**
- Train $\cap$ Test Parent Jobs: **16 overlapping jobs (72.7% of Test jobs leaked)**
- Train $\cap$ Val Windows Leaked: **612 windows (78.7% of Val leaked)**
- Train $\cap$ Test Windows Leaked: **702 windows (89.2% of Test leaked)**

In Candidate A:
- Train $\cap$ Val Parent Jobs: **0 overlapping jobs (0.0% leakage)**
- Train $\cap$ Test Parent Jobs: **0 overlapping jobs (0.0% leakage)**
- Val $\cap$ Test Parent Jobs: **0 overlapping jobs (0.0% leakage)**
- Train $\cap$ Val Windows Leaked: **0 windows (0.0% leakage)**
- Train $\cap$ Test Windows Leaked: **0 windows (0.0% leakage)**

---

## 7. Trade-offs and Scientific Recommendation

### 7.1 Why Candidate B is Rejected
1. **Severe Class Starvation in Validation:** Candidate B assigns only **3 Critical windows** (0.88%) and **4 Warning windows** (1.17%) to the validation split. Validating early warning models on 3 failure windows produces statistically meaningless validation metrics with extreme variance.
2. **Failure of Temporal Separation:** Because parent jobs run concurrently from 5,100s to 10,800s, Candidate B fails to achieve chronological isolation; all three splits span the exact same global time window.

### 7.2 Why Candidate A is Recommended
1. **Zero Information Leakage:** Guarantees 0% parent-job overlap across train, val, and test.
2. **Robust Statistical Power:** Provides **49 Critical and 48 Warning windows** in Validation, and **55 Critical and 49 Warning windows** in Test.
3. **Sound Operational Generalization:** In cloud computing (Borg/Kubernetes), the core operational challenge is **workload generalization** — predicting failures on newly submitted parent jobs that the model has never encountered before. Candidate A directly models and evaluates this exact operational capability.
4. **Transparent Trade-Off:** Because individual jobs vary in size (some jobs have 75 tasks while others have 1 task), total window counts are slightly non-uniform (66.1% Tr, 13.2% Va, 20.7% Te), but minority failure classes are preserved at ~70% / 14% / 16%.

---

## 8. Reproducible Assignment Files and Manifest

All candidate split assignments, verification metadata, and cryptographic manifests have been generated and stored in [results/phase7a_4/](file:///c:/Users/hp/Desktop/Research%20project/results/phase7a_4/):

| File Name | Path | Size (Bytes) | SHA-256 Checksum | Description |
| :--- | :--- | :--- | :--- | :--- |
| `candidate_a_job_grouped_split.csv` | [candidate_a_job_grouped_split.csv](file:///c:/Users/hp/Desktop/Research%20project/results/phase7a_4/candidate_a_job_grouped_split.csv) | 256,213 | `f84bc4fe54f8e0583e2ee52f49b2fbe8e6fcafb4f020c656985a1acc31bb4946` | Window-by-window assignments for Candidate A |
| `candidate_b_temporal_group_split.csv` | [candidate_b_temporal_group_split.csv](file:///c:/Users/hp/Desktop/Research%20project/results/phase7a_4/candidate_b_temporal_group_split.csv) | 256,673 | `a9f31e43b164d8dd12089d0ad6e610e403a5e867615c1c6fcc022cf6fdf46804` | Window-by-window assignments for Candidate B |
| `split_design_verification.json` | [split_design_verification.json](file:///c:/Users/hp/Desktop/Research%20project/results/phase7a_4/split_design_verification.json) | 10,819 | `b8ff4770f648f331eafda0fd59a0a11a66d0a87b6f25e84f856964ef895e3382` | Machine-readable verification data and per-split job ID lists |
| `candidate_splits_manifest.json` | [candidate_splits_manifest.json](file:///c:/Users/hp/Desktop/Research%20project/results/phase7a_4/candidate_splits_manifest.json) | 473 | `dbb7f7291a442bdeb99e08375e2fa7b12d58bf5cbfa8f4e24ebca7cfa4326fbe` | Cryptographic manifest of generated split files |

---

## 9. Official Verdict and Next Steps

### Official Phase 7A.4 Verdict
# **`SPLIT DESIGN VERIFIED`**

### Rationale
- The repaired Phase 7A diagnostic data has been verified with exact SHA-256 hashes.
- Authoritative parent-job identifiers have been verified against raw cluster traces with 100% accuracy.
- Candidate A achieves complete group isolation with **zero parent-job overlap** between Train, Validation, and Test, while maintaining sufficient Critical and Warning examples in all three splits.
- Candidate B has been documented with full empirical evidence showing why chronological splitting fails due to 91.9% job concurrency.
- The spent Phase 6 confirmatory holdout remains strictly untouched and unblinded.
- No models were trained, tuned, or evaluated.

### Required Actions Before Phase 7B Retraining
In accordance with user safeguards, **no split configuration has been activated in production**, and **no models have been trained**.

To proceed to **Phase 7B (Clean Model Retraining and Evaluation)**, the following step requires explicit user authorization:
1. Confirm acceptance of **Candidate A (Stratified Job-Grouped Split, Seed 48)** as the active project split configuration.
2. Authorize updating the training split loaders to read Candidate A assignments for model retraining.
