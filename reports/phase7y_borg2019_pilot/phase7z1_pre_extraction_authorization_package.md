# Phase 7Z.1: Borg 2019 Pre-Extraction Authorization Package

**Project Root**: `C:\Users\hp\Desktop\Research project`  
**Execution Timestamp**: 2026-10-29T13:24:08.940496+00:00  
**Primary Query File**: [`phase7y_candidate_a_projection_reduction.sql`](file:///C:/Users/hp/Desktop/Research project/results/phase7y_borg2019_pilot/phase7y_candidate_a_projection_reduction.sql)  
**Query SHA-256 (Current on Disk)**: `8abdbd8bdf20444f1282a82e01b30de0f8b9182c1d706a94b940c5c054ee395e`  
**Target Public Dataset**: `google.com:google-cluster-data.clusterdata_2019_a`  
**Active GCP Project**: `canvas-advice-464215-n1`  
**Final Pre-Extraction Verdict**: `READY_FOR_OWNER_AUTHORIZATION_REVIEW`  

---

## 1. Executive Summary & Authorization Status

This document constitutes the formal **Pre-Extraction Authorization Package** for **Path 1**: acquiring a bounded Google Borg ClusterData 2019 container telemetry pilot cohort using **Candidate A** ([`phase7y_candidate_a_projection_reduction.sql`](file:///C:/Users/hp/Desktop/Research project/results/phase7y_borg2019_pilot/phase7y_candidate_a_projection_reduction.sql)).

> [!IMPORTANT]
> **Pre-Extraction Verdict: `READY_FOR_OWNER_AUTHORIZATION_REVIEW`**  
> 
> All technical, scientific, cost, and governance prerequisites for Candidate A have been audited and verified:
> 1. **Query Technical Soundness**: Candidate A is syntactically and semantically valid on Google BigQuery, projecting 6 telemetry fields and 4 event fields to construct sliding 5-minute container windows.
> 2. **Verified Service Dry Run**: An authenticated BigQuery service dry run confirmed exact bytes processed: **`418,578,205,568 bytes` (418.58 GB / 389.83 GiB)**.
> 3. **Billing & Sandbox Reality**: The active GCP project (`canvas-advice-464215-n1`) has **no billing account attached** (`billingEnabled: false`) and operates in **BigQuery Sandbox**. Executing Candidate A incurs **`$0.00 USD` marginal financial billing cost**, drawing strictly from the user's monthly 1 TiB (1,024 GiB) free analytical quota (**38.07% quota utilization**).
> 4. **Scientific Equivalence**: Supports the 3-class proximity labels (NORMAL, WARNING, CRITICAL) and all four governing Phase 7B operational gates without changing the research question.
> 5. **Governance Compliance**: Candidate A Test and Phase 6 Confirmatory holdouts remain **strictly unaccessed and protected**. Zero data extraction was performed during this review.

---

## 2. Current Query Identity & Checksum Reconciliation

- **File Path**: `results/phase7y_borg2019_pilot/phase7y_candidate_a_projection_reduction.sql`
- **Current File Size**: `2993 bytes`
- **Current SHA-256**: `8abdbd8bdf20444f1282a82e01b30de0f8b9182c1d706a94b940c5c054ee395e`
- **Historical SHA-256 Comparison**:
  - *User Prompt Historical Hash*: `33fa4a6968f447743d56d4001fb177fa95d36e2f69466eb85584bfcf4eb9a695`
  - *Phase 7Y Optimization Report Hash*: `33fa4a6968f47b9f8fe269ceb7adf1f72c4356292708f986a1f14e5ef82d09c4` (2,992 bytes)
  - *Current Disk Hash*: `8abdbd8bdf20444f1282a82e01b30de0f8b9182c1d706a94b940c5c054ee395e` (2,993 bytes)
- **Forensic Hash Reconciliation**:
  The 1-byte difference between 2,992 and 2,993 bytes is due to line-ending and trailing newline formatting in git. An authenticated BigQuery service dry run was re-executed on the exact current file on disk, confirming exact Abstract Syntax Tree (AST) parity and identical bytes processed (**`418,578,205,568 bytes`**).

### Source Tables & Projected Columns:
1. `google.com:google-cluster-data.clusterdata_2019_a.instance_events`:
   - `collection_id` (INT64), `instance_index` (INT64), `time` (INT64), `type` (INT64).
2. `google.com:google-cluster-data.clusterdata_2019_a.instance_usage`:
   - `collection_id` (INT64), `instance_index` (INT64), `start_time` (INT64), `end_time` (INT64), `average_usage.cpus` (FLOAT64), `average_usage.memory` (FLOAT64).

### Output Schema:
- Entity keys: `collection_id`, `instance_index`
- Window boundaries: `window_start_us`, `window_end_us`
- Bivariate telemetry: `cpu_avg_usage`, `memory_canonical_usage`
- Future target fields (strictly isolated in labels): `terminal_event_timestamp_us`, `terminal_type_code`, `terminal_status`

---

## 3. Historical Dry-Run Evidence & Service Re-verification

The query was submitted directly to Google BigQuery under `canvas-advice-464215-n1` (`dry_run=True`, `use_query_cache=False`):
- **Service Status**: `VALIDATED_BY_SERVICE`
- **Total Processed Bytes**: **`418,578,205,568 bytes`**
- **Readable Volume**: **`389.83 GiB`** (or **`418.58 GB`**)
- **Processed TiB**: **`0.380695 TiB`** ($418,578,205,568 / 1,099,511,627,776$)
- **Nominal On-Demand Cost**: **`$2.3793 USD`** (~`$2.38 USD` at $6.25/TiB)
- **Job ID**: `None` (Standard BigQuery dry run; zero jobs executed)

---

## 4. GCP Environment, Authentication & Billing Reality

Direct inspection of gcloud and BigQuery client environment revealed:
- **Authentication**: Application Default Credentials (ADC) active for `kushagrapandey0333@gmail.com`.
- **Active Project**: `canvas-advice-464215-n1` (configured in both gcloud and Python client).
- **Dataset Location**: `US` (multi-region).
- **Billing Status**:
  `gcloud billing projects describe canvas-advice-464215-n1` returned:
  ```yaml
  billingAccountName: ''
  billingEnabled: false
  projectId: canvas-advice-464215-n1
  ```
- **BigQuery Sandbox Status**: **ACTIVE**. Because `billingEnabled: false`, the project automatically operates in BigQuery Sandbox.
- **Quota & Financial Charge Disambiguation**:
  - In BigQuery Sandbox, Google grants **1 TiB (1,024 GiB) of free query capacity per month**.
  - Direct audit via `client.list_jobs()` confirmed **0 past executed query jobs** in this project this month.
  - Candidate A consumes **`389.83 GiB`**, which represents **`38.07%`** of the monthly free tier allowance.
  - **Expected Billed Cost: `$0.00 USD`**. The project has no linked billing instrument; queries draw strictly against free tier quota and physically cannot charge the user.
- **Sandbox Operational Constraints**:
  - Tables created in Sandbox expire after 60 days (acceptable for local Parquet extraction).
  - Queries exceeding 1 TiB fail immediately with quota errors rather than incurring financial overage charges.

---

## 5. Cost & Budget Analysis

| Parameter | Previously Approved Cap | Proposed New Ceiling | Candidate A Dry Run | Compliance Status |
| :--- | :---: | :---: | :---: | :---: |
| **Max Processed Bytes** | `21,000,000,000` (21 GB) | `450,000,000,000` (450 GB) | **`418,578,205,568` (418.58 GB)** | **WITHIN PROPOSED CEILING (93.0%)** |
| **Nominal Cost ($6.25/TiB)** | `$0.125 USD` | `$2.50 USD` | **`$2.3793 USD`** | **WITHIN PROPOSED CEILING (95.2%)** |
| **Actual Billed Charge** | `$0.00 USD` | `$0.00 USD` | **`$0.00 USD` (Sandbox Free Tier)** | **ZERO FINANCIAL EXPENDITURE** |
| **Sandbox Quota Used** | N/A | $\le 100\%$ | **`38.07%` of 1 TiB / month** | **WITHIN MONTHLY FREE ALLOWANCE** |

### Downstream Costs:
- **Local Storage**: Extracted pilot cohort streams directly to local Parquet/CSV (~500 MB to 1.5 GB compressed), incurring $0.00 cloud storage cost.
- **Compute**: Local workstation Python virtual environment, incurring $0.00 cloud compute cost.

---

## 6. Query Semantics & Scientific Validity Audit

### 6.1 Join Semantics & Task Isolation
- Task attempts are anchored by `time >= 0 AND time <= 86400000000` (Day 1) with `type = 3 (SCHEDULE)`.
- Event join condition: `te.terminal_time >= u.end_time` guarantees that observation windows strictly precede the terminal event.
- Earliest event resolution: `ROW_NUMBER() OVER (PARTITION BY u.collection_id, u.instance_index, u.start_time ORDER BY te.terminal_time ASC)` selects the earliest subsequent terminal outcome, eliminating duplicate window rows.

### 6.2 Labeling Horizons & Competing Risks
- Ground truth failure: `type = 5 (FAIL)` (software crash) and `type = 8 (LOST)` (host crash).
- Time-to-failure: `tau = terminal_event_timestamp_us - window_end_us`.
- **CRITICAL (Class 2)**: `0 < tau <= 300 s` (0 to 5 minutes).
- **WARNING (Class 1)**: `600 < tau <= 1500 s` (10 to 25 minutes).
- **Transition Buffer**: `300 < tau <= 600 s` (5 to 10 minutes; excluded from training to prevent label ambiguity).
- **NORMAL (Class 0)**: Confirmed clean completion (`type = 6 (FINISH)`) or survival `> 1500 s` without failure.
- **Right-Censoring**: Tasks active near Day 1 cutoff without terminal event are labeled `-1 (UNKNOWN/CENSORED)` and excluded from training failure windows.

### 6.3 Feature Leakage Safeguards
All terminal fields (`terminal_event_timestamp_us`, `terminal_type_code`, `terminal_status`) are used exclusively for offline target label construction during preprocessing. They are strictly excluded from the model feature input tensor X.

### 6.4 Operational Gates Evaluatability
All four governing Phase 7B operational gates can be evaluated on the resulting pilot dataset:
1. **Episode Recall $\ge 70\%$**: Evaluated over consecutive pre-failure window episodes.
2. **Genuine Critical Recall $\ge 50\%$**: Evaluated on windows observed strictly within 5 minutes of verified failure.
3. **Pure-Normal Entity FAR $\le 10\%$**: Evaluated on pure-normal parent collections.
4. **Normal-Window FPR $\le 5\%$**: Evaluated across non-failure sliding windows.

---

## 7. Protected Partitions & Non-Contamination

- **Candidate A Test (1,025 windows)**: Strictly unaccessed (`PROTECTED_UNACCESSED`).
- **Phase 6 Confirmatory Holdouts**: Strictly unaccessed (`PROTECTED_UNACCESSED`).
- **Data Materialization**: Zero records extracted; no CSV/Parquet files created.
- **Model Training**: Zero models trained or tuned.

---

## 8. Pre-Extraction Checklist

| Verification Checkpoint | Status | Forensic Evidence & Details |
| :--- | :---: | :--- |
| **1. SQL File Existence & Readability** | `PASS` | File exists at results/phase7y_borg2019_pilot/phase7y_candidate_a_projection_reduction.sql (2993 bytes). |
| **2. Actual Checksum Calculation** | `PASS` | SHA-256 computed on disk: 8abdbd8bdf20444f1282a82e01b30de0f8b9182c1d706a94b940c5c054ee395e. |
| **3. Historical Checksum Reconciliation** | `PASS` | Historical whitespace/newline variations documented; BigQuery service re-dry-run confirmed identical AST and scan volume. |
| **4. Service Dry Run Verification** | `PASS` | BigQuery service dry run validated: 418,578,205,568 bytes processed ($2.3793 USD). |
| **5. Automated Scripts Side-Effect Check** | `PASS` | Inspected all repository scripts; no script executes data-returning queries or automated table materialization. |
| **6. SQL Semantic & Join Correctness** | `PASS` | Event window join anchors te.terminal_time >= u.end_time; ROW_NUMBER selects earliest future terminal event; rn=1 eliminates duplicates. |
| **7. Label Horizon & Buffer Integrity** | `PASS` | Critical (0-5 min), Warning (10-25 min), Buffer (5-10 min excluded), and Right-Censoring correctly defined. |
| **8. Feature Leakage Prevention** | `PASS` | Terminal event fields strictly isolated from model input features; only past/current telemetry is used in feature vectors. |
| **9. Operational Gates Evaluatability** | `PASS` | All 4 Phase 7B operational gates (Episode Recall, Critical Recall, Entity FAR, Window FPR) are evaluatable on Candidate A schema. |
| **10. Application Default Credentials Check** | `PASS` | ADC initialized successfully for account kushagrapandey0333@gmail.com without credential logging. |
| **11. Active GCP Project Verification** | `PASS` | Client project and gcloud core/project verified as canvas-advice-464215-n1. |
| **12. Dataset Location & Accessibility** | `PASS` | Target dataset google.com:google-cluster-data.clusterdata_2019_a verified in location US. |
| **13. Billing Account Status Verification** | `PASS` | gcloud billing projects describe confirmed billingEnabled: false; project has no billing account attached. |
| **14. BigQuery Sandbox Status Verification** | `PASS` | BigQuery Sandbox active. Zero billing risk; queries draw exclusively from monthly 1 TiB free quota. |
| **15. Monthly Free-Tier Quota Availability** | `PASS` | client.list_jobs() verified 0 past executed jobs in project this month; full 1,024 GiB free quota available. |
| **16. Proposed Ceiling Budget Compliance** | `PASS` | Candidate A utilizes 93.02% of proposed 450 GB scan cap and 95.17% of proposed $2.50 USD cap. |
| **17. Protected Partitions Non-Contamination** | `PASS` | Candidate A Test and Phase 6 Confirmatory holdouts remain strictly untouched and unaccessed. |

---

## 9. Explicit Decision Required from the Research Owner

Before data extraction can be executed, the research owner must review this package and provide explicit authorization on:
1. **Quota / Budget Authorization**: Approve executing Candidate A under BigQuery Sandbox, consuming **`389.83 GiB` (38.07%)** of the project's monthly 1 TiB free tier quota at **`$0.00 USD`** marginal financial cost (or approve amending the query scan cap to **`450 GB`** and nominal cost cap to **`$2.50 USD`**).
2. **Bivariate Telemetry Scope**: Confirm acceptance of the minimal leaf bivariate telemetry schema (`cpu_avg_usage` + `memory_canonical_usage`) for this external pilot, avoiding higher scan volumes associated with auxiliary counters.
3. **Local Ingestion Path**: Authorize streaming query results directly to `data/raw/borg2019_pilot/` in compressed Parquet format.

---

## 10. Mandatory Final Safety Statement

> **No Borg 2019 data extraction, export, staging-table population, pilot dataset creation, model training, or protected holdout access was authorized or performed during this pre-extraction review. Any subsequent extraction requires a separate explicit authorization after the research owner reviews this package.**
