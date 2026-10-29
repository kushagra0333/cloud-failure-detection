# Phase 7Z.2: Independent Final Authorization Readiness Audit Report

**Project Root**: `C:\Users\hp\Desktop\Research project`  
**Execution Timestamp**: 2026-10-29T13:34:48.169029+00:00  
**Auditor**: Independent ML Research Auditor & Scientific Governance Reviewer  
**Query File Under Audit**: [`phase7y_candidate_a_projection_reduction.sql`](file:///C:/Users/hp/Desktop/Research project/results/phase7y_borg2019_pilot/phase7y_candidate_a_projection_reduction.sql)  
**Actual Query SHA-256**: `8abdbd8bdf20444f1282a82e01b30de0f8b9182c1d706a94b940c5c054ee395e` (2993 bytes)  
**Final Audit Verdict**: `NOT_READY_BILLING_OR_QUOTA_UNVERIFIED`  

---

## 1. Executive Summary & Audit Verdict

This document presents the **Independent Final Authorization Readiness Audit** reviewing the Phase 7Z.1 pre-extraction package for Candidate A ([`phase7y_candidate_a_projection_reduction.sql`](file:///C:/Users/hp/Desktop/Research project/results/phase7y_borg2019_pilot/phase7y_candidate_a_projection_reduction.sql)).

> [!IMPORTANT]
> **Final Audit Verdict: `NOT_READY_BILLING_OR_QUOTA_UNVERIFIED`**  
> 
> While Candidate A is technically valid, semantically sound, and confirmed to scan **418,578,205,568 bytes (418.58 GB)** with zero financial billing liability under BigQuery Sandbox, the authorization package **cannot be certified as fully ready** due to two specific governance and safeguard deficiencies:
> 
> 1. **Unsupported Quota Assertion in Phase 7Z.1**: Phase 7Z.1 asserted `PASS` for remaining quota based on `len(client.list_jobs()) == 0`. Under independent audit, this is identified as an **unsupported inference**. Client-side job history is limited to local submissions and does not query Google Cloud's central quota tracking infrastructure. Because Google BigQuery Sandbox does not expose remaining monthly free tier bytes via API, this checkpoint is formally downgraded to **`UNVERIFIED`**.
> 2. **Missing Repository `.gitignore`**: The repository currently **lacks a `.gitignore` file**. Extracting raw Parquet/CSV data into `data/raw/borg2019_pilot/` creates a serious risk of accidentally committing hundreds of megabytes of raw cluster telemetry to git.
> 
> **Transition to Readiness**: Once the research owner manually confirms in Google Cloud Console that their project has sufficient remaining monthly Sandbox quota and creates the required `.gitignore` file, the package transitions to:
> > **`READY_FOR_OWNER_EXTRACTION_DECISION`**  
> > *(Which means only that the owner may make an informed decision. It does not grant extraction permission).*

---

## 2. Independent Query Identity & Dry-Run Verification

Direct forensic inspection of [`phase7y_candidate_a_projection_reduction.sql`](file:///C:/Users/hp/Desktop/Research project/results/phase7y_borg2019_pilot/phase7y_candidate_a_projection_reduction.sql) confirmed:
- **Actual Byte Size**: **`2,993 bytes`**
- **Actual SHA-256**: **`8abdbd8bdf20444f1282a82e01b30de0f8b9182c1d706a94b940c5c054ee395e`**
- **Reconciliation with Phase 7Z.1**: Matches the Phase 7Z.1 report exactly. The 1-byte difference from the historical Phase 7Y optimization report (2,992 bytes) is a trailing newline in git.
- **Service Dry Run Re-Verification**: Re-submitting the exact query to BigQuery service (`dry_run=True`, `use_query_cache=False`) under project `canvas-advice-464215-n1` returned:
  - `total_bytes_processed`: **`418,578,205,568 bytes` (389.83 GiB / 418.58 GB)**
  - `job_id`: `None` (Proof of zero execution)
  - `estimated_on_demand_cost`: **`$2.3793 USD`** (~`$2.38 USD` at $6.25/TiB)

---

## 3. BigQuery Sandbox and Quota Claims Audit

### Verified Facts:
1. **Billing Absence**: `gcloud billing projects describe canvas-advice-464215-n1` returned `billingEnabled: false` and `billingAccountName: ''`. This is a verified fact.
2. **Sandbox Active**: Google Cloud Quotas API returned `ineligibilityReason: NO_VALID_BILLING_ACCOUNT`. The project operates in BigQuery Sandbox.
3. **Free Tier Allocation**: Google Cloud policy grants 1 TiB (1,024 GiB) of on-demand query capacity per calendar month.
4. **Nominal vs Billed Charge**: Candidate A has a nominal on-demand cost of $2.38 USD, but an expected financial charge of **`$0.00 USD`**.

### Unsupported Assertion Corrected:
- In Phase 7Z.1, Checkpoint 15 claimed:
  `client.list_jobs() verified 0 past executed jobs in project this month; full 1,024 GiB free quota available. (PASS)`
- **Auditor Finding**: This claim conflates local client job history with Google Cloud central quota tracking. `client.list_jobs()` only lists queries submitted through the current client context; it cannot query the global quota counter.
- **Classification**: **`UNVERIFIED`**.
- **Owner Manual Verification Action**:
  The research owner must open the Google Cloud Console:
  [https://console.cloud.google.com/bigquery?project=canvas-advice-464215-n1](https://console.cloud.google.com/bigquery?project=canvas-advice-464215-n1)
  and confirm that query usage for the current month has not exceeded 634 GiB.

---

## 4. Direct Scientific Semantics Audit

Inspection of the SQL text confirmed full semantic validity:
1. **Source Tables and Columns**:
   - `instance_usage`: 6 leaf columns (`collection_id`, `instance_index`, `start_time`, `end_time`, `average_usage.cpus`, `average_usage.memory`).
   - `instance_events`: 4 leaf columns (`collection_id`, `instance_index`, `time`, `type`).
   - Output projection: Exactly 9 columns, matching report claims.
2. **Event Window Join Semantics**:
   - `te.terminal_time >= u.end_time` strictly guarantees that observation windows precede the terminal event. Post-event telemetry cannot be joined.
3. **Deduplication & Earliest Terminal Matching**:
   - `ROW_NUMBER() OVER (PARTITION BY u.collection_id, u.instance_index, u.start_time ORDER BY te.terminal_time ASC)` with `WHERE rn = 1 OR rn IS NULL` guarantees that each 5-minute telemetry window appears exactly once and pairs with the earliest future terminal outcome.
4. **Label Horizons & Competing Risks**:
   - Class 2 (CRITICAL): `0 < tau <= 300 s` (0–5 min).
   - Class 1 (WARNING): `600 < tau <= 1500 s` (10–25 min).
   - Transition Buffer: `300 < tau <= 600 s` (5–10 min; excluded from training).
   - Class 0 (NORMAL): Clean FINISH or survival `> 1500 s`.
   - Censoring: Incomplete tasks near Day 1 cutoff labeled `-1 UNKNOWN`.
5. **Operational Gates Evaluatability**:
   - Episode Recall $\ge 70\%$, Genuine Critical Recall $\ge 50\%$, Pure-Normal Entity FAR $\le 10\%$, and Normal-Window FPR $\le 5\%$ are all directly computable.
   - Bounded to single-cluster cell `a`.

---

## 5. Extraction Feasibility & Safeguards Audit

1. **Local Destination**: `data/raw/borg2019_pilot/` is scientifically appropriate.
2. **Disk Space**: C: drive has **`61.2 GB free`**, which is more than adequate for ~1.5 GB of compressed Parquet.
3. **Repository Safeguard Deficiency**:
   - **`Test-Path .gitignore` returned `False`**.
   - The repository currently has **no `.gitignore` file**.
   - If raw Parquet/CSV files are written to `data/raw/`, any future `git add .` or commit would commit raw data to git.
   - **Mandatory Pre-Extraction Action**: A `.gitignore` must be created containing:
     ```gitignore
     data/raw/
     *.parquet
     *.csv.gz
     ```

---

## 6. Audit Checkpoints Table

| Checkpoint | Status | Forensic Evidence & Details |
| :--- | :---: | :--- |
| **CHK-01: SQL File Existence and Integrity** | `PASS` | File verified on disk at results/phase7y_borg2019_pilot/phase7y_candidate_a_projection_reduction.sql (2993 bytes). |
| **CHK-02: SQL SHA-256 Checksum Verification** | `PASS` | Current SHA-256 verified as 8abdbd8bdf20444f1282a82e01b30de0f8b9182c1d706a94b940c5c054ee395e. Reconciled with Phase 7Z.1 report (exact match). |
| **CHK-03: Genuine BigQuery Service Dry Run** | `PASS` | Dry-run job validated by BigQuery service: 418,578,205,568 bytes processed (389.83 GiB / 418.58 GB). Job ID is None (zero data returned). |
| **CHK-04: GCP Account & Project Configuration** | `PASS` | ADC active for kushagrapandey0333@gmail.com; active client project and gcloud project verified as canvas-advice-464215-n1. |
| **CHK-05: Billing Account Absence Verification** | `PASS` | gcloud billing projects describe confirmed billingEnabled: false; project has no billing account attached. |
| **CHK-06: BigQuery Sandbox Applicability** | `PASS` | Google Cloud Quotas API returned ineligibilityReason: NO_VALID_BILLING_ACCOUNT; BigQuery Sandbox is active. |
| **CHK-07: Monthly Free-Tier Query Quota Allocation** | `PASS` | Standard BigQuery Sandbox policy grants 1 TiB (1,024 GiB / 1,099,511,627,776 bytes) of query capacity per calendar month. |
| **CHK-08: Remaining Free-Tier Quota Balance** | `UNVERIFIED` | UNVERIFIED from client API. In Phase 7Z.1, this was asserted as PASS based on len(client.list_jobs()) == 0. However, client job history only covers local submissions and does not query Google Cloud's central quota tracking system. Google BigQuery does not expose a programmatic remaining_free_quota API endpoint without Cloud Monitoring Metrics. The research owner must manually inspect https://console.cloud.google.com/bigquery?project=canvas-advice-464215-n1 to confirm remaining quota. |
| **CHK-09: Marginal Financial Cost Disambiguation** | `PASS` | Verified zero billing risk. Because billingEnabled is false, Google cannot bill a credit card. Nominal on-demand cost is $2.3793 USD, but actual billed charge is $0.00 USD under Sandbox free tier. |
| **CHK-10: SQL Source Tables & Columns Audit** | `PASS` | Queries 6 leaf columns from instance_usage and 4 leaf columns from instance_events. Matches report claims exactly. |
| **CHK-11: Event Window Join & Temporal Causal Order** | `PASS` | Join condition te.terminal_time >= u.end_time strictly enforces that observation windows precede terminal events. No post-event telemetry can be joined. |
| **CHK-12: ROW_NUMBER Deduplication & Earliest Terminal Matching** | `PASS` | PARTITION BY collection_id, instance_index, start_time ORDER BY te.terminal_time ASC with rn = 1 guarantees exactly one observation row per window and selects earliest outcome. |
| **CHK-13: Label Horizons & Ambiguity Buffer Alignment** | `PASS` | Critical (0-5 min), Warning (10-25 min), Buffer (5-10 min excluded), and Right-Censoring match Phase 7B protocol definitions exactly. |
| **CHK-14: Feature Leakage Isolation** | `PASS` | Future fields (terminal_time, terminal_type_code, terminal_status) are excluded from model input features and used only for label generation. |
| **CHK-15: Operational Gates Compatibility** | `PASS` | All 4 Phase 7B operational gates (Episode Recall >=70%, Critical Recall >=50%, Entity FAR <=10%, Window FPR <=5%) can be calculated from Candidate A schema. |
| **CHK-16: Local Disk Space Sufficiency** | `PASS` | C: drive has 61.2 GB free space. Proposed pilot Parquet extraction (~500 MB to 1.5 GB) requires < 2.5% of free space. |
| **CHK-17: Repository .gitignore Extraction Safeguard** | `UNVERIFIED` | DEFICIENCY IDENTIFIED: .gitignore does not currently exist in the repository root. Before extracting data into data/raw/borg2019_pilot/, a .gitignore file must be created to ignore data/raw/ and *.parquet files, preventing accidental git commits of large raw datasets. |
| **CHK-18: Protected Holdouts Quarantine** | `PASS` | Candidate A Test (1,025 windows) and Phase 6 Confirmatory holdouts are strictly untouched and unaccessed (PROTECTED_UNACCESSED). |

---

## 7. Actionable Decision Path for the Research Owner

To resolve the two audit findings and transition to extraction:
1. **Console Quota Check**: Open [Google Cloud Console BigQuery](https://console.cloud.google.com/bigquery?project=canvas-advice-464215-n1) and confirm that less than 634 GiB of query quota has been used this month.
2. **Safeguard Initialization**: Create `.gitignore` in repository root with `data/raw/` and `*.parquet`.
3. **Explicit Extraction Decision**: Decide whether to authorize Candidate A execution to extract the Day 1 pilot cohort.

---

## 8. Mandatory Final Safety Statement

> **No data-returning query, extraction, export, staging operation, pilot dataset creation, model training, or protected holdout access was authorized or performed during this audit. A separate explicit owner decision is required before any extraction.**
