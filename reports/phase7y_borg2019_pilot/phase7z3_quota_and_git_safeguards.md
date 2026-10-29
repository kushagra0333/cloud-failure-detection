# Phase 7Z.3: Quota Verification and Git Data Protection Report

**Project Root**: `C:\Users\hp\Desktop\Research project`  
**Execution Timestamp**: 2026-10-29T13:53:33.091938+00:00  
**Primary Query File**: [`phase7y_candidate_a_projection_reduction.sql`](file:///C:/Users/hp/Desktop/Research project/results/phase7y_borg2019_pilot/phase7y_candidate_a_projection_reduction.sql)  
**Query SHA-256**: `8abdbd8bdf20444f1282a82e01b30de0f8b9182c1d706a94b940c5c054ee395e` (2993 bytes)  
**Historical Reports**: [`Phase 7Z.1 Package`](file:///C:/Users/hp/Desktop/Research project/reports/phase7y_borg2019_pilot/phase7z1_pre_extraction_authorization_package.md) | [`Phase 7Z.2 Audit`](file:///C:/Users/hp/Desktop/Research project/reports/phase7y_borg2019_pilot/phase7z2_final_authorization_readiness_audit.md)  
**Final Verdict**: `NOT_READY_BILLING_OR_QUOTA_UNVERIFIED`  

---

## 1. Executive Summary & Audit Resolution

Phase 7Z.3 was initiated to resolve the two outstanding readiness blockers identified in the Phase 7Z.2 independent audit:
1. **Repository Git Data Protection**: Resolve the absence of a root `.gitignore` file.
2. **BigQuery Sandbox Quota Establishment**: Rigorously analyze BigQuery Sandbox quota policy, explain client API limitations, formulate exact quota arithmetic, and provide the research owner with a verified console check procedure.

> [!IMPORTANT]
> **Final Verdict: `NOT_READY_BILLING_OR_QUOTA_UNVERIFIED`**  
> 
> - **Issue 2 (Git Data Protection) is FULLY RESOLVED (`PASS`)**: A root `.gitignore` has been created, tested with `git check-ignore`, and confirmed active. All raw datasets (`/data/raw/`) and Parquet files (`*.parquet`) are strictly ignored, while source code, SQL, reports, and manifests remain trackable.
> - **Issue 1 (Sandbox Quota Availability) remains OUTSTANDING (`UNVERIFIED`)**: Because Google Cloud does not expose an automated API endpoint for remaining Sandbox free tier query bytes under client credentials, remaining quota cannot be established programmatically from local scripts.
> - **Governing Decision Rule**: In accordance with the governing instructions (*"If manual quota verification remains outstanding, report NOT_READY_BILLING_OR_QUOTA_UNVERIFIED"*), the package remains formally classified as `NOT_READY_BILLING_OR_QUOTA_UNVERIFIED` until the research owner performs the 2-minute manual verification in Google Cloud Console.

---

## 2. Active GCP Project & BigQuery Sandbox Quota Analysis

### 2.1 Active Configuration & Billing Status
- **Authenticated Account**: `kushagrapandey0333@gmail.com`
- **Active Project ID**: `canvas-advice-464215-n1` (verified in both gcloud and BigQuery client)
- **Billing Status**:
  `gcloud billing projects describe canvas-advice-464215-n1` returned:
  ```yaml
  billingAccountName: ''
  billingEnabled: false
  projectId: canvas-advice-464215-n1
  ```
- **BigQuery Sandbox Applicability**: **CONFIRMED ACTIVE**. Official Google Cloud documentation specifies that any project without a linked billing account automatically uses BigQuery Sandbox.
- **Financial Cost Reality**: Because `billingEnabled: false`, Google Cloud physically cannot bill a payment method. The expected billed charge for Candidate A is **`$0.00 USD`** (marginal financial cost $0.00).

### 2.2 Programmatic Verification Limitation
Google Cloud BigQuery does not expose a programmatic `remaining_free_tier_bytes` field via the standard BigQuery client or `gcloud` CLI without Cloud Monitoring Metrics API. While `len(client.list_jobs()) == 0` confirms that no query jobs were executed through this local client, it does not prove global quota state across potential other tools or identities. Therefore, the remaining quota balance is formally designated as **`UNVERIFIED`**.

---

## 3. Mathematical Reconciliation of the 634 GiB Threshold

The previous Phase 7Z.2 audit report noted that monthly query usage must remain below 634 GiB. The rigorous mathematical derivation using consistent binary units (IEC 60027-2) is:

| Parameter | Binary Units (IEC) | Decimal Representation | Exact Bytes |
| :--- | :---: | :---: | :---: |
| **Total Monthly Sandbox Quota** | **`1,024.00 GiB`** (1 TiB) | `1,099.51 GB` | `1,099,511,627,776 bytes` |
| **Candidate A Estimated Scan** | **`389.831 GiB`** | `418.578 GB` | `418,578,205,568 bytes` |
| **Minimum Remaining Quota Required** | **`389.831 GiB`** | `418.578 GB` | `418,578,205,568 bytes` |
| **Maximum Allowable Prior Usage** | **`634.169 GiB`** | `680.933 GB` | `680,933,422,208 bytes` |
| **Candidate A with 10% Safety Buffer** | **`428.814 GiB`** | `460.436 GB` | `460,436,026,124 bytes` |
| **Max Prior Usage with 10% Buffer** | **`595.186 GiB`** | `639.075 GB` | `639,075,601,652 bytes` |

### Key Derivation Findings:
1. Candidate A consumes **`38.07%`** of the total 1,024 GiB monthly quota.
2. For Candidate A to execute successfully under Sandbox without hitting quota errors, total query usage in project `canvas-advice-464215-n1` between the 1st of the current calendar month and today must be strictly **<= 634.17 GiB** (634.1689 GiB).
3. If a conservative 10% query safety buffer is applied, prior monthly usage must be **<= 595.19 GiB**.

---

## 4. Root `.gitignore` Implementation & Testing

### 4.1 Implemented `.gitignore` Rules
A root `.gitignore` file was created at `.gitignore`:
```gitignore
# Raw research datasets
/data/raw/
*.parquet

# Python bytecode and virtual environment
__pycache__/
*.py[cod]
.venv/
.pytest_cache/

# Local temporary files
*.tmp
```

### 4.2 `git check-ignore` Test Results
All paths were tested using `git check-ignore -v`:

| Test Path | Status |
| :--- | :--- |
| `data/raw/borg2019_pilot/pilot_telemetry.parquet` | **Ignored (Protected)** |
| `hypothetical_test.parquet` | **Ignored (Protected)** |
| `data/raw/new_file.csv` | **Ignored (Protected)** |
| `scripts/run_phase7z3_safeguards_audit.py` | Trackable |
| `results/phase7y_borg2019_pilot/phase7y_candidate_a_projection_reduction.sql` | Trackable |
| `reports/phase7y_borg2019_pilot/phase7z3_quota_and_git_safeguards.md` | Trackable |

- **Result**: Raw dataset paths and Parquet files are **strictly ignored**. Source code, SQL queries, markdown reports, and configuration files remain **trackable**.

### 4.3 Audit of Existing Tracked Data Files
In accordance with Task 2.3, existing tracked files were audited using `git ls-files`:
- `data/raw/sample_task_events.csv.gz`
- `data/raw/sample_task_events_part1.csv.gz`
- `data/raw/sample_task_events_part2.csv.gz`
- `data/raw/sample_task_usage.csv.gz`
- `data/raw/sample_task_usage_part1.csv.gz`

- **Finding**: Five historical sample gzip CSV files from Borg 2011 are currently tracked in git.
- **Safety Action**: Git ignore rules only apply to untracked files. In accordance with the prompt's instruction (*"Do not remove it, rewrite history, or untrack files without separate approval"*), these existing historical sample files remain untouched in git history.

---

## 5. Checkpoints Table

| Checkpoint | Status | Forensic Evidence & Details |
| :--- | :---: | :--- |
| **CHK-Z3-01: Active Project and Account Verification** | `PASS` | Configured project is canvas-advice-464215-n1; authenticated identity is kushagrapandey0333@gmail.com. |
| **CHK-Z3-02: Billing Account Absence Verification** | `PASS` | gcloud billing projects describe confirmed billingEnabled: false and billingAccountName: ''. No credit card attached. |
| **CHK-Z3-03: BigQuery Sandbox Applicability** | `PASS` | Authoritative Google Cloud documentation confirms projects without billing automatically use BigQuery Sandbox. |
| **CHK-Z3-04: Applicable Monthly Query Quota and Units** | `PASS` | Sandbox provides 1 TiB (1,024 GiB / 1,099,511,627,776 bytes) of on-demand query capacity per calendar month. |
| **CHK-Z3-05: Remaining Quota Balance Verification** | `UNVERIFIED` | UNVERIFIED from client API. Google BigQuery does not expose a programmatic remaining_quota API endpoint. Local job history (client.list_jobs()) does not prove central Google Cloud quota consumption. The research owner must manually inspect https://console.cloud.google.com/bigquery?project=canvas-advice-464215-n1. |
| **CHK-Z3-06: Required Quota versus Available Quota Math** | `PASS` | Candidate A requires 418,578,205,568 bytes (389.83 GiB). Under the 1,024 GiB Sandbox allowance, prior monthly usage must be <= 634.17 GiB (or <= 595.19 GiB with 10% safety margin). Math reconciled exactly. |
| **CHK-Z3-07: Query Checksum and Dry-Run Estimate Parity** | `PASS` | SQL on disk verified: 8abdbd8bdf20444f1282a82e01b30de0f8b9182c1d706a94b940c5c054ee395e (2993 bytes). BigQuery service dry run confirmed: 418,578,205,568 bytes. |
| **CHK-Z3-08: Root .gitignore Implementation and Testing** | `PASS` | Root .gitignore created with /data/raw/ and *.parquet. Verified via git check-ignore: data/raw/ and *.parquet ignored; code/reports trackable. |
| **CHK-Z3-09: Existing Tracked Raw Data Audit** | `PASS` | Identified 5 existing tracked sample files in data/raw/ (historical Borg 2011 samples). Git rules preserve them; no history rewrite or untracking performed without authorization. |
| **CHK-Z3-10: Protected Holdouts Quarantine** | `PASS` | Candidate A Test (1,025 windows) and Phase 6 Confirmatory holdouts remain strictly untouched (PROTECTED_UNACCESSED). |

---

## 6. Actionable Manual Procedure for the Research Owner

Because client APIs cannot query central quota counters, the research owner must execute this 2-minute manual verification:

1. Open the **Google Cloud Console BigQuery Studio**:
   [https://console.cloud.google.com/bigquery?project=canvas-advice-464215-n1](https://console.cloud.google.com/bigquery?project=canvas-advice-464215-n1)
2. In the BigQuery Studio navigation or Quotas page ([https://console.cloud.google.com/apis/api/bigquery.googleapis.com/quotas?project=canvas-advice-464215-n1](https://console.cloud.google.com/apis/api/bigquery.googleapis.com/quotas?project=canvas-advice-464215-n1)):
   - Check the metric: **`Query usage`** (`bigquery.googleapis.com/quota/query/usage`).
3. Confirm that current-month query usage is **<= 634.17 GiB**.
4. Once verified, the research owner may proceed to formally authorize Candidate A extraction.

---

## 7. Mandatory Final Safety Statement

> **No data-returning BigQuery query, extraction, export, staging operation, pilot dataset creation, model training, or protected holdout access was authorized or performed during Phase 7Z.3. Completion of this phase does not authorize data extraction. A separate explicit owner decision is required.**
