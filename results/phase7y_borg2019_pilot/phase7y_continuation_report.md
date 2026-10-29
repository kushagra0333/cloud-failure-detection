# Phase 7Y Continuation Report: BigQuery Sandbox and Authenticated Access Evaluation

**Project Root**: `C:\Users\hp\Desktop\Research project`  
**Execution Timestamp**: 2026-10-29T10:40:04.921716+00:00  
**Target Public Dataset**: `google.com:google-cluster-data.clusterdata_2019_a`  
**Operational Verdict**: `DATA_ACCESS_BLOCKED`  

---

## Executive Summary

Phase 7Y Continuation evaluated whether Google Borg ClusterData 2019 can be queried via BigQuery Sandbox or authenticated local client access, inspected the corrected SQL query plan against the official Borg 2019 schema, recalculated cost and budget constraints under current pricing, and revisited the causal labeling and right-censoring specifications.

### Key Findings:
1. **Authentication Blocker**: Local client initialization threw `google.auth.exceptions.DefaultCredentialsError`. No Application Default Credentials (ADC) file, Google Cloud environment variables, or CLI tools (`gcloud`, `bq`) exist on the local system.
2. **BigQuery Sandbox Reality**: BigQuery Sandbox offers 1 TiB/month of free query scanning and 10 GB of storage without requiring a billing account. **However, BigQuery Sandbox strictly mandates an authenticated Google Cloud Project ID and OAuth 2.0 / ADC bearer tokens.** The BigQuery REST API rejects anonymous, unauthenticated calls.
3. **Budget and Byte Cap Alignment**: Under current BigQuery on-demand analysis pricing ($6.25/TiB):
   - The earlier 25 GB ($25,000,000,000$ bytes) cap implies a maximum query cost of **$0.142 USD**, exceeding the strict $0.125 USD paid-query budget cap.
   - To strictly enforce the $\le \$0.125$ USD budget cap, the maximum byte scan cap is reduced to **$\le 21,000,000,000$ bytes (21.0 GB)**, yielding a maximum cost of **$0.1194 USD**.
   - Under BigQuery Sandbox, a 21.0 GB query consumes **1.91%** of the 1 TiB monthly quota at **$0.00 USD** marginal cost.
4. **Provisional Estimates**: The earlier figures (18.4 GB scan volume, $0.092 USD, 125,000 events) remain **provisional offline theoretical estimates** derived from protobuf column widths. They must not be represented as verified service responses until a live dry run is executed with authenticated credentials.
5. **Operational Verdict**: In strict adherence to experimental governance (Requirement 4: *"If credentials remain unavailable, stop and report the exact blocker. Do not manufacture successful authentication, query estimates or service responses."*), this continuation returns:
   $$\mathbf{DATA\_ACCESS\_BLOCKED}$$

---

## 1. Authentication Status and BigQuery Sandbox Assessment

### 1.1 Local Authentication Environment
- **Python Environment**: `C:\Users\hp\Desktop\Research project\.venv\Scripts\python.exe`
- **Installed Packages**: `google-cloud-bigquery` v3.46.2, `google-auth` v2.43.0.
- **Environment Variables**:
  - `GOOGLE_APPLICATION_CREDENTIALS`: Unset (`False`)
  - `GCLOUD_PROJECT`: Unset (`False`)
  - `GOOGLE_CLOUD_PROJECT`: Unset (`False`)
  - `CLOUDSDK_CORE_PROJECT`: Unset (`False`)
- **Local ADC Credentials File**: Not found at `C:\Users\hp\AppData\Roaming\gcloud\application_default_credentials.json`
- **Command-Line Tools**: `gcloud` and `bq` not found in system `PATH`.
- **Live Client Attempt**:
  ```python
  from google.cloud import bigquery
  client = bigquery.Client()
  # Raises: google.auth.exceptions.DefaultCredentialsError
  ```
- **Security Safeguard Compliance**: Zero credentials, secret keys, or bearer tokens were requested or printed in logs or artifacts.

### 1.2 BigQuery Sandbox Technical Architecture & Restrictions
Google Cloud BigQuery Sandbox provides no-cost exploration of BigQuery subject to specific rules:
- **Free Monthly Query Processing Allowance**: 1 TiB ($1,099,511,627,776$ bytes) per calendar month.
- **Storage Allowance**: 10 GB of active table storage.
- **Table Expiration**: Default partition/table expiration enforced at 60 days.
- **Quota Exceeded Behavior**: Queries exceeding 1 TiB/month fail with quota errors rather than incurring overage charges, protecting users against accidental billing.
- **Authentication Mandate**: BigQuery Sandbox does **not** provide unauthenticated or anonymous API access. Every API request to `bigquery.googleapis.com` must supply a project identifier and OAuth 2.0 / ADC bearer credentials.
- **Conclusion**: Local execution of a live BigQuery dry run is technically impossible without user-provided credentials.

---

## 2. Query Plan Audit and Schema Verification

### 2.1 Query Checksum and Identification
- **Query File**: `results/phase7y_borg2019_pilot/query_plan.sql`
- **File Size**: 3,238 bytes
- **SHA-256 Checksum**: `7b9f3bd7142980b0de3aa31b17d2993de078821c20a4f1d89c706bf01cab9602`
- **Target Dataset**: `google.com:google-cluster-data.clusterdata_2019_a`

### 2.2 Protobuf Schema Verification (`clusterdata_trace_format_v3.proto`)
The query was verified against the official Google cluster trace v3 protobuf specification:
- Nested CPU usage: `u.average_usage.cpus`, `u.maximum_usage.cpus` (Valid v3 `Resources` submessage).
- Nested Memory usage: `u.average_usage.memory` (Valid v3 `Resources` submessage).
- Memory limits and cache: `u.assigned_memory`, `u.page_cache_memory` (Valid scalar fields).
- Event lifecycle enumerations:
  - `type = 3` — `SCHEDULE` (task running).
  - `type = 4` — `EVICT` (scheduler priority preemption).
  - `type = 5` — `FAIL` (abnormal termination ground truth).
  - `type = 6` — `FINISH` (clean completion normal ground truth).
  - `type = 7` — `KILL` (cancellation).
  - `type = 8` — `LOST` (unreachable host ground truth).

### 2.3 Join Architecture Audit
- **Earliest Terminal Event Resolution**: `earliest_terminal` CTE applies `ROW_NUMBER() OVER (PARTITION BY collection_id, instance_index ORDER BY terminal_time ASC)` and joins on `et.rn = 1 AND et.terminal_time >= u.end_time`.
- **Observation on Multi-Attempt Tasks**: In Borg 2019, if an `instance_index` is rescheduled after an initial failure, `rn = 1` captures the initial failure timestamp. If an observation window occurs after `rn = 1`, `et.terminal_time >= u.end_time` evaluates to false, resulting in an unmatched left join. This is anti-leakage conservative, but subsequent pilot pipelines should track alloc attempts or window-specific event joins.

---

## 3. Cost Recalculation and Budget Consistency

### 3.1 Mathematical Budget Reconciliation
| Parameter | Baseline Value | Standard Current Value | Unit |
| :--- | :---: | :---: | :---: |
| On-Demand Query Price | $5.00 | $6.25 | USD / TiB |
| Bytes per TiB | $1,099,511,627,776$ | $1,099,511,627,776$ | bytes |
| Predeclared Budget Cap | $0.125 | $0.125 | USD |
| Old Byte Cap (25 GB) | 25,000,000,000 | 25,000,000,000 | bytes |
| **Old Byte Cap Cost** | **$0.1137** | **$0.1421** | **USD** |
| Old Cap Budget Status | Within Cap | **EXCEEDS BUDGET (+13.7%)** | - |
| **Revised Byte Cap (21 GB)** | **21,000,000,000** | **21,000,000,000** | **bytes** |
| **Revised Byte Cap Cost** | **$0.0955** | **$0.1194** | **USD** |
| Revised Cap Budget Status | **WITHIN CAP** | **WITHIN CAP** | - |

To maintain a strict $\le \$0.125$ USD budget cap at the current $6.25/TiB on-demand rate, the maximum allowed scan cap is strictly tightened from 25.0 GB down to **21.0 GB ($21,000,000,000$ bytes)**.

### 3.2 BigQuery Sandbox Quota Utilization
- **Monthly Free Allowance**: 1 TiB ($1,099,511,627,776$ bytes).
- **Pilot Utilization at 21.0 GB Cap**: **1.91%** of monthly quota.
- **Remaining Monthly Allowance**: **98.09%** (~1,078.5 GB).
- **Cost in Sandbox**: **$0.00 USD**.

### 3.3 Status of Analytical Estimates
The previous estimates (18.4 GB, $0.092 USD, 125,000 events) are strictly classified as **provisional theoretical estimates**. They are based on uncompressed column byte estimates and published cell_a event frequencies. Until an authenticated BigQuery service dry run provides the official `totalBytesProcessed` field, these estimates remain provisional.

---

## 4. Revisited Labeling and Right-Censoring Protocol

In accordance with Requirement 10, the label specification was audited and hardened:
1. **Complete 25-Minute Follow-Up**: Retained Normal windows ($y=0$) must have observed telemetry or lifecycle events for at least 25 minutes ($1,500\times 10^6 \ \mu\text{s}$) after window end.
2. **Post-Event Observation Prohibition**: Observations where $t_{\text{cutoff}} \ge t_{\text{fail}}$ are strictly excluded.
3. **Explicit Right-Censoring Elimination**: Observation windows ending within 25 minutes of trace boundary ($t_{\text{cutoff}} > 84,900\times 10^6 \ \mu\text{s}$) without a prior event are tagged as `RIGHT_CENSORED` and dropped from training and evaluation.
4. **Target Classes**:
   - Class 2 (CRITICAL): $0 < \Delta t \le 300\text{ s}$ ($0–5$ min lead time).
   - Class 1 (WARNING): $600 < \Delta t \le 1,500\text{ s}$ ($10–25$ min lead time).
   - Ambiguous Buffer: $300 < \Delta t \le 600\text{ s}$ ($5–10$ min lead time) — Strictly excluded.
   - Clean Completion: `FINISH` (type = 6) confirms Class 0 (NORMAL).
   - Competing Risks: `KILL` (type = 7) and `EVICT` (type = 4) are dropped from failure evaluation.
   - Sensitivity Target: `LOST` (type = 8) is isolated in `y_lost`.

---

## 5. Governance and Non-Contamination Safeguards

- **Candidate A Test (1,025 windows)**: Strictly untouched.
- **Phase 6 Confirmatory Holdouts**: Strictly untouched.
- **Zero Model Training**: No models trained, evaluated, or fitted.
- **Zero Synthetic Data**: No rows manufactured or simulated.
- **Historical Preservation**: All 15 original Phase 7Y artifacts and `phase7y_manifest.json` are preserved intact.

---

## 6. Actionable Authentication Instructions for User

To authorize the live BigQuery service dry run:
1. Open a local PowerShell window.
2. Ensure the Google Cloud SDK is installed (`https://cloud.google.com/sdk/docs/install`).
3. Run: `gcloud auth application-default login`
4. Set an active GCP Project (BigQuery Sandbox or Billing-enabled): `gcloud config set project <PROJECT_ID>`
5. Re-run the Phase 7Y dry run script.

---

## 7. Operational Verdict

$$\mathbf{DATA\_ACCESS\_BLOCKED}$$

Live BigQuery service dry run and data extraction remain blocked pending authenticated credentials.
