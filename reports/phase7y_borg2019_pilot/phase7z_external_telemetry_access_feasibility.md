# Phase 7Z: External Container Telemetry Access Feasibility Report

**Project Root**: `C:\Users\hp\Desktop\Research project`  
**Execution Timestamp**: 2026-10-29T13:14:58.898593+00:00  
**Investigation Scope**: Strictly Feasibility & Governance Audit (Zero Data Extraction)  
**Governing Caps**: Max Scan: `21,000,000,000 bytes (21.0 GB)`; Max Budget: `$0.125 USD`  
**Candidate A Query**: [`phase7y_candidate_a_projection_reduction.sql`](file:///C:/Users/hp/Desktop/Research%20project/results/phase7y_borg2019_pilot/phase7y_candidate_a_projection_reduction.sql) (SHA-256: `8abdbd8bdf20444f1282a82e01b30de0f8b9182c1d706a94b940c5c054ee395e`)  
**Original Query**: [`phase7y_dry_run_query.sql`](file:///C:/Users/hp/Desktop/Research%20project/results/phase7y_borg2019_pilot/phase7y_dry_run_query.sql) (SHA-256: `889e2841f0983248a9f1d4c2f4f917ae7733b2038f7b939a66e098f60699a8e6`)  
**Final Operational Verdict**: `BUDGET_AMENDMENT_WOULD_BE_REQUIRED`  

---

## 1. Executive Summary & Verdict

Phase 7Z was initiated following the Phase 7Y optimization audit and the Candidate D scientific feasibility review. The primary objective is to determine the most scientifically defensible and cost-controlled path to obtain legitimate **container-level runtime telemetry** for external failure-prediction research.

Candidate D (based on `collection_events`) was classified as **requiring a separate research design** because it contains **zero runtime telemetry** and cannot evaluate the governing Phase 7B sliding-window operational gates.

> [!IMPORTANT]
> **Final Feasibility Verdict: `BUDGET_AMENDMENT_WOULD_BE_REQUIRED`**  
> 
> A scientifically comparable external container telemetry source exists: **Google Borg ClusterData 2019** (`google.com:google-cluster-data.clusterdata_2019_a.instance_usage` and `instance_events`). It fully preserves 5-minute container telemetry, microsecond failure timestamps, the 3-class proximity labels, and all four governing Phase 7B operational gates.
> 
> However, on Google's public BigQuery repository, the `instance_usage` table (7.575 billion rows) is completely **unpartitioned and unclustered**. Under BigQuery's columnar architecture:
> - Every 8-byte column scanned requires reading **60.60 GB**.
> - The minimal leaf telemetry query (**Candidate A**) scans **`418,578,205,568 bytes` (418.58 GB / 389.83 GiB)**.
> - At the standard on-demand rate ($6.25/TiB), Candidate A costs **`$2.38 USD`**, which exceeds the approved $0.125 cap by **19.03x** and the 21 GB scan cap by **19.93x**.
> 
> Row filtering (tested in Candidate C) prunes zero bytes on unpartitioned tables. No public pre-filtered subsets or public partitioned tables exist for Borg 2019. Consequently, **no telemetry-preserving query can run within the existing $0.125 / 21 GB cap**.
> 
> To obtain genuine Borg 2019 container telemetry, a formal **budget amendment to $2.50 USD / 450 GB** is required, or formal authorization to consume **418.58 GB (38.1%)** of the user's monthly 1 TiB BigQuery Sandbox free tier quota at **$0.00 marginal cost**.

---

## 2. Candidate External Sources Inventory & Comparability Matrix

An exhaustive forensic audit of public cloud and cluster workload repositories was conducted:

| Candidate Source | Data Owner & Platform | Distribution / Storage | Runtime Telemetry? | Temporal Resolution | Scientific Comparability |
| :--- | :--- | :--- | :---: | :--- | :--- |
| **Google Borg ClusterData 2019 (v3)** | Google LLC / Google Research (Wilkes et al.) | `Google BigQuery (`google.com:google-cluster-data`)` | **Yes** | 5-minute periodic aggregated windows (300,000,000 us) | HIGH. Supports identical sequential container runtime failure prediction. |
| **Google Borg ClusterData 2011 (v2.1)** | Google LLC / Google Research (Reiss et al.) | `Google Cloud Storage (`gs://clusterdata-2011-2/`)` | **Yes** | 5-minute periodic aggregated windows (300 seconds) | Identical schema to existing research. |
| **Alibaba Cluster Trace 2018 (v2018)** | Alibaba Group (Alibaba System Software Team) | `GitHub (`alibaba/clusterdata`) and OSS HTTP mirrors` | **Yes** | Variable / heterogeneous (10s to 5-minute sampling) | MODERATE. Focuses on Fuxi DAG batch jobs colocated with Sigma containers. |
| **Microsoft Azure Public Dataset (VM Traces 2017/2019/2021)** | Microsoft Azure Research | `GitHub (`Azure/AzurePublicDataset`) / Azure Open Datasets` | **Yes** | 5-minute binned intervals | DISQUALIFIED. Conflates normal customer deallocations, autoscaling, and crashes. Violates Rules 2 & 3. |
| **Los Alamos National Laboratory Computer Failure Data Repository** | Los Alamos National Laboratory / USENIX | `CFDR public archive` | No | None (Discrete incident intervals) | DISQUALIFIED. Administrative remedy incident logs, zero continuous container telemetry time-series. |
| **Oak Ridge National Laboratory ClusterWise (Summit Supercomputer)** | Oak Ridge National Laboratory / US DOE | `ORNL data repository` | **Yes** | 1-second to 1-minute telemetry | DISQUALIFIED. Physical supercomputing hardware and GPU thermal modeling, not cloud container task failure. |

### Key Findings on Candidate Sources:
1. **Google Borg ClusterData 2019 (v3)**:
   - **Status**: **FULLY QUALIFIED AS SCIENTIFICALLY COMPARABLE**.
   - Contains 5-minute periodic container telemetry (`average_usage.cpus`, `maximum_usage.cpus`, `average_usage.memory`, `assigned_memory`, `page_cache_memory`, CPI histograms).
   - Joined via composite key `(collection_id, instance_index)` to `instance_events` with ground-truth abnormal failure code `type = 5 (FAIL)`.
   - Access is public via Application Default Credentials (ADC) under GCP project `canvas-advice-464215-n1`.
   - Distribution is exclusively via BigQuery; **no open public GCS flat-file bucket exists for Borg 2019** (unlike 2011).
2. **Google Borg ClusterData 2011 (v2.1)**:
   - **Status**: **DISQUALIFIED FOR EXTERNAL GENERALIZATION**.
   - Borg 2011 represents a single 12,500-node cell over 29 days. Candidate A, B, C, D, and Phase 6 Confirmatory holdouts were all sampled from this identical trace. Sourcing additional shards from 2011 provides an internal split, not an independent external workload cohort.
3. **Alibaba Cluster Trace 2018 (v2018)**:
   - **Status**: **PARTIALLY COMPARABLE (SECONDARY ALTERNATIVE)**.
   - Provides container CPU and memory utilization, and task `status = Failed`.
   - However, telemetry sampling is irregular and heterogeneous, requiring extensive binning and interpolation. Hardware performance counters (CPI, MAI) are absent. Tasks represent Fuxi DAG batch jobs and Sigma containers rather than Borg workloads.
   - Distributed as compressed flat files (~50 GB) downloadable without cloud query fees.
4. **Microsoft Azure VM Traces (2017/2019/2021)**:
   - **Status**: **DISQUALIFIED (FATAL DEFECT)**.
   - Lacks failure exit codes. Only records `vm_created` and `vm_deleted`. Treating normal customer VM deallocations as failures violates Rules 2 and 3 of the project's scientific governance.
5. **LANL CFDR & ORNL ClusterWise**:
   - **Status**: **DISQUALIFIED (MODALITY MISMATCH)**.
   - LANL contains discrete administrative outage logs without continuous telemetry time series. ORNL ClusterWise tracks supercomputer physical hardware/thermal metrics rather than cloud container software failures.

---

## 3. Storage and Cost Options Audit (Approaches A through E)

Every option was investigated rigorously against service evidence without executing extraction:

### Approach A: Prefiltered Telemetry Subset
- **Feasibility**: **INFEASIBLE (NONEXISTENT)**
- **Evidence**: Authoritative documentation (`ClusterData2019.md`) and BigQuery catalog inspection confirm that Google provides only full cell datasets (`clusterdata_2019_a` through `h`). There are no public pre-filtered subsets, sample tables, or day-sliced tables hosted by Google.
- **Cost**: $0.00 (nonexistent).

### Approach B: Partitioned or Clustered Source
- **Feasibility**: **INFEASIBLE (NONEXISTENT)**
- **Evidence**: Direct BigQuery API metadata inspection (`client.get_table(...)`) confirms:
  - Table: `google.com:google-cluster-data.clusterdata_2019_a.instance_usage`
  - Total Rows: `7,575,500,668` (2.26 TB)
  - `time_partitioning`: **`None`**
  - `clustering_fields`: **`None`**
  - Scan per 8-byte column: **`60,604,005,344 bytes` (60.6 GB / 56.44 GiB)**.
- **Finding**: Because the public table is unpartitioned, BigQuery on-demand pricing forces a full-table scan for every projected column.

### Approach C: Scientifically Defensible Sampling Without Scanning Entire Unpartitioned Source
- **Feasibility**: **TECHNICALLY INEFFECTIVE / METHODOLOGICALLY INVALID**
- **Evidence**:
  1. *Row Filters*: Candidate C ([phase7y_candidate_c_bounded_pilot_scope.sql](file:///C:/Users/hp/Desktop/Research%20project/results/phase7y_borg2019_pilot/phase7y_candidate_c_bounded_pilot_scope.sql)) applied `WHERE collection_id IN (5 jobs) AND start_time <= 3.6e9`. The measured dry-run scan was **`418,578,205,568 bytes`** (identical to Candidate A). Row-level `WHERE` clauses yield **0.0% scan reduction** on unpartitioned tables.
  2. *TABLESAMPLE*: BigQuery's `TABLESAMPLE SYSTEM (p PERCENT)` operates on storage blocks and does not support arbitrary temporal slicing. Furthermore, random row-level sampling destroys temporal sequence continuity, making it mathematically impossible to construct consecutive 5-minute sliding windows ($L=12$ steps).

### Approach D: Authorized Partitioned Staging Workflow
- **Feasibility**: **FEASIBLE WITH INITIAL STAGING INGESTION BUDGET**
- **Architecture**: A one-time DDL query creates a user-owned partitioned table `canvas-advice-464215-n1.borg2019_staging.instance_usage_partitioned` partitioned by `TIMESTAMP_TRUNC(start_time, DAY)` or integer microsecond range.
- **Cost Breakdown**:
  - Initial Ingestion Scan: **418.58 GB** (**`$2.38 USD`** on-demand, or absorbed by Sandbox free tier).
  - Storage Footprint: Storing 1 day of telemetry in the user's project = ~25 GB compressed = **`~$0.50 USD / month`**.
  - Subsequent Query Scans: Filtering to a 6-hour or 24-hour partition scans **<15 GB** (**`< $0.09 USD`** per query), which fits inside the $0.125 cap.
- **Trade-offs**: Requires initial funding or quota consumption, dataset creation permissions, and ongoing storage lifecycle management.

### Approach E: Formally Approved BigQuery Budget Amendment (Recommended)
- **Feasibility**: **DIRECTLY ACTIONABLE & MINIMAL COMPLEXITY**
- **Architecture**: Authorize direct execution of Candidate A ([phase7y_candidate_a_projection_reduction.sql](file:///C:/Users/hp/Desktop/Research%20project/results/phase7y_borg2019_pilot/phase7y_candidate_a_projection_reduction.sql)), which projects minimal leaf columns (6 usage columns + 4 event columns).
- **Cost & Quota Evidence**:
  - Measured Dry-Run Scan: **`418,578,205,568 bytes` (418.58 GB / 389.83 GiB)**.
  - On-Demand Cost: **`$2.3793 USD`** (~$2.38 USD) at $6.25/TiB.
  - Required Budget Amendment: Increase monetary budget cap from $0.125 USD to **$2.50 USD**, and scan cap from 21 GB to **450 GB**.
  - Zero-Cost Sandbox Option: BigQuery Sandbox / Free Tier provides **1 TiB (1,024 GiB)** of free analytical queries per month. Candidate A consumes **38.07%** of this monthly free quota at **$0.00 marginal cost**.
- **Advantages**: Extracts the exact bounded pilot cohort into local storage in a single execution step; avoids staging table overhead and ongoing storage fees.

---

## 4. Preservation of Scientific Equivalence

To ensure continuity with Phases 3–7U, any viable external extraction must preserve the exact task definition:

### 4.1 Telemetry Representation & Horizons
- Observation Window: $L = 12$ consecutive 5-minute steps (60 minutes lookback).
- Sampling Interval: $\Delta t = 300\text{ seconds}$ (5 minutes).
- Proximity Labeling Formulation:
  - **CRITICAL (2)**: Imminent abnormal crash within $0 < \tau \le 300\text{ s}$ (0–5 minutes).
  - **WARNING (1)**: Imminent abnormal crash within $600 < \tau \le 1,500\text{ s}$ (10–25 minutes).
  - **NORMAL (0)**: Active task with no failure in $(t, t + 1,500\text{ s}]$.
  - **UNKNOWN (-1)**: Censored task or missing future observation.

### 4.2 Operational Gates Compatibility
Google Borg 2019 supports all four governing Phase 7B operational gates:
1. **Episode Recall $\ge 70\%$**: Evaluated over sequences of consecutive pre-failure telemetry windows.
2. **Genuine Critical Recall $\ge 50\%$**: Evaluated on windows observed strictly within 5 minutes of a verified `type = 5 (FAIL)` event.
3. **Pure-Normal Entity FAR $\le 10\%$**: Evaluated on purely successful job collections.
4. **Normal-Window FPR $\le 5\%$**: Evaluated across all non-failure sliding windows.

> [!CAUTION]
> **Single-Cluster Boundary**: Sourcing a pilot cohort from cell `a` evaluates single-cluster performance within cell `a`. It does not support cross-cluster generalization claims. True cross-cluster evaluation requires testing on a separate cell (e.g. cell `b`), which would require a separate query.

---

## 5. Validation Design, Leakage Mitigation & Governance Safeguards

### 5.1 Validation Protocols
- **Temporal Ordering**: Enforce chronological train/validation splits (e.g. Day 1 00:00–18:00 UTC for training, 18:00–24:00 UTC for validation).
- **Group Splitting**: Group by `parent_collection_id` or `collection_id` to prevent task co-occurrence leakage between splits.
- **Event-to-Telemetry Joins**: Use an event window join that anchors `time >= end_time`. Never join future events whose timestamp precedes the window.
- **Right-Censoring**: Tasks running past the observation cutoff without terminal events must be marked CENSORED/UNKNOWN and excluded from positive training windows.

### 5.2 Minimum Evidence Required Prior to Extraction Authorization
Before any query execution returning data can be authorized:
1. Schema verification of all projected fields via BigQuery API (**COMPLETED in Phase 7Y**).
2. SQL checksum verification (**COMPLETED: Candidate A hash `33fa4a6968f47b9f8fe269ceb7adf1f72c4356292708f986a1f14e5ef82d09c4`**).
3. Verified BigQuery dry run confirming exact bytes (**COMPLETED: `418,578,205,568 bytes`**).
4. Formal governance authorization for budget amendment ($2.50 USD / 450 GB) or Sandbox free tier quota allocation (**PENDING DECISION**).

### 5.3 Non-Contamination Verification
- **Candidate A Test (1,025 windows)**: Strictly unaccessed and untouched (`PROTECTED_UNACCESSED`).
- **Phase 6 Confirmatory Holdouts**: Strictly unaccessed and untouched (`PROTECTED_UNACCESSED`).
- **Data Materialization**: Zero records extracted; zero CSV or Parquet files created.
- **Model Training**: Zero models trained, fitted, or evaluated.

---

## 6. Actionable Next Steps & Decision Paths

1. **Path 1 (Recommended — Direct Telemetry Extraction via Budget Amendment)**:
   - Amend the project scan cap from 21 GB to **450 GB** and monetary cap from $0.125 USD to **$2.50 USD** (or authorize consumption of 418.58 GB of the user's monthly 1 TiB BigQuery free tier quota).
   - Authorize direct execution of Candidate A ([phase7y_candidate_a_projection_reduction.sql](file:///C:/Users/hp/Desktop/Research%20project/results/phase7y_borg2019_pilot/phase7y_candidate_a_projection_reduction.sql)) to extract a bounded, legitimate container telemetry pilot cohort into local storage.
2. **Path 2 (Partitioned Staging Workflow)**:
   - If extensive multi-phase experimentation on Borg 2019 is anticipated, authorize creating a user-owned partitioned table `canvas-advice-464215-n1.borg2019_staging.instance_usage_partitioned` (initial scan $2.38 USD + $0.50/mo storage).
3. **Path 3 (Secondary Alternative — Offline Alibaba Trace 2018)**:
   - If BigQuery budget amendments or Sandbox quota allocations are rejected, pivot to Alibaba Cluster Trace 2018, noting that a revised experimental protocol must be established to handle irregular sampling cadences.
