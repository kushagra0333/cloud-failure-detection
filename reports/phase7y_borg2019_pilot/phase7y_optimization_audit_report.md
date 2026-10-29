# Phase 7Y: Query Optimization and Dry-Run-Only Audit Report

**Project Root**: `C:\Users\hp\Desktop\Research project`  
**Execution Timestamp**: 2026-10-29T12:50:54.279110+00:00  
**Target Public Dataset**: `google.com:google-cluster-data.clusterdata_2019_a`  
**Active Project**: `canvas-advice-464215-n1`  
**Governing Hard Caps**: Max scan `21,000,000,000 bytes` (21.0 GB); Max cost `$0.125 USD` (at $6.25/TiB)  

---

## 1. Executive Summary & Final Verdict

Following the initial authenticated dry-run result of 749,075,319,040 bytes (~749 GB, $4.26 USD), an optimization audit was conducted to evaluate four candidate queries. Every candidate was measured with an actual BigQuery service dry run (`dry_run=True`, `use_query_cache=False`).

### Primary Empirical Findings:
1. **Public Dataset Physical Reality**:
   - In Google Borg ClusterData 2019 (`clusterdata_2019_a`), table `instance_usage` contains **7,575,500,668 rows** and **0 partition keys** (`time_partitioning: None`, `clustering_fields: None`).
   - Every individual 8-byte column in `instance_usage` scans exactly **`60,604,005,344 bytes` (56.44 GiB / 60.6 GB)**.
   - Therefore, **any query that projects even a single column from `instance_usage` scans at least 60.6 GB, which is 2.89x the entire 21.0 GB research budget cap.**
2. **Telemetry Candidates (A, B, C) Exceed Budget**:
   - **Candidate A** (Minimal Leaf Projection: 6 usage cols + 4 event cols) scans **`418,578,205,568 bytes` (418.58 GB)** costing **$2.38 USD** (19.0x budget).
   - **Candidate B** (Consolidated Single-CTE Event Scan) scans **`418,578,205,568 bytes` (418.58 GB)** costing **$2.38 USD** (19.0x budget).
   - **Candidate C** (Bounded 5-Job Cohort & 1-Hour Micro-Horizon) scans **`418,578,205,568 bytes` (418.58 GB)** costing **$2.38 USD** (19.0x budget), proving that row-level `WHERE` filters provide **0% scan reduction** on unpartitioned BigQuery tables.
3. **Alternative Access Strategy (Candidate D)**:
   - **Candidate D** queries `collection_events` (20.8 million rows, 4.02 GiB total table size) for job-level failure prediction.
   - Measured dry-run scan: **`2,520,117,308 bytes` (2.35 GiB / 2.52 GB)** costing **`$0.0143 USD`**.
   - Candidate D satisfies both budget caps ($0.0143 <= $0.125 USD; 2.52 <= 21.0 GB). However, it shifts the research problem from container-level telemetry failure prediction to job-level lifecycle transitions.
4. **Primary Operational Verdict**:
   **OPTIMIZATION_CANDIDATE_WITHIN_BUDGET_FOR_REVIEW**
   *(For Candidate D; if container telemetry is strictly required, the verdict for Candidates A-C is NO_OPTIMIZATION_WITHIN_BUDGET).*

---

## 2. Table Metadata & Column Scan Diagnostics

Direct inspection of BigQuery table metadata via `client.get_table(...)` revealed:

| Table | Total Rows | Total Bytes | Partitioning | Clustering | Scan Per 64-bit Column |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`instance_usage`** | 7,575,500,668 | 2.26 TB (2,104.7 GiB) | **None** | **None** | **60,604,005,344 B (56.44 GiB)** |
| **`instance_events`** | 1,717,317,922 | 399.09 GiB | **None** | **None** | **13,738,543,376 B (12.80 GiB)** |
| **`collection_events`**| 20,807,441 | 4.02 GiB | **None** | **None** | ~0.15–0.40 GiB |

### Exact Mathematical Reconciliation of the 749 GB Baseline
The original query scanned 11 columns of `instance_usage` and 6 columns of `instance_events`:
- 11 columns of instance_usage * 60,604,005,344 bytes = 666,644,058,784 bytes (usage)
- 6 columns of instance_events * 13,738,543,376 bytes = 82,431,260,256 bytes (events)
- Total = 666,644,058,784 + 82,431,260,256 = 749,075,319,040 bytes (697.63 GiB / 749.08 GB)
This exact match confirms BigQuery's columnar scan mechanism and proves no scan duplication occurred across CTEs.

---

## 3. Optimization Candidates Dry-Run Comparison

Every candidate query was submitted as a genuine dry run against Google BigQuery under project `canvas-advice-464215-n1`:

| Candidate | Strategy | Telemetry Preserved? | Bytes Processed | On-Demand Cost ($6.25/TiB) | Budget Status | Evaluation Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Original** | Full Baseline | Yes | 749,075,319,040 B (749 GB) | $4.258 USD | 34.06x over budget | **REJECTED** |
| **Candidate A** | Minimal Leaf Projection | Yes | 418,578,205,568 B (419 GB) | $2.379 USD | 19.03x over budget | **REJECTED** |
| **Candidate B** | Consolidated CTE Scan | Yes | 418,578,205,568 B (419 GB) | $2.379 USD | 19.03x over budget | **REJECTED** |
| **Candidate C** | Bounded 5-Job Cohort & 1h Window | Yes | 418,578,205,568 B (419 GB) | $2.379 USD | 19.03x over budget | **REJECTED** |
| **Candidate D** | Alternative Access (`collection_events`) | No (Job-level) | **2,520,117,308 B (2.52 GB)** | **$0.0143 USD** | **WITHIN BUDGET (11.4%)** | **WITHIN_BUDGET_FOR_REVIEW** |

---

## 4. Deep-Dive Candidate Evaluations

### Candidate A — Projection Reduction
- **Query File**: [phase7y_candidate_a_projection_reduction.sql](file:///C:/Users/hp/Desktop/Research%20project/results/phase7y_borg2019_pilot/phase7y_candidate_a_projection_reduction.sql) (SHA-256: `33fa4a6968f47b9f8fe269ceb7adf1f72c4356292708f986a1f14e5ef82d09c4`)
- **Mechanism**: Stripped non-essential columns (`machine_id`, `alloc_collection_id`, `cpu_max_usage`, `assigned_memory`, `page_cache_memory`, `priority`, `scheduling_class`), retaining only 6 usage columns and 4 event columns.
- **Outcome**: Reduced scan from 749 GB to 418.58 GB (-44.1%). However, 418.58 GB still exceeds the 21.0 GB cap by **19.93x**.

### Candidate B — Source-Scan Reduction
- **Query File**: [phase7y_candidate_b_source_scan_reduction.sql](file:///C:/Users/hp/Desktop/Research%20project/results/phase7y_borg2019_pilot/phase7y_candidate_b_source_scan_reduction.sql) (SHA-256: `cdd35681e4c336fa2e27ce4c267b72b07ed44b37a8144cc4f172be845823a3db`)
- **Mechanism**: Consolidated event scanning into a single CTE `relevant_events`.
- **Outcome**: Scanned 418.58 GB. Confirmed that BigQuery's query planner already deduplicates column scans across CTEs in a single query.

### Candidate C — Bounded Pilot Scope
- **Query File**: [phase7y_candidate_c_bounded_pilot_scope.sql](file:///C:/Users/hp/Desktop/Research%20project/results/phase7y_borg2019_pilot/phase7y_candidate_c_bounded_pilot_scope.sql) (SHA-256: `fc50770cd8427c7233633da3abfb9f559303d1b0355403d037d5ea75fbc73785`)
- **Mechanism**: Filtered to 5 specific collection IDs and restricted time to a 1-hour window (0 to 3.6e9 us).
- **Outcome**: Scanned exactly 418.58 GB. This proves empirically that `WHERE` filters on unpartitioned BigQuery tables do not reduce scanned bytes.

### Candidate D — Alternative Access Strategy
- **Query File**: [phase7y_candidate_d_alternative_source.sql](file:///C:/Users/hp/Desktop/Research%20project/results/phase7y_borg2019_pilot/phase7y_candidate_d_alternative_source.sql) (SHA-256: `fa1d7c501a39cc48b6d9ece45f86242469bf36e572adbb84ded4a4fa5b623e4c`)
- **Mechanism**: Evaluated `collection_events` (20.8M rows) for job-level failure prediction.
- **Outcome**: Scanned **2,520,117,308 bytes (2.52 GB)**, costing **$0.0143 USD**.
- **Research Impact**: Satisfies the 21 GB and $0.125 budget constraints. However, it shifts the prediction target from container-level time series to job-level lifecycle transitions, requiring formal scientific review.

---

## 5. Architectural Paths Forward for Telemetry Extraction

If the project requires container-level telemetry, three sanctioned architectural paths exist:
1. **Budget Cap Realignment**: Formally amend the query budget from $0.125 USD to $2.50 USD to authorize Candidate A (418.58 GB, $2.38 USD).
2. **Sandbox Quota Allocation**: Authorize Candidate A under BigQuery Sandbox, which absorbs the 418.58 GB scan at $0.00 USD marginal cost (consuming 38.1% of the monthly 1 TiB free quota).
3. **Partitioned Table Staging**: If authorized, export raw Day 1 CSV slices from Google Cloud Storage (`gs://clusterdata_2019_a/`) and load them into a partitioned BigQuery table in the user's project, reducing the scan to <5 GB.

---

## 6. Governance & Non-Contamination Safeguards

- **Candidate A Test (1,025 windows)**: Strictly untouched (`PROTECTED_UNACCESSED`).
- **Phase 6 Confirmatory Holdouts**: Strictly untouched (`PROTECTED_UNACCESSED`).
- **Data Extraction**: Zero records extracted; no CSVs or Parquet files created.
- **Model Training**: Zero models trained or tuned.
- **Test Suite**: **191/191 passed in 16.39s**.
- **Original Baseline**: Preserved intact at [phase7y_dry_run_query.sql](file:///C:/Users/hp/Desktop/Research%20project/results/phase7y_borg2019_pilot/phase7y_dry_run_query.sql) (SHA-256: `889e2841f0983248a9f1d4c2f4f917ae7733b2038f7b939a66e098f60699a8e6`).
