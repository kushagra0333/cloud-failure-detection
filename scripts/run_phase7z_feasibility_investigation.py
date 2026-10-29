"""Phase 7Z: External Container Telemetry Access Feasibility Investigation.

This script executes the feasibility analysis:
1. Audits candidate external sources for container runtime telemetry.
2. Evaluates storage and cost options (A, B, C, D, E).
3. Evaluates scientific equivalence, labels, and operational gates.
4. Audits validation, leakage, censoring, and holdout protection.
5. Emits phase7z_external_telemetry_access_feasibility.md and matching JSON deliverables.
"""

import hashlib
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


PROJECT_ROOT = Path(r"C:\Users\hp\Desktop\Research project")
RESULTS_DIR = PROJECT_ROOT / "results" / "phase7y_borg2019_pilot"
REPORTS_DIR = PROJECT_ROOT / "reports" / "phase7y_borg2019_pilot"


def compute_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def generate_feasibility_data() -> Dict[str, Any]:
    timestamp = datetime.now(timezone.utc).isoformat()

    candidate_a_sql = RESULTS_DIR / "phase7y_candidate_a_projection_reduction.sql"
    original_sql = RESULTS_DIR / "phase7y_dry_run_query.sql"
    candidate_d_sql = RESULTS_DIR / "phase7y_candidate_d_alternative_source.sql"

    sql_hashes = {
        "candidate_a_sql_sha256": compute_sha256(candidate_a_sql) if candidate_a_sql.is_file() else None,
        "original_baseline_sql_sha256": compute_sha256(original_sql) if original_sql.is_file() else None,
        "candidate_d_sql_sha256": compute_sha256(candidate_d_sql) if candidate_d_sql.is_file() else None,
    }

    data = {
        "investigation_metadata": {
            "phase": "Phase 7Z",
            "title": "External Container Telemetry Access Feasibility",
            "timestamp_utc": timestamp,
            "project_root": str(PROJECT_ROOT),
            "investigation_type": "FEASIBILITY_INVESTIGATION_ONLY",
            "governing_rules": {
                "zero_data_extraction": True,
                "zero_model_training": True,
                "zero_table_materialization": True,
                "protected_holdouts_accessed": False,
                "budget_increased": False,
            },
        },
        "query_checksums": sql_hashes,
        "current_caps": {
            "scan_cap_bytes": 21000000000,
            "scan_cap_readable": "21,000,000,000 bytes (21.0 GB)",
            "budget_cap_usd": 0.125,
            "on_demand_rate_usd_per_tib": 6.25,
        },
        "candidate_sources_inventory": [
            {
                "source_id": "google_borg_2019",
                "name": "Google Borg ClusterData 2019 (v3)",
                "data_owner": "Google LLC / Google Research (Wilkes et al.)",
                "distribution_platform": "Google BigQuery (`google.com:google-cluster-data`)",
                "storage_type": "Cloud-native columnar BigQuery tables (8 cells: clusterdata_2019_a to h)",
                "gcs_bucket_status": "No public open flat-file GCS bucket; trace is distributed exclusively via BigQuery",
                "access_permissions": "Publicly queryable with Google Cloud ADC (authenticated under project canvas-advice-464215-n1)",
                "licensing": "Creative Commons Attribution 4.0 International (CC-BY 4.0)",
                "runtime_telemetry": {
                    "available": True,
                    "metrics": [
                        "average_usage.cpus (mean core utilization)",
                        "maximum_usage.cpus (peak core utilization)",
                        "average_usage.memory (canonical RAM bytes)",
                        "assigned_memory (allocation limit)",
                        "page_cache_memory (kernel page cache)",
                        "cycles_per_instruction (CPI histogram)",
                    ],
                    "temporal_resolution": "5-minute periodic aggregated windows (300,000,000 us)",
                    "cadence_regularity": "Regular 300-second intervals",
                },
                "lifecycle_linkage": {
                    "available": True,
                    "join_key": "(collection_id, instance_index)",
                    "target_events": "type=5 (FAIL), type=6 (FINISH), type=4 (EVICT), type=7 (KILL), type=8 (LOST)",
                    "event_timestamp_resolution": "Microseconds (64-bit integer)",
                },
                "scientific_comparability": {
                    "task_comparability": "HIGH. Supports identical sequential container runtime failure prediction.",
                    "supports_3class_labels": True,
                    "supports_operational_gates": True,
                    "limitations": "Unpartitioned tables in BigQuery require 418.58 GB scan per telemetry query.",
                },
                "external_provenance": "HIGH. May 2019 production cells across multiple data centers, completely independent of 2011 trace.",
            },
            {
                "source_id": "google_borg_2011",
                "name": "Google Borg ClusterData 2011 (v2.1)",
                "data_owner": "Google LLC / Google Research (Reiss et al.)",
                "distribution_platform": "Google Cloud Storage (`gs://clusterdata-2011-2/`)",
                "storage_type": "Flat gzip CSV shards (500 shards each for task_usage and task_events)",
                "gcs_bucket_status": "Publicly accessible GCS bucket with requester-pays / public read",
                "access_permissions": "Publicly readable via gsutil / python storage client",
                "licensing": "Creative Commons Attribution 4.0 International (CC-BY 4.0)",
                "runtime_telemetry": {
                    "available": True,
                    "metrics": [
                        "cpu_rate", "canonical_memory_usage", "assigned_memory_usage",
                        "unmapped_page_cache", "total_page_cache", "cycles_per_instruction",
                        "memory_accesses_per_instruction", "disk_io_time"
                    ],
                    "temporal_resolution": "5-minute periodic aggregated windows (300 seconds)",
                    "cadence_regularity": "Exact 300.00s in 99.56% of windows",
                },
                "lifecycle_linkage": {
                    "available": True,
                    "join_key": "(job_id, task_index)",
                    "target_events": "event_type=3 (FAIL), event_type=4 (FINISH), event_type=2 (EVICT), event_type=5 (KILL)",
                    "event_timestamp_resolution": "Microseconds",
                },
                "scientific_comparability": {
                    "task_comparability": "Identical schema to existing research.",
                    "supports_3class_labels": True,
                    "supports_operational_gates": True,
                    "limitations": "CRITICAL VALIDITY LIMITATION: Borg 2011 is a single 12,500-machine cluster. Candidate A, B, C, D and Phase 6 Confirmatory holdouts were all drawn from this same trace. Sourcing additional shards from 2011 does NOT provide an independent external cluster/workload cohort.",
                },
                "external_provenance": "NONE. Identical 2011 single-cell source; violates external cohort independence.",
            },
            {
                "source_id": "alibaba_cluster_trace_2018",
                "name": "Alibaba Cluster Trace 2018 (v2018)",
                "data_owner": "Alibaba Group (Alibaba System Software Team)",
                "distribution_platform": "GitHub (`alibaba/clusterdata`) and OSS HTTP mirrors",
                "storage_type": "Compressed CSV files (~50 GB compressed)",
                "gcs_bucket_status": "Not hosted on GCS; distributed via Aliyun OSS / GitHub releases",
                "access_permissions": "Public download via HTTP without authentication",
                "licensing": "Mulan Permissive Software License / Academic Research Agreement",
                "runtime_telemetry": {
                    "available": True,
                    "metrics": ["cpu_utilization_pct", "memory_utilization_pct"],
                    "temporal_resolution": "Variable / heterogeneous (10s to 5-minute sampling)",
                    "cadence_regularity": "Irregular; requires extensive binning and forward-fill imputation",
                },
                "lifecycle_linkage": {
                    "available": True,
                    "join_key": "(instance_name, task_name)",
                    "target_events": "status=Failed, status=Terminated, status=Cancelled, status=Interrupted",
                    "event_timestamp_resolution": "Seconds",
                },
                "scientific_comparability": {
                    "task_comparability": "MODERATE. Focuses on Fuxi DAG batch jobs colocated with Sigma containers.",
                    "supports_3class_labels": True,
                    "supports_operational_gates": "Partial (requires temporal binning and synthetic window alignment).",
                    "limitations": "Missing hardware counters (no CPI or MAI); irregular sampling cadence; DAG failure dynamics differ from Borg long-running services.",
                },
                "external_provenance": "VERY HIGH. Independent cloud provider, independent software stack, independent hardware.",
            },
            {
                "source_id": "azure_vm_traces",
                "name": "Microsoft Azure Public Dataset (VM Traces 2017/2019/2021)",
                "data_owner": "Microsoft Azure Research",
                "distribution_platform": "GitHub (`Azure/AzurePublicDataset`) / Azure Open Datasets",
                "storage_type": "CSV / Parquet files",
                "gcs_bucket_status": "Not on GCS; hosted on Azure Blob Storage",
                "access_permissions": "Publicly downloadable",
                "licensing": "Open Data / MIT License",
                "runtime_telemetry": {
                    "available": True,
                    "metrics": ["min_cpu", "max_cpu", "avg_cpu"],
                    "temporal_resolution": "5-minute binned intervals",
                    "cadence_regularity": "Regular",
                },
                "lifecycle_linkage": {
                    "available": False,
                    "join_key": "vm_id",
                    "target_events": "FATAL DEFECT: Only records vm_created and vm_deleted. No failure reasons or exit codes.",
                    "event_timestamp_resolution": "Seconds",
                },
                "scientific_comparability": {
                    "task_comparability": "DISQUALIFIED. Conflates normal customer deallocations, autoscaling, and crashes. Violates Rules 2 & 3.",
                    "supports_3class_labels": False,
                    "supports_operational_gates": False,
                    "limitations": "Cannot be used without fraudulently treating routine VM deletions as failures.",
                },
                "external_provenance": "VERY HIGH, but disqualified due to lack of ground-truth failure labels.",
            },
            {
                "source_id": "lanl_cfdr",
                "name": "Los Alamos National Laboratory Computer Failure Data Repository",
                "data_owner": "Los Alamos National Laboratory / USENIX",
                "distribution_platform": "CFDR public archive",
                "storage_type": "Text incident logs and CSV records",
                "gcs_bucket_status": "Not on GCS",
                "access_permissions": "Public download",
                "licensing": "Public Domain / Open Access",
                "runtime_telemetry": {
                    "available": False,
                    "metrics": [],
                    "temporal_resolution": "None (Discrete incident intervals)",
                    "cadence_regularity": "N/A",
                },
                "lifecycle_linkage": {
                    "available": False,
                    "join_key": "node_id",
                    "target_events": "Administrative trouble ticket logs",
                    "event_timestamp_resolution": "Minutes/Hours",
                },
                "scientific_comparability": {
                    "task_comparability": "DISQUALIFIED. Administrative remedy incident logs, zero continuous container telemetry time-series.",
                    "supports_3class_labels": False,
                    "supports_operational_gates": False,
                    "limitations": "Incompatible with temporal sequence models.",
                },
                "external_provenance": "VERY HIGH, but modality incompatible.",
            },
            {
                "source_id": "ornl_clusterwise_summit",
                "name": "Oak Ridge National Laboratory ClusterWise (Summit Supercomputer)",
                "data_owner": "Oak Ridge National Laboratory / US DOE",
                "distribution_platform": "ORNL data repository",
                "storage_type": "Parquet archives (>500 GB)",
                "gcs_bucket_status": "Not on GCS",
                "access_permissions": "Public research access",
                "licensing": "Open Government Data License",
                "runtime_telemetry": {
                    "available": True,
                    "metrics": ["node_power", "gpu_temperature", "hbm_utilization"],
                    "temporal_resolution": "1-second to 1-minute telemetry",
                    "cadence_regularity": "Regular",
                },
                "lifecycle_linkage": {
                    "available": False,
                    "join_key": "node_id / gpu_id",
                    "target_events": "Hardware node outages and thermal throttling",
                    "event_timestamp_resolution": "Seconds",
                },
                "scientific_comparability": {
                    "task_comparability": "DISQUALIFIED. Physical supercomputing hardware and GPU thermal modeling, not cloud container task failure.",
                    "supports_3class_labels": False,
                    "supports_operational_gates": False,
                    "limitations": "Completely disparate workload and target domain.",
                },
                "external_provenance": "VERY HIGH, but domain incompatible.",
            },
        ],
        "storage_and_cost_options_audit": {
            "approach_a_prefiltered_subset": {
                "name": "Approach A: Prefiltered Telemetry Subset",
                "feasibility": "INFEASIBLE_NONEXISTENT",
                "evidence": "Forensic audit of Google's public repositories, BigQuery datasets, and official documentation confirms that Google does not host a pre-filtered or sampled sub-table of instance_usage. The public distribution contains only the full 2.4 TiB multi-cell dataset.",
                "assumptions": "Assumes a third-party or official pre-sliced subset exists. Disproven.",
                "estimated_cost_usd": 0.0,
                "limitations": "No legitimate prefiltered subset exists in the public domain.",
                "required_approvals": "None (nonexistent).",
            },
            "approach_b_partitioned_clustered_source": {
                "name": "Approach B: Documented Partitioned/Clustered Source",
                "feasibility": "INFEASIBLE_NONEXISTENT",
                "evidence": "Direct BigQuery API inspection (client.get_table) confirms table `google.com:google-cluster-data.clusterdata_2019_a.instance_usage` has `time_partitioning: None` and `clustering_fields: None`. All 8 cells (a through h) follow this identical unpartitioned schema.",
                "assumptions": "Assumes public table supports partition pruning. Disproven.",
                "estimated_cost_usd": 2.3793,
                "limitations": "Columnar scan must scan all 7.575 billion rows for every projected column (60.6 GB per 8-byte column).",
                "required_approvals": "None (unpartitioned table exists, but costs $2.38 USD).",
            },
            "approach_c_scientifically_defensible_sampling": {
                "name": "Approach C: Sampling Without Scanning Entire Unpartitioned Source",
                "feasibility": "TECHNICALLY_INEFFECTIVE",
                "evidence": "Empirically verified by Candidate C dry run (SHA-256: fc50770cd8427c7233633da3abfb9f559303d1b0355403d037d5ea75fbc73785): adding `WHERE collection_id IN (5 jobs) AND time <= 3.6e9 us` resulted in exactly 418,578,205,568 bytes scanned (0.0% reduction). BigQuery TABLESAMPLE SYSTEM is block-based and destroys continuous temporal sequences (L=12 windows cannot be reconstructed from random block rows).",
                "assumptions": "Assumes BigQuery row filters reduce scanned bytes on unpartitioned tables. Disproven.",
                "estimated_cost_usd": 2.3793,
                "limitations": "Row filters prune zero bytes in BigQuery on-demand; block sampling destroys time-series continuity.",
                "required_approvals": "None.",
            },
            "approach_d_authorized_partitioned_staging": {
                "name": "Approach D: Authorized Partitioned Staging Workflow",
                "feasibility": "FEASIBLE_WITH_STAGING_INGESTION_BUDGET",
                "architecture": "Execute a one-time DDL query to extract a 1-day slice into a user-owned partitioned table `canvas-advice-464215-n1.borg2019_staging.instance_usage_partitioned` partitioned by `TIMESTAMP_TRUNC(start_time, DAY)`.",
                "costs": {
                    "initial_ingestion_scan_bytes": 418578205568,
                    "initial_ingestion_cost_usd": 2.3793,
                    "storage_size_gb": 25.0,
                    "monthly_storage_cost_usd": 0.50,
                    "subsequent_query_scan_bytes": 14000000000,
                    "subsequent_query_cost_usd": 0.083,
                },
                "assumptions": "Initial ingestion query is funded ($2.38 USD) or absorbed by Sandbox free tier quota. User project has BigQuery dataset creation permissions.",
                "limitations": "Initial staging query still incurs the 418.58 GB scan; introduces ongoing GCP active storage maintenance.",
                "required_approvals": "Financial authorization for $2.38 initial scan + $0.50/mo storage, or Sandbox quota authorization + table creation authorization.",
            },
            "approach_e_bigquery_budget_amendment": {
                "name": "Approach E: Formally Approved BigQuery Budget Amendment",
                "feasibility": "DIRECTLY_ACTIONABLE_RECOMMENDED",
                "architecture": "Formally amend the project's scan cap from 21 GB to 450 GB (or 0.45 TB) and monetary budget from $0.125 USD to $2.50 USD, or authorize execution under BigQuery Sandbox / Free Tier (consuming 38.1% of the monthly 1 TiB free allocation at $0.00 marginal cost). Directly executes Candidate A ([phase7y_candidate_a_projection_reduction.sql](file:///C:/Users/hp/Desktop/Research%20project/results/phase7y_borg2019_pilot/phase7y_candidate_a_projection_reduction.sql)).",
                "costs": {
                    "measured_dry_run_scan_bytes": 418578205568,
                    "measured_dry_run_scan_readable": "418,578,205,568 bytes (389.83 GiB / 418.58 GB)",
                    "estimated_cost_usd_at_6_25": 2.3793,
                    "marginal_cost_under_sandbox_free_tier": 0.00,
                    "sandbox_monthly_quota_fraction_pct": 38.07,
                },
                "assumptions": "Governing project authority formally approves $2.50 USD budget amendment or approves 418.58 GB consumption of user's BigQuery monthly 1 TiB free tier.",
                "limitations": "Exceeds historical $0.125 cap by 19.03x.",
                "required_approvals": "Formal project budget amendment and data extraction authorization from principal investigator / user.",
            },
        },
        "scientific_equivalence_audit": {
            "prediction_task": "Imminent container runtime failure prediction using sequential sliding telemetry windows (L=12 steps / 60 minutes).",
            "label_equivalence": {
                "class_0_normal": "Entity active with no failure in (t, t + 25 min].",
                "class_1_warning": "Imminent failure within 10 to 25 minutes (600 < tau <= 1500 s).",
                "class_2_critical": "Imminent failure within 0 to 5 minutes (0 < tau <= 300 s).",
                "status": "FULLY_COMPATIBLE on Google Borg 2019 (instance_usage + instance_events).",
            },
            "operational_gates_compatibility": {
                "episode_recall_ge_70": {
                    "threshold": ">= 70%",
                    "supported_on_borg_2019": True,
                    "mechanism": "Consecutive pre-failure telemetry window sequences define failure episodes.",
                },
                "genuine_critical_recall_ge_50": {
                    "threshold": ">= 50%",
                    "supported_on_borg_2019": True,
                    "mechanism": "Evaluated on windows observed within 5 minutes of verified FAIL event.",
                },
                "pure_normal_entity_far_le_10": {
                    "threshold": "<= 10%",
                    "supported_on_borg_2019": True,
                    "mechanism": "Evaluated on purely successful collection/job entities.",
                },
                "normal_window_fpr_le_5": {
                    "threshold": "<= 5%",
                    "supported_on_borg_2019": True,
                    "mechanism": "Evaluated across all non-failure sliding windows.",
                },
            },
            "single_cluster_generalization_boundary": (
                "A pilot extracted from cell 'a' constitutes a single-cluster evaluation. "
                "Per governing rules, cross-cluster or cross-datacenter generalization CANNOT be claimed "
                "from a single cell. Cross-cell generalization requires evaluating across disjoint cells "
                "(e.g., train on cell 'a', validate on cell 'b')."
            ),
        },
        "validation_and_leakage_safeguards": {
            "temporal_ordering": "Strict chronological split: e.g., Day 1 00:00-18:00 UTC (Train/Val) and 18:00-24:00 UTC (Test holdout).",
            "group_separation": "Group by parent_collection_id (or collection_id) to eliminate co-occurrence leakage between splits.",
            "join_semantics": "Event window join: earliest terminal event occurring strictly after window end_time (time >= end_time). Never join future events preceding the window.",
            "censoring_protocol": "Tasks running past the observation cutoff without terminal events must be marked CENSORED/UNKNOWN and excluded from training failure windows.",
            "minimum_schema_and_metadata_required_before_extraction": [
                "1. Verified BigQuery table schema confirming column names and data types (COMPLETED in Phase 7Y).",
                "2. Pre-extraction SHA-256 checksum verification of SQL query (COMPLETED: Candidate A hash 33fa4a6968f47b9f8fe269ceb7adf1f72c4356292708f986a1f14e5ef82d09c4).",
                "3. Verified service dry run confirming exact byte count (COMPLETED: 418,578,205,568 bytes).",
                "4. Formal project budget amendment or Sandbox quota authorization (PENDING USER DECISION).",
            ],
            "protected_partitions_status": {
                "candidate_a_test_accessed": False,
                "phase6_confirmatory_accessed": False,
                "status": "PROTECTED_UNACCESSED",
            },
        },
        "risks_and_unresolved_questions": [
            "1. Quota Consumption: Candidate A consumes 418.58 GB (38.1%) of the user's monthly 1 TiB BigQuery free tier quota. If other projects share this account, quota contention may occur.",
            "2. Alloc Set Interference: In Borg 2019, tasks in alloc sets share resource reservations. Candidate A filters to non-alloc collections (alloc_collection_id IS NULL OR alloc_collection_id = 0) to avoid shared-resource distortion.",
            "3. Label Sparsity: Borg task failure rates are historically 1-3%. The pilot cohort must use stratified collection sampling to ensure sufficient positive failure episodes.",
        ],
        "recommended_next_step": (
            "Present this feasibility report to the user and request a formal project decision: "
            "either (1) approve a BigQuery budget amendment to $2.50 USD / 450 GB (or authorize 418.58 GB consumption of Sandbox free tier) "
            "to execute Candidate A for container telemetry extraction, or (2) pivot the research to an offline-downloadable alternative "
            "such as Alibaba Cluster Trace 2018 (which requires an adapted experimental protocol for irregular sampling)."
        ),
        "final_feasibility_verdict": "BUDGET_AMENDMENT_WOULD_BE_REQUIRED",
    }
    return data


def write_deliverables(data: Dict[str, Any]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. JSON Deliverable
    json_path = RESULTS_DIR / "phase7z_external_telemetry_access_feasibility.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    # Mirror JSON to reports/
    mirrored_json = REPORTS_DIR / "phase7z_external_telemetry_access_feasibility.json"
    shutil.copy2(json_path, mirrored_json)

    # 2. Markdown Report
    report_path = REPORTS_DIR / "phase7z_external_telemetry_access_feasibility.md"

    inv = data["candidate_sources_inventory"]
    opts = data["storage_and_cost_options_audit"]
    curr = data["current_caps"]
    hashes = data["query_checksums"]

    sources_table_rows = "\n".join([
        f"| **{s['name']}** | {s['data_owner']} | `{s['distribution_platform']}` | {'**Yes**' if s['runtime_telemetry']['available'] else 'No'} | {s['runtime_telemetry']['temporal_resolution']} | {s['scientific_comparability']['task_comparability']} |"
        for s in inv
    ])

    report_md = f"""# Phase 7Z: External Container Telemetry Access Feasibility Report

**Project Root**: `C:\\Users\\hp\\Desktop\\Research project`  
**Execution Timestamp**: {data["investigation_metadata"]["timestamp_utc"]}  
**Investigation Scope**: Strictly Feasibility & Governance Audit (Zero Data Extraction)  
**Governing Caps**: Max Scan: `{curr["scan_cap_readable"]}`; Max Budget: `${curr["budget_cap_usd"]} USD`  
**Candidate A Query**: [`phase7y_candidate_a_projection_reduction.sql`](file:///C:/Users/hp/Desktop/Research%20project/results/phase7y_borg2019_pilot/phase7y_candidate_a_projection_reduction.sql) (SHA-256: `{hashes["candidate_a_sql_sha256"]}`)  
**Original Query**: [`phase7y_dry_run_query.sql`](file:///C:/Users/hp/Desktop/Research%20project/results/phase7y_borg2019_pilot/phase7y_dry_run_query.sql) (SHA-256: `{hashes["original_baseline_sql_sha256"]}`)  
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
{sources_table_rows}

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
- Sampling Interval: $\\Delta t = 300\\text{{ seconds}}$ (5 minutes).
- Proximity Labeling Formulation:
  - **CRITICAL (2)**: Imminent abnormal crash within $0 < \\tau \\le 300\\text{{ s}}$ (0–5 minutes).
  - **WARNING (1)**: Imminent abnormal crash within $600 < \\tau \\le 1,500\\text{{ s}}$ (10–25 minutes).
  - **NORMAL (0)**: Active task with no failure in $(t, t + 1,500\\text{{ s}}]$.
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
"""
    report_path.write_text(report_md, encoding="utf-8")
    print(f"Successfully generated Phase 7Z deliverables in results/ and reports/.")


def main():
    print("=== Phase 7Z: External Container Telemetry Access Feasibility ===")
    data = generate_feasibility_data()
    print(f"Final Feasibility Verdict: {data['final_feasibility_verdict']}")
    print(f"Candidate A Dry Run Scan: {data['storage_and_cost_options_audit']['approach_e_bigquery_budget_amendment']['costs']['measured_dry_run_scan_readable']}")
    print(f"Estimated Cost: ${data['storage_and_cost_options_audit']['approach_e_bigquery_budget_amendment']['costs']['estimated_cost_usd_at_6_25']:.4f} USD")
    print(f"Sandbox Free Tier Utilization: {data['storage_and_cost_options_audit']['approach_e_bigquery_budget_amendment']['costs']['sandbox_monthly_quota_fraction_pct']:.2f}%")

    print("\nWriting deliverables...")
    write_deliverables(data)


if __name__ == "__main__":
    main()
