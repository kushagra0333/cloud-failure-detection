"""Phase 7Y: Candidate D Scientific Feasibility Review.

This script executes the design-only scientific feasibility review of Candidate D:
1. Formulates the prediction task, observation unit, and feature availability.
2. Audits temporal leakage risks and label feasibility.
3. Evaluates validation feasibility against existing Phase 7B operational gates.
4. Assesses research continuity against frozen Phase 7S-7U telemetry models.
5. Compiles budget and access status from verified dry-run evidence.
6. Emits structured report and machine-readable JSON summary in reports/ and results/.
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


def generate_feasibility_review() -> Dict[str, Any]:
    timestamp = datetime.now(timezone.utc).isoformat()
    candidate_d_file = RESULTS_DIR / "phase7y_candidate_d_alternative_source.sql"
    sql_sha256 = compute_sha256(candidate_d_file) if candidate_d_file.is_file() else None

    review_data = {
        "review_timestamp_utc": timestamp,
        "review_type": "DESIGN_ONLY_SCIENTIFIC_FEASIBILITY_REVIEW",
        "candidate_d_sql_file": "phase7y_candidate_d_alternative_source.sql",
        "candidate_d_sha256": sql_sha256,
        "source_dataset": "google.com:google-cluster-data.clusterdata_2019_a",
        "source_table": "collection_events",
        "dry_run_budget_status": {
            "scanned_bytes": 2520117308,
            "scanned_bytes_readable": "2,520,117,308 bytes (~2.52 GB / 2.35 GiB)",
            "hard_scan_cap_bytes": 21000000000,
            "scan_cap_utilization_pct": 12.0,
            "estimated_cost_usd_at_6_25": 0.0143,
            "budget_cap_usd": 0.125,
            "budget_cap_utilization_pct": 11.4,
            "meets_numerical_budget_caps": True,
            "authorization_note": "Meeting numerical dry-run caps does NOT authorize data extraction or imply scientific equivalence.",
        },
        "task_definition": {
            "unit_of_observation": "Job/Collection submission event (collection_id at submit_time_us)",
            "historical_unit_of_observation": "Fixed 5-minute sliding telemetry windows of individual container tasks (task_id, window_index)",
            "observation_shift": "Shift from continuous dynamic container time series to discrete static job submission metadata",
            "prediction_timestamp": "t_pred = submit_time_us (Job admission time)",
            "lead_time_definition": "delta_t = t_terminal - t_submit (Entire job execution lifetime, ranging from seconds to weeks)",
            "features_available_at_admission": [
                "parent_collection_id (workflow grouping)",
                "user (obfuscated user ID)",
                "collection_name (obfuscated job template name)",
                "priority (Borg priority tier)",
                "scheduling_class (latency sensitivity: 0-3)",
                "collection_type (0=JOB, 1=ALLOC)",
                "scheduler (Borg scheduler queue/algorithm)",
            ],
            "telemetry_features_available": "NONE. collection_events contains zero CPU, memory, CPI, page cache, or time-series usage metrics.",
        },
        "temporal_leakage_audit": {
            "leakage_table": [
                {
                    "feature": "collection_id",
                    "source": "collection_events",
                    "available_at_prediction_time": True,
                    "leakage_risk": "High (Monotonically increasing ID; leaks global cluster epoch/time).",
                },
                {
                    "feature": "parent_collection_id",
                    "source": "collection_events",
                    "available_at_prediction_time": True,
                    "leakage_risk": "High (Workflow group ID; causes co-occurrence leakage between train and val if parent appears in both).",
                },
                {
                    "feature": "user",
                    "source": "collection_events",
                    "available_at_prediction_time": True,
                    "leakage_risk": "Critical (Categorical user ID; models memorize users with buggy code rather than learning system signals).",
                },
                {
                    "feature": "collection_name",
                    "source": "collection_events",
                    "available_at_prediction_time": True,
                    "leakage_risk": "Critical (Job name string; periodic recurring cron jobs share identical names, creating direct target memorization).",
                },
                {
                    "feature": "submit_time_us",
                    "source": "collection_events",
                    "available_at_prediction_time": True,
                    "leakage_risk": "Medium (Diurnal cycle / temporal shift).",
                },
                {
                    "feature": "scheduling_class",
                    "source": "collection_events",
                    "available_at_prediction_time": True,
                    "leakage_risk": "Low (Static admission attribute).",
                },
                {
                    "feature": "priority",
                    "source": "collection_events",
                    "available_at_prediction_time": True,
                    "leakage_risk": "Low (Static admission attribute).",
                },
                {
                    "feature": "terminal_time_us",
                    "source": "earliest_job_terminal",
                    "available_at_prediction_time": False,
                    "leakage_risk": "Fatal (Future target timestamp; strictly excluded from features).",
                },
                {
                    "feature": "terminal_type_code",
                    "source": "earliest_job_terminal",
                    "available_at_prediction_time": False,
                    "leakage_risk": "Fatal (Future target label; strictly excluded from features).",
                },
                {
                    "feature": "terminal_status",
                    "source": "earliest_job_terminal",
                    "available_at_prediction_time": False,
                    "leakage_risk": "Fatal (Future target string; strictly excluded from features).",
                },
            ],
            "leakage_summary": "If restricted to admission-time fields, no future target leakage occurs. However, severe categorical memorization leakage exists for 'user' and 'collection_name' across repeated periodic jobs.",
        },
        "label_feasibility": {
            "failure_target": "FAIL (event code 5): Abnormal job termination where one or more non-retryable tasks failed.",
            "normal_anchor": "FINISH (event code 6): Clean successful completion of all collection tasks.",
            "competing_risks": [
                "KILL (event code 7): Explicit user or parent workflow cancellation.",
                "EVICT (event code 4): Collection-level preemption by scheduler.",
                "LOST (event code 8): Host/network disconnect.",
            ],
            "censoring_and_incomplete_jobs": (
                "Long-running service collections (daemons, databases, web serving tiers) run indefinitely "
                "and never log FINISH or FAIL within Day 1. Truncating at Day 1 results in right-censoring "
                "for all service jobs, severely biasing the sample toward short batch jobs."
            ),
            "applicability_of_original_lead_time_horizons": (
                "INAPPLICABLE. The original 5-minute critical (Class 2) and 10-25 minute warning (Class 1) "
                "horizons are defined relative to sliding 5-minute telemetry windows during active execution. "
                "At job submission, the job has not even been scheduled or assigned to machines. "
                "A job may wait in queue for 15 minutes and execute for 4 hours. A 5-minute failure proximity "
                "label cannot be evaluated at admission time without continuous runtime telemetry."
            ),
        },
        "validation_feasibility": {
            "temporal_split_feasible": True,
            "group_split_feasible": "Feasible by grouping on parent_collection_id or user.",
            "operational_gate_comparability": {
                "Episode_Recall_ge_70": {
                    "meaningful_on_candidate_d": False,
                    "reason": "Episodes were defined as sequences of failure-bound 5-minute telemetry windows. Candidate D has no telemetry windows.",
                },
                "Genuine_Critical_Recall_ge_50": {
                    "reason": "Critical windows were defined as 0-5 minutes prior to container failure. Candidate D has no critical window observations.",
                    "meaningful_on_candidate_d": False,
                },
                "Pure_Normal_Entity_FAR_le_10": {
                    "meaningful_on_candidate_d": True,
                    "reason": "Can be adapted as False Alarm Rate on pure-normal parent collections.",
                },
                "Normal_Window_FPR_le_5": {
                    "meaningful_on_candidate_d": False,
                    "reason": "Normal windows were sliding 5-minute non-failure intervals. Candidate D has no window structure.",
                },
            },
            "comparability_summary": "Candidate D cannot evaluate three of the four governing Phase 7B operational gates. Model outputs cannot be compared against Phase 7B-7U results.",
        },
        "research_continuity_assessment": {
            "continuity_classification": "C. REQUIRES_SEPARATE_RESEARCH_QUESTION_AND_EXPERIMENTAL_PROTOCOL",
            "justification": [
                "1. Modality Mismatch: Phase 7S-7U models (S0, S2, Hybrid, Fuzzy) strictly require input tensors of shape (batch, seq_len=12, features=6) containing standardized CPU and memory telemetry. Candidate D contains zero telemetry.",
                "2. Model Incompatibility: Frozen neural and fuzzy models developed in Phases 7N-7U cannot be tested or evaluated on Candidate D without retraining an entirely different tabular architecture.",
                "3. Problem Re-framing: Candidate D changes the core scientific question from 'Can continuous runtime container telemetry predict imminent software crashes?' to 'Can static submission metadata predict job failure proneness at admission?'.",
            ],
        },
        "governance_and_holdouts": {
            "candidate_a_test_accessed": False,
            "phase6_confirmatory_holdouts_accessed": False,
            "data_extraction_performed": False,
            "models_trained_or_tuned": False,
            "tables_created": False,
        },
        "recommended_decision": "CANDIDATE_D_REQUIRES_NEW_RESEARCH_DESIGN",
        "final_operational_verdict": "CANDIDATE_D_REQUIRES_NEW_RESEARCH_DESIGN",
    }
    return review_data


def write_review_deliverables(review_data: Dict[str, Any]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. JSON Summary
    json_path = RESULTS_DIR / "phase7y_candidate_d_scientific_feasibility_review.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(review_data, f, indent=2)

    # 2. Markdown Report
    report_path = REPORTS_DIR / "phase7y_candidate_d_scientific_feasibility_review.md"
    b_stat = review_data["dry_run_budget_status"]
    t_def = review_data["task_definition"]
    l_aud = review_data["temporal_leakage_audit"]
    l_fea = review_data["label_feasibility"]
    v_fea = review_data["validation_feasibility"]
    r_con = review_data["research_continuity_assessment"]

    leakage_table_header = (
        "| Feature | Source Table | Available at $t_{\\text{pred}}$? | Leakage Risk & Assessment |\n"
        "| :--- | :--- | :---: | :--- |\n"
    )
    leakage_rows = leakage_table_header + "\n".join([
        f"| `{row['feature']}` | `{row['source']}` | {'Yes' if row['available_at_prediction_time'] else '**No**'} | {row['leakage_risk']} |"
        for row in l_aud["leakage_table"]
    ])

    report_md = f"""# Phase 7Y: Candidate D Scientific Feasibility Review Report

**Project Root**: `C:\\Users\\hp\\Desktop\\Research project`  
**Review Type**: Design-Only Scientific Feasibility Review  
**Timestamp**: {review_data["review_timestamp_utc"]}  
**Query File Under Review**: `phase7y_candidate_d_alternative_source.sql` (SHA-256: `{review_data["candidate_d_sha256"]}`)  
**Final Operational Verdict**: `CANDIDATE_D_REQUIRES_NEW_RESEARCH_DESIGN`  

---

## 1. Executive Summary & Final Verdict

This document presents an independent, design-only scientific feasibility review of **Candidate D** ([phase7y_candidate_d_alternative_source.sql](file:///C:/Users/hp/Desktop/Research%20project/results/phase7y_borg2019_pilot/phase7y_candidate_d_alternative_source.sql)), formulated during the Phase 7Y query optimization audit.

> [!IMPORTANT]
> **Final Operational Verdict: `CANDIDATE_D_REQUIRES_NEW_RESEARCH_DESIGN`**  
> While Candidate D successfully satisfies BigQuery numerical budget guardrails ({b_stat["scanned_bytes_readable"]}, costing **$0.0143 USD**, which is only 11.4% of the $0.125 cap), **it cannot serve as an empirical validation or testbed for the frozen Phase 7S–7U failure prediction models**.
> 
> Candidate D fundamentally shifts the scientific task from **runtime container-level telemetry anomaly detection** to **static job-admission failure proneness classification**. It contains **zero runtime telemetry** (no CPU or memory streams), cannot evaluate the Phase 7B sliding-window operational gates, and cannot take the inputs required by the frozen models (S0, S2, Hybrid, Fuzzy). If pursued, it must be governed under an entirely separate research charter and experimental protocol.

---

## 2. Budget and Access Compliance Status

Based on verified BigQuery service dry-run evidence under project `canvas-advice-464215-n1`:
- **Actual Bytes Scanned**: **`2,520,117,308 bytes` (~2.52 GB / 2.35 GiB)**.
- **Estimated On-Demand Cost**: **`$0.0143 USD`** (at $6.25/TiB).
- **Approved Scan Cap**: `21,000,000,000 bytes` (Candidate D utilizes **12.0%** of cap).
- **Approved Cost Cap**: `$0.125 USD` (Candidate D utilizes **11.4%** of budget).
- **Compliance Finding**: Candidate D cleanly satisfies numerical budget guardrails.
- **Access Boundary Principle**: A successful dry run and low monetary cost **do not authorize data extraction**. The project's mandate for external validation of Phase 7 models does not automatically extend to a different problem formulation.

---

## 3. Definition of the Candidate D Prediction Task

### 3.1 Observation Unit and Temporal Alignment
- **Candidate D Unit of Observation**: A **collection submission event** (`collection_id` at `submit_time_us`).
- **Historical Unit of Observation (Phases 3–7U)**: Fixed **5-minute sliding telemetry windows** of individual container task attempts (`task_id`, `window_index`).
- **Prediction Timestamp**: Admission time $t_{{\\text{{pred}}}} = t_{{\\text{{submit}}}}$, prior to scheduling or execution.
- **Target Event**: Terminal outcome of the collection (`FAIL = 5` vs `FINISH = 6`).
- **Lead Time**: $\\Delta t = t_{{\\text{{terminal}}}} - t_{{\\text{{submit}}}}$ represents the total job lifetime (spanning seconds to days), rather than imminent crash proximity.

### 3.2 Feature Modality Mismatch
In Candidate D, the only features available prior to execution are static submission attributes:
- Categorical attributes: `user`, `collection_name`, `parent_collection_id`.
- Scheduler metadata: `priority`, `scheduling_class`, `collection_type`, `scheduler`.
- **Telemetry Reality**: **`collection_events` contains zero runtime telemetry.** There are no CPU histograms, canonical memory readings, assigned memory limits, page cache, or CPI counters.

---

## 4. Temporal Leakage & Feature Availability Audit

{leakage_rows}

### Critical Leakage Findings:
1. **Target Leakage Safeguard**: Features available at submission time do not leak `terminal_time_us` or `terminal_type_code`, provided future fields are strictly isolated in labels.
2. **Categorical Identity Memorization**:
   - In Google cluster traces, automated workflows and cron jobs repeatedly submit identical tasks with the same `user` and `collection_name`.
   - A tabular model will rapidly memorize that specific `user` strings or `collection_name` templates correlate with failure, creating severe out-of-distribution generalization failure when applied to unseen users or workloads.

---

## 5. Label Feasibility Audit

### 5.1 Lifecycle Event Taxonomy in `collection_events`
- `type = 0` (`SUBMIT`): Collection submitted to Borg queue.
- `type = 1` (`QUEUE`), `type = 2` (`ENABLE`), `type = 3` (`SCHEDULE`): Queueing and activation stages.
- `type = 5` (`FAIL`): Abnormal termination of collection (**Failure Target**).
- `type = 6` (`FINISH`): Normal completion (**Normal Anchor**).
- `type = 7` (`KILL`): User or workflow manager cancellation (**Competing Event**).
- `type = 4` (`EVICT`): Scheduler priority preemption (**Competing Event**).
- `type = 8` (`LOST`): Infrastructure node disconnect (**Sensitivity Target**).

### 5.2 Right-Censoring and Long-Running Service Workloads
- Long-running service daemons (web servers, storage servers, pipeline controllers) execute continuously and rarely finish within 24 hours.
- In Candidate D, any collection running past the trace cutoff ($86,400\\text{{ s}}$) lacks a terminal event (`terminal_status IS NULL`).
- If un-terminated collections are discarded or assumed normal, the dataset suffers from severe survivorship and selection bias toward short-lived batch jobs.

### 5.3 Inapplicability of Proximity Horizons
- The governing Phase 7B label specification defines:
  - Class 2 (CRITICAL): Crash within $0 < \\Delta t \\le 300\\text{{ s}}$ (5 min).
  - Class 1 (WARNING): Crash within $600 < \\Delta t \\le 1,500\\text{{ s}}$ (10–25 min).
  - Class 0 (NORMAL): $\\ge 25$ minutes survival without failure.
- **These horizons are completely meaningless for static job submission.** At admission, tasks have not even started executing. Defining a 5-minute pre-failure alert at submission conflates queue waiting time with imminent execution failure.

---

## 6. Validation Feasibility & Operational Gates Comparison

| Operational Gate (Phase 7B Protocol) | Target Threshold | Evaluatable on Candidate D? | Feasibility Analysis |
| :--- | :---: | :---: | :--- |
| **Episode Recall** | $\\ge 70\\%$ | **No** | Episodes represent consecutive pre-failure telemetry windows. Candidate D has no windows. |
| **Genuine Critical Recall** | $\\ge 50\\%$ | **No** | Requires 0–5 minute pre-failure window localization. Inapplicable at submission time. |
| **Pure-Normal Entity FAR** | $\\le 10\\%$ | **Yes (Adapted)** | Can evaluate false alarms on pure-normal parent collections. |
| **Normal-Window FPR** | $\\le 5\\%$ | **No** | Requires sliding 5-minute non-failure telemetry windows. Candidate D has no window structure. |

Candidate D cannot evaluate three of the four governing operational gates without completely redefining the metrics.

---

## 7. Research Continuity & Scientific Classification

Candidate D is classified under:
> **Category C: REQUIRES_SEPARATE_RESEARCH_QUESTION_AND_EXPERIMENTAL_PROTOCOL**

### Scientific Justification:
1. **Modality Incompatibility**: Frozen Phase 7S–7U models (S0 absolute LSTM, S2 relative LSTM, Phase 7T hybrid, Phase 7U fuzzy layer) accept temporal sequence tensors of shape `(batch, 12, 6)`. Candidate D provides a flat tabular row of submission metadata. The frozen models physically cannot ingest Candidate D data.
2. **Distinct Research Questions**:
   - *Phase 7 Research Question*: Can dynamic multivariate resource telemetry detect imminent software failure episodes with sufficient lead time to trigger automated migration?
   - *Candidate D Question*: Can static scheduler metadata predict whether a newly submitted batch job will ultimately fail or succeed?
3. **Non-Comparability**: Results on Candidate D cannot validate, invalidate, or establish external generalizability for the frozen Phase 7 models.

---

## 8. Governance, Non-Contamination & Next Steps

- **Protected Partitions**: Candidate A Test (1,025 windows) and Phase 6 Confirmatory holdouts were **strictly untouched**.
- **Data Extraction**: Zero records extracted; no CSVs or Parquet files created.
- **Model Training**: Zero models trained, tuned, or fitted.
- **Recommendation**:
  - If the project's goal is to test the **frozen telemetry failure prediction models**, Candidate D must be rejected, and the project must pursue authorized access to container telemetry (e.g. BigQuery Sandbox allocation for Candidate A, budget realignment, or partitioned table staging).
  - If the project desires to open an entirely new line of inquiry into **pre-admission job failure classification**, Candidate D may be submitted for a separate research proposal and protocol design.
"""
    report_path.write_text(report_md, encoding="utf-8")

    # Mirror JSON to reports/
    shutil.copy2(json_path, REPORTS_DIR / "phase7y_candidate_d_scientific_feasibility_review.json")
    print(f"Successfully generated feasibility review deliverables in results/ and reports/.")


def main():
    print("=== Step 1: Generating Scientific Feasibility Review for Candidate D ===")
    review_data = generate_feasibility_review()
    print(f"Verdict: {review_data['final_operational_verdict']}")
    print(f"Unit of observation: {review_data['task_definition']['unit_of_observation']}")
    print(f"Telemetry available: {review_data['task_definition']['telemetry_features_available']}")
    print(f"Research continuity: {review_data['research_continuity_assessment']['continuity_classification']}")

    print("\n=== Step 2: Writing Review Deliverables ===")
    write_review_deliverables(review_data)


if __name__ == "__main__":
    main()
