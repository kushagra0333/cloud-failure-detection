# Phase 7Y: Candidate D Scientific Feasibility Review Report

**Project Root**: `C:\Users\hp\Desktop\Research project`  
**Review Type**: Design-Only Scientific Feasibility Review  
**Timestamp**: 2026-10-29T13:07:17.049774+00:00  
**Query File Under Review**: `phase7y_candidate_d_alternative_source.sql` (SHA-256: `0476eefca0076cf7ac18d571160b7060a8dbbee29951746eafda5955837dec14`)  
**Final Operational Verdict**: `CANDIDATE_D_REQUIRES_NEW_RESEARCH_DESIGN`  

---

## 1. Executive Summary & Final Verdict

This document presents an independent, design-only scientific feasibility review of **Candidate D** ([phase7y_candidate_d_alternative_source.sql](file:///C:/Users/hp/Desktop/Research%20project/results/phase7y_borg2019_pilot/phase7y_candidate_d_alternative_source.sql)), formulated during the Phase 7Y query optimization audit.

> [!IMPORTANT]
> **Final Operational Verdict: `CANDIDATE_D_REQUIRES_NEW_RESEARCH_DESIGN`**  
> While Candidate D successfully satisfies BigQuery numerical budget guardrails (2,520,117,308 bytes (~2.52 GB / 2.35 GiB), costing **$0.0143 USD**, which is only 11.4% of the $0.125 cap), **it cannot serve as an empirical validation or testbed for the frozen Phase 7S–7U failure prediction models**.
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
- **Prediction Timestamp**: Admission time $t_{\text{pred}} = t_{\text{submit}}$, prior to scheduling or execution.
- **Target Event**: Terminal outcome of the collection (`FAIL = 5` vs `FINISH = 6`).
- **Lead Time**: $\Delta t = t_{\text{terminal}} - t_{\text{submit}}$ represents the total job lifetime (spanning seconds to days), rather than imminent crash proximity.

### 3.2 Feature Modality Mismatch
In Candidate D, the only features available prior to execution are static submission attributes:
- Categorical attributes: `user`, `collection_name`, `parent_collection_id`.
- Scheduler metadata: `priority`, `scheduling_class`, `collection_type`, `scheduler`.
- **Telemetry Reality**: **`collection_events` contains zero runtime telemetry.** There are no CPU histograms, canonical memory readings, assigned memory limits, page cache, or CPI counters.

---

## 4. Temporal Leakage & Feature Availability Audit

| Feature | Source Table | Available at $t_{\text{pred}}$? | Leakage Risk & Assessment |
| :--- | :--- | :---: | :--- |
| `collection_id` | `collection_events` | Yes | High (Monotonically increasing ID; leaks global cluster epoch/time). |
| `parent_collection_id` | `collection_events` | Yes | High (Workflow group ID; causes co-occurrence leakage between train and val if parent appears in both). |
| `user` | `collection_events` | Yes | Critical (Categorical user ID; models memorize users with buggy code rather than learning system signals). |
| `collection_name` | `collection_events` | Yes | Critical (Job name string; periodic recurring cron jobs share identical names, creating direct target memorization). |
| `submit_time_us` | `collection_events` | Yes | Medium (Diurnal cycle / temporal shift). |
| `scheduling_class` | `collection_events` | Yes | Low (Static admission attribute). |
| `priority` | `collection_events` | Yes | Low (Static admission attribute). |
| `terminal_time_us` | `earliest_job_terminal` | **No** | Fatal (Future target timestamp; strictly excluded from features). |
| `terminal_type_code` | `earliest_job_terminal` | **No** | Fatal (Future target label; strictly excluded from features). |
| `terminal_status` | `earliest_job_terminal` | **No** | Fatal (Future target string; strictly excluded from features). |

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
- In Candidate D, any collection running past the trace cutoff ($86,400\text{ s}$) lacks a terminal event (`terminal_status IS NULL`).
- If un-terminated collections are discarded or assumed normal, the dataset suffers from severe survivorship and selection bias toward short-lived batch jobs.

### 5.3 Inapplicability of Proximity Horizons
- The governing Phase 7B label specification defines:
  - Class 2 (CRITICAL): Crash within $0 < \Delta t \le 300\text{ s}$ (5 min).
  - Class 1 (WARNING): Crash within $600 < \Delta t \le 1,500\text{ s}$ (10–25 min).
  - Class 0 (NORMAL): $\ge 25$ minutes survival without failure.
- **These horizons are completely meaningless for static job submission.** At admission, tasks have not even started executing. Defining a 5-minute pre-failure alert at submission conflates queue waiting time with imminent execution failure.

---

## 6. Validation Feasibility & Operational Gates Comparison

| Operational Gate (Phase 7B Protocol) | Target Threshold | Evaluatable on Candidate D? | Feasibility Analysis |
| :--- | :---: | :---: | :--- |
| **Episode Recall** | $\ge 70\%$ | **No** | Episodes represent consecutive pre-failure telemetry windows. Candidate D has no windows. |
| **Genuine Critical Recall** | $\ge 50\%$ | **No** | Requires 0–5 minute pre-failure window localization. Inapplicable at submission time. |
| **Pure-Normal Entity FAR** | $\le 10\%$ | **Yes (Adapted)** | Can evaluate false alarms on pure-normal parent collections. |
| **Normal-Window FPR** | $\le 5\%$ | **No** | Requires sliding 5-minute non-failure telemetry windows. Candidate D has no window structure. |

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
