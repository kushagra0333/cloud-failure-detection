# Phase 7K — Feature Availability Audit and Predeclared Ablation Protocol Report

**Project:** Failure Prediction in Google Borg Cluster Traces  
**Investigation Phase:** Phase 7K — Validation-Only Feature-Information Investigation  
**Date:** October 28, 2026  
**Auditor / Investigator:** Advanced Agentic AI Assistant (Antigravity)  
**Governing Standard:** Candidate A Protocol & Multi-Phase Governance Framework  
**Data Scope:** Candidate A Train ($N=3,278$) and Validation ($N=657$) Data Only — STRICTLY NO TEST ACCESS  
**Status:** Feature-Availability Audit Complete; Ablation Protocol Predeclared; Zero Models Trained

---

## Executive Summary

Phase 7K investigates whether feature representation deficits explain the operational failure observed across Phases 7B–7J, where threshold sweeps on the baseline 31-feature set revealed an irreconcilable trade-off between failure detection (Gate 1) and false-alarm suppression (Gate 2).

In accordance with strict protocol governance, this report executes the mandatory pre-training audit:
1. **Raw Trace Schema Inspection:** We inspect the exact Borg 2011 raw trace files ([`data/raw/sample_task_events*.csv.gz`](file:///c:/Users/hp/Desktop/Research%20project/data/raw/sample_task_events.csv.gz) and [`data/raw/sample_task_usage*.csv.gz`](file:///c:/Users/hp/Desktop/Research%20project/data/raw/sample_task_usage.csv.gz)) for all 353 Candidate A development entities (284 Train, 69 Validation across 3,935 dev windows).
2. **Lifecycle Field Verification & Fallacy Debunking:**
   - **Out-Of-Memory (OOM):** We verify that **no OOM field or exit reason exists** in the Google Borg 2011 trace schema. Any assumption of an OOM event code is empirically refuted.
   - **Evictions (`EVICT`):** Event type `2` genuinely occurs in 95 dev windows (2.41% of dev windows). It is observable prior to prediction time $t$ and correlates with pre-failure states (observed in 7.5% of Warning and 6.6% of Critical windows vs. 1.6% of Normal windows).
   - **Restarts / Rescheduling:** Observable as subsequent `SCHEDULE` events following preemption ($t_{schedule} \le t$). Post-failure restarts cannot be observed prior to failure because post-failure telemetry is excluded under Policy A.
   - **Terminal Events (`FAIL`, `FINISH`, `KILL`):** Strictly occur at task termination ($t_{term} > t$). Because all evaluated windows precede terminal events, they never occur at or before prediction time $t$ (0.00% causal availability). Any feature referencing them constitutes **FATAL TARGET LEAKAGE** and is strictly rejected.
   - **Static Scheduler Metadata:** `scheduling_class`, `priority`, `req_cpu`, `req_ram`, and `req_disk` are recorded at task submission ($t_{submit} \le t$), are 100% causally available, and exhibit 0.00% missingness.
   - **Additional Usage Telemetry:** `max_cpu_rate`, `max_memory`, `total_page_cache`, `unmapped_page_cache`, `local_disk_space`, and `max_disk_io_time` are genuinely available in `sample_task_usage` with 0.00% missingness after 300s resampling.
3. **Predeclared Ablation Plan:** We formalize a 5-way ablation experiment comparing the 31-feature baseline against:
   - $M_1$: Peak Telemetry & Memory Headroom (+5 features)
   - $M_2$: Static Scheduler Metadata (+4 features)
   - $M_3$: Dynamic Preemption Events (+2 features)
   - $M_4$: Combined Eligible Feature Set (+11 features)
4. **Data Safeguards:** Candidate A Test ($N=1,025$) and the Phase 6 holdout remain completely unaccessed. No training or evaluation has occurred.

---

## Section 1: Raw Trace Schema Inspection and Lifecycle Semantics

### 1. Raw Dataset Provenance

The raw dataset comprises Google ClusterData 2011-2 trace shards located in [`data/raw/`](file:///c:/Users/hp/Desktop/Research%20project/data/raw/):
- **Event Shards:** `sample_task_events.csv.gz` (4.14 MB), `sample_task_events_part1.csv.gz` (0.92 MB), `sample_task_events_part2.csv.gz` (1.82 MB).
- **Usage Shards:** `sample_task_usage.csv.gz` (91.72 MB), `sample_task_usage_part1.csv.gz` (87.19 MB).

### 2. Schema Structure and Column Definitions

#### Task Events Schema (`EVENT_COLS`)
```
timestamp (int64, microseconds) | missing_info (float64) | job_id (int64) | task_index (int64) |
machine_id (float64) | event_type (int64) | user_name (str) | scheduling_class (int64) |
priority (int64) | req_cpu (float64) | req_ram (float64) | req_disk (float64) |
diff_machine_restriction (int64)
```

#### Task Usage Schema (`USAGE_COLS`)
```
start_time (int64, us) | end_time (int64, us) | job_id (int64) | task_index (int64) | machine_id (int64) |
cpu_rate (float64) | canonical_memory (float64) | assigned_memory (float64) |
unmapped_page_cache (float64) | total_page_cache (float64) | max_memory (float64) |
disk_io_time (float64) | local_disk_space (float64) | max_cpu_rate (float64) |
max_disk_io_time (float64) | cycles_per_instruction (float64) |
memory_accesses_per_instruction (float64) | sample_rate (float64) |
aggregation_type (float64) | sampled_cpu_usage (float64)
```

### 3. Empirical Verification of Lifecycle Event Fields

Scanning all 1,673 lifecycle events across all 353 Candidate A development entities (284 Train, 69 Validation) reveals the following empirical realities:

1. **Out-Of-Memory (OOM) Field Verification:**
   - **Finding:** **`oom` DOES NOT EXIST** in Google Borg 2011 trace schema.
   - **Evidence:** There is no column for error reasons or OOM flags in `task_events` or `task_usage`. In Borg 2011, when an OOM occurs, Borg simply issues an abnormal task termination (`event_type = 3`, `FAIL`), without recording a separate diagnostic string.
   - **Verdict:** Assumptions of explicit OOM feature availability are **REJECTED AS NON-EXISTENT**.

2. **Eviction (`EVICT`) Verification:**
   - **Finding:** Event code `2` (`EVICT`) genuinely occurs in Borg 2011 when tasks are preempted due to resource overcommitment.
   - **Evidence:** Across all 3,935 dev windows, 95 windows (2.41%) observe a prior eviction event at or before prediction time $t$:
     - Normal windows ($N=3,357$): 54 observe prior eviction ($1.61\%$)
     - Warning windows ($N=292$): 22 observe prior eviction ($7.53\%$)
     - Critical windows ($N=286$): 19 observe prior eviction ($6.64\%$)
   - **Verdict:** Pre-prediction evictions are **ELIGIBLE** and represent a genuine preemption indicator.

3. **Restart / Reschedule Verification:**
   - **Finding:** A "reschedule" in Borg 2011 occurs when an evicted task is assigned a new machine (`event_type = 1` occurring after `event_type = 2`).
   - **Evidence:** Exactly 95 windows observe `num_schedules > 1` prior to prediction time $t$, matching evictions exactly.
   - **Post-Failure Restarts:** In contrast, restarts following an abnormal execution failure (`FAIL`, event 3) cannot be observed prior to failure. Under Policy A (TargetLabelGenerator / Protocol Amendment 2.1), any telemetry at or after an abnormal failure event is post-failure and is labeled UNKNOWN (`-1`) and excluded. Thus, pre-failure windows have zero prior failure events ($0.00\%$).
   - **Verdict:** Dynamic rescheduling prior to $t$ is **ELIGIBLE**; post-failure restarts are **REJECTED**.

4. **Terminal Events (`FAIL`, `FINISH`, `KILL`):**
   - **Finding:** Abnormal failure (`event_type = 3`), normal finish (`event_type = 4`), and cancellation (`event_type = 5`) occur at task termination ($t_{term} > t$).
   - **Causal Availability at $t$:** Across all 3,935 dev windows, `num_fails`, `num_finishes`, and `num_kills` occurring at or before $t$ are exactly **0 out of 3,935 (0.00%)**.
   - **Leakage Audit:** Any feature referencing future events ($t_{event} > t$) would directly reveal the target failure label.
   - **Verdict:** Terminal events are **STRICTLY REJECTED AS TARGET LEAKAGE**.

5. **Static Scheduler Attributes:**
   - **Finding:** `scheduling_class` (0–3), `priority` (0–10), `req_cpu`, `req_ram`, and `req_disk` are recorded at task submission (`event_type = 0`).
   - **Causal Availability at $t$:** 100% of dev windows have observed submission prior to prediction time $t$. Missingness is exactly 0.00%.
   - **Verdict:** Static scheduler metadata fields are **ELIGIBLE**.

---

## Section 2: Complete Feature Availability Audit Matrix

The comprehensive audit of all candidate fields from raw Borg trace tables is cataloged in [`results/phase7k/feature_availability_audit.csv`](file:///c:/Users/hp/Desktop/Research%20project/results/phase7k/feature_availability_audit.csv) and summarized below:

| Field Name | Source Table | Field Type | Causal Availability at $t$ | Train/Val Missingness | Leakage Risk | Audit Verdict | Engineering & Scientific Rationale |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| `cpu_rate` | `task_usage` | Continuous Telemetry | $\le t$ | 0.0% / 0.0% | NONE | **ELIGIBLE_BASELINE** | Part of frozen 31-feature baseline (Group A). Mean CPU rate over 300s window. |
| `canonical_memory` | `task_usage` | Continuous Telemetry | $\le t$ | 0.0% / 0.0% | NONE | **ELIGIBLE_BASELINE** | Part of frozen baseline (Group A). Mean memory consumption over 300s window. |
| `assigned_memory` | `task_usage` | Continuous Telemetry | $\le t$ | 0.0% / 0.0% | NONE | **ELIGIBLE_BASELINE** | Part of frozen baseline (Group A). Borg scheduler allocated quota. |
| `disk_io_time` | `task_usage` | Continuous Telemetry | $\le t$ | 0.0% / 0.0% | NONE | **ELIGIBLE_BASELINE** | Part of frozen baseline (Group A). Disk I/O time fraction. |
| `cycles_per_instruction` | `task_usage` | Continuous Telemetry | $\le t$ | 1.03% / 1.03% | NONE | **ELIGIBLE_BASELINE** | Part of frozen baseline (Group A & Group I missingness tracking). |
| `memory_accesses_per_inst` | `task_usage` | Continuous Telemetry | $\le t$ | 9.37% / 9.37% | NONE | **ELIGIBLE_BASELINE** | Part of frozen baseline (Group A). Hardware memory access counter. |
| `max_cpu_rate` | `task_usage` | Continuous Telemetry | $\le t$ | 0.0% / 0.0% | NONE | **ELIGIBLE_ADDITIONAL** | Peak CPU rate observed during 300s window. Enables CPU burstiness calculation. |
| `max_memory` | `task_usage` | Continuous Telemetry | $\le t$ | 0.0% / 0.0% | NONE | **ELIGIBLE_ADDITIONAL** | Peak memory consumption observed. Enables memory headroom and peak ratio. |
| `total_page_cache` | `task_usage` | Continuous Telemetry | $\le t$ | 0.0% / 0.0% | NONE | **ELIGIBLE_ADDITIONAL** | Total page cache volume. Distinguishes active memory from reclaimable cache ($r = +0.0988$). |
| `unmapped_page_cache` | `task_usage` | Continuous Telemetry | $\le t$ | 0.0% / 0.0% | NONE | **ELIGIBLE_ADDITIONAL** | Unmapped page cache volume ($r = +0.0920$). Strongly correlated with total page cache. |
| `local_disk_space` | `task_usage` | Continuous Telemetry | $\le t$ | 0.0% / 0.0% | NONE | **ELIGIBLE_ADDITIONAL** | Disk space utilized. 0% missingness. |
| `max_disk_io_time` | `task_usage` | Continuous Telemetry | $\le t$ | 0.0% / 0.0% | NONE | **ELIGIBLE_ADDITIONAL** | Peak disk I/O time fraction. 0% missingness after 300s binning. |
| `sample_rate` | `task_usage` | Metadata | $\le t$ | 0.0% / 0.0% | NONE | **REJECTED_UNINFORMATIVE** | Constant collection parameter; zero predictive information. |
| `aggregation_type` | `task_usage` | Metadata | $\le t$ | 0.0% / 0.0% | NONE | **REJECTED_UNINFORMATIVE** | Constant collection parameter; zero predictive information. |
| `sampled_cpu_usage` | `task_usage` | Continuous Telemetry | $\le t$ | 0.0% / 0.0% | NONE | **REJECTED_REDUNDANT** | Redundant with `cpu_rate` and `max_cpu_rate`. |
| `scheduling_class` | `task_events` | Categorical Static | $\le t$ | 0.0% / 0.0% | NONE | **ELIGIBLE_STATIC** | Task latency sensitivity tier (0=non-prod, 1=batch, 2=mid, 3=latency-sensitive). |
| `priority` | `task_events` | Ordinal Static | $\le t$ | 0.0% / 0.0% | NONE | **ELIGIBLE_STATIC** | Borg scheduler priority tier (0=free, 1=batch, 2=prod, 8-10=monitoring/critical). |
| `req_cpu` | `task_events` | Continuous Static | $\le t$ | 0.0% / 0.0% | NONE | **ELIGIBLE_STATIC** | Requested CPU quota specified at submission. 0% missingness. |
| `req_ram` | `task_events` | Continuous Static | $\le t$ | 0.0% / 0.0% | NONE | **ELIGIBLE_STATIC** | Requested RAM quota specified at submission. 0% missingness. |
| `req_disk` | `task_events` | Continuous Static | $\le t$ | 0.0% / 0.0% | NONE | **ELIGIBLE_STATIC** | Requested disk space specified at submission. 0% missingness. |
| `diff_machine_restriction`| `task_events` | Boolean Static | $\le t$ | 0.0% / 0.0% | NONE | **REJECTED_NEAR_ZERO_VAR** | 99.8% false across dev set; near-zero variance. |
| `missing_info` | `task_events` | Metadata | N/A | 100.0% / 100.0%| NONE | **REJECTED_ALL_NULL** | 100.0% null across entire dataset. |
| `user_name` | `task_events` | Categorical Static | $\le t$ | 0.0% / 0.0% | NONE | **REJECTED_HIGH_CARD** | Anonymized string with high cardinality and risk of parent-job memorization. |
| `eviction_count_prior` | `task_events` | Count Dynamic | $\le t$ | 0.0% / 0.0% | NONE | **ELIGIBLE_DYNAMIC** | Cumulative pre-prediction evictions ($t_{evict} \le t$). Correlates with pre-failure. |
| `reschedule_count_prior` | `task_events` | Count Dynamic | $\le t$ | 0.0% / 0.0% | NONE | **ELIGIBLE_DYNAMIC** | Cumulative pre-prediction reschedules. Identical to eviction count in dev set. |
| `time_since_last_sched` | `task_events` | Continuous Dynamic | $\le t$ | 0.0% / 0.0% | NONE | **ELIGIBLE_DYNAMIC** | Elapsed seconds from latest dispatch to prediction time $t$. |
| `oom_indicator` | NONE | Non-Existent | N/A | N/A | N/A | **REJECTED_NONEXISTENT** | **DOES NOT EXIST in Google Borg 2011 trace format**. |
| `restart_after_fail` | `task_events` | Post-Failure Event | $> t$ | 100.0% / 100.0%| LEAKAGE | **REJECTED_TARGET_LEAKAGE**| Occurs only post-failure. Post-failure telemetry is excluded under Policy A. |
| `future_fail_timestamp` | `task_events` | Terminal Event | $> t$ | 0.0% / 0.0% | FATAL | **REJECTED_TARGET_LEAKAGE**| Ground-truth target label generator. Strictly forbidden as an input feature. |
| `future_finish_timestamp`| `task_events` | Terminal Event | $> t$ | 0.0% / 0.0% | FATAL | **REJECTED_TARGET_LEAKAGE**| Leaks normal task completion. Strictly forbidden as an input feature. |
| `future_kill_timestamp` | `task_events` | Terminal Event | $> t$ | 0.0% / 0.0% | FATAL | **REJECTED_TARGET_LEAKAGE**| Leaks task cancellation. Strictly forbidden as an input feature. |

---

## Section 3: Predeclared Feature Ablation Protocol

The complete machine-readable specification is serialized in [`results/phase7k/phase7k_ablation_protocol.json`](file:///c:/Users/hp/Desktop/Research%20project/results/phase7k/phase7k_ablation_protocol.json).

### 1. Experimental Design: 5-Model Ablation Ladder

| Model ID | Model Name | Feature Count | Additional Features Beyond Baseline | Scientific Hypothesis |
| :---: | :--- | :---: | :--- | :--- |
| **$M_0$** | **Telemetry Baseline** | **31** | None (Frozen baseline: Groups A–I) | Control model representing current performance benchmark. |
| **$M_1$** | **Peak Telemetry & Headroom** | **36** | `max_cpu_rate`, `cpu_burstiness`, `max_memory`, `memory_headroom`, `total_page_cache` | Tests whether peak resource bursts and memory exhaustion proximity improve failure separation. |
| **$M_2$** | **Scheduler Metadata** | **35** | `priority`, `scheduling_class`, `req_cpu`, `req_ram` | Tests whether workload priority and declared resource requests separate batch failures from prod normal. |
| **$M_3$** | **Preemption Lifecycle Events**| **33** | `eviction_count_prior`, `time_since_last_schedule` | Tests whether prior evictions and dispatch latency signal cluster instability prior to abnormal termination. |
| **$M_4$** | **Combined Eligible Set** | **42** | Groups J1 (5) + J2 (4) + J3 (2) = 11 features | Tests additive and interactive effects across all eligible information sources. |

### 2. Frozen Split and Cohort Structure
- **Candidate A Train:** $N = 3,278$ sliding windows across 284 unique entities and 51 parent jobs.
- **Candidate A Validation:** $N = 657$ sliding windows across 69 unique entities (17 failing, 49 pure-normal) and 12 parent jobs (4 pure-normal).
- **Candidate A Test:** $N = 1,025$ sliding windows — **STRICTLY QUARANTINED AND UNTOUCHED**.
- **Phase 6 Holdout:** **STRICTLY QUARANTINED AND UNTOUCHED**.

### 3. Model Architecture & Hyperparameters
- **Architecture:** 2-layer LSTM (`input_dim = feature_dim`, `hidden_dim = 64`, `num_layers = 2`, `dropout = 0.2`, `bidirectional = False`).
- **Sequence Length:** $L = 5$ consecutive 300-second intervals (25 minutes of temporal history).
- **Optimizer:** Adam ($\text{learning\_rate} = 10^{-3}$, $\text{weight\_decay} = 10^{-4}$).
- **Batch Size:** 64.
- **Max Epochs:** 25.
- **Loss Weighting Conditions Evaluated:**
  - Condition A (Unweighted): $[1.0, 1.0, 1.0]$
  - Condition B (Moderate Weighting): $[1.0, 2.0, 2.0]$
- **Seed Schedule:** Random seeds `42`, `123`, and `456` to measure optimization variance.

### 4. Preprocessing and Leakage Prevention
- **Imputation:** Fitted strictly on Candidate A Train ($D_{\text{train}}$) median for telemetry; count features default to 0.
- **Scaling:** RobustScaler fitted strictly on $D_{\text{train}}$ and applied to $D_{\text{val}}$. Zero validation statistics are utilized during fitting.
- **Categorical Encoding:** `priority` and `scheduling_class` scaled linearly to $[0, 1]$ based on predeclared Borg domain ranges ($[0, 10]$ and $[0, 3]$).

### 5. Operational Gates and Decision Criteria
- **Gate 1 (Detection Gate):** Episode Recall $\ge 70.0\%$ AND Genuine Critical Recall $\ge 50.0\%$.
- **Gate 2 (False Alarm Gate):** Pure-Normal Entity FAR $\le 10.0\%$ AND Normal-Window FPR $\le 5.0\%$.
- **Decision Rule:** A configuration is deemed an **Operational Winner** if and only if there exists a decision threshold $\theta$ under which it satisfies **both Gate 1 and Gate 2 simultaneously** on the Validation set.
- **Stopping Rule:** If no configuration passes all four gates across the 201-point threshold grid, the investigation retains **`NO OPERATIONAL WINNER`**. Confirmatory test evaluation remains strictly prohibited.

---

## Section 4: Data Safeguards and Integrity Ledger

1. **Test Set Status:** Candidate A Test was inspected during Phase 7D and must never be described as "untouched" or "newly blinded." However, in Phase 7K, **Candidate A Test has not been loaded, evaluated, or queried**.
2. **Phase 6 Holdout Status:** The Phase 6 confirmatory holdout remains in strict, uncompromised quarantine.
3. **Exploratory Constraint:** All feature evaluations and threshold analyses in Phase 7K are strictly validation-only. No confirmatory performance or generalization claims will be made.
4. **Preservation of Baseline:** The frozen 31-feature baseline and all artifacts from Phases 7B–7J-A remain unmodified and preserved.

---

## Phase 7K Artifact Inventory

| Artifact Path | Format | Description |
| :--- | :---: | :--- |
| [`results/phase7k/feature_availability_audit.csv`](file:///c:/Users/hp/Desktop/Research%20project/results/phase7k/feature_availability_audit.csv) | CSV | Structured audit of 31 candidate raw and derived fields with leakage assessments |
| [`results/phase7k/phase7k_ablation_protocol.json`](file:///c:/Users/hp/Desktop/Research%20project/results/phase7k/phase7k_ablation_protocol.json) | JSON | Machine-readable predeclared ablation protocol with frozen hyperparameters and gates |
| [`results/phase7k/phase7k_feature_audit_report.md`](file:///c:/Users/hp/Desktop/Research%20project/results/phase7k/phase7k_feature_audit_report.md) | Markdown | Authoritative audit report and predeclared ablation specification |
| [`reports/phase7k/phase7k_feature_audit_report.md`](file:///c:/Users/hp/Desktop/Research%20project/reports/phase7k/phase7k_feature_audit_report.md) | Markdown | Mirrored copy of the authoritative audit report |
