# Phase 7Y Continuation: Comprehensive SQL Safety, Task Lifecycle, and Censoring Audit

**Project Root**: `C:\Users\hp\Desktop\Research project`  
**Audit Target File**: `phase7y_dry_run_query.sql`  
**SQL SHA-256 Checksum**: `889e2841f0983248a9f1d4c2f4f917ae7733b2038f7b939a66e098f60699a8e6`  
**Target Public Dataset**: `google.com:google-cluster-data.clusterdata_2019_a`  
**Execution Timestamp**: 2026-10-29T11:43:31.092394+00:00  

---

## 1. Schema and Protobuf v3 Field Name Audit

Every table reference, field projection, nested record, and timestamp unit in `phase7y_dry_run_query.sql` was verified against the official Google Borg 2019 trace specification ([clusterdata_trace_format_v3.proto](https://github.com/google/cluster-data/blob/master/clusterdata_trace_format_v3.proto)):

### 1.1 Table Identifier Alignment
- **Usage Table**: `google.com:google-cluster-data.clusterdata_2019_a.instance_usage` (Single-cell sharded table, isolating Cell A).
- **Event Table**: `google.com:google-cluster-data.clusterdata_2019_a.instance_events` (Single-cell lifecycle event log).

### 1.2 Telemetry Record Field Mapping
In Google ClusterData 2019 (v3), resource metrics are stored within nested `Resources` protocol buffer messages (`message Resources { optional float cpus = 1; optional float memory = 2; }`):
- `u.average_usage.cpus`: **VALID**. Queries average normalized CPU core utilization.
- `u.maximum_usage.cpus`: **VALID**. Queries peak CPU utilization.
- `u.average_usage.memory`: **VALID**. Queries average canonical memory usage.
- `u.assigned_memory`: **VALID**. Direct scalar double representing assigned memory limit.
- `u.page_cache_memory`: **VALID**. Direct scalar double representing kernel page cache allocation.
- **Legacy Error Avoidance**: Avoids invalid Phase 7X assumed paths (`cpu_usage.average`, `memory_usage.canonical_memory_usage`).

### 1.3 Microsecond Timestamp Consistency
- **Units**: All timestamps in Borg 2019 are 64-bit integers expressing elapsed **microseconds** ($\mu\text{s}$) from trace epoch.
- **5-Minute Telemetry Step**: $5 \times 60 \times 10^6 = 300,000,000 \ \mu\text{s}$.
- **24-Hour Observation Boundary**: $24 \times 3600 \times 10^6 = 86,400,000,000 \ \mu\text{s}$.
- **25-Hour Event Follow-up Horizon**: $25 \times 3600 \times 10^6 = 90,000,000,000 \ \mu\text{s}$ (providing a 3,600-second buffer past Day 1).

---

## 2. Task Lifecycle and Join Cardinality Audit

### 2.1 Complete Event Type Taxonomy
In `clusterdata_trace_format_v3.proto`, lifecycle transitions are enumerated as:
- `0: SUBMIT` — Task or alloc instance submitted to system.
- `1: QUEUE` — Placed in queue; not eligible for immediate scheduling.
- `2: ENABLE` — Transitioned to eligible for batch scheduling.
- `3: SCHEDULE` — Assigned to a physical machine and started running (**Active filter anchor**).
- `4: EVICT` — Forcibly descheduled due to priority preemption or resource overcommitment (**Competing event**).
- `5: FAIL` — Task terminated abnormally due to internal failure or software crash (**Primary Target Ground Truth**).
- `6: FINISH` — Task completed execution successfully and normally (**Normal Anchor Ground Truth**).
- `7: KILL` — Explicitly cancelled by user or workflow manager (**Competing event**).
- `8: LOST` — Host machine disconnected or unreachable (**Secondary Sensitivity Target Ground Truth**).
- `9: UPDATE_PENDING` — Configuration update pending.
- `10: UPDATE_RUNNING` — In-place dynamic resource reconfiguration applied while task continues running (**Non-terminal**).

### 2.2 EVICT Handling (Non-Definitive Termination)
- **Technical Reality**: `EVICT` (type = 4) deschedules a task to accommodate higher-priority work. Borg may reschedule the evicted task later under the same `instance_index`.
- **Policy**: `EVICT` is **not** treated as a definitive permanent task termination. In `phase7y_dry_run_query.sql`, `EVICT` is captured as `terminal_status = 'EVICT'` so that downstream labeling can treat it as a competing termination event, preventing preemption artifacts from being misclassified as software crashes.

### 2.3 FINISH and KILL Handling (Clean Completion vs Cancellation)
- **`FINISH` (type = 6)**: Represents clean task completion. An observation window followed by `FINISH` is classified as **Normal ($y=0$)** without requiring an artificial 25-minute survival wait.
- **`KILL` (type = 7)**: Represents cancellation (e.g., pipeline cancelled by developer or parent workflow finished early). It is treated as an unobservable competing risk. Windows followed by `KILL` within 25 minutes are **excluded from the primary failure target** to prevent training models to predict human cancellations.

### 2.4 Resolution of the `earliest_terminal` Join Flaw
The previous query draft in `query_plan.sql` calculated:
```sql
earliest_terminal AS (
    SELECT ..., ROW_NUMBER() OVER (PARTITION BY collection_id, instance_index ORDER BY terminal_time ASC) AS rn
    FROM terminal_events
)
... LEFT JOIN earliest_terminal et ON ... AND et.rn = 1 AND et.terminal_time >= u.end_time
```
**Defect Identified**: Pre-computing `rn = 1` globally across the entire 25-hour trace meant that if an instance had an early retry failure at $t = 5,000$ and was rescheduled from $t = 10,000$ to $t = 30,000$, all observation windows after $t = 5,000$ joined to `et.rn = 1` which failed `terminal_time >= u.end_time`. Consequently, all subsequent observation windows evaluated `et` to `NULL`, missing subsequent failures or clean finishes!

**Correction Implemented in `phase7y_dry_run_query.sql`**:
```sql
usage_with_terminal_candidates AS (
    SELECT ...,
        ROW_NUMBER() OVER (
            PARTITION BY u.collection_id, u.instance_index, u.start_time
            ORDER BY te.terminal_time ASC
        ) AS rn
    FROM `instance_usage` u
    ... LEFT JOIN terminal_events te
        ON u.collection_id = te.collection_id
        AND u.instance_index = te.instance_index
        AND te.terminal_time >= u.end_time
)
... WHERE rn = 1 OR rn IS NULL
```
**Benefits**:
1. Anchored temporally to $u.\text{end\_time}$: For each usage window, `rn = 1` selects the **earliest terminal event strictly occurring at or after that window**.
2. **Zero Cartesian Multiplication**: Partitioning by `(collection_id, instance_index, start_time)` and filtering `rn = 1` guarantees exactly one output row per valid usage window.
3. Multi-attempt tasks correctly identify their respective terminal outcomes.

---

## 3. Future-Label Boundaries and Right-Censoring Specification

The forward lead time is defined strictly as:
$$\Delta t = t_{\text{event}} - t_{\text{cutoff}} \quad \text{where } t_{\text{cutoff}} = u.\text{end\_time}$$

### 3.1 Three-State Proximity Schema
- **Class 2 (CRITICAL)**: $0 < \Delta t \le 300\times 10^6 \ \mu\text{s}$ ($0 < \Delta t \le 5$ minutes).
  - Imminent execution failure requiring urgent preemptive intervention.
- **Class 1 (WARNING)**: $600\times 10^6 < \Delta t \le 1,500\times 10^6 \ \mu\text{s}$ ($10 < \Delta t \le 25$ minutes).
  - Early warning window allowing orderly checkpointing or migration.
- **Ambiguous Buffer (EXCLUDED)**: $300\times 10^6 < \Delta t \le 600\times 10^6 \ \mu\text{s}$ ($5 < \Delta t \le 10$ minutes).
  - Strictly excluded from training and evaluation to eliminate label transition noise.
- **Class 0 (NORMAL)**: $\Delta t > 1,500\times 10^6 \ \mu\text{s}$ ($>25$ minutes) with complete follow-up OR clean completion via `FINISH` (type = 6).

### 3.2 Post-Event Observation Prohibition
- Any observation window where $t_{\text{cutoff}} \ge t_{\text{fail}}$ is strictly prohibited. Telemetry recorded after failure represents restart or post-mortem dynamics and is excluded from feature matrices.

### 3.3 Right-Censoring Elimination at Day 1 Boundary
- **Trace Cutoff**: $T_{\text{max}} = 86,400\times 10^6 \ \mu\text{s}$ (Day 1 end).
- **25-Minute Follow-Up Cutoff**:
  $$t_{\text{censored}} = T_{\text{max}} - 1,500\times 10^6 \ \mu\text{s} = 84,900\times 10^6 \ \mu\text{s} \quad (23.583\text{ hours})$$
- **Rule**: Any observation window ending at $t_{\text{cutoff}} > 84,900\times 10^6 \ \mu\text{s}$ that does NOT have a terminal event prior to $T_{\text{max}}$ suffers from incomplete follow-up. It is flagged as `RIGHT_CENSORED` and dropped from training and evaluation.

---

## 4. Query Budget and Guardrail Verification

- **Max Bytes Billed Guardrail**: $21,000,000,000$ bytes (21.0 GB).
- **Standard On-Demand Price**: $6.25 per TiB ($1 \text{ TiB} = 1,099,511,627,776$ bytes).
- **Maximum Cost at Cap**:
  $$\frac{21,000,000,000}{1,099,511,627,776} \times \$6.25 = \mathbf{\$0.1194 \text{ USD}} \le \mathbf{\$0.125 \text{ USD}}$$
- **BigQuery Sandbox Allowance**: 1 TiB ($1,099,511,627,776$ bytes) free query processing per calendar month. A 21.0 GB query consumes **1.91%** of the monthly quota, leaving **98.09%** remaining at **$0.00 USD** marginal cost.
- **Provisional Estimates**: Earlier analytical estimates (18.4 GB, $0.092 USD, 125,000 events) remain strictly provisional offline projections until verified by a genuine BigQuery service dry run.
