# Borg 2019 Schema Verification & Phase 7X Query Correction Report

**Project Root**: `C:\Users\hp\Desktop\Research project`  
**Reference Specification**: [Google cluster-usage traces v3](https://drive.google.com/file/d/10r6cnJ5cJ89fPWCgj7j4LtLBqYN9RiI9/view) (`clusterdata_trace_format_v3.proto`)  
**Target Dataset**: `google.com:google-cluster-data.clusterdata_2019_a`  

---

## 1. Audit of Phase 7X Draft Query Assumptions

A rigorous cross-reference of the Phase 7X draft query ([results/phase7x_borg2019_pilot/query_plan.sql](file:///c:/Users/hp/Desktop/Research%20project/results/phase7x_borg2019_pilot/query_plan.sql)) against the authoritative `clusterdata_trace_format_v3.proto` schema identified several critical discrepancies that would cause query execution failure:

### Discrepancy 1: Nested Record Field Names in `instance_usage`
- **Phase 7X Draft Assumption**:
  - `u.cpu_usage.average`
  - `u.cpu_usage.maximum`
  - `u.memory_usage.canonical_memory_usage`
  - `u.memory_usage.assigned_memory_usage`
- **Official v3 Protobuf / BigQuery Reality**:
  In Google cluster traces v3, resource measurements use the nested `Resources` message (`message Resources { optional float cpus = 1; optional float memory = 2; }`):
  - **`average_usage.cpus`**: Mean CPU utilization in normalized core fraction.
  - **`maximum_usage.cpus`**: Peak CPU utilization in normalized core fraction.
  - **`average_usage.memory`**: Mean canonical memory usage (RAM bytes normalized).
  - **`assigned_memory`**: Direct scalar float representing memory limit (not nested under `memory_usage`).
  - **`page_cache_memory`**: Direct scalar float representing kernel page cache.
  - *Correction*: The SQL projection must query `u.average_usage.cpus`, `u.maximum_usage.cpus`, `u.average_usage.memory`, and `u.assigned_memory`.

### Discrepancy 2: Join Cardinality and Task Retry Ambiguity
- **Phase 7X Draft Assumption**:
  - Joined `instance_usage` directly with `terminal_events` using `u.collection_id = t.collection_id AND u.instance_index = t.instance_index`.
- **Relational Reality**:
  - A task instance in Borg may experience multiple events over a 24-hour trace (e.g. `SCHEDULE` -> `EVICT` -> `SCHEDULE` -> `FAIL`).
  - A naive join on `(collection_id, instance_index)` without timestamp anchoring produces a **Cartesian product of multiple duplicate rows**, inflating row counts and corrupting lead-time calculations.
  - *Correction*: Terminal failure events must be correlated using an event window join: finding the earliest terminal event occurring strictly after the observation window:
    ```sql
    t.terminal_time = (
        SELECT MIN(time) 
        FROM `google.com:google-cluster-data.clusterdata_2019_a.instance_events` e
        WHERE e.collection_id = u.collection_id 
          AND e.instance_index = u.instance_index 
          AND e.time >= u.end_time 
          AND e.type IN (4, 5, 6, 7, 8)
    )
    ```

### Discrepancy 3: Workload Grouping Hierarchy
- **Phase 7X Draft Assumption**:
  - Assumed `collection_id` directly maps to parent jobs.
- **Relational Reality**:
  - In Borg 2019, `collection_id` identifies individual jobs or alloc sets. Multi-tier workflows (e.g. MapReduce or service tiers) define parent relationships in `collection_events.parent_collection_id`.
  - *Correction*: To enable hierarchical within-job vs cross-job failure prediction evaluation (the core research question established in Phases 7Q–7U), `collection_events` should be queried to resolve `parent_collection_id` (defaulting to `collection_id` when null).

### Discrepancy 4: Event Enumeration & `UPDATE_RUNNING`
- In v3 schema, event types are:
  - `0: SUBMIT`, `1: QUEUE`, `2: ENABLE`, `3: SCHEDULE`
  - `4: EVICT` (scheduler priority preemption; non-failure)
  - `5: FAIL` (**Primary software crash ground truth**)
  - `6: FINISH` (normal completion)
  - `7: KILL` (user/job cancellation; non-failure)
  - `8: LOST` (**Sensitivity node failure**)
  - `9: UPDATE_PENDING`, `10: UPDATE_RUNNING` (dynamic resource reconfiguration; task continues running)
- `UPDATE_RUNNING` is an active state modification, not a termination.

---

## 2. Corrected Query Architecture

The corrected query plan has been formalized in [query_plan.sql](file:///c:/Users/hp/Desktop/Research%20project/results/phase7y_borg2019_pilot/query_plan.sql). It resolves all schema, join, and field name discrepancies while strictly bounding data scan volumes.
