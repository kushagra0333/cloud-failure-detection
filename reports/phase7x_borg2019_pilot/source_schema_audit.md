# Google Borg ClusterData 2019 Source Schema Audit

**Project Root**: `C:\Users\hp\Desktop\Research project`  
**Reference Document**: [Google cluster-usage traces v3](https://drive.google.com/file/d/10r6cnJ5cJ89fPWCgj7j4LtLBqYN9RiI9/view)  
**Primary Repository**: [github.com/google/cluster-data](https://github.com/google/cluster-data)  
**BigQuery Project ID**: `google.com:google-cluster-data`  

---

## 1. Trace Topography and Cell Structure
The Google Borg 2019 dataset captures 31 days (May 1 to May 31, 2019) across **eight distinct production Borg cells**, partitioned into 8 BigQuery datasets:
- `clusterdata_2019_a`
- `clusterdata_2019_b`
- `clusterdata_2019_c`
- `clusterdata_2019_d`
- `clusterdata_2019_e`
- `clusterdata_2019_f`
- `clusterdata_2019_g`
- `clusterdata_2019_h`

Each cell represents an autonomous cluster installation with independent machine pools, job dispatchers, and network fabrics.

---

## 2. Relational Table Schemas

### 2.1 Table: `instance_events`
Tracks lifecycle state transitions for tasks and allocation instances.
- **`time`** (`INTEGER`): Microseconds from trace start ($0 = 2019	ext{-}05	ext{-}01	ext{ 00:00:00 UTC}$).
- **`type`** (`INTEGER`): Enumerated event code:
  - `0: SUBMIT` — Task or alloc instance submitted to system.
  - `1: QUEUE` — Placed in queue; not eligible for immediate scheduling.
  - `2: ENABLE` — Made eligible for scheduling.
  - `3: SCHEDULE` — Assigned to a physical machine and started.
  - `4: EVICT` — Forcibly descheduled due to resource overcommitment or higher-priority arrival.
  - `5: FAIL` — Task terminated abnormally due to internal failure or software error (**Primary Failure Event**).
  - `6: FINISH` — Task completed execution successfully and normally.
  - `7: KILL` — Explicitly cancelled by user or parent collection death.
  - `8: LOST` — Host machine disconnected or unreachable; task state unknown (**Sensitivity Failure Event**).
  - `9: UPDATE_PENDING` — Pending configuration update.
  - `10: UPDATE` — Configuration update applied.
- **`collection_id`** (`INTEGER`): Unique identifier for the parent collection (job / alloc set).
- **`instance_index`** (`INTEGER`): Index of the instance within its parent collection.
- **`machine_id`** (`INTEGER`): Obfuscated identifier of physical host machine.
- **`alloc_collection_id`** (`INTEGER`): Reference ID if instance runs within a shared alloc set.
- **`alloc_instance_index`** (`INTEGER`): Alloc instance index.
- **`priority`** (`INTEGER`): Borg scheduling priority (0 = free/batch, 115–119 = production tier, 360 = monitoring).
- **`scheduling_class`** (`INTEGER`): Latency-sensitivity tier (0 to 3).
- **`resource_request`** (`RECORD`): CPU and memory capacity requested.

### 2.2 Table: `instance_usage`
Logs periodic resource consumption observations (aggregated over 5-minute windows).
- **`start_time`** (`INTEGER`): Microsecond timestamp of window start.
- **`end_time`** (`INTEGER`): Microsecond timestamp of window end (typically `start_time + 300,000,000` $\mu$s).
- **`collection_id`** (`INTEGER`): Foreign key matching `instance_events.collection_id`.
- **`instance_index`** (`INTEGER`): Foreign key matching `instance_events.instance_index`.
- **`machine_id`** (`INTEGER`): Host machine identifier during this measurement interval.
- **`alloc_collection_id`** (`INTEGER`): Associated alloc set reference.
- **`cpu_usage`** (`RECORD`):
  - `cpu_usage.average` (`FLOAT`): Average CPU core utilization over the 5-minute interval.
  - `cpu_usage.maximum` (`FLOAT`): Peak CPU core utilization.
  - `cpu_usage.sample_count` (`INTEGER`): Number of hardware counter samples.
  - `cpu_usage.histogram` (`RECORD`): Detailed percentile histogram bins.
- **`memory_usage`** (`RECORD`):
  - `memory_usage.canonical_memory_usage` (`FLOAT`): Normalized memory consumption.
  - `memory_usage.assigned_memory_usage` (`FLOAT`): Memory reservation allocation.
  - `memory_usage.maximum_memory_usage` (`FLOAT`): Peak memory observed.

### 2.3 Table: `collection_events`
Tracks metadata for high-level collections and jobs:
- `collection_id`, `time`, `type`, `user`, `collection_name`, `parent_collection_id` (enables multi-tier job hierarchies), `max_instances`.

---

## 3. Timestamp Semantics & Composite Entity Keys
1. **Timestamp Units**: All timestamps are 64-bit integers expressing elapsed **microseconds** ($\mu	ext{s}$). One 5-minute step equals $300,000,000 \ \mu	ext{s}$.
2. **Entity Identification**:
   - Because `instance_index` is only unique within a collection, and `collection_id` can collide across cells, the globally unique entity composite key is defined as:
     $$\mathbf{Entity\;Composite\;Key} = (	ext{cell\_id},\;	ext{collection\_id},\;	ext{instance\_index})$$
