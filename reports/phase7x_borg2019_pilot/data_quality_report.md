# Data Quality, Temporal Alignment, and Leakage Assessment

**Project Root**: `C:\Users\hp\Desktop\Research project`  
**Dataset Target**: Google Borg ClusterData 2019  

---

## 1. Temporal Alignment Between Usage and Event Logs
- `instance_usage` records fixed 5-minute sampling intervals (`start_time` to `end_time`), whereas `instance_events` records discrete event timestamps at microsecond resolution.
- Alignment Policy: A lifecycle event occurring at timestamp $T_{\text{event}}$ is associated with the usage window if $W_{\text{start}} \le T_{\text{event}} < W_{\text{end}}$.
- Failure lead time is measured strictly from $W_{\text{end}}$ to $T_{\text{fail}}$:
  $$\Delta t = T_{\text{fail}} - W_{\text{end}}$$
  This strictly guarantees that all usage telemetry in window $W$ was observed prior to the prediction timestamp.

---

## 2. Jitter and Observation Gaps
- In production Borg cells, task migration or machine reboot can induce gaps between 5-minute reporting intervals.
- Data Quality Rule: Tasks must exhibit continuous reporting without gaps exceeding 10 minutes. If a gap exceeds 10 minutes, the historical context sequence is reset to prevent unaligned recurrent states.

---

## 3. Absence of Borg 2011 Hardware Performance Counters
- As established in Phase 7W, Google Borg 2019 does not provide `memory_accesses_per_instruction` (MAI) or identical `cycles_per_instruction` (CPI) point metrics.
- The new benchmark explicitly relies on CPU usage statistics, memory canonical utilization, and rolling burstiness trends, avoiding fabricated or distorted feature proxies.
