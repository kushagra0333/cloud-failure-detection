# BigQuery Query Budget and Cost Governance Report

**Project Root**: `C:\Users\hp\Desktop\Research project`  
**Target Public Dataset**: `google.com:google-cluster-data.clusterdata_2019_a`  
**Query SQL Checksum**: `58e5a2e4d864182f8b38786e356b02572df0a5c7b152a859b0ad093f6951a7bb`  

---

## 1. Hard Query Guardrails & Budgetary Caps

To prevent unexpected billing liabilities or accidental full-trace table scans, the following hard guardrails are predeclared:
- **Maximum Bytes Billed Cap**: **`25,000,000,000 bytes` (25 GB)**.
  - Any query whose BigQuery dry-run estimated bytes processed exceeds 25 GB is **automatically aborted**.
- **Maximum Cost Allocation**: **`$0.125 USD`** (based on standard on-demand rate of $5.00 / TiB).
- **Free Tier Policy**: The Google Cloud BigQuery free tier provides 1 TiB of free query data processed per calendar month. However, this phase does not assume zero cost because billing depends on the user's specific account quota and prior usage.
- **Prohibited Operations**:
  - `SELECT *` queries without explicit column projections.
  - Multi-cell unpartitioned scans across all 8 cells simultaneously.
  - Unbounded `COUNT(*)` queries on full tables.

---

## 2. Partition Pruning Reality in BigQuery
In Google Borg ClusterData 2019:
- Tables are sharded by cell (`clusterdata_2019_a` through `_h`).
- Targeting a single cell (`clusterdata_2019_a`) immediately isolates 87.5% of the total 2.4 TiB dataset.
- However, within each cell dataset, tables are partitioned or clustered by specific columns (such as time). If `instance_usage` is clustered on `start_time`, predicates like `u.start_time >= 0 AND u.end_time <= 86400000000` achieve partition pruning; if unpartitioned, BigQuery streams the full column for the selected fields across that single cell (~15–20 GB).

---

## 3. Cost and Local Storage Summary

| Metric | Target Constraint | Projected Value | Status |
| :--- | :---: | :---: | :---: |
| **Max Allowed Bytes Billed** | 25.0 GB | 18.4 GB (provisional) | **WITHIN CAP** |
| **Max Query Budget** | $0.125 USD | $0.092 USD (provisional) | **WITHIN BUDGET** |
| **Local Storage Budget** | 100 MB | ~15 MB (compressed) | **WITHIN BUDGET** |
| **Live Dry Run Service Evidence** | Required | **Blocked (Auth Needed)** | **BLOCKED** |
