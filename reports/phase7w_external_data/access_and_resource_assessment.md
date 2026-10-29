# External Dataset Access, Storage, and Compute Assessment

**Project Root**: `C:\Users\hp\Desktop\Research project`  
**Assessment Date**: 2026-10-29  
**Scope**: Practical evaluation of hosting, tooling, hardware capacity, and bandwidth for candidate external datasets.

---

## 1. Candidate 1: Google Borg ClusterData 2019 (`clusterdata-2019`)

- **Hosting & Access Protocol**:
  - Hosted directly on **Google Cloud BigQuery** under public project tables.
  - Official Repository: [github.com/google/cluster-data](https://github.com/google/cluster-data).
  - Trace Specification: Documented in `Google cluster-usage traces v3`.
- **Data Scale**:
  - Size: Approximately **2.4 TiB compressed** across 8 cells for May 2019.
  - Scale in uncompressed format: Exceeds 10 TiB.
- **Local Storage Feasibility**:
  - **Local Download Impractical**: Attempting to download the complete 2.4 TiB trace onto a local workstation disk is technically unfeasible and unnecessary.
  - **Recommended Sourcing Strategy**: Utilize BigQuery serverless SQL to filter, aggregate, and export a compact, representative cohort (e.g. 1 cell, 5,000 tasks, 1-week window) of ~500 MB to ~1 GB CSV/Parquet for local experimentation.
- **Hardware & Compute Requirements**:
  - Cloud query execution: BigQuery handles petabyte-scale joins in seconds.
  - Local workstation: 16–32 GB RAM is sufficient to process extracted cohorts.
- **Licensing & Compliance**:
  - Licensed under **Creative Commons Attribution 4.0 International (CC-BY 4.0)**. Completely open for research.

---

## 2. Candidate 2: ClusterWise / OLCF Summit

- **Hosting & Access Protocol**:
  - Hosted on **Hugging Face Hub** under repository [`MachaParfait/ClusterWise`](https://huggingface.co/datasets/MachaParfait/ClusterWise).
  - Distribution: Partitioned Apache Parquet files organized into monthly directories.
- **Data Scale**:
  - Size: **>500 GB compressed**; 1B to 10B rows.
- **Local Storage Feasibility**:
  - Full local download requires substantial disk storage (>500 GB SSD).
  - Out-of-core streaming via `huggingface_hub`, `polars`, or `dask` is required.
- **Hardware & Compute Requirements**:
  - High RAM (32–64 GB) or fast SSD scratch space for parquet scanning.
- **Licensing & Suitability**:
  - CC-BY 4.0. However, the telemetry is strictly physical hardware sensors (GPU power and temperature) on Summit supercomputer, completely unsuited for containerized cloud failure prediction.

---

## 3. Candidate 3: Alibaba Cluster Trace 2018 (`cluster-trace-v2018`)

- **Hosting & Access Protocol**:
  - Hosted on GitHub / Alibaba Open Data: [github.com/alibaba/clusterdata](https://github.com/alibaba/clusterdata).
  - Distribution: Tar/Gzip compressed CSV archives for 6 core tables.
- **Data Scale**:
  - Size: **~50 GB compressed** (~200 GB uncompressed).
- **Local Storage Feasibility**:
  - **High Feasibility**: Download of individual tables (e.g., `batch_task.csv.tar.gz` and `batch_instance.csv.tar.gz`) can be performed directly via HTTP/cURL on a standard workstation.
- **Hardware & Compute Requirements**:
  - Chunked streaming via `pd.read_csv(chunksize=...)` or DuckDB/Polars is readily achievable on 16 GB RAM.
- **Licensing**:
  - Alibaba Open Data Agreement (Academic & Research Use).

---

## 4. Candidate 4: LANL CFDR (Computer Failure Data Repository)

- **Hosting & Access Protocol**:
  - Hosted on USENIX portal: [usenix.org/cfdr-data](https://www.usenix.org/cfdr-data).
  - Distribution: ASCII text tables and gzip compressed logs.
- **Data Scale**:
  - Size: **~100 MB**.
- **Local Storage Feasibility**:
  - Trivial local download (<1 minute).
- **Suitability**:
  - Only contains discrete administrative incident tickets (MTBF/root-cause), with zero continuous multivariate telemetry timeseries.
