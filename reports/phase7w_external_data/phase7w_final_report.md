# Phase 7W Final Report: Governance Correction, External Dataset Sourcing and Compatibility Assessment

**Project Root**: `C:\Users\hp\Desktop\Research project`  
**Execution Date**: 2026-10-29  
**Governing Protocol**: Phase 7B Evaluation Protocol v2.1  
**Governance Incident Status**: **DOCUMENTED_AND_RESOLVED**  
**Final Scientific & Operational Verdict**: **`NEW_BENCHMARK_REQUIRED`**  
**Audit Integrity Status**: `AUDIT_PASSED` (10/10 automated integrity tests passed)

---

## 1. Executive Summary & Core Verdict

Phase 7W was initiated to resolve a recorded governance inconsistency from Phase 7V and to determine which external workload dataset can validly support future failure-prediction research.

### Core Findings:
1. **Governance Correction (Part A)**:
   - The Phase 7V script (`scripts/run_phase7v_audit.py`) executed `pd.read_csv` on `confirmatory_metadata.csv` (line 206) and `confirmatory_l5_metadata.csv` (line 222) to compute entity counts and rows for its inventory table.
   - Consequently, the assertion in Phase 7V reports that these partitions were "strictly unaccessed" was factually inaccurate.
   - However, **zero model training, evaluation, threshold tuning, or feature array access** occurred. The incident was an inventory scanning breach, not an experimental contamination. This discrepancy is formally documented and sealed in [phase7v_governance_correction.md](file:///c:/Users/hp/Desktop/Research%20project/results/phase7w_external_data/phase7v_governance_correction.md).
2. **Frozen-Model Evaluation (C1)**:
   - **Zero external datasets can support faithful frozen-model evaluation**.
   - None of the candidate datasets supply the 31 exact Borg 2011 features (such as hardware performance counters CPI and MAI, disk I/O fraction, and assigned memory ratio).
   - Passing external data into the frozen Phase 7S, 7T, or 7U models would induce severe out-of-distribution feature distortion and violates the frozen preprocessing pipeline.
3. **New External Benchmark (C2)**:
   - **Google Borg ClusterData 2019** is the premier candidate for establishing a new cloud failure-prediction benchmark. It shares the Borg workload hierarchy, microsecond lifecycle event timestamps, and 5-minute sampling cadence.
   - **Alibaba Cluster Trace 2018** is a secondary, locally downloadable candidate for DAG batch job failure prediction.


**PRIMARY VERDICT: NEW_BENCHMARK_REQUIRED**

---

## 2. Governance Correction Addendum Summary

Forensic review of `scripts/run_phase7v_audit.py` (SHA-256: `32b25961dec4db8ba2d595e1fa45720dcbcf4e0e4cc75baaf39f664a6ba79c4b`) revealed that while building `cohort_inventory.csv`, the script invoked `pd.read_csv()` on:
- `data/processed/confirmatory/confirmatory_metadata.csv`
- `data/processed/confirmatory_l5/confirmatory_l5_metadata.csv`

While the script only inspected `len()` and `.nunique()` to report summary statistics, `pd.read_csv` loads the entire CSV into memory.
The Phase 7V governance claim of "zero access" has been formally corrected in [phase7v_governance_correction.md](file:///c:/Users/hp/Desktop/Research%20project/results/phase7w_external_data/phase7v_governance_correction.md). The original Phase 7V artifacts remain intact without silent modification.

---

## 3. External Candidate Evaluation & Compatibility Matrix

All four public candidates were audited across eight dimensions:

| dimension | google_borg_2019 | clusterwise_summit | alibaba_2018 | lanl_cfdr |
| --- | --- | --- | --- | --- |
| 1. Workload / Environment Similarity | HIGH (Google production Borg cells; containerized services & batch workloads) | LOW (HPC supercomputing environment; MPI scientific simulations on POWER9/Volta) | MODERATE (Hybrid cloud; co-located Fuxi DAG batch jobs and Sigma online containers) | LOW (HPC scientific simulation clusters; non-containerized monolithic batch jobs) |
| 2. Failure Labels & Timestamps | MODERATE (Task FAIL/LOST events with microsecond timestamps; no native 3-class proximity labels) | LOW (Node outage state transitions; no container task failure events) | MODERATE (Task/Instance Failed status with second timestamps; no native proximity labels) | LOW (Administrative Remedy incident outage records; no container task failure events) |
| 3. Telemetry Similarity & Feature Mapping | MODERATE (CPU/memory histograms available; MAI absent; CPI/allocs semantically shifted) | VERY LOW (Physical node/GPU power & temperature metrics; 0/31 Borg features match) | LOW (Basic CPU/memory utilization present; zero hardware counters CPI/MAI; 4/31 features) | VERY LOW (Zero continuous multivariate telemetry timeseries; textual syslogs only) |
| 4. Entity / Job Hierarchy & Time-Series Suitability | HIGH (Cell -> Job -> Task -> Alloc set; continuous 5-minute sampling intervals) | MODERATE (Cluster -> Node -> GPU; time-series suited for forecasting, not container tasks) | HIGH (Machine -> Job -> DAG Task -> Instance; variable sampling cadence) | LOW (System -> Node; discrete incident intervals, not continuous time-series) |
| 5. Independent Provenance | HIGH (May 2019 trace across 8 global Borg cells; completely disjoint from 2011 single-cell trace) | VERY HIGH (Completely independent institution: Oak Ridge National Laboratory Summit) | VERY HIGH (Completely independent cloud provider: Alibaba Group production clusters) | VERY HIGH (Completely independent institution: Los Alamos National Laboratory) |
| 6. Dataset Scale & Practical Local Access | LOW LOCAL / HIGH CLOUD (~2.4 TiB compressed; local storage impractical, BigQuery access required) | LOW (>500 GB Parquet; requires distributed compute like Dask/Polars on HPC/workstation) | HIGH (~50 GB compressed; feasible for local workstation download and filtering) | VERY HIGH (~100 MB; easily handled on any local workstation) |
| 7. Suitability for Frozen-Model Evaluation (C1) | INCOMPATIBLE (Fails C1: MAI missing, alloc/CPI shifted, scaler mismatch, proximity labels absent) | INCOMPATIBLE (Fails C1: 0/31 features available; completely disparate modality) | INCOMPATIBLE (Fails C1: Hardware counters missing; feature space mismatch; scaler mismatch) | INCOMPATIBLE (Fails C1: No telemetry timeseries; fails all frozen model requirements) |
| 8. Suitability for New Adapted Benchmark (C2) | VERY HIGH (Best candidate for defining a 2019 cloud failure prediction benchmark with BigQuery) | LOW (Suited for hardware anomaly detection / power forecasting, not cloud task failure) | HIGH (Feasible candidate for defining a batch task failure prediction benchmark locally) | LOW (Suited for log-based root-cause analysis or MTBF estimation, not telemetry prediction) |
| OVERALL RECOMMENDATION | PRIMARY CANDIDATE FOR NEW BENCHMARK (C2) | NOT SUITABLE FOR BORG FAILURE PREDICTION | SECONDARY CANDIDATE FOR LOCAL NEW BENCHMARK (C2) | NOT SUITABLE FOR CONTINUOUS TELEMETRY PREDICTION |

---

## 4. Feature Mapping Analysis (C1 Frozen Evaluation)

Mapping the 31 required features across external datasets demonstrated why **frozen-model evaluation is impossible**:
- **Hardware Performance Counters**: `cycles_per_instruction` (CPI) and `memory_accesses_per_instruction` (MAI) are completely absent in Alibaba 2018, ClusterWise, and LANL CFDR. In Google Borg 2019, MAI is absent and CPI is tracked via different cycle histograms.
- **Resource Definitions**: Memory allocation shifted from static assigned memory in 2011 to shared *alloc sets* in 2019. Disk I/O time tracking shifted to storage service access.
- **Scaler Incompatibility**: The frozen scaler fitted on Candidate A Train cannot be transferred to different hardware architectures or telemetry units.

---

## 5. Specification for the New External Benchmark (C2)

Because C1 is blocked, future research must pursue **C2: An Adapted Failure-Prediction Benchmark**. The recommended protocol specification for **Google Borg 2019** is:

1. **Target Definition**:
   - Binary or multiclass failure prediction based on task event types: `FAIL` (unexpected crash) and `LOST` (unreachable host).
   - Observation window: $L=12$ continuous 5-minute steps (60 minutes).
   - Prediction horizons: Warning interval (10–25 minutes prior) and Critical interval (within 5 minutes).
2. **Feature Schema**:
   - Extract resampled 5-minute continuous series of CPU average usage, CPU max usage, memory usage, and alloc set utilization.
   - Engineer rolling statistical trends (lags, slopes, rolling std, relative burstiness).
3. **Entity & Job Hierarchy**:
   - Group by `job_id` and evaluate cross-job generalization to guard against pure-normal job confounding (the core challenge identified in Phase 7R–7U).
4. **Data Acquisition Strategy**:
   - Use Google Cloud BigQuery sandbox to extract a curated single-cell slice (e.g. 5,000 tasks over 7 days, ~1 GB) to allow local workstation training and evaluation.
5. **Separation from Borg 2011 Results**:
   - Explicitly declare that models trained on Borg 2019 represent a new study and cannot be directly compared against Phase 7S–7U Borg 2011 operational gate metrics.

---

## 6. Deliverables & Sealed Manifest

All deliverables have been generated in `results/phase7w_external_data/` and mirrored to `reports/phase7w_external_data/`.
Manifest sealed in `results/phase7w_external_data/phase7w_manifest.json`.

**Final Primary Verdict**: `NEW_BENCHMARK_REQUIRED`  
**Governance Incident Status**: `DOCUMENTED_AND_RESOLVED`  
**Integrity Verdict**: `AUDIT_PASSED`
