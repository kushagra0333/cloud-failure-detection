# Phase 7X Design: Google Borg 2019 Acquisition, Label Feasibility & Benchmark Protocol

## 1. Scientific Objective
Phase 7X transitions the research from external source selection (Phase 7W verdict `NEW_BENCHMARK_REQUIRED`) to a rigorous protocol and feasibility assessment for a new task-failure prediction benchmark using Google Borg ClusterData 2019.
- **Scope**: Data acquisition feasibility, BigQuery schema audit, 3-class failure-proximity target specification, cell-aware group split design, and feature dictionary construction.
- **Governing Constraints**:
  - No model training, fine-tuning, or operational winner claims.
  - Zero access to Candidate A Test or Phase 6 confirmatory holdouts.
  - If BigQuery credentials/billing are unavailable, report `DATA_ACCESS_BLOCKED` and do not manufacture synthetic data.

## 2. Protocol Components
1. **Source Schema Audit**: Complete relational map of 8 cells (`clusterdata_2019_a` to `clusterdata_2019_h`), tables (`instance_events`, `instance_usage`), and lifecycle event codes.
2. **Access Feasibility**: Dry-run cost and resource estimation; formal blocker documentation.
3. **Query Plan**: Reproducible SQL queries bounded to Cell `a` and first 24 hours.
4. **Target Specification**: Class 0 (Normal >25m), Class 1 (Warning 10-25m), Class 2 (Critical 0-5m), Buffer Exclusion (5-10m), Post-Event Censoring ($t \ge t_{\text{fail}}$).
5. **Leakage-Safe Cell Split**: Train (a, b, c, d), Val (e, f), Test (g, h).
6. **Feature Dictionary**: 12 custom features tailored to Borg 2019 telemetry.
