# Phase 7Y Design: BigQuery Access Enablement, Schema Verification & Borg 2019 Pilot

## 1. Objective and Governance Scope
Phase 7Y is an empirical enablement and schema-verification phase for the proposed Google Borg 2019 failure-prediction benchmark:
- **Scope**: Authenticate to Google Cloud BigQuery, verify table schemas against the official `clusterdata_trace_format_v3.proto`, correct draft query plans, submit a genuine dry-run request, enforce hard query budgets, and extract a bounded pilot cohort.
- **Strict Governance Boundaries**:
  - No model training, threshold tuning, or operational winner claims.
  - Zero access to Candidate A Test or Phase 6 confirmatory holdouts.
  - If authentication requires user intervention, stop execution safely and produce actionable instructions.
  - Never fabricate synthetic pilot data or claim offline estimates are service responses.

## 2. Technical Investigation Sequence
1. **Stage A**: Runtime authentication check; test `bigquery.Client()` initialization.
2. **Stage B**: Schema verification; audit field names (`average_usage.cpus`, `assigned_memory`), join semantics, and event types.
3. **Stage C**: Real dry run submission or empirical blocker documentation.
4. **Stage D**: Hard budgetary caps ($0.125 / 25 GB).
5. **Stage E**: 3-class causal proximity target formulation with right-censoring guards.
6. **Stage F**: Pilot feasibility decision and manifest sealing.
