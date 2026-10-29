# Phase 7V Design: Independent-Cohort Readiness and Frozen-Model Generalization Audit

## 1. Scientific Role & Context
Phase 7V is an independent evaluation readiness audit, **not a model-development phase**.
Through Phases 7S, 7T, and 7U, the research established three distinct representation strategies:
1. **Phase 7S $S_2$**: Causal baseline-relative-only multi-resolution LSTM (neutralizes cross-job baseline confounding; reduces pure-normal Job 6251277720 false alarms from 90.7% to 6.1%; attenuates within-job discrimination from 0.6293 to 0.5345).
2. **Phase 7T Hybrid (80/20)**: Prespecified linear blend $0.80 R_{S2} + 0.20 R_{S0}$ (recovers within-job AUC to 0.5679; keeps Job 6251277720 false alarms at 7.30%; achieves lowest Brier score 0.1559).
3. **Phase 7U Mamdani Fuzzy System**: Type-1 Min-Max Centroid fuzzy rule base (recovers within-job AUC to 0.5607; achieves 66.7% episode recall at tau=0.50, but suffers 43.69% false alarm rate on Job 6251277720 due to Rule 6 elevation).

All three approaches were developed, fitted, and evaluated against the Candidate A development pool (Train: 3,278 windows, 51 parent jobs; Validation: 657 windows, 12 parent jobs).
Because Candidate A Validation has been repeatedly explored, it cannot serve as an independent confirmation cohort.

## 2. Objective & Governing Restrictions
The objective of Phase 7V is to establish whether the project has access to an authorized, genuinely independent external workload cohort to evaluate these frozen models.
- **Strict Governance Restrictions**:
  - Candidate A Test (1,025 windows) was already inspected in Phase 7D and is not virgin.
  - Phase 6 Confirmatory Holdout is designated as a virgin confirmatory partition and is strictly prohibited.
  - Candidate A Validation is an explored development cohort and cannot provide fresh evidence.
- **Allowed Decision Paths**:
  - **Path A**: An eligible, independent cohort exists with verified provenance and compatible features -> execute single-pass blinded evaluation under frozen protocol.
  - **Path B**: No eligible independent cohort exists -> declare `NO_INDEPENDENT_COHORT_AVAILABLE`, audit data provenance across all partitions, verify frozen model readiness on standby, and produce a formal data acquisition specification.
