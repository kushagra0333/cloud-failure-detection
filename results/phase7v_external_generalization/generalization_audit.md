# Technical Generalization Audit: Phase 7V

## 1. Executive Summary
A comprehensive audit of all project data assets was conducted to determine whether an authorized, genuinely independent workload cohort exists to evaluate the frozen failure-prediction approaches developed through Phase 7U.

**Audit Verdict**: **`NO_INDEPENDENT_COHORT_AVAILABLE` (Path B)**

Every data partition in the workspace was inspected, fingerprinted, and audited for entity, parent-job, and temporal overlap with Candidate A:
1. **Candidate A Dev Pool (Train + Validation)**: 3,935 windows, 353 entities, 63 parent jobs. Explored repeatedly across Phases 7B–7U; cannot serve as independent confirmation.
2. **Candidate B Temporal-Group Split**: 4,960 windows across 412 entities and 74 parent jobs. Proved to have **100% entity overlap** and **100% parent-job overlap** with Candidate A. Completely non-independent.
3. **Historical Phase 1–5 Partitions (Random and Chronological Splits)**: Proved to have 53 to 245 overlapping entities with Candidate A Dev from the exact same Google Borg 2011 trace. Completely non-independent.
4. **Prohibited Partitions**: Candidate A Test (inspected in Phase 7D) and Phase 6 Confirmatory Holdouts are strictly prohibited by protocol governance.
5. **Synthetic Benchmark**: `data/synthetic_benchmark/` contains 0 files.
6. **Raw Telemetry**: `data/raw/` contains only the source trace of the existing Borg 2011 cell.

## 2. Comprehensive Cohort Inventory Table
| cohort_name | relative_path | source_cluster | rows | entities | parent_jobs | dev_entity_overlap | dev_job_overlap | classification | independence_status | byte_size | sha256 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Candidate A Train | data/processed/phase7c/train_metadata.csv | Google Borg 2011 (May 2011 trace) | 3278 | 284 | 51 | 284 | 51 | AUTHORIZED_TRAINING_COHORT | DEVELOPMENT_BASELINE | 164062 | 0180ade4098711680c44bbd2ca8c3c5d85834a86042546cc9300b00f1df725ce |
| Candidate A Validation | data/processed/phase7c/val_metadata.csv | Google Borg 2011 (May 2011 trace) | 657 | 69 | 12 | 69 | 12 | DEVELOPMENT_VALIDATION_COHORT | EXPLORED_DEVELOPMENT_DATA | 32677 | 8d22193f11d671dc570d29abf9255d06386880d49c5de83c68d6404a8b41b956 |
| Candidate A Test | data/processed/phase7c/test_metadata.csv | Google Borg 2011 (May 2011 trace) | 1025 | 59 | 11 | 0 | 0 | PROHIBITED_TEST_PARTITION | GOVERNANCE_PROHIBITED | 55645 | 5bf69371833c4b90558afaf0f445244252bc6a6cc5a368cc2cc027d8a38fb4cf |
| Candidate B Temporal-Group Split | results/phase7a_4/candidate_b_temporal_group_split.csv | Google Borg 2011 (May 2011 trace) | 4960 | 412 | 74 | 353 | 63 | ALTERNATIVE_SPLIT_SAME_DATA | NON_INDEPENDENT_OVERLAP | 256673 | a9f31e43b164d8dd12089d0ad6e610e403a5e867615c1c6fcc022cf6fdf46804 |
| Chronological Split (Train) | data/processed/chronological/train_metadata.csv | Google Borg 2011 (May 2011 trace) | 1603 | 401 | 0 | 236 | 0 | HISTORICAL_PHASE3_SPLIT | NON_INDEPENDENT_OVERLAP | 50761 | c0796d3e88e757039afe9734d47f1c21699fe3bab0f7bfa6491200bd06143ec5 |
| Chronological Split (Val) | data/processed/chronological/val_metadata.csv | Google Borg 2011 (May 2011 trace) | 724 | 400 | 0 | 240 | 0 | HISTORICAL_PHASE3_SPLIT | NON_INDEPENDENT_OVERLAP | 22783 | 7018689c3d3199cee8e5b881625fb390b0e46bf656604bd55b2df0bd3c8e7f48 |
| Chronological Split (Test) | data/processed/chronological/test_metadata.csv | Google Borg 2011 (May 2011 trace) | 460 | 236 | 0 | 171 | 0 | HISTORICAL_PHASE3_SPLIT | NON_INDEPENDENT_OVERLAP | 17611 | b1facc699689886796b334f3eedc09fc6894f41bfe37512c2040f4918b5f3d41 |
| Phase 1-5 Random Split (Train) | data/processed/train_metadata.csv | Google Borg 2011 (May 2011 trace) | 4798 | 367 | 0 | 245 | 0 | HISTORICAL_PHASE1_5_SPLIT | NON_INDEPENDENT_OVERLAP | 154858 | 7b427ae2132b7a2a8d2e3b8f94ff0ea24091e8f10b1e5b11ab4428ced9879856 |
| Phase 1-5 Random Split (Val) | data/processed/val_metadata.csv | Google Borg 2011 (May 2011 trace) | 1015 | 78 | 0 | 55 | 0 | HISTORICAL_PHASE1_5_SPLIT | NON_INDEPENDENT_OVERLAP | 34460 | ff6f25567e8dfa01b6108e8bab5dda597029d43a45b7db18a0cd7e75e55e0254 |
| Phase 1-5 Random Split (Test) | data/processed/test_metadata.csv | Google Borg 2011 (May 2011 trace) | 1003 | 80 | 0 | 53 | 0 | HISTORICAL_PHASE1_5_SPLIT | NON_INDEPENDENT_OVERLAP | 33956 | f9abdb39c2bb9694da2b1572bc505f1d9c69244fb55b33210c98a932ab8945d3 |
| Phase 6 Confirmatory Holdout | data/processed/confirmatory/confirmatory_metadata.csv | Google Borg 2011 (May 2011 trace) | 2596 | 188 | 0 | 0 | 0 | PROHIBITED_PHASE6_HOLDOUT | GOVERNANCE_PROHIBITED | 67386 | 8aae825741c5f93b8cb7a38daa8160cd7d094cb63799b575ecc2cc73457869d9 |
| Phase 6 Confirmatory L5 Holdout | data/processed/confirmatory_l5/confirmatory_l5_metadata.csv | Google Borg 2011 (May 2011 trace) | 5511 | 335 | 0 | 0 | 0 | PROHIBITED_PHASE6_HOLDOUT | GOVERNANCE_PROHIBITED | 156696 | ee71e6d72960723f6737c40769bab6fe964e777ce1008076e38bd554710bab3d |
| Synthetic Benchmark | data/synthetic_benchmark/ | N/A (Empty Directory) | 0 | 0 | 0 | 0 | 0 | EMPTY_DIRECTORY | NO_DATA_PRESENT | 0 | N/A (Directory) |
| Raw Google Borg 2011 Telemetry | data/raw/ | Google Borg 2011 (May 2011 trace) | Millions (Raw Events) | Unwindowed | Unwindowed | Source of Candidate A | Source of Candidate A | RAW_TRACE_STORAGE | NON_INDEPENDENT_SOURCE | 185797809 | N/A (Directory) |

## 3. Why Candidate A Validation Cannot Provide Generalization Evidence
Candidate A Validation (657 windows, 69 entities, 12 parent jobs) has been repeatedly evaluated across Phases 7C through 7U:
- It was used to assess Phase 7M multi-resolution context, Phase 7N critical margin loss, Phase 7P calibration, Phase 7Q signal audit, Phase 7R forensics, Phase 7S relative ablation, Phase 7T linear hybrid, and Phase 7U fuzzy logic.
- Furthermore, Validation is heavily skewed: 12 parent jobs, with 49 pure-normal entities and 20 failure episodes.
- Claiming that a model generalizes to diverse cloud workloads based on repeated validation on these 12 parent jobs constitutes a severe methodological fallacy.

## 4. Frozen Model Standby Audit
All three candidate failure-prediction architectures have been audited, cryptographically fingerprinted, and placed on standby under frozen protocol `v3.0-EXTERNAL-GENERALIZATION-STANDBY`:
| model_identifier | model_architecture | parameter_count | checkpoints_verified | preprocessing_pipeline | operating_policy | external_readiness_status |
| --- | --- | --- | --- | --- | --- | --- |
| Phase 7S S2 Relative-Only Model | MultiResolutionLSTMClassifier (Stream A: L=5, F=31 relative; Stream B: F=93 relative summaries; Fused: 80->64->3) | 67027 | results/phase7s_relative_ablation/checkpoints/ckpt_S2_seed*.pt (3 full-train + 15 OOF) | Causal baseline-relative scaling (12-step historical median & MAD scale floor) -> Stream A (recent 5) & Stream B (hist 93 summaries) | P_fail >= 0.50, z2 - z1 >= 0.50 | READY_FOR_STANDBY_EVALUATION |
| Phase 7T Relative-Primary Linear Hybrid (80/20) | Deterministic Ensemble: R_hybrid = 0.80 * R_S2 + 0.20 * R_S0; Critical margin: z2_S2 - z1_S2 | 134,054 (Combined S0 + S2 inference) | results/phase7s_relative_ablation/checkpoints/ and results/phase7q_signal_audit/checkpoints/ | Dual stream: Raw absolute telemetry (S0) + Causal baseline-relative telemetry (S2) | R_hybrid >= 0.7650, z2_S2 - z1_S2 >= 0.50 (Frozen in frozen_operating_policy.json) | READY_FOR_STANDBY_EVALUATION |
| Phase 7U Mamdani Fuzzy Inference System | Type-1 Mamdani Min-Max Centroid (9 relative-primary rules; R_rel & R_abs inputs; z2_S2 - z1_S2 margin) | 134,054 + Rule Base | results/phase7s_relative_ablation/checkpoints/ and results/phase7q_signal_audit/checkpoints/ | Dual stream + empirical membership breakpoints (q20, q50, q80 derived on Train-OOF) | Primary: tau = 1.0000 (0 alerts); Diagnostic: tau = 0.5000 | READY_FOR_STANDBY_EVALUATION |

## 5. Specification for External Cohort Acquisition
To execute a scientifically valid external generalization study, the project requires an external dataset satisfying the following minimum specifications:
1. **Cluster Independence**: Must originate from an entirely separate cluster or cloud provider (e.g., Google Borg 2019 trace, Alibaba Cluster Trace v2018, Azure Public VM trace, or Oak Ridge / LANL supercomputer logs).
2. **Telemetry Coverage**: Must provide continuous CPU, memory, disk I/O, and network usage measurements mappable to the 31 core Borg features.
3. **Temporal Sampling**: Must provide continuous time series sampled at 5-minute intervals, with at least 60 minutes of uninterrupted history per entity ($L=12$).
4. **Workload Hierarchy**: Must preserve parent-job/workload identifiers to allow separation of within-job failure ranking from cross-job baseline confounding.
5. **Ground Truth Failure Semantics**: Must document failure timestamps to distinguish imminent critical states (within 5 min) from warning states (10–25 min).
