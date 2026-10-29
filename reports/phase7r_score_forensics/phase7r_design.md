# Phase 7R Experimental Design: Failure-Score Direction, Label-Semantics, and Cross-Job Signal Forensics

## 1. Executive Summary and Scientific Mission

Phase 7Q established that both the matched multi-resolution baseline (Q0) and the causal relative-deviation feature extension (Q1) exhibited binary failure ROC-AUCs below 0.50:
- **Q0 Train-OOF ROC-AUC**: 0.3066 (AP = 0.1127).
- **Q1 Train-OOF ROC-AUC**: 0.3139 (AP = 0.1137).
- **Q0 Validation ROC-AUC**: 0.4244 (AP = 0.1677).
- **Q1 Validation ROC-AUC**: 0.4435 (AP = 0.1565).
- Phase 7P Platt calibrators produced negative slopes ($a \in [-0.15, -0.12]$), which strictly inverted the ordering of distinct raw scores.
- Phase 7P calibrated Brier score on Validation ($0.125955$) was approximately equal to a trivial static prevalence predictor ($p(1-p) = 0.125844$).

Phase 7R investigates why these failure ROC-AUCs are below 0.50 and conducts a rigorous forensic analysis across the entire modeling, evaluation, and data pipeline.

### Core Objectives
1. **Objective A — Audit Target Labels and Row Alignment (Workstream A)**: Determine whether label semantics, prediction timestamps, row indexing, class column serialization, or probability arithmetic ($P_{\text{fail}} = P_1 + P_2 = 1 - P_0$) contain implementation defects or misalignment bugs.
2. **Objective B — Independently Recompute Score-Ranking Metrics (Workstream B)**: Recompute all ranking, correlation, calibration, and discrimination metrics from raw serialized predictions using a clean-room independent evaluation pipeline without shared helper functions.
3. **Objective C — Score-Direction Diagnostic (Workstream C)**: Test the diagnostic inverse score ($P_{\text{fail\_inv}} = 1 - P_{\text{fail}} = P_0$) mathematically and empirically across seeds, folds, and cohorts to establish whether inversion is a superficial artifact or reflects underlying between-job confounding.
4. **Objective D — Cross-Job and Fold Forensics (Workstream D)**: Dissect the impact of massive parent-job imbalance (e.g. Job 6251277720 with 940 windows, Job 6251414911 with 643 windows, Job 6251995937 with 283 windows, concentrating >57% of train windows in 3 jobs). Test for **Simpson's Paradox** by decomposing ranking into within-job vs between-job components.
5. **Objective E — Verify Model Objective and Checkpoint Inference (Workstream E)**: Recompute predictions directly from saved PyTorch checkpoints to verify loss calculation, class weights ($[1.0, 2.0, 2.0]$), auxiliary margin ($\lambda=0.10, m=0.50$), and in-sample training convergence.
6. **Objective F — Interpretable Binary Baseline (Workstream F)**: Fit a simple, linear logistic regression baseline on recent 5-step features using the frozen 5-fold parent-job partition to test whether linear models exhibit the same cross-job transfer collapse.

---

## 2. Hypotheses Under Investigation

| Hypothesis ID | Description | Primary Diagnostic Test |
| :--- | :--- | :--- |
| **H1: Label Inversion Bug** | Frozen labels were inverted or remapped during data pipeline (e.g. 0 became failure, 1/2 became normal). | Exact row-by-row comparison against frozen `data/processed/phase7c/y_train.npy` and time-to-failure metadata. |
| **H2: Column / Probability Swap Bug** | Model logits or softmax probabilities were swapped during serialization (e.g. $P_0$ exported as $P_1$, or softmax applied along wrong axis). | Independent forward pass of saved PyTorch checkpoints and exact probability difference verification. |
| **H3: Prediction Row Misalignment** | Sorting, merging, or batch shuffling misaligned prediction rows with true label rows in OOF or Validation files. | Exact metadata key alignment (`entity_id`, `prediction_timestamp`, `parent_job_id`). |
| **H4: Training Loss Mis-specification** | Asymmetric margin loss or cross-entropy penalized true failures in the wrong direction during backpropagation. | Check in-sample training performance: if in-sample AUC is high ($>0.80$), the loss did not invert gradients. |
| **H5: Simpson's Paradox / Cross-Job Baseline Confounding** | Model learned absolute telemetry baselines that correlate with job type rather than impending failure. Large pure-normal jobs have high baseline values (predicting high $P_{\text{fail}}$), while failing jobs have lower baseline values, inverting cross-job pooled AUC while within-job AUC remains positive. | Decompose AUC into within-parent-job vs between-parent-job ranking; calculate unweighted vs window-weighted metrics. |
| **H6: Weak / Non-transferable Telemetry Signal** | Raw telemetry contains minimal generalizable signal across unseen parent jobs. | Evaluate simple linear baseline (Workstream F) across the same folds. |

---

## 3. Strict Data Governance and Partitioning

### Authorized Partitions
- **Candidate A Train**: 3,278 windows, 284 entities, 51 parent jobs.
  - Frozen labels: 2,797 Normal (Class 0), 244 Warning (Class 1), 237 Critical (Class 2).
  - Total failure windows: 481 (14.67%).
- **Candidate A Validation**: 657 windows, 69 entities, 12 parent jobs.
  - Frozen labels: 560 Normal (Class 0), 48 Warning (Class 1), 49 Critical (Class 2).
  - Total failure windows: 97 (14.76%).
  - 20 failure episodes, 49 pure-normal entities.

### Forbidden Partitions
- **Candidate A Test** (1,025 windows) — STRICTLY FORBIDDEN.
- **Phase 6 Confirmatory Holdout** — STRICTLY FORBIDDEN.
- Any derivative, reconstructed, or unscaled files originating from forbidden partitions.

### Access Control
- Explicit path allowlist enforced prior to any filesystem operation.
- No tuning of thresholds, loss parameters, or features on Validation outcomes.

---

## 4. Evaluation Protocol and Operational Standard

### Frozen Primary Operating Policy
- Alert triggered when raw $P_{\text{fail}} = P_1 + P_2 \ge 0.50$.
- Imminent severity decision:
  - Predict **Critical** if $z_2 - z_1 \ge 0.50$.
  - Predict **Warning** otherwise.

### Operational Gates (Unchanged)
1. **Gate 1A (Episode Recall)**: $\ge 70.0\%$ (Validation denominator = 20 episodes).
2. **Gate 1B (Genuine Critical Recall)**: $\ge 50.0\%$ (Validation denominator = 49 true Critical windows).
3. **Gate 2A (Pure-Normal Entity FAR)**: $\le 10.0\%$ (Validation denominator = 49 pure-normal entities).
4. **Gate 2B (Normal-Window FPR)**: $\le 5.0\%$ (Validation denominator = 560 Normal windows).

### Operational Verdict Standard
The verdict must be one of:
- `OPERATIONAL_WINNER`: All 4 operational gates pass under the fixed policy.
- `NO_OPERATIONAL_WINNER`: Any gate fails (expected based on prior evidence).
- `AUDIT_BLOCKED`: A material data governance, integrity, or code defect remains unresolved.
