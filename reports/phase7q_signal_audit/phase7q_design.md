# Phase 7Q — Experimental Design: Cross-Job Predictive Signal Audit and Causal Workload-Relative Features

## 1. Context and Objective

Phase 7P demonstrated that probability calibration (Platt scaling) and operating-threshold selection on Train Out-of-Fold (OOF) data could not resolve the failure-interception deficit identified in Phase 7N. The fitted Platt calibrator had negative slopes ($a \in [-0.15, -0.12]$), its Validation Brier score ($0.125955$) was approximately equal to a constant-prevalence baseline ($0.125844$), and its preselected operating threshold ($\tau = 0.2250$) yielded no failure alerts on Candidate A Validation.

Phase 7Q addresses the root cause of this behavior through two controlled workstreams:

- **Workstream A — Predictive Signal Audit:**  
  Independently audit raw N2 failure-risk scores across held-out parent jobs and validation windows. Determine whether raw scores contain genuine failure-ranking signal, quantify the impact of the negative Platt slope (ranking inversion), and analyze fold-level variation and parent-job concentration (especially the single-parent-job Fold 0).
- **Workstream B — Causal Workload-Relative Feature Treatment:**  
  Test whether deviations from an entity's own recent historical baseline provide failure-discriminative information that absolute telemetry values do not capture. Evaluate whether a matched N2 baseline (Q0) and an extended model with a causal relative-deviation branch (Q1) improve ranking and operational interception across unseen parent jobs.

---

## 2. Research Questions & Hypotheses

### Primary Questions
1. Does the raw N2 multi-resolution model contain a useful failure-ranking signal across held-out parent jobs?
2. Why did Phase 7P Platt calibration yield negative slopes? Was it driven by ranking inversion, weak signal, or fold imbalance?
3. Does the calibrated Brier score outperform a static constant-prevalence baseline?
4. Does adding a causal relative-deviation feature branch (Q1) improve OOF failure ranking over the matched baseline (Q0)?
5. Does Q1 improve Critical-versus-Warning separation and genuine Critical Recall?
6. Does Q1 preserve false-alarm bounds under the frozen operational policy?

### Working Hypotheses
- **Hypothesis A (Prevalence Collapse):** Phase 7P calibration improved Brier score solely by shrinking probabilities toward the cohort failure prevalence ($\sim 14.7\%$), without improving individual failure discrimination.
- **Hypothesis B (Cross-Job Shift):** In Google Borg traces, baseline telemetry levels vary substantially across different parent workloads. Absolute resource usage confuses normal windows of heavy jobs with degradation windows of light jobs.
- **Hypothesis C (Causal Relative Deviation):** Normalizing an entity's recent telemetry ($t-20\text{m}$ to $t$) against its own historical baseline ($t-55\text{m}$ to $t-25\text{m}$) using robust scale floors removes job-specific offsets and improves failure ranking across parent jobs.

---

## 3. Strict Data Governance

- **Authorized Partitions:**
  - Candidate A Train: 3,278 windows (2,797 Normal, 244 Warning, 237 Critical; 51 parent jobs).
  - Candidate A Validation: 657 windows (560 Normal, 48 Warning, 49 Critical; 20 episodes, 49 pure-normal entities).
- **Prohibited Partitions:**
  - Candidate A Test (previously inspected in Phase 7D; strictly prohibited).
  - Phase 6 Holdout (strictly prohibited).
  - No derivation, calibration fitting, scale-floor fitting, or hyperparameter selection using Validation outcomes.

---

## 4. Workstream A Design: Signal & Calibration Audit

1. **Prevalence Baseline Formulation:**  
   Compare binary Brier score, log-loss (NLL), and ECE against:
   - Train Constant Prevalence: $p_{\text{tr}} = 481 / 3278 \approx 0.146736$.
   - Validation Constant Prevalence: $p_{\text{val}} = 97 / 657 \approx 0.147641$.
2. **Discriminative Ranking Metrics:**  
   Compute ROC-AUC, Average Precision, Spearman correlation, and Pearson correlation for raw and calibrated scores on Train OOF and Validation.
3. **Fold-Level Diagnostics:**  
   Inspect each of the 5 Phase 7P folds for window count, job count, failure prevalence, raw AUC, and fold-specific Platt parameters.
4. **Conclusion Categories:**  
   Classify Workstream A as exactly one of: `RANKING_SIGNAL_SUPPORTED`, `RANKING_SIGNAL_WEAK_OR_INCONSISTENT`, `RANKING_DIRECTION_OR_IMPLEMENTATION_ISSUE`, or `INSUFFICIENT_EVIDENCE`.

---

## 5. Workstream B Design: Matched Q0 vs. Q1 Treatment

### 5.1 Balanced Grouped 5-Fold Partition
Unlike Phase 7P (which placed 643 windows of 1 job in Fold 0), Phase 7Q uses a capacity-balanced greedy partition ensuring:
- Approximately 10–11 parent jobs per fold across all 5 folds.
- Each fold contains both failing jobs (6–7) and pure-normal jobs (3–4).
- Zero parent jobs cross fold boundaries.
- The three largest jobs (940, 643, 283 windows) are placed into distinct folds.

### 5.2 Baseline Model Q0
- Multi-resolution LSTM classifier identical to Phase 7N N2.
- Stream A: 5 recent steps (25 min, 31 features), 2-layer LSTM (hidden 64, dropout 0.2).
- Stream B: 7 historical steps causal summaries (93 features), MLP (93 -> 32 -> 16).
- Loss: `AsymmetricCriticalMarginLoss` ($\lambda_{\text{crit}} = 0.10$, margin $= 0.50$, class weights `[1.0, 2.0, 2.0]`).
- Trained on 4 folds, evaluated on 1 held-out fold across seeds 42, 123, 456.

### 5.3 Treatment Model Q1
- Retains Stream A and Stream B completely unchanged.
- Adds Branch C (causal workload-relative deviation):
  - For each of 31 features:
    $$\Delta_j = \text{clip}\left( \frac{\text{median}_{\text{recent}, j} - \text{median}_{\text{hist}, j}}{\max(\text{scale}_{\text{hist}, j}, \text{scale\_floor}_j)}, -5.0, +5.0 \right)$$
    where $\text{scale}_{\text{hist}, j} = 1.4826 \times \text{MAD}(\text{hist}_j)$.
  - `scale_floor_j` is computed strictly from Candidate A Train (10th percentile of positive historical scales).
  - Branch C MLP: Linear(31, 32) -> ReLU -> Dropout(0.2) -> Linear(32, 16) -> ReLU.
  - Fusion: Concatenates Stream A (64), Stream B (16), and Branch C (16) = 96 dimensions.
  - Final classification head: Linear(96, 64) -> ReLU -> Dropout(0.2) -> Linear(64, 3).
  - Total parameters: 69,603 (delta: +2,576 parameters vs Q0).

### 5.4 Fixed Operational Policy
Both Q0 and Q1 are evaluated under the frozen operational policy on Candidate A Validation:
- Predict Normal if $P_{\text{fail}} = P_1 + P_2 < 0.50$.
- Else predict Critical if $z_2 - z_1 \ge 0.50$; Warning otherwise.
- No validation threshold tuning is permitted.

---

## 6. Operational Gates & Decision Standard

The frozen operational gates remain:
- **Gate 1A:** Episode Recall $\ge 70\%$ (at least 14 / 20 episodes).
- **Gate 1B:** Genuine Critical Recall $\ge 50\%$ (at least 25 / 49 Critical windows).
- **Gate 2A:** Pure-Normal Entity FAR $\le 10\%$ (at most 4 / 49 pure-normal entities).
- **Gate 2B:** Normal-Window FPR $\le 5\%$ (at most 28 / 560 normal windows).

Verdict must be one of:
- `OPERATIONAL_WINNER` (if all 4 gates pass simultaneously for all seeds under fixed policy).
- `NO_OPERATIONAL_WINNER` (if any gate fails).
- `AUDIT_BLOCKED` (if material integrity failure occurs).
