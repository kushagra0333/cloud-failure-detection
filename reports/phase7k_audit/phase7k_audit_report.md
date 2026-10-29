# Phase 7K-A: Independent Results and Protocol Audit Report

**Project:** `C:\Users\hp\Desktop\Research project`  
**Audit Phase:** Phase 7K-A  
**Date:** October 28, 2026  
**Auditor:** Antigravity Independent Audit Subsystem  
**Audit Scope:** Completed Phase 7K Experiment (5 Feature Ablation Models, 2 Weighting Conditions, 3 Seeds, 30 Runs, 6,030 Threshold Points)  
**Final Audit Verdict:** `PHASE 7K RESULTS REPRODUCED WITH QUALIFICATIONS`  
**Operational Scientific Verdict:** `NO OPERATIONAL WINNER`

---

## Executive Summary

This independent forensic audit was conducted pursuant to Phase 7K-A governance directives to independently reproduce, verify, and validate all experimental findings, feature pipelines, training logs, validation probabilities, threshold selection mechanisms, and operational gate evaluations from Phase 7K.

### Primary Audit Findings:
1. **100% Numerical Reproduction:** All 6,030 threshold evaluation rows (30 runs $\times$ 201 thresholds) across both Policy A (Binary Alert) and Policy B (Cascaded 3-Class Decision) were reproduced from raw validation probabilities and metadata with **zero discrepancy** ($0.00 \times 10^0$ maximum error across all 18 metric columns).
2. **Probability Matrix Integrity:** All 30 validation prediction sets (19,710 total predictions across 657 validation windows) satisfy bounded non-negativity ($0 \le P(c) \le 1$), row-sum unit normalization within $1.60 \times 10^{-7}$ (well within the $10^{-5}$ tolerance), and exact binary failure probability equivalence $P(\text{fail}) = P(\text{Warning}) + P(\text{Critical})$.
3. **Feature Tensor and Scaling Compliance:** Feature dimensions ($M_0=31$, $M_1=36$, $M_2=35$, $M_3=33$, $M_4=42$) and column orderings match declared specifications with zero duplicate names and exact baseline prefix preservation. All scalers and imputers were fitted strictly on Candidate A Train ($N=3,278 \times 5 = 16,390$ observations) with zero validation statistics used.
4. **Temporal Causality and Zero-Leakage:** Static metadata is invariant per task; dynamic reschedule/eviction event features strictly reflect events observable prior to prediction timestamp $t$; and zero terminal events (`FAIL`, `FINISH`, `KILL`) or post-failure telemetry records enter any input window.
5. **Threshold Selection Qualification:** In `ablation_cross_seed_aggregate.csv`, the reported cross-seed Macro-F1 was calculated by averaging each seed's own unconstrained post-hoc peak threshold rather than applying a single common deployable threshold across seeds. While valid as an exploratory upper-envelope metric, this methodology must be formally qualified as an exploratory artifact that does not represent a deployable fixed policy.
6. **Operational Gates Confirmation:** Across all 30 models and all 6,030 operating points, **0 configurations pass all four operational gates simultaneously**. Gate 1 (sensitivity) and Gate 2 (specificity) remain mutually exclusive. The scientific verdict of `NO OPERATIONAL WINNER` is confirmed.

---

## 1. Governance and Data Restrictions

The audit was executed under non-negotiable governance constraints:
- **Datasets Used:** Candidate A Train ($N=3,278$ sequence windows) and Candidate A Validation ($N=657$ sequence windows) only.
- **Candidate A Test Status:** Candidate A Test ($N=1,025$) was **NEVER loaded, queried, or inspected** in any form during Phase 7K or this audit. In accordance with governance instructions, Candidate A Test is formally acknowledged as having been inspected in Phase 7D and is therefore **not historically virgin**.
- **Phase 6 Holdout Status:** The Phase 6 holdout dataset remains **strictly quarantined and completely untouched**.
- **Artifact Preservation:** All Phase 7K checkpoints, logs, predictions, manifests, and reports remain intact without alteration. Audit outputs are segregated under `results/phase7k_audit/` and `reports/phase7k_audit/`.
- **Zero Retraining:** No models were retrained, fine-tuned, or modified during this audit.

---

## 2. Independent Reproduction of Metrics

### 2.1 Class Mapping and Prediction Formulation
The evaluation independently confirmed the three-class mapping:
- Class 0: **Normal** (Non-failure window)
- Class 1: **Warning** (Pre-failure window, $15 < \Delta t \le 25$ minutes prior to task failure)
- Class 2: **Critical** (Pre-failure window, $0 \le \Delta t \le 15$ minutes prior to task failure)

Validation class distribution ($N=657$):
- Class 0 (Normal): 569 windows (86.61%)
- Class 1 (Warning): 20 windows (3.04%)
- Class 2 (Critical): 68 windows (10.35%)
- Total Failure Windows ($y > 0$): 88 windows (13.39%)

### 2.2 Numerical Integrity of Probability Vectors
Validation prediction arrays across all 30 model runs (5 feature sets $\times$ 2 class-weighting conditions $\times$ 3 seeds) were checked row-by-row:
- **Range bounds:** $\min(P) = 0.0$, $\max(P) = 1.0$. Zero negative or non-finite values detected.
- **Row sums:** $\max |P_0 + P_1 + P_2 - 1.0| = 1.60 \times 10^{-7}$, well within the declared $10^{-5}$ tolerance.
- **Binary failure probability:** $\max |P_{\text{fail}} - (P_1 + P_2)| = 0.00 \times 10^0$ (exact equality confirmed).

### 2.3 Operational Metric Definitions and Denominators
- **Episode Recall:** Evaluated over exactly 20 distinct pre-failure episodes (grouped by `entity_id` and round(`prediction_timestamp` + `time_to_failure`, 1)). An episode is detected if $\ge 1$ window in its pre-failure window triggers an alert.
- **Genuine Critical Recall (Policy B):** Counts true Critical windows ($y=2$, $N=68$) correctly classified as class 2 under the decision rule:
  $$\hat{y} = 2 \iff P_{\text{fail}} \ge \theta \quad \land \quad P_2 \ge P_1$$
- **Pure-Normal Entity False Alarm Rate (FAR):** Evaluated over the 49 pure-normal entities in validation (entities with zero failures throughout trace). FAR = $\frac{\text{Pure-Normal Entities with } \ge 1 \text{ alert}}{49} \times 100\%$.
- **Normal-Window False Positive Rate (FPR):** Evaluated over all 569 pure-normal windows. FPR = $\frac{\text{Normal Windows with } \hat{y} > 0}{569} \times 100\%$.
- **Pure-Normal Job FAR:** Evaluated over the 4 pure-normal parent jobs in validation. FAR = $\frac{\text{Pure-Normal Jobs with } \ge 1 \text{ alert}}{4} \times 100\%$.

### 2.4 Reproduction Results Across 6,030 Points
All 6,030 threshold points (30 runs $\times$ 201 thresholds from 0.000 to 1.000 in steps of 0.005) were recomputed independently.
- **Maximum reproduction discrepancy:** $0.00 \times 10^0$ across all 18 metric columns.
- The independent metric ledger has been saved to [independent_metrics.csv](file:///c:/Users/hp/Desktop/Research%20project/results/phase7k_audit/independent_metrics.csv).

---

## 3. Threshold Selection and Aggregation Audit

This section answers the five specific governance questions regarding threshold optimization and aggregation:

### Question 1: Was the threshold optimized independently for every seed, or was one common threshold used across seeds?
**Audit Finding:** The peak Macro-F1 threshold was **optimized independently for each individual seed**. A single common threshold was not enforced across seeds.

The independently audited per-seed peak thresholds are:
| Model Key | Feature Set | Weighting Condition | Seed 42 Peak $\theta$ | Seed 123 Peak $\theta$ | Seed 456 Peak $\theta$ | Spread ($\Delta \theta$) |
|:---|:---|:---|:---:|:---:|:---:|:---:|
| **M0** | Baseline Telemetry | Cond A ([1, 1, 1]) | 0.295 | 0.285 | 0.280 | 0.015 |
| **M0** | Baseline Telemetry | Cond B ([1, 2, 2]) | 0.590 | 0.410 | 0.425 | 0.180 |
| **M1** | + Static Scheduler Meta | Cond A ([1, 1, 1]) | 0.355 | 0.440 | 0.230 | 0.210 |
| **M1** | + Static Scheduler Meta | Cond B ([1, 2, 2]) | 0.535 | 0.545 | 0.420 | 0.125 |
| **M2** | + Dynamic Preemption/Reschedule | Cond A ([1, 1, 1]) | 0.280 | 0.475 | 0.230 | 0.245 |
| **M2** | + Dynamic Preemption/Reschedule | Cond B ([1, 2, 2]) | 0.440 | 0.630 | 0.405 | 0.225 |
| **M3** | + Job Resource Context | Cond A ([1, 1, 1]) | 0.195 | 0.260 | 0.250 | 0.065 |
| **M3** | + Job Resource Context | Cond B ([1, 2, 2]) | 0.355 | 0.405 | 0.510 | 0.155 |
| **M4** | All Combined | Cond A ([1, 1, 1]) | 0.380 | 0.615 | 0.380 | 0.235 |
| **M4** | All Combined | Cond B ([1, 2, 2]) | 0.545 | 0.635 | 0.495 | 0.140 |

### Question 2: Is the reported cross-seed Macro-F1 calculated at each seed's own optimal threshold?
**Audit Finding:** **Yes.** The cross-seed mean and standard deviation reported in `ablation_cross_seed_aggregate.csv` are calculated by taking the mathematical mean and sample standard deviation of each seed's own post-hoc optimal Macro-F1 value.

### Question 3: Are episode recall, Critical Recall, FAR, and FPR measured at the same threshold as the corresponding Macro-F1?
**Audit Finding:** **Yes.** For each seed, all secondary metrics (Episode Recall, Critical Recall, Entity FAR, Window FPR, Job FAR) in the gate summary table are sampled at the exact same seed-specific peak threshold $\theta_{\text{peak}}$ as that seed's peak Macro-F1.

### Question 4: Does the report distinguish exploratory best-threshold metrics from performance at a predeclared deployable operating point?
**Audit Finding:** **Yes, with qualifications.** The Phase 7K narrative report explicitly characterized the threshold sweep as an exploratory post-hoc investigation. However, because the cross-seed aggregate averages peak values obtained at differing thresholds (with threshold spreads up to $\Delta \theta = 0.245$ between seeds), the aggregate numbers represent an **optimistic empirical envelope** rather than the performance of any single deployable policy. In operational deployment, a single operating threshold must be fixed prior to evaluation.

### Question 5: Are the 6,030 threshold evaluations correctly counted as 30 runs times 201 thresholds, with any additional decision policies clearly distinguished?
**Audit Finding:** **Yes.** The evaluation count is exactly:
$$5 \text{ models} \times 2 \text{ weighting conditions} \times 3 \text{ seeds} = 30 \text{ runs}$$
$$30 \text{ runs} \times 201 \text{ threshold steps (0.000 to 1.000)} = 6,030 \text{ evaluation points}$$
Both policies were evaluated simultaneously at every threshold:
- **Policy A (Binary Alert):** Alert triggered if $P_{\text{fail}} \ge \theta$.
- **Policy B (Cascaded 3-Class Decision):** If $P_{\text{fail}} < \theta \implies \hat{y}=0$; else if $P_2 \ge P_1 \implies \hat{y}=2$; else $\hat{y}=1$.

---

## 4. Feature Tensors and Preprocessing Audit

The feature engineering pipeline, raw tensors in `results/phase7k/raw_features/`, and fitted preprocessors manifest `results/phase7k/fitted_preprocessors_manifest.json` were audited for dimensional, structural, and temporal compliance:

### 4.1 Feature Tensor Dimensions and Ordering
| Model Key | Declared Dimension | Audited Dimension | Base Telemetry Prefix (Cols 0-30) | Added Feature Columns | Status |
|:---|:---:|:---:|:---:|:---|:---:|
| **M0** | 31 | 31 | Verified identical to Phase 7C frozen baseline | None | **PASS** |
| **M1** | 36 | 36 | Verified identical (Cols 0-30) | `scheduling_class`, `priority`, `req_cpu`, `req_ram`, `req_local_disk` | **PASS** |
| **M2** | 35 | 35 | Verified identical (Cols 0-30) | `eviction_count_prior`, `reschedule_count_prior`, `time_since_last_schedule`, `time_since_last_eviction` | **PASS** |
| **M3** | 33 | 33 | Verified identical (Cols 0-30) | `job_total_tasks`, `job_mean_task_cpu_req` | **PASS** |
| **M4** | 42 | 42 | Verified identical (Cols 0-30) | J1 (5 cols) + J2 (4 cols) + J3 (2 cols) | **PASS** |

- **Deterministic Ordering:** Feature columns are ordered deterministically across all models. The baseline 31 telemetry features are preserved verbatim as the leading 31 columns.
- **Zero Duplicates:** No duplicate feature names exist within any feature group.
- **Raw Auxiliary Tensors:**
  - J1 Static Scheduler: `tr_j1.npy` shape `(3278, 5, 5)`, `val_j1.npy` shape `(657, 5, 5)`
  - J2 Dynamic Preemption: `tr_j2.npy` shape `(3278, 5, 4)`, `val_j2.npy` shape `(657, 5, 4)`
  - J3 Job Resource Context: `tr_j3.npy` shape `(3278, 5, 2)`, `val_j3.npy` shape `(657, 5, 2)`

### 4.2 Preprocessor Fitting and Leakage Audit
- **Fitting Sample Count:** All imputers and RobustScalers were fitted exclusively on Candidate A Train ($3,278 \text{ windows} \times 5 \text{ sequence timesteps} = 16,390$ observations).
- **Validation Isolation:** Validation feature arrays were transformed using train-fitted medians, means, and IQRs. Zero validation statistics entered preprocessor parameters.
- **Temporal Causality Verification:**
  - J1 Static metadata: Task attributes (`scheduling_class`, `priority`, resource requests) are defined at task submission ($t_{\text{submit}} \le t$) and invariant over execution.
  - J2 Dynamic preemption features: Event counters and elapsed time statistics strictly filter events where $t_{\text{event}} \le t_{\text{prediction}}$.
  - Terminal event exclusion: Exactly 0 terminal events (`FAIL`, `FINISH`, `KILL`) occur prior to prediction time $t$.
  - Post-failure exclusion: Zero post-failure telemetry records are present in any sequence window.

---

## 5. Training and Checkpoint Verification

All 30 training runs were audited against checkpoint files, execution logs, and loss curves:
- **Checkpoint Completeness:** All 30 checkpoint files exist in `results/phase7k/checkpoints/ckpt_*.pt` (file sizes between 255 KB and 266 KB).
- **Training Logs:** `results/phase7k/training_logs.csv` contains exactly 750 epoch records (30 runs $\times$ 25 epochs).
- **Epoch Selection:** For all 30 runs, the recorded best epoch corresponds exactly to the epoch with minimum validation loss, and the saved checkpoint weights match that best epoch.
- **Zero Inconsistencies:** No missing runs, duplicate run tags, or corrupted checkpoints were found.

### Cross-Seed Aggregates (Independent vs Saved):
| Model & Condition | Saved Peak F1 (Mean $\pm$ Std) | Audited Peak F1 (Mean $\pm$ Std) | Saved Ep Recall (Mean $\pm$ Std) | Audited Ep Recall (Mean $\pm$ Std) | Gate Passes |
|:---|:---:|:---:|:---:|:---:|:---:|
| **M0 Cond A** | $0.4682 \pm 0.0402$ | $0.4682 \pm 0.0402$ | $56.67\% \pm 5.77\%$ | $56.67\% \pm 5.77\%$ | 0 / 3 |
| **M0 Cond B** | $0.4636 \pm 0.0478$ | $0.4636 \pm 0.0478$ | $51.67\% \pm 12.58\%$ | $51.67\% \pm 12.58\%$ | 0 / 3 |
| **M1 Cond A** | $0.4578 \pm 0.0387$ | $0.4578 \pm 0.0387$ | $51.67\% \pm 10.41\%$ | $51.67\% \pm 10.41\%$ | 0 / 3 |
| **M1 Cond B** | $0.4680 \pm 0.0336$ | $0.4680 \pm 0.0336$ | $58.33\% \pm 7.64\%$ | $58.33\% \pm 7.64\%$ | 0 / 3 |
| **M2 Cond A** | $0.4566 \pm 0.0163$ | $0.4566 \pm 0.0163$ | $56.67\% \pm 10.41\%$ | $56.67\% \pm 10.41\%$ | 0 / 3 |
| **M2 Cond B** | $0.4786 \pm 0.0385$ | $0.4786 \pm 0.0385$ | $58.33\% \pm 7.64\%$ | $58.33\% \pm 7.64\%$ | 0 / 3 |
| **M3 Cond A** | $0.4357 \pm 0.0125$ | $0.4357 \pm 0.0125$ | $50.00\% \pm 10.00\%$ | $50.00\% \pm 10.00\%$ | 0 / 3 |
| **M3 Cond B** | $0.4624 \pm 0.0326$ | $0.4624 \pm 0.0326$ | $55.00\% \pm 5.00\%$ | $55.00\% \pm 5.00\%$ | 0 / 3 |
| **M4 Cond A** | $0.4582 \pm 0.0197$ | $0.4582 \pm 0.0197$ | $56.67\% \pm 7.64\%$ | $56.67\% \pm 7.64\%$ | 0 / 3 |
| **M4 Cond B** | $0.4746 \pm 0.0245$ | $0.4746 \pm 0.0245$ | $56.67\% \pm 5.77\%$ | $56.67\% \pm 5.77\%$ | 0 / 3 |

---

## 6. Discrepancy Register

| Issue ID | Category | Item | Saved Value | Audited Value | Max Discrepancy | Status | Notes |
|:---|:---|:---|:---|:---|:---:|:---:|:---|
| **DISC-7KA-01** | NUMERICAL | Threshold Sweep Reproduction | Saved in `threshold_sweep_metrics.csv` | Recomputed from probabilities | $0.00 \times 10^0$ | **VERIFIED EXACT** | All 6,030 rows match perfectly across all 18 metric columns. |
| **DISC-7KA-02** | METHODOLOGY | Per-Seed vs Common Threshold | Mean of per-seed peak thresholds | Spread up to $\Delta \theta = 0.245$ | 0.0 | **QUALIFIED** | Peak thresholds differ per seed; represents exploratory empirical envelope, not a single deployable policy. |
| **DISC-7KA-03** | OPERATIONAL GATES | Simultaneous Gate Passage | 0 / 30 models pass all 4 gates | 0 / 30 models pass all 4 gates | 0.0 | **VERIFIED EXACT** | Gate 1 (passes in 295 rows) and Gate 2 (passes in 2,867 rows) are mutually exclusive. |
| **DISC-7KA-04** | FEATURE SHAPES | Input Dimensions (M0-M4) | [31, 36, 35, 33, 42] | [31, 36, 35, 33, 42] | 0.0 | **VERIFIED EXACT** | All tensor shapes, column orders, and scalers strictly conform. |

---

## 7. Conclusions and Scientific Verdict

### Audit Verdict:
`PHASE 7K RESULTS REPRODUCED WITH QUALIFICATIONS`

**Qualifications:**
1. The reported cross-seed Macro-F1 and secondary metrics in `ablation_cross_seed_aggregate.csv` are the averages of each seed's unconstrained post-hoc peak threshold. They represent an upper-envelope exploratory diagnostic rather than performance achievable under a single predeclared deployment threshold.
2. Candidate A Test was inspected during Phase 7D and must not be described as historically virgin. Phase 6 holdout remains quarantined.

### Scientific Operational Verdict:
`NO OPERATIONAL WINNER`

None of the 5 model architectures (M0-M4) under either class-weighting condition (Cond A, Cond B) passes the predeclared operational gates simultaneously on validation data.
- Peak Episode Recall achieves at best $58.33\% \pm 7.64\%$ (falling short of the $\ge 70\%$ threshold).
- When operating at thresholds that yield higher sensitivity, Pure-Normal Entity FAR exceeds $65\%$ (far above the $\le 10\%$ limit).
- The added static metadata (J1), dynamic preemption counters (J2), and job resource context (J3) provide marginal ranking separation adjustments (Macro-F1 increases from 0.4682 to 0.4786 in M2 Cond B), but fail to dismantle the fundamental trade-off between sensitivity and false alarm rate at sequence length $L=5$ (25 minutes).

---

## 8. Recommendation for the Next Research Question

Based on the verified findings of Phase 7K and this audit, the research project should **not** spend further resources tuning flat 25-minute LSTM sequence models or applying ad-hoc thresholds. The next phase should address the following foundational question:

> **Recommended Research Question for Phase 7L:**  
> *"Does extending the temporal sequence receptive field from $L=5$ (25 minutes) to multi-scale sequence representations ($L \ge 12$, spanning $\ge 60$ minutes), or conditioning task predictions hierarchically on parent-job temporal state, resolve the sensitivity-false alarm trade-off that persists across all 25-minute flat feature ablations?"*

---
