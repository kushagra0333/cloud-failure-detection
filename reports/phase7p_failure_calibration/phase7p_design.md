# Phase 7P: Leakage-Safe Failure-Risk Calibration and Operating-Threshold Selection — Experimental Design

**Project Root:** `C:\Users\hp\Desktop\Research project`  
**Phase:** Phase 7P  
**Author / Lead:** Antigravity Senior ML Research Engineering Subsystem  
**Date:** October 29, 2026  
**Status:** Frozen Prior to Training & Calibration Execution  

---

## 1. Scientific Motivation & Problem Statement

Phase 7N resolved the temporal Warning/Critical separation failure of the multi-resolution architecture:
- Condition N2 (Asymmetric Critical Margin Loss) shifted the mean Critical logit margin from negative ($-0.1120$) to positive ($+0.7712 \pm 0.1176$).
- On true Critical validation windows, $100\%$ achieved $P_2 > P_1$, and $83.0\%$ achieved $z_2 - z_1 \ge 0.50$ logit units.
- In exploratory sweeps with lowered thresholds, genuine Critical Recall surged from $0.0\%$ to $85.7\%$.

However, under the predeclared primary evaluation policy ($\theta_{\text{fail}} = 0.50$), genuine Critical Recall remained at $0.0\%$. Forensics revealed that the aggregate failure risk $P_{\text{fail}} = P_1 + P_2$ on Critical windows averaged only $0.1918 \pm 0.0431$. Because $P_{\text{fail}} < 0.50$, the cascade assigned all Critical windows to Class 0 (Normal) before ever evaluating the Warning-versus-Critical margin branch.

The fixed threshold $\theta_{\text{fail}} = 0.50$ assumes an uncalibrated, balanced binary risk prior, which is severely misaligned with the low empirical base rate of failure ($\approx 14.8\%$ in both Train and Validation).

**Phase 7P investigates whether a leakage-safe calibration and threshold-selection procedure—fitted and selected exclusively on Candidate A Train out-of-fold predictions—can identify an operating threshold $\tau$ that unlocks pre-failure event interception while preserving specificity gates and Critical-versus-Warning separation on Candidate A Validation.**

---

## 2. Research Questions & Hypotheses

- **RQ1 — Calibration Fidelity:** Does parent-job-grouped Train OOF Platt calibration improve binary failure-risk calibration on Candidate A Validation (measured via binary Brier score, binary NLL, binary ECE, and reliability bins)?
- **RQ2 — Pre-Failure Interception:** Does an operating threshold $\tau$ selected strictly on Train OOF data increase Episode Recall and genuine Critical Recall on Candidate A Validation?
- **RQ3 — Specificity Retention:** Does the selected threshold preserve the governing specificity constraints: Pure-Normal Entity FAR $\le 10\%$ and Normal-Window FPR $\le 5\%$ on Validation?
- **RQ4 — Margin Stability:** Does the Critical-versus-Warning logit margin ($z_2 - z_1 \ge 0.50$) remain effective in assigning Class 2 once failure alerts are triggered?
- **RQ5 — Operational Feasibility:** Can any leakage-safe calibrated operating policy satisfy all four operational gates simultaneously?

**Core Methodological Hypothesis:** Lowering the effective alert threshold via risk calibration will increase sensitivity (Episode Recall and Critical Recall), but risks triggering false alarms on pure-normal entities and normal windows. The trade-off must be managed without validation snooping.

---

## 3. Frozen Candidate Model Architecture

The model architecture is frozen exactly from Phase 7N Condition N2:
1. **Backbone:** Multi-Resolution Dual-Stream Temporal Classifier (`MultiResolutionLSTMClassifier`, 67,027 parameters).
   - **Stream A:** 2-layer LSTM encoding recent 5 5-minute steps ($t-20\text{m}$ to $t$, $F=31$, hidden dim = 64, dropout = 0.2).
   - **Stream B:** MLP ($93 \to 32 \to 16$, ReLU, dropout = 0.2) encoding 7 historical steps ($t-55\text{m}$ to $t-25\text{m}$) summarized into 93 causal statistics.
   - **Fused Head:** Linear($80 \to 64$) $\to$ ReLU $\to$ Dropout(0.2) $\to$ Linear($64 \to 3$).
2. **Learning Objective:** Asymmetric Critical Margin Loss
   $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}}(w=[1.0, 2.0, 2.0]) + \lambda_{\text{crit}} \times \frac{1}{|\{i: y_i=2\}|} \sum_{i: y_i=2} \operatorname{softplus}(m - (z_{i, 2} - z_{i, 1}))$$
   where $m = 0.50$ logit units, $\lambda_{\text{crit}} = 0.10$.
3. **Seeds:** 42, 123, 456.

---

## 4. Leakage-Free Calibration and Threshold Selection Protocol

### A. Parent-Job-Grouped Train Out-Of-Fold Generation
- Candidate A Train contains 3,278 windows across 51 unique parent jobs (34 failing jobs, 17 pure-normal jobs).
- Parent jobs are partitioned into 5 folds such that:
  - Every parent job belongs to exactly one held-out fold.
  - Zero parent job overlap exists between fold training sets and fold validation sets.
  - Each fold contains positive counts of Normal, Warning, and Critical classes.
- For each seed ($42, 123, 456$) and each fold ($k \in \{0, 1, 2, 3, 4\}$):
  - Train the candidate N2 model on the 4 training folds for 25 epochs.
  - Checkpoint selected at minimum fold validation loss.
  - Predict on the held-out fold to generate out-of-fold (OOF) logits $(z_0, z_1, z_2)$ and probabilities $(P_0, P_1, P_2)$.
- Assemble complete out-of-fold prediction set on Train ($N=3,278$ per seed).

### B. Platt Calibrator Fitting (Train OOF Only)
- Derive failure score:
  $$s_{\text{fail}} = \operatorname{logit}(\operatorname{clip}(P_1 + P_2, \epsilon, 1 - \epsilon)), \quad \epsilon = 10^{-6}$$
- Binary failure target:
  $$y_{\text{fail}} = \begin{cases} 1 & \text{if } y_{\text{train}} \in \{1, 2\} \\ 0 & \text{if } y_{\text{train}} = 0 \end{cases}$$
- Fit Platt logistic regression mapping per seed strictly on Train OOF:
  $$q_{\text{fail}} = \sigma(a \cdot s_{\text{fail}} + b) = \frac{1}{1 + \exp(-(a \cdot s_{\text{fail}} + b))}$$
- Record fitted parameters $(a, b)$ in `calibrator_parameters.json`.

### C. Deterministic Operating-Threshold Selection (Train OOF Only)
- Candidate grid: $\tau \in [0.00, 1.00]$ with step $0.005$ (201 candidate points).
- For each candidate threshold $\tau$:
  - Apply the calibrated decision rule to Train OOF predictions:
    $$\text{If } q_{\text{fail}} < \tau \implies \text{Normal}; \quad \text{Else if } (z_2 - z_1 \ge 0.50) \implies \text{Critical}; \quad \text{Else} \implies \text{Warning}$$
  - Compute operational metrics on Train OOF:
    - Pure-Normal Entity FAR (denominator: 153 pure-normal entities in Train).
    - Normal-Window FPR (denominator: 2,797 Normal windows in Train).
    - Episode Recall (denominator: 131 failure episodes in Train).
    - Genuine Critical Recall (denominator: 237 Critical windows in Train).
- **Selection Decision Rule:**
  1. **Step 1 (Specificity Constraints):** Filter candidate thresholds where mean (across seeds 42, 123, 456) Entity FAR $\le 10.0\%$ AND Window FPR $\le 5.0\%$.
  2. **Step 2 (Sensitivity Maximization):** Among eligible thresholds, select $\tau$ that maximizes mean Episode Recall, with ties broken by highest Genuine Critical Recall.
  3. **Step 3 (Tie-Break):** If multiple thresholds tie on both, select the largest $\tau$ (conservative tie-break).
  4. **Step 4 (Infeasibility):** If no candidate threshold satisfies Step 1 on Train OOF, record `NO_ELIGIBLE_THRESHOLD`.

### D. Policy Freeze
- Save the fitted calibrator parameters, selected shared threshold $\tau$, hashes, and decision logic to `results/phase7p_failure_calibration/frozen_operating_policy.json`.
- Compute and verify its cryptographic SHA-256 hash before Candidate A Validation labels are accessed.

---

## 5. Candidate A Validation Evaluation Policies

After policy freezing, evaluate on Candidate A Validation ($N=657$ windows):

1. **V0 — Historical Raw Reference:**
   - Raw score $P_{\text{fail}} = P_1 + P_2$.
   - Fixed threshold $\theta_{\text{fail}} = 0.50$.
   - Margin rule: Critical if $z_2 - z_1 \ge 0.50$, else Warning.
2. **V1 — Calibrated Primary Treatment (Core Deliverable):**
   - Calibrated score $q_{\text{fail}} = \operatorname{Platt}(P_{\text{fail}})$.
   - Operating threshold: Preselected $\tau$ from `frozen_operating_policy.json`.
   - Margin rule: Critical if $z_2 - z_1 \ge 0.50$, else Warning.
3. **V2 — Optional Isotonic Comparison:**
   - Calibrated score from Train OOF Isotonic Regression.
4. **V3 — Uncalibrated Diagnostic:**
   - Raw score $P_{\text{fail}}$ evaluated at numerical threshold $\tau$.

---

## 6. Frozen Operational Gates

All four gates must pass simultaneously under the primary treatment V1:

| Gate | Metric | Target | Denominator |
|:---|:---|:---:|:---|
| **Gate 1a** | Episode Recall | $\ge 70\%$ | 20 validation failure episodes |
| **Gate 1b** | Genuine Critical Recall | $\ge 50\%$ | 49 true Critical validation windows |
| **Gate 2a** | Pure-Normal Entity FAR | $\le 10\%$ | 49 pure-normal validation entities |
| **Gate 2b** | Normal-Window FPR | $\le 5\%$ | 560 Normal-labelled validation windows |

Final verdict:
- `OPERATIONAL_WINNER` only if all four gates pass simultaneously under V1.
- Otherwise, `NO_OPERATIONAL_WINNER`.
- `AUDIT_BLOCKED` if governance, leakage, or reproducibility fails.
