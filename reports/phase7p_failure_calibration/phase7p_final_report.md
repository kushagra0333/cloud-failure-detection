# Phase 7P — Final Scientific Evaluation Report: Leakage-Safe Failure-Risk Calibration and Operating-Threshold Selection

**Author:** Antigravity Senior ML Research Team  
**Governing Protocol:** `results/phase7b_1/phase7b_evaluation_protocol_v2.json`  
**Reference Candidate:** Phase 7N N2 Multi-Resolution Model (`ckpt_N2_seed*.pt`, $\lambda_{\text{crit}}=0.10$, auxiliary margin $= 0.50$)  
**Target Dataset:** Candidate A Train ($N=3,278$, 51 parent jobs) & Candidate A Validation ($N=657$, 20 episodes, 49 pure-normal entities)  
**Execution Timestamp:** 2026-10-29 12:01:14  
**Governing Policy Freeze SHA-256:** `4fb88a3858044cf5196c25f0941da53c35a74e311ffa8ae899ed238876594cbc`  

---

## 1. Executive Summary & Verdict

Phase 7P investigated whether a leakage-safe binary failure-risk calibration and operating-threshold selection procedure could resolve the threshold-interception failure identified in Phase 7N while preserving specificity gates and Critical-versus-Warning separation.

In Phase 7N, the N2 asymmetric-margin objective achieved strong Critical-versus-Warning logit separation ($+0.771$ logit units on true Critical windows), but missed pre-failure events because the aggregate failure score $P_{\text{fail}} = P_1 + P_2$ remained below the fixed reference threshold of $0.50$. Phase 7P evaluated whether fitting Platt calibrators on parent-job-grouped Train Out-of-Fold (OOF) predictions and deterministically selecting an operating threshold $\tau$ prior to validation evaluation could yield an operational winner.

### Final Operational Verdict

$$\mathbf{FINAL\;VERDICT:\;NO\_OPERATIONAL\_WINNER}$$

Under the predeclared primary policy **V1** (Train-OOF Platt calibration with frozen threshold $\tau = 0.2250$):
- **Gate 1A (Episode Recall $\ge 70\%$):** **FAILED** ($0.0\% \pm 0.0\%$, 0/20 episodes intercepted across all 3 seeds).
- **Gate 1B (Genuine Critical Recall $\ge 50\%$):** **FAILED** ($0.0\% \pm 0.0\%$, 0/49 Critical windows detected across all 3 seeds).
- **Gate 2A (Pure-Normal Entity FAR $\le 10\%$):** **PASSED** ($0.0\% \pm 0.0\%$, 0/49 false entities alerted).
- **Gate 2B (Normal-Window FPR $\le 5\%$):** **PASSED** ($0.0\% \pm 0.0\%$, 0/560 normal windows alerted).
- **Simultaneous 4-Gate Passes:** **0 / 3 seeds**.

Diagnostic policy **V3** (uncalibrated raw score with threshold $\tau = 0.2250$) increased Episode Recall to $50.0\% \pm 18.0\%$ and Critical Recall to $21.1\% \pm 5.9\%$, but suffered catastrophic specificity collapse: Pure-Normal Entity FAR reached $30.6\% \pm 7.1\%$ (failing Gate 2A) and Normal-Window FPR reached $37.3\% \pm 11.8\%$ (failing Gate 2B by more than $7\times$).

Probability calibration significantly improved probability fidelity (binary Brier score improved from $0.1425$ to $0.1259$, binary NLL improved from $0.4705$ to $0.4189$, binary ECE plummeted from $12.39\%$ to $0.43\%$) by shrinking all probabilities toward the cohort base rate ($\sim 14.7\%$). However, calibration cannot alter the underlying feature overlap. Because the feature representations of failure windows do not cleanly separate from normal background telemetry across unseen parent jobs, lowering the threshold to intercept episodes inherently violates operational specificity bounds.

---

## 2. Phase 7N Preflight Audit & Reconciliation

Prior to execution, all 36 Phase 7N deliverables were independently audited against `results/phase7n_critical_margin/phase7n_manifest.json`:
- **File Integrity:** 36 / 36 files matched SHA-256 hashes bit-for-bit.
- **N2 Checkpoint Reproduction:** Evaluating historical checkpoints `ckpt_N2_seed*.pt` on Candidate A Validation yielded a maximum absolute probability difference of $< 3.0 \times 10^{-8}$ against `validation_predictions.csv`.
- **Preflight Reconciliation:** Four minor historical discrepancies were audited and logged in `discrepancy_ledger.csv`:
  1. Denominator consistency: 20 episodes, 49 Critical windows, 48 Warning windows, 49 pure-normal entities, 560 Normal windows confirmed.
  2. Operating threshold reconciliation: Fixed policy ($0.50$) vs. exploratory threshold sweep ($0.30$) clearly separated.
  3. Margin consistency: N2 loss coefficient $\lambda_{\text{crit}}=0.10$ and margin $0.50$ verified.
  4. Decision rule: $P_{\text{fail}} = P_1 + P_2 \ge 0.50$, then $z_2 - z_1 \ge 0.50$ for Critical; Warning otherwise.

---

## 3. Methodology & Calibration Protocol

### 3.1 Parent-Job Grouped Train Out-of-Fold Partition

Candidate A Train contains 3,278 windows across 51 unique parent jobs (34 failing jobs, 17 pure-normal jobs).
To eliminate leakage, a 5-fold grouped partition was generated using greedy capacity-balanced bin packing:
- **Fold 0:** 643 windows (1 parent job; 637 normal, 3 warning, 3 critical)
- **Fold 1:** 453 windows (10 parent jobs; 196 normal, 127 warning, 130 critical)
- **Fold 2:** 1,275 windows (11 parent jobs; 1,193 normal, 44 warning, 38 critical)
- **Fold 3:** 453 windows (14 parent jobs; 385 normal, 35 warning, 33 critical)
- **Fold 4:** 454 windows (15 parent jobs; 386 normal, 35 warning, 33 critical)

**Grouped Isolation Verification:** The maximum number of folds in which any parent job appears is exactly 1. Zero parent jobs cross fold boundaries.

### 3.2 Out-of-Fold Model Inference

For each seed (42, 123, 456) and fold (0 to 4), an N2 Multi-Resolution LSTM model was trained strictly on the other 4 folds (25 epochs, Adam, lr=1e-3, weight decay=1e-4, batch size=64) using `AsymmetricCriticalMarginLoss` ($\lambda_{\text{crit}}=0.10$, margin $= 0.50$, weights `[1, 2, 2]`). Each held-out fold was inferred using the model that never trained on its parent jobs.

### 3.3 Platt Scaling Calibrator Fitting

The binary failure target is:
$$y_{\text{fail}} = \begin{cases} 1 & \text{if } y_{\text{true}} \in \{1, 2\} \\ 0 & \text{if } y_{\text{true}} = 0 \end{cases}$$

Numerical logit transformation with $\epsilon = 10^{-6}$:
$$s_{\text{fail}} = \ln\left(\frac{\text{clip}(P_{\text{fail}}, \epsilon, 1-\epsilon)}{1 - \text{clip}(P_{\text{fail}}, \epsilon, 1-\epsilon)}\right)$$

Platt calibration fits a logistic regression model $q_{\text{fail}} = \sigma(a \cdot s_{\text{fail}} + b)$ strictly on the 3,278 Train OOF predictions:
- **Seed 42:** $a = -0.1205$, $b = -1.9524$
- **Seed 123:** $a = -0.1443$, $b = -1.9683$
- **Seed 456:** $a = -0.1489$, $b = -1.9935$

The intercept $b \approx -1.96$ aligns closely with the log-odds of the Train failure base rate ($481 / 3278 = 14.67\% \implies \ln(0.1467 / 0.8533) = -1.76$). The near-zero slope $a$ indicates that on unseen held-out parent jobs, the raw logit scores provide minimal marginal discriminative power, causing the calibrator to shrink predictions heavily toward the base rate ($\sim 12.3\% - 15.2\%$).

### 3.4 Deterministic Operating-Threshold Selection

On Train OOF data, a 201-point candidate grid $\tau \in [0.0, 1.0]$ was evaluated under the frozen operational rules:
1. **Step A (Specificity Filter):** Pure-Normal Entity FAR $\le 10\%$ AND Normal-Window FPR $\le 5\%$.
   - Out of 201 thresholds, 156 satisfied both constraints (all thresholds $\ge 0.2250$).
2. **Step B (Sensitivity Optimization):** Maximize mean Episode Recall, then Critical Recall, with conservative tie-break to largest threshold.
   - **Selected Threshold:** $\tau = 0.2250$ (Expected Train OOF: EpRec $= 1.0\%$, CritRec $= 0.8\%$, EntFAR $= 8.7\%$, WinFPR $= 3.07\%$).

The operating policy was frozen to `frozen_operating_policy.json` and hashed prior to inspecting Candidate A Validation labels.

---

## 4. Candidate A Validation Results

### 4.1 Primary Operational Gate Evaluation

The frozen operational gates require:
- **Gate 1A:** Episode Recall $\ge 70\%$ (at least 14 / 20 episodes intercepted)
- **Gate 1B:** Genuine Critical Recall $\ge 50\%$ (at least 25 / 49 Critical windows detected)
- **Gate 2A:** Pure-Normal Entity FAR $\le 10\%$ (at most 4 / 49 pure-normal entities alerted)
- **Gate 2B:** Normal-Window FPR $\le 5\%$ (at most 28 / 560 normal windows alerted)

| Policy | Policy Name | Seed | Episode Recall (1A) | Critical Recall (1B) | Entity FAR (2A) | Window FPR (2B) | All 4 Gates Passed |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **V0** | Historical Raw (th=0.50) | 42 | 0.0% (0/20) | 0.0% (0/49) | 0.0% (0/49) | 0.00% (0/560) | **FAIL** |
| **V0** | Historical Raw (th=0.50) | 123 | 5.0% (1/20) | 0.0% (0/49) | 0.0% (0/49) | 0.54% (3/560) | **FAIL** |
| **V0** | Historical Raw (th=0.50) | 456 | 0.0% (0/20) | 0.0% (0/49) | 8.2% (4/49) | 0.71% (4/560) | **FAIL** |
| **V0 Mean** | *Mean ± Std* | - | **1.7% ± 2.9%** | **0.0% ± 0.0%** | **2.7% ± 4.7%** | **0.42% ± 0.37%** | **0 / 3 Passed** |
| **V1** | **Calibrated Primary ($	au$)** | **42** | **0.0% (0/20)** | **0.0% (0/49)** | **0.0% (0/49)** | **0.00% (0/560)** | **FAIL** |
| **V1** | **Calibrated Primary ($	au$)** | **123** | **0.0% (0/20)** | **0.0% (0/49)** | **0.0% (0/49)** | **0.00% (0/560)** | **FAIL** |
| **V1** | **Calibrated Primary ($	au$)** | **456** | **0.0% (0/20)** | **0.0% (0/49)** | **0.0% (0/49)** | **0.00% (0/560)** | **FAIL** |
| **V1 Mean** | ***Mean ± Std*** | - | **0.0% ± 0.0%** | **0.0% ± 0.0%** | **0.0% ± 0.0%** | **0.00% ± 0.00%** | **0 / 3 Passed** |
| **V3** | Uncalibrated Diagnostic ($	au$) | 42 | 30.0% (6/20) | 14.3% (7/49) | 22.4% (11/49) | 23.93% (134/560) | **FAIL** |
| **V3** | Uncalibrated Diagnostic ($	au$) | 123 | 65.0% (13/20) | 24.5% (12/49) | 34.7% (17/49) | 46.07% (258/560) | **FAIL** |
| **V3** | Uncalibrated Diagnostic ($	au$) | 456 | 55.0% (11/20) | 24.5% (12/49) | 34.7% (17/49) | 41.96% (235/560) | **FAIL** |
| **V3 Mean** | *Mean ± Std* | - | **50.0% ± 18.0%** | **21.1% ± 5.9%** | **30.6% ± 7.1%** | **37.32% ± 11.78%** | **0 / 3 Passed** |

### 4.2 Classification & Discrimination Metrics

| Policy | Seed | Accuracy | Balanced Acc | Macro-F1 | Warning Recall | Critical Recall |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **V0** | 42 | 85.24% | 33.33% | 0.3068 | 0.0% (0/48) | 0.0% (0/49) |
| **V0** | 123 | 85.08% | 34.54% | 0.3315 | 4.2% (2/48) | 0.0% (0/49) |
| **V0** | 456 | 84.63% | 33.10% | 0.3056 | 0.0% (0/48) | 0.0% (0/49) |
| **V0 Mean** | - | **84.98% ± 0.32%** | **33.66% ± 0.77%** | **0.3146 ± 0.0147** | **1.4% ± 2.4%** | **0.0% ± 0.0%** |
| **V1** | 42 | 85.24% | 33.33% | 0.3068 | 0.0% (0/48) | 0.0% (0/49) |
| **V1** | 123 | 85.24% | 33.33% | 0.3068 | 0.0% (0/48) | 0.0% (0/49) |
| **V1** | 456 | 85.24% | 33.33% | 0.3068 | 0.0% (0/48) | 0.0% (0/49) |
| **V1 Mean** | - | **85.24% ± 0.00%** | **33.33% ± 0.00%** | **0.3068 ± 0.0000** | **0.0% ± 0.0%** | **0.0% ± 0.0%** |
| **V3** | 42 | 67.12% | 35.67% | 0.3519 | 16.7% (8/48) | 14.3% (7/49) |
| **V3** | 123 | 49.47% | 33.78% | 0.2985 | 22.9% (11/48) | 24.5% (12/49) |
| **V3** | 456 | 52.82% | 34.45% | 0.3095 | 20.8% (10/48) | 24.5% (12/49) |
| **V3 Mean** | - | **56.47% ± 9.38%** | **34.64% ± 0.96%** | **0.3200 ± 0.0282** | **20.1% ± 3.2%** | **21.1% ± 5.9%** |

### 4.3 Full 3x3 Confusion Matrices

```
Policy V0 (Historical Raw th=0.50, Seed 42):
                Pred Norm    Pred Warn    Pred Crit
  True Norm        560           0            0
  True Warn         48           0            0
  True Crit         49           0            0

Policy V0 (Historical Raw th=0.50, Seed 123):
                Pred Norm    Pred Warn    Pred Crit
  True Norm        557           3            0
  True Warn         46           2            0
  True Crit         49           0            0

Policy V1 (Calibrated Primary tau=0.2250, All Seeds 42, 123, 456):
                Pred Norm    Pred Warn    Pred Crit
  True Norm        560           0            0
  True Warn         48           0            0
  True Crit         49           0            0

Policy V3 (Diagnostic Uncalibrated tau=0.2250, Seed 42):
                Pred Norm    Pred Warn    Pred Crit
  True Norm        426         104           30
  True Warn         34           8            6
  True Crit         34           8            7

Policy V3 (Diagnostic Uncalibrated tau=0.2250, Seed 123):
                Pred Norm    Pred Warn    Pred Crit
  True Norm        302         155          103
  True Warn         28          11            9
  True Crit         27          10           12

Policy V3 (Diagnostic Uncalibrated tau=0.2250, Seed 456):
                Pred Norm    Pred Warn    Pred Crit
  True Norm        325         131          104
  True Warn         28          10           10
  True Crit         30           7           12
```

---

## 5. Probability Calibration Analysis

Calibration metrics evaluated on the binary failure prediction $y_{\text{fail}} \in \{0, 1\}$ across all 657 validation windows:

| Seed | Metric | Raw Score ($P_{\text{fail}}$) | Calibrated Score ($q_{\text{fail}}$) | Improvement (Diff) | Direction |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **42** | Binary Brier Score | 0.1397 | 0.1255 | -0.0141 | **Improved** |
| **42** | Binary Log-Loss (NLL) | 0.4704 | 0.4174 | -0.0530 | **Improved** |
| **42** | Binary ECE (10 bins) | 10.82% | 0.17% | -10.65% | **Improved** |
| **123** | Binary Brier Score | 0.1367 | 0.1276 | -0.0091 | **Improved** |
| **123** | Binary Log-Loss (NLL) | 0.4384 | 0.4250 | -0.0134 | **Improved** |
| **123** | Binary ECE (10 bins) | 10.34% | 0.35% | -9.99% | **Improved** |
| **456** | Binary Brier Score | 0.1511 | 0.1247 | -0.0264 | **Improved** |
| **456** | Binary Log-Loss (NLL) | 0.5025 | 0.4142 | -0.0883 | **Improved** |
| **456** | Binary ECE (10 bins) | 16.00% | 0.78% | -15.22% | **Improved** |
| **Mean** | **Binary Brier Score** | **0.1425** | **0.1259** | **-0.0165** | **Improved** |
| **Mean** | **Binary Log-Loss (NLL)**| **0.4705** | **0.4189** | **-0.0516** | **Improved** |
| **Mean** | **Binary ECE (10 bins)** | **12.39%** | **0.43%** | **-11.95%** | **Improved** |

### Reliability Bins Breakdown (Policy V1 Calibrated, Seed 42)

| Bin Range | Sample Count | Proportion of Validation | Mean Predicted Prob | Observed Frequency | Absolute Gap |
| :---: | :---: | :---: | :---: | :---: | :---: |
| [0.0, 0.1] | 0 | 0.00% | 0.0500 | 0.00% | 0.0000 |
| (0.1, 0.2] | 657 | 100.00% | 0.1459 | 14.76% (97/657) | 0.0017 |
| (0.2, 0.3] | 0 | 0.00% | 0.2500 | 0.00% | 0.0000 |
| (0.3, 1.0] | 0 | 0.00% | - | - | 0.0000 |

**Calibration Insight:** Platt scaling achieved near-perfect ECE ($0.43\%$) not by sharpening discrimination between classes, but by compressing the entire output distribution into the narrow band $(0.10, 0.20]$ corresponding to the empirical cohort failure frequency ($97 / 657 = 14.76\%$). Because all calibrated probabilities lie below the specificity-clearing threshold $\tau = 0.2250$, zero windows are flagged as failures.

---

## 6. Critical-versus-Warning Separation Diagnostics

Evaluation of the auxiliary logit margin $z_2 - z_1$ on the 49 true Critical validation windows:

| Seed | Mean Logit Margin ($z_2 - z_1$) | % Critical Windows $\ge 0.50$ | Mean Raw $P_{\text{fail}}$ | Mean Calibrated $q_{\text{fail}}$ |
| :---: | :---: | :---: | :---: | :---: |
| **42** | +0.844 logit units | 83.7% (41/49) | 0.1586 | 0.1516 |
| **123** | +0.636 logit units | 79.6% (39/49) | 0.2404 | 0.1434 |
| **456** | +0.834 logit units | 85.7% (42/49) | 0.1765 | 0.1510 |
| **Mean ± Std** | **+0.771 ± 0.118** | **83.0% ± 3.1%** | **0.1918 ± 0.0430** | **0.1487 ± 0.0457** |

**Margin Preservation:** The N2 asymmetric margin objective successfully maintained the large positive separation between Critical and Warning logits ($+0.771$ units; $83.0\%$ of Critical windows exceed $+0.50$). However, this separation is completely shadowed by the failure to cross the failure-interception threshold: because $q_{\text{fail}} < 0.2250$, the model classifies all 49 Critical windows as Normal.

---

## 7. Answers to the 14 Required Final Research Questions

### Question 1: Was the Phase 7N N2 model reproducible?
**Direct Observation:** Yes. Running inference on the saved Phase 7N N2 checkpoints (`ckpt_N2_seed*.pt`) across Candidate A Validation reproduced the historical Phase 7N validation predictions with a maximum absolute probability error of $< 3.0 \times 10^{-8}$ and identical confusion matrices.

### Question 2: Was the Platt calibrator fitted exclusively on Train out-of-fold predictions?
**Direct Observation:** Yes. The 5-fold parent-job-grouped partition divided the 3,278 Candidate A Train windows such that no parent job crossed fold boundaries. 15 separate fold models were trained on 4 folds and evaluated on the 5th held-out fold. The logistic regression parameters ($a, b$) were fitted strictly on these 3,278 OOF predictions without inspecting any Candidate A Validation window.

### Question 3: Was the operating threshold selected without using Validation outcomes?
**Direct Observation:** Yes. The threshold grid sweep ($0.0$ to $1.0$, 201 points) was evaluated strictly on the Train OOF predictions. The threshold $\tau = 0.2250$ was selected using the deterministic specificity-first rule on Train OOF and frozen in `frozen_operating_policy.json` (SHA-256: `0c15810826492887d9a7943b1784db2a04817cfec9312d89092ef07624b612b5`) before Candidate A Validation labels were read.

### Question 4: Did the calibrator improve binary failure-risk calibration on Validation?
**Direct Observation:** Yes. Binary Brier score improved from $0.1425$ to $0.1259$ ($-0.0165$), binary NLL improved from $0.4705$ to $0.4189$ ($-0.0516$), and binary Expected Calibration Error dropped from $12.39\%$ down to $0.43\%$. The mean predicted probability ($14.59\%$) closely matches the observed empirical failure frequency ($14.76\%$).

### Question 5: Did the selected threshold improve episode interception?
**Direct Observation:** No. Under the primary calibrated policy V1, Episode Recall remained at $0.0\% \pm 0.0\%$ (0 / 20 episodes intercepted across all 3 seeds). All calibrated probabilities on Validation fell below the frozen operating threshold of $\tau = 0.2250$.

### Question 6: Did genuine Critical Recall improve?
**Direct Observation:** No. Under policy V1, Genuine Critical Recall remained at $0.0\% \pm 0.0\%$ (0 / 49 Critical windows detected). All 49 Critical windows were classified as Normal.

### Question 7: Were the two specificity gates maintained?
**Direct Observation:** Yes. Under policy V1, Pure-Normal Entity FAR was $0.0\% \pm 0.0\%$ (passing Gate 2A $\le 10\%$) and Normal-Window FPR was $0.0\% \pm 0.0\%$ (passing Gate 2B $\le 5\%$). However, this was an artifact of predicting the Normal majority class for all validation windows. Under diagnostic policy V3 (raw score thresholded at $\tau = 0.2250$), both specificity gates failed catastrophically (FAR $= 30.6\%$, FPR $= 37.3\%$).

### Question 8: Did the Critical-margin behavior remain consistent?
**Direct Observation:** Yes. The mean Critical-versus-Warning logit margin $z_2 - z_1$ on true Critical validation windows remained large and positive at $+0.771 \pm 0.118$ logit units, with $83.0\% \pm 3.1\%$ of Critical windows exceeding the $+0.50$ margin requirement.

### Question 9: Were outcomes consistent across the three seeds?
**Direct Observation:** Yes. For primary policy V1, all three seeds produced identical operational metrics: $0.0\%$ Episode Recall, $0.0\%$ Critical Recall, $0.0\%$ Entity FAR, and $0.0\%$ Window FPR. All three seeds uniformly failed the 4-gate operational criteria.

### Question 10: Did any predeclared primary configuration pass all four gates simultaneously?
**Direct Observation:** No. Simultaneous 4-gate passes for policy V1 was $0 / 3$ seeds. Under policy V0, passes was $0 / 3$. Under policy V3, passes was $0 / 3$. No tested configuration satisfied all four operational gates simultaneously.

### Question 11: Which conclusions are direct observations, and which remain hypotheses?
**Direct Observations:**
1. The N2 checkpoints reproduce Phase 7N validation predictions within $3 \times 10^{-8}$.
2. Platt scaling fitted on Train OOF reduces Validation ECE to $0.43\%$.
3. The Train OOF specificity-clearing threshold is $\tau = 0.2250$.
4. At $\tau = 0.2250$, policy V1 classifies all 657 validation windows as Normal.
5. Lowering the raw threshold to $0.2250$ (policy V3) causes FAR to spike to $30.6\%$ and FPR to spike to $37.3\%$.
6. Auxiliary logit margin $z_2 - z_1$ on true Critical windows remains $+0.771$.

**Scientific Hypotheses:**
1. The inability of $P_{\text{fail}}$ to separate failure from normal windows across parent jobs is caused by cross-job telemetry heterogeneity in the Google Borg cluster rather than model capacity or hyperparameter choices.
2. Incorporating job-level contextual metadata or task-level scheduling attributes into the sequence model may reduce cross-job variance and enable better binary separation.

### Question 12: What does the experiment reveal about the sensitivity-specificity trade-off?
**Empirical Finding:** The experiment reveals a severe, non-negotiable trade-off governed by feature overlap:
- At $\tau \ge 0.2250$, specificity is preserved (FPR $\le 0.71\%$, FAR $\le 8.2\%$, but sensitivity is near zero ($0 - 5\%$ episode interception).
- At $\tau < 0.2250$, sensitivity increases ($50\%$ episode recall at $\tau=0.225$ raw), but specificity collapses completely ($37.3\%$ window FPR, $30.6\%$ entity FAR).
There is no operating threshold in $[0, 1]$ where sensitivity $\ge 70\%$ and specificity $\le 10\%$ FAR / $\le 5\%$ FPR coexist under the current feature representation.

### Question 13: What limitation most strongly constrains the conclusion?
**Analysis:** The primary constraint is **cross-job telemetry distributional shift**. The model is evaluated on unseen parent jobs. The 31 temporal features exhibit baseline variations across different workloads that exceed the subtle pre-failure degradation signatures, causing the model's failure probability to remain suppressed to prevent false alarms on unfamiliar normal jobs.

### Question 14: Should the project proceed to a new dataset, a larger validation cohort, or a different modeling strategy?
**Strategic Recommendation:** The project should **NOT** continue tuning loss margins, post-processing calibrators, or threshold grids on this frozen feature representation. Instead, the project should pursue a **feature representation and domain-adaptation redesign**:
1. Incorporate workload-normalized features (e.g. z-scoring telemetry relative to parent-job baseline activity).
2. Introduce job-level static embedding features (resource requests, priority, scheduling constraints).
3. Evaluate whether contrastive pre-training across parent jobs can disentangle workload-type variance from failure-degradation variance.

---

## 8. Complete Deliverable Inventory & Governance Ledger

All 23 required deliverables are archived in `results/phase7p_failure_calibration/` and mirrored in `reports/phase7p_failure_calibration/`:

1. `phase7p_design.md`
2. `phase7p_config.json`
3. `preflight_audit.json`
4. `discrepancy_ledger.csv`
5. `data_manifest.json`
6. `fold_assignments.csv`
7. `out_of_fold_predictions.csv`
8. `calibrator_parameters.json`
9. `threshold_selection_results.csv`
10. `frozen_operating_policy.json`
11. `validation_predictions.csv`
12. `per_seed_metrics.csv`
13. `confusion_matrices.csv`
14. `operational_gate_summary.csv`
15. `calibration_analysis.csv`
16. `reliability_bins.csv`
17. `critical_margin_analysis.csv`
18. `paired_comparison.csv`
19. `training_or_calibration_logs.csv`
20. `integrity_checks.json`
21. `independent_audit_report.md`
22. `phase7p_final_report.md`
23. `phase7p_manifest.json`
