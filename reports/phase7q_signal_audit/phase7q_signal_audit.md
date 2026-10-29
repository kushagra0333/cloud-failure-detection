# Phase 7Q — Workstream A: Cross-Job Predictive Signal & Calibration Audit

**Auditor:** Antigravity Senior ML Research & Independent Audit Group  
**Target Environment:** `C:\Users\hp\Desktop\Research project`  
**Evaluation Scope:** Candidate A Train ($N=3,278$, 51 parent jobs) & Candidate A Validation ($N=657$, 20 episodes, 49 pure-normal entities)  
**Governing Inputs:** Phase 7P Out-of-Fold Predictions, Calibrator Parameters, and Validation Predictions  
**Audit Finding Verdict:** `RANKING_SIGNAL_WEAK_OR_INCONSISTENT`

---

## 1. Executive Summary: Why Phase 7P Calibration Failed

Phase 7P investigated whether binary failure-risk calibration (Platt scaling) and a training-only threshold selection procedure could resolve the threshold-interception deficit from Phase 7N. However, the Phase 7P Platt calibrators produced negative slopes ($a \in [-0.15, -0.12]$), yielded a Validation Brier score virtually identical to a static constant-prevalence baseline, and predicted Normal for all 657 validation windows under the frozen threshold $\tau = 0.2250$.

This audit independently investigated the mathematical and empirical causes of this outcome:

1. **Prevalence-Only Collapse:**  
   The binary failure prevalence on Candidate A Train is $481 / 3278 = 14.6736\%$, and on Candidate A Validation it is $97 / 657 = 14.7641\%$. A naive baseline predicting a constant probability $p = 0.146736$ for every window achieves a Brier score of:
   $$\text{Brier}_{\text{const}} = \frac{1}{N} \sum_{i=1}^N (y_i - p)^2 = p^2(1 - \text{prev}) + (1 - p)^2(\text{prev}) \approx 0.125844$$
   The Phase 7P calibrated Brier score on Validation was $0.125955 \pm 0.001487$. The apparent Brier "improvement" over the raw score ($0.1425 \to 0.1259$) was **100% attributable to contracting predictions to the cohort base rate**, not to discriminating failure-associated windows.

2. **Negative Slope and Ranking Inversion:**  
   When the Platt slope $a < 0$, the logistic mapping $q = \sigma(a \cdot s + b)$ is strictly decreasing. Consequently, higher raw failure scores $P_{\text{fail}}$ produce *lower* calibrated probabilities $q_{\text{fail}}$, inverting the ranking of distinct raw scores. On Validation, Seed 123's raw AUC of $0.5382$ was inverted to $0.4618$. Across seeds, the baseline N2 raw score has an average Validation AUC of only $0.4655$, demonstrating no reliable discriminatory ranking across held-out parent jobs.

3. **Fold Imbalance and Cross-Job Distribution Shift:**  
   In Phase 7P, Fold 0 contained 643 windows from a single parent job (Job 3998246322) with only 6 failures (prevalence $0.93\%$) and an inverted raw AUC of $0.2611$ ($a = -0.7218$). Furthermore, Fold 1 had a prevalence of $56.73\%$ (AUC $0.4170$, $a = -0.2870$) and Fold 2 had 1,275 windows with prevalence $6.43\%$ (AUC $0.4664$, $a = -0.0941$). These three distorted folds accounted for $72.3\%$ of the training data. Although Folds 3 and 4 exhibited positive slopes ($a = +0.30, +0.59$) and positive AUCs ($0.65 - 0.70$), the pooled Train OOF data was heavily dominated by cross-job telemetry shifts where normal windows in large jobs received higher raw failure scores than failure windows in other jobs.

---

## 2. Re-Audit of Probability Calibration and Prevalence Baselines

### 2.1 Comparison Against Static Prevalence Predictor

| Metric | Raw N2 Score ($P_{\text{fail}}$) | Calibrated Score ($q_{\text{fail}}$) | Constant Train-Prevalence Baseline ($p=0.1467$) | Difference (Calibrated - Prevalence Baseline) |
| :--- | :---: | :---: | :---: | :---: |
| **Validation Brier (Seed 42)** | 0.139651 | 0.125539 | 0.125844 | -0.000305 |
| **Validation Brier (Seed 123)** | 0.136677 | 0.127593 | 0.125844 | +0.001749 |
| **Validation Brier (Seed 456)** | 0.151114 | 0.124733 | 0.125844 | -0.001111 |
| **Validation Brier Mean** | **0.142481** | **0.125955** | **0.125844** | **+0.000111** |
| **Validation NLL Mean** | **0.470453** | **0.418871** | **0.418598** | **+0.000273** |
| **Validation ECE Mean** | **12.387%** | **0.432%** | **0.080%** | **+0.352%** |

**Scientific Interpretation:** The calibrated score does not outperform a constant prevalence baseline. Its mean Brier score is actually $0.000111$ worse than simply predicting the training cohort failure prevalence for every window.

---

### 2.2 Discrimination and Ranking Metrics on Validation

| Seed | Raw ROC-AUC | Calibrated ROC-AUC | Raw Average Precision | Calibrated Average Precision | Spearman Corr (Raw vs $y$) | Pearson Corr (Raw vs $y$) | Inversion Status |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **42** | 0.4573 | 0.5427 | 0.1415 | 0.1764 | -0.0526 | -0.0461 | Slope $a < 0$ inverts ranking |
| **123** | 0.5382 | 0.4618 | 0.1593 | 0.1384 | +0.0471 | +0.0485 | Slope $a < 0$ degrades AUC below 0.50 |
| **456** | 0.4010 | 0.5990 | 0.1279 | 0.1989 | -0.1265 | -0.1032 | Slope $a < 0$ inverts ranking |
| **Mean** | **0.4655** | **0.5345** | **0.1429** | **0.1712** | **-0.0440** | **-0.0336** | **Inconsistent / Weak Signal** |

*Note:* Empirical random-guess baseline for Average Precision is equal to failure prevalence: $97 / 657 = 0.1476$. Raw AP across seeds averages $0.1429$, which is below the random-guess prevalence baseline.

---

## 3. Fold-Dominance & Train Out-of-Fold Diagnostics

To understand why the pooled Train OOF Platt calibrator fit a negative slope, each fold of the Phase 7P partition was analyzed independently:

| Fold ID | Windows | Parent Jobs | Failing Jobs | Pure-Normal Jobs | Failure Count | Prevalence | Mean Raw AUC | Mean Fold Platt Slope ($a$) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fold 0** | 643 | 1 | 1 | 0 | 6 | 0.93% | **0.2611** | **-0.7218** |
| **Fold 1** | 453 | 10 | 8 | 2 | 257 | 56.73% | **0.4170** | **-0.2870** |
| **Fold 2** | 1,275 | 11 | 4 | 7 | 82 | 6.43% | **0.4664** | **-0.0941** |
| **Fold 3** | 453 | 14 | 10 | 4 | 68 | 15.01% | **0.6521** | **+0.3032** |
| **Fold 4** | 454 | 15 | 11 | 4 | 68 | 14.98% | **0.6978** | **+0.5879** |

### Key Diagnostic Observations:
1. **Severe Job Concentration in Fold 0:** Fold 0 consisted entirely of Job 3998246322 (643 windows, 637 normal, 3 warning, 3 critical). The model assigned elevated failure scores to the normal windows of this job when evaluated out-of-fold, producing an inverted AUC of $0.2611$ and an extreme negative slope of $a = -0.7218$.
2. **Prevalence Distortion in Fold 1 and 2:** Fold 1 had over $56.7\%$ failure prevalence (AUC $0.4170$), while Fold 2 had $1,275$ windows with only $6.4\%$ prevalence (AUC $0.4664$).
3. **Contrast with Folds 3 and 4:** In Folds 3 and 4, where jobs were well-dispersed ($14-15$ jobs per fold) and failure prevalence was near the true cohort rate ($15.0\%$), the model achieved positive AUCs ($0.6521$ and $0.6978$) and positive Platt slopes ($+0.3032$ and $+0.5879$).
4. **Effect of Excluding Fold 0:** Refitting logistic regression on Train OOF excluding Fold 0 shifted the intercept from $-1.95$ to $-1.70$ (matching the true Train log-odds of $-1.76$), but the pooled slope remained negative ($a \in [-0.18, -0.12]$) because Folds 1 and 2 still diluted the signal.

---

## 4. Workstream A Conclusion

**Official Audit Classification:**
$$\mathbf{RANKING\_SIGNAL\_WEAK\_OR\_INCONSISTENT}$$

**Justification:**
The raw N2 failure-risk score does not establish a consistent or robust failure-ranking signal across unseen parent jobs. While localized discrimination exists when evaluating balanced multi-job cohorts (Folds 3 and 4: AUC $\sim 0.67$), cross-job baseline shifts across large dominant parent jobs degrade overall validation ranking to near-chance levels (Validation AUC $= 0.4655$, AP $= 0.1429$ vs prevalence $0.1476$). Platt calibration contracted scores to the cohort prevalence without improving discrimination and inverted the score ranking due to a negative fitted slope.

This empirical finding directly motivates **Workstream B**: investigating whether a causal workload-relative deviation feature branch can remove job-specific baseline offsets and restore genuine ranking signal across parent jobs.
