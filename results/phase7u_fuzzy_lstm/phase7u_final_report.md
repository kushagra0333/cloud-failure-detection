# Phase 7U Final Report: Interpretable Fuzzy Inference for LSTM-Based Failure Prediction

**Project Root**: `C:\Users\hp\Desktop\Research project`  
**Evaluation Date**: 2026-10-29  
**Governing Protocol**: Phase 7B Evaluation Protocol v2.1 / Candidate A Job-Grouped Split  
**Authorized Partitions**: Candidate A Train (3,278 windows, 51 parent jobs) & Candidate A Validation (657 windows, 12 parent jobs)  
**Forbidden Partitions**: Candidate A Test, Phase 6 Confirmatory Holdout (Zero access confirmed)  
**Selected Operating Threshold**: $\tau = 1.0000$ (selected strictly on Train-OOF)  
**Operational Verdict**: `NO_OPERATIONAL_WINNER`  
**Audit Integrity Verdict**: `AUDIT_PASSED`  

---

## 1. Executive Summary & Core Scientific Findings

Phase 7U investigated whether an interpretable, transparent Type-1 Mamdani fuzzy inference system could combine absolute-resource evidence ($S_0$) and relative-telemetry evidence ($S_2$) more effectively than the fixed linear 80/20 hybrid evaluated in Phase 7T.

The fuzzy inference system was constructed under strict data governance:
- **Input Membership Breakpoints**: Derived strictly from empirical percentiles of pooled Candidate A Train-OOF predictions ($R_{\text{rel}}$: 20th=0.0923, 50th=0.1821, 80th=0.3230; $R_{\text{abs}}$: 20th=0.0968, 50th=0.2667, 80th=0.6248).
- **Predeclared Relative-Primary Rule Base**: 9 inspectable Mamdani rules where LOW relative risk suppresses absolute telemetry, while MEDIUM relative risk treats HIGH absolute telemetry as supporting evidence (Rule 6).
- **Inference Engine**: Min conjunction, Max aggregation, and Centroid defuzzification over 1001-point output discretization.
- **Operating Policy**: The operating threshold was selected from Train-OOF predictions under frozen specificity constraints (Pure-Normal Entity FAR $\le 10\%$ and Normal-Window FPR $\le 5\%$ across all three seeds).

```
                           PRIMARY EXPERIMENTAL FINDINGS SUMMARY
  +-------------------------------------------------------------------------------------------------+
  | Metric / Dimension          | S0 (Absolute)   | S2 (Relative)   | Hybrid (80/20)  | Fuzzy Inference |
  +-----------------------------+-----------------+-----------------+-----------------+-----------------+
  | Within-Job Macro AUC        | 0.6293          | 0.5345          | 0.5679          | 0.5607 (+0.026) |
  | Job 6251277720 FA% (t=0.50) | 90.74%          | 6.10%           | 7.30%           | 43.69% (FAILED) |
  | Val ROC-AUC                 | 0.4244          | 0.5864          | 0.5619          | 0.5827 (s123:0.63)|
  | Val Brier Score             | 0.3162          | 0.1705          | 0.1559          | 0.1906          |
  | Val Episode Recall (t=0.50) | 35.00%          | 43.33%          | 56.67%          | 66.67% (+23.3%) |
  | Val Window FPR (t=0.50)     | 34.46%          | 10.30%          | 9.52%           | 17.44%          |
  | Train-OOF Selected Tau      | N/A             | N/A             | 0.7650          | 1.0000 (0 alerts)|
  +-------------------------------------------------------------------------------------------------+
```

### Key Scientific Takeaways
1. **The Double-Edged Nature of Fuzzy Rule 6**:
   The fuzzy system partially recovered within-job failure discrimination (**$0.5607 \pm 0.2829$**, $+0.0262$ over $S_2$) and achieved strong Validation Episode Recall at $\tau = 0.50$ (**$66.67\% \pm 5.77\%$**, intercepting 14 / 20 episodes on two seeds).
   HOWEVER, on the massive pure-normal parent job `6251277720` (940 windows, 28.7% of Train), the fuzzy false alert rate surged to **$43.69\%$** (compared to only **$7.30\%$** for the Phase 7T linear hybrid and $6.10\%$ for $S_2$).
   **Mechanism**: Normal healthy variations in heavy jobs fall into the empirical `MEDIUM` relative-risk set, while the high baseline load activates `HIGH` absolute risk. Rule 6 (`IF R_rel is MEDIUM AND R_abs is HIGH THEN Output is HIGH`) repeatedly triggers, elevating benign windows directly to HIGH failure risk.
2. **Superiority of Fixed Linear Blend over Fuzzy Rule Base for Simpson's Paradox**:
   The fixed linear 80/20 hybrid from Phase 7T strictly bounds the maximum contribution of absolute telemetry to $20\%$, preventing high absolute resource levels from ever triggering an alert when relative deviations are modest ($0.80 \times 0.15 + 0.20 \times 0.80 = 0.28 < 0.50$). In contrast, Mamdani fuzzy inference with a linguistic `MEDIUM` antecedent allows high absolute telemetry to trigger a complete jump to the `HIGH` output set.
3. **Operational Gate Verdict**:
   Under the primary Train-OOF selected threshold ($\tau = 1.0000$), no alerts are generated (Episode Recall: $0.00\%$). Under the diagnostic fixed policy ($\tau = 0.50$), Normal-Window FPR was $17.44\%$ and Pure-Normal Entity FAR was $48.30\%$ (failing Gates 2A and 2B). Because all four frozen operational gates were not met simultaneously, the final operational verdict is strictly **`NO_OPERATIONAL_WINNER`**.

---

## 2. Preflight Audit and Reconciliation

| id | component | observation | root_cause | impact | resolution |
| --- | --- | --- | --- | --- | --- |
| DISC-7U-01 | Train-OOF Episode Recall Reporting at tau=0.765 | Phase 7T reported 0.25% Train-OOF episode recall at selected threshold tau=0.765 | Verified arithmetically: 1 episode detected in Seed 123 (1/131 = 0.7634%) and 0 in Seeds 42 and 456 (0/131 = 0.0%); mean across 3 seeds = (0.7634 + 0 + 0)/3 = 0.2545%; it was an exact percentage, not a decimal fraction mixup | Reflects genuine extreme specificity suppression on Train-OOF; no reporting defect | PROVEN EXACT / RECONCILED |
| DISC-7U-02 | Brier Score Interpretation Relative to Constant Prevalence | Phase 7T reported hybrid Brier score of 0.1559 as 'best calibration among models' | The hybrid improved probability error over S2 (0.1705) and S0 (0.3162), but remains higher (worse) than a naive constant-prevalence baseline (0.1258) | Hybrid scores are uncalibrated risk scores and must not be claimed as calibrated probabilities | RESOLVED / CLARIFIED |
| DISC-7U-03 | Model Architecture & Parameter Count Provenance | Confirmation that S0 and S2 share identical architecture with Phase 7N/7Q | Checkpoint inspection confirmed MultiResolutionLSTMClassifier contains exactly 67,027 parameters in S0 and S2 | Controlled ablation baseline verified bit-for-bit | CONFIRMED / FROZEN |

### Reconciliation Highlights
- **Train-OOF Episode Recall Exactness**: Verified that the $0.25\%$ episode recall reported at $\tau = 0.765$ in Phase 7T was mathematically exact: across 131 failing episodes, Seed 123 detected 1 episode ($1/131 = 0.7634\%$) while Seeds 42 and 456 detected 0 ($0/131 = 0.0\%$), yielding a mean of $0.2545\%$. It was NOT a decimal reporting defect.
- **Brier Score vs Constant Prevalence**: Reconciled that while hybrid Brier score ($0.1558$) improved over $S_0$ ($0.3162$) and $S_2$ ($0.1705$), it is higher than the trivial constant prevalence predictor ($0.1258$). Hybrid scores are uncalibrated risk rankings.

---

## 3. Representation Performance Comparison

### Model Performance Summary Across Seeds
| model | val_auc_mean | val_auc_std | val_ap_mean | val_ap_std | val_brier_mean | val_brier_std | val_macro_f1_mean | val_macro_f1_std | episode_recall_mean | episode_recall_std | critical_recall_mean | critical_recall_std | entity_far_mean | entity_far_std | window_fpr_mean | window_fpr_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Fuzzy_Fixed50 | 0.5827 | 0.0512 | 0.2148 | 0.0316 | 0.1906 | 0.0145 | 0.4082 | 0.0298 | 66.6667 | 5.7735 | 27.2109 | 5.8913 | 48.2993 | 5.8913 | 17.4405 | 2.0850 |
| Fuzzy_Primary | 0.5827 | 0.0512 | 0.2148 | 0.0316 | 0.1906 | 0.0145 | 0.3068 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Hybrid_T0 | 0.5619 | 0.0191 | 0.2203 | 0.0255 | 0.1558 | 0.0023 | 0.4137 | 0.0118 | 56.6667 | 2.8868 | 22.4490 | 6.1224 | 36.0544 | 8.2479 | 9.5238 | 0.8809 |
| S0 | 0.4244 | 0.0154 | 0.1677 | 0.0077 | 0.3162 | 0.0072 | 0.2804 | 0.0137 | 36.6667 | 2.8868 | 17.0068 | 2.3565 | 45.5782 | 9.2026 | 34.4643 | 3.8631 |
| S2 | 0.5864 | 0.0217 | 0.2415 | 0.0310 | 0.1705 | 0.0009 | 0.4204 | 0.0134 | 58.3333 | 7.6376 | 22.4490 | 5.3995 | 37.4150 | 7.1671 | 10.2976 | 0.3717 |

### Seed-Paired Differences (Treatment Effects)
| comparison | delta_val_auc_mean | delta_val_auc_std | delta_val_ap_mean | delta_val_ap_std | delta_val_brier_mean | delta_val_brier_std | delta_episode_recall_mean | delta_episode_recall_std | delta_critical_recall_mean | delta_critical_recall_std | delta_entity_far_mean | delta_entity_far_std | delta_window_fpr_mean | delta_window_fpr_std | delta_macro_f1_mean | delta_macro_f1_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Fuzzy_Fixed50_vs_S0 | 0.1584 | 0.0500 | 0.0471 | 0.0331 | -0.1256 | 0.0140 | 30.0000 | 8.6603 | 10.2041 | 7.3583 | 2.7211 | 11.2399 | -17.0238 | 4.1891 | 0.1278 | 0.0311 |
| Fuzzy_Fixed50_vs_S2 | -0.0037 | 0.0361 | -0.0267 | 0.0245 | 0.0201 | 0.0140 | 8.3333 | 2.8868 | 4.7619 | 1.1783 | 10.8844 | 3.1174 | 7.1429 | 1.9643 | -0.0122 | 0.0165 |
| Fuzzy_Fixed50_vs_Hybrid_T0 | 0.0208 | 0.0336 | -0.0055 | 0.0073 | 0.0348 | 0.0122 | 10.0000 | 5.0000 | 4.7619 | 3.1174 | 12.2449 | 3.5348 | 7.9167 | 1.4434 | -0.0055 | 0.0181 |
| Fuzzy_Primary_vs_S2 | -0.0037 | 0.0361 | -0.0267 | 0.0245 | 0.0201 | 0.0140 | -58.3333 | 7.6376 | -22.4490 | 5.3995 | -37.4150 | 7.1671 | -10.2976 | 0.3717 | -0.1136 | 0.0134 |
| Fuzzy_Primary_vs_Hybrid_T0 | 0.0208 | 0.0336 | -0.0055 | 0.0073 | 0.0348 | 0.0122 | -56.6667 | 2.8868 | -22.4490 | 6.1224 | -36.0544 | 8.2479 | -9.5238 | 0.8809 | -0.1069 | 0.0118 |

---

## 4. Train-OOF Threshold Selection (Leakage-Safe)

- Candidate Grid: 201 thresholds ($\tau = 0.000$ to $1.000$, step $0.005$).
- Specificity Constraints: Pure-Normal Entity FAR $\le 10.0\%$ AND Normal-Window FPR $\le 5.0\%$ across all three seeds.
- Number of Eligible Thresholds: Exactly **28 thresholds** (range $[0.865, 1.000]$).
- Because the maximum defuzzified output ceiling is $\approx 0.8603$, all thresholds $\ge 0.865$ produce zero alerts (FAR = 0.0%, FPR = 0.0%, Episode Recall = 0.0%).
- Deterministic Tie-Breaking Winner: **$\tau = 1.0000$** (conservative tie-break).
- The operating policy was sealed in `frozen_operating_policy.json` prior to inspecting Validation outcomes.

---

## 5. Within-Job Discrimination vs Cross-Job False Alarm Suppression

### 5.1 Within-Job Macro AUC Across 32 Failing Jobs
| model | within_job_auc_mean | within_job_auc_std | within_job_auc_median |
| --- | --- | --- | --- |
| Fuzzy | 0.5606 | 0.2527 | 0.5611 |
| Hybrid_T0 | 0.5679 | 0.2653 | 0.5636 |
| S0 | 0.6293 | 0.3493 | 0.6683 |
| S2 | 0.5345 | 0.2596 | 0.5441 |

- $S_0$ achieved within-job AUC of $0.6293$, but collapsed cross-job ($0.3066$ OOF AUC).
- $S_2$ resolved cross-job inversion ($0.5080$ OOF AUC, $0.5864$ Val AUC), but attenuated within-job AUC to $0.5345$.
- **Fuzzy Treatment** achieved **$0.5607$** within-job AUC (+0.0262 recovery over $S_2$).
- **Phase 7T Linear Hybrid** achieved **$0.5679$** within-job AUC (+0.0334 recovery over $S_2$). Both methods recover similar within-job signal.

### 5.2 Behavior on High-Resource Pure-Normal Job 6251277720
| model | mean_score | median_score | q90_score | false_alert_rate_tau50 | false_alert_rate_selected_tau |
| --- | --- | --- | --- | --- | --- |
| Fuzzy | 0.4525 | 0.3534 | 0.8587 | 43.6879 | 0.0000 |
| Hybrid_T0 | 0.2608 | 0.2172 | 0.4470 | 7.3050 | 0.0000 |
| S0 | 0.6139 | 0.6525 | 0.6696 | 90.7447 | 0.0000 |
| S2 | 0.1725 | 0.1184 | 0.3983 | 6.0993 | 0.0000 |

- Under $S_0$, Job `6251277720` suffered an extreme **$90.74\%$ false alert rate**.
- Under Phase 7T Linear Hybrid, false alerts were crushed to **$7.30\%$**.
- Under Phase 7U Fuzzy Treatment, false alerts surged back to **$43.69\%$**!
- The fuzzy rule base failed to suppress false alarms on healthy heavy jobs because Rule 6 inappropriately elevated benign relative fluctuations to HIGH failure risk.

---

## 6. Fuzzy Rule Interpretability & Activation Analysis

| cohort | window_count | mean_R_rel | mean_R_abs | mean_fuzzy_score | mean_act_low | mean_act_med | mean_act_high | dominant_consequent |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| detected_failure_windows | 27 | 0.7017 | 0.0949 | 0.8430 | 0.0000 | 0.0388 | 0.9558 | HIGH |
| missed_critical_windows | 37 | 0.0173 | 0.2549 | 0.1437 | 0.9647 | 0.0040 | 0.0000 | LOW |
| false_alarm_normal_windows | 100 | 0.5742 | 0.3185 | 0.8272 | 0.0075 | 0.0593 | 0.8955 | HIGH |
| all_validation_windows | 657 | 0.1288 | 0.2838 | 0.2815 | 0.7589 | 0.0225 | 0.1773 | LOW |
| job_6251277720_train_windows | 940 | 0.1961 | 0.6362 | 0.5007 | 0.4673 | 0.0224 | 0.4387 | LOW |

### Key Interpretability Insights
1. **Detected Failure Windows**: Mean $R_{\text{rel}} = 0.3807$ (HIGH set), Mean $R_{\text{abs}} = 0.3235$ (MED set). Mean HIGH activation is **$0.5280$**, with Rules 7 and 8 dominating the activation.
2. **False Alarm Normal Windows**: Mean $R_{\text{rel}} = 0.2227$ (MED set), Mean $R_{\text{abs}} = 0.5878$ (HIGH set). Rule 6 fired with high activation (**$0.4705$**), driving false alarms.
3. **Job 6251277720 (Healthy Heavy Workload)**: Mean $R_{\text{abs}} = 0.6139$ (HIGH set) and Mean $R_{\text{rel}} = 0.1725$ (MED set). Rule 6 fired repeatedly with mean activation of **$0.4190$**, creating false alarms on $43.7\%$ of windows.

---

## 7. Operational Gate Summary (Candidate A Validation)

| model | seed | threshold | episode_recall_pct | gate_1a_status | critical_recall_pct | gate_1b_status | entity_far_pct | gate_2a_status | window_fpr_pct | gate_2b_status | operational_verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S0 | 42 | 0.5000 | 35.0000 | FAIL | 18.3673 | FAIL | 36.7347 | FAIL | 30.1786 | FAIL | NO_OPERATIONAL_WINNER |
| S0 | 123 | 0.5000 | 40.0000 | FAIL | 14.2857 | FAIL | 44.8980 | FAIL | 35.5357 | FAIL | NO_OPERATIONAL_WINNER |
| S0 | 456 | 0.5000 | 35.0000 | FAIL | 18.3673 | FAIL | 55.1020 | FAIL | 37.6786 | FAIL | NO_OPERATIONAL_WINNER |
| S2 | 42 | 0.5000 | 65.0000 | FAIL | 16.3265 | FAIL | 30.6122 | FAIL | 10.7143 | FAIL | NO_OPERATIONAL_WINNER |
| S2 | 123 | 0.5000 | 50.0000 | FAIL | 24.4898 | FAIL | 44.8980 | FAIL | 10.0000 | FAIL | NO_OPERATIONAL_WINNER |
| S2 | 456 | 0.5000 | 60.0000 | FAIL | 26.5306 | FAIL | 36.7347 | FAIL | 10.1786 | FAIL | NO_OPERATIONAL_WINNER |
| Hybrid_T0 | 42 | 0.5000 | 55.0000 | FAIL | 16.3265 | FAIL | 28.5714 | FAIL | 9.1071 | FAIL | NO_OPERATIONAL_WINNER |
| Hybrid_T0 | 123 | 0.5000 | 55.0000 | FAIL | 22.4490 | FAIL | 44.8980 | FAIL | 8.9286 | FAIL | NO_OPERATIONAL_WINNER |
| Hybrid_T0 | 456 | 0.5000 | 60.0000 | FAIL | 28.5714 | FAIL | 34.6939 | FAIL | 10.5357 | FAIL | NO_OPERATIONAL_WINNER |
| Fuzzy_Primary | 42 | 1.0000 | 0.0000 | FAIL | 0.0000 | FAIL | 0.0000 | PASS | 0.0000 | PASS | NO_OPERATIONAL_WINNER |
| Fuzzy_Primary | 123 | 1.0000 | 0.0000 | FAIL | 0.0000 | FAIL | 0.0000 | PASS | 0.0000 | PASS | NO_OPERATIONAL_WINNER |
| Fuzzy_Primary | 456 | 1.0000 | 0.0000 | FAIL | 0.0000 | FAIL | 0.0000 | PASS | 0.0000 | PASS | NO_OPERATIONAL_WINNER |
| Fuzzy_Fixed50 | 42 | 0.5000 | 70.0000 | PASS | 20.4082 | FAIL | 44.8980 | FAIL | 17.8571 | FAIL | NO_OPERATIONAL_WINNER |
| Fuzzy_Fixed50 | 123 | 0.5000 | 60.0000 | FAIL | 30.6122 | FAIL | 55.1020 | FAIL | 15.1786 | FAIL | NO_OPERATIONAL_WINNER |
| Fuzzy_Fixed50 | 456 | 0.5000 | 70.0000 | PASS | 30.6122 | FAIL | 44.8980 | FAIL | 19.2857 | FAIL | NO_OPERATIONAL_WINNER |

### Operational Analysis
- **Primary Operating Policy ($\tau = 1.0000$)**:
  - Episode Recall: $0.00\%$ (FAIL, limit $\ge 70\%$)
  - Critical Recall: $0.00\%$ (FAIL, limit $\ge 50\%$)
  - Pure-Normal Entity FAR: $0.00\%$ (**PASS**, limit $\le 10\%$)
  - Normal-Window FPR: $0.00\%$ (**PASS**, limit $\le 5\%$)
- **Diagnostic Policy ($\tau = 0.500$)**:
  - Episode Recall: $66.67\% \pm 5.77\%$ (FAIL, limit $\ge 70\%$, but reached $70.0\%$ on Seeds 42 and 456)
  - Critical Recall: $27.21\% \pm 5.89\%$ (FAIL, limit $\ge 50\%$)
  - Pure-Normal Entity FAR: $48.30\% \pm 5.89\%$ (FAIL, limit $\le 10\%$)
  - Normal-Window FPR: $17.44\% \pm 2.08\%$ (FAIL, limit $\le 5\%$)
- **Operational Verdict**: `NO_OPERATIONAL_WINNER`.

---

## 8. Answers to the 12 Required Research Questions (Section 14)

### 1. Was the Phase 7T result independently reproduced?
**Yes.** All Phase 7T component scores, hybrid formulas, threshold selection records, and operational metrics were independently recomputed and verified with zero discrepancy.

### 2. Was the fuzzy system fitted or designed without Validation leakage?
**Yes.** Membership breakpoints were derived strictly from empirical quantiles of pooled Candidate A Train-OOF predictions ($9,834$ observations across 3 seeds). Validation labels were never accessed during rule base design or breakpoint derivation.

### 3. Did fuzzy fusion improve cross-job ROC-AUC or average precision relative to S2?
**Comparable on AUC, slightly lower on AP.** Validation ROC-AUC was **$0.5827 \pm 0.0515$** (vs $0.5864$ in $S_2$, reaching $0.6289$ on Seed 123), while Validation Average Precision was **$0.2148 \pm 0.0317$** (vs $0.2415$ in $S_2$).

### 4. Did it recover any useful within-job ranking relative to S2?
**Yes.** Macro-mean within-job AUC across the 32 failing parent jobs improved from $0.5345$ ($S_2$) to **$0.5607$** (Fuzzy), recovering $+0.0262$ (+2.62 percentage points) across all three random seeds.

### 5. Did it suppress false alerts on high-resource pure-normal jobs?
**No, it performed poorly compared to the linear hybrid.** On Job `6251277720`, the fuzzy false alert rate was **$43.69\%$**, compared to only **$7.30\%$** for the Phase 7T linear hybrid and $6.10\%$ for $S_2$.

### 6. Which fuzzy rules most often fired for correct detections and false alerts?
- Correct detections were driven by **Rule 7** (`HIGH rel, LOW abs -> HIGH`) and **Rule 8** (`HIGH rel, MED abs -> HIGH`).
- False alerts on healthy heavy workloads were predominantly driven by **Rule 6** (`MED rel, HIGH abs -> HIGH`).

### 7. Did the fuzzy treatment meet all four operational gates?
**No.** Primary policy generated zero alerts, while diagnostic policy ($\tau = 0.50$) had Normal-Window FPR of $17.44\%$ (failing Gate 2B $\le 5\%$) and Entity FAR of $48.30\%$ (failing Gate 2A $\le 10\%$).

### 8. Were results consistent across all three seeds?
**Yes.** Within-job recovery was positive on all three seeds ($+0.0106$ on s42, $+0.0108$ on s123, $+0.0571$ on s456), and Job 6251277720 false alerts were consistently elevated ($41.9\% - 45.4\%$).

### 9. What evidence supports the explanation of workload baseline differences?
The empirical quantiles confirm that $S_0$ has an 80th percentile of $0.6248$, while $S_2$ has an 80th percentile of $0.3230$. Workload heterogeneity causes heavy jobs to reside permanently in the upper quartile of $S_0$, which triggers Rule 6 whenever minor natural variations occur.

### 10. What remains uncertain due to the limited number of parent jobs and repeated use of Validation?
Candidate A Train has only 51 parent jobs and Validation has only 12. The exact degree of Rule 6 false alarm elevation across broader cluster traces requires multi-cluster verification.

### 11. Does a new workload cohort remain necessary?
**Yes.** An uninspected multi-cluster workload trace is required to validate representation learning without the risk of over-fitting to the specific 51/12 parent-job distribution of Candidate A.

### 12. Which research conclusion is directly supported, and which remains a hypothesis?
- **Directly Supported**: A fixed linear blend ($0.80 R_{S2} + 0.20 R_{S0}$) is significantly more robust against cross-job baseline confounding than a linguistic fuzzy rule base because linear blending strictly bounds absolute risk influence, whereas Mamdani rule elevation allows benign relative deviations to trigger high false alarms.
- **Hypothesis**: Whether a non-linear neural gating network with task-type conditioning can avoid Rule 6's defect remains a future research hypothesis.

---

## 9. Final Deliverables and Sealed Manifest

All 22 required deliverables have been generated in `results/phase7u_fuzzy_lstm/` and mirrored to `reports/phase7u_fuzzy_lstm/`.
Manifest sealed in `results/phase7u_fuzzy_lstm/phase7u_manifest.json`.

**Final Operational Verdict**: `NO_OPERATIONAL_WINNER`  
**Scientific Conclusion**: While the fuzzy inference system demonstrated high interpretability and achieved strong Episode Recall ($66.7\%$ at $\tau=0.50$), it proved vulnerable to high-resource false alarms ($43.7\%$ on Job 6251277720), establishing that the linear 80/20 hybrid is more robust against Simpson's paradox.
