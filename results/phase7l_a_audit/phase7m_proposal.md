# Phase 7M Proposal: Multi-Resolution Temporal Pooling & Decoupled Critical-Margin Classification

**Project:** `C:\Users\hp\Desktop\Research project`  
**Phase:** Phase 7M (Design Specification Only — Execution Strictly Deferred)  
**Artifact:** `phase7m_proposal.md`  
**Date:** October 29, 2026  
**Author:** Senior Machine Learning Research Engineer & Independent Auditor  

---

## 1. Executive Summary & Diagnostic Motivation

The Phase 7L experiment and Phase 7L-A independent audit demonstrated that extending causal telemetry history in a standard flat LSTM from $L=5$ (25 minutes) to $L=12$ (60 minutes) yields a modest improvement in ranking separation for pre-failure episodes ($+8.4\%$ Episode Recall under Condition B), but incurs severe failure modes:
1. **Critical Recall Collapse:** Genuine Critical Recall fell to **$0.0\% \pm 0.0\%$** across all seeds.
2. **Probability Calibration Degradation:** Multiclass Brier score deteriorated from $0.3152$ to $0.3699$, and ECE increased from $0.1653$ to $0.2429$ (overconfidence on routine healthy fluctuations).
3. **Cascaded Margin Suppression:** Because Policy B uses a single aggregate failure threshold followed by pairwise argmax ($P_2 \ge P_1$), the extended flat context pushes mass into Warning ($P_1$), masking imminent Critical crashes ($P_2$).
4. **Operational Impasse:** Gate 1 (sensitivity) and Gate 2 (specificity) remain completely disjoint in threshold space, separated by a gap of $\Delta \theta \ge 0.265$.

Phase 7M is formulated to test whether **explicit multi-resolution temporal representation combined with decoupled class decision heads** can resolve the sensitivity–specificity conflict without introducing uncontrolled confounding factors.

---

## 2. Research Hypothesis

> **Primary Research Hypothesis ($H_1$):**  
> *"Decoupling the temporal representation into a high-frequency short-term sequence ($L_{\text{short}}=5$, covering $t-20\text{m}$ to $t$) and a multi-scale downsampled rolling context summary ($L_{\text{context}}$, covering $t-60\text{m}$ to $t-25\text{m}$), paired with an explicit critical-margin classification head, will significantly improve Genuine Critical Recall ($\ge 25\%$) and restore probability calibration (ECE $\le 0.18$) compared to the flat $L=12$ treatment, while maintaining Pure-Normal Entity FAR $\le 10\%$."*

---

## 3. Predeclared Primary Metric & Secondary Diagnostic Metrics

### 3.1 Predeclared Primary Metric
* **Operational Critical-Intercept Macro-F1 ($\text{F1}_{\text{crit-eval}}$):** Unweighted Macro-F1 evaluated at a **predeclared threshold** $\theta = 0.50$ under Policy B with an explicit Critical detection margin:
  $$\hat{y} = \begin{cases} 
  2 & \text{if } P(\text{fail}) \ge \theta_{\text{fail}} \text{ and } (P_2 - P_1) \ge \delta_{\text{crit}} \\
  1 & \text{if } P(\text{fail}) \ge \theta_{\text{fail}} \text{ and } (P_2 - P_1) < \delta_{\text{crit}} \\
  0 & \text{otherwise}
  \end{cases}$$
  where $\delta_{\text{crit}}$ is calibrated strictly on training quantile margins.

### 3.2 Secondary Operational & Diagnostic Metrics
* **Genuine Critical Recall:** Class 2 recall on true Critical windows ($\text{denominator}=49$).
* **Episode Recall:** Fraction of 20 pre-failure episodes alerted ($\text{denominator}=20$).
* **Pure-Normal Entity FAR:** Falsely alerted pure-normal entities ($\text{denominator}=49$).
* **Normal-Window FPR:** Falsely alerted normal windows ($\text{denominator}=560$).
* **Multiclass Brier Score:** Sum of squared probability error across all 3 classes.
* **Expected Calibration Error (ECE):** 10 uniform confidence bins on $[0, 1]$.

---

## 4. Controlled Comparison: Baseline vs. Treatment

To prevent confounding, Phase 7M holds the underlying 31 baseline features, loss weighting condition, optimizer, training budget, and random seeds strictly invariant.

| Component | Baseline Condition | Treatment Condition (Multi-Resolution) |
|:---|:---|:---|
| **Feature Set** | Authoritative 31 baseline telemetry features | Authoritative 31 baseline telemetry features |
| **Input Structure** | Flat sequence $L=12$ ($12 \times 31 = 372$ values) | Multi-resolution dual stream: <br>1. Short stream: $5 \times 31$ (last 25 min) <br>2. Long summary: $1 \times 31$ mean + $1 \times 31$ max pooled over $t-60\text{m}$ to $t-25\text{m}$ |
| **Model Architecture** | Standard 2-layer LSTM ($h=64$) | Dual-stream: 2-layer LSTM ($h=48$) on short stream concatenated with 1-layer projection ($h=16$) of pooled history; total hidden dim = 64 |
| **Loss Function** | Cross-Entropy with Condition B `[1.0, 2.0, 2.0]` | Cross-Entropy with Condition B `[1.0, 2.0, 2.0]` |
| **Optimizer** | Adam ($lr=0.001$, weight decay=$10^{-4}$) | Adam ($lr=0.001$, weight decay=$10^{-4}$) |
| **Batch Size & Epochs** | Batch size 64, 25 epochs | Batch size 64, 25 epochs |
| **Checkpoint Rule** | Min validation cross-entropy loss | Min validation cross-entropy loss |
| **Random Seeds** | 42, 123, 456 (matched pairing) | 42, 123, 456 (matched pairing) |

---

## 5. Causal Feature Definitions & Temporal Boundaries

Every feature in both streams must be strictly causal:
* **Short Stream:** Observations at timestamps $[t - 1200\text{s}, t - 900\text{s}, t - 600\text{s}, t - 300\text{s}, t]$ (5 steps). Exactly identical to $L=5$.
* **Long Context Summary:** Sliced strictly from the causal window $[t - 3300\text{s}, t - 1500\text{s}]$ (7 steps, covering minutes 25 to 60 prior to prediction).
  * Feature pooling functions: strictly causal `mean` and `max` along the time dimension of the standardized 31 telemetry features.
  * No future values, no lifecycle event fields, and no post-crash observations enter either stream.

---

## 6. Missing Data and Padding Policy

* **Raw Missingness:** Follows the established protocol: genuine raw NaNs in `cycles_per_instruction` and `memory_accesses_per_instruction` are imputed with training medians.
* **Truncation / Short Tasks:**
  * For tasks with history $< 60$ minutes, zero-padding with a binary mask indicator is applied to the unavailable historical steps prior to pooling.
  * No forward-filling from future observations.

---

## 7. Frozen Split & Leakage Isolation

1. **Frozen Candidate A Split:** Partition strictly into Candidate A Train ($N=3,278$) and Validation ($N=657$).
2. **Strict Quarantines:** Candidate A Test ($N=1,025$, non-virgin) and Phase 6 holdout remain **strictly quarantined**.
3. **Preprocessors:** All medians, means, and standard deviations fitted exclusively on Candidate A Train. Zero validation observations are used for scaling or imputation.

---

## 8. Threshold Policy & Unchanged Operational Gates

### 8.1 Four Governing Operational Gates (Non-Negotiable)
* **Gate 1a (Episode Sensitivity):** Episode Recall $\ge 70.0\%$
* **Gate 1b (Critical Sensitivity):** Genuine Critical Recall $\ge 50.0\%$
* **Gate 2a (Entity Specificity):** Pure-Normal Entity False Alarm Rate $\le 10.0\%$
* **Gate 2b (Window Specificity):** Normal-Window False Positive Rate $\le 5.0\%$

### 8.2 Operational Winner Criterion
* `NO OPERATIONAL WINNER` is preserved as the default. A configuration is declared an operational winner **only if** it satisfies all four gates simultaneously at a single threshold $\theta$.

---

## 9. Statistical Analysis Plan

1. **Paired Seed-Matched Comparison:** Recompute paired deltas $\Delta = \text{Treatment} - \text{Baseline}$ for each of the three seeds (42, 123, 456).
2. **Entity-Clustered Resampling:** In addition to point estimates, compute $95\%$ bootstrap confidence intervals by resampling at the **parent job level** ($N=30$ jobs in validation) rather than treating individual windows or entities as independent.
3. **Threshold Stability:** Evaluate sensitivity across a fixed grid $\theta \in [0.10, 0.90]$ with step $0.05$. Report both fixed-threshold performance and sweep ceilings.

---

## 10. Stopping & Rejection Criteria

Phase 7M will be halted and the hypothesis rejected if:
1. **Critical Recall Still Near Zero:** Treatment achieves Critical Recall $< 10.0\%$ across all 3 seeds.
2. **False Alarm Explosion:** Pure-Normal Entity FAR exceeds $20.0\%$ at all points where Episode Recall $\ge 20\%$.
3. **Calibration Further Deteriorates:** ECE increases beyond $0.25$ or Brier score worsens beyond $0.40$.
4. **Any Data Leakage Violation:** Any gradient update or normalization touches validation data.

---

## 11. Single-Cluster Scope Limitation

Even if Phase 7M meets all operational gates, findings remain strictly limited to the Google Borg 2011 production cluster trace. Cross-cluster generalization cannot be claimed without external cluster validation.
