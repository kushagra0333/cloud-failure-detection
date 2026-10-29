# Phase 7L-A: Decision Policy Definition and Threshold Audit

**Project:** `C:\Users\hp\Desktop\Research project`  
**Phase:** Phase 7L-A Audit  
**Artifact:** `policy_definition_audit.md`  
**Date:** October 29, 2026  
**Auditor:** Independent ML Research Engineer Subsystem  

---

## 1. Executive Summary

This audit establishes the exact mathematical and operational semantics of the decision policies evaluated in Phase 7L (`run_phase7l_experiment.py`). It resolves how thresholds are applied, how three-class predictions are formed, whether policies were mixed, and whether post-hoc threshold selection introduces optimism.

### Key Audit Findings:
1. **Primary Threshold Target:** In both Policy A and Policy B, the operational threshold $\theta \in [0.0, 1.0]$ is applied to the aggregate failure probability $P(\text{fail}) = P_1 + P_2 = P(\text{Warning}) + P(\text{Critical})$, not to any class-specific probability.
2. **Cascaded Partitioning (Policy B):** Windows with $P(\text{fail}) < \theta$ are assigned Class 0 (Normal). Windows with $P(\text{fail}) \ge \theta$ are partitioned into Warning (Class 1) or Critical (Class 2) via pairwise comparison: if $P_2 \ge P_1$, predict Class 2; otherwise predict Class 1. There is **no secondary threshold** on $P_2$.
3. **Macro-F1 is Threshold-Dependent:** The reported "Peak Macro-F1" is computed from the threshold-dependent cascaded predictions $\hat{y}_{\text{Policy B}}(\theta)$, **not** from ordinary three-class argmax. Under ordinary three-class argmax ($\hat{y} = \text{argmax}(P_0, P_1, P_2)$), models predict Class 0 (Normal) for almost 100% of windows, yielding a flat Macro-F1 of only $0.3029$ to $0.3068$.
4. **Post-Hoc Optimization:** In the reported tables, each seed selected a distinct threshold $\theta_{\text{peak}} \in [0.280, 0.690]$ that retrospectively maximized Macro-F1 on the validation set. That same maximized Macro-F1 is presented as evidence. This represents an optimistic in-sample ceiling rather than a fixed deployable rule.
5. **No Incompatible Policy Mixing:** Policy B was used consistently across Macro-F1, Episode Recall, Critical Recall, Entity FAR, and Window FPR within each reported comparison.

---

## 2. Answers to Governing Audit Questions

### Question 1: Is the threshold applied to $P(\text{Warning}) + P(\text{Critical})$ or to a class-specific probability?
* **Finding:** The threshold $\theta$ is applied strictly to the sum $P(\text{fail}) = P_1 + P_2$.
* **Code Reference:** [`run_phase7l_experiment.py:L286`](file:///C:/Users/hp/Desktop/Research%20project/scripts/run_phase7l_experiment.py#L286):
  ```python
  p_fail = p1 + p2
  # Policy A (L312):
  alert_a = (p_fail >= th).astype(int)
  # Policy B (L343):
  mask_alert = (p_fail >= th)
  ```
* **Evaluation Implication:** A window triggers a failure alert based on cumulative non-normal evidence. Individual probabilities $P_1$ or $P_2$ do not need to exceed $\theta$ independently.

---

### Question 2: What happens after a binary failure alert is triggered under Policy B?
* **Finding:** Once $P(\text{fail}) \ge \theta$, the window is deemed anomalous. Policy B branches into a 2-way tie-breaker between Class 1 and Class 2. If $P(\text{fail}) < \theta$, the window is definitively assigned $\hat{y} = 0$ (Normal), even if $P_1 > P_0$ or $P_2 > P_0$ (which can occur when $\theta > 0.5$).

---

### Question 3: Are Warning and Critical assigned using probability argmax, a second threshold, or another rule?
* **Finding:** They are assigned via conditional pairwise argmax:
  $$\hat{y} = \begin{cases} 0 & \text{if } P_1 + P_2 < \theta \\ 2 & \text{if } P_1 + P_2 \ge \theta \text{ and } P_2 \ge P_1 \\ 1 & \text{if } P_1 + P_2 \ge \theta \text{ and } P_1 > P_2 \end{cases}$$
* **Code Reference:** [`run_phase7l_experiment.py:L344-L347`](file:///C:/Users/hp/Desktop/Research%20project/scripts/run_phase7l_experiment.py#L344-L347):
  ```python
  mask_crit = mask_alert & (p2 >= p1)
  mask_warn = mask_alert & (p1 > p2)
  pred_b[mask_crit] = 2
  pred_b[mask_warn] = 1
  ```
* **Audit Assessment:** No secondary threshold $\theta_{\text{crit}}$ exists. If $P_1$ marginally exceeds $P_2$ (e.g. $P_1 = 0.28, P_2 = 0.27$), the window is assigned Class 1, completely suppressing Class 2 predictions.

---

### Question 4: Is peak Macro-F1 calculated from threshold-dependent predictions or from ordinary three-class argmax predictions?
* **Finding:** Peak Macro-F1 is calculated strictly from threshold-dependent predictions $\hat{y}_{\text{Policy B}}(\theta)$.
* **Audit Verification:**
  * At standard argmax $\hat{y} = \text{argmax}(P_0, P_1, P_2)$, 10 of the 12 models predict Class 0 for every single validation sample ($N=657$). Class 1 F1 is 0.0, Class 2 F1 is 0.0, and Class 0 F1 is $\frac{2 \times 560}{560 + 657} = 0.9203$, giving Macro-F1 = $\frac{0.9203}{3} = \mathbf{0.3068}$.
  * Only through threshold sweeping on Policy B does Macro-F1 rise above $0.3068$ (to peak values between $0.3155$ and $0.3763$).
  * Therefore, the metric called "Peak Macro-F1" is an empirical post-hoc optimum over the threshold sweep.

---

### Question 5: Is the same policy used when calculating Macro-F1, Episode Recall, Critical Recall, FAR, and FPR in each reported table?
* **Finding:** YES. The primary reported tables (Table 3.1 and Table 3.2 in `phase7l_report.md`) evaluate all metrics simultaneously at the specific threshold $\theta_{\text{peak}}$ that maximizes Macro-F1 under Policy B.
* **Consistency Check:** There is no mixing of Policy A and Policy B within those primary tables. Policy A metrics were computed in parallel in the sweep CSV but were not substituted into the Policy B report tables.

---

### Question 6: Does the aggregate table's metrics correspond to the threshold maximizing Macro-F1 for each seed or to a fixed, predeclared threshold?
* **Finding:** The aggregate table corresponds to the threshold maximizing Macro-F1 for **each seed independently**.
* **Observed Operating Thresholds ($\theta_{\text{peak}}$):**
  * `L5_cond_B_seed42`: $\theta = 0.590$
  * `L5_cond_B_seed123`: $\theta = 0.410$
  * `L5_cond_B_seed456`: $\theta = 0.425$
  * `L12_cond_B_seed42`: $\theta = 0.510$
  * `L12_cond_B_seed123`: $\theta = 0.605$
  * `L12_cond_B_seed456`: $\theta = 0.540$
* **Audit Assessment:** Because the optimal threshold varies substantially across seeds (from $0.410$ to $0.605$), the cross-seed aggregate reflects an average of post-hoc tuned operating points, not a single deployable operational model.

---

### Question 7: Does the gate summary evaluate the same saved predictions and policies as the threshold sweep?
* **Finding:** YES. `gate_summary.csv` extracts exact operating points from `threshold_sweep_metrics.csv`, which in turn operates on the identical probability columns in `validation_predictions.csv`. Numerical equality was verified bit-for-bit.

---

### Question 8: Was any threshold selected after examining the same metric that is subsequently presented as evidence of model superiority?
* **Finding:** YES. The threshold $\theta_{\text{peak}}$ was explicitly selected by maximizing `polB_macro_f1` over the validation set. That same maximum value is then reported as "Peak Macro-F1" to support the claim that $L=12$ achieves $+0.0253$ higher performance than $L=5$.
* **Methodological Caveat:** This validation threshold selection is structurally symmetric (both $L=5$ and $L=12$ were granted the identical post-hoc sweep), making the comparison fair between conditions. However, the resulting metric is an upper bound and cannot be guaranteed under a predeclared threshold on unseen data.

---

## 3. Decision Policy Comparison Matrix

| Property | Standard 3-Class Argmax | Policy A (Binary Alert) | Policy B (Cascaded 3-Class Decision) |
|:---|:---|:---|:---|
| **Input Probabilities** | $P_0, P_1, P_2 \in [0, 1]$ | $P_{\text{fail}} = P_1 + P_2$ | $P_{\text{fail}} = P_1 + P_2$ and $P_1, P_2$ |
| **Operational Threshold** | None (implicit 0.33) | Single threshold $\theta \in [0, 1]$ | Single threshold $\theta \in [0, 1]$ |
| **Output Classes** | $\{0, 1, 2\}$ | $\{0, 1\}$ (Binary Alert) | $\{0, 1, 2\}$ (Three-Class Decision) |
| **Normal Decision Rule** | $P_0 \ge P_1 \text{ and } P_0 \ge P_2$ | $P_{\text{fail}} < \theta$ | $P_{\text{fail}} < \theta$ |
| **Warning Decision Rule** | $P_1 > P_0 \text{ and } P_1 \ge P_2$ | Grouped with Failure | $P_{\text{fail}} \ge \theta \text{ and } P_1 > P_2$ |
| **Critical Decision Rule** | $P_2 > P_0 \text{ and } P_2 > P_1$ | Grouped with Failure | $P_{\text{fail}} \ge \theta \text{ and } P_2 \ge P_1$ |
| **Validation Macro-F1 Range** | $0.3029 - 0.3068$ (flat) | N/A (Binary F1 only) | $0.3068 - 0.3763$ (tunable) |
| **Operational Gates Evaluated** | Gates not satisfied | Partial (Gate 1a, 2a, 2b only) | Full (Gate 1a, 1b, 2a, 2b) |

---

## 4. Policy B Decision Logic Flowchart

```
                          [ Input Window x_t ]
                                   │
                                   ▼
                       [ Model Output: P_0, P_1, P_2 ]
                                   │
                                   ▼
                      [ Compute P_fail = P_1 + P_2 ]
                                   │
                                   ▼
                        /─────────────────────\
                       <  P_fail >= theta ?    >
                        \─────────────────────/
                               /         \
                       No     /           \  Yes
                             ▼             ▼
                     [ Class 0: Normal ]  /──────────────────\
                                         <   P_2 >= P_1 ?     >
                                          \──────────────────/
                                                 /      \
                                         Yes    /        \  No
                                               ▼          ▼
                                     [ Class 2: Crit ]  [ Class 1: Warn ]
```

---

## 5. Audit Conclusion on Policy Semantics

1. Policy B is an operationally sensible cascaded structure for severe class imbalance, allowing false alarms on healthy entities to be controlled via $\theta$ while retaining multiclass resolution.
2. However, the pairwise rule ($P_2 \ge P_1$) without a dedicated critical threshold creates an acute vulnerability: if longer sequences shift mass from Class 2 to Class 1, all alerted windows are labeled Warning, driving Genuine Critical Recall to **zero**.
3. Future architectures (Phase 7M) must decouple the Critical detection boundary from the Warning detection boundary.
