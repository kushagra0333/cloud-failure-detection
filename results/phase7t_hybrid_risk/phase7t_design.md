# Phase 7T Design: Relative-Primary Hybrid Risk Scoring and Cross-Job Robustness

## 1. Scientific Motivation & Hypotheses
Phase 7S established that baseline-relative-only normalization ($S_2$) eliminates the catastrophic cross-job score inversion seen in absolute-only models ($S_0$), raising Train-OOF ROC-AUC from $0.3066$ to $0.5080$ and Validation ROC-AUC from $0.4244$ to $0.5864$. Furthermore, $S_2$ reduced false alarms on healthy high-resource Job `6251277720` from $90.74\%$ down to $6.10\%$.
However, $S_2$ also exhibited a trade-off: within-job failure discrimination decreased from $0.6293$ ($S_0$) to $0.5345$ ($S_2$), because purely relative normalization strips away absolute saturation cues.

**Phase 7T Central Research Question:**
Can a fixed 80/20 linear blend of relative-only ($S_2$) and absolute-only ($S_0$) risk scores recover valuable absolute saturation cues while preserving the cross-job robustness and false-alarm suppression of $S_2$?

## 2. Prespecified Model Specification
- **Hybrid Risk Score Formula**:
  $$R_{\text{hybrid}} = 0.80 \cdot R_{S2} + 0.20 \cdot R_{S0}$$
  where $R_{S2} = P_{1,S2} + P_{2,S2}$ and $R_{S0} = P_{1,S0} + P_{2,S0}$.
- **Critical-versus-Warning Separation**:
  Retain the audited $S_2$ logit margin:
  $$M_{\text{critical}} = z_{2,S2} - z_{1,S2}$$
- **Decision Logic**:
  1. If $R_{\text{hybrid}} < \tau$, predict Normal (Class 0).
  2. Else if $M_{\text{critical}} \ge 0.50$, predict Critical (Class 2).
  3. Else predict Warning (Class 1).

## 3. Leak-Free Threshold Selection (Train-OOF Only)
- Evaluate 201 candidate thresholds: $\tau \in \{0.000, 0.005, \dots, 1.000\}$.
- A threshold is **eligible** if and only if BOTH specificity gates pass on Train-OOF predictions for **every random seed**:
  - $\text{Pure-Normal Entity FAR} \le 10.0\%$
  - $\text{Normal-Window FPR} \le 5.0\%$
- Tie-breaking:
  1. Maximize mean Episode Recall across seeds.
  2. Maximize mean Genuine Critical Recall across seeds.
  3. Choose largest $\tau$ as conservative tie-break.
- The resulting operating policy is frozen in `frozen_operating_policy.json` BEFORE inspecting Validation outcomes.
