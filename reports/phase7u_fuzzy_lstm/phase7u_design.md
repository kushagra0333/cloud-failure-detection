# Phase 7U Design: Interpretable Fuzzy Inference for LSTM-Based Failure Prediction

## 1. Scientific Motivation & Research Question
Phase 7T evaluated a fixed linear blend:
$$R_{\text{hybrid}} = 0.80 \cdot R_{S2} + 0.20 \cdot R_{S0}$$
While the linear hybrid recovered $+0.0334$ within-job AUC over $S_2$ and preserved false-alarm suppression on Job `6251277720` ($7.30\%$ vs $90.74\%$ in $S_0$), it imposed a uniform linear weighting across all operating regimes.

**Phase 7U Research Question:**
Can a relative-primary fuzzy inference system use absolute telemetry risk as conditional supporting evidence, without allowing high absolute resource usage alone to trigger too many false alarms?

## 2. Fuzzy Architecture Specification
- **Type**: Type-1 Mamdani Fuzzy Inference System
- **Inputs**:
  1. $R_{\text{rel}}$: Failure probability from relative-only model ($S_2$)
  2. $R_{\text{abs}}$: Failure probability from absolute-only model ($S_0$)
- **Input Breakpoints** (Derived strictly from Candidate A Train-OOF percentiles):
  - $R_{\text{rel}}$: 20th = 0.0923, 50th = 0.1821, 80th = 0.3230
  - $R_{\text{abs}}$: 20th = 0.0968, 50th = 0.2667, 80th = 0.6248
- **Fuzzy Terms**: LOW, MEDIUM, HIGH (Ruspini partition with overlapping linear ramps)
- **Output Sets**: LOW ($[0.0, 0.4]$), MEDIUM ($[0.2, 0.8]$), HIGH ($[0.6, 1.0]$)
- **T-norm**: Minimum (antecedent conjunction)
- **S-norm**: Maximum (rule output aggregation)
- **Defuzzification**: Centroid over 1001-point output discretization

## 3. Predeclared Relative-Primary Rule Base
| Relative Risk ($R_{\text{rel}}$) | Absolute Risk ($R_{\text{abs}}$) | Output Term | Intended Semantic Interpretation |
| :--- | :--- | :--- | :--- |
| **LOW** | LOW | **LOW** | Both streams indicate low failure risk |
| **LOW** | MEDIUM | **LOW** | Moderate absolute load is ignored when relative deviation is absent |
| **LOW** | HIGH | **LOW** | High absolute telemetry represents healthy heavy workload, NOT failure |
| **MEDIUM** | LOW | **MEDIUM** | Moderate relative deviation produces moderate risk |
| **MEDIUM** | MEDIUM | **MEDIUM** | Moderate relative deviation with moderate load produces moderate risk |
| **MEDIUM** | HIGH | **HIGH** | Moderate relative deviation reinforced by high resource load triggers HIGH alert |
| **HIGH** | LOW | **HIGH** | Strong relative deviation indicates failure regardless of low baseline |
| **HIGH** | MEDIUM | **HIGH** | Strong relative deviation confirmed by moderate load |
| **HIGH** | HIGH | **HIGH** | Overwhelming evidence of impending failure across both streams |

## 4. Leakage-Free Operating Policy Selection
Operating threshold $\tau_{\text{fuzzy}}$ is selected from Train-OOF fuzzy predictions across a 201-point grid under frozen specificity constraints (FAR $\le 10\%$ and FPR $\le 5\%$ for all 3 seeds).
Critical classification is governed by $M_{\text{crit}} = z_{2,S2} - z_{1,S2} \ge 0.50$.
