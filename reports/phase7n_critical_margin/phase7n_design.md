# Phase 7N: Critical-Class Separation and Asymmetric Loss Remediation — Experimental Design

**Project Root:** `C:\Users\hp\Desktop\Research project`  
**Phase:** Phase 7N  
**Author / Lead:** Antigravity Senior ML Research Engineering Subsystem  
**Date:** October 29, 2026  
**Status:** Frozen Prior to Training Execution  

---

## 1. Research Motivation & Diagnostic Context

In Phase 7M, the multi-resolution dual-stream architecture successfully resolved the temporal dilution and calibration degradation associated with flat long sequences:
- Multiclass Brier score improved from $0.3152$ to $0.2872 \pm 0.0118$ ($-8.9\%$).
- Expected Calibration Error (ECE) dropped from $0.1653$ to $0.1244 \pm 0.0305$ ($-24.7\%$).
- Under predeclared $\theta = 0.50$, Pure-Normal Entity FAR fell from $10.2\%$ to $3.4\%$ and Window FPR dropped to $0.5\%$.
- Pre-failure episode interception improved at peak F1 from $8.3\%$ to $18.3\%$.

However, Phase 7M failed to produce genuine Critical Recall ($0.0\% \pm 0.0\%$ at $\theta = 0.50$ and across the entire threshold sweep). Forensics revealed:
$$\operatorname{mean}(P_2 - P_1) = -0.0110 \quad \text{on true Critical validation windows}$$
Under symmetric loss weighting (`[1.0, 2.0, 2.0]`), the network minimizes expected cross-entropy by assigning higher probability to the earlier, more frequent Warning state (Class 1) than to the imminent Critical state (Class 2). Under Policy B's pairwise cascade ($\operatorname{argmax}(P_1, P_2)$), alerted Critical windows are $100\%$ classified as Warning.

Phase 7N tests whether an **explicitly asymmetric learning objective** can enforce Warning/Critical logit separation on true Critical examples, unlocking genuine Critical Recall while retaining Phase 7M's false-alarm suppression and calibration benefits.

---

## 2. Research Questions & Hypotheses

- **RQ1 — Critical Discrimination:** Does asymmetric training improve genuine Critical Recall compared with the unchanged Phase 7M multi-resolution baseline?
- **RQ2 — Warning/Critical Separation:** Does it increase $P_2 - P_1$ and $\text{logit}_2 - \text{logit}_1$ on true Critical windows without causing excessive false Critical predictions on Normal or Warning windows?
- **RQ3 — False-Alarm Retention:** Does the model retain Phase 7M's improvement in Pure-Normal Entity FAR ($\le 10\%$) and Normal-Window FPR ($\le 5\%$)?
- **RQ4 — Calibration:** Does the modified loss damage probability calibration (Brier score, NLL, ECE)?
- **RQ5 — Operational Feasibility:** Can any predeclared model and decision policy satisfy all four operational gates simultaneously?

**Primary Hypothesis:** An objective penalizing negative Critical-versus-Warning margins on true Critical training samples can shift probability mass toward Critical, but may risk increased false alarms on Normal or Warning windows if not properly regularized.

---

## 3. Frozen Multi-Resolution Architecture

The architectural backbone is frozen from Phase 7M:
1. **Stream A (Recent High-Resolution Telemetry):**
   - Span: Most recent 5 5-minute steps ($t-20\text{m}$ to $t$).
   - Features: 31 standardized telemetry signals.
   - Encoder: 2-layer compact LSTM, hidden dimension = 64, dropout = 0.2.
2. **Stream B (Historical Causal Summaries):**
   - Span: Earlier 7 5-minute steps ($t-55\text{m}$ to $t-25\text{m}$).
   - Features: 93 causal statistics (31 bin means, 31 bin standard deviations, 31 slope trends: late minus early bin).
   - Preprocessing: StandardScaler fitted exclusively on Candidate A Train ($N=3,278$). Zero validation access.
   - Encoder: MLP ($93 \to 32 \to 16$), ReLU, dropout = 0.2.
3. **Fused Classification Head:**
   - Concatenation: $z = [z_{\text{recent}}, z_{\text{hist}}] \in \mathbb{R}^{80}$.
   - Projection: $\text{Linear}(80, 64) \to \text{ReLU} \to \text{Dropout}(0.2) \to \text{Linear}(64, 3)$.
   - Parameter Count: Exactly 67,027 trainable parameters.

---

## 4. Experimental Conditions

Four controlled conditions across seeds $42, 123, 456$ (12 runs total):

### Condition N0 — Reproduced Phase 7M Baseline
- Loss: Standard Weighted Cross-Entropy
  $$\mathcal{L}_{\text{N0}} = \frac{1}{\sum_{i=1}^B w_{y_i}} \sum_{i=1}^B w_{y_i} \left(-\log \frac{\exp(z_{i, y_i})}{\sum_{c=0}^2 \exp(z_{i, c})}\right)$$
- Class Weights: $w = [1.0, 2.0, 2.0]$ (moderate weighting from Phase 7E/7M).
- Baseline reference to verify exact reproduction.

### Condition N1 — Critical-Weight Sensitivity
- Loss: Standard Weighted Cross-Entropy with doubled Critical weight.
- Class Weights: $w = [1.0, 2.0, 4.0]$.
- Tests whether scalar weight rebalancing alone shifts the Warning/Critical decision boundary.

### Condition N2 — Asymmetric Critical Margin Loss
- Loss: Base Cross-Entropy ($w = [1.0, 2.0, 2.0]$) plus softplus margin penalty on true Critical training samples:
  $$\mathcal{L}_{\text{margin}} = \frac{1}{|\{i: y_i = 2\}|} \sum_{i: y_i = 2} \operatorname{softplus}(m - (z_{i, 2} - z_{i, 1}))$$
  $$\mathcal{L}_{\text{N2}} = \mathcal{L}_{\text{ce}} + \lambda_{\text{crit}} \times \mathcal{L}_{\text{margin}}$$
- Predeclared Parameters: Margin $m = 0.50$ logit units, $\lambda_{\text{crit}} = 0.10$.
- Computed strictly on training samples ($y_i == 2$). Zero validation label access during gradient updates.

### Condition N3 — Focal Loss Comparator
- Loss: Multiclass Focal Loss with class weights $\alpha = [1.0, 2.0, 2.0]$ and focusing parameter $\gamma = 2.0$:
  $$\mathcal{L}_{\text{N3}} = \frac{1}{\sum_i \alpha_{y_i}} \sum_{i=1}^B -\alpha_{y_i} (1 - p_{i, y_i})^\gamma \log(p_{i, y_i})$$
- Tests whether dynamically down-weighting easy examples addresses Critical separation without an explicit logit margin.

---

## 5. Predeclared Primary Decision Policy

To prevent post-hoc threshold selection bias, the primary operating policy is predeclared prior to training:
1. **Failure Interception:** $P_{\text{fail}} = P_1 + P_2$.
   - If $P_{\text{fail}} < 0.50$, predict **Class 0 (Normal)**.
2. **Critical vs Warning Separation:**
   - In alerted windows ($P_{\text{fail}} \ge 0.50$):
     - Predict **Class 2 (Critical)** if $\text{logit}_2 - \text{logit}_1 \ge 0.50$.
     - Otherwise, predict **Class 1 (Warning)**.

### Secondary Diagnostic Policies
- **Argmax Policy:** $\operatorname{argmax}(P_0, P_1, P_2)$ (evaluates intrinsic classification without thresholding).
- **Exploratory Threshold Sweep:** 201 thresholds ($\theta \in [0.0, 1.0]$, step 0.005), clearly labeled non-deployable and exploratory.

---

## 6. Governing Operational Gates

All four gates must pass simultaneously under the predeclared primary policy:

| Gate | Metric | Target | Denominator |
|:---|:---|:---:|:---|
| **Gate 1a** | Episode Recall | $\ge 70\%$ | 20 validation failure episodes |
| **Gate 1b** | Genuine Critical Recall | $\ge 50\%$ | 49 true Critical validation windows |
| **Gate 2a** | Pure-Normal Entity FAR | $\le 10\%$ | 49 pure-normal validation entities |
| **Gate 2b** | Normal-Window FPR | $\le 5\%$ | 560 Normal-labelled validation windows |

Verdict rule:
- `OPERATIONAL_WINNER` only if all 4 gates pass simultaneously.
- Otherwise, `NO_OPERATIONAL_WINNER`.
- `AUDIT_BLOCKED` if data governance, leakage, or reproducibility fails.

---

## 7. Training Budget & Controls

- Hardware: CPU deterministic execution.
- Seeds: 42, 123, 456.
- Epochs: 25 per run.
- Optimizer: Adam, $\text{lr} = 0.001$, $\text{weight\_decay} = 0.0001$.
- Batch size: 64.
- Checkpoint selection: Minimum validation loss checkpoint saved per run.
