# Phase 7M: Multi-Resolution Temporal Redesign and Critical-Failure Detection Specification

**Project:** `C:\Users\hp\Desktop\Research project`  
**Phase:** Phase 7M  
**Date:** October 29, 2026  
**Auditor / Investigator:** Antigravity Senior ML Research Engineering Subsystem  

---

## 1. Preflight System Audit & Context

### 1.1 Verified Governing Artifacts
* **Project Root:** `C:\Users\hp\Desktop\Research project`
* **Frozen Candidate A Split:** [`results/phase7a_4/candidate_a_job_grouped_split.csv`](file:///C:/Users/hp/Desktop/Research%20project/results/phase7a_4/candidate_a_job_grouped_split.csv) (`SHA-256: f84bc4fe54f8e0583e2ee52f49b2fbe8e6fcafb4f020c656985a1acc31bb4946`).
* **Evaluation Protocol:** [`results/phase7b_1/phase7b_evaluation_protocol_v2.json`](file:///C:/Users/hp/Desktop/Research%20project/results/phase7b_1/phase7b_evaluation_protocol_v2.json).
* **Frozen Cohort Structure:**
  * Candidate A Train: $N=3,278$ windows.
  * Candidate A Validation: $N=657$ windows.
  * Authoritative Validation Labels: 560 Normal, 48 Warning, 49 Critical (Total: 657).
  * Operational Groups: 20 pre-failure episodes across 20 failing entities; 49 pure-normal entities; 4 pure-normal parent jobs.
* **Data Access Restrictions:**
  * Candidate A Test ($N=1,025$) was inspected in Phase 7D and is disqualified as a virgin benchmark. Zero access in Phase 7M.
  * Phase 6 confirmatory holdout remains strictly quarantined. Zero access.

### 1.2 Diagnostic Baseline Findings from Phase 7L & 7L-A
Phase 7L tested flat sequence length extension from $L=5$ (25m) to $L=12$ (60m). Phase 7L-A audit verified:
1. **Macro-F1 Gain:** $L=12$ yielded modest peak Macro-F1 improvement under Condition B ($0.3306 \to 0.3559$, $+7.65\%$) and raised Episode Recall from $8.3\%$ to $16.7\%$.
2. **Critical Recall Collapse:** Genuine Critical Recall collapsed to **$0.0\% \pm 0.0\%$** across all seeds.
3. **Compound Failure Mechanism:**
   * $84.4\%$ of true Critical windows fail to cross the elevated peak-F1 threshold ($\theta \approx 0.51 - 0.60$) and are classified as Normal.
   * The remaining $15.6\%$ of Critical windows produce $P(\text{Warning}) > P(\text{Critical})$, causing Policy B's pairwise cascade to assign them to Warning (Class 1).
4. **Calibration Degradation:** Multiclass Brier score worsened by $+0.0547$ ($0.3152 \to 0.3699$), and ECE worsened by $+0.0776$ ($0.1653 \to 0.2429$), increasing Pure-Normal Entity FAR from $8.2\%$ to $12.9\%$.
5. **Operational Impasse:** Gate 1 (sensitivity) and Gate 2 (specificity) are separated by a disjoint threshold gap $\Delta \theta \ge 0.265$.

---

## 2. Research Question & Formal Hypothesis

### 2.1 Primary Research Question
*Can a multi-resolution temporal model preserve short-term signals associated with imminent failures while incorporating longer-term trends, improving genuine Critical Recall and episode interception without violating the existing false-alarm limits?*

### 2.2 Formal Hypothesis ($H_1$)
A model that separately represents recent fine-grained telemetry ($L=5$, last 25m) and longer-term causal trends ($t-55\text{m}$ to $t-25\text{m}$), paired with an explicit classification head, will improve Genuine Critical Recall and episode interception compared with a flat sequence LSTM, by preserving imminent failure dynamics while providing baseline context without dilution.

---

## 3. Multi-Resolution Architecture Design

### 3.1 Dual-Stream Inputs
The full 60-minute causal sequence ($L=12$ steps) is partitioned into two disjoint, non-overlapping streams:
1. **Stream A (Recent High-Resolution Telemetry):**
   * Temporal span: $[t - 20\text{m}, t]$ (steps 7, 8, 9, 10, 11 of the 12-step sequence).
   * Shape: $(N, 5, 31)$ where $31$ is the authoritative baseline telemetry feature set.
   * Encoding: 2-layer LSTM ($input\_dim=31, hidden\_dim=64, num\_layers=2, dropout=0.2$).
   * Output representation: Final hidden state $z_{\text{recent}} \in \mathbb{R}^{64}$.
2. **Stream B (Longer-Term Historical Trends):**
   * Temporal span: $[t - 55\text{m}, t - 25\text{m}]$ (steps 0, 1, 2, 3, 4, 5, 6; 7 five-minute steps).
   * Summary features ($93$ total):
     * `hist_mean_all` ($31$ dims): Mean across all 7 historical steps.
     * `hist_std_all` ($31$ dims): Temporal standard deviation across all 7 historical steps.
     * `hist_trend_delta` ($31$ dims): Late history mean (steps 4..6, $t-35\text{m}$ to $t-25\text{m}$) minus Early history mean (steps 0..3, $t-55\text{m}$ to $t-40\text{m}$).
   * Preprocessing: Standardized using a Train-only fitted StandardScaler.
   * Encoding: Multi-layer perceptron: `nn.Linear(93, 32)` $\to$ `nn.ReLU()` $\to$ `nn.Dropout(0.2)` $\to$ `nn.Linear(32, 16)`.
   * Output representation: $z_{\text{history}} \in \mathbb{R}^{16}$.

### 3.2 Stream Fusion and Classification Head
* **Fusion:** $z_{\text{fused}} = [z_{\text{recent}}, z_{\text{history}}] \in \mathbb{R}^{64 + 16 = 80}$.
* **Classification Head:**
  $$\text{classifier} = \text{Dropout}(0.2) \to \text{Linear}(80, 64) \to \text{ReLU}() \to \text{Dropout}(0.2) \to \text{Linear}(64, 3)$$
* **Total Parameter Counts:**
  * M0 (Phase 7L baseline L=5): $61,955$ parameters.
  * M1 (Recent-only control, $z_{\text{history}}$ omitted): $62,467$ parameters.
  * M2 (Multi-resolution treatment, Stream A + Stream B): $67,027$ parameters ($+7.3\%$ vs M1).

---

## 4. Controlled Tripartite Ablation Plan

To strictly isolate experimental factors, Phase 7M evaluates three matched model conditions:

1. **M0 — Baseline:** Existing $L=5$ LSTM from Phase 7L.
2. **M1 — Recent-Only Control:** Recent 5-step stream ($5 \times 31$) using the updated modular architecture and classification head, trained under the exact Phase 7M budget.
3. **M2 — Multi-Resolution Treatment:** Recent 5-step stream ($5 \times 31$) plus historical summary stream ($93$ features).

### Factor Isolation Constraints:
* **Features:** Same 31 baseline telemetry features for all models.
* **Loss Function:** Cross-entropy with Condition B `[1.0, 2.0, 2.0]` (primary) and Condition A `[1.0, 1.0, 1.0]` (reference).
* **Optimization:** Adam ($lr=0.001, weight\_decay=0.0001$), batch size 64, 25 epochs.
* **Model Selection:** Minimum validation loss.
* **Seeds:** Fixed set $\{42, 123, 456\}$.

---

## 5. Decision Policies and Operational Gates

### 5.1 Predeclared Primary Policy & Threshold
* **Primary Operating Point:** Policy B evaluated at fixed predeclared threshold $\theta = 0.50$.
* **Unthresholded Baseline:** Standard 3-class argmax ($\hat{y} = \text{argmax}(P_0, P_1, P_2)$).
* **Threshold Sweep:** 201 thresholds ($\theta \in [0.0, 1.0]$, step 0.005) under Policy B and Policy A.
* **Exploratory Margin Analysis:** Critical margin rule ($P_2 - P_1 \ge \delta_{\text{crit}}$) evaluated across candidate grid $\delta \in \{-0.10, -0.05, -0.02, 0.00, +0.02, +0.05, +0.10\}$.

### 5.2 Four Unchanged Operational Gates
1. **Gate 1a (Episode Sensitivity):** Episode Recall $\ge 70.0\%$
2. **Gate 1b (Critical Sensitivity):** Genuine Critical Recall $\ge 50.0\%$
3. **Gate 2a (Entity Specificity):** Pure-Normal Entity False Alarm Rate $\le 10.0\%$
4. **Gate 2b (Window Specificity):** Normal-Window False Positive Rate $\le 5.0\%$

Verdict criterion: `NO OPERATIONAL WINNER` is preserved unless all four gates pass simultaneously at a single threshold.
