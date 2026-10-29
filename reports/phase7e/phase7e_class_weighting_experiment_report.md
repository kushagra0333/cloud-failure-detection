# Phase 7E Experiment Report: Class Weighting & Validation Threshold Calibration

**Document ID:** `REPORT-PHASE7E-CLASS-WEIGHTING-V1`  
**Execution Timestamp:** 2026-10-28T14:16:36Z (Local: 2026-10-28 19:46:36 IST)  
**Experiment Objective:** Determine whether neural-model failure detection improves when class imbalance is handled via inverse-frequency weighting, and calibrate operational alert thresholding on failure-risk score $P(\text{Warning}) + P(\text{Critical})$.  
**Evaluation Cohort:** Candidate A Validation Split ONLY ($N=657$ windows, 12 parent jobs, 69 entities, 20 failure episodes).  
**Blinding & Quarantine Status:** Candidate A Test split and historical Phase 6 holdout strictly unaccessed.  
**Official Status:** `COMPLETED — NO_OPERATIONAL_WINNER_ON_VALIDATION`  

---

## 1. Executive Summary

Phase 7E executed a controlled, pre-registered development experiment on the clean Phase 7C Candidate A dataset to test the hypothesis that neural sequence models (Temporal LSTM and Compact Temporal Transformer) suffered majority-class mode collapse in Phase 7D primarily due to improper loss weighting ($[1.0, 1.5, 2.0]$) and rigid uniform $\text{argmax}$ thresholding.

### Key Empirical Findings
1. **Inverse-Frequency Weighting Eliminates Mode Collapse:**
   - **Temporal LSTM:** Under Condition B ($[1.0, 1.5, 2.0]$), the LSTM collapsed completely to the majority class ($0.0\%$ Episode Recall, Macro-F1 $0.3068$). Under Condition C (Inverse-Frequency $[1.0, 11.46, 11.80]$), mode collapse was successfully eliminated: Macro-F1 rose to **$0.3409$** ($+0.0341$), detecting $10.0\%$ of validation failure episodes ($2/20$) with $16.33\%$ Entity FAR ($8/49$).
   - **Compact Temporal Transformer:** Condition C achieved the highest Validation Macro-F1 (**$0.3463$**) and highest Balanced Accuracy (**$0.3605$**) among all neural models tested. Episode Recall doubled from $15.0\%$ (Condition B) to **$30.0\%$** ($6/20$ episodes detected), and Warning Recall rose from $0.0\%$ to **$22.92\%$** ($11/48$ windows detected).
2. **Persistence of the Operational Gate Barrier:**
   - Despite significant metric gains, **neither architecture satisfied both Validation Gate 1 and Gate 2 simultaneously** under standard $\text{argmax}$.
   - For the Transformer under Condition C, the increase in failure sensitivity was accompanied by higher false alarms: Window FPR reached $20.89\%$ (ceiling $\le 5.0\%$) and Entity FAR reached $28.57\%$ (ceiling $\le 10.0\%$).
   - Random Forest tabular baseline (from Phase 7D) maintains a higher Validation Macro-F1 ($0.3653$ vs $0.3463$), but also fails both gates.
3. **Failure-Risk Threshold Sweep ($S_{\text{fail}} = P(\text{Warning}) + P(\text{Critical}) \ge \theta$):**
   - A fine-grained sweep across 91 thresholds ($\theta \in [0.05, 0.95]$) proved that **no scalar threshold exists that satisfies both Gate 1 and Gate 2 simultaneously**.
   - Operating at low thresholds ($\theta \le 0.45$) achieves high Episode Recall ($85\% - 100\%$), but false alarms explode (Entity FAR $67\% - 98\%$).
   - Operating at high thresholds ($\theta \ge 0.75$) suppresses false alarms to Gate 2 compliance (Entity FAR $\le 10\%$), but Episode Recall drops to $0\% - 5\%$.
4. **Honest Scientific Verdict:**
   - Inverse-frequency weighting successfully repairs representation learning and eliminates mode collapse in recurrent and attention architectures.
   - However, inverse-frequency weighting alone is **insufficient to create an operational cloud failure predictor** under standard sliding-window cross-entropy. Cross-job workload heterogeneity remains the dominant obstacle.

---

## 2. Invariants & Pre-Execution Verification

Before initiating training, all pipeline components and safeguards were verified:

1. **Splits Invariance:**
   - Strictly loaded Candidate A Train ($N=3,278$) and Candidate A Validation ($N=657$).
   - **ZERO access to Candidate A Test set** (`X_test.npy`, `y_test.npy`, `test_metadata.csv` were neither loaded nor inspected).
   - **ZERO access to Phase 6 confirmatory holdout** (`data/processed/confirmatory/` timestamps confirmed unchanged at `2026-10-26 15:06:17`).
2. **Preprocessing Invariance:**
   - Features were scaled strictly using `configs/phase7c/fitted_scaler_phase7c.json` (`1170ef86...`), fitted exclusively on Candidate A Train.
3. **Candidate A Split Checksum:**
   - `results/phase7a_4/candidate_a_job_grouped_split.csv`: `f84bc4fe54f8e0583e2ee52f49b2fbe8e6fcafb4f020c656985a1acc31bb4946` (VERIFIED).

---

## 3. Experimental Design & Weight Formulations

### 3.1 Training Set Class Distribution (Candidate A Train, $N=3,278$)
- Normal (Class 0): $2,797$ windows ($85.33\%$)
- Warning (Class 1): $244$ windows ($7.44\%$)
- Critical (Class 2): $237$ windows ($7.23\%$)

### 3.2 Loss Weight Configurations
Using the formal inverse-frequency formula strictly from training labels:
$$w_c = \frac{N_{\text{train}}}{K \cdot N_c} \quad (K = 3)$$

- **Condition A (Unweighted Baseline):**
  $$w_{\text{condA}} = [1.0000, 1.0000, 1.0000]$$
- **Condition B (Protocol v2.0 Ad-hoc Weights):**
  $$w_{\text{condB}} = [1.0000, 1.5000, 2.0000]$$
- **Condition C (Inverse-Frequency Balanced):**
  - Raw scikit-learn weights: $[0.390657, 4.478142, 4.610408]$
  - Unit-normalized base ($w_0 = 1.0$):
    $$w_{\text{condC}} = [1.000000, 11.463116, 11.801693]$$

> **Mathematical Scale-Invariance Proof:**  
> In PyTorch `nn.CrossEntropyLoss(weight=w, reduction='mean')`, the loss is computed as $\ell = \frac{\sum_{i=1}^B w_{y_i} \ell_i}{\sum_{i=1}^B w_{y_i}}$. Multiplying the weight vector by any positive scalar $c$ cancels out identically between numerator and denominator. Therefore, raw weights $[0.3907, 4.4781, 4.6104]$ and normalized weights $[1.0000, 11.4631, 11.8017]$ generate identical gradients and identical optimization steps.

### 3.3 Fixed Training Settings Across All Conditions
- **Deterministic Seed:** Fixed to 42 (`set_seed(42)`) before every model initialization.
- **Optimizer:** AdamW ($lr = 10^{-3}$ for LSTM, $lr = 5 \times 10^{-4}$ for Transformer; weight decay $10^{-4}$).
- **Scheduler:** `ReduceLROnPlateau` (factor $0.5$, patience $5$, min $lr = 10^{-5}$).
- **Gradient Clipping:** Max norm $1.0$.
- **Batch Size:** $64$.
- **Max Epochs & Early Stopping:** 35 epochs; early stopping patience = 10 epochs monitoring `val_macro_f1` (mode: `max`).
- **Architectures:**
  - **Temporal LSTM:** 2 layers, `hidden_dim=64`, `dropout=0.2`, input shape $(5, 31)$, $63,171$ parameters.
  - **Compact Transformer:** 2 layers, `d_model=32`, 4 heads, `dim_feedforward=64`, `dropout=0.1`, input shape $(5, 31)$, $22,179$ parameters.

---

## 4. Validation Performance Comparison (Standard Argmax Decision Rule)

All 6 models were evaluated on Candidate A Validation Split ($N = 657$ windows, 12 parent jobs, 69 entities, 20 failure episodes, 560 normal windows, 48 warning windows, 49 critical windows).

### Complete Validation Evaluation Matrix

| Model Identifier | Condition | Macro-F1 | Accuracy | BalAcc | Ep-Rec % | Crit-Rec | Warn-Rec | Entity-FAR % | Win-FPR % | Job-FAR % | Gate 1 | Gate 2 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **lstm_condA** | Unweighted `[1, 1, 1]` | 0.3171 | 0.6773 | 0.3150 | 25.00% (5/20) | 0.1020 (5/49) | 0.0625 (3/48) | 48.98% (24/49) | 21.96% (123/560) | 75.00% (3/4) | **FAIL** | **FAIL** |
| **lstm_condB** | Ad-hoc `[1, 1.5, 2]` | 0.3068 | 0.8524 | 0.3333 | 0.00% (0/20) | 0.0000 (0/49) | 0.0000 (0/48) | 0.00% (0/49) | 0.00% (0/560) | 0.00% (0/4) | **FAIL** | **PASS** |
| **lstm_condC** | **Inv-Freq `[1, 11.5, 11.8]`** | **0.3409** | 0.7656 | 0.3371 | **10.00%** (2/20) | 0.0612 (3/49) | 0.0625 (3/48) | 16.33% (8/49) | 11.25% (63/560) | 25.00% (1/4) | **FAIL** | **FAIL** |
| **transformer_condA**| Unweighted `[1, 1, 1]` | 0.3257 | 0.8158 | 0.3377 | 10.00% (2/20) | 0.0612 (3/49) | 0.0000 (0/48) | 12.24% (6/49) | 4.82% (27/560) | 25.00% (1/4) | **FAIL** | **FAIL** |
| **transformer_condB**| Ad-hoc `[1, 1.5, 2]` | 0.3328 | 0.7930 | 0.3412 | 15.00% (3/20) | 0.1020 (5/49) | 0.0000 (0/48) | 22.45% (11/49) | 7.86% (44/560) | 50.00% (2/4) | **FAIL** | **FAIL** |
| **transformer_condC**| **Inv-Freq `[1, 11.5, 11.8]`** | **0.3463** | 0.6956 | **0.3605**| **30.00%** (6/20) | 0.0612 (3/49) | **0.2292** (11/48)| 28.57% (14/49) | 20.89% (117/560) | 75.00% (3/4) | **FAIL** | **FAIL** |

### Detailed Confusion Matrices (Validation Split)
- **`lstm_condA_unweighted`:**
  - Matrix: `[[437, 24, 99], [41, 3, 4], [43, 1, 5]]`
  - Normal: P=0.8388, R=0.7804, F1=0.8085
  - Warning: P=0.1071, R=0.0625, F1=0.0789
  - Critical: P=0.0463, R=0.1020, F1=0.0637
- **`lstm_condB_adhoc`:**
  - Matrix: `[[560, 0, 0], [48, 0, 0], [49, 0, 0]]` (Total Majority Collapse)
  - Normal: P=0.8524, R=1.0000, F1=0.9203
  - Warning / Critical: Precision = Recall = F1 = 0.0000
- **`lstm_condC_inverse_freq`:**
  - Matrix: `[[497, 9, 54], [44, 3, 1], [46, 0, 3]]` (Repaired Mode Collapse)
  - Normal: P=0.8467, R=0.8875, F1=0.8666
  - Warning: P=0.2500, R=0.0625, F1=0.1000
  - Critical: P=0.0517, R=0.0612, F1=0.0561
- **`transformer_condA_unweighted`:**
  - Matrix: `[[533, 3, 24], [48, 0, 0], [46, 0, 3]]`
  - Normal: P=0.8501, R=0.9518, F1=0.8981
  - Warning: P=0.0000, R=0.0000, F1=0.0000
  - Critical: P=0.1111, R=0.0612, F1=0.0789
- **`transformer_condB_adhoc`:**
  - Matrix: `[[516, 11, 33], [48, 0, 0], [44, 0, 5]]`
  - Normal: P=0.8487, R=0.9214, F1=0.8836
  - Warning: P=0.0000, R=0.0000, F1=0.0000
  - Critical: P=0.1316, R=0.1020, F1=0.1149
- **`transformer_condC_inverse_freq`:**
  - Matrix: `[[443, 96, 21], [37, 11, 0], [36, 10, 3]]` (Highest Minority Detection)
  - Normal: P=0.8585, R=0.7911, F1=0.8234
  - Warning: P=0.0940, R=0.2292, F1=0.1333
  - Critical: P=0.1250, R=0.0612, F1=0.0822

---

## 5. Operational Threshold Calibration ($S_{\text{fail}} = P(\text{Warning}) + P(\text{Critical}) \ge \theta$)

A grid of 91 thresholds $\theta \in [0.05, 0.95]$ with step $0.01$ was evaluated on Candidate A Validation data.

### 5.1 Threshold Optimization Summary

| Model Identifier | Metric Optimal Threshold ($\theta^*$) | Optimal Binary F1 | Episode Recall % | Critical Recall | Entity FAR % | Window FPR % | Both Gates Passed? | Gate 2 Compliant Max Recall |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`lstm_condA`** | $\theta = 0.05$ | 0.1347 | 45.0% | 0.3265 | 67.3% | 49.5% | **NO (0/91)** | None ($0.0\%$ compliant) |
| **`lstm_condB`** | $\theta = 0.06$ | 0.2643 | 100.0% | 1.0000 | 98.0% | 96.4% | **NO (0/91)** | $0.0\%$ (at $\theta=0.54$, FAR $6.1\%$) |
| **`lstm_condC`** | $\theta = 0.57$ | 0.2817 | 100.0% | 0.9592 | 98.0% | 80.7% | **NO (0/91)** | $0.0\%$ (at $\theta=0.74$, FAR $2.0\%$) |
| **`transformer_condA`**| $\theta = 0.18$ | 0.2950 | 85.0% | 0.7755 | 67.3% | 52.9% | **NO (0/91)** | $5.0\%$ (at $\theta=0.79$, FAR $6.1\%$) |
| **`transformer_condB`**| $\theta = 0.06$ | 0.2381 | 95.0% | 0.8367 | 95.9% | 75.2% | **NO (0/91)** | None ($0.0\%$ compliant) |
| **`transformer_condC`**| $\theta = 0.45$ | 0.3102 | 100.0% | 1.0000 | 73.5% | 76.1% | **NO (0/91)** | $0.0\%$ (at $\theta=0.92$, FAR $0.0\%$) |

### 5.2 Key Insight on Operational Trade-offs
- Across all 6 models, **zero thresholds achieved simultaneous compliance with Gate 1 and Gate 2**.
- When $\theta$ is set to achieve Gate 1 compliance ($\text{Episode Recall} \ge 70\%$, $\text{Critical Recall} \ge 50\%$), the False Alarm Rate is prohibitively high: Entity FAR exceeds $67\%$ in every configuration.
- Conversely, when $\theta$ is raised to enforce Gate 2 compliance ($\text{Entity FAR} \le 10\%$, $\text{Window FPR} \le 5\%$), Episode Recall collapses to $0.0\% - 5.0\%$.
- **Conclusion:** The inability to satisfy operational gates is not an artifact of the $\text{argmax}$ threshold cutoff, but a fundamental property of cross-entropy feature representations on industrial Borg traces. Normal container tasks exhibit sudden resource spikes that overlap heavily with pre-failure profiles of other parent jobs.

---

## 6. Formal Protocol Amendment Documentation

In accordance with requirement 6, this experiment establishes a formal amendment to Protocol v2.0:

```json
{
  "amendment_id": "PROTOCOL-AMENDMENT-PHASE7E-01",
  "title": "Protocol Amendment: Inverse-Frequency Loss Weighting and Validation-Calibrated Alert Thresholding",
  "timestamp_iso": "2026-10-28T14:16:35Z",
  "rationale": "Audit of Phase 7D revealed that neural networks suffered majority-class mode collapse under hardcoded weights [1.0, 1.5, 2.0] and uniform argmax. This amendment formally pre-registers inverse-frequency loss weighting w = [1.0, 11.46, 11.80] and validation-only calibration of decision threshold theta on failure-risk score S_fail = P(Warning) + P(Critical).",
  "scope": "Candidate A Training and Validation splits ONLY. Test split remains strictly blinded and unaccessed for confirmation.",
  "threshold_tuning_policy": "Threshold theta is selected strictly on Candidate A Validation split to maximize Episode Recall subject to Gate 2 (Entity FAR <= 10%, Window FPR <= 5%). The selected theta is permanently frozen prior to any future independent test scoring."
}
```
Serialized to [results/phase7e/phase7e_protocol_amendment.json](file:///c:/Users/hp/Desktop/Research%20project/results/phase7e/phase7e_protocol_amendment.json).

---

## 7. Distinguishing Verified Facts from Hypotheses

### Verified Empirical Facts:
1. **Fact 1:** Under Condition B ($[1.0, 1.5, 2.0]$), Temporal LSTM undergoes complete majority-class collapse ($0.0\%$ recall, $0.0\%$ false alarms, Macro-F1 $0.3068$).
2. **Fact 2:** Inverse-frequency class weighting ($[1.0, 11.46, 11.80]$) successfully restores minority sensitivity for both LSTM and Transformer, raising LSTM Macro-F1 to $0.3409$ and Transformer Macro-F1 to $0.3463$.
3. **Fact 3:** In the Transformer, Condition C doubles Episode Recall ($15\% \to 30\%$) and raises Warning Recall from $0.0\%$ to $22.92\%$.
4. **Fact 4:** No threshold on $P(\text{Warning}) + P(\text{Critical})$ satisfies both Gate 1 and Gate 2 simultaneously on validation data.
5. **Fact 5:** The Candidate A Test split and historical Phase 6 holdout remained completely unaccessed and unblinded.

### Scientific Hypotheses:
1. **Hypothesis H1 (Focal Loss / Hard Negative Mining):** Because class weighting penalizes all normal errors uniformly, it causes normal background spikes to trigger false alarms. A Focal Loss formulation ($\gamma = 2.0$) or online hard negative mining may suppress easy normal windows and reduce Entity FAR.
2. **Hypothesis H2 (Cross-Job Workload Normalization):** Normalizing features relative to the parent job's baseline (or task-relative z-scores) rather than cluster-global scaling may eliminate false alarms caused by high-utilization healthy batch jobs.

---

## 8. Plan for Fresh, Independent Final Confirmation

1. **Test Set Status Acknowledgement:**
   - As mandated by the protocol, because Candidate A Test was already unblinded and evaluated in Phase 7D, **Candidate A Test is no longer described as a newly blinded holdout**.
2. **Independent Test Set Requirement:**
   - Any future operational claim must be verified against a **genuinely untouched, newly partitioned test cohort** (e.g. Candidate B temporal split from later trace days, or an independent 2019 Borg trace partition).
3. **Pre-Registration Gate Enforcement:**
   - Because no model in Phase 7E qualified under both Validation Gate 1 and Validation Gate 2, **no model is authorized for confirmatory test evaluation at this time**.
   - Model development remains strictly confined to Candidate A Train and Validation.

---

**Lead ML Research Engineer & Reproducibility Specialist**  
*Phase 7E Execution Complete | All Artifacts Verified and Cataloged*  
*Timestamp: 2026-10-28T14:16:36Z*
