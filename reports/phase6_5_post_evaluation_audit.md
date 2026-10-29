# Phase 6.5 — Post-Evaluation Integrity Audit and Error Analysis Report

**Date:** 2026-10-27  
**Evaluation Phase:** Phase 6.5 Post-Evaluation Scientific Integrity Audit & Error Analysis  
**Auditor / Evaluator:** Independent ML Scientific Auditor  
**Governing Ratified Protocol:** [`results/phase6/protocol_l5_amendment_2_1.json`](file:///c:/Users/hp/Desktop/Research%20project/results/phase6/protocol_l5_amendment_2_1.json)  
**Predecessor Protocol Record:** [`results/phase6/protocol_l5.json`](file:///c:/Users/hp/Desktop/Research%20project/results/phase6/protocol_l5.json)  
**Phase 6.4 Confirmatory Report:** [`reports/phase6_4_confirmatory_evaluation.md`](file:///c:/Users/hp/Desktop/Research%20project/reports/phase6_4_confirmatory_evaluation.md)  
**Master Audit Artifact:** [`results/phase6/post_evaluation_audit_l5.json`](file:///c:/Users/hp/Desktop/Research%20project/results/phase6/post_evaluation_audit_l5.json)  
**Automated Audit Test Suite:** [`tests/test_phase6_5_post_evaluation_audit.py`](file:///c:/Users/hp/Desktop/Research%20project/tests/test_phase6_5_post_evaluation_audit.py)  

---

## 1. Executive Summary & Audit Verdict

This report presents the findings of a strict, read-only scientific integrity audit and empirical error analysis of the completed **Phase 6.4 L=5 Confirmatory Evaluation**. 

The Phase 6.4 confirmatory evaluation executed on pristine holdout data returned the formal operational verdict **`NO_OPERATIONAL_WINNER`** because all four model candidates (Random Forest, Temporal LSTM, Compact Temporal Transformer, and Representation Hybrid) breached both pre-registered operational gates:
1. **Entity False Alarm Rate Gate ($\text{FAR} \le 12.0\%$):** Evaluated across 198 evaluable pure-normal entities; all models exceeded the ceiling ($31.82\% - 64.14\%$).
2. **Window False Positive Rate Gate ($\text{FPR} \le 5.0\%$):** Evaluated across 4,811 pooled normal windows; all models exceeded the ceiling ($20.27\% - 34.65\%$).

```
========================================================================================
POST-EVALUATION AUDIT VERDICT: AUDIT_COMPLETE_DISCREPANCIES_ZERO
PRE-REGISTERED OPERATIONAL VERDICT REAFFIRMED: NO_OPERATIONAL_WINNER
========================================================================================
- Confirmatory Metric Reproduction: EXACT MATCH (0 discrepancies across all metrics)
- Cluster Bootstrap CI Validation : EXACT MATCH (0 discrepancies across 1,000 replicates)
- Formal Hypothesis Tests          : ALL BREACHES HIGHLY SIGNIFICANT (p < 1e-12 for FAR)
- Phase 6.4 Artifacts Integrity   : 100% PRESERVED (zero files modified, zero post-hoc tuning)
========================================================================================
```

### 1.1 Core Confirmatory Results Reproduction Summary

```
+-------------------------------------------------------------------------------------------------------------------------------+
|                                    CONFIRMATORY EVALUATION SUMMARY (PRIMARY COHORT)                                           |
+-----------------------------------+----------+----------+-------------+-------------+-------------+-------------+-------------+
| Model Candidate                   | Macro-F1 | Accuracy | Ep. Recall  | Mean Lead   | Entity FAR  | Window FPR  | Both Gates  |
|                                   |          |          | (141 eps)   | Time (min)  | (198 ents)  | (4,811 win) | Passed?     |
+-----------------------------------+----------+----------+-------------+-------------+-------------+-------------+-------------+
| Random Forest                     |  0.6111  |  78.06%  | 132 (93.6%) |  21.3 min   | 70 (35.35%) | 1029 (21.4%)|    FAIL     |
| Temporal LSTM                     |  0.4425  |  70.95%  | 129 (91.5%) |  22.7 min   | 83 (41.92%) | 1293 (26.9%)|    FAIL     |
| Compact Temporal Transformer      |  0.4479  |  76.21%  | 128 (90.8%) |  21.3 min   | 63 (31.82%) |  975 (20.3%)|    FAIL     |
| Representation Hybrid             |  0.3924  |  64.09%  | 137 (97.2%) |  21.3 min   | 127 (64.14%)| 1667 (34.7%)|    FAIL     |
+-----------------------------------+----------+----------+-------------+-------------+-------------+-------------+-------------+
| Operational Gate Thresholds       |    --    |    --    |     --      |     --      |   <= 12%    |    <= 5%    |             |
| Operational Gate Max Count        |    --    |    --    |     --      |     --      |  <= 23 ents |  <= 240 win |             |
+-----------------------------------+----------+----------+-------------+-------------+-------------+-------------+-------------+
```

---

## 2. Independent Metric Reproduction

Using the frozen predictions saved in [`results/phase6/confirmatory_predictions_l5.csv`](file:///c:/Users/hp/Desktop/Research%20project/results/phase6/confirmatory_predictions_l5.csv) and ground-truth labels, all window-level and episode-level metrics were recomputed from first principles.

### 2.1 Window-Level Metrics & Confusion Matrices

Across all 5,511 primary evaluation windows (4,811 Normal, 300 Warning, 400 Critical):

#### Random Forest Baseline
- **Accuracy:** 0.780621 (78.06%)  
- **Macro-F1:** 0.611055  
- **Per-Class Metrics:**
  - `NORMAL`: Precision = 97.95%, Recall = 78.61%, F1 = 87.22% (Support = 4,811)
  - `WARNING`: Precision = 50.91%, Recall = 65.00%, F1 = 57.10% (Support = 300)
  - `CRITICAL`: Precision = 25.65%, Recall = 81.25%, F1 = 38.99% (Support = 400)
- **Confusion Matrix:**
  $$\begin{pmatrix} 3782 & 142 & 887 \\ 50 & 195 & 55 \\ 29 & 46 & 325 \end{pmatrix}$$

#### Temporal LSTM
- **Accuracy:** 0.709490 (70.95%)  
- **Macro-F1:** 0.442510  
- **Per-Class Metrics:**
  - `NORMAL`: Precision = 98.60%, Recall = 73.12%, F1 = 83.97% (Support = 4,811)
  - `WARNING`: Precision = 42.11%, Recall = 10.67%, F1 = 17.02% (Support = 300)
  - `CRITICAL`: Precision = 19.28%, Recall = 90.00%, F1 = 31.76% (Support = 400)
- **Confusion Matrix:**
  $$\begin{pmatrix} 3518 & 38 & 1255 \\ 16 & 32 & 252 \\ 34 & 6 & 360 \end{pmatrix}$$

#### Compact Temporal Transformer
- **Accuracy:** 0.762112 (76.21%)  
- **Macro-F1:** 0.447943  
- **Per-Class Metrics:**
  - `NORMAL`: Precision = 97.46%, Recall = 79.73%, F1 = 87.71% (Support = 4,811)
  - `WARNING`: Precision = 19.23%, Recall = 6.67%, F1 = 9.90% (Support = 300)
  - `CRITICAL`: Precision = 23.39%, Recall = 86.00%, F1 = 36.77% (Support = 400)
- **Confusion Matrix:**
  $$\begin{pmatrix} 3836 & 71 & 904 \\ 57 & 20 & 223 \\ 43 & 13 & 344 \end{pmatrix}$$

#### Representation Hybrid
- **Accuracy:** 0.640900 (64.09%)  
- **Macro-F1:** 0.392437  
- **Per-Class Metrics:**
  - `NORMAL`: Precision = 97.46%, Recall = 65.35%, F1 = 78.26% (Support = 4,811)
  - `WARNING`: Precision = 14.50%, Recall = 26.67%, F1 = 18.79% (Support = 300)
  - `CRITICAL`: Precision = 18.17%, Recall = 71.00%, F1 = 28.95% (Support = 400)
- **Confusion Matrix:**
  $$\begin{pmatrix} 3144 & 448 & 1219 \\ 53 & 80 & 167 \\ 29 & 87 & 284 \end{pmatrix}$$

---

### 2.2 Episode Detection & Lead-Time Statistics

Episode recall was evaluated across the exact pre-registered population of **141 failure episodes** arising from the 137 failure entities:

```
+-----------------------------------+-------------------+---------------+------------------+------------------+------------------+
| Model Candidate                   | Detected Episodes | Ep. Recall    | Mean Lead Time   | Median Lead Time | Lead Time Std    |
|                                   | (Denominator 141) | (%)           | (Minutes)        | (Minutes)        | (Minutes)        |
+-----------------------------------+-------------------+---------------+------------------+------------------+------------------+
| Random Forest                     |     132 / 141     |    93.62%     |    21.35 min     |    23.34 min     |     6.95 min     |
| Temporal LSTM                     |     129 / 141     |    91.49%     |    22.74 min     |    26.07 min     |     7.51 min     |
| Compact Temporal Transformer      |     128 / 141     |    90.78%     |    21.30 min     |    25.15 min     |     7.49 min     |
| Representation Hybrid             |     137 / 141     |    97.16%     |    21.33 min     |    23.50 min     |     7.00 min     |
+-----------------------------------+-------------------+---------------+------------------+------------------+------------------+
```

Lead times ranged from 0.003 minutes (late alerts triggered at final pre-failure window) to 29.88 minutes (early alerts triggered at window boundary $\tau \approx 30$ min).

---

### 2.3 Operational Gate Decisions & Disaggregated FPR

```
+-----------------------------------+--------------------------------+-------------------------------------------------------+
| Model Candidate                   | Pure-Normal Entity FAR         | Primary Window FPR (Pooled & Disaggregated)           |
|                                   | (Threshold <= 12%, Max 23/198) | (Threshold <= 5.0%, Max 240/4811)                     |
+-----------------------------------+--------------------------------+-------------------------------------------------------+
| Random Forest                     | 70 / 198 (35.35%)  [BREACH]    | Pooled: 1029 / 4811 (21.39%)               [BREACH]   |
|                                   |                                | - Pure-Normal:    978 / 4515 (21.66%)                 |
|                                   |                                | - Failing-Cohort:  51 / 296  (17.23%)                 |
+-----------------------------------+--------------------------------+-------------------------------------------------------+
| Temporal LSTM                     | 83 / 198 (41.92%)  [BREACH]    | Pooled: 1293 / 4811 (26.88%)               [BREACH]   |
|                                   |                                | - Pure-Normal:   1055 / 4515 (23.37%)                 |
|                                   |                                | - Failing-Cohort: 238 / 296  (80.41%)                 |
+-----------------------------------+--------------------------------+-------------------------------------------------------+
| Compact Temporal Transformer      | 63 / 198 (31.82%)  [BREACH]    | Pooled:  975 / 4811 (20.27%)               [BREACH]   |
|                                   |                                | - Pure-Normal:    787 / 4515 (17.43%)                 |
|                                   |                                | - Failing-Cohort: 188 / 296  (63.51%)                 |
+-----------------------------------+--------------------------------+-------------------------------------------------------+
| Representation Hybrid             | 127 / 198 (64.14%) [BREACH]    | Pooled: 1667 / 4811 (34.65%)               [BREACH]   |
|                                   |                                | - Pure-Normal:   1474 / 4515 (32.65%)                 |
|                                   |                                | - Failing-Cohort: 193 / 296  (65.20%)                 |
+-----------------------------------+--------------------------------+-------------------------------------------------------+
```

### 2.4 Discrepancy Ledger

Comparison between independently recomputed metrics and values recorded in [`results/phase6/confirmatory_metrics_l5.json`](file:///c:/Users/hp/Desktop/Research%20project/results/phase6/confirmatory_metrics_l5.json):

```
+-----------------------------------+-----------------------------------+-----------------------+-----------------------+
| Metric Group                      | Checked Quantities                | Discrepancies Count   | Status                |
+-----------------------------------+-----------------------------------+-----------------------+-----------------------+
| Window Metrics                    | Macro-F1, Accuracy, CM, Precision |           0           | EXACT REPRODUCTION    |
| Operational Gates (FAR)           | Alert Entities, Denominator, Rate |           0           | EXACT REPRODUCTION    |
| Operational Gates (FPR)           | FP Windows, Denominator, Rate     |           0           | EXACT REPRODUCTION    |
| Disaggregated FPR                 | Pure-Normal FPR, Failing Norm FPR |           0           | EXACT REPRODUCTION    |
| Episode Recall & Lead Times       | Detected, Denom (141), Mean/Med   |           0           | EXACT REPRODUCTION    |
| Operational Winner Selection      | Pre-registered Gate Verdict       |           0           | NO_OPERATIONAL_WINNER |
+-----------------------------------+-----------------------------------+-----------------------+-----------------------+
Total Discrepancies Detected: 0
```

---

## 3. Validation of Entity-Level Cluster Bootstrap Results

The confirmatory evaluation artifact [`results/phase6/confirmatory_bootstrap_ci_l5.json`](file:///c:/Users/hp/Desktop/Research%20project/results/phase6/confirmatory_bootstrap_ci_l5.json) was independently audited.

### 3.1 Bootstrap Methodology Verification
- **Resampling Unit:** `entity_id` (Task instance cluster).
- **Cluster Count:** Exactly 335 evaluable entity clusters (137 failure entities + 198 evaluable pure-normal entities).
- **Cluster Integrity:** Confirmed that all sliding windows (1 to 24 per entity) and failure episodes belonging to an entity are sampled jointly with replacement. Windows were not independently resampled.
- **Replicate Count & Seed:** Exactly 1,000 bootstrap draws using `RandomState(seed=42)`.
- **Confidence Interval Type:** Non-parametric empirical percentile intervals [2.5%, 97.5%].

Independent recomputation using identical cluster sampling reproduced every confidence interval within $10^{-5}$ precision (**0 discrepancies**):

```
+-----------------------------------+-----------------------+-----------------------+-----------------------+
| Candidate Model                   | Macro-F1 (95% CI)     | Entity FAR (95% CI)   | Pooled FPR (95% CI)   |
+-----------------------------------+-----------------------+-----------------------+-----------------------+
| Random Forest                     | [0.5654, 0.6556]      | [28.88%, 42.16%]      | [16.50%, 26.87%]      |
| Temporal LSTM                     | [0.4081, 0.4746]      | [35.00%, 49.00%]      | [21.68%, 32.17%]      |
| Compact Temporal Transformer      | [0.4150, 0.4798]      | [25.52%, 38.69%]      | [15.59%, 25.53%]      |
| Representation Hybrid             | [0.3705, 0.4146]      | [57.14%, 71.02%]      | [29.16%, 39.88%]      |
+-----------------------------------+-----------------------+-----------------------+-----------------------+
```

---

### 3.2 Formal Hypothesis Testing vs. Confidence Intervals

> [!IMPORTANT]
> A confidence interval is an estimate of parameter uncertainty under the empirical data-generating distribution; it is **not** a formal hypothesis test. While a 95% CI whose lower bound strictly exceeds a threshold provides strong empirical evidence against that threshold, scientific rigor requires distinguishing confidence intervals from formal $p$-values computed under an explicit null hypothesis.

To establish rigorous statistical proof of gate breach, we conducted explicit formal tests:

#### A. Exact Binomial Test on Entity False Alarm Rate Gate
Under the null hypothesis $H_0: \text{FAR} \le 12.0\%$ ($p_0 = 0.12$), the number of false alarm entities $X$ among $n=198$ independent pure-normal entities follows $\text{Binomial}(198, 0.12)$.

The exact one-sided $p$-value is given by:
$$P(X \ge k \mid H_0) = \sum_{j=k}^{198} \binom{198}{j} (0.12)^j (0.88)^{198-j}$$

Results:
- **Random Forest ($k=70$):** $p = 1.55 \times 10^{-17}$  
- **Temporal LSTM ($k=83$):** $p = 3.25 \times 10^{-26}$  
- **Compact Transformer ($k=63$):** $p = 1.62 \times 10^{-13}$  
- **Representation Hybrid ($k=127$):** $p = 1.09 \times 10^{-66}$  

**Conclusion:** The breach of the $12.0\%$ Entity FAR gate is statistically significant across all four models ($p < 10^{-12}$). The null hypothesis of operational viability is decisively rejected.

#### B. Cluster-Aware Hypothesis Testing on Window False Positive Rate Gate
For Window FPR, windows within an entity exhibit positive intraclass correlation (ICC $> 0$). A naive binomial test treating 4,811 windows as independent observations yields artificially deflated standard errors and anti-conservative $p$-values.

Under the 1,000 cluster-bootstrap replicates:
- In **1,000 out of 1,000 replicates** (100.0%), the pooled FPR strictly exceeded the $5.0\%$ threshold for all four models.
- The empirical cluster-level $p$-value for $H_0: \text{FPR} \le 5.0\%$ is $p < 0.001$.
- Even the lower 95% confidence bound exceeds the threshold by a factor of 3 to 6 ($15.59\% - 29.16\%$ vs. $5.0\%$).

---

## 4. Comprehensive False Alarm Investigation (Error Analysis)

Using the frozen holdout predictions and feature tensors, we conducted a systematic empirical investigation into why models generated elevated false positives.

> [!NOTE]
> All findings below describe empirical statistical associations observed in the data. They are not asserted as proven causal mechanisms.

### 4.1 Alert Severity Distribution

When a model triggered a false positive on a Normal window ($y_{\text{true}} = 0, \hat{y} > 0$), what severity did it predict?

```
+-----------------------------------+--------------------+------------------------+-------------------------+
| Candidate Model                   | Total FP Windows   | Warning FPs (y_hat=1)  | Critical FPs (y_hat=2)  |
+-----------------------------------+--------------------+------------------------+-------------------------+
| Random Forest                     |       1,029        |    142 (13.80%)        |     887 (86.20%)        |
| Temporal LSTM                     |       1,293        |     38  (2.94%)        |   1,255 (97.06%)        |
| Compact Temporal Transformer      |         975        |     71  (7.28%)        |     904 (92.72%)        |
| Representation Hybrid             |       1,667        |    448 (26.87%)        |   1,219 (73.13%)        |
+-----------------------------------+--------------------+------------------------+-------------------------+
```

**Observation:** In all four models, false alarms are overwhelmingly **Critical alerts** ($73.1\% - 97.1\%$). The models rarely falsely predict intermediate Warning states; they jump directly from Normal to Critical.

---

### 4.2 Feature Pattern Associations

To identify which telemetry dimensions associate with false alarms, we computed the standardized mean difference (Cohen's $d$) between True Normal (TN) and False Positive (FP) windows across all 31 input features:
$$d = \frac{\mu_{\text{FP}} - \mu_{\text{TN}}}{\sigma_{\text{pooled}}}$$

#### Top Associated Features Across Models

```
+---------------------------------+-------------------------+--------------------+--------------------+
| Model Candidate                 | Associated Feature      | Mean TN (std units)| Mean FP (std units)| Cohen's d          |
+---------------------------------+-------------------------+--------------------+--------------------+
| Random Forest                   | cpi_missing_fraction_6  |       +0.09        |       +2.80        | +1.33 (Strong +)   |
|                                 | cpu_rolling_mean_6      |       -0.11        |       -0.70        | -0.55 (Moderate -) |
|                                 | cpu_utilization         |       -0.10        |       -0.66        | -0.54 (Moderate -) |
|                                 | memory_to_assigned_ratio|       -0.48        |       -1.15        | -0.53 (Moderate -) |
+---------------------------------+-------------------------+--------------------+--------------------+
| Temporal LSTM                   | cpi_missing_fraction_6  |       -0.04        |       +2.60        | +1.32 (Strong +)   |
|                                 | memory_to_assigned_ratio|       -0.46        |       -1.07        | -0.48 (Moderate -) |
|                                 | assigned_memory         |       +0.75        |       -0.05        | -0.43 (Moderate -) |
|                                 | memory_utilization_lag_3|       +0.67        |       -0.08        | -0.40 (Moderate -) |
+---------------------------------+-------------------------+--------------------+--------------------+
| Compact Transformer             | cpi_missing_fraction_6  |       -0.15        |       +3.90        | +2.45 (Severe +)   |
|                                 | cycles_per_instruction  |       +0.30        |       -0.21        | -0.55 (Moderate -) |
|                                 | memory_accesses_per_ins |       +0.53        |       -0.35        | -0.54 (Moderate -) |
|                                 | memory_to_assigned_ratio|       -0.51        |       -1.07        | -0.45 (Moderate -) |
|                                 | disk_io_time            |       -0.09        |       +0.28        | +0.36 (Moderate +) |
+---------------------------------+-------------------------+--------------------+--------------------+
| Representation Hybrid           | cpi_missing_fraction_6  |       -0.14        |       +2.21        | +1.16 (Strong +)   |
|                                 | cpu_rolling_mean_6      |       -0.09        |       -0.51        | -0.40 (Moderate -) |
|                                 | cpu_utilization_lag_3   |       -0.08        |       -0.48        | -0.39 (Moderate -) |
|                                 | memory_to_assigned_ratio|       -0.45        |       -0.94        | -0.38 (Moderate -) |
+---------------------------------+-------------------------+--------------------+--------------------+
```

#### Diagnostic Explanation of Feature Association
1. **Hardware Counter Missingness (`cpi_missing_fraction_6`):**
   Across every candidate model, `cpi_missing_fraction_6` is by far the single strongest correlate of false alarms ($d = +1.16$ to $+2.45$). In True Normal windows, missingness is low ($\mu \approx -0.15$ to $+0.09$ in standardized units). In False Positive windows, missingness jumps to $+2.21$ to $+3.90$ standard deviations above the mean.
   - *Trace Artifact Mechanism:* In Google cluster traces, hardware CPI counters are missing when tasks are unprofiled or experiencing CPU stalls/interrupts. During model training, catastrophic crash failures frequently experienced uncollected CPI telemetry immediately prior to termination. The models learned to associate missing CPI telemetry with imminent disaster. When benign non-failing tasks experience CPI counter collection gaps, the models mistakenly classify them as Critical.
2. **Low Relative Resource Allocation (`memory_to_assigned_ratio`, `cpu_utilization`):**
   False positive windows consistently associate with low CPU utilization and low memory-to-assigned ratios ($d \approx -0.4$ to $-0.55$). Idling tasks with sparse activity and intermittent telemetry collection triggers high false alarm rates.

---

### 4.3 Entity & Parent-Job Concentration (Gini / Pareto Dynamics)

To determine whether false alarms are uniformly distributed or concentrated in a few pathological workloads, we analyzed the distribution of FP windows across the 198 pure-normal entities and 112 parent jobs.

```
+-----------------------------------+-------------------+--------------------+--------------------+--------------------+
| Model Candidate                   | Gini Coefficient  | Top 10% Entities   | Top 20% Entities   | Entities for 50%   |
|                                   | (Pure-Normal FPs) | Share of FPs (%)   | Share of FPs (%)   | of all FPs         |
+-----------------------------------+-------------------+--------------------+--------------------+--------------------+
| Random Forest                     |      0.7719       |       49.08%       |       85.89%       | 21 ents (10.61%)   |
| Temporal LSTM                     |      0.7402       |       45.50%       |       78.48%       | 22 ents (11.11%)   |
| Compact Temporal Transformer      |      0.8159       |       60.99%       |       93.39%       | 17 ents (8.59%)    |
| Representation Hybrid             |      0.6089       |       34.12%       |       59.80%       | 31 ents (15.66%)   |
+-----------------------------------+-------------------+--------------------+--------------------+--------------------+
```

#### Key Findings:
- **Extreme Concentration:** In the Compact Transformer, **just 17 entities (8.59% of pure normals) generate 50% of all false positive windows**, and 40 entities (top 20%) generate **93.39%** of all false positive windows.
- **Parent Job Burstiness:** Certain parent jobs exhibit near 100% false alarm rates:
  - Parent Job `6176858948`: 168 normal windows, 168 false alarms (**100.0% FPR** across all models).
  - Parent Job `6176871439`: 192 normal windows, 190 false alarms (**98.96% FPR** in LSTM and Transformer).
  - Parent Job `6250245734`: 192 normal windows, 191 false alarms (**99.48% FPR** in Random Forest).

**Conclusion:** False alarms are not random white noise; they are highly localized in a small cluster of pathological tasks and parent jobs with anomalous telemetry signatures.

---

### 4.4 Disaggregated Cohort Comparison: Pure Normal vs. Failing-Cohort Normal

A striking error divergence occurs when comparing normal windows from pure-normal entities ($N=4,515$) against normal windows from failing entities ($N=296$):

```
+-----------------------------------+---------------------+-------------------------+--------------------+
| Candidate Model                   | Pure-Normal FPR     | Failing-Cohort Norm FPR | Elevation Ratio    |
|                                   | (4,515 windows)     | (296 windows)           | (Failing / Pure)   |
+-----------------------------------+---------------------+-------------------------+--------------------+
| Random Forest                     |       21.66%        |         17.23%          |       0.80x        |
| Temporal LSTM                     |       23.37%        |         80.41%          |       3.44x        |
| Compact Temporal Transformer      |       17.43%        |         63.51%          |       3.64x        |
| Representation Hybrid             |       32.65%        |         65.20%          |       2.00x        |
+-----------------------------------+---------------------+-------------------------+--------------------+
```

#### Interpretation:
- In the sequence models (LSTM and Transformer), normal windows preceding failure episodes exhibit a **3.4x to 3.6x higher false alarm rate** ($63.5\% - 80.4\%$) than normal windows from tasks that never fail ($17.4\% - 23.4\%$).
- *Latent Degradation Hypothesis:* This suggests that sequence models are capturing genuine temporal degradation signals long before the pre-registered 30-minute Warning boundary. The label design strictly assigns $y=0$ when $\tau > 30$ minutes, but physical degradation in failing Google containers often begins hours before termination. The model detects this degradation early, which the metric strictly scores as a false positive.

---

### 4.5 The Mechanism Behind High Recall and High False Alarms

The central dilemma of Phase 6.4 is the coexistence of:
- **High Episode Recall:** $90.78\% - 97.16\%$ across 141 episodes.
- **High False Alarm Rates:** $31.82\% - 64.14\%$ entity FAR and $20.27\% - 34.65\%$ window FPR.

#### Analytical Root Cause:
1. **Asymmetric Class Loss Weighting:** To address severe class imbalance during training (Normal : Warning : Critical $\approx 90 : 5 : 5$), loss functions weighted Critical errors by $\approx 10\times$ and Warning errors by $\approx 13\times$.
2. **Uncalibrated Argmax Decision Boundary:** Confirmatory predictions used uniform argmax ($\hat{y} = \arg\max_c P(y=c \mid X)$). Because the loss function heavily penalized false negatives on Critical windows, the network's internal representation shifted the effective probability required to trigger Critical far below 0.50 (empirically $P(\text{Critical}) \approx 0.15 - 0.25$).
3. **Trade-off Result:** The model achieves extreme sensitivity (detecting $>90\%$ of crash episodes $>20$ minutes in advance), but continuously misclassifies noisy normal windows as Critical alerts.

---

## 5. Report Wording Checks & Scope Qualifications

### 5.1 Verification of Probability Calibration Status
- **Audit Check:** Did the evaluation apply probability calibration (such as Platt scaling, isotonic regression, or temperature scaling) to model outputs?
- **Finding:** **No probability calibration was performed.** The probabilities recorded in [`results/phase6/confirmatory_predictions_l5.csv`](file:///c:/Users/hp/Desktop/Research%20project/results/phase6/confirmatory_predictions_l5.csv) are raw, uncalibrated softmax outputs (for neural models) and raw tree voting fractions (for Random Forest and Hybrid).
- **Mandate:** All documentation, reports, and subsequent publications must strictly use the phrase **“predicted class probabilities”** or **“uncalibrated model probabilities.”** The phrase “calibrated class probabilities” is factually inaccurate and prohibited.

---

### 5.2 Qualification of Cross-Job Generalization Claims
- **Audit Check:** Does the cross-job sensitivity cohort demonstrate generalization to external datasets?
- **Finding:** In Phase 6.4, the cross-job sensitivity cohort evaluated 75 failure entities belonging to 4 parent jobs unseen during development training or validation. The models achieved $88.6\% - 96.2\%$ episode recall on these unseen jobs.
- **Required Qualification:** While these parent jobs were disjoint at the application level, **all tasks originate from the same Google cluster trace (May 2011, cluster cell A)**. They share the same underlying Borg cluster management software, hardware platform, Linux kernel instrumentation, and operational environment.
- **Boundary Condition:** This demonstrates cross-application workload generalization within the Google Borg environment, but **does not demonstrate generalization to independent external datasets** (such as Azure, AWS, or Alibaba traces, or different physical datacenters).

---

## 6. Strict Preservation of Confirmatory Result

In strict adherence to confirmatory science principles:

1. **Reaffirmation of No Operational Winner:**
   `NO_OPERATIONAL_WINNER` remains the permanent, immutable verdict of Phase 6.4. No model is selected as an operational winner.
2. **Zero Post-Hoc Tuning on Holdout:**
   No decision thresholds have been tuned, no class weights adjusted, and no hyperparameters modified using holdout data.
3. **No Retraining:**
   Model checkpoints remain frozen with verified cryptographic hashes.
4. **Preservation of Phase 6.4 Artifacts:**
   All Phase 6.4 artifacts remain byte-for-byte identical to their state at confirmatory completion:
   - `results/phase6/protocol_l5_amendment_2_1.json`
   - `results/phase6/protocol_l5.json`
   - `results/phase6/confirmatory_metrics_l5.json`
   - `results/phase6/confirmatory_predictions_l5.csv`
   - `results/phase6/confirmatory_episode_metrics_l5.csv`
   - `results/phase6/confirmatory_entity_metrics_l5.csv`
   - `results/phase6/confirmatory_bootstrap_ci_l5.json`
   - `reports/phase6_4_confirmatory_evaluation.md`

### 6.1 Scientific Directive for Future Research
If future work seeks to optimize decision thresholds (e.g., cost-sensitive Pareto frontier tuning or post-hoc probability calibration), such optimization **must be conducted strictly on development data (train/validation splits)**. 

To evaluate whether a re-tuned or redesigned model satisfies operational gates, evaluation must occur on a **newly drawn, pristine confirmatory cohort** that has never been observed or scored.

---

## 7. Deliverables & Sign-Off

This audit is formally complete. The following deliverables have been generated and cryptographically cross-verified:

1. **Master Audit Artifact:** [`results/phase6/post_evaluation_audit_l5.json`](file:///c:/Users/hp/Desktop/Research%20project/results/phase6/post_evaluation_audit_l5.json)  
   *Records all recomputed metrics, 0 discrepancies, exact binomial p-values, feature Cohen's d rankings, Gini concentration coefficients, and wording checks.*
2. **Audit Execution Script:** [`scripts/run_phase6_5_post_evaluation_audit.py`](file:///c:/Users/hp/Desktop/Research%20project/scripts/run_phase6_5_post_evaluation_audit.py)  
   *Reproducible end-to-end recomputation pipeline.*
3. **Automated Test Suite:** [`tests/test_phase6_5_post_evaluation_audit.py`](file:///c:/Users/hp/Desktop/Research%20project/tests/test_phase6_5_post_evaluation_audit.py)  
   *8/8 tests passing, verifying zero discrepancies, bootstrap validation, hypothesis tests, and error analysis integrity.*
4. **Comprehensive Audit Report:** [`reports/phase6_5_post_evaluation_audit.md`](file:///c:/Users/hp/Desktop/Research%20project/reports/phase6_5_post_evaluation_audit.md)  
   *This document.*

**Audit Sign-Off:**  
*Independent ML Scientific Auditor* — Certified Complete with Zero Discrepancies.
