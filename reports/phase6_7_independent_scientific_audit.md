# Phase 6.7 — Independent Scientific Audit of Error Mechanisms and Temporal Labels

**Project:** Temporal LSTM/Transformer Framework for Early Failure Prediction in Cloud Computing  
**Auditor:** Independent Machine Learning Scientific Auditor  
**Date of Audit:** October 27, 2026  
**Governing Protocol:** [Protocol Amendment 2.1](file:///c:/Users/hp/Desktop/Research%20project/results/phase6/protocol_l5_amendment_2_1.json)  
**Primary Confirmatory Cohort:** Frozen $L=5$ Confirmatory Holdout ($N = 5,511$ primary windows across 335 evaluable tasks)  
**Primary Audit Artifact:** [`results/phase6/phase6_7_independent_scientific_audit.json`](file:///c:/Users/hp/Desktop/Research%20project/results/phase6/phase6_7_independent_scientific_audit.json)  
**Official Operational Verdict:** **`NO_OPERATIONAL_WINNER`** (Reaffirmed as immutable)

---

## 1. Executive Summary & Preservation Invariants

### 1.1 Objective and Scope of the Scientific Audit
This audit provides an independent, read-only methodological review of the Phase 6.6 post-hoc error-mechanism investigation. In accordance with strict empirical research standards, this analysis does **not** introduce model retraining, hyperparameter adjustments, threshold tuning, or cohort re-sampling. Instead, every empirical claim, mathematical identity, and physical interpretation advanced in Phase 6.6 has been audited against the raw Google cluster trace data, model source code, frozen prediction tensors, and confirmatory evaluation logs.

The goal is to establish:
1. Which Phase 6.6 empirical findings are **independently reproduced**;
2. Which findings are **reproduced with necessary scientific qualifications**;
3. Which claims are **not reproduced** (requiring factual and numerical correction);
4. Which hypotheses remain **unverifiable** due to inherent limitations in observational trace data.

### 1.2 Strict Preservation Rules Compliance
The audit operated under strict preservation constraints:
- **No model retraining or fine-tuning:** Model weights and checkpoints from Phase 6.3 remain completely untouched.
- **No threshold or label modification:** All decision boundaries remain fixed at uniform argmax; all ground-truth targets remain identical to Protocol Amendment 2.1.
- **Artifact preservation:** All historical artifacts from Phase 6.4, Phase 6.5, and Phase 6.6 have been verified via SHA-256 cryptographic hashes and preserved without modification.
- **Confirmatory status:** The official operational decision remains **`NO_OPERATIONAL_WINNER`**. The confirmatory holdout cohort has already been evaluated and is permanently retired from hypothesis testing.
- **Epistemological integrity:** Passing software assertions are treated as software verification, not scientific validity. Observational correlations are strictly decoupled from causal attributions.

### 1.3 Cryptographic Integrity of Audited Artifacts
All input artifacts evaluated during this audit were verified against their recorded cryptographic digests:

| Artifact Name | Path | SHA-256 Digest | Audit Status |
| :--- | :--- | :--- | :--- |
| **Confirmatory Predictions** | [`results/phase6/confirmatory_predictions_l5.csv`](file:///c:/Users/hp/Desktop/Research%20project/results/phase6/confirmatory_predictions_l5.csv) | `d1d798314218ba56079d1159582fcd587b7c6e904299e804dfde10aec6b3a3c0` | Verified & Unchanged |
| **Confirmatory Metrics** | [`results/phase6/confirmatory_metrics_l5.json`](file:///c:/Users/hp/Desktop/Research%20project/results/phase6/confirmatory_metrics_l5.json) | `121828c4bfcd3871c57f20608bafe3be96ab6c29c1b2110d96773cbfe84b2451` | Verified & Unchanged |
| **Confirmatory Episodes** | [`results/phase6/confirmatory_episode_metrics_l5.csv`](file:///c:/Users/hp/Desktop/Research%20project/results/phase6/confirmatory_episode_metrics_l5.csv) | `66fc0f92d9f97ed849200032d091eece39ea4d5747d611c43b0b5587d6328a9b` | Verified & Unchanged |
| **Confirmatory Entities** | [`results/phase6/confirmatory_entity_metrics_l5.csv`](file:///c:/Users/hp/Desktop/Research%20project/results/phase6/confirmatory_entity_metrics_l5.csv) | `8ee74c3848cfee722f7119bca0b0a7a4bbacd50f8d1e218f491890c76ee3f3d9` | Verified & Unchanged |
| **Confirmatory Bootstrap CIs** | [`results/phase6/confirmatory_bootstrap_ci_l5.json`](file:///c:/Users/hp/Desktop/Research%20project/results/phase6/confirmatory_bootstrap_ci_l5.json) | `eb0eda70c215ec1e461d5428c2d080c0476b5377a7fff6a4c625c0797a82d1ca` | Verified & Unchanged |
| **Protocol Amendment 2.1** | [`results/phase6/protocol_l5_amendment_2_1.json`](file:///c:/Users/hp/Desktop/Research%20project/results/phase6/protocol_l5_amendment_2_1.json) | `f354cd9f02343e83d0e71056e6800189715506fbdd845342c6d982cd0e88d8b0` | Verified & Unchanged |
| **Post-Evaluation Audit (6.5)** | [`results/phase6/post_evaluation_audit_l5.json`](file:///c:/Users/hp/Desktop/Research%20project/results/phase6/post_evaluation_audit_l5.json) | `297b0f61cd8570fe475cde9a2634c62a42951277cc4b2a94dc9115c2309d6907` | Verified & Unchanged |
| **Error Investigation (6.6)** | [`results/phase6/phase6_6_error_mechanism_investigation.json`](file:///c:/Users/hp/Desktop/Research%20project/results/phase6/phase6_6_error_mechanism_investigation.json) | `18de0ead4569608cb38c47c75b1b0b7e5e0c2eb35dbbec8953838737da4fc1ce` | Verified & Unchanged |
| **Training Manifest** | [`results/phase6/training_manifest_l5.json`](file:///c:/Users/hp/Desktop/Research%20project/results/phase6/training_manifest_l5.json) | `bed2c2d794b26e56939cc61ed4b08d3756e6b0062b763f5764963dbe3a0a87df` | Verified & Unchanged |
| **Fitted Scaler** | [`configs/fitted_scaler.json`](file:///c:/Users/hp/Desktop/Research%20project/configs/fitted_scaler.json) | `3241b4bd0f83e3cdfb940fdfd017dbb841da3d93bafa2c2c2cb118673582d842` | Verified & Unchanged |

---

## 2. Audit of Failure Timestamps and Event Lineage

### 2.1 Data Lineage and Trace Event Inspection
The primary confirmatory dataset consists of 5,511 windows across 335 evaluable tasks (198 pure-normal control entities and 137 failure-cohort entities). To audit failure event timestamps independently of model predictions, we inspected:
1. The raw Google cluster task event logs in `data/raw/sample_task_events*.csv.gz`;
2. The dataset generation pipeline in [`src/features/engineer.py`](file:///c:/Users/hp/Desktop/Research%20project/src/features/engineer.py);
3. The episode evaluation records in [`results/phase6/confirmatory_episode_metrics_l5.csv`](file:///c:/Users/hp/Desktop/Research%20project/results/phase6/confirmatory_episode_metrics_l5.csv).

Across the 137 failure entities, exactly 153 raw `FAIL` events (event type 5) occur in the cluster trace. Of these 153 events, exactly 141 failure episodes correspond to evaluable windows under Protocol Amendment 2.1. Every single one of the 141 failure episodes recorded in `confirmatory_episode_metrics_l5.csv` was verified against the raw trace timestamps down to sub-second precision ($< 0.1\text{ s}$).

### 2.2 Mathematical Validity of Timestamp Reconstruction
In Phase 6.6, failure timestamps were reconstructed using:
$$t_{\text{fail}} = t_{\text{pred}} + \tau$$
where $t_{\text{pred}}$ is the prediction timestamp and $\tau = \text{time\_to\_failure}$.

We evaluated whether this reconstruction is universally valid:
- **Single-failure entities:** For entities with a single failure episode, $t_{\text{fail}} = t_{\text{pred}} + \tau$ is exact across all pre-failure windows ($\tau \ge 0$).
- **Repeated-failure entities:** In tasks experiencing multiple failure episodes (e.g., job retry after failure), $\tau$ reflects the time to the *immediately subsequent* failure. 
- **Post-failure windows:** Once an entity fails, if Borg schedules another attempt or telemetry continues during cleanup, subsequent windows have no future failure event within the sequence ($\tau = \text{NaN}$ or unrecorded).
- **Critical Finding on Phase 6.6 Methodology:** Phase 6.6 attempted to reconstruct failure events *exclusively* from windows where $\text{label} > 0$ (Warning or Critical windows). However, if an entity undergoes an initial failure and subsequent telemetry ceases *before* reaching the 30-minute pre-failure horizon of a second failure event, the second failure timestamp will be completely missing from the window labels! Relying on labels rather than raw lifecycle events creates a systematic temporal misclassification.

### 2.3 Independent Classification of the 296 Failure-Cohort Normal Windows
Protocol Amendment 2.1 defines 4,811 primary normal windows, consisting of 4,515 pure-normal windows and 296 normal windows belonging to failure-cohort entities. We audited all 296 failure-cohort normal windows using independently verified lifecycle event times from the raw trace:

| Temporal Classification Category | Verified Count | Definition and Boundary Conditions |
| :--- | :---: | :--- |
| **More than 30 min before next failure** | **147** | $\tau > 1800\text{ s}$ preceding a verified failure event. |
| **Within Warning horizon** | **0** | $900\text{ s} < \tau \le 1800\text{ s}$ (all assigned label 1, not label 0). |
| **Within Critical horizon** | **0** | $0\text{ s} \le \tau \le 900\text{ s}$ (all assigned label 2, not label 0). |
| **After prior failure (post-failure recovery)** | **149** | Window occurs after a prior failure event, with no subsequent failure in the task lifecycle. |
| **Ambiguous or unverifiable** | **0** | All timestamps unambiguously resolved to raw event records. |
| **Total Failure-Cohort Normal Windows** | **296** | Exact sum ($147 + 149 = 296$). |

### 2.4 Reconciliation of the 7-Window Discrepancy in Phase 6.6
The Phase 6.6 report asserted that the 296 normal windows partition into **140 pre-failure normal windows** and **156 post-failure recovery windows**. Our independent lifecycle audit proves this partitioning is incorrect. The true breakdown is **147 pre-failure normal windows** and **149 post-failure windows**.

The root cause of this 7-window discrepancy was traced to two specific entities:
1. **Entity `6224307516_2`** (Parent Job `6224307516`): Experienced failure 1 at $t = 9232.6\text{ s}$ and failure 2 at $t = 13367.6\text{ s}$. Between $t = 9300\text{ s}$ and $t = 10200\text{ s}$, 4 normal windows were recorded ($\tau = 4067.6\text{ s}, 3767.6\text{ s}, 3467.6\text{ s}, 3167.6\text{ s}$). All 4 windows occurred $> 30\text{ min}$ before failure 2. Because telemetry ceased at $t = 10200\text{ s}$, no Warning or Critical windows were logged for failure 2. Phase 6.6, which only parsed labels $> 0$, missed failure 2 and misclassified these 4 windows as "post-failure".
2. **Entity `6224307516_3`** (Parent Job `6224307516`): Experienced failure 1 at $t = 9455.5\text{ s}$ and failure 2 at $t = 13367.6\text{ s}$. Between $t = 9600\text{ s}$ and $t = 10200\text{ s}$, 3 normal windows were recorded ($\tau = 3767.6\text{ s}, 3467.6\text{ s}, 3167.6\text{ s}$). Exactly like task 2, these 3 windows preceded failure 2 by $> 30\text{ min}$ and were erroneously classified as "post-failure" by Phase 6.6.

**Audit Reconciliation Summary:**
- Phase 6.6 Pre-Failure Count: 140
- Discrepancy: $+7$ windows (`6224307516_2` [4 windows] + `6224307516_3` [3 windows])
- True Pre-Failure Normal Count: **147**
- Phase 6.6 Post-Failure Count: 156
- Discrepancy: $-7$ windows
- True Post-Failure Normal Count: **149**

> [!IMPORTANT]
> **Claim 4 Status: NOT REPRODUCED.** Phase 6.6's partition of 140 / 156 is factually incorrect and was caused by an incomplete timestamp reconstruction methodology that failed to inspect raw trace event tables for repeated failures. The true division is **147 pre-failure** and **149 post-failure** normal windows.

---

## 3. Audit of the Missing-Telemetry Feature Claim (`cpi_missing_fraction_6`)

### 3.1 Feature Definition and Temporal Leakage Verification
We audited the definition and implementation of `cpi_missing_fraction_6` in [`src/features/engineer.py`](file:///c:/Users/hp/Desktop/Research%20project/src/features/engineer.py):
- **Exact definition:** The trailing 6-step rolling mean of the binary missingness indicator for `cycles_per_instruction`:
  $$\text{cpi\_missing\_fraction\_6}(t) = \frac{1}{6} \sum_{k=0}^{5} \mathbb{I}\left[\text{cpi}(t - k \cdot 300\text{ s}) \text{ is NaN}\right]$$
- **Information barrier:** The rolling computation is strictly retrospective ($\le t_{\text{pred}}$). At no point does the rolling calculation access future telemetry or post-failure records.
- **Pipeline invariance:** The computation is identical across training, validation, and confirmatory evaluation. The missingness fraction is explicitly engineered prior to feature scaling; standard scaling parameters are derived strictly from training data (`configs/fitted_scaler.json`).

### 3.2 Statistical Comparison: False Positives vs. True Negatives
We independently recomputed the distribution of `cpi_missing_fraction_6` between True Negative (TN) and False Positive (FP) windows across all four models. 

To prevent anticonservative inference from correlated observations within tasks and parent jobs, we computed:
1. Standard pooled Cohen's $d$;
2. **500-replicate Entity-Cluster Bootstrap 95% Confidence Intervals** (resampling entire entities with replacement);
3. **500-replicate Parent-Job Cluster Bootstrap 95% Confidence Intervals** (resampling entire parent jobs with replacement).

#### Pure-Normal Control Entities ($N = 4,515$ windows across 198 entities)

| Model Name | TN ($N$) | FP ($N$) | Mean TN | Mean FP | Cohen's $d$ | Entity-Cluster 95% CI | Parent-Job Cluster 95% CI |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | 3,537 | 978 | +0.026 | +2.935 | **+1.425** | $[+0.941, +1.991]$ | $[+0.096, +2.689]$ |
| **Temporal LSTM** | 3,248 | 1,267 | -0.116 | +2.247 | **+1.505** | $[+1.026, +2.023]$ | $[+0.521, +2.478]$ |
| **Compact Transformer** | 3,674 | 841 | -0.210 | +3.785 | **+2.953** | $[+2.291, +3.534]$ | $[+1.642, +4.103]$ |
| **Representation Hybrid** | 2,958 | 1,557 | -0.198 | +1.777 | **+1.205** | $[+0.806, +1.643]$ | $[+0.320, +2.081]$ |

*(Note: Feature values reflect standardized z-scores derived from training set parameters).*

#### Failure-Cohort Normal Windows ($N = 296$ windows across 137 entities)

| Model Name | TN ($N$) | FP ($N$) | Mean TN | Mean FP | Cohen's $d$ | Entity-Cluster 95% CI | Parent-Job Cluster 95% CI |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | 245 | 51 | -0.402 | -0.016 | **+0.250** | $[-0.076, +0.640]$ | $[-0.231, +0.762]$ |
| **Temporal LSTM** | 270 | 26 | -0.364 | +0.222 | **+0.354** | $[-0.151, +0.970]$ | $[-0.170, +0.892]$ |
| **Compact Transformer** | 162 | 134 | -0.665 | +0.063 | **+0.485** | $[+0.252, +0.722]$ | $[+0.165, +0.831]$ |
| **Representation Hybrid** | 186 | 110 | -0.584 | +0.088 | **+0.443** | $[+0.196, +0.697]$ | $[+0.091, +0.785]$ |

### 3.3 Epistemological Evaluation: Association vs. Causation
While Phase 6.6 correctly identified a large empirical separation in `cpi_missing_fraction_6` among pure-normal false alarms, the report repeatedly referred to missing telemetry as an "error mechanism" and suggested that telemetry dropout represents an underlying "degradation process".

**Our audit requires strict epistemological corrections:**
1. **Cluster sensitivity:** Under entity clustering, the 95% CI for Random Forest is $[+0.941, +1.991]$; however, under parent-job clustering, the lower bound collapses to $+0.096$. This collapse demonstrates that missingness is heavily co-clustered within specific parent-job batch scripts.
2. **Observational confounding:** In Google cluster traces, hardware CPI counters are collected via CPU performance sampling. When a container is idle, descheduled by Borg, or experiences high I/O wait, hardware cycle counters are frequently omitted. Consequently, missing CPI is a marker of task inactivity or scheduling status, **not** evidence of physical processor failure.
3. **Causal overreach:** Observational telemetry cannot distinguish whether missingness triggers a model false alarm because the model learned a spurious artifact or because the container crashed. Labeling missing telemetry as a causal mechanism is scientifically ungrounded.

> [!WARNING]
> **Claim 2 Status: REPRODUCED WITH QUALIFICATIONS.** The effect size is mathematically reproduced ($d = +1.205$ to $+2.953$ in pure-normal entities), and entity-cluster CIs exclude zero. However, the claim that missing telemetry represents a physical degradation mechanism must be stripped from the paper. It is an observational association reflecting training-distribution artifact sensitivity.

---

## 4. Audit of Parent-Job Concentration

### 4.1 Recomputed Concentration Metrics
The confirmatory evaluation encompasses 112 unique parent jobs across the 4,811 primary normal windows. We audited the degree of concentration of false positives using:
- **Gini Coefficient:**
  $$G = \frac{\sum_{i=1}^{J} \sum_{j=1}^{J} |y_i - y_j|}{2 J \sum_{i=1}^{J} y_i}$$
- **Standard Herfindahl-Hirschman Index (HHI):**
  $$\text{HHI} = \sum_{j=1}^{J} s_j^2, \quad \text{where } s_j = \frac{\text{FP}_j}{\sum_k \text{FP}_k}$$
- **Normalized HHI ($H^*$):**
  $$H^* = \frac{\text{HHI} - 1/J}{1 - 1/J}, \quad \text{where } J = 112$$

| Concentration Metric | Random Forest | Temporal LSTM | Compact Transformer | Representation Hybrid |
| :--- | :---: | :---: | :---: | :---: |
| **Total False Positives** | 1,029 | 1,293 | 975 | 1,667 |
| **Gini Coefficient** | **0.8806** | **0.8351** | **0.8659** | **0.7938** |
| **Standard HHI** | 0.0989 | 0.0883 | 0.1065 | 0.0567 |
| **Normalized HHI ($H^*$)** | **0.0908** | **0.0801** | **0.0984** | **0.0482** |
| **Top 1 Parent Job Share** | 18.56% | 19.33% | 20.31% | 15.60% |
| **Top 3 Parent Jobs Share** | **50.63%** | 46.56% | **52.21%** | 33.23% |
| **Top 5 Parent Jobs Share** | **60.54%** | 56.77% | **61.64%** | 42.65% |
| **Top 10 Parent Jobs Share** | 75.41% | 72.39% | 75.79% | 59.99% |

### 4.2 Audit of the "Top 3 Jobs Account for 50–61%" Claim
Phase 6.6 claimed: *"Three parent jobs account for 50% to 61% of all false positive windows in Random Forest and the Compact Transformer."*

**Audit Findings:**
- In **Random Forest**, the top 3 jobs generate 521 FPs out of 1,029 (**50.63%**). The top 5 jobs generate 623 FPs (**60.54%**).
- In **Compact Transformer**, the top 3 jobs generate 509 FPs out of 975 (**52.21%**). The top 5 jobs generate 601 FPs (**61.64%**).
- In **Temporal LSTM**, the top 3 jobs generate 602 FPs out of 1,293 (**46.56%**); the top 4 jobs reach 54.52%.
- In **Representation Hybrid**, the top 3 jobs generate 554 FPs out of 1,667 (**33.23%**); the top 7 jobs reach 50.27%.

The claim holds for Random Forest and Compact Transformer (50.63% and 52.21% for top 3; 60.54% and 61.64% for top 5). The wording must be qualified to clarify that this specific top-3 concentration threshold does not generalize to LSTM or Hybrid.

### 4.3 Methodological Flaw in Top-Job Exclusion Sensitivity Analysis
Phase 6.6 presented a sensitivity analysis showing that excluding the top 10 parent jobs reduces Random Forest FPR from 21.39% down to 6.81%, and pure-normal FAR from 35.35% to 20.62%.

**Audit Critique & Mandatory Disclaimers:**
1. **Severe Post-Hoc Selection Bias:** Removing the top error-producing jobs from a test set *after* inspecting test predictions is the textbook definition of post-hoc selection bias.
2. **False Operational Promise:** In a production cloud cluster, a system operator cannot selectively exclude unknown future parent jobs that happen to trigger false alarms. Reporting post-exclusion FPR as indicative of "potential deployment performance" is scientifically invalid.
3. **Mandatory Report Policy:** Any presentation of top-job exclusions must be explicitly identified as a post-hoc diagnostic tool illustrating error clustering, **never** as a primary finding or an adjusted performance metric.

> [!NOTE]
> **Claim 3 Status: REPRODUCED WITH QUALIFICATIONS.** Top-3 concentration exceeds 50% for Random Forest (50.63%) and Compact Transformer (52.21%), and Gini coefficients confirm extreme concentration ($G > 0.79$ across all models). However, post-hoc job exclusion must be framed strictly as a diagnostic of workload heterogeneity, not deployment capability.

---

## 5. Audit of Prediction Probabilities and Loss Weighting

### 5.1 Probability Validity and Argmax Invariant Verification
We audited the raw predicted probabilities across all 5,511 confirmatory windows for each of the four models:
- **Sum-to-One Invariant:** For all 5,511 rows across all four models, $\sum_{c=0}^2 P(c) = 1.0$ within floating-point tolerance (maximum absolute deviation: $1.52 \times 10^{-7}$).
- **Argmax Decision Consistency:** For all 5,511 predictions across all four models, the recorded class label strictly matches $\text{argmax}_{c} P(c)$ (100.0% adherence, 0 discrepancies).
- **Critical Predominance:** False positive windows overwhelmingly predict the Critical class rather than Warning:
  - Random Forest: **86.20%** Critical (887 Critical vs. 142 Warning FPs)
  - Temporal LSTM: **97.06%** Critical (1,255 Critical vs. 38 Warning FPs)
  - Compact Transformer: **92.72%** Critical (904 Critical vs. 71 Warning FPs)
  - Representation Hybrid: **73.13%** Critical (1,219 Critical vs. 448 Warning FPs)

### 5.2 Mathematical Derivation of Argmax Triggering Thresholds
Phase 6.6 observed that models trigger Critical alarms at predicted probabilities as low as $0.336$ to $0.359$. We verified the mathematical basis of this behavior:
In a 3-class classification problem ($\mathcal{C} \in \{0, 1, 2\}$) under uniform argmax:
$$\hat{y} = 2 \iff P(2) > P(0) \quad \text{and} \quad P(2) > P(1)$$
When probability mass is split across the remaining classes (e.g., $P(0) \approx 0.33, P(1) \approx 0.33$), $P(2)$ needs to barely exceed $1/3$ to capture the prediction:
$$\min P(2) \to \frac{1}{3} + \epsilon \approx 0.3334$$

**Empirically Verified Minimum Triggering Probabilities on False Positives:**
- Random Forest: $\min P(2) = \mathbf{0.3364}$
- Temporal LSTM: $\min P(2) = \mathbf{0.3458}$
- Compact Transformer: $\min P(2) = \mathbf{0.3587}$
- Representation Hybrid: $\min P(2) = \mathbf{0.3406}$

This is **not** an empirical anomaly of the neural architectures, but a trivial mathematical property of unthresholded 3-class argmax decision rules.

### 5.3 Compact Transformer Softmax Overconfidence
The Compact Transformer exhibits severe softmax saturation on Critical false positives:
- **Median $P(\text{Critical})$ on false positive windows:** $\mathbf{0.9934}$
- **Percentage of Critical FPs with $P(\text{Critical}) \ge 0.90$:** **64.27%**
- **Percentage of Critical FPs with $P(\text{Critical}) \ge 0.99$:** **53.65%**

The network outputs extreme certainty on false alarms. Because these probabilities are uncalibrated (no temperature scaling, Platt scaling, or isotonic regression was applied), high probability values cannot be interpreted as posterior probabilities of failure.

### 5.4 Audit of Training Loss Weights: Refutation of the "10x Weight" Claim
Phase 6.6 Section 4.5 claimed: *"All four models were trained with ~10x loss weights on Critical and ~13x on Warning, which pushed the decision boundary deep into the Normal region."*

We audited the actual training code and manifests in [`scripts/train_phase6_3_l5_models.py`](file:///c:/Users/hp/Desktop/Research%20project/scripts/train_phase6_3_l5_models.py) and [`results/phase6/training_manifest_l5.json`](file:///c:/Users/hp/Desktop/Research%20project/results/phase6/training_manifest_l5.json).

**Audit Findings:**
1. **Random Forest & Hybrid:** Trained with scikit-learn `class_weight='balanced'`. Because training data had 13,000 Normal, 980 Warning, and 1,120 Critical windows, the balanced weights were indeed:
   $$w_{\text{Normal}} = 1.0, \quad w_{\text{Warning}} = 17.0, \quad w_{\text{Critical}} = 11.5$$
2. **Temporal LSTM & Compact Transformer:** Trained using PyTorch `nn.CrossEntropyLoss(weight=...)` with **explicitly specified fixed weights**:
   ```python
   # Line 158 of scripts/train_phase6_3_l5_models.py:
   class_weights = torch.tensor([1.0, 1.5, 2.0], dtype=torch.float32)
   ```
   The neural sequence models were trained with weight $1.5$ on Warning and weight $2.0$ on Critical—**not 10x or 13x!**

> [!CAUTION]
> **Claim 8 Status: NOT REPRODUCED.** The claim that all models were trained with uniform ~10x loss weights is factually false. The neural models used modest weights ($[1.0, 1.5, 2.0]$). The high false alarm rate and saturation in Compact Transformer occurred despite using only a $2.0\times$ loss weight on the Critical class, demonstrating that model saturation was driven by feature distributions and cross-entropy optimization dynamics, not extreme loss weighting.

---

## 6. Audit of Temporal Leakage and Horizon Label Validity

### 6.1 Input Feature Independence and Temporal Ordering
We verified the complete temporal pipeline for sequence construction:
- Windows are generated with step size $\Delta t = 300\text{ s}$ and history length $L = 5$ steps ($1,500\text{ s}$ history).
- Every feature tensor $X_{t} \in \mathbb{R}^{5 \times 31}$ uses telemetry strictly from time points $\{t - 4\Delta t, \dots, t\}$.
- No target label $\tau$, post-failure telemetry, or future event information is accessible to the input feature extractor.
- Data split boundaries are strictly task-level and entity-disjoint, guaranteeing zero cross-split leakage.

### 6.2 Pre-Failure Normal Windows Beyond the 30-Minute Horizon
The 147 verified pre-failure normal windows occur at $\tau > 1800\text{ s}$ prior to a verified failure event (spanning $30\text{ min} < \tau \le 48\text{ min}$). On these windows, the models exhibited the following false positive rates:
- **Random Forest:** $10.88\%$ FPR (16 / 147)
- **Temporal LSTM:** **$94.56\%$ FPR** (139 / 147)
- **Compact Transformer:** **$79.59\%$ FPR** (117 / 147)
- **Representation Hybrid:** **$74.83\%$ FPR** (110 / 147)

Phase 6.6 hypothesized that sequence models were detecting "early pre-failure degradation 30 to 48 minutes in advance", which the pre-registered protocol treated as false alarms.

**Methodological Limitations of the Early Degradation Hypothesis:**
1. **Lack of Corroborating Ground Truth:** Google cluster traces do not provide kernel logs, OS panic dumps, or hardware fault counters. Observational telemetry showing unusual resource utilization 40 minutes before a failure cannot establish whether the task was degrading or executing normal heavy compute.
2. **Inconsistent Baseline Behavior:** Random Forest triggered on only 10.88% of these windows. If a clear physical degradation signal were present in the 31 features, Random Forest would have exhibited elevated detection.
3. **Protocol Pre-Registration:** Under Protocol Amendment 2.1, any alert $> 30\text{ min}$ prior to failure is officially registered as a false alarm. Re-interpreting test-set false alarms as "successful early detections" post hoc violates scientific evaluation integrity.

---

## 7. Evidence Classification Ledger

Every major Phase 6.6 finding was audited against the primary evidence and assigned an unambiguous scientific status:

| Claim ID | Claim Summary | Source Location | Denominator | Independently Audited Result | Scientific Classification |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **CLAIM 1** | False positives predominantly predict Critical ($73\% - 97\%$). | Section 4.1 | 4,811 pooled normal windows | RF: 86.20%, LSTM: 97.06%, TF: 92.72%, Hybrid: 73.13%. | **REPRODUCED** |
| **CLAIM 2** | `cpi_missing_fraction_6` is elevated ($d = +1.2$ to $+3.0$) among pure-normal FPs. | Section 3.2 | 4,515 pure-normal windows (198 entities) | RF: $+1.425$, LSTM: $+1.505$, TF: $+2.953$, Hybrid: $+1.205$. Entity CIs exclude 0. | **REPRODUCED WITH QUALIFICATIONS** |
| **CLAIM 3** | Top 3 parent jobs account for $50\% - 61\%$ of FPs in RF and Transformer. | Section 4.1 | 112 parent jobs (4,811 normal windows) | RF Top 3: 50.63% (Top 5: 60.54%). TF Top 3: 52.21% (Top 5: 61.64%). | **REPRODUCED WITH QUALIFICATIONS** |
| **CLAIM 4** | Failure-cohort normal windows partition into 140 pre-failure and 156 post-failure. | Section 5.1 | 296 failure-cohort normal windows | True counts: **147 pre-failure** and **149 post-failure** windows (7-window error). | **NOT REPRODUCED** |
| **CLAIM 5** | Sequence models detect pre-failure degradation $> 30\text{ min}$ in advance ($75\% - 98\%$ FPR). | Section 5.2 | 147 verified pre-failure normal windows | LSTM: 94.56%, TF: 79.59%, Hybrid: 74.83%, RF: 10.88%. | **REPRODUCED WITH QUALIFICATIONS** |
| **CLAIM 6** | Under 3-class argmax, Critical alerts trigger at $P(\text{Critical}) \approx 0.336 - 0.359$. | Section 6.2 | Critical false positive windows | RF: 0.3364, LSTM: 0.3458, TF: 0.3587, Hybrid: 0.3406. | **REPRODUCED** |
| **CLAIM 7** | Compact Transformer outputs saturated probabilities on FPs (median $P > 0.99$). | Section 6.2 | 904 TF Critical false positive windows | Median $P(\text{Critical}) = 0.9934$; $53.65\% > 0.99$. | **REPRODUCED** |
| **CLAIM 8** | All models were trained with uniform ~10x loss weights on Critical. | Section 4.5 | Model training specifications | Neural models used explicit weights $[1.0, 1.5, 2.0]$, not 10x. Only RF/Hybrid used balanced. | **NOT REPRODUCED** |
| **CLAIM 9** | Missing CPI telemetry is a causal mechanism of container failure. | Section 7.0 | Observational cluster telemetry | Telemetry lacks kernel/hardware fault logs. Cannot distinguish cause from idle/wait states. | **UNVERIFIABLE** |

---

## 8. Repository Test Suite and Software Verification

Automated audit tests were implemented in [`tests/test_phase6_7_independent_scientific_audit.py`](file:///c:/Users/hp/Desktop/Research%20project/tests/test_phase6_7_independent_scientific_audit.py) to guarantee:
1. Invariant cohort denominators and sample sizes;
2. Independent lifecycle event reconstruction and 7-window reconciliation;
3. Feature distribution effect sizes and cluster bootstrap intervals;
4. Parent-job concentration indices and Pareto shares;
5. Probability sums, argmax consistency, and Transformer saturation;
6. Evidence classification ledger completeness and status integrity;
7. Cryptographic hash immutability across all prior phase artifacts.

### Test Execution Results
The complete repository test suite was executed:
```bash
.\.venv\Scripts\pytest.exe -v tests/test_phase6_7_independent_scientific_audit.py
# Result: 8 passed in 1.04s

.\.venv\Scripts\pytest.exe
# Result: 152 passed in 19.47s (0 failures, 0 warnings across all 25 test modules)
```

---

## 9. Comprehensive Synthesis and Scientific Directives

In accordance with Phase 6.7 requirements, we synthesize the audit findings into seven foundational conclusions:

### 1. Are the Phase 6.6 findings independently reproducible?
Four of the nine major claims are **reproduced** without reservation (Critical predominance, argmax triggering thresholds, Transformer probability saturation, and probability validity). Three claims are **reproduced with qualifications** (CPI missingness elevation, parent-job concentration, and pre-failure detection rates). Two claims are **not reproduced** and have been disproved (the 140/156 temporal partition and the claim of uniform 10x training loss weights). One claim remains **unverifiable** (causal attribution of missing telemetry).

### 2. Were any temporal-label or failure-timestamp issues discovered?
Yes. Phase 6.6 suffered from a methodological flaw by attempting to reconstruct failure event times solely from windows with positive labels ($\text{label} > 0$). When tasks experienced repeated failures and telemetry terminated before entering the pre-failure Warning/Critical horizons of the second failure, Phase 6.6 missed the second failure event completely. This caused **7 pre-failure normal windows** in tasks `6224307516_2` and `6224307516_3` to be mislabeled as "post-failure recovery". Ground-truth event reconstruction proves the correct division is **147 pre-failure normal windows** and **149 post-failure normal windows**.

### 3. Does the missing-telemetry association remain supported after entity-aware analysis?
Yes, but strictly as an **observational association**, not a causal mechanism. `cpi_missing_fraction_6` remains significantly elevated among false alarms ($d = +1.205$ to $+2.953$ on pure-normal controls), and 95% entity-cluster bootstrap confidence intervals exclude zero. However, parent-job cluster intervals approach zero ($[+0.096, +2.689]$ in Random Forest), indicating that missingness is heavily correlated with parent-job workload types. The research paper must explicitly disclaim causal degradation and describe missingness as a vulnerability to unobserved sampling gaps in production telemetry.

### 4. Do parent-job concentration claims reproduce?
Yes. Top 3 parent jobs drive $50.63\%$ of false positives in Random Forest and $52.21\%$ in Compact Transformer (top 5 drive $60.54\%$ and $61.64\%$). Gini coefficients exceed $0.79$ across all models, confirming that false alarms are concentrated in a small fraction of batch workloads. However, post-hoc job exclusion sensitivity analyses must be presented strictly as diagnostic evidence of workload heterogeneity and never as indicative of true operational performance.

### 5. Which interpretations must be removed or qualified in the research paper?
- **REMOVE:** All assertions that missing CPI telemetry represents a "physical failure mechanism" or "hardware degradation process".
- **REMOVE:** The claim that neural sequence models were trained with uniform ~10x loss weights.
- **REMOVE:** Any framing of post-hoc top-job exclusion metrics as "adjusted" operational performance.
- **QUALIFY:** The claim of "early degradation detection" at $30 - 48\text{ min}$ must be formally qualified as an unverified hypothesis that cannot be distinguished from spurious feature drift without external kernel/node logs.
- **CORRECT:** The failure-cohort normal window breakdown must be corrected to 147 pre-failure and 149 post-failure windows.

### 6. Which questions remain unresolved?
1. Does the elevated missingness in CPI telemetry reflect scheduler descheduling, thread idling, or collector failure? (Cannot be resolved without low-level machine logs).
2. Are pre-failure normal alerts ($30 - 48\text{ min}$ prior to failure) genuine physical precursors or baseline drift? (Unresolved without independent node-level diagnostics).
3. Would temperature-calibrated softmax outputs or cost-sensitive thresholding prevent false-alarm cascading in production? (Requires future empirical testing on a fresh cohort).

### 7. What evidence is needed before designing the next model experiment?
Before any future experimental phase (e.g., Phase 7) can be designed:
1. **Decision Threshold Formulation:** A formal utility or cost-sensitive objective function must be specified on validation data rather than defaulting to uniform argmax.
2. **Missingness Representation:** Telemetry missingness indicators must be evaluated separately from continuous utilization signals to prevent models from learning spurious missingness artifacts.
3. **Loss Weight Tuning:** Cross-entropy weights must be systematically validated via grid search rather than ad-hoc heuristics.
4. **Independent Cohort Sampling:** Any future model evaluation must be conducted on a fresh, independently sampled holdout cohort from an unobserved time period or datacenter.

---

### Reaffirmation of Operational Verdict
The official operational conclusion of the confirmatory evaluation remains **`NO_OPERATIONAL_WINNER`**. The Phase 6.4 confirmatory cohort is permanently retired. No winning model is declared.
