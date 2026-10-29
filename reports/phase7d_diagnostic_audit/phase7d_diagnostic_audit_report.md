# Phase 7D Diagnostic Audit Report: Forensic Analysis of Operational Failure Modes

**Document ID:** `REPORT-PHASE7D-DIAGNOSTIC-AUDIT-V1`  
**Execution Timestamp:** 2026-10-28T14:15:00Z (Local: 2026-10-28 19:45:00 IST)  
**Audit Scope:** Read-Only Forensic Audit of Phase 7D Models, Predictions, Checkpoints, and Protocol Implementation  
**Governing Standard:** Read-Only Diagnostic Protocol (Strict Zero-Retraining Invariant)  
**Primary Finding:** Operational failure across all four architectures confirmed; identified severe loss weighting asymmetry and decision-rule discrepancy as primary mathematical drivers of neural model collapse.

---

## 1. Executive Summary

Following the completion of Phase 7D under Protocol v2.0 (Path A), in which no candidate model achieved the operational gate requirements (`NO_OPERATIONAL_WINNER`), this independent, read-only diagnostic audit was conducted to investigate why the models failed, why certain Phase 7B and Phase 7D metrics appeared identical, and whether implementation defects or protocol mismatches contributed to the outcome.

### Core Audit Conclusions
1. **Strict Non-Negotiable Safeguards Maintained:**
   - **Zero Retraining:** No models were trained, retrained, fine-tuned, or re-evaluated against the test set during this audit.
   - **Holdout Isolation:** The quarantined Phase 6 confirmatory holdout (`data/processed/confirmatory/`) remained 100% unaccessed, unopened, and untouched (filesystem timestamp unchanged at `2026-10-26 15:06:17`).
   - **Data Immutability:** No existing datasets, split files, scalers, reports, or historical artifacts were modified.
2. **Independent Metric Reproduction (100% Verified):**
   - Every metric reported in `results/phase7d/test_evaluation_results.json` was independently recomputed from the saved predictions in `results/phase7d/test_predictions.csv`. All confusion matrices, per-class precisions, recalls, F1 scores, Balanced Accuracies, Episode Recalls, Window FPRs, and Entity FARs match bit-for-bit.
3. **Forensic Resolution of Identical Phase 7B / 7D Metrics:**
   - **Verified Fact:** The model checkpoints for Temporal LSTM and Compact Temporal Transformer in Phase 7D are **completely distinct** from Phase 7B (different training runs, different durations, different checkpoint SHA-256 hashes).
   - **Verified Fact:** Continuous softmax output probabilities and PR-AUC values are **different** between Phase 7B and Phase 7D (e.g., max probability difference of $0.1515$ for LSTM and $0.2353$ for Transformer; PR-AUC Critical is $0.1485$ vs $0.1214$ for LSTM).
   - **Root Cause of Identical Discrete Metrics:**
     - **Temporal LSTM:** Suffered complete **mode collapse to the majority class** (Class 0: Normal). In both Phase 7B and Phase 7D, 100% of the 1,025 test windows had $\text{argmax}(\hat{P}) = 0$. When a model predicts Class 0 for every instance, its discrete metrics are mathematically determined entirely by the test set ground truth distribution, guaranteeing identical confusion matrices despite different underlying neural weights.
     - **Compact Temporal Transformer:** Suffered an **extreme conservative collapse**. In both 7B and 7D, it outputted Class 2 for the exact same 3 consecutive windows of a single failing task (`6238841354_6` at timestamps 7800.0, 8100.0, 8400.0) and Class 0 for the other 1,022 windows. Because all 3 windows happen to be true Critical windows, its discrete confusion matrix (921 Normal correct, 0 Warning correct, 3 Critical correct) is identical.
4. **Primary Structural Defect Discovered — Loss Weighting Asymmetry:**
   - In Protocol v2.0, Random Forest and Representation Hybrid were granted `class_weight='balanced'`, which dynamically penalized minority errors by **$\approx 11.5\times$ (Warning) and $11.8\times$ (Critical)**.
   - In contrast, Temporal LSTM and Compact Temporal Transformer were hardcoded to manual weights of **$[1.0, 1.5, 2.0]$**. Because Normal constitutes **$85.3\%$** of training data, a $1.5\times / 2.0\times$ penalty still left Normal accounting for **$76.9\%$ of the total cross-entropy loss gradient**, heavily penalizing the neural networks for attempting to predict minority classes.
5. **Primary Protocol Discrepancy — Decision Rule Mismatch:**
   - The master engineering configuration (`configs/default_config.yaml`) specified an operational decision rule: $P(\text{WARNING}) + P(\text{CRITICAL}) \ge 0.50$.
   - Protocol v2.0 instead enforced standard 3-class uniform $\text{argmax}$. On true test failure windows, the Transformer actually learned a median combined failure probability of **$57.17\%$**; however, because the failure probability was split between Warning ($28\%$) and Critical ($29\%$), the Normal probability ($43\%$) won under uniform $\text{argmax}$, classifying the failure as Normal.

---

## 2. Independent Reproduction of Phase 7D Test Metrics

All metrics reported in `results/phase7d/test_evaluation_results.json` were independently audited and recomputed from raw window predictions in `results/phase7d/test_predictions.csv` ($N = 1,025$ windows across 11 parent jobs and 59 entities).

### Comprehensive Prediction Distribution & Evaluation Metrics

| Metric Category | Metric | Random Forest ($M^*$) | Temporal LSTM | Compact Transformer | Representation Hybrid |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Prediction Counts** | Class 0 (Normal) | 891 (86.93%) | 1,025 (100.00%) | 1,022 (99.71%) | 784 (76.49%) |
| | Class 1 (Warning) | 82 (8.00%) | 0 (0.00%) | 0 (0.00%) | 55 (5.37%) |
| | Class 2 (Critical) | 52 (5.07%) | 0 (0.00%) | 3 (0.29%) | 186 (18.15%) |
| **Confusion Matrix** | `[True 0, Pred (0, 1, 2)]` | [818, 72, 31] | [921, 0, 0] | [921, 0, 0] | [750, 43, 128] |
| | `[True 1, Pred (0, 1, 2)]` | [37, 4, 8] | [49, 0, 0] | [49, 0, 0] | [19, 4, 26] |
| | `[True 2, Pred (0, 1, 2)]` | [36, 6, 13] | [55, 0, 0] | [52, 0, 3] | [15, 8, 32] |
| **Normal Class (0)** | Precision / Recall / F1 | 0.9181 / 0.8882 / 0.9029 | 0.8985 / 1.0000 / 0.9466 | 0.9012 / 1.0000 / 0.9480 | 0.9566 / 0.8143 / 0.8798 |
| **Warning Class (1)**| Precision / Recall / F1 | 0.0488 / 0.0816 / 0.0611 | 0.0000 / 0.0000 / 0.0000 | 0.0000 / 0.0000 / 0.0000 | 0.0727 / 0.0816 / 0.0769 |
| **Critical Class (2)**| Precision / Recall / F1 | 0.2500 / 0.2364 / 0.2430 | 0.0000 / 0.0000 / 0.0000 | 1.0000 / 0.0545 / 0.1034 | 0.1720 / 0.5818 / 0.2656 |
| **Summary Metrics** | **Macro-F1** | **0.4023** | **0.3155** | **0.3505** | **0.4074** |
| | **Overall Accuracy** | 0.8146 | 0.8985 | 0.9015 | 0.7668 |
| | **Balanced Accuracy** | 0.4021 | 0.3333 | 0.3515 | 0.4926 |
| | **Critical Recall** | 0.2364 (13/55) | 0.0000 (0/55) | 0.0545 (3/55) | 0.5818 (32/55) |
| | **Episode Recall %** | 42.86% (9/21) | 0.00% (0/21) | 4.76% (1/21) | 80.95% (17/21) |
| | **Window FPR %** | 11.18% (103/921)| 0.00% (0/921) | 0.00% (0/921) | 18.57% (171/921) |
| | **Entity FAR %** | 65.79% (25/38) | 0.00% (0/38) | 0.00% (0/38) | 57.89% (22/38) |
| | **Parent-Job FAR %** | 100.00% (3/3) | 0.00% (0/3) | 0.00% (0/3) | 100.00% (3/3) |
| **Reproduction** | **Exact Verification** | **VERIFIED (100%)** | **VERIFIED (100%)** | **VERIFIED (100%)** | **VERIFIED (100%)** |

All metrics match the reported JSON file to 6 decimal places. Output saved in `results/phase7d_diagnostic_audit/independent_metric_reproduction.json`.

---

## 3. Forensic Investigation: Why LSTM and Transformer Metrics Were Identical in 7B and 7D

During the Phase 7D review, a major question arose: *Why were the reported test metrics for Temporal LSTM (Macro-F1 0.3155) and Compact Transformer (Macro-F1 0.3505) identical to those in Phase 7B?*

A line-by-line forensic comparison was executed across checkpoints, predictions, and output probabilities.

### 3.1 Checkpoint Provenance Audit
The checkpoints are completely distinct files with distinct SHA-256 hashes and training logs:

| Architecture | Phase 7B Checkpoint & SHA-256 | Phase 7D Checkpoint & SHA-256 | Identity Check |
| :--- | :--- | :--- | :--- |
| **Temporal LSTM** | `models/checkpoints_phase7b/best_lstm_checkpoint_phase7b.pt`<br>`bdb5083b2f9efac08170966fc61c8d62...` | `models/checkpoints_phase7d/best_lstm_checkpoint_phase7d.pt`<br>`8a29df1e77f3492f9af579c31d387a33...` | **DISTINCT (No Reuse)** |
| **Compact Transformer** | `models/checkpoints_phase7b/best_transformer_checkpoint_phase7b.pt`<br>`54267c7b9daaedf167a9990f11895e6d...` | `models/checkpoints_phase7d/best_transformer_checkpoint_phase7d.pt`<br>`a1218bd2ee756c0ae37852d13663ce0a...` | **DISTINCT (No Reuse)** |
| **Random Forest** | `random_forest_phase7b.joblib`<br>`3dcd7704bab47790c7e360902767d414...` | `random_forest_phase7d.joblib`<br>`7a0dc3f8a3e920d1f37c8b8ef265cc21...` | **DISTINCT (No Reuse)** |
| **Representation Hybrid**| `representation_hybrid_phase7b.joblib`<br>`e3ae8bcbb92a643c0f6adf668f11aa8c...` | `representation_hybrid_phase7d.joblib`<br>`ed53d3d14782b2baf536b07ce4ef3ce2...` | **DISTINCT (No Reuse)** |

### 3.2 Row Alignment and Prediction Difference Audit
The test prediction CSV in Phase 7B was sorted by original trace task indices, whereas Phase 7D was sorted by `parent_job_id` and `entity_id`. When aligned strictly by `(entity_id, prediction_timestamp)`, the forensic audit revealed:

| Model Architecture | Aligned Discrete Agreement | Max Probability Delta | Mean Probability Delta | Class Distribution (7B vs 7D) |
| :--- | :---: | :---: | :---: | :--- |
| **Random Forest** | 94.05% (964/1025) | $0.2082$ | $0.0444$ | 7B: [922, 55, 48] $\to$ 7D: [891, 82, 52] |
| **Representation Hybrid** | 91.90% (942/1025) | $0.4091$ | $0.0566$ | 7B: [796, 51, 178] $\to$ 7D: [784, 55, 186] |
| **Temporal LSTM** | **100.00%** (1025/1025) | $0.1515$ | $0.0343$ | 7B: [1025, 0, 0] $\to$ 7D: [1025, 0, 0] |
| **Compact Transformer** | **100.00%** (1025/1025) | $0.2353$ | $0.0328$ | 7B: [1022, 0, 3] $\to$ 7D: [1022, 0, 3] |

### 3.3 The Mathematical Cause of Identical Metrics
1. **Temporal LSTM:**
   - Both the 7B and 7D LSTM models experienced majority-class mode collapse.
   - For all 1,025 test windows, the maximum softmax probability was Class 0 (Normal).
   - Although the continuous probabilities shifted significantly (mean absolute shift of $3.43\%$ per window, maximum shift of $15.15\%$), $\text{argmax}$ mapped 100% of windows to Class 0 in both models.
   - The continuous PR-AUC was **not identical**: PR-AUC Warning was $0.1230$ in 7B vs **$0.1996$** in 7D; PR-AUC Critical was $0.1214$ in 7B vs **$0.1485$** in 7D.
2. **Compact Temporal Transformer:**
   - The Transformer learned an extremely peaked, conservative decision boundary.
   - In both 7B and 7D, exactly 3 windows out of 1,025 triggered Class 2, and they were the **exact same 3 consecutive windows** of task `6238841354_6` (timestamps 7800.0, 8100.0, 8400.0).
   - In both runs, all other 1,022 windows had $P(\text{Normal}) > \max(P(\text{Warning}), P(\text{Critical}))$.
   - The continuous PR-AUC shifted from $0.2219 \to \mathbf{0.2290}$ (Warning) and $0.2623 \to \mathbf{0.2635}$ (Critical).
   - **Conclusion:** The identical discrete metrics were **not** caused by cached predictions or checkpoint reuse, but by the extreme rigidity of the uniform $\text{argmax}$ decision rule operating on heavily unbalanced loss representations.

---

## 4. Input Pipelines, Checkpoint Integrity, and Label Mapping Verification

The training and inference pipelines were audited for data flow integrity, checkpoint loading correctness, and label mappings:

1. **Input Tensors and Dimensions:**
   - Train input: $(3278, 5, 31)$, float32, zero NaNs.
   - Validation input: $(657, 5, 31)$, float32, zero NaNs.
   - Test input: $(1025, 5, 31)$, float32, zero NaNs.
   - Random Forest tabular inputs correctly extracted $X[:, -1, :]$ ($t_{\text{pred}}$ step).
   - Hybrid classifier correctly concatenated 32-dimensional Transformer sequence latent vectors with 31-dimensional tabular vectors to produce $\mathbb{R}^{63}$.
2. **Label Mappings:**
   - In `y_train.npy`, `y_val.npy`, `y_test.npy`: labels are integers in $\{0, 1, 2\}$.
   - Mapping: `0 -> NORMAL`, `1 -> WARNING`, `2 -> CRITICAL`.
   - In PyTorch CrossEntropyLoss: target tensor is `torch.int64`, loss criterion matches PyTorch class indexing convention $\{0, 1, 2\}$.
   - In Scikit-Learn: `rf.classes_` is confirmed as `array([0, 1, 2])`.
   - **Verdict:** No label inversion, no off-by-one errors, and no dimension mismatch occurred.

---

## 5. Training Class Distributions, Weighting Asymmetry, and Window Overlap

An in-depth empirical audit of Candidate A Train ($N = 3,278$) uncovered the structural reasons for deep model collapse.

### 5.1 Training Class Distribution
- **Class 0 (Normal):** $2,797$ windows ($85.33\%$)
- **Class 1 (Warning):** $244$ windows ($7.44\%$)
- **Class 2 (Critical):** $237$ windows ($7.23\%$)
- Ratio: $\approx 11.8\text{ Normal} : 1.0\text{ Warning} : 1.0\text{ Critical}$.

### 5.2 The Loss Weighting Asymmetry Defect
To counteract severe class imbalance, the standard formulation (as implemented in `src/training/trainer.py#compute_training_class_weights`) computes inverse class frequency:
$$w_c = \frac{N}{K \cdot N_c}$$
For Candidate A Train:
- $w_0 = \frac{3,278}{3 \times 2,797} = 0.3906 \implies \mathbf{1.00}$ (normalized)
- $w_1 = \frac{3,278}{3 \times 244} = 4.4781 \implies \mathbf{11.46}$ (normalized)
- $w_2 = \frac{3,278}{3 \times 237} = 4.6104 \implies \mathbf{11.80}$ (normalized)

**The Discrepancy:**
- **Random Forest & Representation Hybrid** used `class_weight='balanced'`, which applied the true $1.0 : 11.5 : 11.8$ ratio.
- **Temporal LSTM & Compact Transformer** used hardcoded protocol weights:
  $$\text{weights} = [1.0, 1.5, 2.0]$$
- **Mathematical Impact:**
  - Effective Normal contribution to loss: $2,797 \times 1.0 = 2,797$ (**$76.9\%$**)
  - Effective Warning contribution to loss: $244 \times 1.5 = 366$ ($10.1\%$)
  - Effective Critical contribution to loss: $237 \times 2.0 = 474$ ($13.0\%$)
  - Because Normal still generated **$76.9\%$ of the gradient**, any epoch where the neural network attempted to predict Class 1 or Class 2 increased loss dramatically on the 2,797 Normal samples. Early stopping was triggered at epoch 1 (LSTM) and epoch 17 (Transformer), effectively forcing the network to stay in a majority-class regime.

### 5.3 Sliding Window Overlap Autocorrelation
- Windows slide by 1 step ($300\text{ s}$). For sequence length $L=5$, consecutive windows share 4 out of 5 observations (**$80\%$ autocorrelation**).
- A long-running normal container executing for 12 hours yields $\approx 140$ nearly identical Normal training windows.
- Conversely, a failure episode lasts at most 30 minutes, yielding at most 6 pre-failure windows.
- This creates massive sample repetition of background normal patterns during mini-batch gradient descent.

---

## 6. Literature & Specification Conformance Review

A line-by-line audit comparing the repository code against `configs/default_config.yaml` and the original literature specifications was performed.

| Specification Aspect | Literature / Default Config Standard | Protocol v2.0 Implementation | Conformance Status |
| :--- | :--- | :--- | :--- |
| **Observation Grid** | 300-second (5 min) fixed intervals | 300-second fixed intervals | **CONFORMANT** |
| **Prediction Horizon** | 0–15 min (Critical), 15–30 min (Warning) | 0–15 min (Critical), 15–30 min (Warning) | **CONFORMANT** |
| **Target Labeling** | Normal=0, Warning=1, Critical=2, Unknown=-1 | Normal=0, Warning=1, Critical=2, Unknown=-1 | **CONFORMANT** |
| **Feature Set** | 31 causal telemetry features | 31 causal telemetry features | **CONFORMANT** |
| **Sequence Length ($L$)** | $L=15$ (75 min history in default config) | $L=5$ (25 min history in Protocol v2.0) | **INTENTIONAL AMENDMENT**<br>*(Ratified in Phase 6.1 due to Borg task lifetime limitations)* |
| **Decision Rule** | $P(\text{WARNING}) + P(\text{CRITICAL}) \ge 0.50$ | Uniform 3-class $\text{argmax}$ | **MATERIAL MISMATCH**<br>*(Argmax splits failure probability and suppresses alerts)* |
| **Loss Weighting** | Balanced inverse frequency or Focal Loss | RF: Balanced; Neural: $[1.0, 1.5, 2.0]$ | **MATERIAL ASYMMETRY**<br>*(Under-weighted minority classes in neural nets)* |
| **Partitioning** | Grouped by entity or chronological | Grouped by parent job (Candidate A) | **SUPERIOR STANDARD**<br>*(Eliminated severe cross-job leakage)* |

---

## 7. Audit Findings, Confirmed Defects, and Hypotheses

### 7.1 Confirmed Defects and Protocol Mismatches
1. **Defect 1 (Loss Weighting Asymmetry):** PyTorch neural models were restricted to $[1.0, 1.5, 2.0]$, leaving them heavily suppressed by the $85.3\%$ majority class, while Random Forest received true balanced weighting ($[1.0, 11.5, 11.8]$).
2. **Defect 2 (Argmax Alert Suppression):** Using uniform $\text{argmax}$ across three classes instead of a combined failure threshold ($P(\text{Fail}) \ge \theta$) suppressed valid warnings. In the Transformer, median $P(\text{Fail})$ on true failures was **$57.17\%$**, yet $\text{argmax}$ selected Normal because probability was split between Warning ($28\%$) and Critical ($29\%$).
3. **Defect 3 (Sequence Length Truncation Disadvantage):** Reducing sequence length to $L=5$ (25 min) severely attenuated the temporal advantage of recurrent/attention mechanisms over static tabular features $x_{\text{last}}$.

### 7.2 Unresolved Hypotheses
1. **Hypothesis H1 (Inverse-Frequency Rectification):** Training Temporal LSTM and Compact Transformer with true balanced weights ($[1.0, 11.46, 11.80]$) or Focal Loss will eliminate majority-class mode collapse and allow them to compete with Random Forest on Validation Macro-F1.
2. **Hypothesis H2 (Validation-Calibrated Thresholding):** Calibrating an operational alert threshold $\theta$ on validation data using combined failure risk ($P(\text{Warning}) + P(\text{Critical}) \ge \theta$) will allow deep models to reach the $\ge 70\%$ Episode Recall gate without exceeding the $\le 10\%$ Entity FAR gate.
3. **Hypothesis H3 (Workload Domain Shift as Upper Bound):** The primary root cause of high false alarms across all models is cross-job workload domain shift between training parent jobs and unseen validation/test parent jobs, which cannot be solved by architecture alone.

---

## 8. Prioritized Proposal for a Pre-Registered Development Experiment

Before any final model is considered for confirmatory testing, a strictly pre-registered development experiment must be conducted on **Candidate A Train ($N=3,278$) and Validation ($N=657$) only**. Candidate A Test must remain completely blinded.

### Experiment Design (Stage 1 — Weighting Comparison)
Train the four architectures across three controlled weighting regimes on Candidate A Train:
1. **Condition A (Unweighted Baseline):** Standard cross-entropy with uniform weights $[1.0, 1.0, 1.0]$.
2. **Condition B (Protocol v2.0 Ad-hoc Baseline):** Cross-entropy with weights $[1.0, 1.5, 2.0]$.
3. **Condition C (True Balanced Inverse-Frequency):** Cross-entropy with dynamically fitted weights:
   $$w = [1.00, 11.46, 11.80]$$

### Contingent Stage 2 (If Justified by Validation Macro-F1)
If Condition C eliminates mode collapse on Validation, evaluate:
1. **Focal Loss Formulation:**
   $$\text{FL}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t) \quad (\gamma = 2.0, \alpha = \text{balanced})$$
2. **Class-Balanced Mini-Batch Sampling:**
   Using `WeightedRandomSampler` to ensure each mini-batch contains equal representations of Normal, Warning, and Critical sequences.
3. **Validation Threshold Calibration:**
   Tuning $\theta \in [0.20, 0.60]$ on Validation data to optimize operational trade-offs ($F_{\beta}$ or constrained FAR).

---

## 9. Plan for Fresh, Independent Final Evaluation

To guarantee complete scientific integrity and prevent test overfitting:
1. **Test Set Blinding:** Candidate A Test ($N = 1,025$) will remain completely untouched and unblinded during the development experiment.
2. **Pre-Registration Gate:** Only if a model architecture and thresholding scheme successfully passes both Validation Gate 1 (Episode Recall $\ge 70\%$, Critical Recall $\ge 50\%$) and Validation Gate 2 (Entity FAR $\le 10\%$, Window FPR $\le 5\%$) will it be locked as a candidate.
3. **One-Shot Evaluation:** Once frozen, the candidate will be scored on Candidate A Test in a single, irreversible pass.
4. **Holdout Quarantine:** The historical Phase 6 holdout (`data/processed/confirmatory/`) will remain quarantined and unaccessed for all development and diagnostic work.

---

**Lead ML Research Engineer & Independent Pipeline Auditor**  
*Audit Artifacts Cataloged in `results/phase7d_diagnostic_audit/`*  
*Timestamp: 2026-10-28T14:15:00Z*
