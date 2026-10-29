# Phase 7D Retraining and Confirmatory Evaluation Report

**Document ID:** `REPORT-PHASE7D-RETRAINING-EVALUATION-V1`  
**Execution Timestamp:** 2026-10-28T13:36:57Z (Local: 2026-10-28 19:06:57 IST)  
**Governing Standard:** Protocol v2.0 (`results/phase7b_1/phase7b_evaluation_protocol_v2.json`)  
**Execution Path:** Path A (Strict Protocol v2.0 Preservation Without Amendments)  
**Final Confirmatory Status:** `COMPLETED — NO_OPERATIONAL_WINNER`  

---

## 1. Executive Summary

Phase 7D executed the authorized retraining, validation candidate qualification, and one-shot confirmatory test evaluation across four prescribed model architectures using the clean, independently reconstructed Phase 7C Borg trace dataset under strict adherence to Protocol v2.0.

### Key Findings
1. **Candidate Selection on Validation Set ($N=657$):**
   - No architecture satisfied both Validation Gate 1 (Episode Recall $\ge 70\%$, Critical Recall $\ge 50\%$) and Validation Gate 2 (Entity FAR $\le 10\%$, Window FPR $\le 5\%$).
   - Per Protocol v2.0 pre-registered fallback rules, **Random Forest** achieved the highest Validation Macro-F1 ($0.3653$) and was permanently frozen as the validated operational candidate ($M^*$) prior to test scoring.
2. **Confirmatory Test Evaluation on Candidate A Test ($N=1,025$):**
   - $M^*$ (Random Forest) achieved Test Macro-F1 of $0.4023$ and Accuracy of $81.46\%$.
   - **Test Gate 1 Result: FAIL** (Episode Recall = $42.86\%$, Critical Recall = $23.64\%$; thresholds $\ge 70\%$ and $\ge 50\%$).
   - **Test Gate 2 Result: FAIL** (Entity False Alarm Rate = $65.79\%$, Window False Positive Rate = $11.18\%$; thresholds $\le 10\%$ and $\le 5\%$).
   - Consequently, the protocol-mandated confirmatory verdict is **`NO_OPERATIONAL_WINNER`**.
3. **Descriptive Benchmark Evaluations:**
   - **Temporal LSTM:** Completely collapsed to the majority class on both validation and test sets (Test Macro-F1 $0.3155$, $0.0\%$ episode and critical recall, $0.0\%$ false alarms).
   - **Compact Temporal Transformer:** High specificity ($0.0\%$ false alarms), but extremely low sensitivity (Test Macro-F1 $0.3505$, Episode Recall $4.76\%$, Critical Recall $5.45\%$).
   - **Representation Hybrid:** High sensitivity (Test Episode Recall $80.95\%$, Critical Recall $58.18\%$), but severe false alarms (Entity FAR $57.89\%$, Window FPR $18.57\%$, Test Macro-F1 $0.4074$). Because Hybrid underperformed Random Forest on validation Macro-F1 ($0.2880$ vs $0.3653$), it was not selected as $M^*$ and remains descriptive.
4. **Diagnostic Comparison with Historical Phase 7B:**
   - On the test set, Random Forest showed slight increases in Episode Recall ($38.10\% \to 42.86\%$) and Critical Recall ($20.00\% \to 23.64\%$), but suffered higher false alarms (Entity FAR $44.74\% \to 65.79\%$, Window FPR $8.14\% \to 11.18\%$).
   - Both Phase 7B and Phase 7D produced the identical operational conclusion: **`NO_OPERATIONAL_WINNER`**.
   - Fixing the data pipeline proved that legacy preprocessing contamination was not masking a viable operational model. Under strict causal calibration, observed differences cannot be solely attributed to contamination removal, as underlying class imbalance and cross-job domain shift dominate operational failure.
5. **Safeguard Compliance:**
   - The historical Phase 6 confirmatory holdout remained 100% unaccessed and bit-for-bit untouched.
   - All input tensors, scalers, and protocol files matched their pre-registered cryptographic hashes.

---

## 2. Input Verification & Artifact Provenance

Before initiating training, all input artifacts were cryptographically verified against their immutable SHA-256 checksums:

| Artifact Description | Path | SHA-256 Checksum | Verification Status |
| :--- | :--- | :--- | :--- |
| **Candidate A Split CSV** | `results/phase7a_4/candidate_a_job_grouped_split.csv` | `f84bc4fe54f8e0583e2ee52f49b2fbe8e6fcafb4f020c656985a1acc31bb4946` | MATCH / IMMUTABLE |
| **Protocol v2.0 Specification** | `results/phase7b_1/phase7b_evaluation_protocol_v2.json` | `f919cbb6d79ee790b41eb08cd1b023f1ea5f887390eaef39f8bdac94bc0e9dab` | MATCH / IMMUTABLE |
| **Fitted Scaler (Train Only)** | `configs/phase7c/fitted_scaler_phase7c.json` | `1170ef8689ec64594874f502d7b7ff5c3f195e188244d21cc8f975e31a0db5fb` | MATCH / IMMUTABLE |
| **Train Metadata CSV** | `data/processed/phase7c/train_metadata.csv` | `0180ade4098711680c44bbd2ca8c3c5d85834a86042546cc9300b00f1df725ce` | MATCH / VERIFIED |
| **Val Metadata CSV** | `data/processed/phase7c/val_metadata.csv` | `8d22193f11d671dc570d29abf9255d06386880d49c5de83c68d6404a8b41b956` | MATCH / VERIFIED |
| **Test Metadata CSV** | `data/processed/phase7c/test_metadata.csv` | `5bf69371833c4b90558afaf0f445244252bc6a6cc5a368cc2cc027d8a38fb4cf` | MATCH / VERIFIED |
| **Train Sequences Tensor ($X$)** | `data/processed/phase7c/X_train.npy` | `6d3a7aca61f84732da301ede11c276afeafd7cbf863b2ce3b5ba4e71e843702d` | MATCH / VERIFIED |
| **Train Labels Tensor ($y$)** | `data/processed/phase7c/y_train.npy` | `355ca59b1609780181fabe8f06c6ab51cc5c53cfbec91cb822d76b8e373fab1e` | MATCH / VERIFIED |
| **Val Sequences Tensor ($X$)** | `data/processed/phase7c/X_val.npy` | `ce25dc6e5783730db200ef7c3ee85b9023be76b344f2982347ff28fb6bf7479e` | MATCH / VERIFIED |
| **Val Labels Tensor ($y$)** | `data/processed/phase7c/y_val.npy` | `eb968a74ae86dcc574f3341055ff45ca9fcf77ec5a1b4cee5ad8cb4c1e2631b9` | MATCH / VERIFIED |
| **Test Sequences Tensor ($X$)** | `data/processed/phase7c/X_test.npy` | `1199fce77261166a19f8885b9d92f34fadf308af85b05f43dd55b6c67984ea14` | MATCH / VERIFIED |
| **Test Labels Tensor ($y$)** | `data/processed/phase7c/y_test.npy` | `c073fe68540017728fd2ddde25bf7334c3b95cdcc760cd556764ca23a0c307b0` | MATCH / VERIFIED |

### Partition Dimensions
- **Train Split:** $N = 3,278$ windows, shape $(3,278, 5, 31)$, 37 parent jobs, 191 entities.
- **Validation Split:** $N = 657$ windows, shape $(657, 5, 31)$, 12 parent jobs, 69 entities.
- **Test Split:** $N = 1,025$ windows, shape $(1,025, 5, 31)$, 11 parent jobs, 59 entities.
- Zero NaN, infinite, or missing values detected.

---

## 3. Pre-Training Safeguards & Holdout Quarantine Audit

The research repository strictly isolates the historical Phase 6 confirmatory holdout (`data/processed/confirmatory/`).

- **Quarantine Invariant:** The holdout files were not opened, read, loaded, or inspected by any execution script.
- **Filesystem Audit:**
  - `data/processed/confirmatory/confirmatory_metadata.csv`: LastWriteTime = 2026-10-26 15:06:17 (UNTOUCHED)
  - `data/processed/confirmatory/X_confirmatory.npy`: LastWriteTime = 2026-10-26 15:06:17 (UNTOUCHED)
  - `data/processed/confirmatory/y_confirmatory.npy`: LastWriteTime = 2026-10-26 15:06:17 (UNTOUCHED)
- **Zero Leakage Invariant:** Features were normalized strictly using `configs/phase7c/fitted_scaler_phase7c.json`, fitted exclusively on the 3,278 training windows.

---

## 4. Stage 1: Model Training & Checkpoint Registry

All four models were trained strictly on Candidate A Train ($N=3,278$) using random seed 42.

| Architecture | Model Name | Input Geometry | Hyperparameters & Setup | Duration | Checkpoint Path & Checksum |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Model 1** | Random Forest Baseline | $\mathbb{R}^{31}$ (tabular $x_{\text{last}}$) | 100 trees, max depth 15, balanced class weights, seed 42 | 0.31s | `models/checkpoints_phase7d/random_forest_phase7d.joblib`<br>`6f2a89df881fbe3277ba5bb396a849ce...` |
| **Model 2** | Temporal LSTM | $\mathbb{R}^{5 \times 31}$ | 2 layers, hidden 64, dropout 0.2, cross-entropy weights $[1.0, 1.5, 2.0]$, AdamW $10^{-3}$, best epoch 1 | 6.63s | `models/checkpoints_phase7d/best_lstm_checkpoint_phase7d.pt`<br>`d1152a4209cb69b3f9ff7e9420dbad74...` |
| **Model 3** | Compact Temporal Transformer | $\mathbb{R}^{5 \times 31}$ | 2 layers, $d_{\text{model}}=32$, 4 heads, $d_{\text{ff}}=64$, dropout 0.1, weights $[1.0, 1.5, 2.0]$, best epoch 17 | 14.88s | `models/checkpoints_phase7d/best_transformer_checkpoint_phase7d.pt`<br>`c1aa128e073c683884e930191cfa9ba2...` |
| **Model 4** | Representation Hybrid | $\mathbb{R}^{63}$ (32-dim latent + 31-dim tabular) | Frozen Transformer backbone + RF (100 trees, depth 15, balanced weights, seed 42) | 0.27s | `models/checkpoints_phase7d/representation_hybrid_phase7d.joblib`<br>`9233633a69ef115bbce07328e1858c8e...` |

---

## 5. Stage 2: Validation Gate Qualification & Candidate Selection

Models were evaluated on Candidate A Validation Split ($N=657$ windows, 12 parent jobs, 69 entities) to identify qualifying candidates.

### Gate Threshold Criteria:
- **Validation Gate 1:** Episode Recall $\ge 70.0\%$ AND Critical Recall $\ge 50.0\%$.
- **Validation Gate 2:** Pure-Normal Entity FAR $\le 10.0\%$ AND Window FPR $\le 5.0\%$.
- **Secondary Diagnostic Metric:** Pure-Normal Parent-Job FAR (recorded descriptively).

### Validation Evaluation Matrix:

| Model | Macro-F1 | Accuracy | Episode Recall % | Critical Recall | Entity FAR % | Window FPR % | Parent-Job FAR % | Gate 1 | Gate 2 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | **0.3653** | 0.7306 | 50.00% (10/20) | 0.2041 (10/49) | 36.73% (18/49) | 16.79% (94/560) | 75.00% (3/4) | **FAIL** | **FAIL** |
| **Temporal LSTM** | 0.3068 | 0.8524 | 0.00% (0/20) | 0.0000 (0/49) | 0.00% (0/49) | 0.00% (0/560) | 0.00% (0/4) | **FAIL** | **PASS** |
| **Compact Transformer** | 0.3328 | 0.7930 | 15.00% (3/20) | 0.1020 (5/49) | 22.45% (11/49) | 7.86% (44/560) | 50.00% (2/4) | **FAIL** | **FAIL** |
| **Representation Hybrid** | 0.2880 | 0.6088 | 45.00% (9/20) | 0.1633 (8/49) | 48.98% (24/49) | 30.18% (169/560) | 75.00% (3/4) | **FAIL** | **FAIL** |

### Selection Verdict:
- **No model qualified through both Validation Gate 1 and Validation Gate 2.**
- Per Protocol v2.0 pre-registered candidate selection fallback:
  > *"If no model passes both gates simultaneously, select the model with the highest validation Macro-F1 as the frozen candidate $M^*$ for diagnostic test scoring, but final confirmatory status cannot be OPERATIONAL_WINNER unless test gates are met."*
- **Frozen Candidate $M^*$:** **Random Forest** (Validation Macro-F1 = $0.3653$).
- Candidate $M^*$ was permanently locked before unblinding test split predictions.

---

## 6. Stage 3: One-Shot Confirmatory Test Evaluation

Candidate A Test Split ($N=1,025$ windows, 11 parent jobs, 59 entities, 21 failure episodes) was unblinded and scored in a single pass.

### Comprehensive Test Performance Matrix:

| Model Architecture | Role in Protocol | Macro-F1 | Accuracy | Episode Recall % | Critical Recall | Warning Recall | Entity FAR % | Window FPR % | Parent-Job FAR % |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | **Confirmatory Winner Candidate ($M^*$)** | **0.4023** | **0.8146** | **42.86%** (9/21) | **0.2364** (13/55) | **0.0816** (4/49) | **65.79%** (25/38) | **11.18%** (103/921) | **100.00%** (3/3) |
| **Temporal LSTM** | Descriptive Benchmark | 0.3155 | 0.8985 | 0.00% (0/21) | 0.0000 (0/55) | 0.0000 (0/49) | 0.00% (0/38) | 0.00% (0/921) | 0.00% (0/3) |
| **Compact Transformer** | Descriptive Benchmark | 0.3505 | 0.9015 | 4.76% (1/21) | 0.0545 (3/55) | 0.0000 (0/49) | 0.00% (0/38) | 0.00% (0/921) | 0.00% (0/3) |
| **Representation Hybrid** | Descriptive Benchmark | 0.4074 | 0.7668 | 80.95% (17/21) | 0.5818 (32/55) | 0.0816 (4/49) | 57.89% (22/38) | 18.57% (171/921) | 100.00% (3/3) |

### Confirmatory Gate Evaluation of $M^*$ (Random Forest):
- **Test Gate 1 Criterion:** Episode Recall $\ge 70.0\%$ AND Critical Recall $\ge 50.0\%$.
  - Result: Episode Recall = $42.86\%$ (FAIL), Critical Recall = $23.64\%$ (FAIL).
  - **Test Gate 1: FAIL**
- **Test Gate 2 Criterion:** Entity FAR $\le 10.0\%$ AND Window FPR $\le 5.0\%$.
  - Result: Entity FAR = $65.79\%$ (FAIL), Window FPR = $11.18\%$ (FAIL).
  - **Test Gate 2: FAIL**
- **Official Protocol v2.0 Operational Verdict:**
  $$\mathbf{NO\_OPERATIONAL\_WINNER}$$
  *(Candidate Random Forest failed both Test Gate 1 and Test Gate 2)*

### Detailed Class-Level Metrics for $M^*$ (Random Forest):
- **Normal Class (Class 0, Support = 921):** Precision = $0.9181$, Recall = $0.8882$, F1 = $0.9029$
- **Warning Class (Class 1, Support = 49):** Precision = $0.0488$, Recall = $0.0816$, F1 = $0.0611$, PR-AUC = $0.0415$
- **Critical Class (Class 2, Support = 55):** Precision = $0.2500$, Recall = $0.2364$, F1 = $0.2430$, PR-AUC = $0.2272$
- **Episode Detection Dynamics ($N=21$ episodes):**
  - Episodes Detected: $9$ of $21$ ($42.86\%$)
  - Mean Lead Time: $1,181.1\text{ s}$ ($19.68\text{ minutes}$)
  - Median Lead Time: $1,273.9\text{ s}$ ($21.23\text{ minutes}$)
  - Min / Max Lead Time: $526.6\text{ s}$ ($8.78\text{ min}$) / $1,797.5\text{ s}$ ($29.96\text{ min}$)

---

## 7. Uncertainty Estimation & Clustered Bootstrap Analysis

In accordance with Section 8 of Protocol v2.0, dual-cluster bootstrap resampling was executed ($B=1,000$ iterations, seed 42) across both hierarchical statistical clustering dimensions:
1. **Primary Cluster Unit:** Parent Job (`parent_job_id`, $J=11$ clusters in test split).
2. **Complementary Cluster Unit:** Monitored Entity (`entity_id`, $N=59$ clusters in test split).

### Clustered Bootstrap Confidence Intervals for $M^*$ (Random Forest):

| Metric | Point Estimate | Primary Parent-Job Cluster (J=11)<br>Mean [95% CI] | Valid Replicates<br>(Parent-Job) | Complementary Entity Cluster (N=59)<br>Mean [95% CI] | Valid Replicates<br>(Entity) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Macro-F1** | 0.4023 | 0.3874 [0.2972, 0.4885] | 1000/1000 (100.0%) | 0.4009 [0.3405, 0.4622] | 1000/1000 (100.0%) |
| **Accuracy** | 0.8146 | 0.7492 [0.4821, 0.9212] | 1000/1000 (100.0%) | 0.8132 [0.7525, 0.8677] | 1000/1000 (100.0%) |
| **Critical Recall** | 0.2364 | 0.2523 [0.0780, 0.5556] | 1000/1000 (100.0%) | 0.2344 [0.0655, 0.4348] | 1000/1000 (100.0%) |
| **Warning Recall** | 0.0816 | 0.0749 [0.0000, 0.1538] | 1000/1000 (100.0%) | 0.0814 [0.0200, 0.1552] | 1000/1000 (100.0%) |
| **Episode Recall %** | 42.86% | 48.72% [31.25%, 85.71%] | 1000/1000 (100.0%) | 43.07% [22.71%, 65.00%] | 1000/1000 (100.0%) |
| **Window FPR %** | 11.18% | 15.92% [6.08%, 39.36%] | 1000/1000 (100.0%) | 11.25% [6.69%, 16.52%] | 1000/1000 (100.0%) |
| **Entity FAR %** | 65.79% | 77.31% [61.19%, 100.00%] | 999/1000 (99.9%) | 65.71% [50.00%, 80.56%] | 1000/1000 (100.0%) |
| **Parent-Job FAR %** | 100.00% | 100.00% [100.00%, 100.00%] | 969/1000 (96.9%) | N/A (Not defined) | 0/1000 (0.0%) |

### Critical Bootstrap Limitations & Replicate Dynamics:
1. **Severe Imbalance in Cluster Sizes:**
   - In parent-job resampling, the window count per bootstrap replicate varied wildly from **122 to 4,114 windows** (mean: $998.6$, median: $958.0$). This 33-fold variation is an inherent consequence of having only 11 parent jobs in Candidate A test, where a single large parent job comprises hundreds of telemetry windows.
   - In entity-level resampling ($N=59$), the replicate window count remained much more tightly bounded (**891 to 1,158 windows**, mean: $1025.3$), producing narrower, more stable confidence bounds.
2. **Undefined Denominators (Zero-Division Handling):**
   - In $31$ of $1,000$ parent-job bootstrap replicates ($3.1\%$), the resampled clusters contained zero pure-normal parent jobs (all sampled clusters experienced at least one failure), resulting in an undefined denominator. These were recorded as NaN and handled with valid replicate reporting ($96.9\%$ valid).
   - In $1$ of $1,000$ parent-job replicates ($0.1\%$), no pure-normal entity was present ($99.9\%$ valid).
3. **Statistical Robustness of Failure Decision:**
   - For Gate 1 (Critical Recall $\ge 50\%$), the complementary entity bootstrap upper bound ($43.48\%$) falls strictly below the operational threshold.
   - For Gate 2 (Entity FAR $\le 10\%$, Window FPR $\le 5\%$), the lower bounds of the 95% CIs across both clustering dimensions ($50.00\%$ and $6.08\%$) strictly exceed the maximum allowable false alarm thresholds.
   - The operational rejection of Random Forest is statistically decisive across both clustering dimensions.

---

## 8. Comparative Analysis: Phase 7D vs Historical Phase 7B

Phase 7B operated on the legacy dataset contaminated by future-window imputation leakage and global scaling. Phase 7D operated on the independently reconstructed, clean Phase 7C dataset.

### Side-by-Side Performance Comparison (Candidate A Test):

| Architecture | Evaluation Metric | Historical Phase 7B | Clean Phase 7D | Observed Absolute Delta |
| :--- | :--- | :---: | :---: | :---: |
| **Random Forest ($M^*$)** | **Macro-F1** | 0.3965 | **0.4023** | $+0.0058$ |
| | **Accuracy** | 0.8390 | **0.8146** | $-0.0244$ |
| | **Episode Recall %** | 38.10% (8/21) | **42.86%** (9/21) | $+4.76\%$ ($+1$ episode) |
| | **Critical Recall** | 0.2000 (11/55) | **0.2364** (13/55) | $+0.0364$ ($+2$ windows) |
| | **Warning Recall** | 0.0612 (3/49) | **0.0816** (4/49) | $+0.0204$ ($+1$ window) |
| | **Entity FAR %** | 44.74% (17/38) | **65.79%** (25/38) | $+21.05\%$ ($+8$ entities) |
| | **Window FPR %** | 8.14% (75/921) | **11.18%** (103/921) | $+3.04\%$ ($+28$ windows) |
| | **Parent-Job FAR %** | 100.00% (3/3) | **100.00%** (3/3) | $0.00\%$ |
| **Temporal LSTM** | Macro-F1 | 0.3155 | 0.3155 | $0.0000$ |
| | Episode Recall % | 0.00% | 0.00% | $0.00\%$ |
| | Critical Recall | 0.0000 | 0.0000 | $0.0000$ |
| | Entity FAR % | 0.00% | 0.00% | $0.00\%$ |
| **Compact Transformer** | Macro-F1 | 0.3505 | 0.3505 | $0.0000$ |
| | Episode Recall % | 4.76% (1/21) | 4.76% (1/21) | $0.00\%$ |
| | Critical Recall | 0.0545 (3/55) | 0.0545 (3/55) | $0.0000$ |
| | Entity FAR % | 0.00% | 0.00% | $0.00\%$ |
| **Representation Hybrid**| Macro-F1 | 0.4206 | 0.4074 | $-0.0132$ |
| | Episode Recall % | 80.95% (17/21) | 80.95% (17/21) | $0.00\%$ |
| | Critical Recall | 0.6182 (34/55) | 0.5818 (32/55) | $-0.0364$ ($-2$ windows) |
| | Entity FAR % | 55.26% (21/38) | 57.89% (22/38) | $+2.63\%$ ($+1$ entity) |
| | Window FPR % | 17.05% (157/921)| 18.57% (171/921)| $+1.52\%$ ($+14$ windows) |

### Evidence-Calibrated Interpretation (Causation vs. Correlation):
1. **Operational Equivalence:**
   - The primary research objective of Phase 7D was to determine whether correcting pipeline contamination would uncover a viable operational winner.
   - The result is an unambiguous **NO**: both contaminated Phase 7B and clean Phase 7D selected Random Forest as $M^*$ on validation, and both resulted in `NO_OPERATIONAL_WINNER` on test.
2. **Random Forest Performance Shift:**
   - In Phase 7D, Random Forest detected 1 additional failure episode ($9/21$ vs $8/21$) and 2 additional critical windows ($13/55$ vs $11/55$), raising Macro-F1 slightly ($0.3965 \to 0.4023$).
   - However, this slight gain in recall was accompanied by a noticeable increase in false alarms: Entity FAR increased from $44.74\%$ to $65.79\%$, and Window FPR increased from $8.14\%$ to $11.18\%$.
3. **Calibrated Attribution Warning:**
   - **Do not claim the experiment proves contamination was the sole cause of these differences.**
   - The observed variations between Phase 7B and Phase 7D reflect multiple compounding factors:
     a) Removal of backward-looking multi-failure event sequence contamination.
     b) Replacement of global min-max scaling with training-only quantile clipping and robust scaling.
     c) The shift in empirical feature distributions caused by exact raw trace re-aggregation.
   - The decisive scientific conclusion is that **pipeline contamination was not the reason models failed operational thresholds**. Severe class imbalance (only 55 critical windows out of 1,025 in test) and cross-job behavioral variance are the true root causes of operational failure in cloud workload failure prediction.

---

## 9. Scientific Limitations & Key Insights

1. **Failure of Temporal Deep Learning Models:**
   - Both the Temporal LSTM and Compact Temporal Transformer failed to learn balanced multi-class failure representations under standard cross-entropy loss, despite class weighting.
   - The LSTM collapsed completely into predicting class 0 (Normal).
   - The Transformer predicted critical failure on only 3 windows. While highly precise ($100\%$ precision), it is operationally useless ($4.76\%$ episode recall).
   - This underscores a critical domain reality: without advanced ranking objectives, contrastive pretraining, or focal loss formulated specifically for extreme temporal imbalance, standard recurrent and attention-based architectures fail on industrial trace telemetry.
2. **Trade-Off Profile of the Representation Hybrid:**
   - The Representation Hybrid demonstrated the highest raw failure sensitivity across both phases ($80.95\%$ episode recall, $58.18\%$ critical recall).
   - However, its false alarm rate ($57.89\%$ Entity FAR, $18.57\%$ Window FPR) renders it impractical for autonomous operations without human-in-the-loop filtering.
3. **Test Cluster Count Limitation ($J=11$):**
   - The test split comprises only 11 parent-job clusters. This creates significant cluster-level sampling variance during parent-job bootstrap resampling.
   - Future experimental protocols must prioritize larger multi-month trace collections to ensure at least 30–50 independent parent-job clusters in holdout partitions.

---

## 10. Final Verification Sign-Off & Status Declaration

### Protocol Requirement Checklist:
- [x] Frozen inputs verified with SHA-256 before training.
- [x] Zero access to quarantined Phase 6 confirmatory holdout.
- [x] Historical artifacts preserved without modification.
- [x] Four prescribed architectures trained strictly on Candidate A Train ($N=3,278$).
- [x] Validation candidate $M^*$ frozen before test evaluation.
- [x] One-shot confirmatory test evaluation applied strictly to $M^*$.
- [x] Non-selected models reported strictly as descriptive scientific benchmarks.
- [x] Dual-cluster bootstrap intervals computed ($B=1,000$) with valid fraction accounting.
- [x] Output artifacts isolated in `results/phase7d/` and `models/checkpoints_phase7d/`.
- [x] Calibrated scientific attribution documented.

### Official Status Declaration:
$$\mathbf{COMPLETED\ —\ NO\_OPERATIONAL\_WINNER}$$

**Lead ML Research Engineer & Pipeline Auditor**  
*Signature Verified via Automated Checksum Registry*  
*Timestamp: 2026-10-28T13:36:57Z*
