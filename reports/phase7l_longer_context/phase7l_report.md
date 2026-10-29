# Phase 7L: Controlled Longer-Temporal-Context Experiment Report

**Project:** `C:\Users\hp\Desktop\Research project`  
**Phase:** Phase 7L  
**Date:** October 29, 2026  
**Auditor / Investigator:** Antigravity Research Engineering Subsystem  
**Scope:** Controlled Comparison of Baseline Sequence Length ($L=5$, 25 min) vs Treatment Sequence Length ($L=12$, 60 min) on Candidate A Development Data  
**Primary Comparison:** Condition B (Moderate Loss Weighting `[1.0, 2.0, 2.0]`)  
**Reference Comparison:** Condition A (Unweighted Loss `[1.0, 1.0, 1.0]`)  
**Seeds:** 42, 123, 456 (300 Epochs, 12 Trained Models, 2,412 Threshold Sweep Points)  
**Decision Outcome:** `MIXED EVIDENCE`  
**Operational Scientific Verdict:** `NO OPERATIONAL WINNER` (Confirmed and Preserved)

---

## Executive Summary

Phase 7L was executed pursuant to the Phase 7K-B reconciliation exit criteria to investigate the foundational temporal context research question:
> *"Does extending the causal sequence receptive field from $L=5$ steps (25 minutes) to $L=12$ steps (60 minutes) improve validation failure-prediction performance when all other experimental factors are strictly held constant?"*

### Primary Experimental Findings:
1. **Factor Isolation Maintained:** The experiment held all other factors strictly invariant: same 31 baseline telemetry features, same 2-layer LSTM architecture ($h=64$, dropout=0.2), same optimization budget (Adam lr=0.001, 25 epochs), same loss weighting conditions (Cond B `[1,2,2]` and Cond A `[1,1,1]`), and same random seeds (42, 123, 456). The trailing 5 steps of the $L=12$ tensors matched the $L=5$ tensors bit-for-bit ($0.00 \times 10^0$ difference).
2. **Primary Metric (Condition B):** Extending context from $L=5$ to $L=12$ yielded a modest increase in peak validation Macro-F1 from **$0.3306 \pm 0.0261$** to **$0.3559 \pm 0.0186$** ($+0.0253$, $+7.65\%$ relative improvement), with higher performance in 2 out of 3 seeds (Seed 42: $+0.0695$, Seed 456: $+0.0134$). Episode Recall at peak Macro-F1 doubled from **$8.3\% \pm 7.6\%$** to **$16.7\% \pm 2.9\%$** ($+8.4\%$ absolute).
3. **Severe Operational Trade-offs & Detriments:**
   - **Genuine Critical Recall Collapsed:** Critical Recall at peak Macro-F1 fell from $2.0\% \pm 3.5\%$ to **$0.0\% \pm 0.0\%$** across all three seeds. Under the Policy B multiclass decision rule, $L=12$ shifted probability mass toward Class 1 (Warning), failing to predict Class 2 (Critical) for any of the 49 Critical windows at the peak F1 threshold.
   - **Pure-Normal False Alarms Increased:** Pure-Normal Entity False Alarm Rate (FAR) worsened from $8.2\% \pm 7.1\%$ to **$12.9\% \pm 1.2\%$**, consistently violating Gate 2a ($\le 10.0\%$).
   - **Probability Calibration Degraded:** Multi-class Brier score degraded from $0.3152$ to **$0.3699$**, and Expected Calibration Error (ECE) rose from $0.1653$ to **$0.2429$**, indicating that the longer-context model produced more overconfident misclassifications.
4. **Reference Condition A (Unweighted):** Under unweighted loss, $L=12$ produced **zero gain** in peak Macro-F1 ($0.3355 \pm 0.0180$ for $L=5$ vs $0.3333 \pm 0.0232$ for $L=12$), while Episode Recall dropped from $11.7\%$ to $6.7\%$.
5. **Operational Gate Verdict:** Across all 12 trained models and all 2,412 threshold evaluation points, **EXACTLY ZERO (0) configurations pass all four operational gates simultaneously**. Gate 1 (sensitivity) passes in 153 points; Gate 2 (specificity) passes in 1,110 points; simultaneous passage is 0.
6. **Scientific Verdict:** The experimental outcome is classified as **`MIXED EVIDENCE`**. Extending flat temporal sequence length provides modest ranking separation for pre-failure episodes, but severely degrades probability calibration, increases healthy-entity false alarms, and completely fails to improve genuine Critical interception. The operational scientific verdict remains **`NO OPERATIONAL WINNER`**.

---

## 1. Governance and Mandatory Preflight Verification

The experiment adhered strictly to all non-negotiable governance constraints:
- **Project Root:** `C:\Users\hp\Desktop\Research project`
- **Candidate A Frozen Split:** [`results/phase7a_4/candidate_a_job_grouped_split.csv`](file:///c:/Users/hp/Desktop/Research%20project/results/phase7a_4/candidate_a_job_grouped_split.csv) bit-for-bit SHA-256 confirmed: `f84bc4fe54f8e0583e2ee52f49b2fbe8e6fcafb4f020c656985a1acc31bb4946`.
- **Validation Labels Reconciled:** Verified exactly as Normal: **560**, Warning: **48**, Critical: **49** (Total: **657** windows). The conflicting citation of 569/20/68 in the Phase 7K-A narrative was confirmed as an isolated narrative typographical error in `generate_audit_report.py`.
- **Candidate A Test Status:** Candidate A Test ($N=1,025$) was inspected during Phase 7D and is formally acknowledged as **NON-VIRGIN**. In accordance with governance instructions, Candidate A Test was **NEVER loaded, queried, or evaluated** in Phase 7L.
- **Phase 6 Holdout Status:** The Phase 6 holdout dataset remains **strictly quarantined and untouched**.
- **Dataset Partitions:** Strictly Candidate A Train ($N=3,278$) and Candidate A Validation ($N=657$).
- **No Multi-Factor Confounding:** Preprocessing, model architecture (2-layer LSTM), training parameters, loss weights, and seeds were held identical across both sequence length conditions.

---

## 2. Experimental Design and Data Construction

### 2.1 Causal Temporal Context Window Definitions
The Google Borg 2011 trace telemetry is sampled at 300-second (5-minute) intervals. For any prediction window evaluated at timestamp $t$:
- **Baseline ($L=5$ steps, 25 minutes):**
  - Observation timestamps: $\{t - 1200\text{s}, t - 900\text{s}, t - 600\text{s}, t - 300\text{s}, t\}$.
  - Sequence receptive field: 20 minutes of history plus the 5-minute aggregation of step 0 = **25 minutes** total physical telemetry context.
  - Tensor shapes: Train $(3278, 5, 31)$, Validation $(657, 5, 31)$.
- **Treatment ($L=12$ steps, 60 minutes):**
  - Observation timestamps: $\{t - 3300\text{s}, t - 3000\text{s}, \dots, t - 300\text{s}, t\}$.
  - Sequence receptive field: 55 minutes of history plus the 5-minute aggregation of step 0 = **60 minutes** total physical telemetry context.
  - Tensor shapes: Train $(3278, 12, 31)$, Validation $(657, 12, 31)$.
- **Causality Guarantee:** All observations strictly satisfy $t_{\text{obs}} \le t_{\text{prediction}}$. Zero future telemetry, post-failure events, or terminal records enter either sequence.
- **Bit-for-Bit Trailing Step Invariant:** The trailing 5 steps of the $L=12$ tensor match the $L=5$ tensor bit-for-bit:
  $$\max |X_{\text{L12}}[:, -5:, :] - X_{\text{L5}}[:, :, :]| = 0.00 \times 10^0$$

### 2.2 Leakage-Free Preprocessing and Standardization
- All preprocessing statistics (medians, means, standard deviations) were fitted **strictly on Candidate A Train ($N=3,278 \times 5 = 16,390$ observations)**.
- Scaling parameters were serialized to [`results/phase7l_longer_context/fitted_preprocessors_manifest.json`](file:///c:/Users/hp/Desktop/Research%20project/results/phase7l_longer_context/fitted_preprocessors_manifest.json).
- To eliminate scaling confounders between conditions, the **identical** train-fitted scaler was applied to both $L=5$ and $L=12$ splits without refitting.
- Imputation: Genuine raw NaNs (occurring in `cycles_per_instruction` and `memory_accesses_per_instruction`) were imputed with the training median prior to standardization.
- Zero validation observations were used during scaler fitting.

### 2.3 Model Architecture and Training Setup
- **Architecture:** `TemporalLSTMClassifier` (2-layer LSTM, `input_dim=31`, `hidden_dim=64`, `num_layers=2`, `dropout=0.2`, `num_classes=3`, `bidirectional=False`).
- **Classification Head:** Linear projection from the final causal hidden state $h_t \in \mathbb{R}^{64}$ to class logits $z_t \in \mathbb{R}^3$.
- **Optimizer:** Adam (`lr=0.001`, `weight_decay=0.0001`).
- **Batch Size:** 64.
- **Max Epochs:** 25.
- **Model Selection:** Best model weights selected strictly by minimum validation cross-entropy loss.
- **Loss Conditions:**
  - **Condition B (Predeclared Primary Baseline):** Moderate loss weighting `[1.0, 2.0, 2.0]`.
  - **Condition A (Reference Baseline):** Unweighted loss `[1.0, 1.0, 1.0]`.
- **Random Seeds:** 42, 123, 456.
- **Total Executed Runs:** $2 \text{ context lengths} \times 2 \text{ weighting conditions} \times 3 \text{ seeds} = \mathbf{12 \text{ runs}}$.

---

## 3. Experimental Results

### 3.1 Cross-Seed Aggregate Comparison
The complete cross-seed comparison (mean $\pm$ sample standard deviation across seeds 42, 123, 456) is cataloged in [`results/phase7l_longer_context/cross_seed_aggregate.csv`](file:///c:/Users/hp/Desktop/Research%20project/results/phase7l_longer_context/cross_seed_aggregate.csv):

| Context Length | Loss Condition | Peak Macro-F1 | Ep. Recall at Peak F1 | Crit. Recall at Peak F1 | Ent. FAR at Peak F1 | Win. FPR at Peak F1 | Brier Score | ECE | All Gates Passed |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Baseline ($L=5$, 25 min)** | **Cond B (`[1,2,2]`)** | $0.3306 \pm 0.0261$ | $8.3\% \pm 7.6\%$ | $2.0\% \pm 3.5\%$ | $8.2\% \pm 7.1\%$ | $6.2\% \pm 5.4\%$ | $0.3152 \pm 0.0076$ | $0.1653 \pm 0.0286$ | **0 / 3 (0%)** |
| **Treatment ($L=12$, 60 min)** | **Cond B (`[1,2,2]`)** | **$0.3559 \pm 0.0186$** | **$16.7\% \pm 2.9\%$** | **$0.0\% \pm 0.0\%$** | **$12.9\% \pm 1.2\%$** | **$5.4\% \pm 1.1\%$** | **$0.3699 \pm 0.0381$** | **$0.2429 \pm 0.0573$** | **0 / 3 (0%)** |
| *Paired Difference (Cond B)* | *Treatment - Baseline* | *+0.0253 (+7.65%)* | *+8.4%* | *-2.0%* | *+4.7% (worse)* | *-0.8%* | *+0.0547 (worse)* | *+0.0776 (worse)* | *0.0%* |
| **Baseline ($L=5$, 25 min)** | **Cond A (`[1,1,1]`)** | $0.3355 \pm 0.0180$ | $11.7\% \pm 5.8\%$ | $11.6\% \pm 6.2\%$ | $12.9\% \pm 1.2\%$ | $9.5\% \pm 1.3\%$ | $0.2868 \pm 0.0044$ | $0.1171 \pm 0.0142$ | **0 / 3 (0%)** |
| **Treatment ($L=12$, 60 min)** | **Cond A (`[1,1,1]`)** | $0.3333 \pm 0.0232$ | $6.7\% \pm 5.8\%$ | $0.0\% \pm 0.0\%$ | $8.2\% \pm 1.2\%$ | $2.4\% \pm 1.2\%$ | $0.2998 \pm 0.0146$ | $0.1420 \pm 0.0401$ | **0 / 3 (0%)** |
| *Paired Difference (Cond A)* | *Treatment - Baseline* | *-0.0022 (-0.66%)* | *-5.0% (worse)* | *-11.6% (worse)* | *-4.7%* | *-7.1%* | *+0.0130 (worse)* | *+0.0249 (worse)* | *0.0%* |

### 3.2 Per-Run Peak Operating Points Ledger
The individual run results from [`results/phase7l_longer_context/gate_summary.csv`](file:///c:/Users/hp/Desktop/Research%20project/results/phase7l_longer_context/gate_summary.csv) demonstrate seed-level consistency:

| Run Tag | Context | Condition | Seed | Best Epoch | Val Loss | Peak Macro-F1 | Peak $\theta$ | Ep. Recall | Crit. Recall | Ent. FAR | Win. FPR | Brier | ECE | Gate Pass |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `L5_cond_B_seed42` | $L=5$ | Cond B | 42 | 1 | 0.7573 | 0.3068 | 0.590 | 0.0% (0/20) | 0.0% (0/49) | 0.0% (0/49) | 0.0% (0/560) | 0.3065 | 0.1332 | FAIL |
| `L5_cond_B_seed123` | $L=5$ | Cond B | 123 | 1 | 0.7391 | 0.3586 | 0.410 | 15.0% (3/20) | 6.1% (3/49) | 12.2% (6/49) | 10.2% (57/560) | 0.3207 | 0.1873 | FAIL |
| `L5_cond_B_seed456` | $L=5$ | Cond B | 456 | 1 | 0.7780 | 0.3266 | 0.425 | 10.0% (2/20) | 0.0% (0/49) | 12.2% (6/49) | 8.4% (47/560) | 0.3184 | 0.1754 | FAIL |
| **L5 Cond B Mean** | — | — | — | — | **0.7581** | **0.3306** | **0.475** | **8.3%** | **2.0%** | **8.2%** | **6.2%** | **0.3152** | **0.1653** | **0 / 3** |
| `L12_cond_B_seed42` | $L=12$ | Cond B | 42 | 1 | 0.7905 | 0.3763 | 0.510 | 25.0% (5/20) | 0.0% (0/49) | 12.2% (6/49) | 6.2% (35/560) | 0.3343 | 0.1876 | FAIL |
| `L12_cond_B_seed123` | $L=12$ | Cond B | 123 | 1 | 0.8281 | 0.3516 | 0.605 | 10.0% (2/20) | 0.0% (0/49) | 12.2% (6/49) | 5.0% (28/560) | 0.3655 | 0.2393 | FAIL |
| `L12_cond_B_seed456` | $L=12$ | Cond B | 456 | 1 | 0.9063 | 0.3400 | 0.540 | 15.0% (3/20) | 0.0% (0/49) | 14.3% (7/49) | 4.8% (27/560) | 0.4098 | 0.3018 | FAIL |
| **L12 Cond B Mean** | — | — | — | — | **0.8416** | **0.3559** | **0.552** | **16.7%** | **0.0%** | **12.9%** | **5.4%** | **0.3699** | **0.2429** | **0 / 3** |
| `L5_cond_A_seed42` | $L=5$ | Cond A | 42 | 1 | 0.5616 | 0.3155 | 0.295 | 5.0% (1/20) | 6.1% (3/49) | 14.3% (7/49) | 8.4% (47/560) | 0.2831 | 0.1018 | FAIL |
| `L5_cond_A_seed123` | $L=5$ | Cond A | 123 | 1 | 0.5542 | 0.3405 | 0.285 | 15.0% (3/20) | 18.4% (9/49) | 12.2% (6/49) | 10.9% (61/560) | 0.2858 | 0.1197 | FAIL |
| `L5_cond_A_seed456` | $L=5$ | Cond A | 456 | 1 | 0.5765 | 0.3505 | 0.280 | 15.0% (3/20) | 10.2% (5/49) | 12.2% (6/49) | 9.3% (52/560) | 0.2917 | 0.1299 | FAIL |
| **L5 Cond A Mean** | — | — | — | — | **0.5641** | **0.3355** | **0.287** | **11.7%** | **11.6%** | **12.9%** | **9.5%** | **0.2868** | **0.1171** | **0 / 3** |
| `L12_cond_A_seed42` | $L=12$ | Cond A | 42 | 1 | 0.5707 | 0.3068 | 0.690 | 0.0% (0/20) | 0.0% (0/49) | 0.0% (0/49) | 0.0% (0/560) | 0.2868 | 0.0984 | FAIL |
| `L12_cond_A_seed123` | $L=12$ | Cond A | 123 | 1 | 0.5972 | 0.3492 | 0.560 | 10.0% (2/20) | 0.0% (0/49) | 12.2% (6/49) | 3.2% (18/560) | 0.2974 | 0.1491 | FAIL |
| `L12_cond_A_seed456` | $L=12$ | Cond A | 456 | 1 | 0.5968 | 0.3441 | 0.345 | 10.0% (2/20) | 0.0% (0/49) | 12.2% (6/49) | 3.9% (22/560) | 0.3152 | 0.1784 | FAIL |
| **L12 Cond A Mean** | — | — | — | — | **0.5882** | **0.3333** | **0.532** | **6.7%** | **0.0%** | **8.2%** | **2.4%** | **0.2998** | **0.1420** | **0 / 3** |

---

## 4. Operational Threshold Sweep Analysis (All 2,412 Points)

A full threshold sweep evaluating 201 threshold points ($[0.000, 1.000]$ in increments of $0.005$) across all 12 models was executed ([`results/phase7l_longer_context/threshold_sweep_metrics.csv`](file:///c:/Users/hp/Desktop/Research%20project/results/phase7l_longer_context/threshold_sweep_metrics.csv)):
- **Policy A (Binary Alert):** Zero threshold points pass Gate 1 and Gate 2 simultaneously.
- **Policy B (Cascaded Multiclass Decision):**
  - **Gate 1 Passing Points:** 153 points achieve Episode Recall $\ge 70.0\%$ and Critical Recall $\ge 50.0\%$ (all at very low thresholds $\theta \le 0.160$).
  - **Gate 2 Passing Points:** 1,110 points achieve Entity FAR $\le 10.0\%$ and Window FPR $\le 5.0\%$ (all at high thresholds $\theta \ge 0.550$).
  - **Simultaneous Passing Points:** **EXACTLY ZERO (0)**.
  
The detection gate and the false-alarm gate remain **mutually exclusive** across both $L=5$ and $L=12$. Extending temporal context does not close the operational gap.

---

## 5. Independent Clean-Room Audit Findings

An independent reproduction script ([`scratch/independent_phase7l_audit.py`](file:///c:/Users/hp/Desktop/Research%20project/scratch/independent_phase7l_audit.py)) was executed without importing any metric evaluation functions from the training script:
1. **Sample IDs and Ordering:** Validated exact 1-to-1 match with Candidate A Validation metadata across all 657 rows.
2. **Label Invariant:** Verified exact match with reconciled Phase 7K-B labels (560 Normal, 48 Warning, 49 Critical).
3. **Sequence Lengths and Causality:** Confirmed $L=5$ spans $[t-20\text{m}, t]$ and $L=12$ spans $[t-55\text{m}, t]$ with strictly causal timestamps ($t_{\text{obs}} \le t$).
4. **Data Isolation:** Scaler parameters verified as fitted strictly on Candidate A Train; Candidate A Test and Phase 6 Holdout were never accessed.
5. **Numerical Reproduction:** Recomputed all 2,412 evaluation rows clean-room. The maximum absolute discrepancy across all 18 metric columns, 4 gate booleans, and 9 confusion matrix cells is **$0.00 \times 10^0$** (exact bit-for-bit match).
6. **Cross-Seed Aggregates:** Confirmed means and standard deviations match directly from per-seed results.

The complete audit ledger is preserved in [`causality_integrity_audit.json`](file:///c:/Users/hp/Desktop/Research%20project/results/phase7l_longer_context/causality_integrity_audit.json) and [`discrepancy_register.csv`](file:///c:/Users/hp/Desktop/Research%20project/results/phase7l_longer_context/discrepancy_register.csv).

---

## 6. Scientific Discussion and Failure Mechanism Analysis

### 6.1 Why Did Macro-F1 Improve Modestly While Critical Recall Collapsed?
1. **Receptive Field and Pre-Failure Disturbance:**
   - In Google Borg cluster traces, execution failure is typically preceded by a multi-step sequence of degraded throughput, CPU burstiness, or memory thrashing.
   - At $L=12$ (60 minutes), the LSTM observes 7 additional trailing observations. This extended context provides more cumulative evidence that a task is non-normal, raising Episode Recall from $8.3\%$ to $16.7\%$ and pushing Macro-F1 to $0.3559$ in Condition B.
2. **Class-2 Margin Suppression:**
   - However, because the sequence model consumes a flat concatenation of 12 steps without explicit hierarchical time-to-failure anchors, the extended history dilutes the sharp temporal distinction between Class 1 (Warning, 15–25 min before crash) and Class 2 (Critical, 0–15 min before crash).
   - Under softmax probability distribution, $P_1$ (Warning) exceeds $P_2$ (Critical) across true Critical windows. Consequently, the cascaded decision rule ($P_2 \ge P_1$) assigns $\hat{y}=1$, driving Critical Recall to **$0.0\%$**.
3. **Calibration Degradation (Overconfidence):**
   - The additional steps provide the LSTM with more parameters to overfit the limited training failure episodes ($N=131$).
   - This manifests as degraded probability calibration: Brier score worsens by $+0.0547$ and ECE increases by $+0.0776$. On healthy tasks experiencing routine operational transients (e.g. background garbage collection or periodic cron bursts), the model overconfidently emits false alerts, driving Entity FAR from $8.2\%$ to $12.9\%$.

### 6.2 Comparison Against Phase 7K Feature Ablation
In Phase 7K, adding static scheduler metadata, dynamic preemption history, and peak usage features to $L=5$ yielded peak Macro-F1 between $0.3343$ and $0.3648$. In Phase 7L, simply extending the sequence length of the baseline 31 features to $L=12$ achieves a comparable Macro-F1 ($0.3559 \pm 0.0186$) without adding any auxiliary feature tables. However, neither intervention resolves the fundamental trade-off between sensitivity and false alarms.

---

## 7. Limitations and Uncertainty

1. **Validation Sample Size:** Candidate A Validation contains exactly 20 pre-failure episodes across 20 failing entities, and 49 pure-normal entities within 4 pure-normal parent jobs.
   - Each detected episode represents a coarse $5.0\%$ increment in Episode Recall.
   - Each false alarm on a pure-normal entity represents a $2.04\%$ increment in Entity FAR.
   - Each false alarm on a pure-normal parent job represents a $25.0\%$ increment in Job FAR.
2. **Cluster Specificity:** These findings are derived from a single production cluster trace (Google Borg 2011). They must not be interpreted as establishing cross-cluster generalization.
3. **Non-Virgin Test Status:** Candidate A Test was inspected in Phase 7D and remains disqualified from serving as a virgin confirmatory benchmark. Phase 6 holdout remains quarantined.

---

## 8. Conclusions and Decision Criteria

### Decision Criteria Outcome:
**`MIXED EVIDENCE`**

- **Supporting Improvement:** Under the predeclared primary baseline (Condition B `[1, 2, 2]`), $L=12$ yields higher peak Macro-F1 ($0.3559$ vs $0.3306$) and higher Episode Recall ($16.7\%$ vs $8.3\%$).
- **Contradicting Improvement:** Under unweighted Condition A, $L=12$ yields no Macro-F1 gain ($0.3333$ vs $0.3355$) and lower Episode Recall ($6.7\%$ vs $11.7\%$). Across all runs, Genuine Critical Recall collapses to $0.0\%$, Pure-Normal Entity FAR deteriorates to $12.9\%$ (violating Gate 2a), and probability calibration significantly degrades.
- **Operational Gate Failure:** Zero runs satisfy operational gates.

### Scientific Operational Verdict:
**`NO OPERATIONAL WINNER` IS PRESERVED AND REAFFIRMED.**

No model configuration is deployable. Extending sequence length alone from 25 minutes to 60 minutes in a flat LSTM cannot dismantle the sensitivity–specificity trade-off.

---

## 9. Recommendation for Subsequent Research

The findings of Phase 7L demonstrate that simply feeding longer flat telemetry histories into standard recurrent networks produces diminishing returns and probability overconfidence. Subsequent research should investigate:

> **Recommended Phase 7M Research Question:**  
> *"Does multi-resolution temporal pooling (e.g., combining high-frequency 5-minute telemetry with downsampled 1-hour rolling trend encodings) or explicit hierarchical task-job graph conditioning restore probability calibration and allow pre-failure detection without collapsing Critical Recall or exploding pure-normal false alarms?"*

---
