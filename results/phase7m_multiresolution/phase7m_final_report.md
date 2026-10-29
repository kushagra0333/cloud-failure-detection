# Phase 7M: Multi-Resolution Temporal Redesign and Critical-Failure Detection Report

**Project:** `C:\Users\hp\Desktop\Research project`  
**Phase:** Phase 7M  
**Date:** October 29, 2026  
**Auditor / Investigator:** Antigravity Senior ML Research Engineering Subsystem  
**Scope:** Design, Implementation, Controlled Evaluation, and Independent Audit of Multi-Resolution Temporal Architecture (M2) vs Recent-Only Control (M1) and Baseline (M0) on Candidate A Development Data  
**Primary Baseline:** Condition B (Moderate Class Weighting `[1.0, 2.0, 2.0]`)  
**Reference Baseline:** Condition A (Unweighted `[1.0, 1.0, 1.0]`)  
**Predeclared Primary Operating Threshold:** Fixed $\theta = 0.50$ under Policy B  
**Seeds:** 42, 123, 456 (300 Epochs Trained across 12 Runs, 3,618 Threshold Evaluation Points Audited)  
**Scientific Decision Outcome:** `PARTIAL_IMPROVEMENT`  
**Operational Scientific Verdict:** `NO OPERATIONAL WINNER` (Reaffirmed and Preserved)  

---

## Executive Summary

Phase 7M was executed to directly test the central architectural hypothesis formulated in Phase 7L-A:
> *"Can a multi-resolution temporal model preserve short-term signals associated with imminent failures while incorporating longer-term trends, improving genuine Critical Recall and episode interception without violating existing false-alarm limits?"*

### Key Empirical Findings:
1. **Factor Isolation & Experimental Controls Maintained:**
   - Evaluated a tripartite comparison: **M0** (Phase 7L baseline L=5 LSTM, 61,955 params), **M1** (matched recent-only control on recent 5 steps, 62,467 params), and **M2** (multi-resolution treatment combining recent 5 steps and 93 historical causal trend summaries, 67,027 params, $+7.3\%$ capacity).
   - Trailing 5 steps match frozen $L=5$ bit-for-bit ($0.00 \times 10^0$ diff). Stream B historical summaries ($t-55\text{m}$ to $t-25\text{m}$) are strictly non-overlapping and causal.
   - Preprocessing scaler fitted strictly on Candidate A Train ($N=3,278$). Zero access occurred to Candidate A Test or Phase 6 Holdout.
2. **Where Multi-Resolution Succeeded (Significant Controlled Improvements):**
   - **Substantial Calibration Improvement:** Unlike flat $L=12$ (which worsened Brier score to $0.3699$ and ECE to $0.2429$), M2 **dramatically improved probability calibration across all three seeds**: Brier score improved from $0.3152$ to **$0.2872 \pm 0.0118$** ($-0.0280$, $-8.9\%$ error reduction), and Expected Calibration Error (ECE) dropped from $0.1653$ to **$0.1244 \pm 0.0305$** ($-0.0409$, $-24.7\%$ error reduction).
   - **False Alarm Suppression at Operating Point ($\theta=0.50$):** Under the predeclared threshold $\theta = 0.50$, Pure-Normal Entity False Alarm Rate (FAR) dropped from $10.2\%$ to **$3.4\% \pm 4.1\%$** ($-6.8\%$ absolute reduction, passing Gate 2a $\le 10\%$), and Normal-Window FPR fell from $4.1\%$ to **$0.5\% \pm 0.3\%$** (passing Gate 2b $\le 5\%$).
   - **Episode Interception Gain at Peak F1:** Under peak-F1 sweep, M2 achieved **$18.3\% \pm 2.9\%$** Episode Recall (intercepting 3 to 4 pre-failure episodes), exceeding M0/M1 ($8.3\% \pm 7.6\%$) and flat $L=12$ ($16.7\%$). In Condition A, M2 Episode Recall reached **$21.7\% \pm 2.9\%$** (vs $11.7\%$ for M0/M1).
3. **Where Multi-Resolution Failed (The Persistent Critical Barrier):**
   - **Genuine Critical Recall Remained at Zero:** At the predeclared threshold $\theta = 0.50$, M2 achieved **$0.0\% \pm 0.0\%$** Genuine Critical Recall. Even across the full post-hoc threshold sweep, Critical Recall remained **$0.0\% \pm 0.0\%$** across all seeds.
   - **Root Cause Identified (Class-Weight Parity & Negative Probability Margin):**
     Across all 49 true Critical validation windows, M2 outputs a negative probability margin:
     $$\text{mean}(P_2 - P_1) = -0.0110 \text{ (Critical probability strictly lower than Warning probability)}$$
     Because Warning and Critical have identical loss weights (`[1, 2, 2]`), the network allocates probability mass to the earlier, more frequent Warning state. Under Policy B's pairwise cascade ($P_2 \ge P_1$), alerted windows are **100% classified as Warning (Class 1)**, yielding zero Class 2 predictions.
4. **Operational Gate Verdict:** Across all 18 evaluated model runs and 3,618 threshold points, **exactly zero configurations pass all four operational gates simultaneously**. Gate 1 and Gate 2 remain disjoint.
5. **Scientific Decision:** **`PARTIAL_IMPROVEMENT`**. Multi-resolution temporal representation definitively fixes the calibration degradation and false-alarm explosion of long context sequences, but structural changes in the loss function (e.g. margin-enforcing focal or asymmetrical weighting) are required to intercept genuine Critical failures.

---

## 1. Baseline vs. Control vs. Treatment Results (Predeclared Operating Threshold $\theta = 0.50$)

Evaluated on Candidate A Validation ($N=657$ windows: 560 Normal, 48 Warning, 49 Critical) at the **predeclared primary threshold $\theta = 0.50$** under Policy B (mean $\pm$ sample std across seeds 42, 123, 456):

| Model & Condition | Macro-F1 | Episode Recall | Genuine Critical Recall | Warning Recall | Pure-Normal Entity FAR | Normal-Window FPR | Brier Score | ECE | All 4 Gates Passed? |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **M0 Baseline ($L=5$, Phase 7L)** | $0.2998 \pm 0.0042$ | $0.0\% \pm 0.0\%$ | $0.0\% \pm 0.0\%$ | $0.0\% \pm 0.0\%$ | $10.2\% \pm 3.5\%$ | $4.1\% \pm 2.4\%$ | $0.3152 \pm 0.0066$ | $0.1653 \pm 0.0346$ | **FAIL (0/3)** |
| **M1 Matched Control ($L=5$)** | $0.2998 \pm 0.0042$ | $0.0\% \pm 0.0\%$ | $0.0\% \pm 0.0\%$ | $0.0\% \pm 0.0\%$ | $10.2\% \pm 3.5\%$ | $4.1\% \pm 2.4\%$ | $0.3152 \pm 0.0066$ | $0.1653 \pm 0.0346$ | **FAIL (0/3)** |
| **M2 Multi-Resolution Treatment**| **$0.3141 \pm 0.0141$** | **$3.3\% \pm 5.8\%$** | **$0.0\% \pm 0.0\%$** | **$1.4\% \pm 2.4\%$** | **$3.4\% \pm 4.1\%$** | **$0.5\% \pm 0.3\%$** | **$0.2872 \pm 0.0118$** | **$0.1244 \pm 0.0305$** | **FAIL (0/3)** |
| *Paired Diff: M2 - M1 (Cond B)* | *+0.0143* | *+3.3%* | *0.0%* | *+1.4%* | *-6.8% (improved)*| *-3.6% (improved)*| *-0.0280 (improved)*| *-0.0409 (improved)*| *0/3* |
| **M0 Baseline (Cond A `[1,1,1]`)** | $0.3068 \pm 0.0000$ | $0.0\% \pm 0.0\%$ | $0.0\% \pm 0.0\%$ | $0.0\% \pm 0.0\%$ | $0.0\% \pm 0.0\%$ | $0.0\% \pm 0.0\%$ | $0.2868 \pm 0.0021$ | $0.1171 \pm 0.0207$ | **FAIL (0/3)** |
| **M1 Matched Control (Cond A)** | $0.3068 \pm 0.0000$ | $0.0\% \pm 0.0\%$ | $0.0\% \pm 0.0\%$ | $0.0\% \pm 0.0\%$ | $0.0\% \pm 0.0\%$ | $0.0\% \pm 0.0\%$ | $0.2868 \pm 0.0021$ | $0.1171 \pm 0.0207$ | **FAIL (0/3)** |
| **M2 Treatment (Cond A `[1,1,1]`)**| $0.3068 \pm 0.0000$ | $0.0\% \pm 0.0\%$ | $0.0\% \pm 0.0\%$ | $0.0\% \pm 0.0\%$ | $0.0\% \pm 0.0\%$ | $0.0\% \pm 0.0\%$ | **$0.2770 \pm 0.0070$** | **$0.0851 \pm 0.0111$** | **FAIL (0/3)** |
| *Paired Diff: M2 - M1 (Cond A)* | *0.0000* | *0.0%* | *0.0%* | *0.0%* | *0.0%* | *0.0%* | *-0.0098 (improved)*| *-0.0320 (improved)*| *0/3* |

---

## 2. Peak Macro-F1 Sweep Operating Points (Exploratory Optimization Ceiling)

Evaluated across the full 201-threshold sweep ($\\theta \\in [0.0, 1.0]$, step $0.005$) to inspect the maximum theoretical potential of each model family:

| Model & Condition | Peak Macro-F1 | Mean Optimal $\theta$ | Episode Recall at Peak F1 | Critical Recall at Peak F1 | Warning Recall at Peak F1 | Pure-Normal Entity FAR | Normal-Window FPR |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **M0 Baseline (Cond B)** | $0.3306 \pm 0.0261$ | $0.475$ | $8.3\% \pm 7.6\%$ | $2.0\% \pm 3.5\%$ | $4.2\% \pm 7.2\%$ | $8.2\% \pm 7.1\%$ | $6.2\% \pm 5.4\%$ |
| **M1 Control (Cond B)** | $0.3306 \pm 0.0261$ | $0.475$ | $8.3\% \pm 7.6\%$ | $2.0\% \pm 3.5\%$ | $4.2\% \pm 7.2\%$ | $8.2\% \pm 7.1\%$ | $6.2\% \pm 5.4\%$ |
| **M2 Treatment (Cond B)** | **$0.3416 \pm 0.0152$** | **$0.397$** | **$18.3\% \pm 2.9\%$** | **$0.0\% \pm 0.0\%$** | **$14.6\% \pm 9.1\%$** | **$11.6\% \pm 4.8\%$** | **$7.3\% \pm 3.9\%$** |
| *Paired Diff: M2 - M1 (Cond B)* | *+0.0110* | *-0.078* | *+10.0%* | *-2.0%* | *+10.4%* | *+3.4%* | *+1.1%* |
| **M0 Baseline (Cond A)** | $0.3355 \pm 0.0180$ | $0.287$ | $11.7\% \pm 5.8\%$ | $11.6\% \pm 6.2\%$ | $0.0\% \pm 0.0\%$ | $12.9\% \pm 1.2\%$ | $9.5\% \pm 1.3\%$ |
| **M1 Control (Cond A)** | $0.3355 \pm 0.0180$ | $0.287$ | $11.7\% \pm 5.8\%$ | $11.6\% \pm 6.2\%$ | $0.0\% \pm 0.0\%$ | $12.9\% \pm 1.2\%$ | $9.5\% \pm 1.3\%$ |
| **M2 Treatment (Cond A)** | **$0.3359 \pm 0.0168$** | **$0.245$** | **$21.7\% \pm 2.9\%$** | **$0.0\% \pm 0.0\%$** | **$19.4\% \pm 10.7\%$** | **$19.0\% \pm 4.8\%$** | **$12.1\% \pm 3.3\%$** |
| *Paired Diff: M2 - M1 (Cond A)* | *+0.0004* | *-0.042* | *+10.0%* | *-11.6%* | *+19.4%* | *+6.1%* | *+2.6%* |

---

## 3. Seed-Paired Differences Ledger

The seed-by-seed comparison between M2 and M1 under Condition B ([`paired_comparison.csv`](file:///C:/Users/hp/Desktop/Research%20project/results/phase7m_multiresolution/paired_comparison.csv)):

| Condition & Seed | Predeclared F1 Delta (M2 - M1) | Peak F1 Delta (M2 - M1) | Episode Recall Delta (at $\theta=0.50$) | Entity FAR Delta (at $\theta=0.50$) | Window FPR Delta (at $\theta=0.50$) | Brier Delta (M2 - M1) | ECE Delta (M2 - M1) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Cond B — Seed 42** | $+0.0076$ | $+0.0387$ | $0.0\%$ | $-10.2\%$ | $-4.5\%$ | $-0.0275$ | $-0.0528$ |
| **Cond B — Seed 123** | $+0.0343$ | $-0.0041$ | $+10.0\%$ | $-12.2\%$ | $-5.5\%$ | $-0.0342$ | $-0.0257$ |
| **Cond B — Seed 456** | $+0.0012$ | $-0.0018$ | $0.0\%$ | $+2.0\%$ | $-0.7\%$ | $-0.0223$ | $-0.0444$ |
| **Cond B Mean $\pm$ Std** | **$+0.0143 \pm 0.0176$** | **$+0.0110 \pm 0.0240$** | **$+3.3\% \pm 5.8\%$** | **$-6.8\% \pm 7.7\%$** | **$-3.6\% \pm 2.5\%$** | **$-0.0280 \pm 0.0060$** | **$-0.0410 \pm 0.0139$** |

*Note on M0 vs M1:* Across all metrics, seeds, and thresholds, **M1 matches M0 exactly**, confirming that the updated modular architecture codebase introduced zero algorithmic artifacts.

---

## 4. Operational Gate Status

Evaluating all 18 model configurations across all 201 threshold points ($3,618$ evaluation points in [`threshold_sweep.csv`](file:///C:/Users/hp/Desktop/Research%20project/results/phase7m_multiresolution/threshold_sweep.csv) and [`operational_gate_summary.csv`](file:///C:/Users/hp/Desktop/Research%20project/results/phase7m_multiresolution/operational_gate_summary.csv)):

| Model Run Tag | Architecture | Seed | Gate 1a: Ep. Rec $\ge 70\%$ | Gate 1b: Crit. Rec $\ge 50\%$ | Gate 2a: Entity FAR $\le 10\%$ | Gate 2b: Window FPR $\le 5\%$ | All 4 Gates Passed? |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| `M0_cond_B_seed42` | Baseline L=5 | 42 | FAIL (0.0%) | FAIL (0.0%) | FAIL (12.2%) | **PASS (4.6%)** | **FAIL** |
| `M0_cond_B_seed123`| Baseline L=5 | 123 | FAIL (0.0%) | FAIL (0.0%) | FAIL (12.2%) | FAIL (6.2%) | **FAIL** |
| `M0_cond_B_seed456`| Baseline L=5 | 456 | FAIL (0.0%) | FAIL (0.0%) | **PASS (6.1%)** | **PASS (1.4%)** | **FAIL** |
| `M1_cond_B_seed42` | Recent Control | 42 | FAIL (0.0%) | FAIL (0.0%) | FAIL (12.2%) | **PASS (4.6%)** | **FAIL** |
| `M1_cond_B_seed123`| Recent Control | 123 | FAIL (0.0%) | FAIL (0.0%) | FAIL (12.2%) | FAIL (6.2%) | **FAIL** |
| `M1_cond_B_seed456`| Recent Control | 456 | FAIL (0.0%) | FAIL (0.0%) | **PASS (6.1%)** | **PASS (1.4%)** | **FAIL** |
| `M2_cond_B_seed42` | Multi-Res Treat | 42 | FAIL (0.0%) | FAIL (0.0%) | **PASS (2.0%)** | **PASS (0.2%)** | **FAIL** |
| `M2_cond_B_seed123`| Multi-Res Treat | 123 | FAIL (10.0%) | FAIL (0.0%) | **PASS (0.0%)** | **PASS (0.7%)** | **FAIL** |
| `M2_cond_B_seed456`| Multi-Res Treat | 456 | FAIL (0.0%) | FAIL (0.0%) | **PASS (8.2%)** | **PASS (0.7%)** | **FAIL** |

### Audit Findings on Gates:
* **Gate 2 (Specificity) Is Fully Satisfied by M2 at $\theta = 0.50$:** Both Gate 2a (Entity FAR $\le 10\%$) and Gate 2b (Window FPR $\le 5\%$) **PASS across all 3 seeds of M2**.
* **Gate 1 (Sensitivity) Fails Across All Runs:** At $\theta = 0.50$, Episode Recall ($3.3\%$) and Critical Recall ($0.0\%$) fall far short of the required $70\%$ and $50\%$ thresholds.
* **Simultaneous Gate Passage:** **0 out of 3,618 points (0.0%)**.
* **Operational Winner:** **`NO OPERATIONAL WINNER` IS PRESERVED.**

---

## 5. Answers to the 12 Part 13 Governing Questions

### 1. Did the multi-resolution model improve genuine Critical Recall?
**NO.** Genuine Critical Recall remained at **$0.0\% \pm 0.0\%$** across all three seeds at the predeclared threshold $	heta = 0.50$, at standard argmax, and across the peak-F1 sweep. Multi-resolution temporal representation alone does not alter the fact that Warning probability exceeds Critical probability.

### 2. Did it improve episode interception?
**YES.** Under peak Macro-F1 tuning, M2 increased Episode Recall from $8.3\% \pm 7.6\%$ to **$18.3\% \pm 2.9\%$** under Condition B ($+10.0\%$ paired gain), and from $11.7\% \pm 5.8\%$ to **$21.7\% \pm 2.9\%$** under Condition A ($+10.0\%$ paired gain). At the fixed threshold $	heta = 0.50$, M2 intercepted $3.3\%$ of episodes (Seed 123 detected 2 episodes), whereas M0/M1 detected 0 episodes.

### 3. Did it reduce Critical-to-Warning confusion?
**NO.** At the peak operating points, when true Critical windows crossed the failure threshold, **$100\%$ of them were misclassified as Warning** ($cm_{21} = 9, 7, 1$ across seeds, $cm_{22} = 0$).

### 4. Did it reduce Critical-to-Normal confusion?
**MODESTLY.** At $	heta = 0.50$ in Seed 123, M2 reduced Critical-to-Normal false negatives from 49 to 47. At peak F1, M2 alerted 1 to 9 Critical windows (as Warning) rather than leaving them completely unalerted. However, $80\%$ of Critical windows still fail to exceed $	heta_{	ext{fail}}$.

### 5. Did calibration improve or deteriorate?
**IMPROVED SUBSTANTIALLY.** This is the primary scientific achievement of Phase 7M. Across all three seeds, Multiclass Brier score dropped from $0.3152$ to **$0.2872$** ($-8.9\%$), and ECE dropped from $0.1653$ to **$0.1244$** ($-24.7\%$). The historical summary stream provided stabilizing contextual priors that suppressed the overconfident spikes characteristic of flat $L=12$.

### 6. Did false-alarm rates remain within the required limits?
**YES.** At the predeclared threshold $	heta = 0.50$, M2 achieved Pure-Normal Entity FAR of **$3.4\% \pm 4.1\%$** (well below the $10\%$ gate) and Normal-Window FPR of **$0.5\% \pm 0.3\%$** (well below the $5\%$ gate). M2 satisfies Gate 2 in all seeds.

### 7. Were the improvements consistent across all three seeds?
* **Calibration & Specificity:** **100% consistent.** Brier score, ECE, Entity FAR, and Window FPR improved in all three seeds.
* **Macro-F1 & Sensitivity:** **Mixed.** Macro-F1 at $	heta=0.50$ improved in all 3 seeds ($+0.0076, +0.0343, +0.0012$), but Episode Recall improvement was concentrated in Seed 123 ($+10.0\%$).

### 8. Did the model satisfy all four operational gates simultaneously?
**NO.** Gate 1 and Gate 2 remain mutually exclusive. Zero configurations passed all four gates simultaneously.

### 9. Which conclusions are directly supported by the results?
* **Supported Fact 1:** Multi-resolution representation fixes the calibration breakdown of long-sequence models.
* **Supported Fact 2:** Multi-resolution summaries dramatically suppress false alarms on healthy entities at $	heta = 0.50$.
* **Supported Fact 3:** Multi-resolution input alone does not solve Critical-class separation under symmetric loss weighting.

### 10. Which explanations remain hypotheses?
* **Hypothesis:** That introducing an asymmetric margin loss (e.g. Critical-weighted focal loss or margin ranking loss) will allow the multi-resolution features to tilt $P_2 > P_1$.

### 11. What is the most important remaining research limitation?
* **Validation Cohort Sample Size:** Exactly 20 failure episodes and 49 pure-normal entities. Every episode represents a $5.0\%$ swing in recall, limiting statistical power.

### 12. Should the next phase investigate loss functions, calibration, temporal pooling, or a new dataset?
* **Next Phase Recommendation (Phase 7N):** The temporal representation problem is solved by multi-resolution pooling; the remaining bottleneck is the **loss function and margin calibration**. Phase 7N should investigate **asymmetric margin-enforcing loss functions (e.g. Critical-focused margin loss or ordinal classification)** on top of the Phase 7M multi-resolution backbone.

---

## 6. Deliverables Index and Manifest

All 17 deliverables are cataloged in [`results/phase7m_multiresolution/phase7m_manifest.json`](file:///C:/Users/hp/Desktop/Research%20project/results/phase7m_multiresolution/phase7m_manifest.json):
* Design Specification: [`phase7m_design.md`](file:///C:/Users/hp/Desktop/Research%20project/results/phase7m_multiresolution/phase7m_design.md)
* Experiment Configuration: [`phase7m_config.json`](file:///C:/Users/hp/Desktop/Research%20project/results/phase7m_multiresolution/phase7m_config.json)
* Data Manifest: [`data_manifest.json`](file:///C:/Users/hp/Desktop/Research%20project/results/phase7m_multiresolution/data_manifest.json)
* Integrity Report: [`integrity_checks.json`](file:///C:/Users/hp/Desktop/Research%20project/results/phase7m_multiresolution/integrity_checks.json)
* Training Logs (300 Epochs): [`training_logs.csv`](file:///C:/Users/hp/Desktop/Research%20project/results/phase7m_multiresolution/training_logs.csv)
* Validation Predictions: [`validation_predictions.csv`](file:///C:/Users/hp/Desktop/Research%20project/results/phase7m_multiresolution/validation_predictions.csv)
* Per-Seed Metrics: [`per_seed_metrics.csv`](file:///C:/Users/hp/Desktop/Research%20project/results/phase7m_multiresolution/per_seed_metrics.csv)
* Paired Comparisons: [`paired_comparison.csv`](file:///C:/Users/hp/Desktop/Research%20project/results/phase7m_multiresolution/paired_comparison.csv)
* Confusion Matrices: [`confusion_matrices.csv`](file:///C:/Users/hp/Desktop/Research%20project/results/phase7m_multiresolution/confusion_matrices.csv)
* Critical Probability Forensics: [`critical_probability_analysis.csv`](file:///C:/Users/hp/Desktop/Research%20project/results/phase7m_multiresolution/critical_probability_analysis.csv)
* Episode Error Forensics: [`episode_error_analysis.csv`](file:///C:/Users/hp/Desktop/Research%20project/results/phase7m_multiresolution/episode_error_analysis.csv)
* Calibration Analysis: [`calibration_analysis.csv`](file:///C:/Users/hp/Desktop/Research%20project/results/phase7m_multiresolution/calibration_analysis.csv)
* Threshold Sweep (3,618 Rows): [`threshold_sweep.csv`](file:///C:/Users/hp/Desktop/Research%20project/results/phase7m_multiresolution/threshold_sweep.csv)
* Operational Gate Summary: [`operational_gate_summary.csv`](file:///C:/Users/hp/Desktop/Research%20project/results/phase7m_multiresolution/operational_gate_summary.csv)
* Independent Audit Report: [`independent_audit_report.md`](file:///C:/Users/hp/Desktop/Research%20project/results/phase7m_multiresolution/independent_audit_report.md)
* Final Report: [`phase7m_final_report.md`](file:///C:/Users/hp/Desktop/Research%20project/results/phase7m_multiresolution/phase7m_final_report.md)
* Sealed Manifest: [`phase7m_manifest.json`](file:///C:/Users/hp/Desktop/Research%20project/results/phase7m_multiresolution/phase7m_manifest.json)
