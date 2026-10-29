# Phase 7J-A — Independent Metric Reproduction and Artifact Consistency Audit Report

**Project:** Failure Prediction in Google Borg Cluster Traces  
**Audit Phase:** Phase 7J-A  
**Date:** October 28, 2026  
**Auditor:** Independent Scientific Audit Agent (Antigravity)  
**Execution Environment:** Windows (PowerShell, Python 3.11 Virtual Environment)  
**Evaluated Data Split:** Candidate A Validation Set Only ($N = 657$ windows)  
**Evaluation Status:** Validation-Only Diagnostic Audit — STRICTLY NO TEST ACCESS — NO RETRAINING  
**Executive Verdict:** **`NO OPERATIONAL WINNER` CONFIRMED (0 of 804 Operating Points Pass All Gates)**

---

## Executive Summary

Phase 7J-A was executed as an independent audit and reproducibility study of Phase 7J without retraining any models or evaluating any test data. The objective was to resolve discrepancies between preliminary threshold sweep outputs, the final investigation scripts, saved CSV/JSON artifacts, and [`phase7j_report.md`](file:///c:/Users/hp/Desktop/Research%20project/results/phase7j/phase7j_report.md).

### Core Audit Findings

1. **Independent Metric Reproduction:**  
   Across all 4 loss-weighting conditions ($A: [1,1,1]$, $B: [1,2,2]$, $C: [1,3,3]$, $D: [1,5,5]$) and all 201 threshold steps ($\theta \in [0.000, 1.000]$ in increments of $0.005$, yielding 804 condition-threshold combinations and 1,608 policy evaluations), **all 18 evaluated metric columns match the saved Phase 7J artifacts exactly** (maximum absolute numerical difference $< 10^{-5}$).

2. **Resolution of Reported Discrepancies:**  
   - **Condition A at $\theta = 0.140$:** The preliminary sweep text claiming Episode Recall 85%, Critical Recall 59.2%, Entity FAR 63.3%, and Window FPR 46.6% was identified as premature ungrounded draft text emitted into the chat transcript while background task `task-4902` was still executing. When `task-4902` and the final script completed, their true empirical outputs and saved artifacts confirmed the authoritative metrics: **Episode Recall = 85.0% (17/20)**, **Critical Recall = 53.06% (26/49)**, **Pure-Normal Entity FAR = 97.96% (48/49)**, and **Normal-Window FPR = 68.93% (386/560)**. The 59.18% (29/49) figure actually corresponds to $\theta = 0.130$, whereas 63.3% and 46.6% were ungrounded speculative interpolations.
   - **Peak Macro-F1 across Conditions A–D:** The preliminary values ($0.4578, 0.4418, 0.4079, 0.3702$) were likewise premature speculative text drafted while background task `task-4907` was running. Once completed, the true empirical peaks were confirmed as:
     - Condition A: $\theta = 0.205$, Macro-F1 = $0.3923$
     - Condition B: $\theta = 0.325$, Macro-F1 = $0.3792$
     - Condition C: $\theta = 0.410$, Macro-F1 = $0.4129$
     - Condition D: $\theta = 0.510$, Macro-F1 = $0.4287$
     The saved artifacts and final report reflect these true empirical numbers.
   - **Operating-Point Counts:** The count ambiguity between 804 and 1,608 is fully resolved. There are exactly 804 unique condition-threshold combinations (4 conditions $\times$ 201 thresholds). Because each combination was evaluated under two distinct decision rules (Policy A: Binary Alert and Policy B: Cascaded Three-Class), there are exactly $804 \times 2 = 1,608$ policy evaluations.

3. **Recheck of the Four Operational Gates:**  
   - Gate 1 (Detection): Episode Recall $\ge 70\%$ AND Genuine Critical Recall $\ge 50\%$.  
   - Gate 2 (False Alarms): Pure-Normal Entity FAR $\le 10\%$ AND Normal-Window FPR $\le 5\%$.  
   **Result:** Under both Policy A and Policy B, exactly **0 out of 804 condition-threshold points (0 out of 1,608 policy evaluations)** satisfy both Gate 1 and Gate 2 simultaneously. The failure detection domain and the low-false-alarm domain are completely disjoint in threshold space.

4. **Preserved Final Decision:**  
   The verdict **`NO OPERATIONAL WINNER`** is definitively upheld. No confirmatory test evaluation is authorized.

---

## Section A: Source Input Identification and Cryptographic Hash Audit

All inputs utilized in Phase 7J and audited in Phase 7J-A were verified for cryptographic integrity, row counts, dimensional shapes, and label alignments.

### 1. Cryptographic Hashes and File Specifications

| File Description | Relative Path | Size (Bytes) | Shape / Dimensions | SHA-256 Hash |
| :--- | :--- | :---: | :---: | :--- |
| **Validation Predictions** | [`results/phase7i/phase7i_validation_predictions.csv`](file:///c:/Users/hp/Desktop/Research%20project/results/phase7i/phase7i_validation_predictions.csv) | 149,783 | 657 rows $\times$ 21 cols | `9e405514b98e043409585b3e2b1120a68d75afcb10d982667bd6b20661d6f1fc` |
| **Validation Metadata** | [`data/processed/phase7c/val_metadata.csv`](file:///c:/Users/hp/Desktop/Research%20project/data/processed/phase7c/val_metadata.csv) | 32,677 | 657 rows $\times$ 6 cols | `8d22193f11d671dc570d29abf9255d06386880d49c5de83c68d6404a8b41b956` |
| **Validation Labels** | [`data/processed/phase7c/y_val.npy`](file:///c:/Users/hp/Desktop/Research%20project/data/processed/phase7c/y_val.npy) | 5,384 | (657,) | `eb968a74ae86dcc574f3341055ff45ca9fcf77ec5a1b4cee5ad8cb4c1e2631b9` |
| **Training Metadata** | [`data/processed/phase7c/train_metadata.csv`](file:///c:/Users/hp/Desktop/Research%20project/data/processed/phase7c/train_metadata.csv) | 129,742 | 2,618 rows $\times$ 6 cols | `0180ade4098711680c44bbd2ca8c3c5d85834a86042546cc9300b00f1df725ce` |
| **Training Labels** | [`data/processed/phase7c/y_train.npy`](file:///c:/Users/hp/Desktop/Research%20project/data/processed/phase7c/y_train.npy) | 21,072 | (2,618,) | `355ca59b1609780181fabe8f06c6ab51cc5c53cfbec91cb822d76b8e373fab1e` |

### 2. Class Ordering and Output Mapping

The validation predictions file [`phase7i_validation_predictions.csv`](file:///c:/Users/hp/Desktop/Research%20project/results/phase7i/phase7i_validation_predictions.csv) contains softmax class probabilities indexed as:
- Class Index `0`: `prob_<cond>_c0` = Normal class probability ($P_0$)
- Class Index `1`: `prob_<cond>_c1` = Warning class probability ($P_1$)
- Class Index `2`: `prob_<cond>_c2` = Critical class probability ($P_2$)
- Aggregate Failure Probability: `p_fail_<cond>` $= P_1 + P_2 = 1.0 - P_0$

Probabilities across all rows are strictly bounded in $[0, 1]$, finite, and satisfy $\sum_{c=0}^2 P_c \approx 1.0$ with a maximum floating-point deviation of $1.36 \times 10^{-7}$. There are zero ties where $P_1 = P_2$.

### 3. Row Alignment and Cohort Structure

Exact row-for-row alignment was verified across all 657 rows between `val_metadata.csv` and `phase7i_validation_predictions.csv`:
- `entity_id`: 100% exact match (657/657)
- `prediction_timestamp`: 100% exact match (657/657)
- `time_to_failure`: 100% exact match (657/657)
- `label` vs `y_true`: 100% exact match (657/657)
- `parent_job_id`: 100% exact match (657/657)

**Validation Cohort Composition:**
- Total Windows: $N = 657$
- Normal Windows (Class 0): $N_{norm} = 560$ (85.24%)
- Warning Windows (Class 1): $N_{warn} = 48$ (7.31%)
- Critical Windows (Class 2): $N_{crit} = 49$ (7.46%)
- Pre-Failure Windows (Class 1 + Class 2): $N_{fail} = 97$ (14.76%)
- Pre-Failure Episodes: $N_{ep} = 20$ (grouped by `entity_id` and `fail_timestamp` $= \text{round}(t + \text{TTF}, 1)$)
- Failing Entities: $N_{failing\_entities} = 17$
- Pure-Normal Entities: $N_{pure\_norm\_entities} = 49$ (entities with zero pre-failure windows)
- Pure-Normal Parent Jobs: $N_{pure\_norm\_jobs} = 4$ (parent jobs with zero pre-failure windows)
- Total Parent Jobs: $N_{jobs} = 8$

### 4. Comparison of Preliminary Script vs. Final Investigation Script

A line-by-line comparison between [`scratch/test_threshold_sweep.py`](file:///c:/Users/hp/Desktop/Research%20project/scratch/test_threshold_sweep.py) and [`scripts/run_phase7j_investigation.py`](file:///c:/Users/hp/Desktop/Research%20project/scripts/run_phase7j_investigation.py) confirms:
- **Input Arrays & Metadata:** Identical files loaded (`phase7i_validation_predictions.csv`, `val_metadata.csv`, `y_val.npy`).
- **Policy Implementations:** Identical decision rules (Policy A: alert if $P_{fail} \ge \theta$; Policy B: alert if $P_{fail} \ge \theta$, then Critical if $P_2 \ge P_1$ else Warning).
- **Metric Definitions:** Identical formulas and aggregations.
- **Threshold Grid Difference:** `scratch/test_threshold_sweep.py` was a diagnostic prototype evaluating selected threshold points and coarse grids, whereas `scripts/run_phase7j_investigation.py` executed the full predeclared 201-point uniform grid ($\theta \in [0.000, 1.000]$, step $0.005$).

---

## Section B: Independent Metric Reproduction Methodology

The audit script [`scratch/independent_phase7j_reproduction.py`](file:///c:/Users/hp/Desktop/Research%20project/scratch/independent_phase7j_reproduction.py) implemented the metric evaluations independently from the original Phase 7J script.

### 1. Decision Policies Evaluated

- **Policy A (Binary Alert):**  
  An alert is raised if aggregate pre-failure probability $P_{fail} = P_1 + P_2 \ge \theta$:
  $$\hat{y}_{bin} = \begin{cases} 1 & \text{if } P_{fail} \ge \theta \\ 0 & \text{if } P_{fail} < \theta \end{cases}$$

- **Policy B (Cascaded Three-Class Decision):**  
  A cascaded decision first applies the threshold $\theta$ to separate Normal from failure, then disambiguates between Warning and Critical:
  $$\hat{y}_{3class} = \begin{cases} 0 & \text{if } P_{fail} < \theta \\ 2 & \text{if } P_{fail} \ge \theta \text{ and } P_2 \ge P_1 \\ 1 & \text{if } P_{fail} \ge \theta \text{ and } P_1 > P_2 \end{cases}$$

### 2. Mathematical Metric Definitions

1. **Binary Episode Recall:**  
   $$\text{Episode Recall} = \frac{\sum_{e=1}^{N_{ep}} \mathbb{I}\left( \exists w \in e : \hat{y}(w) > 0 \right)}{N_{ep}} \times 100\% \quad (N_{ep} = 20)$$
   An episode is detected if at least one evaluation window in the episode triggers an alert.

2. **Binary Critical Interception:**  
   $$\text{Critical Interception} = \frac{\sum_{w : y(w) = 2} \mathbb{I}\left( \hat{y}(w) > 0 \right)}{N_{crit}} \times 100\% \quad (N_{crit} = 49)$$
   Measures whether a Critical window triggers an alert under Policy A or Policy B, regardless of whether it is classified as Warning or Critical.

3. **Genuine Cascaded Critical Recall (Three-Class):**  
   $$\text{Critical Recall} = \frac{\sum_{w : y(w) = 2} \mathbb{I}\left( \hat{y}(w) = 2 \right)}{N_{crit}} \times 100\% \quad (N_{crit} = 49)$$
   Requires the model to explicitly predict Class 2 ($\hat{y} = 2$) on true Class 2 windows.

4. **Warning Recall:**  
   $$\text{Warning Recall} = \frac{\sum_{w : y(w) = 1} \mathbb{I}\left( \hat{y}(w) = 1 \right)}{N_{warn}} \times 100\% \quad (N_{warn} = 48)$$

5. **Pure-Normal Entity False Alarm Rate (FAR):**  
   $$\text{Entity FAR} = \frac{\sum_{i \in \mathcal{E}_{norm}} \mathbb{I}\left( \exists w \in i : \hat{y}(w) > 0 \right)}{|\mathcal{E}_{norm}|} \times 100\% \quad (|\mathcal{E}_{norm}| = 49)$$
   A pure-normal entity incurs a false alarm if *any* of its windows triggers an alert.

6. **Normal-Window False Positive Rate (FPR):**  
   $$\text{Window FPR} = \frac{\sum_{w : y(w) = 0} \mathbb{I}\left( \hat{y}(w) > 0 \right)}{N_{norm}} \times 100\% \quad (N_{norm} = 560)$$
   The denominator is explicitly defined as all ground-truth Normal windows ($N_{norm} = 560$).

7. **Pure-Normal Parent-Job False Alarm Rate (FAR):**  
   $$\text{Job FAR} = \frac{\sum_{j \in \mathcal{J}_{norm}} \mathbb{I}\left( \exists w \in j : \hat{y}(w) > 0 \right)}{|\mathcal{J}_{norm}|} \times 100\% \quad (|\mathcal{J}_{norm}| = 4)$$

8. **Macro-F1 and Balanced Accuracy:**  
   Evaluated using standard multi-class formulations across all three classes $\{0, 1, 2\}$, with zero-division set to $0.0$.

---

## Section C: Forensic Discrepancy Resolution Register

All six reported discrepancies were investigated forensically against transcripts, background task outputs, and filesystem artifacts.

| Issue ID | Metric / Setting | Preliminary / Speculative Value | Authoritative Reproduced Value | Source File / Output | Resolution Status | Evidence-Based Explanation |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **DISC-01** | Condition A, Policy B, $\theta = 0.140$ | EpRec: 85.0%, CritRec: 59.2%, EntFAR: 63.3%, WinFPR: 46.6% | **EpRec: 85.0% (17/20)**, **CritRec: 53.06% (26/49)**, **EntFAR: 97.96% (48/49)**, **WinFPR: 68.93% (386/560)** | Transcript vs. [`phase7j_threshold_metrics.csv`](file:///c:/Users/hp/Desktop/Research%20project/results/phase7j/phase7j_threshold_metrics.csv) | **RESOLVED** (Premature Draft Interpolation) | The preliminary sweep values were ungrounded speculative text emitted in the assistant turn while background task `task-4902` was still in flight. When `task-4902` completed, its true console output confirmed 85.0%, 53.1%, 98.0%, 68.9%. In fact, 59.18% (29/49) Critical Recall actually occurs at $\theta = 0.130$, whereas 63.3% (31/49) and 46.6% (261/560) were ungrounded interpolations. The saved CSV and report contain the true empirical metrics. |
| **DISC-02** | Condition A Peak Macro-F1 | $\theta = 0.180$, Macro-F1 = 0.4578 | **$\theta = 0.205$, Macro-F1 = 0.3923** (0.392347) | `task-4907` log vs. [`phase7j_gate_summary.csv`](file:///c:/Users/hp/Desktop/Research%20project/results/phase7j/phase7j_gate_summary.csv) line 2 | **RESOLVED** (Premature Draft Interpolation) | Emitted as preliminary ungrounded text while `task-4907` was executing. Upon completion, the true empirical peak was confirmed at $\theta = 0.205$ with Macro-F1 = 0.3923. At $\theta = 0.180$, true Policy B Macro-F1 is 0.3754. The saved CSV and final report reflect the verified peak. |
| **DISC-03** | Condition B Peak Macro-F1 | $\theta = 0.300$, Macro-F1 = 0.4418 | **$\theta = 0.325$, Macro-F1 = 0.3792** (0.379190) | `task-4907` log vs. [`phase7j_gate_summary.csv`](file:///c:/Users/hp/Desktop/Research%20project/results/phase7j/phase7j_gate_summary.csv) line 3 | **RESOLVED** (Premature Draft Interpolation) | Emitted as premature ungrounded text during `task-4907` execution. The true empirical peak is $\theta = 0.325$, Macro-F1 = 0.3792. At $\theta = 0.300$, true Policy B Macro-F1 is 0.3536. |
| **DISC-04** | Condition C Peak Macro-F1 | $\theta = 0.370$, Macro-F1 = 0.4079 | **$\theta = 0.410$, Macro-F1 = 0.4129** (0.412872) | `task-4907` log vs. [`phase7j_gate_summary.csv`](file:///c:/Users/hp/Desktop/Research%20project/results/phase7j/phase7j_gate_summary.csv) line 4 | **RESOLVED** (Premature Draft Interpolation) | Emitted as premature ungrounded text during `task-4907` execution. The true empirical peak is $\theta = 0.410$, Macro-F1 = 0.4129. At $\theta = 0.370$, true Policy B Macro-F1 is 0.3465. |
| **DISC-05** | Condition D Peak Macro-F1 | $\theta = 0.485$, Macro-F1 = 0.3702 | **$\theta = 0.510$, Macro-F1 = 0.4287** (0.428708) | `task-4907` log vs. [`phase7j_gate_summary.csv`](file:///c:/Users/hp/Desktop/Research%20project/results/phase7j/phase7j_gate_summary.csv) line 5 | **RESOLVED** (Premature Draft Interpolation) | Emitted as premature ungrounded text during `task-4907` execution. The true empirical peak is $\theta = 0.510$, Macro-F1 = 0.4287. At $\theta = 0.485$, true Policy B Macro-F1 is 0.3833. |
| **DISC-06** | Operating-Point Count Ambiguity | Described interchangeably as 804 and 1,608 | **804 condition-threshold combinations; 1,608 policy evaluations** | [`phase7j_threshold_metrics.csv`](file:///c:/Users/hp/Desktop/Research%20project/results/phase7j/phase7j_threshold_metrics.csv) | **RESOLVED** (Terminological Distinction) | There are exactly 4 conditions $\times$ 201 threshold steps = 804 condition-threshold combinations (the 804 rows of the metrics CSV). Evaluating two policies (Policy A and Policy B) per row yields $804 \times 2 = 1,608$ policy evaluations. Both numbers are exact. |

All discrepancies are completely resolved. No unexplained discrepancies remain.

---

## Section D: Recheck of the Four Operational Gates

The four operational deployment gates established in Phase 7I were evaluated across all 804 operating points under both Policy A and Policy B.

### 1. Gate Definitions
- **Gate 1 (Detection Gate):** Episode Recall $\ge 70.0\%$ AND Genuine Critical Recall $\ge 50.0\%$.
- **Gate 2 (False Alarm Gate):** Pure-Normal Entity FAR $\le 10.0\%$ AND Normal-Window FPR $\le 5.0\%$.
- **Operational Winner Requirement:** A model/operating point must pass **both Gate 1 and Gate 2 simultaneously**.

### 2. Independent Gate Verification Summary

| Condition | Description | Policy A Passing Operating Points | Policy B Passing Operating Points | All Gates Passed (Any Policy) | Best Detection when Gate 2 Passes ($\theta$, EpRec, CritRec, EntFAR, WinFPR) | Best False Alarm when Gate 1 Passes ($\theta$, EpRec, CritRec, EntFAR, WinFPR) |
| :--- | :--- | :---: | :---: | :---: | :--- | :--- |
| **Condition A** | Unweighted `[1, 1, 1]` | **0 / 201** | **0 / 201** | **FALSE (0 / 402)** | $\theta = 0.215$: EpRec 35.0%, CritRec 22.4%, EntFAR 6.1%, WinFPR 4.6% | $\theta = 0.140$: EpRec 85.0%, CritRec 53.1%, EntFAR 98.0%, WinFPR 68.9% |
| **Condition B** | Moderate `[1, 2, 2]` | **0 / 201** | **0 / 201** | **FALSE (0 / 402)** | $\theta = 0.355$: EpRec 15.0%, CritRec 4.1%, EntFAR 8.2%, WinFPR 2.5% | $\theta = 0.255$: EpRec 85.0%, CritRec 53.1%, EntFAR 95.9%, WinFPR 72.0% |
| **Condition C** | Stronger `[1, 3, 3]` | **0 / 201** | **0 / 201** | **FALSE (0 / 402)** | $\theta = 0.490$: EpRec 0.0%, CritRec 0.0%, EntFAR 8.2%, WinFPR 1.4% | None (Gate 1 never passes above $\theta = 0.400$) |
| **Condition D** | Higher `[1, 5, 5]` | **0 / 201** | **0 / 201** | **FALSE (0 / 402)** | $\theta = 0.625$: EpRec 0.0%, CritRec 0.0%, EntFAR 6.1%, WinFPR 0.9% | None (Gate 1 never passes above $\theta = 0.500$) |
| **Total** | **All 4 Conditions** | **0 / 804** | **0 / 804** | **FALSE (0 / 1608)** | — | — |

### 3. Detailed Numerators, Denominators, and Confusion Matrices for Key Operating Points

#### Condition A (Unweighted `[1, 1, 1]`)

1. **Detection Gate Threshold ($\theta = 0.140$):**
   - **Policy A:** EpRec = $85.00\%$ ($17/20$), CritInt = $81.63\%$ ($40/49$), EntFAR = $97.96\%$ ($48/49$), WinFPR = $68.93\%$ ($386/560$), Binary Prec = $0.1590$, Binary Rec = $0.7526$, Binary F1 = $0.2626$.
   - **Policy B:** EpRec = $85.00\%$ ($17/20$), CritRec = $53.06\%$ ($26/49$), WarnRec = $10.42\%$ ($5/48$), EntFAR = $97.96\%$ ($48/49$), WinFPR = $68.93\%$ ($386/560$), Macro-F1 = $0.2268$, BalAcc = $0.3152$.
   - **Confusion Matrix B:**
     $$\begin{bmatrix} \text{True Normal (560)} \\ \text{True Warning (48)} \\ \text{True Critical (49)} \end{bmatrix} \longrightarrow \begin{bmatrix} 174 & 208 & 178 \\ 15 & 5 & 28 \\ 9 & 14 & 26 \end{bmatrix}$$
   - **Gate Outcome:** Gate 1 PASSES ($85.0\% \ge 70\%$, $53.1\% \ge 50\%$), but Gate 2 FAILS CATASTROPHICALLY ($98.0\% > 10\%$, $68.9\% > 5\%$). **48 out of 49 pure-normal entities trigger false alarms.**

2. **Peak Macro-F1 Operating Point ($\theta = 0.205$):**
   - **Policy B:** Macro-F1 = $0.3923$, BalAcc = $0.4116$, EpRec = $50.00\%$ ($10/20$), CritRec = $30.61\%$ ($15/49$), WarnRec = $0.00\%$ ($0/48$), EntFAR = $6.12\%$ ($3/49$), WinFPR = $7.14\%$ ($40/560$).
   - **Confusion Matrix B:**
     $$\begin{bmatrix} \text{True Normal (560)} \\ \text{True Warning (48)} \\ \text{True Critical (49)} \end{bmatrix} \longrightarrow \begin{bmatrix} 520 & 9 & 31 \\ 29 & 0 & 19 \\ 29 & 5 & 15 \end{bmatrix}$$
   - **Gate Outcome:** Gate 1 FAILS ($50.0\% < 70\%$, $30.6\% < 50\%$), and Gate 2 FAILS ($7.14\% > 5.0\%$).

3. **Best Detection when Gate 2 Passes ($\theta = 0.215$):**
   - **Policy B:** EpRec = $35.00\%$ ($7/20$), CritRec = $22.45\%$ ($11/49$), WarnRec = $0.00\%$ ($0/48$), EntFAR = $6.12\%$ ($3/49$), WinFPR = $4.64\%$ ($26/560$), Macro-F1 = $0.3836$, BalAcc = $0.3927$.
   - **Confusion Matrix B:**
     $$\begin{bmatrix} \text{True Normal (560)} \\ \text{True Warning (48)} \\ \text{True Critical (49)} \end{bmatrix} \longrightarrow \begin{bmatrix} 534 & 5 & 21 \\ 35 & 0 & 13 \\ 36 & 2 & 11 \end{bmatrix}$$
   - **Gate Outcome:** Gate 2 PASSES ($6.12\% \le 10\%$, $4.64\% \le 5\%$), but Gate 1 COLLAPSES ($35.0\% < 70\%$, $22.5\% < 50\%$).

#### Condition B (Moderate `[1, 2, 2]`)

1. **Detection Gate Threshold ($\theta = 0.255$):**
   - **Policy B:** EpRec = $85.00\%$ ($17/20$), CritRec = $53.06\%$ ($26/49$), WarnRec = $20.83\%$ ($10/48$), EntFAR = $95.92\%$ ($47/49$), WinFPR = $71.96\%$ ($403/560$), Macro-F1 = $0.2223$.
   - **Confusion Matrix B:**
     $$\begin{bmatrix} \text{True Normal (560)} \\ \text{True Warning (48)} \\ \text{True Critical (49)} \end{bmatrix} \longrightarrow \begin{bmatrix} 157 & 178 & 225 \\ 5 & 10 & 33 \\ 5 & 18 & 26 \end{bmatrix}$$
   - **Gate Outcome:** Gate 1 PASSES, Gate 2 FAILS CATASTROPHICALLY ($95.9\% > 10\%$, $72.0\% > 5\%$).

2. **Peak Macro-F1 Operating Point ($\theta = 0.325$):**
   - **Policy B:** Macro-F1 = $0.3792$, BalAcc = $0.3984$, EpRec = $55.00\%$ ($11/20$), CritRec = $28.57\%$ ($14/49$), WarnRec = $4.17\%$ ($2/48$), EntFAR = $22.45\%$ ($11/49$), WinFPR = $13.21\%$ ($74/560$).
   - **Gate Outcome:** Gate 1 FAILS ($55.0\% < 70\%$), Gate 2 FAILS ($22.5\% > 10\%$, $13.2\% > 5\%$).

3. **Best Detection when Gate 2 Passes ($\theta = 0.355$):**
   - **Policy B:** EpRec = $15.00\%$ ($3/20$), CritRec = $4.08\%$ ($2/49$), WarnRec = $0.00\%$ ($0/48$), EntFAR = $8.16\%$ ($4/49$), WinFPR = $2.50\%$ ($14/560$), Macro-F1 = $0.3236$.
   - **Confusion Matrix B:**
     $$\begin{bmatrix} \text{True Normal (560)} \\ \text{True Warning (48)} \\ \text{True Critical (49)} \end{bmatrix} \longrightarrow \begin{bmatrix} 546 & 0 & 14 \\ 42 & 0 & 6 \\ 46 & 1 & 2 \end{bmatrix}$$
   - **Gate Outcome:** Gate 2 PASSES, but Gate 1 COLLAPSES to near zero ($15.0\% < 70\%$, $4.1\% < 50\%$).

#### Condition C (Stronger `[1, 3, 3]`) and Condition D (Higher `[1, 5, 5]`)

- In Conditions C and D, the failure probability mass is shifted upward by heavy training loss weights.
- When $\theta$ is raised high enough to satisfy Gate 2 ($\theta \ge 0.490$ for Cond C, $\theta \ge 0.625$ for Cond D), **Episode Recall and Critical Recall drop to exactly $0.00\%$ ($0/20$ episodes detected, $0/49$ critical windows detected)**.
- At Peak Macro-F1 ($\theta = 0.410$ for Cond C with Macro-F1 = $0.4129$; $\theta = 0.510$ for Cond D with Macro-F1 = $0.4287$), False Alarms remain excessive (EntFAR = $22.45\%$, WinFPR = $15.18\%$–$18.04\%$), while Critical Recall remains deficient ($14.29\%$ and $12.24\%$, far below the $50\%$ requirement).

### 4. Summary of Operational Failure

Across the entire continuum of 804 operating points, the model exhibits a severe trade-off barrier:
- Thresholds low enough to detect failures ($\theta \le 0.140$ in Condition A, $\theta \le 0.255$ in Condition B) flag 95%–98% of pure-normal entities and 68%–72% of normal windows.
- Thresholds high enough to suppress false alarms below operational limits ($\theta \ge 0.215$ in Condition A, $\theta \ge 0.355$ in Condition B) reduce Episode Recall to $\le 35\%$ and Critical Recall to $\le 22\%$, well below operational viability.
- **Therefore, threshold tuning alone cannot bridge the gap between detection and precision on Candidate A Validation data.**

---

## Section E: Corrected Provenance and Methodological Rigor

To maintain scientific integrity and prevent invalid generalization claims, provenance and data boundaries must be stated with strict precision.

### 1. Accurate Historical Context for Candidate A Test Set
- **Past Inspection in Phase 7D:** Candidate A Test tensors, labels, and performance were previously evaluated and inspected during Phase 7D. Candidate A Test must **never** be described as "newly blinded," "completely untouched," or "uncontaminated" across the research history.
- **Non-Access in Phase 7J and Phase 7J-A:** Phase 7J and Phase 7J-A maintained complete isolation from Candidate A Test. No test files, test labels, or test predictions were loaded, inspected, or queried during these phases. All analyses were conducted strictly on Candidate A Validation data ($N = 657$).

### 2. Status of Phase 6 Confirmatory Holdout
- The Phase 6 confirmatory holdout remains in strict, uncompromised quarantine. No script in Phase 7 has accessed, evaluated, or touched Phase 6 holdout data.

### 3. Static Code Verification vs. Retrospective OS-Level Limitations
- **Static Code Inspection:** Static inspection of [`scripts/run_phase7j_investigation.py`](file:///c:/Users/hp/Desktop/Research%20project/scripts/run_phase7j_investigation.py), [`scratch/test_threshold_sweep.py`](file:///c:/Users/hp/Desktop/Research%20project/scratch/test_threshold_sweep.py), and [`scratch/independent_phase7j_reproduction.py`](file:///c:/Users/hp/Desktop/Research%20project/scratch/independent_phase7j_reproduction.py) confirms that only validation paths are coded into data-loading functions.
- **Execution Log Verification:** Execution logs confirm that the running process touched only validation files and produced outputs solely derived from validation arrays.
- **Retrospective Epistemological Limitation:** While static code and active execution logs prove that current scripts did not touch test data, they cannot mathematically reconstruct or prove lower-level OS reads from past user sessions. Provenance claims must strictly distinguish current verifiable execution from retrospective historical assertions.

### 4. Exploratory Nature of Threshold Selection
- Any threshold tuning or operating-point exploration on validation data is strictly exploratory. Threshold selection on validation data does **not** constitute independent generalization or confirmatory performance.

---

## Final Decision Rule and Roadmap Recommendations

### 1. Final Decision Rule
Under the predeclared protocol and audit standards:
- If no operating point passes all four gates after independent reproduction, the system must retain **`NO OPERATIONAL WINNER`**.
- Because 0 of 804 condition-threshold points (0 of 1,608 policy evaluations) pass all four operational gates, **`NO OPERATIONAL WINNER` IS RETAINED AND CONFIRMED**.

### 2. Prohibition on Confirmatory Evaluation
- No confirmatory test evaluation is authorized. Evaluating Candidate A Test or Phase 6 holdout on these models would produce invalid, negative results while squandering holdout validity.

### 3. Transition to Subsequent Research Phases
- Phase 7K must **not** be initiated automatically in this turn.
- The outcome of Phase 7J and Phase 7J-A demonstrates that modifying decision thresholds on the current LSTM architecture cannot resolve the separation deficit between failure windows and normal windows. Future research must address feature representation, sequence length, or model architecture rather than relying on post-hoc threshold adjustment.

---

## Audit Artifact Manifest

The complete set of audit artifacts produced by Phase 7J-A is cataloged below:

| Artifact Path | Format | Description |
| :--- | :---: | :--- |
| [`results/phase7j_audit/phase7j_audit_report.md`](file:///c:/Users/hp/Desktop/Research%20project/results/phase7j_audit/phase7j_audit_report.md) | Markdown | Comprehensive Phase 7J-A audit report (authoritative copy) |
| [`reports/phase7j_audit/phase7j_audit_report.md`](file:///c:/Users/hp/Desktop/Research%20project/reports/phase7j_audit/phase7j_audit_report.md) | Markdown | Mirrored Phase 7J-A audit report |
| [`results/phase7j_audit/independent_threshold_metrics.csv`](file:///c:/Users/hp/Desktop/Research%20project/results/phase7j_audit/independent_threshold_metrics.csv) | CSV | Independently recomputed threshold metrics across all 804 points (804 rows $\times$ 49 cols) |
| [`results/phase7j_audit/independent_gate_summary.csv`](file:///c:/Users/hp/Desktop/Research%20project/results/phase7j_audit/independent_gate_summary.csv) | CSV | Independent gate verification summary across all 4 conditions (5 rows $\times$ 21 cols) |
| [`results/phase7j_audit/metric_discrepancy_register.csv`](file:///c:/Users/hp/Desktop/Research%20project/results/phase7j_audit/metric_discrepancy_register.csv) | CSV | Complete register of investigated discrepancies (DISC-01 to DISC-06) with evidence explanations |
| [`results/phase7j_audit/phase7j_audit_integrity.json`](file:///c:/Users/hp/Desktop/Research%20project/results/phase7j_audit/phase7j_audit_integrity.json) | JSON | Cryptographic input hashes, safeguards verification, and alignment ledger |
| [`results/phase7j_audit/phase7j_audit_manifest.json`](file:///c:/Users/hp/Desktop/Research%20project/results/phase7j_audit/phase7j_audit_manifest.json) | JSON | Cryptographic manifest of all generated audit files with SHA-256 hashes |
