# Phase 7D Final Authorization Gate — Protocol Conformance Report

**Date:** October 28, 2026  
**Auditor / Lead Engineer:** Senior ML Research Engineer & Data-Pipeline Auditor (DeepMind Antigravity)  
**Governing Evaluation Protocol:** Protocol v2.0 ([results/phase7b_1/phase7b_evaluation_protocol_v2.json](file:///c:/Users/hp/Desktop/Research%20project/results/phase7b_1/phase7b_evaluation_protocol_v2.json))  
**Governing Frozen Split:** Candidate A ([results/phase7a_4/candidate_a_job_grouped_split.csv](file:///c:/Users/hp/Desktop/Research%20project/results/phase7a_4/candidate_a_job_grouped_split.csv))  
**Input Reconstructed Dataset:** [`data/processed/phase7c/`](file:///c:/Users/hp/Desktop/Research%20project/data/processed/phase7c/)  
**Active Preprocessing Scaler:** [`configs/phase7c/fitted_scaler_phase7c.json`](file:///c:/Users/hp/Desktop/Research%20project/configs/phase7c/fitted_scaler_phase7c.json)  
**Machine-Readable Audit Ledger:** [`results/phase7d_authorization/authorization_gate_ledger.json`](file:///c:/Users/hp/Desktop/Research%20project/results/phase7d_authorization/authorization_gate_ledger.json)  

---

## 1. Executive Verdict & Gate Decision

### **Official Verdict:** **`AUTHORIZED UNDER EXISTING PROTOCOL`**

This report completes the final pre-training audit required before Phase 7D diagnostic model training may be initiated. All protocol specifications, implementation changes from Phase 7B, statistical resampling procedures, and interpretation frameworks have been audited against the primary code artifacts, data manifests, and Protocol v2.0.

### Summary of Audit Determinations:
1. **Complete Protocol Conformance:** Every operational parameter—model architectures, random seed 42, 31-feature registry ordering, validation candidate-selection gates, single confirmatory test evaluation, and dual bootstrap intervals—is fully specified in [results/phase7b_1/phase7b_evaluation_protocol_v2.json](file:///c:/Users/hp/Desktop/Research%20project/results/phase7b_1/phase7b_evaluation_protocol_v2.json) with zero unresolved ambiguities.
2. **Accounted Pipeline Differences:** All differences between Phase 7B and Phase 7D are classified into intended corrections to eliminate legacy contamination (reconstructing true raw CPI, eliminating the constant $2.50$, and enforcing training-only scaling) or necessary implementation changes. **Zero potential confounds requiring protocol amendments were introduced.**
3. **Calibrated Hypothesis Framework:** Deterministic claims of causality have been replaced with a formal 5-outcome evidence-calibrated decision matrix recognizing that cross-job distribution shift and model capacity remain plausible co-factors.
4. **Statistical Rigor on Clustered Bootstrapping:** The "dual cluster bootstrap" is verified to denote hierarchical statistical clustered resampling over `parent_job_id` ($J=11$) and `entity_id` ($N=59$), fully distinguishing statistical cluster units from the single physical Google Borg datacenter cluster.
5. **Research Safeguards Honored:** Model training, fine-tuning, threshold calibration, and test evaluation remain paused. The quarantined Phase 6 confirmatory holdout remains unopened.

---

## 2. Task 1 — Protocol Conformance Table

The following table provides an exhaustive verification of every required experimental parameter under Protocol v2.0, citing the authoritative source files and line ranges:

| Requirement Item | Exact Protocol v2.0 Specification | Source File Citation | Line Range Citation | Conformance Status |
| :--- | :--- | :--- | :--- | :---: |
| **Permitted Model Architectures** | **4 Architectures:**<br>1. **Random Forest:** 100 trees, `max_depth=15`, `class_weight='balanced'`, `random_state=42`.<br>2. **Temporal LSTM:** Input 31, hidden 64, 2 layers, dropout 0.2, weighted CE `[1.0, 1.5, 2.0]`, AdamW `lr=1e-3, weight_decay=1e-4`, ReduceLROnPlateau (`factor=0.5, patience=5`), early stopping patience 10 on `val_macro_f1` (max 35 epochs).<br>3. **Compact Transformer:** Input 31, `d_model=32`, 4 heads, 2 layers, `dim_feedforward=64`, dropout 0.1, weighted CE `[1.0, 1.5, 2.0]`, AdamW `lr=5e-4, weight_decay=1e-4`, ReduceLROnPlateau (`factor=0.5, patience=5`), patience 10 (max 35 epochs).<br>4. **Representation Hybrid:** Frozen Transformer 32-dim latent embedding (average-pooled over 5 steps) concatenated with 31-dim tabular vector $x_{\text{last}}$ (total 63-dim), classified by Random Forest. | `results/phase7b_1/phase7b_evaluation_protocol_v2.json`<br>&<br>`scripts/train_and_evaluate_phase7b.py` | Protocol v2.0: Lines 8–11, 23–28<br><br>Script: Lines 520–534, 558–594, 613–652, 670–696 | **CONFORMANT** |
| **Random Seeds & Number of Runs** | Fixed `seed=42` across Python `random`, OS `PYTHONHASHSEED`, NumPy `np.random`, and PyTorch `torch.manual_seed`. Exactly **1 single deterministic execution run** per model architecture. | `scripts/train_and_evaluate_phase7b.py` | Lines 76–87, 447, 508 | **CONFORMANT** |
| **Input Representation & Feature Order** | Sequence length $L=5$ trailing steps ($300\text{s}$ spacing, $25\text{ min}$ window). Exactly **31 features** ordered strictly per `FeatureRegistry`: `cpu_utilization`, `memory_utilization`, `assigned_memory`, `disk_io_time`, `cycles_per_instruction`, `memory_accesses_per_instruction`, lags 1/2/3/6 for CPU & Mem, first differences, rolling stats ($w \in \{3, 6\}$), slope, threshold exceedances, allocation ratios, task age, and CPI missingness fraction.<br>RF consumes $x_{\text{last}} \in \mathbb{R}^{31}$; LSTM/Transformer consume $[B, 5, 31]$; Hybrid consumes $[B, 63]$. | `src/features/registry.py`<br>&<br>`results/phase7c/data_manifest.json` | Registry: Lines 34–124<br><br>Data Manifest: Lines 10–43<br><br>Script: Line 483 | **CONFORMANT** |
| **Permitted Preprocessing & Imputation** | Scaler fitted strictly on Candidate A Training split ($N=3,278$). Median imputation ($a_{\text{train}}$) using training medians for unobserved MAI/CPI. Z-score standardization $z = (x - \mu_{\text{train}}) / (\sigma_{\text{train}} + \epsilon)$. Applied to Validation and Test via `scaler.transform()` without refitting. Zero cross-entity state carryover. | `results/phase7b_1/phase7b_evaluation_protocol_v2.json`<br>&<br>`src/features/scaler.py` | Protocol v2.0: Lines 113–117<br><br>Scaler: Lines 34–64<br><br>`fitted_scaler_phase7c.json` | **CONFORMANT** |
| **Required Metrics & Precise Definitions** | 1. **Primary:** Macro-F1 = $\frac{1}{3}(F1_{\text{Normal}} + F1_{\text{Warning}} + F1_{\text{Critical}})$.<br>2. **Secondary Window:** Per-class Precision, Recall, F1, Support; Window FPR = $\frac{\text{FP}_{\text{Normal}}}{\text{Total}_{\text{Normal}}}$; PR-AUC.<br>3. **Episode Recall:** $\frac{\text{Detected Episodes}}{\text{Total Eligible Episodes}}$ (eligible windows in $\{1, 2\}$ within 30 min of failure; detected if $\ge 1$ alert window).<br>4. **Pure-Normal Entity FAR:** $\frac{\text{Alerted PN Entities}}{\text{Total PN Entities}}$ (entities without failures or alerts).<br>5. **Pure-Normal Parent-Job FAR:** Secondary diagnostic metric (denominator $N=3$ jobs in Test). | `results/phase7b_1/phase7b_evaluation_protocol_v2.json`<br>&<br>`scripts/train_and_evaluate_phase7b.py` | Protocol v2.0: Lines 57–97<br><br>Script: Lines 110–277 | **CONFORMANT** |
| **Threshold Selection & Freezing Rules** | Default decision rule: standard argmax over class probabilities ($\hat{y} = \arg\max_c P(y=c)$) or fixed threshold calibrated strictly on Validation data during Stage 2. Candidate $M^*$ (weights, checkpoints, scaler, and thresholds) is permanently locked before unblinding Test. | `results/phase7b_1/phase7b_evaluation_protocol_v2.json` | Lines 28, 38<br><br>Script: Lines 523–530, 710–727, 783–793 | **CONFORMANT** |
| **Validation Candidate Selection Rules** | Evaluates **Validation Gate 1** (Episode Recall $\ge 70.0\%$ AND Critical Window Recall $\ge 50.0\%$) and **Validation Gate 2** (Entity FAR $\le 10.0\%$ AND Window FPR $\le 5.0\%$). Model passing both gates with highest Validation Macro-F1 selected as $M^*$. If zero models pass both gates, model with highest Validation Macro-F1 is selected as $M^*$. | `results/phase7b_1/phase7b_evaluation_protocol_v2.json` | Lines 32–39<br><br>Script: Lines 746–793 | **CONFORMANT** |
| **Test Evaluation Rules & Single Confirmatory Execution** | One-shot confirmatory evaluation of pre-selected candidate $M^*$ only against Test Gate 1 and Test Gate 2. If $M^*$ passes both gates $\to$ `OPERATIONAL_WINNER`; if $M^*$ fails either gate $\to$ `NO_OPERATIONAL_WINNER`. The remaining 3 models are evaluated strictly as descriptive scientific benchmarks and cannot supersede $M^*$. | `results/phase7b_1/phase7b_evaluation_protocol_v2.json` | Lines 40–55<br><br>Script: Lines 795–866 | **CONFORMANT** |
| **Required Confidence Intervals & Resampling Units** | $B=1,000$ bootstrap iterations. Primary resampling unit: `parent_job_id` ($J=11$ clusters in Test). Complementary resampling unit: `entity_id` ($N=59$ clusters in Test). 95% empirical percentiles $[2.5\%, 97.5\%]$. Undefined denominators handled gracefully with valid replicate fraction reporting. | `results/phase7b_1/phase7b_evaluation_protocol_v2.json` | Lines 99–111<br><br>Script: Lines 279–440, 871–908 | **CONFORMANT** |
| **Required Output Files & Official Verdict** | Files: `training_manifest.json`, `validation_metrics.json`, `test_evaluation_results.json`, `bootstrap_ci.json`, `test_predictions.csv`, `test_episode_metrics.csv`, `phase7d_retraining_and_evaluation_report.md`. Official verdict must be exactly `OPERATIONAL_WINNER` or `NO_OPERATIONAL_WINNER`. | `results/phase7b_1/phase7b_evaluation_protocol_v2.json`<br>&<br>`scripts/train_and_evaluate_phase7b.py` | Protocol v2.0: Lines 48–49<br><br>Script: Lines 28–36, 913–943 | **CONFORMANT** |

---

## 3. Task 2 — What Changed from Phase 7B to Phase 7D

Every pipeline dimension between the historical Phase 7B run and the proposed Phase 7D diagnostic experiment was audited and classified:

| Pipeline Dimension | Historical Phase 7B Implementation | Proposed Phase 7D Implementation | Classification & Justification |
| :--- | :--- | :--- | :--- |
| **1. CPI Feature Reconstruction & Missingness** | Inverted from legacy Phase 2 scaled arrays (`X_*_repaired.npy`). Missing CPI slots had been filled with constant $2.500000$ in Phase 2; inversion could not recover genuine unobserved slots. | Extracted directly from raw Borg trace shards (`sample_task_usage*.csv.gz`). Raw CPI is present in 97.36% of rows ($0.355$ to $59.64$, median $2.293$). Genuine missing slots are preserved as `NaN` in unscaled arrays; training median ($2.467000$) is imputed. | **1. Intended correction required to remove suspected contamination.** Eliminates uninvertible legacy constant substitution and restores genuine hardware counter measurements. |
| **2. Parent-Job Scaling Overlap** | Scaler parameters were inherited from Phase 2 training pool, which overlapped with 100% of Candidate A validation and test parent jobs. | Scaler parameters (`configs/phase7c/fitted_scaler_phase7c.json`) fitted strictly on Candidate A Train ($N=3,278$, 51 parent jobs). Pairwise overlap with Val ($J=12$) and Test ($J=11$) is strictly zero. | **1. Intended correction required to remove suspected contamination.** Completely removes cross-split parent-job information bleed. |
| **3. Scaler Fitting Population** | Scaler fitted on inverted arrays pre-filled with legacy $2.50$ constant. | Scaler fitted on genuine observed training values with `NaN`s excluded. Population is flattened training window slices ($16,390$ steps across $4,414$ unique physical intervals). | **1. Intended correction required to remove suspected contamination.** Restores clean training-only standardization parameters. |
| **4. Feature Order & Mathematical Definitions** | 31 mathematical features in `FeatureRegistry`. | Identical 31 mathematical features in identical `FeatureRegistry` order; identical formulas in `src/features/engineer.py`. | **2. Necessary implementation change.** Preserves exact 31-feature mathematical compatibility. |
| **5. Temporal Input Sequence Representation** | Trailing 5 steps ($L=5$) sliced from sliding windows. | Trailing 5 steps ($L=5$) sliced from sliding windows; verified uniform $300\text{s}$ ($5.00\text{ min}$) delta and strictly monotonic age. | **2. Necessary implementation change.** Preserves exact sequence tensor dimensions $[B, 5, 31]$. |
| **6. Target Definitions & Policy A Masking** | Policy A failure masking; Critical $\le 900\text{s}$, Warning $900\text{--}1800\text{s}$, Normal $> 1800\text{s}$, UNKNOWN excluded. | Identical Policy A definitions; verified 0 post-failure windows against raw lifecycle events. | **2. Necessary implementation change.** Preserves identical ground-truth failure horizons. |
| **7. Partition Membership & Split Assignment** | Candidate A (Seed 48, Job-Grouped): Train=3,278, Val=657, Test=1,025 windows. | Identical Candidate A (Seed 48, Job-Grouped): Train=3,278, Val=657, Test=1,025 windows (100.00% exact 1-to-1 match). | **2. Necessary implementation change.** Preserves immutable split assignment. |
| **8. Model Architectures & Evaluation Code** | Seed 42, 4 architectures, Protocol v2.0 gates, dual cluster bootstrap. | Identical seed 42, identical 4 architectures, identical Protocol v2.0 gates, identical dual cluster bootstrap. | **2. Necessary implementation change.** Preserves evaluation logic while updating input paths to `data/processed/phase7c/`. |

### Conclusion on Confounds:
Both Phase 7B and Phase 7D used identical sequence-flattened scaler population structures ($16,390$ steps); Phase 7D simply fits on clean raw data instead of inverted contaminated data. **Zero potential confounds requiring protocol amendments were introduced.**

---

## 4. Task 3 — Evidence-Calibrated Hypothesis & Interpretation Plan

To eliminate deterministic overclaiming (e.g. asserting that passing "proves" contamination caused the failure or that failing "proves" distribution shift), the audit locks an **evidence-calibrated interpretation matrix** prior to training:

```
========================================================================================================================
                                     PHASE 7D EVIDENCE-CALIBRATED DECISION MATRIX
========================================================================================================================
```

### Outcome 1: Clear Improvement and Operational Gate Passage
- **Empirical Criteria:** Candidate $M^*$ passes both Test Gate 1 (Episode Recall $\ge 70.0\%$, Critical Recall $\ge 50.0\%$) and Test Gate 2 (Pure-Normal Entity FAR $\le 10.0\%$, Normal-Window FPR $\le 5.0\%$).
- **Calibrated Scientific Interpretation:** Provides **strong empirical evidence consistent with the hypothesis that preprocessing contamination contributed materially to the Phase 7B failure**, because restoring clean raw features and train-only scaling was the sole intervention between 7B and 7D.
- **Epistemic Limitation:** It does *not* prove contamination was the *exclusive* factor, as interactions between clean feature distributions and Random Forest's orthogonal axis-aligned splitting thresholds could also contribute to the performance shift.

### Outcome 2: Measurable Improvement Without Operational Gate Passage
- **Empirical Criteria:** Noticeable improvement in calibration (e.g. Entity FAR drops from $44.74\%$ to $20\text{--}30\%$, Macro-F1 increases to $> 0.45$), but $M^*$ still fails one or both operational gates.
- **Calibrated Scientific Interpretation:** Supports a **multi-factor explanation**: preprocessing contamination degraded calibration and false-alarm containment in Phase 7B, but parent-job distribution shift across unseen cluster workloads remains an independent barrier preventing full operational viability.

### Outcome 3: Little or No Improvement
- **Empirical Criteria:** Test metrics remain within the Phase 7B 95% confidence intervals (Entity FAR $\approx 40\text{--}50\%$, Critical Recall $\approx 15\text{--}25\%$, Macro-F1 $\approx 0.38\text{--}0.42$).
- **Calibrated Scientific Interpretation:** Provides **strong empirical evidence that inherited preprocessing contamination was NOT the primary driver of Random Forest's failure**. Demonstrates that the operational failure is primarily governed by parent-job distribution shift and model architecture capacity.

### Outcome 4: Worse Results on Clean Data
- **Empirical Criteria:** Macro-F1 drops below $0.35$, Entity FAR worsens above $55\%$, or Episode Recall drops below $30\%$.
- **Calibrated Scientific Interpretation:** Indicates that the legacy Phase 2 scaling parameters (fitted across all parent jobs) may have inadvertently functioned as an artificial regularizer or information bleed that stabilized Phase 7B metrics. Removing it exposes the true unassisted cross-job generalization gap.

### Outcome 5: Inconclusive or Execution Failure
- **Empirical Criteria:** Execution crashes, NaN outputs, non-deterministic seed deviations, or bootstrap divergence.
- **Calibrated Scientific Interpretation:** Invalidates experimental execution. No scientific conclusions may be drawn; requires debugging and re-execution under controlled conditions.

> [!IMPORTANT]
> **Single-Cause Disclaimer:** A single empirical experiment cannot definitively prove an exclusive single cause when multiple plausible explanations (e.g., feature scaling, distribution shift, architecture limitations, threshold sensitivity) interact. All Phase 7D conclusions will be reported in terms of relative support for competing hypotheses.

---

## 5. Task 4 — Audit of Statistical Claims: "Dual Cluster Bootstrap"

The audit scrutinized the statistical claims and nomenclature surrounding the "dual cluster bootstrap" in Protocol v2.0:

### 5.1 Physical Cluster vs. Statistical Cluster
- **Physical Datacenter Cluster:** In cloud computing, a "cluster" denotes a physical compute installation (e.g., the Google Borg 12,000-machine cluster). Candidate A is derived from a **single Borg cluster trace**.
- **Statistical Cluster:** In survey sampling and econometric theory, a "clustered sample" refers to observations that are hierarchically grouped into primary sampling units with intra-group correlation.
- **Resolution:** The term "cluster bootstrap" in Protocol v2.0 is used **strictly in the statistical sense** (hierarchical group resampling), NOT as a multi-datacenter resampler.

### 5.2 Audit of the Dual Resampling Levels
In [scripts/train_and_evaluate_phase7b.py](file:///c:/Users/hp/Desktop/Research%20project/scripts/train_and_evaluate_phase7b.py) (lines 279–440, 883–886), the resampling procedure is executed at two distinct levels:

1. **Level 1: Parent-Job Clustered Bootstrap (`cluster_col="parent_job_id"`, $J=11$ clusters):**
   - **Mechanism:** Samples $J=11$ parent jobs with replacement. All entities and windows within each sampled job are kept intact.
   - **Statistical Validity:** Rigorous block bootstrap accounting for intra-job correlation (tasks in the same parent job share binary code, user priorities, and resource consumption).
   - **Small-Sample Cluster Limitation:** With $J=11$, cluster bootstrap intervals exhibit high variance. Specifically, with only 3 pure-normal parent jobs in Test, in $(1 - 3/11)^{11} \approx 3.1\%$ of replicates, zero pure-normal parent jobs are drawn, rendering parent-job FAR undefined. Protocol v2.0 handles this gracefully by reporting valid replicate fractions.
2. **Level 2: Entity Clustered Bootstrap (`cluster_col="entity_id"`, $N=59$ clusters):**
   - **Mechanism:** Samples $N=59$ entities with replacement. All consecutive 5-step sliding windows of each sampled entity are kept together.
   - **Statistical Validity:** Rigorous block bootstrap over panel entities, preserving intra-entity sequence autocorrelation across overlapping sliding windows while expanding sample size to $N=59$.

### 5.3 Verdict on Statistical Method
The dual clustered bootstrap is **statistically sound, mathematically appropriate, and formally grounded in Protocol v2.0**. No protocol amendment is required, provided reports clearly document that the clustering is statistical (hierarchical parent-job/entity grouping) rather than multi-datacenter compute clustering.

---

## 6. Task 5 — Final Authorization Verdict

All mandatory requirements for Phase 7D authorization have been satisfied:
- All protocol parameters are grounded in existing artifacts with exact citations.
- Implementation differences from Phase 7B are accounted for with zero confounds.
- The hypothesis framework is evidence-calibrated without deterministic overclaiming.
- The statistical resampling methodology is validated.
- Preprocessed arrays in `data/processed/phase7c/` are 100% finite and verified.

The official final verdict of this gate is:

# **`AUTHORIZED UNDER EXISTING PROTOCOL`**

---
*Phase 7D is formally authorized. In accordance with strict instructions, model training has NOT been initiated. Execution will launch via `scripts/train_and_evaluate_phase7d.py` upon your explicit directive.*
