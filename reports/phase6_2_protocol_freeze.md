# Phase 6.2 — Final Protocol Freeze and L=5 Training Readiness Audit

**Audit Date:** October 26, 2026  
**Auditor:** Independent ML Scientific Auditor  
**Protocol Status:** **`FROZEN_PRE_REGISTERED_UNTOUCHED`**  
**Readiness Verdict:** **`READY FOR L=5 MODEL TRAINING`**  
**Deliverables:**
- Protocol Specification Report: [`reports/phase6_2_protocol_freeze.md`](file:///c:/Users/hp/Desktop/Research%20project/reports/phase6_2_protocol_freeze.md)
- Machine-Readable Protocol Artifact: [`results/phase6/protocol_l5.json`](file:///c:/Users/hp/Desktop/Research%20project/results/phase6/protocol_l5.json)
- Automated Verification Suite: [`tests/test_phase6_2_protocol_freeze.py`](file:///c:/Users/hp/Desktop/Research%20project/tests/test_phase6_2_protocol_freeze.py) (9/9 tests passed)

---

## Executive Summary & Official Audit Verdict

### Final Readiness Verdict: `READY FOR L=5 MODEL TRAINING`

Following the Phase 6.1 independent verification that confirmed the feasibility of the $L=5$ sequence length design on Google Borg Trace 2011, this audit formally freezes the experimental protocol, reconciles cohort definitions, audits training readiness, and specifies pre-registered analysis plans.

**Zero models were trained, tuned, or scored during Phase 6.2.** The sole purpose of this phase is to establish strict pre-registration barriers before training new models under the $L=5$ specification.

### Verification of Methodological Prohibitions & Safeguards
| Methodological Safeguard / Prohibition | Compliance Status | Independent Audit Verification Evidence |
| :--- | :---: | :--- |
| **No Holdout Scoring** | **COMPLIANT** | Zero candidate models were evaluated or scored on holdout entities. |
| **No Holdout Threshold Tuning** | **COMPLIANT** | Decision thresholds remain fixed at argmax defaults; zero tuning performed. |
| **Phase 5 Artifacts Unmodified** | **COMPLIANT** | [`results/phase5/confirmatory_metrics.json`](file:///c:/Users/hp/Desktop/Research%20project/results/phase5/confirmatory_metrics.json) remains intact and quarantined. |
| **No Episode Independence Assumption** | **COMPLIANT** | Protocol explicitly forbids treating 141 episodes as independent clusters. Clustering is strictly at `entity_id` level. |
| **No Reuse of L=15 Checkpoints** | **COMPLIANT** | Reusing legacy $L=15$ weights is formally prohibited; new models must be trained from scratch on $L=5$. |
| **Passing Tests != Scientific Proof** | **COMPLIANT** | Methodological validity is established via raw trace provenance, causal feature invariants, and frozen isolation. |

---

## 1. Cohort Reconciliation & Freezing

### 1.1 Reconciling the 100/200 Proposal vs. 137 Eligible Failure Entities
In the initial Phase 6 exploratory concept ([`reports/phase6_proposed_confirmatory_protocol.md`](file:///c:/Users/hp/Desktop/Research%20project/reports/phase6_proposed_confirmatory_protocol.md)), a tentative cohort of 100 failure entities and 200 normal entities was proposed based on preliminary estimates.

However, the independent full-trace census in Phase 6.1 proved that the entire Google Borg trace contains **exactly 137 eligible failing entities** possessing valid pre-failure telemetry under $L=5$ outside development splits.

### 1.2 Scientific Decision: Use ALL 137 Eligible Failure Entities
The protocol formally mandates using **all 137 eligible failure entities (100% of the available failure population)** rather than an arbitrary 100-entity subsample:
1. **Preserving Statistical Power:** Discarding 37 verified failure entities (27.0% of available cases) and ~35 failure episodes without scientific rationale arbitrarily widens uncertainty intervals and degrades detection lead-time estimates.
2. **Preventing Subsampling Bias:** Using the complete census eliminates sampling variance and prevents any post-hoc suspicion of cherry-picking failure cases.
3. **Preserving the Cross-Job Sensitivity Cohort:** Exactly 75 failure entities belong to the 4 parent jobs completely absent from development. Subsampling down to 100 would fragment this critical subset.

### 1.3 Normal Cohort Pairing & Lookahead Coverage Verification
The 137 failure entities are paired with the **200 frozen normal entities** pre-registered from the 238,171 pure-normal pool (seed 42):
- **Telemetry Verification:** All 200 normal entities possess verified telemetry in `sample_task_usage*.csv.gz` across 109 distinct parent jobs.
- **Unpadded $L=5$ Window Yield:** 198 normal entities possess $\ge 5$ consecutive 300s bins, generating **4,515 valid sliding windows**. Exactly 2 entities had $< 5$ bins and are excluded before scoring.
- **Complete Horizon Coverage:** 100% of these 4,515 windows possess complete 30-minute lookahead observation coverage or normal task completion (`FINISH = 4`). Zero steps are mislabeled as Normal (all 286 censored steps at trace end are classified as label `-1` and excluded).

### Table 1: End-to-End Hierarchical Cohort Accounting
| Level | Metric / Entity Category | Total Holdout | Disjoint Parent Jobs (Sensitivity) | Shared Parent Jobs |
| :--- | :--- | :---: | :---: | :---: |
| **Parent Jobs** | Unique Failure Parent Jobs | 9 | 4 (`6250345153`, `6250804554`, `6251723929`, `6252225776`) | 5 (`5664371117`, `6177502198`, `6224307516`, `6251812952`, `6251995937`) |
| | Unique Normal Parent Jobs | 109 | N/A | N/A |
| | **Total Parent Jobs** | **118** | **113** | **5** |
| **Entities** | Failure Entities | 137 | 75 | 62 |
| | Evaluable Normal Entities | 198 | N/A | N/A |
| | Excluded Normal Entities ($< 5$ steps) | 2 | N/A | N/A |
| | **Total Evaluated Entities** | **335** (337 frozen) | **273** | **62** |
| **Episodes** | Evaluated Failure Episodes | **141** | **79** | **62** |
| **Windows** | Pre-Failure Warning ($y=1$) | 300 | 147 | 153 |
| | Pre-Failure Critical ($y=2$) | 400 | 200 | 200 |
| | **Total Pre-Failure Windows** | **700** | **347** | **353** |
| | Normal Windows (Failing Cohort) | 512 | 266 | 246 |
| | Normal Windows (Normal Cohort) | 4,515 | N/A | N/A |
| | **Total Evaluable Windows** | **5,727** | **5,128** | **599** |

All entity IDs, episode metadata, and parent-job mappings are frozen in [`results/phase6/protocol_l5.json`](file:///c:/Users/hp/Desktop/Research%20project/results/phase6/protocol_l5.json).

---

## 2. L=5 Training and Modeling Readiness Audit

### 2.1 Formal Designation as a New Experimental Design
$L=5$ is a **new experimental design**, not a continuation or parameter tweak of Phase 5. Phase 5 evaluated a 75-minute historical dependency assumption ($L=15$) and proved it empirically unviable for holdout failure tasks. Phase 6.2 formally pre-registers a 20-minute endpoint / 25-minute collection span design ($L=5$).

### 2.2 Mathematical Audit of Model Architectures & Input Shapes
| Model Architecture | Input Tensor Dimensions | Feature Representation & Encoding | Training Loss / Objective |
| :--- | :---: | :--- | :--- |
| **Random Forest Baseline** | `[batch_size, 31]` | Causal trailing feature vector at $t_{\text{pred}}$ from Groups A–I. | Gini Impurity, 100 trees, balanced weights, max depth 15. |
| **Temporal LSTM** | `[batch_size, 5, 31]` | 5 consecutive 300s steps ($L=5$, $D=31$). 2 LSTM layers, hidden size 64, dropout 0.2. | Class-weighted CrossEntropyLoss ($w = [1.0, 1.5, 2.0]$). |
| **Compact Temporal Transformer** | `[batch_size, 5, 31]` | 5 consecutive 300s steps. Linear projection to $d=32$, max_len=5 sinusoidal PE, 2 layers, 4 heads, $d_{\text{ff}}=64$. | Class-weighted CrossEntropyLoss ($w = [1.0, 1.5, 2.0]$). |
| **Representation Hybrid** | `[batch_size, 63]` | 32-dim latent sequence embedding from frozen Transformer backbone concatenated with 31-dim tabular vector. | Random Forest classifier fitted on 63-dim fused embeddings. |

### 2.3 Strict Prohibition on Reusing Legacy L=15 Checkpoints
> [!CAUTION]
> **Reusing legacy checkpoints (`best_lstm_checkpoint.pt`, `best_transformer_checkpoint.pt`) is strictly prohibited.**
> 
> *Mathematical Rationale:* Legacy checkpoints were trained with temporal positional encodings and recurrent cell dynamics parameterized for $T=15$ steps (70 minutes). Injecting $L=5$ inputs into an $L=15$ Transformer model produces out-of-distribution self-attention patterns and corrupts temporal contextualization. Completely new models must be trained from scratch on $L=5$ training tensors.

### 2.4 Development Partition Characterization & Quarantine
- **Development Splits:** The 525 development entities (`train_metadata.csv`=400, `val_metadata.csv`=50, `test_metadata.csv`=75) were used extensively during Phases 2–4 for baseline tuning and architecture exploration.
- **Exploratory Status:** The validation set is an exploratory optimization benchmark for early stopping and checkpoint selection. It must never be presented as an untouched confirmatory benchmark.
- **Strict Quarantine:** The confirmatory holdout ($N=137$ failure, $N=200$ normal) shall remain 100% quarantined until model checkpoints, preprocessing scalers, and decision thresholds are permanently frozen.
- **Preprocessing Freeze:** Standardization and median imputation are frozen in [`configs/fitted_scaler.json`](file:///c:/Users/hp/Desktop/Research%20project/configs/fitted_scaler.json), fitted strictly on the 400 training tasks. Holdout evaluation operates strictly in **transform-only mode**.

---

## 3. Primary and Sensitivity Analyses Specification

### 3.1 Primary Analysis: Pre-Registered Full Cohort Evaluation
- **Population:** All 137 failure entities (141 episodes, 700 pre-failure windows) + 200 normal entities (4,515 normal windows).
- **Cluster Bootstrap Specification:**
  - Resampling unit: `entity_id` ($N=337$ clusters: 137 failure entities, 200 normal entities).
  - Paired bootstrap iterations: $B = 1,000$ resamples with replacement.
  - Confidence interval: Empirical 2.5th and 97.5th percentiles (95% CI).
  - **Non-Independence Prohibition:** Treating 141 episodes as 141 independent entity clusters is **strictly prohibited**. Multiple failure episodes belonging to the same entity (e.g. within job `6250345153` or `6250804554`) must be resampled together as a cluster to preserve within-entity correlation.

### 3.2 Sensitivity Analysis: Cross-Job Generalization
- **Population:** The 75 failure entities (79 episodes, 347 pre-failure windows) belonging to the 4 parent jobs completely absent from development partitions (`6250345153`, `6250804554`, `6251723929`, `6252225776`) + 200 normal entities.
- **Purpose:** Rigorously tests application-level generalizability. Evaluates whether failure detection mechanisms reflect universal workload degradation dynamics rather than memorization of job-specific execution patterns.

### 3.3 Metric Separation & Pre-Registered Operational Decision Rule
1. **Metric Separation:**
   - **Window False Positive Rate ($\text{FPR}_{\text{window}}$):** Computed over all 5,027 true normal windows:
     $$\text{FPR}_{\text{window}} = \frac{\sum_{i=1}^{5027} \mathbb{I}(\hat{y}_i \in \{1, 2\})}{5027}$$
   - **Entity False Alarm Rate ($\text{FAR}_{\text{entity}}$):** Computed across the 198 evaluable pure-normal entities:
     $$\text{FAR}_{\text{entity}} = \frac{\sum_{e=1}^{198} \mathbb{I}(\exists t : \hat{y}_{e, t} \in \{1, 2\})}{198}$$
2. **Fixed Operational Gates:**
   - **Entity FAR Gate:** $\text{FAR}_{\text{entity}} \le 12.0\%$
   - **Window FPR Gate:** $\text{FPR}_{\text{window}} \le 5.0\%$
3. **Model Selection & Operational Ranking Policy:**
   - **Eligibility:** Only models satisfying **BOTH** operational gates are eligible to be declared an operational winner.
   - **Ranking Metric:** Eligible models are ranked strictly by **Episode Detection Recall** on the primary cohort.
   - **Tie-Breaker:** Pre-registered tie-breaker is **Macro-F1** across all three classes.
   - **Default Fallback Clause:** If zero candidate models pass both operational gates, the protocol formally mandates declaring **NO OPERATIONAL WINNER**. All models must be reported in full regardless of gate outcome.

---

## 4. Deliverables and Automated Test Evidence

1. [`reports/phase6_2_protocol_freeze.md`](file:///c:/Users/hp/Desktop/Research%20project/reports/phase6_2_protocol_freeze.md): This comprehensive protocol specification report.
2. [`results/phase6/protocol_l5.json`](file:///c:/Users/hp/Desktop/Research%20project/results/phase6/protocol_l5.json): The immutable, machine-readable protocol freezing all 137 failure entities, 200 normal entities, 141 episodes, parent jobs, model architectures, and operational gates.
3. [`tests/test_phase6_2_protocol_freeze.py`](file:///c:/Users/hp/Desktop/Research%20project/tests/test_phase6_2_protocol_freeze.py): 9 automated unit tests verifying:
   - Protocol artifact existence, schema, and `READY FOR L=5 MODEL TRAINING` verdict.
   - Methodological safeguards and prohibitions (no scoring, no tuning, no checkpoint reuse).
   - Cohort counts across entity, parent-job, episode, and window levels.
   - Zero entity overlap between holdout and development partitions.
   - Parent job decomposition (75 disjoint vs 62 shared entities).
   - $L=5$ sequence length and target horizon definitions.
   - Training readiness and model input shapes.
   - Operational gates (FAR $\le 12\%$, FPR $\le 5\%$) and ranking policies.
   - Entity-level cluster bootstrap specification and prohibition of episode independence.

### Test Execution Status
- **Phase 6.2 Unit Tests:** 9/9 PASSED in 1.04s.
- **Phase 6.1 Cohort Tests:** 8/8 PASSED in 1.04s.
- **Full Repository Test Suite:** 73/73 PASSED in 18.97s.

---

## Official Verdict

```
================================================================================
FINAL VERDICT: READY FOR L=5 MODEL TRAINING
================================================================================
```

The experimental protocol is completely frozen, mathematically reconciled, and leakage-safe. The research pipeline is fully prepared to proceed to Phase 6.3 ($L=5$ model training on development data) without risk of confirmatory contamination.
