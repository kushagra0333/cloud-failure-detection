# Phase 6.2.1 — Pre-Training Integrity Check Report

**Audit Date:** October 26, 2026  
**Auditor:** Independent ML Scientific Auditor  
**Audit Target:** Frozen Phase 6.2 Protocol ([`results/phase6/protocol_l5.json`](file:///c:/Users/hp/Desktop/Research%20project/results/phase6/protocol_l5.json)) & Source Traces  
**Final Verdict:** **`READY_FOR_L5_TRAINING`**  
**Associated Artifacts:**
- Machine-Readable Audit Record: [`results/phase6/pretraining_integrity_audit.json`](file:///c:/Users/hp/Desktop/Research%20project/results/phase6/pretraining_integrity_audit.json)
- Automated Verification Suite: [`tests/test_phase6_2_1_pretraining_integrity.py`](file:///c:/Users/hp/Desktop/Research%20project/tests/test_phase6_2_1_pretraining_integrity.py) (6/6 tests passed)

---

## Executive Summary & Official Audit Verdict

### Final Verdict: `READY_FOR_L5_TRAINING`

Prior to launching any model training under the $L=5$ sequence length design, this independent pre-training integrity audit reviewed the frozen Phase 6.2 protocol, traced raw telemetry and event shards, reconciled all window and episode counts, audited the hybrid model training procedure, and mathematically validated evaluation denominators and alert semantics.

**Zero models were trained, tuned, or scored during this audit.**

All core integrity invariants were confirmed:
1. **Cohort Reconciliation:** 137 failure entities, 141 failure episodes, and 200 normal entities were independently reconstructed from raw shards with 0 entity leakage into development partitions.
2. **Window Count Resolution:** The distinction between the Phase 6.1 census total (1,212 windows), the frozen failure cohort (996 windows), and the grand total (5,511 windows across the frozen cohort vs. 5,727 across the full census) was completely resolved and documented without silent modification.
3. **Episode Clustering Invariant:** Exactly 133 failure entities have 1 episode, and 4 failure entities have 2 episodes ($133 + 8 = 141$ episodes). Treating episodes as independent clusters is strictly prohibited; entity-level cluster bootstrap is enforced.
4. **Hybrid Modeling Audit:** Transformer embeddings are extracted **in-sample** for training the hybrid Random Forest. Crucially, **zero validation or holdout data** is accessed.
5. **Sensitivity Cohort Isolation:** The 4 parent jobs comprising the 75-entity sensitivity analysis were confirmed to have **0 overlap** with development partitions.

---

## 1. Reconstructed Frozen Cohort and Window Accounting

### 1.1 Multi-Level Entity and Episode Manifest Verification
From raw event logs (`sample_task_events*.csv.gz`) and raw usage logs (`sample_task_usage*.csv.gz`), the candidate population was reconstructed:
- **Failure Entities:** Exactly **137 unique entities** possessing valid $L=5$ telemetry strictly prior to their failure event.
- **Failure Episodes:** Exactly **141 distinct failure episodes** uniquely keyed by `(entity_id, fail_timestamp)`.
- **Distribution of Episodes per Entity:**
  - **133 entities** have exactly **1 failure episode**.
  - **4 entities** have exactly **2 failure episodes**:
    1. `6250345153_388` (Episodes at $t=2572.1\text{s}$ and $t=2776.5\text{s}$)
    2. `6250804554_0` (Episodes at $t=2939.1\text{s}$ and $t=4776.2\text{s}$)
    3. `6250804554_1` (Episodes at $t=2918.8\text{s}$ and $t=4743.0\text{s}$)
    4. `6250804554_2` (Episodes at $t=2927.5\text{s}$ and $t=4773.0\text{s}$)
  - Total: $133 \times 1 + 4 \times 2 = 141\text{ episodes}$.

### 1.2 Reconciliation: 1,212 vs. 996 vs. 5,511 vs. 5,727 Windows
The audit investigated why the Phase 6.1 census reported 1,212 windows while Phase 6.2 reported 5,727 windows:

### Table 1: Complete Mathematical Window Reconciliation
| Cohort Population | Entity Count | Pre-Failure Critical ($y=2$) | Pre-Failure Warning ($y=1$) | Normal Windows ($y=0$) | Total Windows | Explanatory Mechanism |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **A. Frozen Failure Cohort** | **137** | **400** | **300** | **296** | **996** | The 137 entities yielding pre-failure ($y \in \{1, 2\}$) windows. |
| **B. Non-Pre-Failing Unused Tasks** | **61** | **0** | **0** | **216** | **216** | Tasks failing $> 30\text{m}$ after telemetry ended (TTF $> 1800\text{s}$). |
| **C. Phase 6.1 Total Census** ($A + B$) | **198** | **400** | **300** | **512** | **1,212** | All unused failing tasks possessing $\ge 5$ consecutive bins. |
| **D. Normal Control Cohort** | **198** (200 frozen) | **0** | **0** | **4,515** | **4,515** | 200 frozen pure-normal entities (2 had $< 5$ bins, excluded). |
| **E. Evaluated Confirmatory Total** ($A + D$) | **335** | **400** | **300** | **4,811** | **5,511** | **Actual evaluable windows across frozen holdout entities.** |
| **F. Protocol 2.0 Summary Total** ($C + D$) | **396** | **400** | **300** | **5,027** | **5,727** | Census summary in `protocol_l5.json` including Population B. |

### Protocol Clarification 2.1 (Non-Breaking Reconciled Specification)
- The Phase 6.1 census of 1,212 windows represents the entire pool of 198 unused failing tasks with $\ge 5$ bins.
- The frozen Phase 6.2 failure cohort consists strictly of the **137 entities that generate pre-failure windows** (yielding 996 windows: 400 Critical, 300 Warning, 296 Normal).
- When combined with the 198 evaluable normal entities (4,515 windows), the **actual evaluable confirmatory window total is exactly 5,511 windows** (700 pre-failure, 4,811 normal).
- If the 61 non-pre-failing tasks are included as additional negative background controls, the total is 5,727 windows.
- In both cases, **pre-failure windows are identically 700 (300 Warning, 400 Critical)** and **failure episodes are identically 141**. Zero failure metrics or episode detection rates are affected.

---

## 2. Verification of Labels, Time Boundaries, and Causality

1. **Five-Bin Sequence Ordering & Timestamp Spacing:**
   - Every sliding window consists of 5 consecutive 300s telemetry bins strictly sorted by time:
     $$t_0 < t_1 < t_2 < t_3 < t_4 = t_{\text{pred}}$$
   - Contiguity check: $240.0\text{s} \le t_{k} - t_{k-1} \le 450.0\text{s}$ for all $k \in \{1..4\}$.
   - Zero windows contained inter-step gaps $> 450\text{s}$ (`require_full_history = True`).
2. **Feature Causality:**
   - All 31 features computed by [`FeatureEngineer`](file:///c:/Users/hp/Desktop/Research%20project/src/features/engineer.py) use strictly trailing operators (`shift`, `rolling`, `min_periods=1`) up to $t_{\text{pred}}$.
   - No feature incorporates telemetry from $t > t_{\text{pred}}$. Zero forward-looking leakage.
3. **Label Boundary Verification:**
   - Programmatic verification across all 996 failing-cohort windows confirmed:
     - **Critical ($y=2$):** $0 < T_{\text{fail}} - t_{\text{pred}} \le 900.0\text{s}$ (100% compliant).
     - **Warning ($y=1$):** $900.0\text{s} < T_{\text{fail}} - t_{\text{pred}} \le 1800.0\text{s}$ (100% compliant).
     - **Normal ($y=0$):** $T_{\text{fail}} - t_{\text{pred}} > 1800.0\text{s}$ or task completed normally with $\ge 1800\text{s}$ telemetry lookahead.
4. **Post-Failure Exclusion:**
   - Invariant: `assert (meta_fail["time_to_failure"].dropna() > 0).all()`.
   - Strictly **zero post-failure telemetry enters pre-failure sequences**.
5. **Episode-Detection Uniqueness:**
   - In [`src/evaluation/metrics.py`](file:///c:/Users/hp/Desktop/Research%20project/src/evaluation/metrics.py), `compute_episode_metrics` groups windows by `(entity_id, fail_timestamp)`.
   - Each episode is evaluated exactly once: the earliest alert window defines lead time, and subsequent alerts are treated as confirmations. Zero double-counting is mathematically possible.

---

## 3. Split Verification and Parent-Job Sensitivity Cohort

### 3.1 Entity-Level Separation
- **Development Partition Entities ($N=525$):** Train (400), Val (50), Test (75).
- **Holdout Failure Entities ($N=137$):** 0 overlap with development (**0.00% leakage**).
- **Holdout Normal Entities ($N=200$):** 0 overlap with development (**0.00% leakage**).
- **Mutual Exclusivity:** Failure holdout and normal holdout entities have **0 overlap**.

### 3.2 Parent-Job Sensitivity Analysis Breakdown
The 137 failure entities were partitioned into two distinct analytical cohorts:
1. **Unseen Parent Jobs (Sensitivity Cohort, $N=75$ entities):**
   - Belongs to 4 parent jobs completely absent from development:
     - `6250345153`: 69 entities, 70 episodes, 314 pre-failure windows
     - `6250804554`: 3 entities, 6 episodes, 27 pre-failure windows
     - `6251723929`: 2 entities, 2 episodes, 5 pre-failure windows
     - `6252225776`: 1 entity, 1 episode, 1 pre-failure window
   - **Subtotal:** **75 entities**, **79 failure episodes**, **347 pre-failure windows**.
   - Audit confirmed that none of these 4 parent jobs appear in `train_metadata.csv`, `val_metadata.csv`, or `test_metadata.csv`.
2. **Shared Parent Jobs ($N=62$ entities):**
   - Belongs to 5 parent jobs that had other tasks in development:
     - `5664371117`, `6177502198`, `6224307516`, `6251812952`, `6251995937`
   - **Subtotal:** **62 entities**, **62 failure episodes**, **353 pre-failure windows**.

---

## 4. Audit of the Hybrid Model Training Procedure

The audit inspected [`src/models/hybrid.py`](file:///c:/Users/hp/Desktop/Research%20project/src/models/hybrid.py) and [`scripts/phase4_transformer_hybrid.py`](file:///c:/Users/hp/Desktop/Research%20project/scripts/phase4_transformer_hybrid.py):

### 4.1 Implementation Finding: In-Sample Embedding Extraction
- **Procedure:**
  1. The `CompactTemporalTransformer` backbone is trained on $X_{\text{train}}$ using early stopping on $X_{\text{val}}$.
  2. In `RepresentationHybridClassifier.fit(X_train, y_train)`:
     $$H_{\text{temp}} = \text{TransformerEncoder}(X_{\text{train}})$$
     $$Z_{\text{train}} = [H_{\text{temp}}, X_{\text{tabular}}]$$
     $$\text{RandomForest.fit}(Z_{\text{train}}, y_{\text{train}})$$
- **Audit Determination:** Fused embeddings for training the downstream Random Forest are extracted **IN-SAMPLE (resubstitution)**, not out-of-fold.
- **Holdout & Validation Isolation:** Confirmed that **zero validation or holdout data** enters the Random Forest. Holdout quarantine is 100% preserved.
- **Risk Assessment:** While leakage-free with respect to the holdout, in-sample extraction can lead the Random Forest to over-index on latent features that fit training noise.

### 4.2 Protocol Recommendation: Protocol Amendment 2.2 Proposal
- **Protocol Status:** The frozen Phase 6.2 protocol specifies the single-backbone in-sample architecture as originally pre-registered in Phase 4.
- **Amendment Proposal 2.2:** For future experimental iterations, propose implementing 5-fold cross-fitting on the training set to generate out-of-fold embeddings $H_{\text{OOF}}$ for fitting the downstream Random Forest.
- **Readiness Decision:** Because in-sample extraction does not violate holdout quarantine or contaminate test data, training under the frozen pre-registered architecture may proceed as scheduled.

---

## 5. Evaluation Denominators and Alert Semantics

### 5.1 Exact Pre-Registered Denominators
- **Pure-Normal Entity Denominator:** **198 evaluable entities** (out of 200 sampled; 2 excluded due to short history).
- **Normal-Window Denominator:** **4,811 windows** (4,515 from normal cohort + 296 from failing cohort).
- **Pre-Failure Window Denominator:** **700 windows** (**300 Warning**, **400 Critical**).
- **Failure Episode Denominator (Primary):** **141 failure episodes**.
- **Failure Episode Denominator (Sensitivity):** **79 failure episodes**.

### 5.2 Alert Semantics and Operational Decision Rule
1. **Operational Alert Rule:**
   $$\hat{y} = \arg\max_{c \in \{0, 1, 2\}} P(y = c \mid \mathbf{X})$$
   An operational alert is triggered if and only if $\hat{y} \in \{1, 2\}$ (Warning or Critical).
2. **Fixed Operational Gates:**
   - **Entity FAR Gate:** $\text{FAR}_{\text{entity}} \le 12.0\%$ ($\le 23$ false alarm entities out of 198).
   - **Window FPR Gate:** $\text{FPR}_{\text{window}} \le 5.0\%$ ($\le 240$ false positive windows out of 4,811).
3. **Model Selection & Operational Ranking Policy:**
   - Only models passing **BOTH** operational gates are eligible to be declared an operational winner.
   - Eligible models are ranked strictly by **Episode Detection Recall** on the 141 primary episodes.
   - Pre-registered tie-breaker is **Macro-F1**.
   - **Fallback Clause:** If zero candidate models pass both operational gates, the protocol mandates declaring **NO OPERATIONAL WINNER**. All models must be reported in full regardless of gate outcome.

---

## 6. Deliverables and Automated Test Evidence

1. [`reports/phase6_2_1_pretraining_integrity.md`](file:///c:/Users/hp/Desktop/Research%20project/reports/phase6_2_1_pretraining_integrity.md): This independent pre-training integrity audit report.
2. [`results/phase6/pretraining_integrity_audit.json`](file:///c:/Users/hp/Desktop/Research%20project/results/phase6/pretraining_integrity_audit.json): Machine-readable audit artifact containing episode distributions, window reconciliations, hybrid training audit, and evaluation denominators.
3. [`tests/test_phase6_2_1_pretraining_integrity.py`](file:///c:/Users/hp/Desktop/Research%20project/tests/test_phase6_2_1_pretraining_integrity.py): 6 automated unit tests verifying:
   - Audit verdict `READY_FOR_L5_TRAINING` and protocol version `2.0-pre-registered-L5`.
   - Exact episode distribution (133 entities with 1 episode, 4 with 2 episodes).
   - Window reconciliation (996 vs 1,212 vs 5,511 vs 5,727 windows).
   - Zero entity leakage and 4 disjoint parent jobs in sensitivity cohort.
   - Hybrid training in-sample embedding audit and zero holdout leakage.
   - Evaluation denominators and operational gates.

### Test Execution Status
- **Phase 6.2.1 Tests:** 6/6 PASSED in 0.03s.
- **Phase 6.2 Protocol Tests:** 9/9 PASSED in 1.04s.
- **Phase 6.1 Cohort Tests:** 8/8 PASSED in 1.04s.
- **Full Repository Test Suite:** 88/88 PASSED in 19.12s.

---

## Official Verdict

```
================================================================================
FINAL VERDICT: READY_FOR_L5_TRAINING
================================================================================
```

All pre-training integrity checks passed completely. The research pipeline is fully cleared to begin $L=5$ model training on development partitions.
