# Phase 6.2.3 — Formal Ratification of Protocol Amendment 2.1

**Evaluation State:** Confirmatory Protocol Formulation (Pre-Training Gate)  
**Ratification Target:** Protocol Amendment 2.1 ([`results/phase6/protocol_l5_amendment_2_1.json`](file:///c:/Users/hp/Desktop/Research%20project/results/phase6/protocol_l5_amendment_2_1.json))  
**Immutable Historical Baseline:** Phase 6.2 Protocol ([`results/phase6/protocol_l5.json`](file:///c:/Users/hp/Desktop/Research%20project/results/phase6/protocol_l5.json))  
**Final Scientific Verdict:** `AMENDMENT_RATIFIED_READY_FOR_L5_TRAINING`

---

## 1. Executive Summary & Purpose

The Phase 6.2.2 Protocol Consistency Audit uncovered a mathematical divergence within the original Phase 6.2 protocol artifact ([`results/phase6/protocol_l5.json`](file:///c:/Users/hp/Desktop/Research%20project/results/phase6/protocol_l5.json)). While the machine-readable entity manifests locked down 137 failure entities and 200 normal entities (physically producing **5,511 sliding windows** and **4,811 normal windows**), the embedded summary table cited **5,727 total windows** and **5,027 normal windows**, which originated from an expanded 198-task trace census that included 61 non-pre-failing tasks.

Because this numerical discrepancy shifts the pre-registered **Window False Positive Rate (FPR) operational gate denominator by 216 normal windows** (and alters the allowable false-positive threshold from 240 to 251 windows), model training could not ethically or scientifically proceed without a formal, versioned protocol amendment.

This report documents the formal ratification of **Protocol Amendment 2.1** ([`results/phase6/protocol_l5_amendment_2_1.json`](file:///c:/Users/hp/Desktop/Research%20project/results/phase6/protocol_l5_amendment_2_1.json)). Amendment 2.1 locks the exact primary confirmatory population, denominators, operational thresholds, disaggregated reporting requirements, and downstream evaluation code prior to any model training.

```
       ========================================================================
                         PHASE 6.2.3 RATIFICATION STATUS
       ========================================================================
       Original Protocol 2.0 (protocol_l5.json) : PRESERVED UNCHANGED (IMMUTABLE)
       Ratified Protocol Amendment 2.1          : RATIFIED_PRE_REGISTERED_FROZEN
       Primary Confirmatory Population          : EXACTLY 5,511 WINDOWS
       Primary Normal Window Denominator        : EXACTLY 4,811 WINDOWS
       Pure-Normal Entity Denominator           : EXACTLY 198 ENTITIES
       Excluded Census Tasks (Quarantined)      : EXACTLY 61 TASKS (216 WINDOWS)
       Entity FAR Gate (<= 12.0%)               : MAX 23 FALSE ALARM ENTITIES
       Window FPR Gate (<= 5.0%)                : MAX 240 FALSE POSITIVE WINDOWS
       Downstream Code Integration              : FORMALLY DIRECTED & TESTED
       Final Verdict                            : AMENDMENT_RATIFIED_READY_FOR_L5_TRAINING
       ========================================================================
```

---

## 2. Safeguards Compliance Verification

All mandatory non-negotiable research safeguards were strictly enforced during Phase 6.2.3:

| Constraint / Safeguard | Protocol Mandate | Implementation & Proof | Status |
| :--- | :--- | :--- | :---: |
| **No Model Training** | Zero weights fitted or updated | No models trained or checkpoints generated in this phase | **COMPLIANT** |
| **No Holdout Scoring** | Zero inference on holdout data | Confirmatory candidate data was neither loaded nor evaluated by models | **COMPLIANT** |
| **No Threshold Tuning** | Zero inspection or optimization of decision cutoffs | Uniform argmax rule and operational gates locked pre-hoc | **COMPLIANT** |
| **Phase 5 Artifacts Unmodified** | Quarantine historical Phase 5 results | All artifacts in [`results/phase5/`](file:///c:/Users/hp/Desktop/Research%20project/results/phase5/) remain untouched | **COMPLIANT** |
| **Original Protocol Preserved** | Preserve `protocol_l5.json` as historical record | SHA-256 verified invariant: `ebf692598f2cb7115919c427127782f44ba054c655b3b76100aadb5955377bd1` | **COMPLIANT** |
| **No Cohort Alterations** | Cohort membership & labels unchanged | Primary 137 failure entities, 200 normal entities, and 141 episodes strictly identical | **COMPLIANT** |
| **Entity-Level Resampling** | No episode-level independence assumption | Cluster bootstrap is pre-registered strictly at entity cluster level ($N=337$) | **COMPLIANT** |

---

## 3. Mathematical Reconciliation of Cohorts and Denominators

The mathematical relationship between the primary confirmatory manifest and the expanded trace census is formally resolved and locked as follows:

| Population Component | Entity Count | Critical ($y=2$) | Warning ($y=1$) | Normal ($y=0$) | Total Windows | Role in Protocol Amendment 2.1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **A. Primary Failure Entities** | **137** | **400** | **300** | **296** | **996** | Primary confirmatory failure cohort (100% of eligible failure cases) |
| **B. Evaluable Pure-Normal Entities** | **198** | **0** | **0** | **4,515** | **4,515** | Primary pure-normal control cohort ($\ge 5$ bins, label 0) |
| **C. Primary Evaluation Total ($A + B$)** | **335** | **400** | **300** | **4,811** | **5,511** | **SOLE CONFIRMATORY POPULATION FOR GATES & METRICS** |
| **D. Excluded Census Failing Tasks** | **61** | **0** | **0** | **216** | **216** | Quarantined descriptive census; zero entry into evaluation manifest |
| **E. Excluded Short-History Normal Tasks**| **2** | **0** | **0** | **0** | **0** | Excluded prior to scoring due to short history ($< 5$ bins) |
| **F. Grand Trace Census ($C + D + E$)** | **398** | **400** | **300** | **5,027** | **5,727** | Historical exploratory census in `protocol_l5.json` summary block |

### The Forensic Mechanism Behind the 61 Excluded Tasks
In Borg Trace 2011, 198 unused failing tasks possessed at least 5 consecutive telemetry steps before failure. However:
1. **137 tasks** experienced their failure within 30 minutes of telemetry, generating qualifying Warning ($900\text{s} < \tau \le 1800\text{s}$) or Critical ($0 < \tau \le 900\text{s}$) windows (700 pre-failure windows in total).
2. **61 tasks** had failures that occurred more than 30 minutes after telemetry terminated, or were late-failing executions where all valid $L=5$ observation windows had $\tau > 1800\text{s}$. As a result, they produced **216 Normal windows and exactly 0 pre-failure windows**.
3. Relegating these 61 tasks to an excluded descriptive census ensures that the primary failure cohort consists exclusively of entities that provide evaluable early failure prediction targets.
4. **Denominators are now sealed:** Primary normal windows $= 4,811$ ($4,515 + 296$); primary total windows $= 5,511$.

---

## 4. The 10 Ratified Locked Tenets of Protocol Amendment 2.1

Protocol Amendment 2.1 ([`results/phase6/protocol_l5_amendment_2_1.json`](file:///c:/Users/hp/Desktop/Research%20project/results/phase6/protocol_l5_amendment_2_1.json)) formally codifies the following 10 pre-registered tenets:

### Tenet 1: Primary Cohort Definition
- **Failure Cohort:** Exactly 137 failure entities representing 100% of available, eligible failure entities in Borg Trace 2011 with qualifying pre-failure telemetry under unpadded $L=5$.
- **Normal Control Cohort:** Exactly 200 deterministically sampled pure-normal entities, of which exactly 198 possess evaluable $L=5$ histories ($\ge 5$ bins).
- **Excluded Short-History Normal Entities:** Exactly 2 entities (`4974636262_330` and `4974636615_100`) have $< 5$ bins and are excluded prior to scoring.
- **Total Primary Cohort:** 337 sampled entities, 335 evaluable entities (Case-Control ratio $\approx 1:1.46$).

### Tenet 2: Primary Evaluation Window Denominators
- **Total Primary Windows:** Exactly **5,511 sliding windows**.
- **Critical Windows ($y=2$):** Exactly **400 windows** ($0 < \tau \le 900\text{s}$, prevalence $= 7.258\%$).
- **Warning Windows ($y=1$):** Exactly **300 windows** ($900\text{s} < \tau \le 1800\text{s}$, prevalence $= 5.444\%$).
- **Total Pre-Failure Windows:** Exactly **700 windows** (prevalence $= 12.702\%$).
- **Normal Windows ($y=0$):** Exactly **4,811 windows** (prevalence $= 87.298\%$).

### Tenet 3: Disaggregated Normal Window Composition
- **Pure-Normal Cohort Windows:** Exactly **4,515 windows** from the 198 evaluable pure-normal entities (100% verified label 0).
- **Failure-Cohort Normal Windows:** Exactly **296 windows** from the 137 failure entities where lookahead $\tau > 1800\text{s}$.
- **Sum:** $4,515 + 296 = 4,811$ primary normal windows.

### Tenet 4: Quarantine of Excluded Descriptive Census
- **Excluded Census Tasks:** Exactly 61 tasks (`scratch/excluded_61_failing_tasks.json`).
- **Census Normal Windows:** Exactly 216 windows (all label 0).
- **Census Pre-Failure Windows:** Exactly 0 windows.
- **Mandate:** These 61 tasks are strictly quarantined from the primary evaluation manifest. They cannot enter the evaluation dataset, cannot alter the FPR denominator, and cannot influence model selection. They may only be reported in an ancillary, non-gating descriptive appendix.

### Tenet 5: Locked Entity False Alarm Rate (FAR) Operational Gate
- **Evaluation Population:** Evaluated strictly across the **198 evaluable pure-normal entities**.
- **Formula:** $\text{FAR}_{\text{entity}} = \frac{\text{Count of pure-normal entities triggering } \ge 1 \text{ alert}}{198} \times 100\%$.
- **Operational Gate:** $\text{FAR}_{\text{entity}} \le 12.0\%$.
- **Mathematical Bound:** $198 \times 0.120 = 23.76 \implies$ **At most 23 false alarm entities are permitted** ($23 / 198 = 11.616\% \le 12.0\%$, whereas $24 / 198 = 12.121\% > 12.0\%$).

### Tenet 6: Locked Window False Positive Rate (FPR) Operational Gate
- **Evaluation Population:** Evaluated strictly across all **4,811 primary normal windows**.
- **Formula:** $\text{FPR}_{\text{primary}} = \frac{\text{False Positive Windows on Primary Normal Windows}}{4,811} \times 100\%$.
- **Operational Gate:** $\text{FPR}_{\text{primary}} \le 5.0\%$.
- **Mathematical Bound:** $4,811 \times 0.050 = 240.55 \implies$ **At most 240 false positive windows are permitted** ($240 / 4,811 = 4.9886\% \le 5.0\%$, whereas $241 / 4,811 = 5.0094\% > 5.0\%$).

### Tenet 7: Disaggregated False Positive Reporting Mandate
To ensure transparency between non-failing workload behavior and early pre-failure telemetry dynamics, evaluation reports must report:
1. $\text{FPR}_{\text{pure\_normal}} = \frac{\text{FP}_{\text{pure\_normal}}}{4,515} \times 100\%$
2. $\text{FPR}_{\text{failing\_normal}} = \frac{\text{FP}_{\text{failing\_normal}}}{296} \times 100\%$
3. $\text{FPR}_{\text{pooled\_primary}} = \frac{\text{FP}_{\text{pure\_normal}} + \text{FP}_{\text{failing\_normal}}}{4,811} \times 100\%$

### Tenet 8: Cross-Job Generalization Sensitivity Cohort
- **Population:** Exactly **75 failure entities** belonging to the **4 parent jobs completely absent from development partitions** (`6250345153`, `6250804554`, `6251723929`, `6252225776`), comprising **79 distinct failure episodes** and **347 pre-failure windows**.
- **Role:** Preserved as an independent sensitivity analysis to evaluate out-of-distribution application generalization without confounding by shared job templates.

### Tenet 9: Entity-Level Cluster Bootstrap Uncertainty Specification
- **Clustering Unit:** Strictly `entity_id` ($N=337$ clusters: 137 failure entities + 200 normal entities).
- **Clustering Invariant:** Resampling with replacement is performed strictly at the entity level. All episodes (141) and windows belonging to a selected entity remain bundled together. Treating episodes as independent clusters is strictly prohibited.
- **Replicates:** 1,000 paired bootstrap resamples; 95% percentile confidence intervals.

### Tenet 10: Model Selection and Operational Ranking Rule
- **Mandatory Operational Gate Hurdle:** A candidate model is eligible to be declared an operational winner **if and only if it passes BOTH operational gates**:
  1. $\text{FAR}_{\text{entity}} \le 12.0\%$ ($\le 23 / 198$ false alarm entities)
  2. $\text{FPR}_{\text{primary}} \le 5.0\%$ ($\le 240 / 4,811$ false positive windows)
- **Primary Ranking Metric:** Eligible models are ranked strictly by **Episode Detection Recall** on the primary cohort.
- **Pre-Registered Tie-Breaker:** **Macro-F1** across all 3 classes (Normal, Warning, Critical).
- **Mandatory Fallback Clause:** If no candidate model satisfies both operational gates, the protocol mandates declaring **NO OPERATIONAL WINNER**. All models must be reported regardless of gate outcome.

---

## 5. Downstream Evaluation Code Integration & Quarantine Architecture

To ensure that downstream scripts cannot accidentally read ambiguous counts or ingest quarantined tasks, the following software components were implemented and verified:

1. **Ratified Protocol Loader ([`src/evaluation/confirmatory_evaluator_l5.py`](file:///c:/Users/hp/Desktop/Research%20project/src/evaluation/confirmatory_evaluator_l5.py)):**
   - Function [`load_ratified_l5_protocol()`](file:///c:/Users/hp/Desktop/Research%20project/src/evaluation/confirmatory_evaluator_l5.py#L31-L85): Loads and verifies Protocol Amendment 2.1. Strictly rejects unamended or legacy artifacts.
   - Function [`validate_evaluation_manifest()`](file:///c:/Users/hp/Desktop/Research%20project/src/evaluation/confirmatory_evaluator_l5.py#L88-L187): Programmatically verifies that evaluation manifests contain exactly 5,511 windows across the 335 approved evaluable entities. Throws a fatal `ValueError` if any of the 61 quarantined census tasks are detected.
   - Function [`evaluate_l5_operational_gates()`](file:///c:/Users/hp/Desktop/Research%20project/src/evaluation/confirmatory_evaluator_l5.py#L190-L270): Dynamically evaluates the twin operational gates against the locked denominators (198 entities and 4,811 windows) and computes disaggregated FPRs.
   - Function [`select_operational_winner()`](file:///c:/Users/hp/Desktop/Research%20project/src/evaluation/confirmatory_evaluator_l5.py#L273-L340): Programmatically enforces gate compliance and tie-breaker ranking, defaulting to `NO_OPERATIONAL_WINNER` when gates are violated.

2. **Dedicated L=5 Extraction Script ([`scripts/extract_confirmatory_data_l5.py`](file:///c:/Users/hp/Desktop/Research%20project/scripts/extract_confirmatory_data_l5.py)):**
   - Ingests strictly the ratified primary population pre-registered in Amendment 2.1.
   - Enforces $L=5$ sequence generation (`window_size=5, require_full_history=True, exclude_unknown=True`).
   - Uses the frozen scaler fitted strictly on training data ([`configs/fitted_scaler.json`](file:///c:/Users/hp/Desktop/Research%20project/configs/fitted_scaler.json)).

---

## 6. Verification and Test Suite Results

A dedicated automated test suite ([`tests/test_phase6_2_3_protocol_amendment.py`](file:///c:/Users/hp/Desktop/Research%20project/tests/test_phase6_2_3_protocol_amendment.py)) was constructed to independently audit the ratification deliverables:

```bash
.\.venv\Scripts\pytest -v tests/test_phase6_2_3_protocol_amendment.py
```

### Test Suite Execution Summary
- `test_original_protocol_preservation_and_sha256`: **PASSED** (SHA-256 hash `ebf692598f2cb7115919c427127782f44ba054c655b3b76100aadb5955377bd1` verified invariant).
- `test_amendment_artifact_ratification_and_metadata`: **PASSED** (Version 2.1, status `RATIFIED_PRE_REGISTERED_FROZEN`, 10 tenets verified).
- `test_locked_denominators_and_reconstruction`: **PASSED** (Reconstructed 400 Critical, 300 Warning, 4,811 Normal, 5,511 total windows from source metadata).
- `test_quarantine_of_61_excluded_census_tasks`: **PASSED** (61 tasks verified quarantined; fatal exception raised if injected).
- `test_manifest_validation_on_exact_primary_population`: **PASSED** (Primary manifest validates cleanly against 335 evaluable entities).
- `test_operational_gates_boundary_conditions`: **PASSED** (Exact mathematical boundaries verified: 23 vs. 24 entities for FAR; 240 vs. 241 windows for FPR).
- `test_model_selection_and_ranking_rule`: **PASSED** (Operational winner selection and fallback to `NO_OPERATIONAL_WINNER` verified).
- `test_cross_job_sensitivity_cohort_preservation`: **PASSED** (75 entities, 79 episodes across 4 unseen jobs verified).
- `test_bootstrap_cluster_level_invariant`: **PASSED** (Cluster unit locked strictly at `entity_id` level, $N=337$).

**Result:** 9 of 9 Phase 6.2.3 tests passed.

---

## 7. Artifact Deliverables Summary

| Artifact File | Description | SHA-256 / Status |
| :--- | :--- | :--- |
| [`results/phase6/protocol_l5.json`](file:///c:/Users/hp/Desktop/Research%20project/results/phase6/protocol_l5.json) | Immutable original Phase 6.2 protocol record | `ebf692598f2cb7115919c427127782f44ba054c655b3b76100aadb5955377bd1` (Preserved) |
| [`results/phase6/protocol_l5_amendment_2_1.json`](file:///c:/Users/hp/Desktop/Research%20project/results/phase6/protocol_l5_amendment_2_1.json) | Versioned ratified protocol amendment locking 5,511 windows | `RATIFIED_PRE_REGISTERED_FROZEN` |
| [`src/evaluation/confirmatory_evaluator_l5.py`](file:///c:/Users/hp/Desktop/Research%20project/src/evaluation/confirmatory_evaluator_l5.py) | Confirmatory evaluation framework enforcing Amendment 2.1 | Verified & Exported |
| [`scripts/extract_confirmatory_data_l5.py`](file:///c:/Users/hp/Desktop/Research%20project/scripts/extract_confirmatory_data_l5.py) | Downstream data extraction script pointing to Amendment 2.1 | Implemented |
| [`tests/test_phase6_2_3_protocol_amendment.py`](file:///c:/Users/hp/Desktop/Research%20project/tests/test_phase6_2_3_protocol_amendment.py) | Formal test suite validating amendment invariants | 9/9 Tests Passing |
| [`reports/phase6_2_3_protocol_amendment.md`](file:///c:/Users/hp/Desktop/Research%20project/reports/phase6_2_3_protocol_amendment.md) | Comprehensive scientific audit and ratification report | Finalized |

---

## 8. Final Audit Verdict

```
################################################################################
#                                                                              #
#                        FINAL RATIFICATION VERDICT:                           #
#                 AMENDMENT_RATIFIED_READY_FOR_L5_TRAINING                     #
#                                                                              #
################################################################################
```

**Verdict Rationale:**  
Protocol Amendment 2.1 has formally resolved the window count discrepancy between summary tables and manifests. The primary evaluation population is permanently locked at exactly **5,511 windows** (400 Critical, 300 Warning, 4,811 Normal: 4,515 pure-normal and 296 failing-cohort normal). The operational gates are formally locked to $\le 23$ false alarm entities ($\text{FAR} \le 12.0\%$) and $\le 240$ false positive windows ($\text{FPR} \le 5.0\%$). The 61 non-pre-failing tasks are strictly quarantined. Downstream code directives and unit tests enforce these invariants without ambiguity.

With all pre-registered safeguards intact and the confirmatory protocol formally ratified, the codebase is scientifically ready to proceed to **Phase 6.3: L=5 Model Training**.
