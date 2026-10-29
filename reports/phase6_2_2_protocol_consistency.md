# Phase 6.2.2 — Frozen Protocol Consistency Audit Report

**Audit Date:** October 26, 2026  
**Auditor:** Independent ML Scientific Auditor  
**Audit Scope:** Frozen Phase 6.2 Protocol ([`results/phase6/protocol_l5.json`](file:///c:/Users/hp/Desktop/Research%20project/results/phase6/protocol_l5.json)), Audit Artifacts ([`results/phase6/pretraining_integrity_audit.json`](file:///c:/Users/hp/Desktop/Research%20project/results/phase6/pretraining_integrity_audit.json)), Reports, and Evaluation Code  
**Final Audit Verdict:** **`NOT_READY_REQUIRES_PROTOCOL_AMENDMENT`**  
**Associated Artifacts:**
- Machine-Readable Consistency Audit: [`results/phase6/protocol_consistency_audit.json`](file:///c:/Users/hp/Desktop/Research%20project/results/phase6/protocol_consistency_audit.json)
- Automated Verification Suite: [`tests/test_phase6_2_2_protocol_consistency.py`](file:///c:/Users/hp/Desktop/Research%20project/tests/test_phase6_2_2_protocol_consistency.py) (6/6 tests passed)

---

## Executive Summary & Official Audit Verdict

### Final Verdict: `NOT_READY_REQUIRES_PROTOCOL_AMENDMENT`

This consistency audit independently reviewed the frozen Phase 6.2 protocol specification, cross-referenced all candidate manifests against raw trace shards, traced evaluation code data loaders, and audited operational false-alarm denominators.

### Reason for Verdict
While all fundamental scientific boundaries remain 100% sound (137 failure entities, 141 failure episodes, 200 normal entities, 0 entity leakage, and 4 completely disjoint parent jobs), a **material mathematical discrepancy exists within the frozen protocol artifact [`results/phase6/protocol_l5.json`](file:///c:/Users/hp/Desktop/Research%20project/results/phase6/protocol_l5.json)**:
- The authoritative entity manifest `frozen_entity_ids` encodes **137 failure entities** and **200 normal entities**, which physically yields **5,511 total evaluable windows** (400 Critical, 300 Warning, 4,811 Normal).
- In contrast, the summary block `cohort_counts_summary.sliding_windows` cites **5,727 total evaluable windows** (5,027 Normal windows), carrying over the expanded 198-task census from Phase 6.1.
- This discrepancy shifts the pre-registered Window False Positive Rate ($\text{FPR}_{\text{window}} \le 5.0\%$) denominator from 4,811 to 5,027, shifting the allowable false positive threshold from **240 to 251 windows**.

In strict confirmatory machine learning, **denominators and evaluation populations must never be ambiguous, and must never be altered after seeing model predictions**. Therefore, this audit formally issues a `NOT_READY_REQUIRES_PROTOCOL_AMENDMENT` verdict and submits **Protocol Amendment 2.1** to definitively lock the population before any model training or evaluation occurs.

---

## 1. Resolving the Evaluation Population

### 1.1 Explicit Population Distinction
To eliminate confusion between the trace census and the evaluable confirmatory holdout, three distinct populations are distinguished:
1. **The Primary Confirmatory Manifest Cohort ($N=335$ evaluable entities):**
   - **137 failure entities** yielding **700 pre-failure windows** (300 Warning, 400 Critical) and **296 normal windows** from earlier in their lifespans.
   - **198 evaluable pure-normal entities** (out of 200 frozen; 2 excluded due to $< 5$ bins) yielding **4,515 normal windows**.
   - **Total Evaluated Windows:** **5,511 windows** (**700 pre-failure**, **4,811 normal**).
2. **The Non-Pre-Failing Unused Tasks ($N=61$ tasks):**
   - 61 unused failing tasks in the raw trace possessed $\ge 5$ consecutive bins, but their failure occurred $> 30$ minutes after telemetry ended ($\text{TTF} > 1800\text{s}$).
   - They generate **216 Normal windows** and **0 pre-failure windows**.
   - Their entity IDs are **NOT** present in `frozen_entity_ids`.
3. **The Expanded Trace Census ($N=396$ evaluable entities):**
   - Combines the 198 census failure tasks ($137 + 61$) with the 198 evaluable normal entities.
   - Yields **5,727 windows** (**700 pre-failure**, **5,027 normal**).

### Table 1: Complete Population and Window Resolution
| Subgroup / Population | Entities | Critical Windows ($y=2$) | Warning Windows ($y=1$) | Normal Windows ($y=0$) | Total Windows | Status in Protocol Artifact |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **1. Frozen Failure Entities** | **137** | **400** | **300** | **296** | **996** | Encoded in `frozen_entity_ids.all_failure_entities` |
| **2. Non-Pre-Failing Unused Tasks** | **61** | **0** | **0** | **216** | **216** | **Omitted from `frozen_entity_ids`** |
| **3. Full Unused Failing Census** ($1+2$) | **198** | **400** | **300** | **512** | **1,212** | Recorded in Phase 6.1 census table |
| **4. Frozen Normal Cohort** | **198** (200 frozen) | **0** | **0** | **4,515** | **4,515** | Encoded in `frozen_entity_ids.normal_entities` |
| **A. Primary Manifest Population** ($1+4$) | **335** | **400** | **300** | **4,811** | **5,511** | **Physically scored by evaluation scripts** |
| **B. Expanded Census Population** ($3+4$) | **396** | **400** | **300** | **5,027** | **5,727** | Recorded in `cohort_counts_summary` |

### 1.2 Evaluation Script & Manifest Behavior
- **Manifest Reality:** When an evaluation script loads `results/phase6/protocol_l5.json` and filters data using `frozen_entity_ids["all_failure_entities"]` and `frozen_entity_ids["normal_entities"]`, it extracts exactly **335 evaluable entities** and **5,511 sliding windows**.
- **The Discrepancy:** It will **never score 5,727 windows**, because the 61 non-pre-failing tasks do not exist in the manifest!
- **Operational Risk:** If an automated pipeline enforces `assert len(X_holdout) == protocol["cohort_counts_summary"]["sliding_windows"]["total_evaluable_windows"]` (5,727), the test will fail.

---

## 2. False-Alarm Calculations & Denominators

### 2.1 Entity False Alarm Rate ($\text{FAR}_{\text{entity}}$)
- **Denominator:** Strictly **198 evaluable pure-normal entities**.
- **Formula:**
  $$\text{FAR}_{\text{entity}} = \frac{\sum_{e=1}^{198} \mathbb{I}(\exists t : \hat{y}_{e, t} \in \{1, 2\})}{198} \times 100\%$$
- **Operational Gate ($\le 12.0\%$):**
  $$\text{Max Allowable False Alarm Entities} = \lfloor 0.12 \times 198 \rfloor = \mathbf{23}\text{ entities}$$
- **Audit Determination:** This denominator is completely unaffected by the 61 additional tasks or the 137 failure entities.

### 2.2 Window False Positive Rate ($\text{FPR}_{\text{window}}$)
- **Authoritative Primary Denominator:** **4,811 normal windows** (4,515 from pure-normal cohort + 296 from 137 failure entities).
- **Formula:**
  $$\text{FPR}_{\text{primary}} = \frac{\text{FP}_{\text{pure\_normal}} + \text{FP}_{\text{failing\_normal}}}{4,811} \times 100\%$$
- **Operational Gate ($\le 5.0\%$):**
  $$\text{Max Allowable False Positive Windows} = \lfloor 0.05 \times 4811 \rfloor = \mathbf{240}\text{ windows}$$
- **Discrepancy Impact:** If the expanded denominator of 5,027 were used, the allowable FP threshold would be $\lfloor 0.05 \times 5027 \rfloor = 251$ windows (an unearned 11-window leniency).

### 2.3 Disaggregated False Alarm Reporting Mandate
To ensure full transparency between pure false alarms on healthy machines and premature alerts on degrading tasks:
1. **Pure-Normal Window FPR:**
   $$\text{FPR}_{\text{pure\_normal}} = \frac{\text{FP}_{\text{pure\_normal}}}{4,515} \times 100\%$$
2. **Failing-Cohort Normal Window FPR:**
   $$\text{FPR}_{\text{failing\_normal}} = \frac{\text{FP}_{\text{failing\_normal}}}{296} \times 100\%$$
3. Both must be reported separately in confirmatory result tables alongside the pooled primary FPR ($4,811$).

---

## 3. Cross-Artifact Consistency Matrix

The audit compared [`results/phase6/protocol_l5.json`](file:///c:/Users/hp/Desktop/Research%20project/results/phase6/protocol_l5.json), [`reports/phase6_2_protocol_freeze.md`](file:///c:/Users/hp/Desktop/Research%20project/reports/phase6_2_protocol_freeze.md), [`results/phase6/pretraining_integrity_audit.json`](file:///c:/Users/hp/Desktop/Research%20project/results/phase6/pretraining_integrity_audit.json), and existing test suites:

### Table 2: Cross-Artifact Agreement & Disagreement Matrix
| Item / Specification | Manifests (`protocol_l5.json`) | Summary Block (`protocol_l5.json`) | Report 6.2 | Audit 6.2.1 | Agreement Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Failure Entities** | 137 | 137 | 137 | 137 | **CONSISTENT (100%)** |
| **Normal Entities** | 200 (198 evaluable) | 200 (198 evaluable) | 200 (198) | 200 (198) | **CONSISTENT (100%)** |
| **Failure Episodes** | 141 (79 disjoint, 62 shared) | 141 (79 disjoint, 62 shared) | 141 | 141 | **CONSISTENT (100%)** |
| **Pre-Failure Windows** | 700 (300 Warn, 400 Crit) | 700 (300 Warn, 400 Crit) | 700 | 700 | **CONSISTENT (100%)** |
| **Failing Normal Windows** | **296** | **512** | 512 | 296 (manifest) / 512 (census) | **DISAGREEMENT (216 windows)** |
| **Total Normal Windows** | **4,811** | **5,027** | 5,027 | 4,811 (manifest) / 5,027 (census) | **DISAGREEMENT (216 windows)** |
| **Total Windows** | **5,511** | **5,727** | 5,727 | 5,511 (manifest) / 5,727 (census) | **DISAGREEMENT (216 windows)** |
| **Sensitivity Parent Jobs** | 4 (`6250345153`, etc.) | 4 | 4 | 4 | **CONSISTENT (100%)** |
| **Bootstrap Unit** | `entity_id` | `entity_id` | `entity_id` | `entity_id` | **CONSISTENT (100%)** |
| **Operational Gates** | FAR $\le 12\%$, FPR $\le 5\%$ | FAR $\le 12\%$, FPR $\le 5\%$ | Same | Same | **CONSISTENT (100%)** |

---

## 4. Hybrid Model Training Audit Confirmation

- **Audit Finding:** In [`src/models/hybrid.py`](file:///c:/Users/hp/Desktop/Research%20project/src/models/hybrid.py), the `RepresentationHybridClassifier` extracts temporal embeddings $H_{\text{temp}} = \text{TransformerEncoder}(X_{\text{train}})$ **in-sample** on the training partition.
- **Protocol Status:** The frozen protocol explicitly specifies and accepts this in-sample representation fusion.
- **Quarantine Integrity:** Confirmed that **zero validation or holdout data** enters the Random Forest. Holdout quarantine is 100% intact.
- **Decision:** As instructed, ad-hoc cross-fitting shall **not** be introduced into the current experiment. Training under the pre-registered in-sample architecture is formally validated to proceed.

---

## 5. Formal Protocol Amendment 2.1 Proposal

To eliminate any ambiguity before model training, the following versioned amendment is formally submitted:

```markdown
### Protocol Amendment 2.1 — Primary Confirmatory Population & Denominator Lock

1. Authoritative Primary Population:
   The primary confirmatory evaluation shall be executed strictly on the 337 entities
   encoded in frozen_entity_ids (137 failure entities + 200 normal entities).
   In unpadded L=5 mode, exactly 335 entities are evaluable (137 failure, 198 normal).

2. Authoritative Window Counts:
   The primary holdout evaluation shall score exactly 5,511 windows:
   - 400 Critical windows (0 < tau <= 15m)
   - 300 Warning windows (15m < tau <= 30m)
   - 4,811 Normal windows (4,515 pure-normal + 296 failing-entity normal)

3. Authoritative Operational Gates:
   - Entity FAR Gate: FAR <= 12.0% evaluated across the 198 evaluable pure-normal entities
     (Maximum allowable false alarm entities = 23).
   - Window FPR Gate: FPR <= 5.0% evaluated across the 4,811 primary normal windows
     (Maximum allowable false positive windows = 240).

4. Exclusion of the 61 Non-Pre-Failing Tasks:
   The 61 unused failing tasks that generated no pre-failure windows (216 normal windows)
   are formally excluded from the primary confirmatory evaluation and designated as an
   optional secondary descriptive census.

5. Disaggregated Reporting:
   Confirmatory reports must report FPR_pure_normal (FP / 4,515) and FPR_failing_normal (FP / 296)
   separately, alongside the pooled primary FPR (FP / 4,811).
```

---

## Official Verdict

```
================================================================================
FINAL VERDICT: NOT_READY_REQUIRES_PROTOCOL_AMENDMENT
================================================================================
```

The underlying dataset, entity manifests, and experimental designs are sound and ready. However, formal ratification of **Protocol Amendment 2.1** (locking the primary population to 5,511 windows and the FPR denominator to 4,811) is required to ensure that evaluation denominators are unambiguous and locked before training and scoring begin.
