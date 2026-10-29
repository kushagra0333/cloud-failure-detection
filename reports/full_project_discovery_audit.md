# Comprehensive Scientific Audit & Research Discovery Report

**Project Title:** Temporal LSTM/Transformer Framework for Early Failure Prediction in Cloud Computing  
**Auditor:** Independent Machine Learning Scientific Auditor & Senior Pair Programmer  
**Date of Audit:** October 27, 2026  
**Audited Codebase:** `c:\Users\hp\Desktop\Research project`  
**Governing Protocol:** [Protocol Amendment 2.1](file:///c:/Users/hp/Desktop/Research%20project/results/phase6/protocol_l5_amendment_2_1.json)  
**Primary Confirmatory Cohort:** Frozen $L=5$ Confirmatory Holdout ($N = 5,511$ windows across 335 evaluable tasks)  
**Official Operational Status:** **`NO_OPERATIONAL_WINNER`** (Reaffirmed as immutable)  
**Confirmatory Validity Finding:** **`MATERIAL PROTOCOL VIOLATION FOUND`** (Phase 6.8 Audit)

---

## Executive Summary

This report delivers a complete, read-only scientific investigation of the research project **"Temporal LSTM/Transformer Framework for Early Failure Prediction in Cloud Computing."** It is written specifically for the student researcher to provide full clarity into the research problem, the codebase implementation, the historical evolution across Phases 1 through 6.8, the recent discovery of data-censoring defects, and the realistic paths forward.

### The Core Finding in Brief
1. **The Research Goal:** The project asks whether deep temporal sequence models (LSTM, Transformer, and a Representation-Fused Hybrid) can anticipate container task failures in cloud computing clusters 0 to 30 minutes in advance, outperforming non-temporal baselines (Random Forest) while maintaining low false alarm rates suitable for autonomous cloud operations.
2. **The Confirmatory Verdict:** In Phase 6.4, an unblinded evaluation on 5,511 frozen holdout windows resulted in an official verdict of **`NO_OPERATIONAL_WINNER`**. All four models failed both pre-registered operational gates by substantial margins:
   - Window False Positive Rate (FPR) ranged from **$20.27\%$ to $34.65\%$** (exceeding the $\le 5.0\%$ operational ceiling by $4\times$ to $7\times$).
   - Entity False Alarm Rate (FAR) ranged from **$31.82\%$ to $64.14\%$** (exceeding the $\le 12.0\%$ operational ceiling by $2.6\times$ to $5.3\times$).
   - Random Forest achieved the highest Macro-F1 ($0.6111$), while deep temporal models suffered from extreme softmax overconfidence ($P(\text{Critical}) > 0.99$).
3. **The Material Data-Censoring Violation (Phase 6.8):** A forensic code audit of [`src/labeling/target_generator.py`](file:///c:/Users/hp/Desktop/Research%20project/src/labeling/target_generator.py#L73-L77) revealed an implementation omission: when generating labels, the code checked only for *future* failures. When a task had already failed in the past but had telemetry extending $\ge 30$ minutes post-failure, the code silently assigned `label = 0` (`NORMAL`) instead of `label = -1` (`UNKNOWN`).
   - In the confirmatory holdout, **149 post-failure recovery windows** were incorrectly included as "Normal".
   - Across the development sets, **1,130 of 4,188 training normal windows (27.0%)**, **197 of 840 validation normal windows (23.5%)**, and **145 of 864 test normal windows (16.8%)** were post-failure observations.
4. **Operational Invariance:** Diagnostic counterfactual re-evaluations prove that censoring these post-failure windows changes the operational metrics only marginally. All four models still fail both operational gates by wide margins (Window FPR remains $> 17.4\%$, Entity FAR remains $> 31.8\%$). The operational failure of the models is a robust scientific truth.
5. **The Scientific Value:** This research is **not a failure**; it is a vital, publication-worthy negative result and empirical audit. It proves that deep sequence models under uniform argmax thresholding fail under extreme class imbalance, that observational missingness creates severe confounding, and that parent-job correlation must be accounted for in cloud machine learning.

---

# PART 1 — Understanding the Entire Project

### 1.1 The Exact Research Problem
Modern cloud datacenters (such as Google, AWS, and Azure) run millions of containerized tasks across tens of thousands of physical machines. In multi-tenant environments, tasks frequently terminate abnormally due to software bugs, out-of-memory errors, CPU starvation, machine reboots, or dependency timeouts. 

Current cluster schedulers (like Google Borg or Kubernetes) are purely **reactive**: they detect that a container has died only *after* it crashes, triggering restarts, rescheduled retries, and cascading dependency failures. The research problem is:
> **Can we build a predictive system that monitors runtime resource telemetry (CPU, memory, disk, hardware counters) and accurately warns cluster operators 15 to 30 minutes before a container crashes?**

### 1.2 The Main Hypothesis
The central hypothesis was that **deep temporal sequence models** (LSTM and Self-Attention Transformers) would outperform traditional static tabular models (Random Forest) because they could learn complex sequential temporal patterns and gradual degradation signatures over trailing time windows.

### 1.3 What We Are Predicting & Why It Matters
The model solves a **3-class sequential classification problem**:
- **Class 0 (`NORMAL`):** The container is healthy and will execute actively for at least the next 30 minutes without failing.
- **Class 1 (`WARNING`):** An abnormal failure will occur in **15 to 30 minutes** ($900\text{ s} < \tau \le 1800\text{ s}$).
- **Class 2 (`CRITICAL`):** An abnormal failure is imminent within **0 to 15 minutes** ($0\text{ s} < \tau \le 900\text{ s}$).

**Why this is useful in production:**
- A **Warning alert (15–30 min)** provides enough lead time to initiate live task migration, spin up backup replica containers, or throttle incoming query loads.
- A **Critical alert (0–15 min)** allows the system to force an immediate application-level memory checkpoint, flush database transaction logs, and gracefully notify dependent jobs.

### 1.4 What "Early Failure Prediction" Means in This Project
"Early prediction" strictly requires **positive lead time**:
$$\tau = T_{\text{fail}} - t_{\text{pred}} > 0$$
All telemetry data used to make the prediction must be observed strictly at or before $t_{\text{pred}}$ ($t \le t_{\text{pred}}$). Predicting at or after a crash ($t_{\text{pred}} \ge T_{\text{fail}}$) is post-mortem root-cause analysis, not early anticipation.

### 1.5 System Inputs and Outputs
- **Input:** A 3-dimensional tensor $X \in \mathbb{R}^{N \times L \times 31}$, where:
  - $N$ is the number of sliding window examples.
  - $L$ is the historical sequence length ($L=5$ steps, representing trailing 5-minute measurements: $t - 20\text{m}, t - 15\text{m}, t - 10\text{m}, t - 5\text{m}, t$).
  - $31$ is the number of engineered feature signals per step.
- **Output:** A probability vector $\hat{P} = [P(\text{Normal}), P(\text{Warning}), P(\text{Critical})]$. Under uniform argmax, the predicted class is $\hat{y} = \text{argmax}_{c \in \{0, 1, 2\}} P(c)$.

### 1.6 The Complete Pipeline: From Raw Data to Prediction
```
[Raw Google Trace CSV.GZ] (Events & Usage)
           │
           ▼
[Ingestion & Schema Validation] (src/ingestion/schema.py, validator.py)
           │  • Microsecond to second conversion (timestamp / 1e6)
           │  • Map event codes (FAIL = 3 in Borg 2011)
           ▼
[300s Grid Alignment & Cleaning]
           │  • Aggregate raw telemetry into 5-minute fixed bins
           │  • Resolve duplicate timestamps via mean aggregation
           ▼
[Feature Engineering] (src/features/engineer.py)
           │  • 31 causal features (rates, rolling means, lags, slopes, missingness)
           ▼
[Z-Score Scaling] (configs/fitted_scaler.json)
           │  • StandardScaler fitted strictly on training data (mean & scale)
           ▼
[Target Labeling] (src/labeling/target_generator.py)
           │  • Assign Normal (0), Warning (1), Critical (2), Unknown (-1)
           ▼
[Sliding Window Generation] (src/labeling/window_generator.py)
           │  • Construct L=5 sequences; assert unbroken 25-min contiguity
           │  • Exclude UNKNOWN (-1)
           ▼
[Leakage-Safe Partitioning] (src/splitting/leakage_safe_split.py)
           │  • Partition into Train, Validation, Test splits
           ▼
[Model Training] (scripts/train_phase6_3_l5_models.py)
           │  • Train Random Forest, LSTM, Transformer, Hybrid
           ▼
[Inference & Confirmatory Evaluation] (src/evaluation/confirmatory_evaluator_l5.py)
           │  • Evaluate on frozen holdout; compute FPR, FAR, Episode Recall, Lead Time
           ▼
[Auditing & Integrity Verification] (Phase 6.5, 6.6, 6.7, 6.8 Audits)
```

### 1.7 The Dataset: Origin, Structure, and Limitations
- **Origin:** The **Google Cluster Trace 2011 (version 2.1)**, published by Wilkes, Reiss, et al.
- **Scope:** Traces from an internal Google production cluster containing ~12,500 machines over 29 continuous days in May 2011.
- **Tables Used:**
  1. `task_events`: Records state transitions (`SUBMIT=0`, `SCHEDULE=1`, `EVICT=2`, `FAIL=3`, `FINISH=4`, `KILL=5`, `LOST=6`).
  2. `task_usage`: Records periodic 5-minute resource telemetry (CPU rate, memory usage, assigned memory, disk I/O, Cycles Per Instruction).
- **Inherent Limitations:**
  - *No kernel error logs:* The trace records only that a task experienced `FAIL` (3); it provides no exit codes, stack traces, or OS panic strings.
  - *Telemetry aggregation:* Resource metrics are pre-averaged over 5-minute windows by Google's collection daemon.
  - *Hardware counter omissions:* Hardware performance counters (CPI, MAI) are often absent when containers are idle or descheduled.
  - *Batch bias:* 2011 Borg workloads were dominated by MapReduce batch jobs rather than modern microservices.

### 1.8 Key Concept Definitions
- **Row:** In telemetry, a single 5-minute measurement. In predictions, a single window evaluated at $t_{\text{pred}}$.
- **Entity / Task (`entity_id`):** In this codebase, defined as `f"{job_id}_{task_index}"` (e.g., `6224307516_2`). Represents a single container instance.
- **Parent Job (`parent_job_id`):** A higher-level application (e.g., MapReduce job `6224307516`) consisting of multiple tasks. Tasks in the same parent job execute the same binary code and share resource patterns.
- **Failure Event:** A raw trace record where `event_type == 3`. (A task can experience multiple failure events if Borg retried it!).
- **Failure Episode:** A specific failure event preceded by evaluable pre-failure telemetry ($0 < \tau \le 30\text{ min}$). In the confirmatory holdout, there are 141 episodes across 137 entities.
- **Window / Sequence:** A contiguous series of $L=5$ consecutive 5-minute steps ending at prediction time $t_{\text{pred}}$ ($1,200\text{ s}$ span between step 0 and step 4, $1,500\text{ s}$ physical collection span).

---

# PART 2 — Repository Architecture & Project Map

### 2.1 Directory Structure
```
c:\Users\hp\Desktop\Research project\
├── configs/               # Master YAML and fitted scaler JSON files
├── data/
│   ├── raw/               # Compressed raw Google trace CSV.GZ tables
│   └── processed/         # Processed NPY tensors and metadata CSVs
│       ├── l5/            # Development L=5 tensors (train, val, test)
│       └── confirmatory_l5/# Frozen primary confirmatory dataset
├── models/
│   ├── checkpoints/       # Phase 3-4 historical L=15 checkpoints
│   └── checkpoints_l5/    # Phase 6.3 ratified L=5 checkpoints (RF, LSTM, TF, Hybrid)
├── reports/               # 32 markdown audit and phase reports
├── results/               # Experiment metric JSONs, prediction CSVs, and audit artifacts
│   ├── phase3a/ - phase5/ # Historical phase results
│   ├── phase6/            # Confirmatory evaluation & Phase 6.5-6.8 audits
│   └── project_discovery/ # Master discovery inventory, matrices, and plans
├── scratch/               # Analysis scripts and diagnostic tools
├── scripts/               # 34 execution scripts for pipeline and audits
├── src/                   # Core Python library
│   ├── evaluation/        # Confirmatory evaluator and metric calculation
│   ├── features/          # Feature registry, causal engineering, and scaling
│   ├── ingestion/         # Schema definitions and data validation
│   ├── labeling/          # Target label and sliding window generators
│   ├── models/            # RF, LSTM, Transformer, and Hybrid architectures
│   ├── splitting/         # Leakage-safe chronological and entity splitters
│   └── training/          # Model trainers with early stopping
└── tests/                 # 27 automated test suites (159 passing tests)
```

### 2.2 Key Files and Checkpoints
All primary L=5 model checkpoints are located in [`models/checkpoints_l5/`](file:///c:/Users/hp/Desktop/Research%20project/models/checkpoints_l5):
1. **Random Forest:** [`random_forest_l5.joblib`](file:///c:/Users/hp/Desktop/Research%20project/models/checkpoints_l5/random_forest_l5.joblib) (4.79 MB, 100 trees, max depth 15, balanced class weights).
2. **Temporal LSTM:** [`best_lstm_checkpoint_l5.pt`](file:///c:/Users/hp/Desktop/Research%20project/models/checkpoints_l5/best_lstm_checkpoint_l5.pt) (255.7 KB, 63,171 parameters, 2 layers, hidden dim 64).
3. **Compact Transformer:** [`best_transformer_checkpoint_l5.pt`](file:///c:/Users/hp/Desktop/Research%20project/models/checkpoints_l5/best_transformer_checkpoint_l5.pt) (92.3 KB, 22,179 parameters, 2 layers, $d_{\text{model}}=32$, 4 attention heads).
4. **Representation Hybrid:** [`representation_hybrid_l5.joblib`](file:///c:/Users/hp/Desktop/Research%20project/models/checkpoints_l5/representation_hybrid_l5.joblib) (3.83 MB, Random Forest on 63-dimensional fused representations).

### 2.3 Master Protocols and Confirmatory Prediction Files
- **Authoritative Ratified Protocol:** [`results/phase6/protocol_l5_amendment_2_1.json`](file:///c:/Users/hp/Desktop/Research%20project/results/phase6/protocol_l5_amendment_2_1.json)
- **Primary Predictions File:** [`results/phase6/confirmatory_predictions_l5.csv`](file:///c:/Users/hp/Desktop/Research%20project/results/phase6/confirmatory_predictions_l5.csv) (5,511 rows, 22 columns containing ground-truth labels and predictions/probabilities from all four models).
- **Reported Metric JSON:** [`results/phase6/confirmatory_metrics_l5.json`](file:///c:/Users/hp/Desktop/Research%20project/results/phase6/confirmatory_metrics_l5.json)

---

# PART 3 — Chronological Research Timeline (Phases 1 to 6.8)

| Phase | Core Objective | What Was Actually Implemented | Data Used | Key Result | Problem or Defect Discovered | Evidence File |
| :---: | :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | Ingestion & Schema | Raw trace parsing, schema mapping (`ClusterEventTypeV2`). | Sample raw shards | 300s grid alignment verified. | Categorical enum differences in Borg trace format. | `phase1_completion_report.md` |
| **2** | Feature Pipeline & Labeling | 31 causal features, target generator, window generator ($L=15$). | Sample trace usage & events | Initial train/val/test splits created ($L=15$). | **Censoring Bug Introduced:** `target_generator.py` omitted past-failure check. | `phase2_completion_report.md` |
| **3A** | Baseline Models | Majority classifier, Logistic Regression, Random Forest. | $L=15$ development split | Random Forest achieved Macro-F1 = 0.58. | Class imbalance caused standard logistic regression to collapse. | `phase3a_baseline_results.md` |
| **3B** | Temporal LSTM | 2-layer LSTM ($d_{\text{hidden}}=64$) trained on $L=15$ sequences. | $L=15$ development split | LSTM achieved Macro-F1 = 0.62 on validation. | Extreme class imbalance required weighted cross-entropy. | `phase3b_lstm_results.md` |
| **3C** | Methodology Gate | Robustness audit, entity-cluster bootstrap framework. | Development validation set | High episode recall reported on validation. | **Parent-Job Overlap:** 75% of validation jobs overlapped with train. | `phase3c_1_methodology_gate.md` |
| **4** | Transformer & Hybrid | Compact Transformer and Representation Hybrid. | $L=15$ development split | Hybrid reported Macro-F1 = 0.64 on validation. | Hybrid trained Random Forest on in-sample neural embeddings. | `phase4_transformer_hybrid_results.md` |
| **5** | Confirmatory Evaluation ($L=15$) | First unblinded test on 100 frozen failure tasks. | 100 failure entities ($L=15$) | **TOTAL FAILURE:** 0 pre-failure windows met $L=15$ invariant! | **Dataset Infeasibility:** Tasks failed in $< 75$ min. $L=15$ was impossible. | `phase5_confirmatory_results.md` |
| **6.1** | Feasibility Audit | Trace mining to evaluate sequence length reduction. | Full trace census | Proved $L=5$ (25 min) allows 137 entities to yield 700 windows. | Protocol redesign required. | `phase6_dataset_feasibility.md` |
| **6.2** | Protocol Freeze & Amendment 2.1 | Formal pre-registration of $L=5$ protocol. | 137 fail + 200 normal controls | Locked 5,511 primary windows (4,811 normal, 700 pre-fail). | Mismatch between 5,727 census summary and 5,511 manifest. | `protocol_l5_amendment_2_1.json` |
| **6.3** | $L=5$ Model Training | Retrained all 4 models from scratch on $L=5$ dev data. | $L=5$ sliced dev sets | RF, LSTM, Transformer, Hybrid checkpoints saved. | Neural models used explicit weights $[1, 1.5, 2]$, not balanced. | `training_manifest_l5.json` |
| **6.4** | Confirmatory Evaluation ($L=5$) | Unblinded evaluation of 5,511 primary windows. | Frozen $L=5$ holdout (335 tasks) | **NO_OPERATIONAL_WINNER:** All 4 models failed both gates! | High FPR ($20\% - 35\%$), high FAR ($31\% - 64\%$). | `confirmatory_metrics_l5.json` |
| **6.5** | Post-Evaluation Audit | Independent verification of metrics and gate failure. | 5,511 confirmatory predictions | Verified that all 4 models failed FPR ($\le 5\%$) and FAR ($\le 12\%$). | Softmax overconfidence observed in Transformer ($P > 0.99$). | `post_evaluation_audit_l5.json` |
| **6.6** | Error Mechanism Investigation | Post-hoc analysis of false-alarm drivers. | 5,511 confirmatory predictions | Found CPI missingness elevation and top parent-job clustering. | Mispartitioned failure normal windows (140 vs 156); causal overreach. | `phase6_6_error_mechanism_investigation.json` |
| **6.7** | Independent Scientific Audit | Read-only audit of Phase 6.6 claims against raw trace. | Raw trace events & predictions | Corrected partition to 147 pre-fail / 149 post-fail; refuted 10x loss weights. | Discovered repeated failures in tasks `6224307516_2` and `6224307516_3`. | `phase6_7_independent_scientific_audit.json` |
| **6.8** | Validity & Censoring Audit | Investigated whether evaluated windows violated protocol. | Raw events & 5,511 predictions | **VERDICT C (Material Violation):** 149 post-failure windows were included! | Proved operational failure is invariant to censoring corrections. | `phase6_8_confirmatory_validity_audit.json` |

---

# PART 4 — Forensic Data & Labeling Audit

### 4.1 How Raw Events Become Labels
In [`src/labeling/target_generator.py`](file:///c:/Users/hp/Desktop/Research%20project/src/labeling/target_generator.py), the function `derive_label_for_task` evaluates an entity at timestamp $t$:
1. It queries raw task events for `event_type == 3` (`FAIL`) occurring at `time_sec > t`.
2. If future failures exist, it computes $\tau = T_{\text{fail}}^{\text{earliest}} - t$:
   - If $0 < \tau \le 900\text{ s}$ (15 min) $\to$ **CRITICAL (2)**.
   - If $900\text{ s} < \tau \le 1800\text{ s}$ (30 min) $\to$ **WARNING (1)**.
   - If $\tau > 1800\text{ s}$ $\to$ **NORMAL (0)**.
3. If no future failure exists, it checks if telemetry continues for at least $1800\text{ s}$ or if a normal `FINISH` (4) occurred $\to$ **NORMAL (0)**.
4. If telemetry terminates without `FINISH` in $< 1800\text{ s}$, or if the task was killed/evicted $\to$ **UNKNOWN (-1)**.

### 4.2 The Phase 6.8 Finding: The Missing Prior-Failure Check
Line 201 of Protocol Amendment 2.1 explicitly mandates:
> *"censored lookahead (< 1800s without FINISH) or post-failure step -> label -1 (strictly excluded)"*

However, in [`src/labeling/target_generator.py`](file:///c:/Users/hp/Desktop/Research%20project/src/labeling/target_generator.py#L73-L77), step 2 of `derive_label_for_task` was implemented as:
```python
# Check if telemetry extends at least 30 minutes ahead
if telemetry_max_time_sec >= t + self.warning_horizon_sec:
    return FailureLabel.NORMAL, None  # <-- BUG: OMITTED CHECK FOR PRIOR FAILURES!
```
The code **never checked whether a failure event had already occurred in the past** (`time_sec <= t`)!

### 4.3 Scope of the Data Contamination
Our independent trace audit revealed the exact impact of this bug across the entire project:
- **Confirmatory Holdout Dataset (`confirmatory_predictions_l5.csv`):**
  - **149 post-failure recovery windows** (belonging to 41 failure tasks) were mislabeled as `NORMAL` (0) and included in evaluation.
  - **7 pre-failure normal windows** (in tasks `6224307516_2` and `6224307516_3`) occurred $> 30\text{ min}$ before failure 2, but their trailing 5-step history crossed failure 1.
  - **140 pre-failure normal windows** were clean pre-failure observations ($\tau > 1800\text{ s}$, zero prior failure).
  - Total failure-cohort normal windows: $140 + 7 + 149 = 296$.
- **Development Datasets:**
  - **Training Set (`train_metadata.csv`):** 1,130 of 4,188 Normal windows (**27.0%**) were post-failure observations! 311 windows had histories crossing failure events.
  - **Validation Set (`val_metadata.csv`):** 197 of 840 Normal windows (**23.5%**) were post-failure observations!
  - **Test Set (`test_metadata.csv`):** 145 of 864 Normal windows (**16.8%**) were post-failure observations!

### 4.4 Why Earlier Test Suites Missed This Bug
In Phase 3c, Phase 4, and Phase 5, tests asserted:
```python
assert (prefails["time_to_failure"] > 0.0).all()
assert (meta_fail["time_to_failure"].dropna() > 0).all()
```
Because the buggy generator set `time_to_failure = None` (`NaN`) when labeling post-failure steps as `NORMAL`, calling `.dropna()` completely dropped all post-failure normal windows! The assertion passed with flying colors while hundreds of post-failure windows remained silently in the dataset.

---

# PART 5 — Experimental Design & Methodological Audit

### 5.1 Why $L=15$ Failed and Forced the Pivot to $L=5$
In Phase 2 through Phase 4, the sequence length was fixed at $L=15$ observations (75 minutes unbroken history at 5-minute sampling). In Phase 5, when the unblinded confirmatory holdout was scored, **all 100 failure entities failed within their first 10 to 45 minutes of execution**. Zero tasks accumulated 75 minutes of pre-failure history! Consequently, Phase 5 produced zero pre-failure windows, making failure recall mathematically undefined.

In Phase 6, the project shortened the history invariant to $L=5$ observations ($1,200\text{ s}$ span between step 0 and 4, $1,500\text{ s}$ physical collection span). This scientifically defensible pivot allowed 137 failure entities to yield 700 pre-failure windows (400 Critical, 300 Warning).

### 5.2 Parent-Job Leakage in Development Splits
In [`src/splitting/leakage_safe_split.py`](file:///c:/Users/hp/Desktop/Research%20project/src/splitting/leakage_safe_split.py), the `entity_grouped` strategy partitioned examples by container identity (`job_id_task_index`).
- **Entity Disjointness:** Strictly 0 overlapping entities between train, val, and test.
- **Parent Job Confounding:** **24 of 32 validation parent jobs (75.0%)** and **25 of 31 test parent jobs (80.6%)** were already present in the training set!
- **Consequence:** Tasks sharing a parent job run identical application code. Deep models memorized the specific resource signatures of these batch applications, creating inflated validation scores that collapsed when tested on unseen parent jobs in the confirmatory holdout.

### 5.3 The Confirmatory Holdout is Spent
The Phase 6.4 confirmatory holdout (5,511 windows) has been fully unblinded, audited, and analyzed. Under strict scientific protocol, **this dataset is permanently retired**. It cannot be rescored with adjusted thresholds or cleaned labels to declare a new "winner."

---

# PART 6 — Deep-Dive Model Architecture Audit

| Model Name | What It Learns | Input Shape & Features | Key Architecture Details | Training Parameters & Loss Weights | Strengths & Failure Modes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Random Forest** | Orthogonal threshold splits on instantaneous resource metrics. | $[N, 31]$ (final time step $t_{\text{pred}}$) | 100 trees, max depth 15, balanced class weights (`class_weight='balanced'`). | Effective loss weights: Normal 1.0, Warning 17.0, Critical 11.5. | **Best Macro-F1 (0.6111);** resilient to missingness; but lacks sequence awareness and fails operational FPR gate (21.39%). |
| **Temporal LSTM** | Recurrent hidden-state accumulation over 5 steps. | $[N, 5, 31]$ (3D sequence tensor) | 2 LSTM layers, $d_{\text{hidden}}=64$, dropout 0.2, linear head (64 $\to$ 64 $\to$ 3). | AdamW ($10^{-3}$), fixed weights $[1.0, 1.5, 2.0]$, patience 10. | High episode recall (78.7%), but massive false alarm rates (FPR 26.88%, FAR 41.92%); bypasses Warning to trigger Critical. |
| **Compact Transformer** | Multi-head self-attention across 5 historical steps. | $[N, 5, 31]$ (3D sequence tensor) | Linear proj (31 $\to$ 32), sinusoidal pos enc, 2 encoder layers, 4 heads, $d_{\text{ff}}=64$. | AdamW ($5 \times 10^{-4}$), fixed weights $[1.0, 1.5, 2.0]$, patience 10. | **Softmax Saturation:** Median $P(\text{Critical}) = 0.9934$ on false alarms; overfits to missing CPI counters; FPR 20.27%, FAR 31.82%. |
| **Representation Hybrid** | Tree ensemble on fused neural embeddings + tabular features. | $[N, 63]$ (32-dim Transformer embedding + 31 tabular features) | Frozen Transformer backbone extracts 32-dim latent vector; classified by Random Forest (100 trees). | Fitted on in-sample training embeddings; `class_weight='balanced'`. | **Worst Operational Performer:** Severe in-sample embedding overfitting; Window FPR 34.65%, Entity FAR 64.14%. |

---

# PART 7 — Confirmatory Results & Operational Invariance

### 7.1 Confirmatory Performance Metrics (As Evaluated in Phase 6.4)

| Metric | Random Forest | Temporal LSTM | Compact Transformer | Representation Hybrid | Pre-Registered Gate |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Accuracy** | $80.20\%$ | $74.67\%$ | $81.20\%$ | $67.43\%$ | Baseline |
| **Macro-F1** | **0.6111** | 0.4425 | 0.4479 | 0.3924 | Tie-breaker |
| **Critical Precision / Recall** | $0.2452$ / $0.7225$ | $0.2185$ / $0.8775$ | $0.2644$ / $0.8100$ | $0.1874$ / $0.7800$ | Unbalanced |
| **Warning Precision / Recall** | $0.2971$ / $0.4700$ | $0.3448$ / $0.0667$ | $0.3486$ / $0.1267$ | $0.1479$ / $0.4067$ | Skipped by deep models |
| **Episode Detection Recall** | $74.47\%$ (105/141) | **78.72%** (111/141) | $76.60\%$ (108/141) | $77.30\%$ (109/141) | Primary ranking |
| **Mean / Median Lead Time** | 12.3 / 11.2 min | **13.5 / 13.8 min** | 12.6 / 12.5 min | 12.4 / 11.5 min | Target $\ge 10\text{ min}$ |
| **Window FPR (Pooled Normal)** | **21.39%** (1029/4811) | $26.88\%$ (1293/4811) | **20.27%** (975/4811) | $34.65\%$ (1667/4811) | **FAIL (Gate: $\le 5.0\%$)** |
| **Entity FAR (Pure-Normal)** | $35.35\%$ (70/198) | $41.92\%$ (83/198) | **31.82%** (63/198) | $64.14\%$ (127/198) | **FAIL (Gate: $\le 12.0\%$)** |
| **Operational Gate Status** | **FAILED BOTH** | **FAILED BOTH** | **FAILED BOTH** | **FAILED BOTH** | **NO WINNER** |

### 7.2 Proof of Operational Invariance Across Censoring Corrections
Our diagnostic counterfactual analysis evaluated what happens when the 149 post-failure windows or all 296 failure-cohort normal windows are removed:

```
Window False Positive Rate (FPR) Comparison:
  Scenario 0 (As Frozen, N=4,811):     RF: 21.39% | LSTM: 26.88% | TF: 20.27% | Hybrid: 34.65%  [Gate <= 5.0%]
  Scenario 1 (Censoring 149 Post-Fail): RF: 21.24% | LSTM: 25.59% | TF: 19.31% | Hybrid: 33.91%  [Gate <= 5.0%]
  Scenario 2 (Pure-Normal Only, N=4,515): RF: 21.66% | LSTM: 23.37% | TF: 17.43% | Hybrid: 32.67%  [Gate <= 5.0%]

Entity False Alarm Rate (FAR) Comparison (Evaluated strictly on 198 Pure-Normal tasks):
  All Scenarios (0, 1, 2, 3):          RF: 35.35% | LSTM: 41.92% | TF: 31.82% | Hybrid: 64.14%  [Gate <= 12.0%]
```
**Conclusion:** Entity FAR is mathematically invariant (0.00% change) because it depends solely on pure-normal tasks. Window FPR drops at most to $17.43\%$, remaining $3.5\times$ above the allowable $5.0\%$ ceiling. **Every model fails both gates in every scenario.**

---

# PART 8 — False-Alarm Mechanisms & Confounding

1. **The Argmax Trap:** In a 3-class system under uniform argmax, the model chooses Critical whenever $P(\text{Critical}) > P(\text{Normal})$ and $P(\text{Critical}) > P(\text{Warning})$. If non-critical mass is split ($P(\text{Normal}) \approx 0.33, P(\text{Warning}) \approx 0.33$), Critical alerts trigger at predicted probabilities as low as **$0.3364$ to $0.3587$**.
2. **Softmax Saturation:** In the Compact Transformer, cross-entropy training drove output logits to extremes on false alarms (median $P(\text{Critical}) = 0.9934$). The model is intensely overconfident on out-of-distribution data.
3. **CPI Missingness Confounding:** False alarms exhibited extreme elevation in `cpi_missing_fraction_6` ($d = +1.205$ to $+2.953$). In cluster traces, missing CPI is not a physical hardware fault; it occurs when Borg deschedules a container or when a thread is idle. Models learned a spurious artifact: *"if telemetry is missing, sound an alarm."*
4. **Parent-Job Concentration:** Just **3 parent jobs accounted for 50.63% (RF) and 52.21% (Transformer) of all false positives**. Gini coefficients exceeded $0.79$, proving that false alarms were concentrated in a tiny fraction of unusual batch workloads.

---

# PART 9 — Statistical Validity & Methodology Audit

- **Bootstrap Resampling:** Multi-level hierarchical bootstrap (entity-level clustering with 500–1000 replicates) correctly preserved within-entity correlation.
- **Parent-Job Cluster Collapse:** When bootstrap clustering was elevated from entity level to parent-job level, the 95% confidence interval for CPI missingness in Random Forest collapsed from $[+0.941, +1.991]$ down to $[+0.096, +2.689]$. This proved that the missingness signal was heavily confounded by parent-job workload types.
- **Testing Trap:** Test suites in earlier phases verified software execution and array shapes, but failed to catch scientific protocol violations because assertions encoded flawed assumptions (e.g., asserting that non-null TTF is positive rather than asserting that post-failure steps have label -1).

---

# PART 10 — Contradiction Register & Reconciliation

| Issue | Conflicting Sources | Empirical Ground Truth | Status & Resolution |
| :--- | :--- | :--- | :--- |
| **Post-Failure Censoring** | Phase 2–5 reports ("zero post-failure telemetry") vs Phase 6.8 Audit | 149 post-failure windows were included in confirmatory data; >1,400 across train/val/test. | **Defect Confirmed:** Omitted past-failure check in `target_generator.py`. |
| **Normal Window Partition** | Phase 6.6 (140 pre / 156 post) vs Phase 6.7/6.8 (147 pre / 149 post) | Correct partition is 147 pre-fail and 149 post-fail. 7 windows traced to repeated failures in tasks `6224307516_2` and `6224307516_3`. | **Resolved & Corrected:** Raw trace event matching down to 0.1s. |
| **Window Counts: 5,727 vs 5,511** | Phase 6.2 summary (5,727) vs Manifest (5,511) | 5,727 included 61 non-pre-failing census tasks. Manifest encoded 5,511 primary windows. | **Resolved:** Amendment 2.1 locked 5,511 as the sole primary population. |
| **Entity Counts: 337 vs 335** | Sampling plan (337) vs Manifest (335) | 2 sampled pure-normal controls had $< 5$ steps of total telemetry and were excluded. | **Resolved:** Amendment 2.1 ratified 335 evaluable tasks (137 fail + 198 normal). |
| **Neural Class Loss Weights** | Phase 6.6 (claimed ~10x weights) vs Code | LSTM and Transformer used explicit weights $[1.0, 1.5, 2.0]$. Only RF used balanced (~11.5x). | **Resolved:** Phase 6.7 refuted the claim from source training code. |
| **Probability Calibration** | Reports citing "posterior probabilities" vs Models | Models output uncalibrated scores; no calibration procedure was ever applied. | **Resolved:** Disclaimed. Softmax saturation documented as uncalibrated confidence. |

---

# PART 11 — Research Novelty & Academic Contribution

### 11.1 What This Study Demonstrates (The Real Novelty)
1. **Rigorous Empirical Benchmark:** Provides one of the most transparent, leakage-safe empirical evaluations of deep temporal sequence architectures vs. classical ensembles on production cloud traces.
2. **The Failure of Uniform Argmax in Imbalanced Cloud Ops:** Demonstrates that standard 3-class argmax decision rules fail catastrophically in operational cloud settings, causing severe false alarm cascades.
3. **Forensic Identification of Observational Confounding:** Uncovers that deep sequence models overfit to collector missingness artifacts (CPI omission) and parent-job batch profiles rather than learning genuine degradation signatures.
4. **Data-Censoring Pitfalls in Trace Processing:** Identifies how retrospective sliding window generators can silently leak post-failure recovery data if lifecycle state machines fail to verify prior failure history.

### 11.2 What the Study Fails to Demonstrate
- It does **not** demonstrate that deep learning improves cloud failure prediction over simple tree ensembles.
- It does **not** demonstrate that the Representation Hybrid architecture is viable.
- It does **not** demonstrate that models can achieve operational deployment gates ($\le 5\%$ FPR, $\le 12\%$ FAR) without cost-sensitive threshold tuning.

---

# PART 12 — Recommended Next Steps (Prioritized Roadmap)

### Phase A: Essential Pipeline Repairs (Do First)
1. **Patch `TargetLabelGenerator`:** Add explicit past-failure checks (`task_events["time_sec"] <= t`) to assign `FailureLabel.UNKNOWN` (-1) to any post-failure observation.
2. **Unit & Regression Test Suite:** Implement tests asserting that post-failure steps, repeated failures, and contaminated histories are strictly excluded.
3. **Parent-Job Grouped Splitting:** Update `LeakageSafeSplitter` to guarantee that all tasks belonging to the same `job_id` reside in the same split.

### Phase B: Corrected Development Experiments
4. **Generate Clean Development Dataset:** Regenerate train/val/test splits in `data/processed/clean_dev/` without overwriting historical datasets.
5. **Cost-Sensitive Decision Thresholding:** Replace uniform argmax with a utility-maximizing threshold optimizer tuned strictly on validation data.
6. **Temperature Scaling & Calibration:** Implement temperature scaling on validation logits to calibrate output probabilities.

### Phase C: Future Research & Publication (Immediate)
7. **Write the Research Paper as a High-Integrity Forensic Audit:** Reframe the manuscript to focus on the empirical comparison, the operational failure modes, the confounding of missing telemetry, and the lessons learned in data censoring.
8. **Permanent Cohort Quarantine:** Lock the Phase 6 confirmatory holdout as historical; do not tune models on it. If a new confirmatory experiment is desired, sample an untouched cohort from an unobserved time period.

---

# PART 13 — Beginner-Friendly Explanation & Professor Interview Guide

### 13.1 What are we building, in plain English?
Imagine you are managing an airport control tower. Planes (cloud tasks) are taking off and landing continuously. Most flights land normally, but occasionally an engine fails (task crash). We are trying to build an automated radar warning system that listens to engine sensor telemetry (CPU, memory, disk activity) and sounds a buzzer 15 to 30 minutes *before* a plane crashes, giving the ground crew time to prepare emergency equipment.

### 13.2 How does it work, step by step?
1. Every 5 minutes, we take a snapshot of the container's vital signs (CPU, memory, disk I/O, instructions per cycle).
2. We look at the last 5 snapshots (a 25-minute movie of the container's health).
3. We feed this 25-minute movie into our AI models (Random Forest, LSTM, Transformer).
4. The AI outputs three numbers that add up to 100%: the chance it's Normal, the chance of a Warning (crash in 15–30 min), and the chance of Critical (crash in 0–15 min).
5. If the AI thinks a crash is the most likely outcome, it sounds the alarm.

### 13.3 What went wrong? Why did the models fail?
Imagine your radar warning buzzer goes off 35 times out of every 100 normal flights when nothing is wrong! The airport manager would turn off the buzzer because it causes complete chaos.
- **The False Alarm Disaster:** Our models triggered false alarms on 20% to 35% of all normal 5-minute windows, and alerted on 32% to 64% of healthy tasks! The pre-registered operational rule said false alarms must be under 5% of windows and under 12% of tasks.
- **The Argmax Trap:** In a 3-choice problem, if the AI is confused and gives Normal 33%, Warning 33%, and Critical 34%, the computer picks Critical because 34% is the largest number—even though the AI was 66% sure the flight was NOT in critical danger!
- **Missing Sensor Glitches:** When a container was idling, the Google hardware sensor stopped recording "Cycles Per Instruction" (CPI). The AI mistook this missing sensor data for a dying container and panicked.

### 13.4 Why is the Phase 6.8 finding important?
We discovered that our automated labeling script had a blind spot: when checking whether a window was Normal, it only looked forward in time to see if a crash was coming. It forgot to look backward!
If a task crashed, and the server spent 45 minutes cleaning up its memory, our script labeled those cleanup windows as "Normal"! In reality, the plane had already crashed. This accidentally leaked 149 post-crash windows into our final evaluation, and over 1,000 into our training data.

### 13.5 Does this mean our research failed?
**No, absolutely not!** In real science, discovering *why* complex models fail in real-world deployment is just as valuable as claiming high accuracy in a toy experiment. Hundreds of published papers claim 95% accuracy because they test models on easy, artificial datasets where parent jobs overlap. By conducting an honest confirmatory evaluation and a forensic audit, we uncovered structural reasons why deep temporal models fail in production cloud clusters. This is the hallmark of real scientific integrity.

---

### 13.6 Questions a Professor Might Ask You (and How to Answer Them)

#### Q1: "Why did your Random Forest baseline beat your advanced Transformer and LSTM models?"
> **Answer:** "In our experiments, Random Forest achieved a Macro-F1 of 0.61, while LSTM and Transformer achieved 0.44 to 0.45. Deep sequence models have thousands of parameters that easily overfit to spurious collection artifacts—such as missing CPI telemetry and specific parent-job resource profiles. Random Forest uses orthogonal threshold splits on instantaneous metrics, making it far more robust to uncalibrated probability drift than deep self-attention networks."

#### Q2: "Why didn't you just change the decision threshold from argmax to something higher like 0.80?"
> **Answer:** "Under our pre-registered scientific protocol, all decision thresholds were locked prior to evaluating the holdout to prevent p-hacking and optimistic bias. Post-hoc analysis showed that Compact Transformer probabilities saturated above 0.99 on false alarms due to uncalibrated softmax cross-entropy optimization. Simply shifting the threshold post hoc on uncalibrated outputs without a proper validation-calibration split would violate confirmatory evaluation integrity."

#### Q3: "What happened in Phase 6.8 regarding the 149 post-failure windows?"
> **Answer:** "Our forensic audit discovered that `TargetLabelGenerator` lacked a past-failure condition when future failures were absent. Consequently, 149 post-failure recovery windows were mislabeled as Normal. However, we performed a counterfactual diagnostic sensitivity analysis proving that our operational conclusion is completely invariant: even when all 149 windows are censored, all four models still fail the operational gates by wide margins. The verdict `NO_OPERATIONAL_WINNER` is mathematically robust."

#### Q4: "What is your primary scientific contribution if no model passed the operational gate?"
> **Answer:** "Our study provides a rigorous empirical audit demonstrating three critical findings: first, deep sequence models fail under uniform argmax when deployed on highly imbalanced cloud telemetry; second, parent-job correlation creates severe observational confounding if not strictly separated during splitting; and third, retrospective sliding window generation requires strict past-event verification to prevent post-failure data contamination."

---

## Final Verification & Deliverable Summary

All discovery artifacts have been written to the repository:
- Master Report: [`reports/full_project_discovery_audit.md`](file:///c:/Users/hp/Desktop/Research%20project/reports/full_project_discovery_audit.md)
- Project Inventory: [`results/project_discovery/project_inventory.json`](file:///c:/Users/hp/Desktop/Research%20project/results/project_discovery/project_inventory.json)
- Claims-Evidence Matrix: [`results/project_discovery/claims_evidence_matrix.csv`](file:///c:/Users/hp/Desktop/Research%20project/results/project_discovery/claims_evidence_matrix.csv)
- Contradiction Register: [`results/project_discovery/contradiction_register.csv`](file:///c:/Users/hp/Desktop/Research%20project/results/project_discovery/contradiction_register.csv)
- Prioritized Next Steps: [`results/project_discovery/next_steps_plan.md`](file:///c:/Users/hp/Desktop/Research%20project/results/project_discovery/next_steps_plan.md)
- Unverified Items: [`results/project_discovery/unverified_items.md`](file:///c:/Users/hp/Desktop/Research%20project/results/project_discovery/unverified_items.md)

All 159 repository unit and regression tests pass cleanly. No existing code, datasets, checkpoints, or historical reports were modified.
