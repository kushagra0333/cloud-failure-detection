# Phase 7B.1 — Evaluation Protocol Review and Methodological Correction

**Author:** Antigravity (AI Research Assistant)  
**Date:** October 28, 2026  
**Local Time:** 00:05:00+05:30  
**Phase:** Phase 7B.1 (Pre-Training Evaluation Protocol Review & Methodological Hardening)  
**Status:** Complete — Read-Only Protocol Audit  
**Official Verdict:** **`PROTOCOL READY FOR TRAINING APPROVAL`**

---

## 1. Executive Summary

This report performs a targeted, read-only audit of the Phase 7B evaluation protocol and resolves five critical methodological questions prior to model training.

### Major Inconsistencies Identified and Resolved
1. **Test-Set Model Selection Flaw Eliminated:** The original protocol selected the "Operational Winner" by taking the highest Test Macro-F1 among models passing test-set gates. This leaked the test set into model selection. We have replaced this with a **strict three-stage evaluation architecture**: model selection, threshold calibration, and gate qualification occur **strictly on the Validation split**. Exactly one model is frozen as the *Validated Operational Candidate* ($M^*$) before unblinding Test. Test evaluation is strictly confirmatory for $M^*$.
2. **Parent-Job False-Alarm Rate Inconsistency Resolved:** The original protocol mentioned a 15% Parent-Job FAR target but omitted it from Gate 2. An audit of denominators revealed that the Candidate A Test split contains **only 3 pure-normal parent jobs** (and Validation contains **4**). A 15% threshold is mathematically impossible for non-zero alerts because a single false alarm causes FAR to jump to **33.3%** ($1/3$). We have resolved this by keeping Gate 2 on **Pure-Normal Entity FAR ($\le 10\%$)** and **Normal-Window FPR ($\le 5\%$)**, while officially designating **Parent-Job FAR as a Secondary Diagnostic Metric** reported with exact integer counts.
3. **Mathematical Precision for Episode Detection:** Specified the complete episode-matching algorithm with formal pseudocode and a 4-scenario hand-checkable walkthrough covering correct detections, missed failures, post-failure alerts, and repeated failures.
4. **Cluster Bootstrap Stability on 11 Clusters:** Evaluated the parent-job cluster bootstrap on the 11 test clusters ($B=1,000$ simulation). Identified that Job `6127642779` (660 windows) causes window sample sizes to vary from 122 to 4,114, and in 3.1% of replicates pure-normal jobs are omitted. Defined explicit missing-denominator rules and mandated dual reporting (cluster bootstrap CIs alongside entity-level CIs).
5. **Preprocessing & Feature Lookback Verified:** Documented the training-only scaler fitting, verified that `df.groupby("entity_id")` ensures zero carryover between container tasks, audited the 81.3% hardware counter (CPI) missingness, and verified the 50-minute deepest lookback.

---

## 2. Issue 1: Correcting the Test-Set Model-Selection Problem

### 2.1 The Methodological Violation in Protocol v1
Protocol v1 defined:
> *"A model is declared an OPERATIONAL_WINNER if and only if it satisfies both gates simultaneously on the Test split... If multiple models pass, the model with the highest Macro-F1 on the Test split wins."*

**Why this is scientifically invalid:**
If four models are trained and evaluated on the Test split, and the "winner" is selected based on which model achieved the highest score on the Test split, the Test split is being used for **model selection** (hyper-model optimization). This causes:
- Selection bias (winner's curse): the model with the most favorable random test sample variation is selected.
- Optimism bias: the reported test score of the winner is no longer an unbiased estimate of future generalization.

---

### 2.2 The Revised Three-Stage Evaluation Architecture (Protocol v2)

To eliminate test leakage, Protocol v2 establishes three strictly sequential, firewalled stages:

```
┌────────────────────────────────────────────────────────────────────────┐
│ STAGE 1: Model Training & Checkpointing (Train Split Only, N=3,278)     │
│ - Train Random Forest, LSTM, Transformer, Hybrid strictly on Train.    │
│ - Select epoch checkpoints and tune thresholds strictly on Val.       │
│ - TEST SPLIT REMAINS SEALED AND UNTOUCHED.                            │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ STAGE 2: Validation Gate Qualification & Candidate Selection (Val, N=657)│
│ - Evaluate all 4 models against Validation Gate 1 and Validation Gate 2│
│   * Gate 1: Val Episode Recall >= 70.0% AND Val Critical Recall >= 50% │
│   * Gate 2: Val Entity FAR <= 10.0% AND Val Window FPR <= 5.0%        │
│ - SELECTION RULE:                                                      │
│   Among models passing both Val gates, select the ONE model with the   │
│   highest Validation Macro-F1 as the "Validated Candidate" (M*).      │
│   If zero models pass both Val gates -> NO_VALIDATED_OPERATIONAL_CAND. │
│ - PERMANENT FREEZE:                                                    │
│   Model M* (checkpoint weights, scaler, thresholds) is locked.        │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ STAGE 3: One-Shot Confirmatory Test Evaluation (Test Split, N=1,025)   │
│ - Score M* on Test split once:                                         │
│   * Test Gate 1: Test Episode Recall >= 70% AND Test Crit Recall >= 50%│
│   * Test Gate 2: Test Entity FAR <= 10% AND Test Window FPR <= 5.0%    │
│ - FINAL VERDICT:                                                       │
│   * If M* passes both Test gates -> OPERATIONAL_WINNER.                │
│   * If M* fails either Test gate -> NO_OPERATIONAL_WINNER.             │
│ - DESCRIPTIVE BENCHMARKS:                                              │
│   The other 3 models are reported solely as descriptive comparisons.   │
│   THEY CANNOT REPLACE M* AS WINNER UNDER ANY CIRCUMSTANCES.            │
└────────────────────────────────────────────────────────────────────────┘
```

**Key Guarantee:** The operational winner claim depends entirely on whether the *pre-selected validation winner* confirms its performance on the held-out test split. The test set is **never used to select a model or tune a threshold**.

---

## 3. Issue 2: Resolving the Parent-Job False-Alarm Gate Inconsistency

### 3.1 Empirical Audit of Pure-Normal Denominators
We analyzed the exact counts of pure-normal parent jobs and pure-normal entities in Candidate A:

| Split | Total Parent Jobs | Failing Parent Jobs | **Pure-Normal Parent Jobs** | Total Entities | Failing Entities | **Pure-Normal Entities** | Normal Windows |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **TRAIN** | 51 | 34 | **17 jobs** (33.3%) | 284 | 131 | **153 entities** | 2,797 |
| **VAL** | 12 | 8 | **4 jobs** (33.3%) | 69 | 20 | **49 entities** | 560 |
| **TEST** | 11 | 8 | **3 jobs** (27.3%) | 59 | 21 | **38 entities** | 921 |

### 3.2 The Mathematical Conflict
If `Pure-Normal Parent-Job FAR <= 15.0%` were enforced as a mandatory operational gate:
- In the Test split, the denominator is **3 parent jobs**:
  - 0 jobs alert $\to 0/3 = 0.0\%$ (Passes $\le 15\%$)
  - 1 job alerts $\to 1/3 = \mathbf{33.3\%}$ (**Fails $\le 15\%$**)
  - 2 jobs alert $\to 2/3 = \mathbf{66.7\%}$ (**Fails**)
- In the Validation split, the denominator is **4 parent jobs**:
  - 0 jobs alert $\to 0/4 = 0.0\%$ (Passes $\le 15\%$)
  - 1 job alerts $\to 1/4 = \mathbf{25.0\%}$ (**Fails $\le 15\%$**)

**Finding:** Because individual parent jobs can be massive (e.g., Job `6127642779` has 33 tasks and 660 windows), a single false positive across 660 windows would cause that job to alert, instantly jumping the Job FAR to 33.3% and causing immediate gate failure. With a denominator of only 3 jobs, a 15% threshold is practically an unrealistic "zero-defect" gate.

### 3.3 The Methodological Resolution
To establish a statistically continuous, defensible gate while retaining complete visibility into parent-job false alarms:
1. **Primary Operational Gate 2:**
   $$\text{Pure-Normal Entity FAR} \le 10.0\% \quad \text{AND} \quad \text{Normal-Window FPR} \le 5.0\%$$
   - Entity FAR has a robust denominator ($N=49$ in Val, $N=38$ in Test). In Test, allowing up to 3 false-alarm entities gives $3/38 = 7.89\% \le 10\%$.
   - Window FPR has a large denominator ($N=560$ in Val, $N=921$ in Test).
2. **Pure-Normal Parent-Job FAR Designated as Secondary Diagnostic Metric:**
   - Parent-Job FAR will be computed, reported, and presented with exact integer fractions (e.g. `0/3`, `1/3`) and bootstrap confidence intervals.
   - It is removed from the binary gate pass/fail criteria to prevent extreme discretization artifacts.
   - The written protocol, JSON schema, and decision logic now agree 100%.

---

## 4. Issue 3: Precise Specification of Episode Detection

### 4.1 Formal Matching Algorithm

```python
def compute_episode_metrics(metadata_df, y_pred):
    """
    Inputs:
      metadata_df: DataFrame containing columns:
        - 'entity_id': task identifier (job_id_task_index)
        - 'prediction_timestamp': window prediction anchor t (in seconds)
        - 'time_to_failure': seconds to next failure (NaN for pure-normal)
        - 'label': ground truth in {0: NORMAL, 1: WARNING, 2: CRITICAL}
      y_pred: Array of model predictions in {0, 1, 2}
      
    Rules:
      1. An observation is eligible for episode detection iff label in {1, 2}.
         Under Phase 7A.2 repair, this guarantees:
         t_pred < t_fail AND 0 < (t_fail - t_pred) <= 1800.0s (30 minutes).
      2. Group eligible rows by Episode ID: (entity_id, fail_timestamp), where:
         fail_timestamp = round(prediction_timestamp + time_to_failure, 1).
      3. For each episode group:
         - An alert window is any row with y_pred in {1, 2}.
         - If alert_windows is non-empty:
             detected_episodes += 1
             first_alert_time = min(alert_windows['prediction_timestamp'])
             lead_time = fail_timestamp - first_alert_time
             lead_times.append(lead_time)
      4. Episode Recall = detected_episodes / total_episodes.
    """
```

### 4.2 Verifications of Episode Invariants
- **Strictly Pre-Failure:** Guaranteed because $t_{\text{pred}} < t_{\text{fail}}$ for all rows with $y \in \{1, 2\}$.
- **Strictly Within 30-Minute Horizon:** Only windows with $\tau \le 1,800\text{s}$ receive $y \in \{1, 2\}$.
- **Post-Failure Alerts Excluded:** Any window at or after failure has $y = -1$ (censored/dropped in Phase 7A.2) or $y=0$. It is not in the eligible pre-failure pool and cannot credit episode recall.
- **Single Credit Per Episode:** An episode is counted as detected at most once (`detected_episodes += 1`), regardless of whether 1, 2, or 5 consecutive windows triggered an alert.
- **Consistent Lead Time:** Measured from the *first* alert in the 30-minute window to $t_{\text{fail}}$:
  $$\tau_{\text{lead}} = t_{\text{fail}} - \min_{w \in \text{Alerts}} t_{\text{pred}, w}$$

---

### 4.3 Hand-Checkable Walkthrough (4 Benchmark Scenarios)

Consider a scenario with 4 tasks evaluated over time ($W_{\text{warn}} = 1,800\text{s}$, 300s window spacing):

| Entity ID | Window Time $t$ | Ground Truth $y$ | Failure Time $t_{\text{fail}}$ | Predicted $\hat{y}$ | In Pre-Failure Pool? | Episode ID | Episode Status / Lead Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Case 1: Correct Detection** | | | | | | | |
| `Task_A` | 5,400s | 1 (Warning) | 6,000s | 0 (Normal) | Yes | `(Task_A, 6000)` | No alert yet |
| `Task_A` | 5,700s | 2 (Critical) | 6,000s | 2 (Critical) | Yes | `(Task_A, 6000)` | **First Alert!** Lead time = $6000 - 5700 = 300\text{s}$ |
| `Task_A` | 6,000s | *Failure Event* | 6,000s | — | — | — | **Result: Episode DETECTED, Lead Time = 300s** |
| **Case 2: Missed Episode** | | | | | | | |
| `Task_B` | 6,600s | 1 (Warning) | 7,200s | 0 (Normal) | Yes | `(Task_B, 7200)` | Missed |
| `Task_B` | 6,900s | 2 (Critical) | 7,200s | 0 (Normal) | Yes | `(Task_B, 7200)` | Missed |
| `Task_B` | 7,200s | *Failure Event* | 7,200s | — | — | — | **Result: Episode MISSED (0 lead time)** |
| **Case 3: Post-Failure Alert** | | | | | | | |
| `Task_C` | 5,100s | 1 (Warning) | 5,400s | 0 (Normal) | Yes | `(Task_C, 5400)` | Missed pre-failure alert |
| `Task_C` | 5,400s | *Failure Event* | 5,400s | — | — | — | Failure occurs |
| `Task_C` | 5,700s | 0 (Post-fail) | None | 2 (Critical) | **No** ($y=0$) | None | **Result: False Positive Window! (NOT episode detection)** |
| **Case 4: Repeated Failures** | | | | | | | |
| `Task_D` | 5,100s | 2 (Critical) | 5,400s | 1 (Warning) | Yes | `(Task_D, 5400)` | **Detected Ep 1!** Lead time = $5400 - 5100 = 300\text{s}$ |
| `Task_D` | 5,400s | *Fail 1* | 5,400s | — | — | — | First failure concludes |
| `Task_D` | 8,100s | 1 (Warning) | 8,700s | 0 (Normal) | Yes | `(Task_D, 8700)` | Missed pre-failure window for Ep 2 |
| `Task_D` | 8,400s | 2 (Critical) | 8,700s | 2 (Critical) | Yes | `(Task_D, 8700)` | **Detected Ep 2!** Lead time = $8700 - 8400 = 300\text{s}$ |
| `Task_D` | 8,700s | *Fail 2* | 8,700s | — | — | — | Second failure concludes |
| `Task_D` Summary | — | — | — | — | — | — | **Result: 2 distinct episodes, BOTH detected independently** |

---

## 5. Issue 4: Audit of Uncertainty Estimation

We conducted an empirical simulation of the proposed **Parent-Job Cluster Bootstrap** on the 11 test parent jobs ($B=1,000$ iterations).

### 5.1 Simulation Findings on 11 Clusters
- **Class Coverage:**
  - Critical Class missing replicates: **0 out of 1,000 (0.0%)**
  - Warning Class missing replicates: **0 out of 1,000 (0.0%)**
  - Pure-Normal Jobs missing replicates: **31 out of 1,000 (3.1%)**
- **Window Count Instability:**
  Because Job `6127642779` contains 660 windows (64.4% of the test set), resampling 11 jobs with replacement causes huge sample size fluctuations:
  - Minimum sample size: **122 windows** (when Job `6127642779` is sampled 0 times)
  - Maximum sample size: **4,114 windows** (when Job `6127642779` is sampled 5 times)
  - Median sample size: **958 windows**

### 5.2 Mathematical Protocol for Undefined Denominators
When a bootstrap replicate samples zero pure-normal jobs (3.1% of replicates), the denominator for Pure-Normal Job FAR is 0 ($0/0$).
- **Rule:** If a replicate has an undefined denominator, that metric is marked `NaN` for that replicate. Percentiles $[2.5\%, 97.5\%]$ are computed over the valid replicates, and the valid fraction is explicitly reported (e.g., `Valid Replicates: 969/1000`).

### 5.3 Limitations and Multi-Level Uncertainty Reporting
Resampling from only $J=11$ clusters cannot generate new independent information. Cluster bootstrap intervals with $J=11$ will be wide and primarily reflect sensitivity to large jobs.
To avoid misleading readers, Protocol v2 mandates **dual uncertainty reporting**:
1. **Primary Cluster Bootstrap 95% CIs** (resampling unit = parent job, capturing intra-cluster correlation).
2. **Secondary Entity-Level Bootstrap 95% CIs** ($N=59$ entities, capturing task-level variation).
3. **Point Estimates and Exact Numerators/Denominators** for every metric.

---

## 6. Issue 5: Verification of Feature Preprocessing and Missingness

We inspected the feature preprocessing pipeline in [src/features/engineer.py](file:///c:/Users/hp/Desktop/Research%20project/src/features/engineer.py) and [src/features/scaler.py](file:///c:/Users/hp/Desktop/Research%20project/src/features/scaler.py):

### 6.1 Training-Only Fitting and Scaling Parameters
- `LeakageSafeScaler.fit()` computes feature medians $a_j$, means $\mu_j$, and standard deviations $\sigma_j$ **strictly on the Candidate A Training split** ($N=3,278$ windows, 16,390 steps).
- Validation and Test splits are transformed using the frozen training parameters:
  $$z_{i,j} = \frac{x_{i,j} - \mu_{j,\text{train}}}{\sigma_{j,\text{train}} + \epsilon}$$
- Missing values in Validation and Test are imputed using $a_{j,\text{train}}$ (the training median).

### 6.2 Raw Telemetry Missingness
In Google Cluster Trace 2011:
- `cpu_utilization`: **0.0% missing**
- `memory_utilization`: **0.0% missing**
- `assigned_memory`: **0.0% missing**
- `cycles_per_instruction` (CPI): **81.3% missing** in raw trace (Google sampled hardware counters only on a minority of machines). Imputed via training median.

### 6.3 GroupBy Isolation (Zero Cross-Entity Carryover)
In [src/features/engineer.py:L48-L72](file:///c:/Users/hp/Desktop/Research%20project/src/features/engineer.py#L48-L72):
```python
grp = df.groupby("entity_id", sort=False)
df["cpu_utilization_lag_k"] = grp["cpu_utilization"].shift(k)
```
Because `shift(k)` is executed on the pandas GroupBy object, pandas strictly enforces that lag shifts **never cross entity boundaries**. The first $k$ rows of each container task receive `NaN`. There is zero carryover between entities.

### 6.4 The 50-Minute Deepest Lookback
- Window endpoints span $[t - 1200\text{s}, t]$ (20 minutes).
- Underlying 5 telemetry bins span $[t - 1500\text{s}, t]$ (25 minutes).
- At sequence step 0 ($t - 1200\text{s}$), lag-6 features look back an additional 6 steps ($1,800\text{s}$).
- Earliest raw telemetry required: $t - 1200\text{s} - 1800\text{s} = \mathbf{t - 3,000\text{ seconds (50 minutes)}}$.

---

## 7. Deliverables Summary

| Artifact Path | Format | Status | Description |
| :--- | :---: | :---: | :--- |
| [results/phase7b_1/phase7b_evaluation_protocol_v2.json](file:///c:/Users/hp/Desktop/Research%20project/results/phase7b_1/phase7b_evaluation_protocol_v2.json) | JSON | **Frozen v2.0** | Revised predeclared evaluation protocol |
| [results/phase7b_1/evaluation_protocol_review.json](file:///c:/Users/hp/Desktop/Research%20project/results/phase7b_1/evaluation_protocol_review.json) | JSON | **Complete** | Audit verification data and resolved issue register |
| [reports/phase7b_1/evaluation_protocol_review.md](file:///c:/Users/hp/Desktop/Research%20project/reports/phase7b_1/evaluation_protocol_review.md) | Markdown | **Complete** | Comprehensive methodological review report |

---

## 8. Exact Protocol Differences (v1.0 vs v2.0)

| Protocol Element | Original Protocol v1.0 | Revised Protocol v2.0 | Scientific Benefit |
| :--- | :--- | :--- | :--- |
| **Model Selection** | Winner selected by highest **Test** Macro-F1 among test gate passers | Winner candidate selected by highest **Validation** Macro-F1 among validation gate passers | Eliminates test-set selection bias (zero test leakage) |
| **Test Set Role** | Four models competed on test set to decide winner | Pre-selected model $M^*$ evaluated confirmatory on test; other 3 models reported as descriptive baselines | Preserves strict confirmatory integrity of test set |
| **Parent-Job FAR** | Listed as $\le 15\%$ target, omitted from Gate 2 | Formally designated as **Secondary Diagnostic Metric** (reported with counts & CIs) | Avoids discrete $1/3$ quantization failure on $N=3$ jobs |
| **Episode Detection** | High-level textual description | Formal pseudocode, pre-failure window filtering ($y \in \{1, 2\}$), single-credit rule | Mathematical clarity, eliminates ambiguity |
| **Bootstrap Audit** | Cluster bootstrap proposed without missing-denominator rule | Evaluated on 11 clusters; defined empty-replicate handling; mandated dual reporting | Transparent disclosure of cluster sample size limits |

---

## 9. Official Verdict and Next Step

### Official Phase 7B.1 Verdict
# **`PROTOCOL READY FOR TRAINING APPROVAL`**

### Summary of Verdict:
- All test-set selection leakage has been removed.
- Gate 2 and Parent-Job FAR inconsistencies have been resolved.
- Episode matching and lead times are mathematically defined and verified.
- Uncertainty estimation boundaries are audited and disclosed.
- Feature preprocessing and missingness are verified.

**I have paused execution in accordance with your instructions. Please review this revised evaluation protocol. Once you confirm, I will begin model retraining under these hardened boundaries.**
