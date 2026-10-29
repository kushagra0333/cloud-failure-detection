# Phase 7K — Protocol Hardening and Controlled Feature Ablation Report

**Project:** Failure Prediction in Google Borg Cluster Traces  
**Investigation Phase:** Phase 7K — Controlled Feature Ablation on Stabilized Sequence Representation  
**Date:** October 28, 2026  
**Auditor / Lead Investigator:** Advanced Agentic AI Assistant (Antigravity)  
**Governing Standard:** Multi-Phase Protocol Governance Framework  
**Evaluated Data Split:** Candidate A Train ($N = 3,278$) and Validation ($N = 657$) Data Only — STRICTLY NO TEST ACCESS  
**Final Executive Verdict:** **`NO OPERATIONAL WINNER` CONFIRMED (0 of 30 Models / 0 of 6,030 Operating Points Pass All Four Gates)**  
**Confirmatory Test Evaluation:** **STRICTLY PROHIBITED**

---

## Executive Summary

Phase 7K was executed as a controlled, validation-only investigation into whether expanding the feature representation beyond the 31-feature telemetry baseline could resolve the operational detection-versus-false-alarm trade-off established in Phases 7B–7J.

Following rigorous pre-execution temporal causality hardening, a 5-model ablation experiment was trained and evaluated afresh on Candidate A development data:
- **$M_0$ (Baseline, 31 features):** Existing telemetry baseline (Groups A–I).
- **$M_1$ (Peak Telemetry & Headroom, 36 features):** $M_0$ + 5 usage features (`max_cpu_rate`, `cpu_burstiness`, `max_memory`, `memory_headroom`, `total_page_cache`).
- **$M_2$ (Scheduler Metadata, 35 features):** $M_0$ + 4 static scheduler attributes (`priority`, `scheduling_class`, `req_cpu`, `req_ram`).
- **$M_3$ (Dynamic Preemption Events, 33 features):** $M_0$ + 2 preemption-history features (`eviction_count_prior`, `time_since_last_schedule`).
- **$M_4$ (Combined Eligible Set, 42 features):** $M_0$ + Groups J1 + J2 + J3 (11 additional features).

### Core Empirical Findings

1. **Step 1 Causal Audit:** All 6 temporal causality checks passed with empirical proof. Explicit OOM indicators were confirmed non-existent in Borg 2011; dynamic preemption events were strictly causal; and all terminal events (`FAIL`, `FINISH`, `KILL`) were strictly excluded.
2. **Controlled Training Execution:** Exactly 30 models (5 models $\times$ 2 class weighting conditions $\times$ 3 random seeds: 42, 123, 456) were trained under identical 2-layer LSTM architectures ($L=5$, $H=64$, Adam, lr=0.001, epochs=25) with train-only standardization.
3. **Operational Gate Failure:** Across all 30 trained models and all 6,030 threshold evaluation points ($\theta \in [0.000, 1.000]$, step 0.005), **exactly 0 models and 0 operating points satisfy both Gate 1 and Gate 2 simultaneously**.
4. **Impact of Additional Features:**
   - Group J1 ($M_1$) achieved the highest cross-seed Peak Macro-F1 ($0.3648 \pm 0.0334$ under moderate weighting vs. $0.3306 \pm 0.0261$ for $M_0$), but Critical Recall collapsed to $4.1 \pm 7.1\%$ when false alarms were controlled below gate limits.
   - Group J2 ($M_2$, scheduler metadata) provided negligible separation (Macro-F1 $0.3343 \pm 0.0101$).
   - Group J3 ($M_3$, preemption history) slightly improved episode detection at low thresholds (Episode Recall $31.7 \pm 14.4\%$), but inflated pure-normal Entity FAR to $25.2\%$ and Window FPR to $18.2\%$.
   - Group J4 ($M_4$, combined) achieved Peak Macro-F1 of $0.3600 \pm 0.0206$, but suffered the identical operational barrier.
5. **Final Decision:** The verdict **`NO OPERATIONAL WINNER`** is definitively preserved. No confirmatory test evaluation is authorized.

---

## Step 1 — Audit Temporal Causality and Schema Verification

Before any model training, we audited the raw trace schema, feature-generation code, and aggregation semantics across all 353 Candidate A development entities (284 Train, 69 Validation across 3,935 development windows). All 6 mandated checks passed:

### Check 1: Raw Usage Timestamp Semantics
- **Question:** Do raw usage timestamps denote interval start, interval end, or another reference time?
- **Finding:** In Google Borg 2011 traces ([`sample_task_usage*.csv.gz`](file:///c:/Users/hp/Desktop/Research%20project/data/raw/sample_task_usage.csv.gz)), `start_time` and `end_time` are microsecond timestamps defining the measurement interval. Across all 3,935 dev windows, interval duration ($end\_time - start\_time$) is strictly positive, with a nominal span of 300.0 seconds.
- **Causality Guarantee:** In our pipeline, `prediction_timestamp = t` is mapped to the end of the interval ($t = end\_time\_sec$). Thus, all telemetry in the window reflects execution strictly at or before $t$.

### Check 2: Observability of Peak and Cache Telemetry
- **Question:** Do `max_cpu_rate`, `max_memory`, `total_page_cache`, `unmapped_page_cache`, `local_disk_space`, and `max_disk_io_time` use only measurements fully observable by prediction time?
- **Finding:** These fields represent hardware counters and kernel page-table statistics aggregated over the identical $[t - 300\text{s}, t]$ window. They are logged simultaneously with `cpu_rate` and `canonical_memory` at interval completion ($t$).
- **Causality Guarantee:** 100% causally observable at prediction time $t$; 0.00% missingness after 300s resampling.

### Check 3: Raw Unit Compatibility for Derived Telemetry
- **Question:** Are `cpu_burstiness = max_cpu_rate - cpu_rate` and `memory_headroom = assigned_memory - max_memory` calculated from compatible raw units before scaling?
- **Finding:** In Borg 2011 traces:
  - `cpu_rate` and `max_cpu_rate` are both expressed as core fractions ($[0.0 - 1.0+]$). Empirical verification across all dev records confirms $max\_cpu\_rate \ge cpu\_rate$ in 100.0% of records. Thus, `cpu_burstiness` is strictly non-negative ($\ge 0$) and physically meaningful.
  - `assigned_memory`, `canonical_memory`, and `max_memory` are all normalized relative to total machine memory capacity. Verification confirms $max\_memory \ge canonical\_memory$ in 100.0% of records.
- **Causality Guarantee:** Both derived quantities are calculated from identical physical dimensions before applying standard scaling.

### Check 4: Static Scheduler Metadata Causality
- **Question:** Is static scheduler metadata available by the prediction timestamp and joined to the correct task?
- **Finding:** `scheduling_class` (0–3), `priority` (0–10), `req_cpu`, `req_ram`, and `req_disk` are recorded at task submission (`event_type = 0`, `SUBMIT`).
- **Causality Guarantee:** In 100% of dev windows (3,935/3,935), $t_{submit} \le t_{pred}$. Verification across the entire trace confirms 0 entities had modifications to these fields (`UPDATE_PENDING` and `UPDATE_RUNNING` counts are 0). They are strictly invariant and causally valid.

### Check 5: Dynamic Preemption Events Causality
- **Question:** Do `eviction_count_prior` and `time_since_last_schedule` use only events with timestamps at or before prediction time and associate with the correct task?
- **Finding:** Pre-prediction evictions (`event_type = 2`, `EVICT`) and reschedules (`event_type = 1`, `SCHEDULE`) were filtered strictly by $t_{event} \le t_{pred}$.
- **Causality Guarantee:** Zero causal timestamp violations across all 3,935 dev windows. The minimum elapsed duration to dispatch is $210.1$ seconds ($\ge 0$).

### Check 6: Target Leakage and Terminal Events Audit
- **Question:** Does any feature depend on future terminal events, future usage, labels, or post-failure records?
- **Finding:**
  - Abnormal failures (`event_type = 3`), normal completions (`event_type = 4`), and cancellations (`event_type = 5`) occur strictly at task termination ($t_{term} > t_{pred}$). Across all 3,935 dev windows, terminal events occurring at or before $t$ are exactly **0 out of 3,935 (0.00%)**.
  - Under Policy A, all post-failure telemetry is excluded.
  - An explicit `oom_indicator` **DOES NOT EXIST** in Borg 2011 trace format.
- **Causality Guarantee:** All future terminal timestamps and post-failure data are strictly excluded from the input feature space.

---

## Step 2 — Experimental Control and Preprocessing Pipeline

### 1. Feature Set Definitions

| Model | Descriptor | Feature Dim ($D$) | Constituent Feature Groups |
| :---: | :--- | :---: | :--- |
| **$M_0$** | Baseline Telemetry | **31** | Groups A–I: raw CPU/mem/disk, lags (1,2,3,6), diffs, rolling stats (3,6), OLS slope, exceedance, ratio, age, CPI missingness. |
| **$M_1$** | Peak & Headroom | **36** | $M_0$ (31) + Group J1 (5): `max_cpu_rate`, `cpu_burstiness`, `max_memory`, `memory_headroom`, `total_page_cache`. |
| **$M_2$** | Scheduler Metadata | **35** | $M_0$ (31) + Group J2 (4): `priority` (scaled /10), `scheduling_class` (scaled /3), `req_cpu`, `req_ram`. |
| **$M_3$** | Preemption History | **33** | $M_0$ (31) + Group J3 (2): `eviction_count_prior`, `time_since_last_schedule` (in minutes). |
| **$M_4$** | Combined Eligible Set | **42** | $M_0$ (31) + Groups J1 (5) + J2 (4) + J3 (2) = 11 additional features. |

### 2. Leakage-Free Standardization
- All preprocessing statistics (medians, means, standard deviations) were fitted **strictly on Candidate A Train ($D_{\text{train}}$, $N=3,278 \times 5 = 16,390$ observations)**.
- Scaler parameters were serialized to [`results/phase7k/fitted_preprocessors_manifest.json`](file:///c:/Users/hp/Desktop/Research%20project/results/phase7k/fitted_preprocessors_manifest.json) before training.
- Candidate A Validation tensors ($N=657 \times 5$) were transformed using the fitted training statistics without refitting.

---

## Step 3 & 4 — Controlled Training and Threshold Sweep Results

### 1. Training Setup
- **Architecture:** 2-layer LSTM (`input_dim = D`, `hidden_dim = 64`, `num_layers = 2`, `dropout = 0.2`, `seq_len = 5`).
- **Optimizer:** Adam ($\text{lr} = 0.001$, $\text{weight\_decay} = 0.0001$).
- **Batch Size:** 64.
- **Max Epochs:** 25.
- **Conditions:** Condition A (`[1, 1, 1]`, unweighted) and Condition B (`[1, 2, 2]`, moderate weighting).
- **Seeds:** 42, 123, 456.
- **Total Executed Runs:** 30 complete training and evaluation runs.

### 2. Complete Cross-Seed Aggregate Summary

The aggregate comparison across all 5 models and 2 loss conditions (mean $\pm$ standard deviation across seeds 42, 123, and 456) is cataloged in [`results/phase7k/ablation_cross_seed_aggregate.csv`](file:///c:/Users/hp/Desktop/Research%20project/results/phase7k/ablation_cross_seed_aggregate.csv):

| Model | Condition | Features ($D$) | Peak Macro-F1 | Ep. Recall at Peak F1 | Crit. Recall at Peak F1 | Ent. FAR at Peak F1 | Win. FPR at Peak F1 | Gate 1 Pass | Gate 2 Pass | **All Gates Passed** |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$M_0$** | Cond A (`[1,1,1]`) | 31 | $0.3355 \pm 0.0180$ | $11.7 \pm 5.8\%$ | $11.6 \pm 6.2\%$ | $12.9 \pm 1.2\%$ | $9.5 \pm 1.3\%$ | 0 / 3 | 0 / 3 | **0 / 3 (0%)** |
| **$M_0$** | Cond B (`[1,2,2]`) | 31 | $0.3306 \pm 0.0261$ | $8.3 \pm 7.6\%$ | $2.0 \pm 3.5\%$ | $8.2 \pm 7.1\%$ | $6.2 \pm 5.4\%$ | 0 / 3 | 1 / 3 | **0 / 3 (0%)** |
| **$M_1$** | Cond A (`[1,1,1]`) | 36 | $0.3375 \pm 0.0218$ | $18.3 \pm 12.6\%$ | $4.8 \pm 8.2\%$ | $9.5 \pm 13.0\%$ | $7.1 \pm 7.6\%$ | 0 / 3 | 1 / 3 | **0 / 3 (0%)** |
| **$M_1$** | Cond B (`[1,2,2]`) | 36 | **$0.3648 \pm 0.0334$** | $21.7 \pm 2.9\%$ | $4.1 \pm 7.1\%$ | $9.5 \pm 11.2\%$ | $6.1 \pm 2.4\%$ | 0 / 3 | 1 / 3 | **0 / 3 (0%)** |
| **$M_2$** | Cond A (`[1,1,1]`) | 35 | $0.3211 \pm 0.0097$ | $18.3 \pm 7.6\%$ | $12.2 \pm 12.2\%$ | $22.4 \pm 17.7\%$ | $16.6 \pm 8.1\%$ | 0 / 3 | 0 / 3 | **0 / 3 (0%)** |
| **$M_2$** | Cond B (`[1,2,2]`) | 35 | $0.3343 \pm 0.0101$ | $16.7 \pm 5.8\%$ | $12.2 \pm 12.2\%$ | $19.0 \pm 11.8\%$ | $13.9 \pm 7.2\%$ | 0 / 3 | 0 / 3 | **0 / 3 (0%)** |
| **$M_3$** | Cond A (`[1,1,1]`) | 33 | $0.3396 \pm 0.0216$ | $31.7 \pm 14.4\%$ | $6.8 \pm 11.8\%$ | $25.2 \pm 11.2\%$ | $18.2 \pm 6.9\%$ | 0 / 3 | 0 / 3 | **0 / 3 (0%)** |
| **$M_3$** | Cond B (`[1,2,2]`) | 33 | $0.3484 \pm 0.0141$ | $21.7 \pm 10.4\%$ | $2.0 \pm 3.5\%$ | $17.7 \pm 5.1\%$ | $11.5 \pm 5.3\%$ | 0 / 3 | 0 / 3 | **0 / 3 (0%)** |
| **$M_4$** | Cond A (`[1,1,1]`) | 42 | $0.3442 \pm 0.0200$ | $8.3 \pm 2.9\%$ | $4.1 \pm 7.1\%$ | $6.1 \pm 8.9\%$ | $3.1 \pm 2.3\%$ | 0 / 3 | 2 / 3 | **0 / 3 (0%)** |
| **$M_4$** | Cond B (`[1,2,2]`) | 42 | $0.3600 \pm 0.0206$ | $16.7 \pm 7.6\%$ | $4.1 \pm 7.1\%$ | $8.2 \pm 8.2\%$ | $4.9 \pm 4.0\%$ | 0 / 3 | 1 / 3 | **0 / 3 (0%)** |

### 3. Analysis of Individual Feature Group Contributions

1. **Peak Telemetry & Memory Headroom ($M_1$):**
   - Incorporating `max_cpu_rate`, `cpu_burstiness`, `max_memory`, `memory_headroom`, and `total_page_cache` yielded the highest Macro-F1 across all ablations ($0.3648 \pm 0.0334$ in Condition B).
   - *Failure Mechanism:* While peak usage slightly improves probability separation, at the peak F1 threshold ($\theta \approx 0.535$), Genuine Critical Recall is only **$4.1\%$** ($2/49$). To achieve $\ge 50\%$ Critical Recall, the threshold must be lowered to $\theta \le 0.200$, where Entity FAR explodes to $>85\%$.

2. **Static Borg Scheduler Metadata ($M_2$):**
   - Adding `priority`, `scheduling_class`, `req_cpu`, and `req_ram` did not improve predictive performance (Macro-F1 $0.3211$–$0.3343$, below $M_0$ and $M_1$).
   - *Failure Mechanism:* Task-level priority and declared quotas do not change dynamically prior to a crash. Because both healthy and failing jobs span diverse priority tiers, static metadata fails to provide a fine-grained temporal distress signal.

3. **Dynamic Preemption History ($M_3$):**
   - Adding `eviction_count_prior` and `time_since_last_schedule` increased Episode Recall to **$31.7 \pm 14.4\%$** at peak F1.
   - *Failure Mechanism:* Prior evictions correlate with cluster resource contention, not necessarily execution failure. Consequently, pure-normal tasks running on overloaded machines also experience evictions, driving Entity FAR up to **$25.2\%$** and Window FPR to **$18.2\%$** (violating Gate 2).

4. **Combined Eligible Feature Set ($M_4$):**
   - Combining all 11 additional features ($D=42$) produced competitive Macro-F1 ($0.3600 \pm 0.0206$) and controlled false alarms (FAR $8.2\%$, FPR $4.9\%$), but Genuine Critical Recall remained pinned at **$4.1\%$**.

### 4. Full Threshold Sweep Evaluation (All 6,030 Points)

Evaluating all 201 thresholds across all 30 trained models (6,030 evaluations in [`results/phase7k/threshold_sweep_metrics.csv`](file:///c:/Users/hp/Desktop/Research%20project/results/phase7k/threshold_sweep_metrics.csv)) reveals:
- **Gate 1 Passing Operating Points:** 295 points satisfy Gate 1 (Episode Recall $\ge 70\%$, Critical Recall $\ge 50\%$).
- **Gate 2 Passing Operating Points:** 2,867 points satisfy Gate 2 (Entity FAR $\le 10\%$, Window FPR $\le 5\%$).
- **Simultaneous Passing Operating Points:** **EXACTLY 0 (ZERO)**.

The detection gate and false-alarm gate remain completely disjoint in threshold space across every model architecture and feature representation tested.

---

## Step 5 — Uncertainty, Limitations, and Methodological Conclusions

### 1. Uncertainty and Sample-Size Limitations
- **Validation Cohort Constraints:** Candidate A Validation contains exactly 20 pre-failure episodes across 17 failing entities, and 49 pure-normal entities within 4 pure-normal parent jobs.
- **Statistical Resolution:** Each detected episode represents exactly a 5.0% increment in Episode Recall. Each false alarm on a pure-normal entity represents a 2.04% increment in Entity FAR. Each false alarm on a pure-normal job represents a 25.0% increment in Job FAR.
- **Cross-Seed Variance:** Across the 3 random seeds, standard deviations in peak Macro-F1 were low ($\pm 0.010$ to $\pm 0.033$), confirming that the observed operational failure is robust and not an artifact of random weight initialization.

### 2. Methodological Conclusion
The hypothesis that the operational failure observed in Phases 7B–7J was caused solely by omitting task-level scheduler metadata, peak telemetry, memory headroom, or preemption events is **EMPIRICALLY REFUTED**.

While peak usage telemetry ($M_1$) provides a modest boost in ranking quality, the underlying LSTM sequence model consuming 25 minutes of context ($L=5$) cannot reliably distinguish pre-failure resource anomalies from routine operational transients in cloud cluster traces.

### 3. Preserved Operational Verdict
Under the governing protocol and predeclared decision criteria:
- **`NO OPERATIONAL WINNER` IS RETAINED AND CONFIRMED.**
- **No confirmatory test evaluation is authorized.**
- Candidate A Test ($N=1,025$) and Phase 6 holdout remain strictly quarantined.

---

## Phase 7K Complete Artifact Inventory

| Artifact Path | Format | Description |
| :--- | :---: | :--- |
| [`results/phase7k/temporal_causality_audit.json`](file:///c:/Users/hp/Desktop/Research%20project/results/phase7k/temporal_causality_audit.json) | JSON | Step 1 causal audit verifying timestamps, units, and zero terminal leakage |
| [`results/phase7k/feature_availability_audit.csv`](file:///c:/Users/hp/Desktop/Research%20project/results/phase7k/feature_availability_audit.csv) | CSV | Complete audit of 31 candidate raw and derived fields with eligibility verdicts |
| [`results/phase7k/phase7k_ablation_protocol.json`](file:///c:/Users/hp/Desktop/Research%20project/results/phase7k/phase7k_ablation_protocol.json) | JSON | Machine-readable predeclared ablation protocol with frozen hyperparameters and gates |
| [`results/phase7k/fitted_preprocessors_manifest.json`](file:///c:/Users/hp/Desktop/Research%20project/results/phase7k/fitted_preprocessors_manifest.json) | JSON | Train-only fitted scaling means, stds, and medians for all 5 feature sets |
| [`results/phase7k/training_logs.csv`](file:///c:/Users/hp/Desktop/Research%20project/results/phase7k/training_logs.csv) | CSV | Per-epoch training and validation loss logs across all 30 model runs (750 epochs) |
| [`results/phase7k/validation_predictions.csv`](file:///c:/Users/hp/Desktop/Research%20project/results/phase7k/validation_predictions.csv) | CSV | Validation softmax probabilities ($P_0, P_1, P_2, P_{fail}$) across all 30 runs ($N=657$) |
| [`results/phase7k/threshold_sweep_metrics.csv`](file:///c:/Users/hp/Desktop/Research%20project/results/phase7k/threshold_sweep_metrics.csv) | CSV | Full 201-point threshold sweep metrics across all 30 runs (6,030 operating points) |
| [`results/phase7k/ablation_gate_summary.csv`](file:///c:/Users/hp/Desktop/Research%20project/results/phase7k/ablation_gate_summary.csv) | CSV | Per-run operational gate summary and peak Macro-F1 operating points |
| [`results/phase7k/ablation_cross_seed_aggregate.csv`](file:///c:/Users/hp/Desktop/Research%20project/results/phase7k/ablation_cross_seed_aggregate.csv) | CSV | Cross-seed aggregate mean $\pm$ standard deviation for all 5 models and 2 conditions |
| [`results/phase7k/phase7k_report.md`](file:///c:/Users/hp/Desktop/Research%20project/results/phase7k/phase7k_report.md) | Markdown | Authoritative comprehensive Phase 7K investigation and ablation report |
| [`reports/phase7k/phase7k_report.md`](file:///c:/Users/hp/Desktop/Research%20project/reports/phase7k/phase7k_report.md) | Markdown | Mirrored authoritative report |
| [`results/phase7k/phase7k_manifest.json`](file:///c:/Users/hp/Desktop/Research%20project/results/phase7k/phase7k_manifest.json) | JSON | Cryptographic SHA-256 manifest sealing all Phase 7K artifacts |
