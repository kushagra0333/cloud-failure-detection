# Prioritized Research & Engineering Next-Steps Plan

**Project:** Temporal LSTM/Transformer Framework for Early Failure Prediction in Cloud Computing  
**Auditor:** Independent Machine Learning Scientific Auditor  
**Date:** October 27, 2026  
**Status:** Read-Only Strategic Roadmap (Not Yet Executed)

---

## 1. Overview of Roadmap Principles

The Phase 6.8 Confirmatory Cohort Validity Audit established that:
1. The confirmatory evaluation cohort evaluated in Phase 6.4 contained 149 post-failure recovery windows due to an omission in [`src/labeling/target_generator.py`](file:///c:/Users/hp/Desktop/Research%20project/src/labeling/target_generator.py#L73-L77).
2. The same labeling defect affected the development sets: 1,130 of 4,188 Normal training windows (27.0%), 197 of 840 Normal validation windows (23.5%), and 145 of 864 Normal test windows (16.8%) were post-failure observations.
3. In development splits, 75%–80% of parent jobs overlapped between training and validation, enabling models to memorize workload-specific resource profiles rather than learning general failure dynamics.
4. Despite these data anomalies, diagnostic counterfactual evaluations proved that the operational failure of the models is **100% robust and invariant**: across all models and all censoring interpretations, Window FPR ($17.4\% - 34.6\%$) and Entity FAR ($31.8\% - 64.1\%$) massively exceeded pre-registered operational ceilings ($\le 5.0\%$ and $\le 12.0\%$).
5. The Phase 6.4 confirmatory holdout is **unblinded, evaluated, and permanently spent**. It cannot be rescored or reused for new confirmatory claims.

The following roadmap outlines the strictly prioritized, evidence-based steps needed to transition the project from its current audited state into a publication-ready scientific study.

---

## 2. Phase A: Essential Methodological & Pipeline Repairs (Priority 1)

These repairs resolve confirmed defects in data generation and testing before any new experiments are contemplated.

### Step 1: Patch `TargetLabelGenerator` to Enforce Post-Failure Censoring
- **Objective:** Fix the root-cause labeling bug in `src/labeling/target_generator.py` so that any telemetry following a task failure event is explicitly assigned `FailureLabel.UNKNOWN` (-1).
- **Why Needed:** Currently, `derive_label_for_task` evaluates only future failure events (`task_events["time_sec"] > t`). When a task has failed in the past but has $\ge 30$ minutes of subsequent telemetry, it silently returns `FailureLabel.NORMAL` (0).
- **Supporting Evidence:** Discovered in Phase 6.8 audit; verified across confirmatory data (149 windows) and training data (1,130 windows).
- **Target Files:** [`src/labeling/target_generator.py`](file:///c:/Users/hp/Desktop/Research%20project/src/labeling/target_generator.py) lines 55–80.
- **Proposed Logic:**
  ```python
  # Check if task already experienced an abnormal failure in the past
  past_fails = task_events[
      (task_events["event_type"] == self.event_cls.FAIL) &
      (task_events["time_sec"] <= t)
  ]
  if not past_fails.empty:
      return FailureLabel.UNKNOWN, None
  ```
- **Risks & Tradeoffs:** Low technical risk. Will correctly eliminate post-failure contamination from all future generated datasets.
- **Prerequisite / Dependencies:** None. Can be implemented immediately.

### Step 2: Implement Comprehensive Unit & Regression Tests for Temporal Horizons
- **Objective:** Prevent silent regressions in labeling, repeated failures, history crossings, and lookahead boundaries.
- **Why Needed:** Earlier test suites passed because assertions checked only `time_to_failure.dropna() > 0`, which completely missed the post-failure normal windows because their `time_to_failure` was `NaN`.
- **Supporting Evidence:** Phase 6.8 finding that tests checked non-null fields rather than checking absence of prior failure events.
- **Target Files:** [`tests/test_target_generation.py`](file:///c:/Users/hp/Desktop/Research%20project/tests/test_target_generation.py), [`tests/test_window_generation.py`](file:///c:/Users/hp/Desktop/Research%20project/tests/test_window_generation.py).
- **Test Scenarios to Add:**
  1. Task with single failure: verify that all steps after failure are labeled `-1` (`UNKNOWN`).
  2. Task with repeated failures: verify that steps between failure 1 and failure 2 are correctly labeled (either pre-failure for episode 2, or censored if history spans across failure 1).
  3. Lookahead boundary: verify that steps with $< 1800\text{ s}$ telemetry before trace end without `FINISH` are labeled `-1`.
  4. Contiguity: verify that sequence histories spanning across a prior failure timestamp are flagged or rejected.
- **Risks & Tradeoffs:** Zero risk; pure verification.

### Step 3: Enforce Parent-Job Disjoint Splitting in `LeakageSafeSplitter`
- **Objective:** Add a `job_grouped` splitting strategy to [`src/splitting/leakage_safe_split.py`](file:///c:/Users/hp/Desktop/Research%20project/src/splitting/leakage_safe_split.py) ensuring that all tasks belonging to the same `job_id` are placed in the same split.
- **Why Needed:** In the current development split, 75%–80% of parent jobs overlap between training and validation/test. This allowed models to memorize parent-job-specific resource signatures, inflating validation scores while failing on truly unseen jobs in holdout.
- **Supporting Evidence:** Independent check revealed 24 of 32 validation jobs and 25 of 31 test jobs appear in the training set.
- **Target Files:** [`src/splitting/leakage_safe_split.py`](file:///c:/Users/hp/Desktop/Research%20project/src/splitting/leakage_safe_split.py).
- **Risks & Tradeoffs:** Low risk. Reduces training set size slightly if large jobs are clustered, but guarantees true out-of-distribution application generalization.

---

## 4. Phase B: Corrected Development Experimentation (Priority 2)

These steps re-run development experiments on a clean, corrected development partition without touching the retired confirmatory holdout.

### Step 4: Regenerate Clean Development Datasets (`data/processed/clean_dev/`)
- **Objective:** Generate a clean development partition (train/val/test) using the patched `TargetLabelGenerator` and `job_grouped` splitting, saving to a new directory without overwriting existing files.
- **Why Needed:** Models must be trained on data where "NORMAL" means genuine healthy execution, not post-crash container cleanup.
- **Target Files:** New script `scripts/build_clean_development_datasets.py`, outputs in `data/processed/clean_dev/`.
- **Risks & Tradeoffs:** Requires ~5 minutes of trace processing. Preserves historical datasets untouched.

### Step 5: Replace Ad-Hoc Argmax with Cost-Sensitive Decision Thresholding
- **Objective:** Implement a cost-sensitive utility optimization framework on validation data to determine decision thresholds, rather than relying on uniform argmax.
- **Why Needed:** In 3-class classification, uniform argmax triggers Critical alerts at $P(\text{Critical}) \approx 0.336 - 0.359$ whenever non-critical probability is split. Under extreme class imbalance (98% Normal, 1% Warning, 1% Critical), argmax is mathematically guaranteed to generate massive false alarm rates.
- **Supporting Evidence:** Phase 6.6 and 6.7 mathematical proof of argmax triggering thresholds.
- **Target Files:** New module `src/inference/threshold_optimizer.py`.
- **Risks & Tradeoffs:** Thresholds must be tuned strictly on validation data to avoid optimistic bias.

### Step 6: Evaluate Temperature Scaling & Probability Calibration
- **Objective:** Fit temperature scaling (`T`) on validation logits to calibrate softmax probabilities before applying thresholding.
- **Why Needed:** Compact Transformer outputs saturated probabilities ($P(\text{Critical}) > 0.99$ on 53.7% of false positives). Without calibration, predicted probabilities cannot be interpreted as posterior failure likelihoods.
- **Supporting Evidence:** Phase 6.6 Section 6.2 and Phase 6.7 Claim 7.
- **Target Files:** `src/models/calibration.py`.
- **Risks & Tradeoffs:** Temperature scaling preserves argmax ordering but improves probability-threshold separation.

### Step 7: Decouple Telemetry Missingness from Utilization Signals
- **Objective:** Isolate missingness indicators (e.g., `cpi_missing_fraction_6`) from continuous utilization features, testing whether deep models overfit to collector dropout.
- **Why Needed:** Phase 6.6 and 6.7 proved that missing CPI had an extreme effect size ($d = +1.2$ to $+3.0$) on false alarms because the model learned that missing hardware counters indicate failure.
- **Supporting Evidence:** Observational confounding in Google trace collector sampling.
- **Target Files:** `src/features/engineer.py`.
- **Risks & Tradeoffs:** Low risk. Verifies whether models can predict failure from resource dynamics rather than telemetry artifacts.

---

## 5. Phase C: Genuinely New Confirmatory Evaluation (Priority 3)

These steps govern future confirmatory hypothesis testing if the user decides to run a new benchmark.

### Step 8: Sample a Fresh, Untouched Holdout Cohort (Phase 7 Protocol)
- **Objective:** Sample a brand-new confirmatory cohort from unobserved Google trace shards or an independent datacenter time window (e.g., Days 20–29 of the Borg trace).
- **Why Needed:** The Phase 6 confirmatory cohort has been scored, audited in Phase 6.5, investigated in Phase 6.6, and re-audited in Phase 6.7 and 6.8. Re-evaluating models on the same holdout is scientifically invalid.
- **Prerequisites:** All Phase A and Phase B steps must be complete and locked.
- **Risks & Tradeoffs:** Requires pre-registering a formal Protocol 3.0 before extracting data.

---

## 6. Phase D: Academic Paper Revisions & Contributions (Immediate)

These steps summarize how to write the research paper based on the evidence already discovered.

### Step 9: Reframe the Research Paper as a Rigorous Empirical Evaluation & Forensic Audit
- **Core Narrative:** Present the research as an honest, high-integrity scientific investigation that evaluated deep temporal architectures for cloud failure prediction, uncovered critical operational failure modes (false alarm cascading, argmax threshold vulnerability, workload clustering), and performed a forensic audit identifying data-labeling and censoring pitfalls.
- **Why This Succeeds Academically:** Negative results in machine learning—when accompanied by rigorous root-cause forensic analysis and counterfactual invariance proofs—are highly valued in top venues (e.g., IEEE Transactions on Cloud Computing, ACM SoCC, Empirical Software Engineering). They prevent other researchers from wasting compute on flawed formulations.
- **Key Claims to Make:**
  1. Deep temporal models (LSTM, Transformer) do not automatically outperform Random Forest on cloud failure prediction under realistic multi-tenant trace conditions.
  2. Uniform argmax decision rules fail catastrophically under extreme class imbalance.
  3. Workload clustering by parent job creates severe observational confounding.
  4. Post-failure telemetry censoring requires strict past-event verification in lifecycle data pipelines.

---

## 7. Summary Dependency Matrix

```
[Step 1: TargetLabelGenerator Patch] ────► [Step 2: Regression Tests] ────► [Step 3: Job-Grouped Splitter]
                                                                                      │
                                                                                      ▼
[Step 6: Calibration] ◄── [Step 5: Cost-Sensitive Thresholds] ◄── [Step 4: Clean Dev Dataset]
          │
          ▼
[Step 7: Decouple Missingness] ────► [Step 8: Fresh Confirmatory Cohort (Phase 7)]
          │
          ▼
[Step 9: Academic Paper Manuscript Revisions (Immediate)]
```
