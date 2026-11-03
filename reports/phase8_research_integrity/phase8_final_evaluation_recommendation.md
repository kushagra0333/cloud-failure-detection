# Phase 8: Final Evaluation Dataset & Reporting Recommendation

**Project**: Machine Learning Framework for Cloud Workload Failure Prediction  
**Date**: `2026-11-03T14:09:42.024431+00:00`  
**Audit Finding**: `NO_VALID_INDEPENDENT_TEST_SET_IDENTIFIED`  
**Governing Context**: Prioritizing Research Paper Completion on Borg 2011 Data  

---

## 1. Core Question & Definitive Recommendation

### Question: Which dataset should be used for final model evaluation?

### Evidence-Based Recommendation:
**No valid, untouched independent test partition exists within the Google Borg 2011 dataset. We strongly recommend reporting the existing experimental series as comprehensive development and validation results, with the test set history and trace exhaustion transparently disclosed as an experimental limitation.**

Under no circumstances should the research team attempt to fabricate an artificial test set by:
1. Re-splitting previously inspected Candidate A data.
2. Unsealing the spent Phase 6 Confirmatory holdout.
3. Claiming Candidate A Test is an untouched holdout (it was evaluated in Phase 7D).
4. Sampling from the raw trace (which is completely exhausted of unused failing tasks).

---

## 2. Scientific Justification Across Evaluated Options

### Option 1 — Existing Untouched Partition (`NOT_AVAILABLE`)
- **Audit Finding**: All existing partitions in `data/processed/` have been previously evaluated, exposed to iterative model comparisons, contaminated by pre-Phase 7A labeling bugs, or are quarantined holdouts.
- **Why It Fails**: Presenting an already-evaluated dataset as an independent holdout introduces severe publication bias and violates research ethics.

### Option 2 — New Chronological Entity-Level Partition from Raw Trace (`NOT_FEASIBLE`)
- **Audit Finding**: Forensic census of the raw Borg 2011 trace shards in `data/raw/` proved that all tasks with $\ge 5$ pre-failure observations were previously allocated to Candidate A or the Phase 6 holdout.
- **Why It Fails**: Any newly sampled partition from `data/raw/` would contain **zero failure episodes**. Evaluating models on a test set containing only pure-normal tasks would make it mathematically impossible to measure Episode Recall, Critical Recall, or failure detection.

### Option 3 — Transparent Disclosure of Limitation (`RECOMMENDED & SELECTED`)
- **Why This is the Only Scientifically Defensible Choice**:
  1. **Honesty & Reproducibility**: High-impact peer-reviewed systems and ML venues (e.g. IEEE Transactions, ACM TOCS, OSDI/NSDI, NeurIPS/ICLR applied tracks) value transparent methodology over manufactured holdout claims.
  2. **Wealth of Existing Empirical Evidence**: The project has already completed 14 rigorous, controlled experimental phases (Phases 7D through 7U), covering baseline comparisons, class weighting, focal loss, hierarchical classification, feature ablations, context length variations, multiresolution modeling, calibration, relative telemetry, linear ensembling, and fuzzy inference.
  3. **Genuine Scientific Contribution**: The paper's core contribution is the rigorous discovery that **standard supervised deep sequence models suffer a fundamental operational conflict in cloud telemetry**: high-resource batch jobs mimic failure spikes under absolute telemetry, while relative telemetry suppresses false alarms but struggles with abrupt failures.

---

## 3. Recommended Paper Structure & Framing

Rather than claiming a triumphant "operational winner" that does not exist, the paper should be structured around these verified empirical findings:

1. **Title / Theme**: Realistic Early Failure Detection in Cloud Clusters: An Empirical Evaluation of Deep Sequence Models, Operational Constraints, and Telemetry Representations on Google Borg Traces.
2. **Empirical Benchmark**:
   - Compare Random Forest ($M^*$, Macro-F1 $0.3653$), Temporal LSTM ($0.3068$), Compact Transformer ($0.3463$), Representation Hybrid ($0.2880$), Relative Telemetry $S_2$ ($0.3685$), Linear Hybrid ($0.3702$), and Fuzzy Inference ($0.3650$).
   - Show why unweighted sequence models suffer mode collapse, and why class weighting restores recall at the expense of high Entity False Alarm Rates ($>20\% - 48\%$).
3. **Core Architectural Insights**:
   - **Relative Telemetry ($S_2$)**: Show how task-relative normalization reduces Window FPR from $30.18\%$ to $10.30\%$ and Entity FAR from $36.73\%$ to $30.61\%$.
   - **Hybrid Blending**: Demonstrate that blending relative ($S_2$) and absolute ($S_0$) risk signals recovers $+0.033$ within-job AUC.
4. **Transparent Methodological Disclosures**:
   - Explicitly document the Phase 7A labeling repair (showing how post-failure recovery windows distort benchmarks if unhandled).
   - Document the exhaustion of early-failing workloads in Borg 2011.
   - Include the frozen future evaluation protocol (Section 11 of the audit report) as an open benchmark for future research on larger traces (e.g. Borg 2019).

---

## 4. Mandatory Safety Statement

> **No Google Cloud query was executed. No Borg 2019 data was extracted. No Phase 6 Confirmatory holdout was accessed. No Candidate A Test data was newly accessed during this audit. No model training or evaluation was performed. No existing research results were overwritten.**
