"""Build results/project_discovery/contradiction_register.csv."""

import csv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
out_p = PROJECT_ROOT / "results" / "project_discovery" / "contradiction_register.csv"

contradictions = [
    {
        "issue_id": "CONTRA_01",
        "topic": "Post-Failure Telemetry Censoring",
        "conflicting_sources": "Phase 2-5 reports ('zero post-failure telemetry evaluated; all post-failure steps excluded') vs Phase 6.8 audit",
        "empirical_ground_truth": "TargetLabelGenerator (src/labeling/target_generator.py:73-77) omitted past-failure checks. 149 post-failure windows were included in confirmatory data; 1,130 in train, 197 in val, 145 in test.",
        "resolution_status": "RESOLVED_DEFECT_CONFIRMED",
        "scientific_impact": "Earlier assertions passed because they checked only non-null time_to_failure. Post-failure windows had label 0 and time_to_failure=NaN, silently bypassing checks."
    },
    {
        "issue_id": "CONTRA_02",
        "topic": "Timing Partition of 296 Failure-Cohort Normal Windows",
        "conflicting_sources": "Phase 6.6 report (140 pre-failure / 156 post-failure) vs Phase 6.7/6.8 audit (147 pre-failure / 149 post-failure)",
        "empirical_ground_truth": "Independent lifecycle audit proves exactly 147 pre-failure normal windows and 149 post-failure windows. Discrepancy of 7 windows traced to repeated failures in 6224307516_2 and 6224307516_3.",
        "resolution_status": "RESOLVED_CORRECTED",
        "scientific_impact": "Phase 6.6 reconstructed failure times solely from positive-label windows, missing failure 2 which had no pre-failure telemetry. Phase 6.7/6.8 corrected the counts from raw trace events."
    },
    {
        "issue_id": "CONTRA_03",
        "topic": "Primary Window Count: 5,727 vs 5,511",
        "conflicting_sources": "Original Phase 6.2 protocol summary table (5,727 windows) vs Evaluation Manifest & Amendment 2.1 (5,511 windows)",
        "empirical_ground_truth": "5,727 included 61 non-pre-failing census tasks that generated 216 normal windows and 0 pre-failure windows. Manifest encoded 5,511 windows from 137 pre-failing tasks + 198 pure-normal controls.",
        "resolution_status": "RESOLVED_RATIFIED",
        "scientific_impact": "Resolved in Phase 6.2.3 via Protocol Amendment 2.1, formally locking 5,511 windows as the sole primary confirmatory cohort and relegating the 61 tasks to an excluded descriptive census."
    },
    {
        "issue_id": "CONTRA_04",
        "topic": "Cohort Entity Count: 337 vs 335",
        "conflicting_sources": "Protocol sampling plan (337 entities) vs Evaluable Primary Manifest (335 entities)",
        "empirical_ground_truth": "337 entities were sampled (137 failure entities + 200 normal controls). Exactly 2 sampled normal controls had < 5 steps of total telemetry, leaving 198 evaluable pure-normal entities (137 + 198 = 335).",
        "resolution_status": "RESOLVED_DOCUMENTED",
        "scientific_impact": "Ratified in Amendment 2.1 Tenet 1. The 2 short-history controls were legitimately excluded under the require_full_history=True invariant."
    },
    {
        "issue_id": "CONTRA_05",
        "topic": "Entities (137) vs Episodes (141) vs Raw FAIL Events (153)",
        "conflicting_sources": "Varying counts in historical documents for failure occurrences in holdout",
        "empirical_ground_truth": "137 failure entities experienced 153 raw FAIL events (event_type=3). 8 entities experienced repeated failures. 141 of these events had qualifying pre-failure observation windows, forming the 141 evaluated episodes.",
        "resolution_status": "RESOLVED_VERIFIED",
        "scientific_impact": "All 141 episodes match raw trace timestamps down to 0.1s. Discrepancy between 153 events and 141 episodes represents 12 failure retries that occurred too quickly to accumulate evaluable pre-failure history."
    },
    {
        "issue_id": "CONTRA_06",
        "topic": "Sequence Length Feasibility: L=15 vs L=5",
        "conflicting_sources": "Phase 1-4 design (L=15 observations = 75 min) vs Phase 5 confirmatory failure vs Phase 6 (L=5 = 25 min)",
        "empirical_ground_truth": "In Borg cluster traces, batch tasks that fail do so within 10-45 minutes. Across 100 Phase 5 holdout tasks, zero tasks accumulated 15 pre-failure steps prior to failure. L=15 resulted in 0 pre-failure windows.",
        "resolution_status": "RESOLVED_PIVOT_DOCUMENTED",
        "scientific_impact": "Phase 6 formally demonstrated dataset infeasibility for L=15 and pre-registered L=5. Shorter history allowed 137 entities to yield 700 pre-failure windows (400 Critical, 300 Warning)."
    },
    {
        "issue_id": "CONTRA_07",
        "topic": "Training Loss Weights for Neural Models",
        "conflicting_sources": "Phase 6.6 report (claimed neural models used ~10x loss weights on Critical) vs Actual Training Code",
        "empirical_ground_truth": "scripts/train_phase6_3_l5_models.py (lines 321 & 415) trained LSTM and Transformer with explicit weights [1.0, 1.5, 2.0]. Only RF and Hybrid used balanced class weights (~11.5x on Critical).",
        "resolution_status": "RESOLVED_CORRECTED",
        "scientific_impact": "Disproved Phase 6.6 claim in Phase 6.7 audit. High false alarm rates and saturation in neural sequence models were driven by feature distribution sensitivity and cross-entropy dynamics, not extreme loss weighting."
    },
    {
        "issue_id": "CONTRA_08",
        "topic": "Probability Calibration Claims",
        "conflicting_sources": "Occasional report descriptions of model outputs as 'posterior failure probabilities' vs Actual Model Implementations",
        "empirical_ground_truth": "All models output raw uncalibrated scores (RF tree frequency votes, PyTorch softmax logits). No temperature scaling, Platt scaling, or isotonic regression was ever fitted or evaluated.",
        "resolution_status": "RESOLVED_DISCLAIMED",
        "scientific_impact": "Probabilities cannot be treated as calibrated probabilities of failure. Softmax saturation in Transformer (P > 0.99) reflects uncalibrated confidence on out-of-distribution feature combinations."
    },
    {
        "issue_id": "CONTRA_09",
        "topic": "Validation Generalization vs Confirmatory Collapse",
        "conflicting_sources": "Phase 3/4 validation results (promising F1 and detection) vs Phase 6.4 confirmatory evaluation (complete operational failure)",
        "empirical_ground_truth": "In validation splits, 75%-80% of parent jobs overlapped with the training set. In confirmatory holdout, 95 of 116 parent jobs (81.9%) were completely unseen. Models overfit to parent-job resource profiles.",
        "resolution_status": "RESOLVED_ROOT_CAUSE_IDENTIFIED",
        "scientific_impact": "Proves that apparent success in development was an artifact of parent-job leakage. When tested on truly unseen parent jobs in confirmatory holdout, generalization collapsed."
    },
    {
        "issue_id": "CONTRA_10",
        "topic": "Normative Censoring Rule vs Ratified Evaluation Manifest",
        "conflicting_sources": "Protocol text line 201 ('post-failure step -> label -1 (strictly excluded)') vs Manifest Tenet 3 (locked 296 failure-cohort normal windows)",
        "empirical_ground_truth": "Tenet 7 assumed all 296 windows were 'early pre-failure telemetry dynamics'. In reality, 149 were post-failure recovery windows. The ratified manifest contradicted the normative textual censoring rule.",
        "resolution_status": "RESOLVED_DOCUMENTED",
        "scientific_impact": "Phase 6.8 formally declared Verdict C (Material Protocol Violation). Counterfactual evaluation proved NO_OPERATIONAL_WINNER remains invariant regardless of whether these 149 windows are included or censored."
    }
]

with open(out_p, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(contradictions[0].keys()))
    writer.writeheader()
    writer.writerows(contradictions)

print(f"Contradiction register written to {out_p} ({len(contradictions)} items)")
