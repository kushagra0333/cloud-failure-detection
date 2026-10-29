import json
from pathlib import Path

PROJECT_ROOT = Path("c:/Users/hp/Desktop/Research project")
out_dir = PROJECT_ROOT / "results" / "phase7b_1"
out_dir.mkdir(parents=True, exist_ok=True)

# 1. Versioned Revised Evaluation Protocol (v2.0)
protocol_v2 = {
    "protocol_name": "Phase 7B Predeclared Evaluation Protocol",
    "version": "2.0-REVISED-FROZEN",
    "timestamp_iso": "2026-10-27T23:55:00+05:30",
    "predecessor_version": "1.0-FROZEN (results/phase7b/phase7b_evaluation_protocol.json)",
    "revisions_summary": [
        "Eliminated test-set model-selection leakage: operational candidate is selected strictly on Validation data before test evaluation.",
        "Resolved Gate 2 inconsistency: Pure-Normal Parent-Job FAR is explicitly designated as a Secondary Diagnostic Metric due to N=3 pure-normal job denominator in test, while Entity FAR (<=10%) and Window FPR (<=5%) form the primary Gate 2.",
        "Formally specified episode matching algorithm with strict pre-failure 30-minute windowing, single-credit detection, and lead-time definition.",
        "Documented cluster bootstrap behavior on J=11 test clusters, including empty-denominator handling and dual uncertainty reporting.",
        "Verified feature preprocessing pipeline: training-only fitting, median imputation for CPI and lags, zero cross-entity carryover."
    ],
    "frozen_split_reference": {
        "candidate_a_assignment_csv": "results/phase7a_4/candidate_a_job_grouped_split.csv",
        "candidate_a_sha256": "f84bc4fe54f8e0583e2ee52f49b2fbe8e6fcafb4f020c656985a1acc31bb4946",
        "train_windows": 3278,
        "val_windows": 657,
        "test_windows": 1025,
        "total_windows": 4960,
        "zero_parent_job_overlap": True
    },
    "three_stage_evaluation_architecture": {
        "stage_1_training_and_tuning": {
            "training_split": "Candidate A Train (N=3,278 windows, 51 parent jobs)",
            "validation_split": "Candidate A Val (N=657 windows, 12 parent jobs)",
            "rules": [
                "All 4 models trained strictly on Train split.",
                "Epoch selection, checkpointing, and threshold calibration performed strictly on Validation split.",
                "Test split remains completely sealed and untouched during Stage 1."
            ]
        },
        "stage_2_validation_selection_and_gate_evaluation": {
            "validation_gates": {
                "gate_1_failure_recall": "Validation Episode Recall >= 70.0% AND Validation Critical Window Recall >= 50.0%",
                "gate_2_false_alarm_containment": "Validation Pure-Normal Entity FAR <= 10.0% AND Validation Normal-Window FPR <= 5.0%"
            },
            "selection_rule": "The model satisfying both Validation Gate 1 and Validation Gate 2 with the highest Validation Macro-F1 is selected as the 'Validated Operational Candidate' (M*). If zero models pass both validation gates, verdict is NO_VALIDATED_OPERATIONAL_CANDIDATE.",
            "freezing_rule": "Model M* (its weights, checkpoint, scaler, and decision thresholds) is permanently locked before any test set scoring."
        },
        "stage_3_one_shot_confirmatory_test_evaluation": {
            "test_split": "Candidate A Test (N=1,025 windows, 11 parent jobs)",
            "confirmatory_evaluation_of_winner": {
                "subject": "Pre-selected model M* only",
                "test_gates": {
                    "gate_1": "Test Episode Recall >= 70.0% AND Test Critical Window Recall >= 50.0%",
                    "gate_2": "Test Pure-Normal Entity FAR <= 10.0% AND Test Normal-Window FPR <= 5.0%"
                },
                "verdict_rule": "If M* passes both Test Gate 1 and Test Gate 2, M* is declared the OPERATIONAL_WINNER. If M* fails either gate, final verdict is NO_OPERATIONAL_WINNER."
            },
            "descriptive_scientific_benchmarks": {
                "subject": "The remaining three non-selected models",
                "role": "Reported strictly as descriptive benchmark context for research transparency.",
                "restriction": "Under no circumstances can a non-selected model replace M* as the winner based on test performance."
            }
        }
    },
    "metrics_specification": {
        "primary_metric": {
            "name": "Macro-F1",
            "formula": "mean(F1_normal, F1_warning, F1_critical)",
            "scope": "Evaluated on Test split for M* confirmatory performance; on Validation split for model selection"
        },
        "secondary_window_metrics": [
            "Critical Class (y=2) Precision, Recall, F1",
            "Warning Class (y=1) Precision, Recall, F1",
            "Normal Class (y=0) Precision, Recall, F1",
            "Overall Accuracy",
            "Confusion Matrix (3x3)",
            "PR-AUC (Precision-Recall Area Under Curve) per class"
        ],
        "episode_detection_specification": {
            "episode_identifier": "(entity_id, fail_timestamp)",
            "eligible_window_condition": "label in {1, 2} (pre-failure windows within 30 min of failure)",
            "alert_condition": "y_pred in {1, 2}",
            "matching_rule": "An alert window w matches episode (eid, ftime) iff w.entity_id == eid AND 0 < (ftime - w.prediction_timestamp) <= 1800.0s",
            "episode_recall_formula": "Detected Episodes / Total Eligible Episodes",
            "lead_time_formula": "ftime - min(prediction_timestamp of matching alert windows)"
        },
        "false_alarm_metrics": {
            "primary_gate_metrics": [
                {
                    "metric": "Pure-Normal Entity FAR",
                    "denominator": "Total pure-normal entities in split (49 in Val, 38 in Test)",
                    "threshold": "<= 10.0%"
                },
                {
                    "metric": "Normal-Window FPR",
                    "denominator": "Total ground-truth normal windows in split (560 in Val, 921 in Test)",
                    "threshold": "<= 5.0%"
                }
            ],
            "secondary_diagnostic_metric": {
                "metric": "Pure-Normal Parent-Job FAR",
                "denominator": "Total pure-normal parent jobs in split (4 in Val, 3 in Test)",
                "status": "Reported as secondary diagnostic metric with exact integer counts. Excluded from primary binary gate because denominator of 3 or 4 causes discrete single-event quantization (33.3% or 25.0% per failure)."
            }
        }
    },
    "uncertainty_estimation_protocol": {
        "primary_cluster_bootstrap": {
            "resampling_unit": "parent_job_id (J=11 in Test)",
            "iterations": 1000,
            "confidence_level": 0.95,
            "interval_type": "Empirical percentiles [2.5%, 97.5%]",
            "undefined_denominator_handling": "If a bootstrap replicate samples 0 instances of a required denominator (e.g. 0 pure-normal jobs in 3.1% of replicates), that metric is tracked as undefined for that replicate and excluded from the percentile calculation; the fraction of valid replicates is reported."
        },
        "complementary_uncertainty_reporting": [
            "Entity-level bootstrap 95% CIs (N=59 entities)",
            "Analytical standard errors for Macro-F1 and Window FPR",
            "Explicit reporting of small-sample cluster limitation (J=11)"
        ]
    },
    "feature_pipeline_lock": {
        "scaler_fitted_on": "Candidate A Train split strictly",
        "imputation_policy": "Training-split median per feature for NaNs resulting from lag initialization and missing CPI",
        "cross_entity_independence": "Verified GroupBy('entity_id') ensures zero state carryover across entities"
    }
}

v2_path = out_dir / "phase7b_evaluation_protocol_v2.json"
with open(v2_path, "w") as f:
    json.dump(protocol_v2, f, indent=2)

# 2. Protocol Review Document JSON
review_summary = {
    "review_phase": "Phase 7B.1 Evaluation Protocol Review",
    "timestamp_iso": "2026-10-27T23:55:00+05:30",
    "resolved_issues": [
        {
            "issue_id": "TEST_SET_MODEL_SELECTION_LEAKAGE",
            "status": "RESOLVED",
            "original_flaw": "Protocol v1 evaluated all 4 models on Test and picked winner with highest Test Macro-F1 among those passing Test gates, leaking test data for model selection.",
            "resolution": "Implemented 3-stage architecture: Model selection and operational gates occur strictly on Validation split. Exactly one model (M*) is frozen as Validated Operational Candidate before test unblinding. Test evaluation is strictly confirmatory for M*."
        },
        {
            "issue_id": "PARENT_JOB_FAR_GATE_INCONSISTENCY",
            "status": "RESOLVED",
            "original_flaw": "Parent-job FAR (<=15%) was listed in metrics hierarchy but omitted from Gate 2 acceptance definition.",
            "resolution": "Audited denominators: Test has only 3 pure-normal jobs; Val has only 4. A single false alarm causes Job FAR to jump to 33.3% (Test) or 25.0% (Val). Job FAR is formally designated as a Secondary Diagnostic Metric with integer reporting, while Entity FAR (<=10%) and Window FPR (<=5%) form Gate 2."
        },
        {
            "issue_id": "EPISODE_DETECTION_AMBIGUITY",
            "status": "RESOLVED",
            "original_flaw": "Matching algorithm was described in high-level prose without handling repeated alerts, repeated failures, or post-failure exclusion.",
            "resolution": "Formally specified matching algorithm on pre-failure windows (y in {1, 2}), single-credit counting, lead-time formula, and provided 4-scenario hand-checkable walkthrough."
        },
        {
            "issue_id": "CLUSTER_BOOTSTRAP_STABILITY_ON_11_CLUSTERS",
            "status": "RESOLVED",
            "original_flaw": "Bootstrap was proposed without addressing small cluster count (J=11), job size skew (job 6127642779 has 660 of 1025 windows), or undefined denominators.",
            "resolution": "Simulated 1,000 cluster replicates: proved 0% missing classes, 3.1% missing pure-normal jobs. Defined explicit NaN-replicate exclusion and mandated dual reporting (cluster bootstrap CIs + entity-level CIs + point estimates)."
        },
        {
            "issue_id": "FEATURE_PREPROCESSING_SCALER_ALIGNMENT",
            "status": "RESOLVED",
            "original_flaw": "Unclear whether features were normalized using old entity split scaler or new Candidate A training scaler.",
            "resolution": "Verified that features can be scaled cleanly using Candidate A training split statistics; verified GroupBy('entity_id') prevents cross-entity leakage; documented 81.3% CPI missingness and lag initialization."
        }
    ],
    "verdict": "PROTOCOL READY FOR TRAINING APPROVAL"
}

review_json_path = out_dir / "evaluation_protocol_review.json"
with open(review_json_path, "w") as f:
    json.dump(review_summary, f, indent=2)

print(f"Saved revised protocol v2 to: {v2_path}")
print(f"Saved review summary JSON to:  {review_json_path}")
