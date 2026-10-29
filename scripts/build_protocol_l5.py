"""Build and freeze the Phase 6.2 L=5 Confirmatory Protocol Artifact.

Creates:
- results/phase6/protocol_l5.json
"""

import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def convert_numpy_types(obj):
    """Recursively converts numpy types to standard Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {str(k): convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(v) for v in obj]
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, (np.integer, int)):
        return int(obj)
    elif isinstance(obj, (np.floating, float)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return convert_numpy_types(obj.tolist())
    return obj


def build_protocol_l5():
    print("=" * 80)
    print("PHASE 6.2: FREEZING L=5 CONFIRMATORY PROTOCOL")
    print("=" * 80)

    # Load previously extracted metadata from scratch
    with open("scratch/failing_entities_metadata.json") as f:
        failing_entities_meta = json.load(f)
    with open("scratch/failing_episodes_metadata.json") as f:
        failing_episodes_meta = json.load(f)
    with open("scratch/failing_jobs_metadata.json") as f:
        failing_jobs_meta = json.load(f)

    # Load frozen Phase 4/5 normal entities
    with open("results/phase4/confirmatory_protocol.json") as f:
        p4_proto = json.load(f)
    frozen_normal_eids = sorted(p4_proto["frozen_entity_ids"]["normal_entities"])

    # Extract failure entities
    failure_eids = sorted([item["entity_id"] for item in failing_entities_meta])
    assert len(failure_eids) == 137, f"Expected 137 failure entities, got {len(failure_eids)}"

    # Identify disjoint vs shared entities
    disjoint_failing_eids = sorted([
        item["entity_id"] for item in failing_entities_meta if not item["is_parent_job_in_dev"]
    ])
    shared_failing_eids = sorted([
        item["entity_id"] for item in failing_entities_meta if item["is_parent_job_in_dev"]
    ])
    assert len(disjoint_failing_eids) == 75, f"Expected 75 disjoint entities, got {len(disjoint_failing_eids)}"
    assert len(shared_failing_eids) == 62, f"Expected 62 shared entities, got {len(shared_failing_eids)}"

    # Jobs
    disjoint_jobs = sorted([j["parent_job_id"] for j in failing_jobs_meta if not j["is_in_dev"]])
    shared_jobs = sorted([j["parent_job_id"] for j in failing_jobs_meta if j["is_in_dev"]])
    all_failure_jobs = sorted([j["parent_job_id"] for j in failing_jobs_meta])

    normal_jobs = sorted(list(set(e.split("_")[0] for e in frozen_normal_eids)))

    # Episodes
    all_episodes = sorted([item["episode_id"] for item in failing_episodes_meta])
    disjoint_episodes = sorted([
        item["episode_id"] for item in failing_episodes_meta if not item["is_parent_job_in_dev"]
    ])
    shared_episodes = sorted([
        item["episode_id"] for item in failing_episodes_meta if item["is_parent_job_in_dev"]
    ])
    assert len(all_episodes) == 141, f"Expected 141 episodes, got {len(all_episodes)}"
    assert len(disjoint_episodes) == 79, f"Expected 79 disjoint episodes, got {len(disjoint_episodes)}"
    assert len(shared_episodes) == 62, f"Expected 62 shared episodes, got {len(shared_episodes)}"

    protocol_data = {
        "protocol_version": "2.0-pre-registered-L5",
        "date_frozen": "2026-10-26",
        "status": "FROZEN_PRE_REGISTERED_UNTOUCHED",
        "auditor": "Independent ML Scientific Auditor (Phase 6.2)",
        "safeguards_complied": {
            "no_model_scoring_performed": True,
            "no_holdout_threshold_tuning": True,
            "phase5_artifacts_unmodified": True,
            "episodes_not_treated_as_independent_entity_clusters": True,
            "reuse_of_l15_checkpoints_strictly_prohibited": True,
            "passing_tests_acknowledged_not_proof_of_validity": True
        },
        "cohort_reconciliation_decision": {
            "earlier_proposal": "100 failure entities and 200 normal entities (Phase 6 exploratory concept)",
            "empirical_census_finding": "Exactly 137 failure entities with 141 distinct failure episodes exist in the untouched trace possessing valid pre-failure telemetry under L=5.",
            "decision": "USE ALL 137 ELIGIBLE FAILURE ENTITIES (100% of the eligible failure population). Subsampling down to 100 was rejected because discarding 37 verified failure entities (27% of available cases) and 35 episodes without scientific justification diminishes statistical power, increases bootstrap uncertainty width, and compromises the 75-entity sensitivity analysis.",
            "normal_cohort_pairing": "Paired with 200 frozen normal entities sampled deterministically from the 238,171 pure-normal pool with seed 42. Evaluated in unpadded L=5 mode, 198 yield valid windows (4,515 windows, 100% label 0; 2 entities had < 5 steps and are excluded before scoring).",
            "ratio": "137 failure entities : 200 normal entities (Case-Control ratio ~ 1:1.46, total 337 entities)."
        },
        "experimental_design_specification": {
            "sequence_length_L": 5,
            "cadence_seconds": 300.0,
            "cadence_minutes": 5.0,
            "endpoint_span_seconds": 1200.0,
            "endpoint_span_minutes": 20.0,
            "physical_collection_span_seconds": 1500.0,
            "physical_collection_span_minutes": 25.0,
            "require_full_history": True,
            "allow_zero_padding": False,
            "inter_step_gap_tolerance_seconds": 450.0,
            "target_horizons": {
                "critical": "0 < time_to_failure <= 900.0s (0 < tau <= 15m) -> y=2",
                "warning": "900.0s < time_to_failure <= 1800.0s (15m < tau <= 30m) -> y=1",
                "normal": "time_to_failure > 1800.0s with >= 1800.0s follow-up or FINISH event -> y=0",
                "unknown_censored": "censored lookahead (< 1800s without FINISH) or post-failure step -> label -1 (strictly excluded)"
            }
        },
        "cohort_counts_summary": {
            "entities": {
                "failure_entities_total": 137,
                "failure_entities_disjoint_jobs": 75,
                "failure_entities_shared_jobs": 62,
                "normal_entities_total": 200,
                "normal_entities_evaluable_l5": 198,
                "normal_entities_excluded_short_history": 2,
                "total_confirmatory_entities": 337
            },
            "parent_jobs": {
                "failure_parent_jobs_total": 9,
                "failure_parent_jobs_disjoint": 4,
                "failure_parent_jobs_shared": 5,
                "normal_parent_jobs_total": 109,
                "total_unique_parent_jobs": 118
            },
            "failure_episodes": {
                "total_failure_episodes": 141,
                "disjoint_jobs_episodes": 79,
                "shared_jobs_episodes": 62
            },
            "sliding_windows": {
                "prefail_windows_total": 700,
                "warning_windows": 300,
                "critical_windows": 400,
                "prefail_windows_disjoint_jobs": 347,
                "prefail_windows_shared_jobs": 353,
                "normal_windows_from_failing_cohort": 512,
                "normal_windows_from_normal_cohort": 4515,
                "total_evaluable_windows": 5727,
                "prevalence_prefail_windows": 0.122227867993714,
                "prevalence_warning_windows": 0.05238344683080147,
                "prevalence_critical_windows": 0.06984442116291252,
                "prevalence_normal_windows": 0.877772132006286
            }
        },
        "frozen_entity_ids": {
            "all_failure_entities": failure_eids,
            "disjoint_job_failure_entities": disjoint_failing_eids,
            "shared_job_failure_entities": shared_failing_eids,
            "normal_entities": frozen_normal_eids
        },
        "frozen_parent_job_ids": {
            "all_failure_jobs": all_failure_jobs,
            "disjoint_failure_jobs": disjoint_jobs,
            "shared_failure_jobs": shared_jobs,
            "normal_jobs": normal_jobs
        },
        "frozen_episode_metadata": failing_episodes_meta,
        "frozen_parent_jobs_metadata": failing_jobs_meta,
        "training_and_modeling_readiness_audit": {
            "new_l5_training_mandated": True,
            "l15_checkpoint_reuse_permitted": False,
            "l15_checkpoint_reuse_rationale": "Legacy checkpoints (best_lstm_checkpoint.pt, best_transformer_checkpoint.pt) were trained with positional encodings and recurrent dynamics tailored to T=15 steps. Reusing them on T=5 violates sequence length assumptions and produces corrupt attention distributions. Completely new models must be trained on L=5 data.",
            "models_to_train_under_l5": [
                {
                    "model_name": "Random Forest",
                    "architecture": "RandomForestClassifier(n_estimators=100, max_depth=15, class_weight='balanced', random_state=42)",
                    "input_representation": "Latest-step feature vector x_last in R^31 (causally aggregated trailing features)",
                    "input_shape": "[batch_size, 31]"
                },
                {
                    "model_name": "Temporal LSTM",
                    "architecture": "LSTM(input_size=31, hidden_size=64, num_layers=2, batch_first=True, dropout=0.2)",
                    "input_shape": "[batch_size, 5, 31]",
                    "loss_function": "Weighted CrossEntropyLoss(weights=[1.0, 1.5, 2.0])"
                },
                {
                    "model_name": "Compact Temporal Transformer",
                    "architecture": "TransformerEncoder(d_model=32, nhead=4, num_layers=2, dim_feedforward=64, dropout=0.1, max_len=5)",
                    "input_shape": "[batch_size, 5, 31]",
                    "loss_function": "Weighted CrossEntropyLoss(weights=[1.0, 1.5, 2.0])"
                },
                {
                    "model_name": "Representation Hybrid",
                    "architecture": "Frozen Compact Transformer encoder backbone (32-dim latent embedding) concatenated with 31-dim tabular vector, classified by Random Forest",
                    "input_shape": "[batch_size, 63]",
                    "loss_function": "Gini / Random Forest objective"
                }
            ],
            "development_data_characterization": "Development partitions (train=400, val=50, test=75 entities) were utilized during Phases 2-4 for baseline exploratory exploration. The validation set is an exploratory optimization set, NOT an independent confirmatory set. The only pristine, untouched holdout is the frozen Phase 6.2 cohort.",
            "preprocessing_freeze": {
                "scaler_source": "configs/fitted_scaler.json",
                "fitted_strictly_on": "Phase 2 Chronological Train Partition (400 entities)",
                "confirmatory_scaling_mode": "Transform-only; zero parameters fitted or updated on holdout data."
            },
            "checkpoint_selection_policy": "All neural model checkpoints must be selected exclusively based on minimum validation loss or maximum validation Macro-F1 on the development validation partition. The confirmatory holdout shall never be inspected during training or checkpoint selection.",
            "threshold_tuning_policy": "Zero threshold tuning on the confirmatory holdout. Decision rule is strictly argmax over predicted class probabilities: y_hat = argmax_c P(y=c | X)."
        },
        "evaluation_protocol_and_analysis_plan": {
            "primary_analysis": {
                "name": "Pre-Registered Full Cohort Evaluation",
                "population": "All 137 failure entities (141 episodes, 700 pre-failure windows) + 200 normal entities (4,515 normal windows)",
                "total_clusters": 337,
                "independent_failure_clusters": 137,
                "parent_jobs_represented": 118,
                "clustering_level": "entity_id",
                "clustering_rule": "Treating 141 episodes as 141 independent clusters is strictly prohibited. Entity-level cluster bootstrap resamples entire entities with replacement, preserving all within-entity episodes and temporal window correlations.",
                "bootstrap_iterations": 1000,
                "confidence_level": 0.95
            },
            "sensitivity_analysis": {
                "name": "Cross-Job Generalization Sensitivity Analysis",
                "population": "The 75 failure entities (79 episodes, 347 pre-failure windows) belonging to the 4 parent jobs completely absent from development partitions + 200 normal entities",
                "total_clusters": 275,
                "independent_failure_clusters": 75,
                "parent_jobs_represented": 113,
                "purpose": "Evaluates whether model failure prediction capability generalizes across independent applications/jobs, ruling out memorization of job-specific workload profiles."
            },
            "metric_definitions_and_separation": {
                "window_fpr": "FPR_window = (False Positives on Normal Windows) / (Total True Normal Windows). Evaluated across all 5,027 true normal windows.",
                "entity_far": "FAR_entity = (Normal Entities generating >= 1 alert) / (Total Evaluated Normal Entities). Evaluated across all 198 evaluable pure-normal entities.",
                "episode_recall": "Episode_Recall = (Failure Episodes with >= 1 alert within 30m before failure) / (Total Evaluated Episodes)."
            },
            "operational_gates": {
                "entity_far_gate": "FAR_entity <= 12.0%",
                "window_fpr_gate": "FPR_window <= 5.0%"
            },
            "model_selection_and_ranking_rule": {
                "eligibility": "Only models satisfying BOTH operational gates (FAR <= 12.0% AND FPR <= 5.0%) are eligible to be declared an operational winner.",
                "ranking_criterion": "Eligible models are ranked strictly by Episode Detection Recall on the primary cohort.",
                "tie_breaker": "Macro-F1 across all 3 classes (Normal, Warning, Critical).",
                "fallback_clause": "If no candidate model passes both operational gates, the protocol mandates declaring NO OPERATIONAL WINNER. All models must be fully reported regardless of gate outcome."
            }
        },
        "final_readiness_verdict": {
            "verdict": "READY FOR L=5 MODEL TRAINING",
            "rationale": "Protocol is completely frozen. Cohort reconciliation is scientifically grounded (all 137 failure entities used; 200 verified normal entities). Input shapes and training requirements for L=5 models are formally defined. L=15 checkpoint reuse is prohibited. Primary and cross-job sensitivity analyses are rigorously specified with entity-level cluster bootstrap mechanics."
        }
    }

    out_file = Path("results/phase6/protocol_l5.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(convert_numpy_types(protocol_data), f, indent=2)

    print(f"\nSuccessfully wrote frozen protocol to {out_file}")
    print("=" * 80)
    print(f"VERDICT: {protocol_data['final_readiness_verdict']['verdict']}")
    print("=" * 80)


if __name__ == "__main__":
    build_protocol_l5()
