"""Unit tests for Phase 6.2: Final Protocol Freeze and L=5 Training Readiness Audit."""

import json
from pathlib import Path
import pandas as pd
import pytest


@pytest.fixture(scope="module")
def protocol():
    protocol_path = Path("results/phase6/protocol_l5.json")
    assert protocol_path.exists(), "results/phase6/protocol_l5.json must exist"
    with open(protocol_path) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def dev_entities():
    tr = pd.read_csv("data/processed/train_metadata.csv")
    va = pd.read_csv("data/processed/val_metadata.csv")
    te = pd.read_csv("data/processed/test_metadata.csv")
    return set(tr["entity_id"]) | set(va["entity_id"]) | set(te["entity_id"])


def test_protocol_artifact_exists_and_status(protocol):
    """Verifies that the protocol exists, is properly versioned and frozen, with READY verdict."""
    assert protocol["protocol_version"] == "2.0-pre-registered-L5"
    assert protocol["status"] == "FROZEN_PRE_REGISTERED_UNTOUCHED"
    verdict = protocol["final_readiness_verdict"]
    assert verdict["verdict"] == "READY FOR L=5 MODEL TRAINING"
    assert len(verdict["rationale"]) > 50


def test_safeguards_and_prohibitions(protocol):
    """Verifies that all scientific safeguards and methodological prohibitions are enforced."""
    safeguards = protocol["safeguards_complied"]
    assert safeguards["no_model_scoring_performed"] is True
    assert safeguards["no_holdout_threshold_tuning"] is True
    assert safeguards["phase5_artifacts_unmodified"] is True
    assert safeguards["episodes_not_treated_as_independent_entity_clusters"] is True
    assert safeguards["reuse_of_l15_checkpoints_strictly_prohibited"] is True
    assert safeguards["passing_tests_acknowledged_not_proof_of_validity"] is True


def test_cohort_counts_and_reconciliation(protocol):
    """Verifies exact candidate counts across entity, parent-job, episode, and window levels."""
    counts = protocol["cohort_counts_summary"]
    ent = counts["entities"]
    assert ent["failure_entities_total"] == 137
    assert ent["failure_entities_disjoint_jobs"] == 75
    assert ent["failure_entities_shared_jobs"] == 62
    assert ent["failure_entities_disjoint_jobs"] + ent["failure_entities_shared_jobs"] == 137
    assert ent["normal_entities_total"] == 200
    assert ent["normal_entities_evaluable_l5"] == 198
    assert ent["normal_entities_excluded_short_history"] == 2
    assert ent["total_confirmatory_entities"] == 337

    ep = counts["failure_episodes"]
    assert ep["total_failure_episodes"] == 141
    assert ep["disjoint_jobs_episodes"] == 79
    assert ep["shared_jobs_episodes"] == 62
    assert ep["disjoint_jobs_episodes"] + ep["shared_jobs_episodes"] == 141

    win = counts["sliding_windows"]
    assert win["prefail_windows_total"] == 700
    assert win["warning_windows"] == 300
    assert win["critical_windows"] == 400
    assert win["warning_windows"] + win["critical_windows"] == 700
    assert win["prefail_windows_disjoint_jobs"] == 347
    assert win["prefail_windows_shared_jobs"] == 353
    assert win["prefail_windows_disjoint_jobs"] + win["prefail_windows_shared_jobs"] == 700
    assert win["normal_windows_from_failing_cohort"] == 512
    assert win["normal_windows_from_normal_cohort"] == 4515
    assert win["total_evaluable_windows"] == 5727


def test_frozen_entity_ids_integrity_and_no_dev_leakage(protocol, dev_entities):
    """Verifies complete entity-level isolation between holdout and development data."""
    f_eids = set(protocol["frozen_entity_ids"]["all_failure_entities"])
    n_eids = set(protocol["frozen_entity_ids"]["normal_entities"])

    assert len(f_eids) == 137
    assert len(n_eids) == 200

    # Mutual exclusivity between failure and normal holdout cohorts
    assert len(f_eids & n_eids) == 0, "Failure and normal cohorts must be disjoint"

    # Zero leakage into development set
    assert len(f_eids & dev_entities) == 0, "No failure entity may overlap with development"
    assert len(n_eids & dev_entities) == 0, "No normal entity may overlap with development"


def test_parent_job_decomposition_and_sensitivity_cohort(protocol):
    """Verifies the exact parent job split and sensitivity analysis cohort."""
    disjoint_eids = set(protocol["frozen_entity_ids"]["disjoint_job_failure_entities"])
    shared_eids = set(protocol["frozen_entity_ids"]["shared_job_failure_entities"])
    all_f_eids = set(protocol["frozen_entity_ids"]["all_failure_entities"])

    assert len(disjoint_eids) == 75
    assert len(shared_eids) == 62
    assert disjoint_eids | shared_eids == all_f_eids
    assert len(disjoint_eids & shared_eids) == 0

    disjoint_jobs = protocol["frozen_parent_job_ids"]["disjoint_failure_jobs"]
    shared_jobs = protocol["frozen_parent_job_ids"]["shared_failure_jobs"]

    assert sorted(disjoint_jobs) == ["6250345153", "6250804554", "6251723929", "6252225776"]
    assert sorted(shared_jobs) == ["5664371117", "6177502198", "6224307516", "6251812952", "6251995937"]


def test_sequence_length_and_target_horizons(protocol):
    """Verifies sequence length L=5 and target horizon specifications."""
    design = protocol["experimental_design_specification"]
    assert design["sequence_length_L"] == 5
    assert design["cadence_seconds"] == 300.0
    assert design["endpoint_span_seconds"] == 1200.0
    assert design["physical_collection_span_seconds"] == 1500.0
    assert design["require_full_history"] is True
    assert design["allow_zero_padding"] is False

    horizons = design["target_horizons"]
    assert "0 < time_to_failure <= 900.0s" in horizons["critical"]
    assert "900.0s < time_to_failure <= 1800.0s" in horizons["warning"]
    assert "time_to_failure > 1800.0s" in horizons["normal"]


def test_training_and_modeling_readiness(protocol):
    """Verifies modeling architecture, input shapes, and training readiness."""
    tm = protocol["training_and_modeling_readiness_audit"]
    assert tm["new_l5_training_mandated"] is True
    assert tm["l15_checkpoint_reuse_permitted"] is False
    assert len(tm["l15_checkpoint_reuse_rationale"]) > 30

    models = {m["model_name"]: m for m in tm["models_to_train_under_l5"]}
    assert "Random Forest" in models
    assert "Temporal LSTM" in models
    assert "Compact Temporal Transformer" in models
    assert "Representation Hybrid" in models

    assert models["Temporal LSTM"]["input_shape"] == "[batch_size, 5, 31]"
    assert models["Compact Temporal Transformer"]["input_shape"] == "[batch_size, 5, 31]"
    assert models["Representation Hybrid"]["input_shape"] == "[batch_size, 63]"

    # Preprocessing
    prep = tm["preprocessing_freeze"]
    assert prep["scaler_source"] == "configs/fitted_scaler.json"
    assert "train" in prep["fitted_strictly_on"].lower()
    assert Path(prep["scaler_source"]).exists()


def test_operational_gates_and_ranking_rule(protocol):
    """Verifies operational gates (FAR <= 12%, FPR <= 5%) and ranking tie-breakers."""
    plan = protocol["evaluation_protocol_and_analysis_plan"]
    gates = plan["operational_gates"]
    assert gates["entity_far_gate"] == "FAR_entity <= 12.0%"
    assert gates["window_fpr_gate"] == "FPR_window <= 5.0%"

    ranking = plan["model_selection_and_ranking_rule"]
    assert "BOTH operational gates" in ranking["eligibility"]
    assert "Episode Detection Recall" in ranking["ranking_criterion"]
    assert "Macro-F1" in ranking["tie_breaker"]
    assert "NO OPERATIONAL WINNER" in ranking["fallback_clause"]


def test_cluster_bootstrap_specification(protocol):
    """Verifies that clustering is defined at entity_id level and episodes are not treated as independent."""
    primary = protocol["evaluation_protocol_and_analysis_plan"]["primary_analysis"]
    assert primary["clustering_level"] == "entity_id"
    assert "strictly prohibited" in primary["clustering_rule"].lower()
    assert primary["independent_failure_clusters"] == 137
    assert primary["bootstrap_iterations"] == 1000
    assert primary["confidence_level"] == 0.95
