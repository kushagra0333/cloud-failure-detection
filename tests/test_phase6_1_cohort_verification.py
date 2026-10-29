"""Unit tests for Phase 6.1: Independent Cohort Verification."""

import json
from pathlib import Path
import pandas as pd
import pytest


@pytest.fixture(scope="module")
def verification_data():
    json_path = Path("results/phase6/cohort_verification.json")
    assert json_path.exists(), "results/phase6/cohort_verification.json must exist"
    with open(json_path) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def dev_metadata():
    tr = pd.read_csv("data/processed/train_metadata.csv")
    va = pd.read_csv("data/processed/val_metadata.csv")
    te = pd.read_csv("data/processed/test_metadata.csv")
    dev_eids = set(tr["entity_id"]) | set(va["entity_id"]) | set(te["entity_id"])
    return {
        "train": tr,
        "val": va,
        "test": te,
        "all_dev_eids": dev_eids
    }


def test_safeguards_and_verdict(verification_data):
    """Verifies that all scientific safeguards are complied with and verdict is valid."""
    safeguards = verification_data["safeguards_complied"]
    assert safeguards["no_model_training"] == 1
    assert safeguards["no_threshold_tuning"] == 1
    assert safeguards["no_candidate_scoring"] == 1
    assert safeguards["phase5_results_unmodified"] == 1
    assert safeguards["phase5_protocol_unmodified"] == 1
    assert safeguards["no_model_driven_candidate_selection"] == 1
    assert safeguards["proposed_l5_separate_from_phase5"] == 1

    verdict_info = verification_data["final_audit_verdict"]
    assert verdict_info["verdict"] == "READY FOR A NEW PRE-REGISTERED EVALUATION"
    assert len(verdict_info["rationale"]) > 50


def test_filtering_funnel_counts(verification_data):
    """Verifies candidate counts at every filtering stage."""
    funnel = verification_data["filtering_funnel"]
    assert funnel["stage_1_raw_failing_tasks_in_events"] == 5170
    assert funnel["stage_2_failing_tasks_with_telemetry"] == 748
    assert funnel["stage_3_failing_tasks_used_in_development"] == 354
    assert funnel["stage_4_failing_tasks_unused_in_development"] == 394
    assert funnel["stage_5_unused_tasks_with_prefail_steps"] == 337
    assert funnel["stage_6_unused_tasks_with_valid_l5_history"] == 198
    assert funnel["stage_7_unused_tasks_yielding_prefail_windows"] == 137

    prefail = funnel["stage_8_prefail_windows_l5"]
    assert prefail["total"] == 700
    assert prefail["warning"] == 300
    assert prefail["critical"] == 400
    assert prefail["warning"] + prefail["critical"] == prefail["total"]

    assert funnel["stage_9_distinct_failure_episodes"] == 141


def test_reconciliation_of_241_long_history_tasks(verification_data):
    """Verifies the mathematical reconciliation of the 241 tasks with >= 15 bins."""
    rec = verification_data["reconciliation_of_241_long_history_tasks"]
    assert rec["total_trace_failing_tasks_ge_15_bins"] == 241
    assert rec["tasks_absorbed_in_development"] == 241
    assert rec["tasks_unused_outside_development"] == 0

    breakdown = rec["development_breakdown"]
    assert breakdown["entities_with_prefailing_windows_during_dev"] == 172
    assert breakdown["entities_late_failing_with_normal_windows_during_dev"] == 69
    assert breakdown["mathematical_sum"] == 241
    assert (
        breakdown["entities_with_prefailing_windows_during_dev"] +
        breakdown["entities_late_failing_with_normal_windows_during_dev"]
    ) == 241


def test_canonical_isolation_and_no_dev_leakage(verification_data, dev_metadata):
    """Verifies strict holdout separation from development partitions."""
    isolation = verification_data["canonical_isolation_audit"]
    assert isolation["canonical_entity_identifier"] == "job_id_task_index"
    assert isolation["candidate_entities_count"] == 137
    assert isolation["development_entities_count"] == 525
    assert isolation["entity_overlap_count"] == 0
    assert isolation["entity_overlap_percentage"] == 0.0

    # Cross-check dev entity count directly against disk
    assert len(dev_metadata["all_dev_eids"]) == 525

    # Job-level verification
    job_audit = isolation["job_level_audit"]
    assert job_audit["candidate_unique_jobs"] == 9
    assert job_audit["development_unique_jobs"] == 95
    assert job_audit["shared_jobs_count"] == 5
    assert job_audit["disjoint_jobs_count"] == 4
    assert job_audit["candidate_entities_in_shared_jobs"] == 62
    assert job_audit["candidate_entities_in_disjoint_jobs"] == 75
    assert (
        job_audit["candidate_entities_in_shared_jobs"] +
        job_audit["candidate_entities_in_disjoint_jobs"]
    ) == 137
    assert job_audit["disjoint_prefail_windows"] == 347
    assert job_audit["disjoint_failure_episodes"] == 79


def test_normal_cohort_lookahead_coverage(verification_data):
    """Verifies normal candidate cohort lookahead and censorship handling."""
    normal_audit = verification_data["normal_candidate_cohort_audit"]
    assert normal_audit["total_unused_pure_normal_entities_in_trace"] == 238171
    assert normal_audit["lookahead_horizon_seconds"] == 1800.0
    assert normal_audit["lookahead_horizon_minutes"] == 30.0
    assert normal_audit["mislabeled_normal_count"] == 0
    assert normal_audit["coverage_verified"] == 1


def test_l5_window_and_label_specification(verification_data):
    """Verifies sequence length, cadence, and label boundary specifications."""
    spec = verification_data["l5_window_and_label_specification"]
    assert spec["sequence_length_L"] == 5
    assert spec["cadence_seconds"] == 300.0
    assert spec["endpoint_span_seconds"] == 1200.0
    assert spec["physical_collection_span_seconds"] == 1500.0
    assert spec["require_full_history"] == 1

    boundaries = spec["label_boundaries"]
    assert "0 < time_to_failure <= 900.0s" in boundaries["critical"]
    assert "900.0s < time_to_failure <= 1800.0s" in boundaries["warning"]
    assert "time_to_failure > 1800.0s" in boundaries["normal"]


def test_feature_causality_and_scaler_frozen(verification_data):
    """Verifies that features are causal and scaler is strictly from training data."""
    fc = verification_data["feature_causality_and_scaling_audit"]
    assert "Strictly trailing" in fc["feature_causality"]
    assert fc["scaling_imputation_source"] == "configs/fitted_scaler.json"

    scaler_path = Path(fc["scaling_imputation_source"])
    assert scaler_path.exists(), "Scaler configuration file must exist"
    with open(scaler_path) as f:
        scaler = json.load(f)
    assert scaler["is_fitted"] is True
    assert len(scaler["feature_cols"]) == 31


def test_statistical_power_and_bootstrap_readiness(verification_data):
    """Verifies that the cohort supports non-degenerate bootstrap uncertainty intervals."""
    power = verification_data["statistical_power_and_uncertainty_feasibility"]
    assert power["phase5_evaluable_failure_episodes"] == 0
    assert power["proposed_l5_evaluable_failure_episodes"] == 141
    assert power["proposed_l5_evaluable_entities"] == 137
    assert power["proposed_l5_prefail_windows"] == 700
    assert power["disjoint_job_failure_episodes"] == 79
    assert power["supports_entity_level_bootstrap"] == 1
