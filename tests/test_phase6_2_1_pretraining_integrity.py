"""Unit tests for Phase 6.2.1: Pre-Training Integrity Audit."""

import json
from pathlib import Path
import pytest


@pytest.fixture(scope="module")
def audit_data():
    audit_path = Path("results/phase6/pretraining_integrity_audit.json")
    assert audit_path.exists(), "results/phase6/pretraining_integrity_audit.json must exist"
    with open(audit_path) as f:
        return json.load(f)


def test_verdict_and_schema(audit_data):
    """Verifies that audit verdict is READY_FOR_L5_TRAINING and protocol version is 2.0-pre-registered-L5."""
    assert audit_data["verdict"] == "READY_FOR_L5_TRAINING"
    assert audit_data["protocol_version_audited"] == "2.0-pre-registered-L5"
    assert len(audit_data["verdict_rationale"]) > 50


def test_episode_distribution_audit(audit_data):
    """Verifies the exact distribution of episodes per entity (133 with 1, 4 with 2)."""
    ep_audit = audit_data["episode_distribution_audit"]
    assert ep_audit["total_episodes"] == 141
    assert ep_audit["total_entities"] == 137
    dist = ep_audit["distribution"]
    assert dist["1"] == 133 or dist[1] == 133
    assert dist["2"] == 4 or dist[2] == 4
    assert ep_audit["multi_episode_entities_count"] == 4
    expected_multi = ["6250345153_388", "6250804554_0", "6250804554_1", "6250804554_2"]
    assert sorted(list(ep_audit["multi_episode_entities"].keys())) == sorted(expected_multi)


def test_window_reconciliation(audit_data):
    """Verifies reconciliation between 996 vs 1,212 vs 5,511 vs 5,727 windows."""
    w_rec = audit_data["window_reconciliation"]
    fail_w = w_rec["frozen_failing_entities_windows"]
    assert fail_w["critical_windows"] == 400
    assert fail_w["warning_windows"] == 300
    assert fail_w["normal_windows"] == 296
    assert fail_w["total_windows"] == 996

    census_w = w_rec["phase6_1_unused_failing_tasks_windows"]
    assert census_w["critical_windows"] == 400
    assert census_w["warning_windows"] == 300
    assert census_w["normal_windows"] == 512
    assert census_w["total_windows"] == 1212

    diff = w_rec["difference_explanation"]
    assert diff["tasks_count"] == 61
    assert diff["windows_count"] == 216
    assert diff["window_type"] == "Normal (y=0)"

    norm_w = w_rec["normal_control_cohort_windows"]
    assert norm_w["entities_sampled"] == 200
    assert norm_w["entities_evaluable_l5"] == 198
    assert norm_w["entities_excluded_short_history"] == 2
    assert norm_w["normal_windows"] == 4515

    totals = w_rec["grand_total_evaluable_windows"]
    assert totals["frozen_cohort_total"] == 5511
    assert totals["protocol_summary_census_total"] == 5727


def test_isolation_audit(audit_data):
    """Verifies zero entity leakage and absolute absence of disjoint parent jobs from dev."""
    iso = audit_data["isolation_audit"]
    assert iso["entity_leakage_count"] == 0
    assert iso["holdout_failure_entities_count"] == 137
    assert iso["holdout_normal_entities_count"] == 200
    assert iso["disjoint_parent_jobs_in_dev"] == 0
    assert sorted(iso["disjoint_parent_jobs"]) == ["6250345153", "6250804554", "6251723929", "6252225776"]


def test_hybrid_training_audit(audit_data):
    """Verifies the hybrid training audit: in-sample embeddings, zero holdout leakage."""
    hyb = audit_data["hybrid_training_audit"]
    assert hyb["embedding_extraction_mode"] == "IN_SAMPLE"
    assert hyb["out_of_fold_cross_fitting"] is False
    assert hyb["validation_or_holdout_leakage"] is False
    assert hyb["holdout_quarantine_preserved"] is True
    assert "Amendment 2.2" in hyb["recommendation_amendment_2_2"]


def test_evaluation_denominators(audit_data):
    """Verifies exact metric denominators and operational gates."""
    denom = audit_data["evaluation_denominators"]
    assert denom["pure_normal_entities_denominator"] == 198
    assert denom["prefail_windows_denominator"] == 700
    assert denom["warning_windows_denominator"] == 300
    assert denom["critical_windows_denominator"] == 400
    assert denom["failure_episodes_denominator_primary"] == 141
    assert denom["failure_episodes_denominator_disjoint"] == 79
    assert denom["failure_episodes_denominator_shared"] == 62

    gates = denom["operational_gates"]
    assert "12.0%" in gates["entity_far_gate"]
    assert "5.0%" in gates["window_fpr_gate"]
