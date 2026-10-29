"""Unit tests for Phase 6.2.2: Frozen Protocol Consistency Audit."""

import json
from pathlib import Path
import pytest


@pytest.fixture(scope="module")
def consistency_data():
    audit_path = Path("results/phase6/protocol_consistency_audit.json")
    assert audit_path.exists(), "results/phase6/protocol_consistency_audit.json must exist"
    with open(audit_path) as f:
        return json.load(f)


def test_verdict_and_amendment_requirement(consistency_data):
    """Verifies that the consistency audit verdict is NOT_READY_REQUIRES_PROTOCOL_AMENDMENT."""
    assert consistency_data["verdict"] == "NOT_READY_REQUIRES_PROTOCOL_AMENDMENT"
    assert consistency_data["protocol_version_audited"] == "2.0-pre-registered-L5"
    assert len(consistency_data["verdict_rationale"]) > 50


def test_population_reconciliation_distinction(consistency_data):
    """Verifies the distinction between the primary manifest cohort (5,511) and expanded census (5,727)."""
    pop = consistency_data["population_reconciliation"]
    primary = pop["primary_manifest_cohort"]
    census = pop["expanded_census_cohort"]

    # Primary Manifest Totals
    assert primary["failure_entities"] == 137
    assert primary["normal_entities_evaluable"] == 198
    assert primary["prefail_windows_total"] == 700
    assert primary["prefail_critical_windows"] == 400
    assert primary["prefail_warning_windows"] == 300
    assert primary["failing_cohort_normal_windows"] == 296
    assert primary["pure_normal_cohort_windows"] == 4515
    assert primary["total_normal_windows"] == 4811
    assert primary["total_evaluable_windows"] == 5511

    # Expanded Census Totals
    assert census["failure_tasks_total"] == 198
    assert census["prefail_yielding_failure_entities"] == 137
    assert census["non_prefail_failing_tasks"] == 61
    assert census["total_normal_windows"] == 5027
    assert census["total_evaluable_windows"] == 5727

    # Discrepancy analysis
    delta = pop["discrepancy_analysis"]["delta"]
    assert delta["entities_omitted_from_manifest"] == 61
    assert delta["normal_windows_difference"] == 216
    assert delta["prefail_windows_difference"] == 0
    assert delta["failure_episodes_difference"] == 0


def test_false_alarm_specifications(consistency_data):
    """Verifies entity FAR and window FPR denominators and operational gates."""
    fa = consistency_data["false_alarm_specifications"]
    
    # Entity FAR
    assert fa["entity_far"]["denominator"] == 198
    assert fa["entity_far"]["max_allowable_false_alarm_entities"] == 23
    assert "FAR <= 12.0%" in fa["entity_far"]["operational_gate"]

    # Window FPR Primary
    assert fa["window_fpr_primary_manifest"]["denominator"] == 4811
    assert fa["window_fpr_primary_manifest"]["max_allowable_false_positive_windows"] == 240
    assert "FPR <= 5.0%" in fa["window_fpr_primary_manifest"]["operational_gate"]

    # Window FPR Expanded
    assert fa["window_fpr_expanded_census"]["denominator"] == 5027
    assert fa["window_fpr_expanded_census"]["max_allowable_false_positive_windows"] == 251

    # Disaggregated Reporting Mandate
    disagg = fa["disaggregated_reporting_mandate"]
    assert disagg["pure_normal_windows_denominator"] == 4515
    assert disagg["failing_cohort_normal_windows_denominator"] == 296


def test_artifact_consistency_matrix(consistency_data):
    """Verifies agreements and documented disagreements across artifacts."""
    matrix = consistency_data["artifact_consistency_matrix"]

    assert matrix["entity_counts"]["agreement"] is True
    assert matrix["episode_counts"]["agreement"] is True
    assert matrix["prefail_window_counts"]["agreement"] is True
    assert matrix["normal_window_counts"]["agreement"] is False
    assert matrix["total_window_counts"]["agreement"] is False
    assert matrix["sensitivity_cohort"]["agreement"] is True
    assert matrix["bootstrap_resampling_unit"]["agreement"] is True


def test_hybrid_model_audit_confirmation(consistency_data):
    """Verifies that in-sample training embeddings are confirmed accepted in the protocol."""
    hyb = consistency_data["hybrid_model_audit"]
    assert hyb["status"] == "CONFIRMED_ACCEPTED_IN_PROTOCOL"
    assert "in-sample" in hyb["details"].lower()


def test_proposed_protocol_amendment_2_1(consistency_data):
    """Verifies the structure and stipulations of Protocol Amendment 2.1."""
    amend = consistency_data["proposed_protocol_amendment_2_1"]
    assert amend["amendment_id"] == "Protocol-Amendment-2.1"
    assert amend["status"] == "PROPOSED_PENDING_REVIEW"
    assert len(amend["formal_stipulations"]) == 5
    assert any("5,511 windows" in s for s in amend["formal_stipulations"])
    assert any("4,811 normal windows" in s for s in amend["formal_stipulations"])
    assert any("240 false positive windows" in s for s in amend["formal_stipulations"])
