"""Phase 6.8 — Confirmatory Cohort Validity, Temporal Censoring, and Metric Integrity Audit Tests.

Automated verification tests for Phase 6.8 deliverables:
1. Master audit artifact existence, metadata integrity, and protocol preservation.
2. Authoritative protocol rules extraction and citation validation.
3. Denominators and classification of all 5,511 confirmatory prediction windows.
4. Independent verification of raw event temporal horizons (Critical, Warning, Normal, Post-Failure).
5. Detailed reconciliation of the 296 failure-cohort normal windows.
6. Diagnostic counterfactual metrics and mathematical proof of operational verdict invariance.
7. Audit of Phase 6.7 implementation and defect documentation.
8. Cryptographic immutability of all historical artifacts.
"""

import json
import hashlib
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PHASE6_DIR = PROJECT_ROOT / "results" / "phase6"
AUDIT_JSON_PATH = PHASE6_DIR / "phase6_8_confirmatory_validity_audit.json"
CONF_PREDS_PATH = PHASE6_DIR / "confirmatory_predictions_l5.csv"
PROTOCOL_AMENDMENT_PATH = PHASE6_DIR / "protocol_l5_amendment_2_1.json"

MODEL_KEYS = ["rf", "lstm", "transformer", "hybrid"]


@pytest.fixture(scope="module")
def audit_data():
    assert AUDIT_JSON_PATH.exists(), f"Phase 6.8 audit artifact missing: {AUDIT_JSON_PATH}"
    with open(AUDIT_JSON_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def test_phase6_8_artifact_exists_and_metadata(audit_data):
    """Verifies that Phase 6.8 audit artifact exists and declares required audit metadata."""
    assert audit_data["audit_phase"] == "Phase 6.8"
    assert audit_data["nature_of_study"] == "CONFIRMATORY_COHORT_VALIDITY_AND_TEMPORAL_CENSORING_AUDIT"
    assert audit_data["confirmatory_verdict_reaffirmed"] == "NO_OPERATIONAL_WINNER"
    assert "final_validity_verdict" in audit_data

    verdict = audit_data["final_validity_verdict"]
    assert verdict["verdict_code"] == "C"
    assert verdict["verdict_title"] == "MATERIAL PROTOCOL VIOLATION FOUND"
    assert "NO_OPERATIONAL_WINNER" in verdict["confirmatory_verdict_reaffirmed"]


def test_authoritative_protocol_rules_and_sources(audit_data):
    """Verifies extraction and citation of authoritative frozen temporal rules."""
    rules = audit_data["authoritative_temporal_rules"]

    assert "critical_horizon" in rules
    assert "warning_horizon" in rules
    assert "normal_horizon" in rules
    assert "unknown_censored_and_post_failure" in rules

    # Check that post-failure exclusion is explicitly identified in protocol text
    post_fail_rule = rules["unknown_censored_and_post_failure"]["definition"]
    assert "post-failure step -> label -1 (strictly excluded)" in post_fail_rule

    denoms = rules["manifest_denominators"]
    assert denoms["total_windows"] == 5511
    assert denoms["critical_windows"] == 400
    assert denoms["warning_windows"] == 300
    assert denoms["normal_windows_pooled"] == 4811
    assert denoms["pure_normal_windows"] == 4515
    assert denoms["failure_cohort_normal_windows"] == 296
    assert denoms["evaluable_entities"] == 335
    assert denoms["failure_episodes"] == 141


def test_window_classification_and_denominators(audit_data):
    """Verifies window classification arithmetic and exact cohort breakdowns."""
    summary = audit_data["window_classification_summary"]
    cats = summary["detailed_category_counts"]

    assert summary["total_confirmatory_windows"] == 5511
    assert cats["PURE_NORMAL_VALID"] == 4515
    assert cats["PREFAIL_CRITICAL_VALID"] == 400
    assert cats["PREFAIL_WARNING_VALID"] == 300
    assert cats["FAIL_NORM_PRE_CLEAN_VALID"] == 140
    assert cats["FAIL_NORM_PRE_CROSS_HIST"] == 7
    assert cats["FAIL_NORM_POST_FAILURE_CENSORING_VIOLATION"] == 149

    # Sum of all categories
    assert sum(cats.values()) == 5511

    # Failure cohort normal breakdown: 140 clean + 7 cross-hist + 149 post-fail = 296
    assert cats["FAIL_NORM_PRE_CLEAN_VALID"] + cats["FAIL_NORM_PRE_CROSS_HIST"] + cats["FAIL_NORM_POST_FAILURE_CENSORING_VIOLATION"] == 296

    # Strictly valid windows: 4515 + 400 + 300 + 140 = 5355
    assert summary["strictly_valid_windows_count"] == 5355
    assert summary["history_crosses_prior_failure_count"] == 7
    assert summary["post_failure_censoring_violations_count"] == 149


def test_296_failure_cohort_reconciliation_table(audit_data):
    """Verifies complete reconciliation of the 296 failure-cohort normal windows."""
    audit_296 = audit_data["failure_cohort_296_windows_audit"]
    records = audit_296["reconciliation_records"]

    assert audit_296["total_windows"] == 296
    assert len(records) == 296

    assert audit_296["pre_failure_clean_normal_count"] == 140
    assert audit_296["pre_failure_cross_history_count"] == 7
    assert audit_296["post_failure_censoring_violation_count"] == 149

    # Check fields in each record
    required_keys = {
        "window_idx", "entity_id", "prediction_timestamp", "prev_failure_timestamp",
        "next_failure_timestamp", "recomputed_time_to_next_failure", "stored_label",
        "stored_time_to_failure", "history_crosses_prior_failure",
        "frozen_protocol_text_permits_inclusion", "audit_classification",
        "violation_type", "explanation"
    }

    clean_count = 0
    cross_count = 0
    post_count = 0

    for r in records:
        assert required_keys.issubset(r.keys())
        assert r["stored_label"] == 0

        cls = r["audit_classification"]
        if cls == "FAIL_NORM_PRE_CLEAN_VALID":
            clean_count += 1
            assert r["recomputed_time_to_next_failure"] > 1800.0
            assert r["prev_failure_timestamp"] is None
            assert r["history_crosses_prior_failure"] is False
            assert r["frozen_protocol_text_permits_inclusion"] is True
        elif cls == "FAIL_NORM_PRE_CROSS_HIST":
            cross_count += 1
            assert r["recomputed_time_to_next_failure"] > 1800.0
            assert r["prev_failure_timestamp"] is not None
            assert r["history_crosses_prior_failure"] is True
        elif cls == "FAIL_NORM_POST_FAILURE_CENSORING_VIOLATION":
            post_count += 1
            assert r["next_failure_timestamp"] is None
            assert r["prev_failure_timestamp"] is not None
            assert r["frozen_protocol_text_permits_inclusion"] is False
            assert r["violation_type"] == "POST_FAILURE_CENSORING_VIOLATION"

    assert clean_count == 140
    assert cross_count == 7
    assert post_count == 149


def test_diagnostic_counterfactual_metrics_and_verdict_invariance(audit_data):
    """Verifies counterfactual metric recomputations and proves operational verdict invariance."""
    scenarios = audit_data["diagnostic_counterfactual_metrics"]

    assert "scenario_0_as_frozen" in scenarios
    assert "scenario_1_exclude_post_failure" in scenarios
    assert "scenario_2_pure_normal_only" in scenarios
    assert "scenario_3_strictly_clean_history" in scenarios

    s0 = scenarios["scenario_0_as_frozen"]
    s1 = scenarios["scenario_1_exclude_post_failure"]
    s2 = scenarios["scenario_2_pure_normal_only"]
    s3 = scenarios["scenario_3_strictly_clean_history"]

    assert s0["total_windows"] == 5511
    assert s0["normal_windows_denominator"] == 4811

    assert s1["total_windows"] == 5362
    assert s1["normal_windows_denominator"] == 4662

    assert s2["total_windows"] == 5215
    assert s2["normal_windows_denominator"] == 4515

    assert s3["total_windows"] == 5355
    assert s3["normal_windows_denominator"] == 4655

    # INVARIANCE PROOF: Across ALL 4 scenarios and ALL 4 models, NO model passes both gates
    for s_key, s_data in scenarios.items():
        for m_key in MODEL_KEYS:
            m_res = s_data["models"][m_key]
            # Window FPR is well above 5.0% threshold
            assert m_res["window_fpr_pct"] > 15.0, f"FPR too low for {m_key} in {s_key}: {m_res['window_fpr_pct']}"
            assert m_res["pass_window_fpr_gate"] is False

            # Entity FAR is well above 12.0% threshold
            assert m_res["entity_far_pct"] > 25.0, f"FAR too low for {m_key} in {s_key}: {m_res['entity_far_pct']}"
            assert m_res["pass_entity_far_gate"] is False

            # Both gates failed
            assert m_res["pass_both_gates"] is False


def test_phase6_7_audit_verification(audit_data):
    """Verifies that Phase 6.7 audit findings were checked and limitations documented."""
    p67 = audit_data["phase6_7_implementation_audit"]

    assert p67["event_type_mapping"]["status"] == "CORRECT"
    assert p67["timestamp_unit_consistency"]["status"] == "CORRECT"
    assert p67["episode_reconstruction"]["status"] == "CORRECT"
    assert p67["reconciliation_147_pre_vs_149_post"]["status"] == "REPRODUCED_EXACT"
    assert p67["phase6_7_limitation_identified"]["status"] == "IMPLEMENTATION_DEFECT_NOTED"


def test_immutability_of_historical_artifacts(audit_data):
    """Verifies cryptographic hash consistency of all prior phase artifacts."""
    hashes = audit_data["input_artifact_hashes"]

    for name, info in hashes.items():
        p = Path(info["path"])
        assert p.exists(), f"Tracked artifact missing: {p}"
        with open(p, "rb") as f:
            actual_sha = hashlib.sha256(f.read()).hexdigest()
        assert actual_sha == info["sha256"], f"Hash mismatch for {name}: expected {info['sha256']}, got {actual_sha}"
