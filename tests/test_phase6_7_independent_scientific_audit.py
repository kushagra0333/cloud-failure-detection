"""Phase 6.7 — Independent Scientific Audit of Error Mechanisms and Temporal Labels.

Automated verification tests for Phase 6.7 deliverables:
1. Master audit artifact existence, metadata integrity, and protocol preservation.
2. Independent denominators and cohort arithmetic invariants.
3. Timestamp reconstruction from ground-truth lifecycle events and 7-window reconciliation.
4. Feature distribution analysis, Cohen's d recomputations, and entity/parent-job cluster bootstrap CIs.
5. Parent-job concentration metrics (Gini, HHI, normalized HHI, Pareto shares) and post-hoc sensitivity caveats.
6. Probability distributions, argmax consistency, Critical triggering minima, and Transformer saturation.
7. Evidence classification ledger integrity and categorization statuses.
8. Immutability and cryptographic integrity of all prior phase artifacts.
"""

import json
import hashlib
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PHASE6_DIR = PROJECT_ROOT / "results" / "phase6"
AUDIT_JSON_PATH = PHASE6_DIR / "phase6_7_independent_scientific_audit.json"
CONF_METRICS_PATH = PHASE6_DIR / "confirmatory_metrics_l5.json"
CONF_PREDS_PATH = PHASE6_DIR / "confirmatory_predictions_l5.csv"
CONF_EPISODES_PATH = PHASE6_DIR / "confirmatory_episode_metrics_l5.csv"
AUDIT_PHASE6_5_PATH = PHASE6_DIR / "post_evaluation_audit_l5.json"
INVESTIGATION_PHASE6_6_PATH = PHASE6_DIR / "phase6_6_error_mechanism_investigation.json"
PROTOCOL_AMENDMENT_PATH = PHASE6_DIR / "protocol_l5_amendment_2_1.json"

MODEL_KEYS = ["random_forest", "temporal_lstm", "compact_transformer", "representation_hybrid"]


@pytest.fixture(scope="module")
def audit_data():
    assert AUDIT_JSON_PATH.exists(), f"Phase 6.7 audit artifact missing: {AUDIT_JSON_PATH}"
    with open(AUDIT_JSON_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def test_phase6_7_artifact_exists_and_metadata(audit_data):
    """Verifies that Phase 6.7 audit artifact exists and adheres to strict scientific audit metadata."""
    assert audit_data["audit_phase"] == "Phase 6.7"
    assert audit_data["nature_of_study"] == "INDEPENDENT_SCIENTIFIC_AUDIT_OF_ERROR_MECHANISMS"
    assert audit_data["confirmatory_verdict_reaffirmed"] == "NO_OPERATIONAL_WINNER"
    assert audit_data["governing_protocol"] == "results/phase6/protocol_l5_amendment_2_1.json"
    assert "input_artifact_hashes" in audit_data


def test_independent_denominators_and_cohort_invariants(audit_data):
    """Verifies all sample sizes, window counts, and cohort partitioning denominators."""
    denoms = audit_data["independent_denominators"]

    assert denoms["evaluable_pure_normal_entities"] == 198
    assert denoms["evaluable_failure_entities"] == 137
    assert denoms["total_evaluable_entities"] == 335
    assert denoms["evaluable_pure_normal_entities"] + denoms["evaluable_failure_entities"] == denoms["total_evaluable_entities"]

    assert denoms["pure_normal_windows"] == 4515
    assert denoms["failing_cohort_normal_windows"] == 296
    assert denoms["primary_normal_windows_total"] == 4811
    assert denoms["pure_normal_windows"] + denoms["failing_cohort_normal_windows"] == denoms["primary_normal_windows_total"]

    assert denoms["prefail_warning_windows"] == 300
    assert denoms["prefail_critical_windows"] == 400
    assert denoms["total_confirmatory_windows"] == 5511
    assert denoms["primary_normal_windows_total"] + denoms["prefail_warning_windows"] + denoms["prefail_critical_windows"] == denoms["total_confirmatory_windows"]

    assert denoms["total_failure_episodes"] == 141


def test_timing_reconstruction_and_discrepancy_reconciliation(audit_data):
    """Verifies independent event timing reconstruction, normal window classification, and 7-window reconciliation."""
    timing = audit_data["timing_reconstruction_audit"]

    assert timing["raw_fail_events_count"] == 153
    assert timing["reconstructed_episodes_count"] == 141
    assert timing["episodes_matching_raw_events_count"] == 141

    breakdown = timing["failing_cohort_normal_windows_breakdown"]
    assert breakdown["more_than_30m_before_next_failure"] == 147
    assert breakdown["after_prior_failure_no_subsequent_failure"] == 149
    assert breakdown["within_warning_horizon"] == 0
    assert breakdown["within_critical_horizon"] == 0
    assert breakdown["ambiguous_or_unverifiable"] == 0
    assert breakdown["total_failing_cohort_normal_windows"] == 296
    assert breakdown["more_than_30m_before_next_failure"] + breakdown["after_prior_failure_no_subsequent_failure"] == 296

    recon = timing["phase6_6_discrepancy_reconciliation"]
    assert recon["phase6_6_reported_pre_failure"] == 140
    assert recon["independent_verified_pre_failure"] == 147
    assert recon["discrepancy_count"] == 7
    assert sorted(recon["discrepancy_entities"]) == ["6224307516_2", "6224307516_3"]


def test_feature_investigation_and_cluster_bootstrap(audit_data):
    """Verifies feature distribution comparison, Cohen's d effect sizes, and cluster bootstrap intervals."""
    feat_audit = audit_data["feature_investigation_audit"]

    for m_key in MODEL_KEYS:
        assert m_key in feat_audit
        pn_data = feat_audit[m_key]["pure_normal_entities"]
        assert pn_data["sample_size_tn"] + pn_data["sample_size_fp"] == 4515

        # Check cpi_missing_fraction_6 is the top feature with positive effect
        top_feature = pn_data["top_5_features"][0]
        assert top_feature["feature"] == "cpi_missing_fraction_6"
        assert top_feature["cohens_d"] > 1.0

        # Cluster bootstrap intervals must be valid
        ent_ci = top_feature["entity_cluster_bootstrap_ci_95"]
        assert len(ent_ci) == 2
        assert ent_ci[0] < ent_ci[1]
        assert ent_ci[0] > 0.5  # Positive effect persists under entity clustering

        pj_ci = top_feature["parent_job_cluster_bootstrap_ci_95"]
        assert len(pj_ci) == 2
        assert pj_ci[0] < pj_ci[1]


def test_parent_job_concentration_metrics(audit_data):
    """Verifies concentration indices (Gini, HHI) and top-job false positive shares."""
    pj_audit = audit_data["parent_job_concentration_audit"]

    for m_key in MODEL_KEYS:
        assert m_key in pj_audit
        m_conc = pj_audit[m_key]
        assert m_conc["gini_coefficient"] > 0.60
        assert 0.0 <= m_conc["herfindahl_hirschman_index_normalized"] <= 1.0

        top1 = m_conc["top_1_job_share_pct"]
        top3 = m_conc["top_3_jobs_share_pct"]
        top5 = m_conc["top_5_jobs_share_pct"]
        top10 = m_conc["top_10_jobs_share_pct"]
        assert top1 <= top3 <= top5 <= top10

    # Check top 3 jobs drive > 50% in RF and Compact Transformer
    rf_top3 = pj_audit["random_forest"]["top_3_jobs_share_pct"]
    tf_top3 = pj_audit["compact_transformer"]["top_3_jobs_share_pct"]
    assert rf_top3 > 50.0  # ~50.63%
    assert tf_top3 > 50.0  # ~52.21%
    assert pj_audit["random_forest"]["claim_top_3_jobs_driving_50_to_61_pct_verified"] is True


def test_prediction_probabilities_and_argmax_invariants(audit_data):
    """Verifies probability validity, argmax rule adherence, Critical predominance, and Transformer saturation."""
    prob_audit = audit_data["prediction_probabilities_and_weights_audit"]

    for m_key in MODEL_KEYS:
        assert m_key in prob_audit
        m_prob = prob_audit[m_key]
        assert m_prob["probabilities_finite_and_valid"] is True
        assert m_prob["probabilities_sum_to_one"] is True
        assert m_prob["max_sum_deviation"] < 1e-4
        assert m_prob["predictions_strictly_match_argmax"] is True

        crit_share = m_prob["critical_fp_share_pct"]
        assert crit_share > 70.0  # All models are predominantly Critical

        min_crit_prob = m_prob["min_probability_to_trigger_critical"]
        assert min_crit_prob < 0.40  # Triggering occurs as low as 0.336-0.359 under 3-class argmax

    # Transformer softmax saturation
    tf_prob = prob_audit["compact_transformer"]
    assert tf_prob["median_probability_on_critical_fps"] > 0.99
    assert tf_prob["fraction_critical_fps_saturated_above_90_pct"] > 0.50


def test_evidence_classification_ledger(audit_data):
    """Verifies that all 9 Phase 6.6 claims are classified into valid evidence statuses."""
    ledger = audit_data["evidence_classifications"]
    assert len(ledger) == 9

    valid_statuses = {"REPRODUCED", "REPRODUCED WITH QUALIFICATIONS", "NOT REPRODUCED", "UNVERIFIABLE"}
    claim_ids = [c["claim_id"] for c in ledger]

    assert len(claim_ids) == len(set(claim_ids)), "Duplicate claim IDs found in ledger"

    expected_statuses = {
        "CLAIM_1_CRITICAL_PREDOMINANCE": "REPRODUCED",
        "CLAIM_2_CPI_MISSINGNESS_ELEVATION": "REPRODUCED WITH QUALIFICATIONS",
        "CLAIM_3_PARENT_JOB_CONCENTRATION": "REPRODUCED WITH QUALIFICATIONS",
        "CLAIM_4_TIMING_SUBGROUPS_PARTITION": "NOT REPRODUCED",
        "CLAIM_5_PRE_FAILURE_DEGRADATION_DETECTION": "REPRODUCED WITH QUALIFICATIONS",
        "CLAIM_6_ARGMAX_TRIGGERING_PROBABILITY": "REPRODUCED",
        "CLAIM_7_TRANSFORMER_SOFTMAX_SATURATION": "REPRODUCED",
        "CLAIM_8_UNIFORM_10X_LOSS_WEIGHTS": "NOT REPRODUCED",
        "CLAIM_9_CPI_MISSINGNESS_CAUSALITY": "UNVERIFIABLE",
    }

    for claim in ledger:
        cid = claim["claim_id"]
        status = claim["status"]
        assert status in valid_statuses, f"Invalid status {status} for {cid}"
        assert status == expected_statuses[cid], f"Status mismatch for {cid}: expected {expected_statuses[cid]}, got {status}"
        assert claim["denominator"] != ""
        assert claim["source_location"] != ""
        assert claim["interpretation"] != ""


def test_immutability_of_prior_phase_artifacts(audit_data):
    """Verifies that all Phase 6.4, 6.5, and 6.6 artifacts exist and remain unmodified."""
    hashes = audit_data["input_artifact_hashes"]

    for name, info in hashes.items():
        p = Path(info["path"])
        assert p.exists(), f"Tracked artifact missing: {p}"
        with open(p, "rb") as f:
            actual_sha = hashlib.sha256(f.read()).hexdigest()
        assert actual_sha == info["sha256"], f"Artifact hash changed for {name}: expected {info['sha256']}, got {actual_sha}"
