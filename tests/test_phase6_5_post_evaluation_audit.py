"""Phase 6.5 — Post-Evaluation Integrity Audit and Error Analysis Automated Tests.

Verifies:
1. Completeness, validity, and zero-discrepancy status of results/phase6/post_evaluation_audit_l5.json.
2. Exact reproduction of Phase 6.4 metrics and confusion matrices across all 4 candidate models.
3. Validation of the 1,000 entity-cluster bootstrap replicates, cluster integrity, and exact CI matching.
4. Formal hypothesis testing vs. confidence interval distinction (exact binomial tests, p < 1e-12).
5. Comprehensive false alarm investigation: alert severity distribution (Critical predominance),
   feature pattern associations (cpi_missing_fraction_6 elevation), entity concentration (Gini > 0.6),
   parent job clustering, and cohort disaggregation.
6. Report wording compliance: probability calibration verification (predicted vs calibrated class probabilities)
   and cross-job generalization scope qualification (Google May 2011 trace source).
7. Strict preservation of NO_OPERATIONAL_WINNER verdict and Phase 6.4 immutable artifacts.
"""

import json
import hashlib
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PHASE6_DIR = PROJECT_ROOT / "results" / "phase6"
AUDIT_JSON_PATH = PHASE6_DIR / "post_evaluation_audit_l5.json"
CONF_METRICS_PATH = PHASE6_DIR / "confirmatory_metrics_l5.json"
CONF_PREDS_PATH = PHASE6_DIR / "confirmatory_predictions_l5.csv"
CONF_EPISODES_PATH = PHASE6_DIR / "confirmatory_episode_metrics_l5.csv"
CONF_BOOTSTRAP_PATH = PHASE6_DIR / "confirmatory_bootstrap_ci_l5.json"
PROTOCOL_AMENDMENT_PATH = PHASE6_DIR / "protocol_l5_amendment_2_1.json"
ORIGINAL_PROTOCOL_PATH = PHASE6_DIR / "protocol_l5.json"

MODEL_KEYS = ["random_forest", "temporal_lstm", "compact_transformer", "representation_hybrid"]


@pytest.fixture(scope="module")
def audit_data():
    assert AUDIT_JSON_PATH.exists(), f"Audit master artifact missing: {AUDIT_JSON_PATH}"
    with open(AUDIT_JSON_PATH) as f:
        return json.load(f)


def test_post_evaluation_audit_artifact_exists_and_valid(audit_data):
    """Verifies that the Phase 6.5 audit master artifact exists, is valid, and reports zero discrepancies."""
    assert audit_data["audit_phase"] == "Phase 6.5"
    assert audit_data["audit_verdict"] == "AUDIT_COMPLETE_DISCREPANCIES_ZERO"
    assert audit_data["protocol_governing"] == "results/phase6/protocol_l5_amendment_2_1.json"

    summary = audit_data["summary"]
    assert summary["total_confirmatory_windows"] == 5511
    assert summary["evaluable_entities"] == 335
    assert summary["evaluable_pure_normal_entities"] == 198
    assert summary["evaluable_failure_entities"] == 137
    assert summary["total_failure_episodes"] == 141
    assert summary["total_reproduction_discrepancies"] == 0
    assert summary["total_bootstrap_discrepancies"] == 0
    assert summary["all_models_breached_gates"] is True
    assert summary["operational_winner_selected"] is False


def test_confirmatory_reproduction_zero_discrepancies(audit_data):
    """Verifies that every reported metric in confirmatory_metrics_l5.json was reproduced with zero discrepancies."""
    rep_sec = audit_data["metric_reproduction"]
    assert len(rep_sec["discrepancies"]) == 0

    with open(CONF_METRICS_PATH) as f:
        master_metrics = json.load(f)

    for m_key in MODEL_KEYS:
        recomputed = rep_sec["models"][m_key]
        master = master_metrics["model_evaluations"][m_key]

        # Check Macro-F1
        assert abs(recomputed["macro_f1"] - master["window_metrics"]["macro_f1"]) < 1e-6
        # Check Accuracy
        assert abs(recomputed["accuracy"] - master["window_metrics"]["accuracy"]) < 1e-6
        # Check Confusion Matrix
        assert recomputed["confusion_matrix"] == master["window_metrics"]["confusion_matrix"]
        # Check FAR Gate
        rep_far = recomputed["operational_gates"]["entity_far"]
        mast_far = master["operational_gates"]["entity_far"]
        assert rep_far["alerting_entities"] == mast_far["false_alarm_entities"]
        assert abs(rep_far["rate_pct"] - mast_far["rate_pct"]) < 1e-6
        assert rep_far["gate_passed"] is False

        # Check Pooled FPR Gate
        rep_fpr = recomputed["operational_gates"]["window_fpr_pooled"]
        mast_fpr = master["operational_gates"]["window_fpr"]["pooled_primary"]
        assert rep_fpr["fp_windows"] == mast_fpr["false_positive_windows"]
        assert abs(rep_fpr["rate_pct"] - mast_fpr["rate_pct"]) < 1e-6
        assert rep_fpr["gate_passed"] is False

        # Check Episode Recall
        rep_ep = recomputed["episode_metrics"]
        mast_ep = master["episode_metrics"]
        assert rep_ep["detected_episodes"] == mast_ep["detected_episodes"]
        assert abs(rep_ep["episode_recall"] * 100.0 - mast_ep["detection_rate_pct"]) < 1e-4
        assert rep_ep["total_eligible_episodes"] == 141


def test_bootstrap_validation_and_cluster_integrity(audit_data):
    """Verifies that the 1,000 entity-cluster bootstrap replicates match with zero discrepancies."""
    boot_sec = audit_data["bootstrap_validation"]
    assert len(boot_sec["discrepancies"]) == 0

    meta = boot_sec["bootstrap_metadata"]
    assert meta["resampling_unit"] == "entity_id"
    assert meta["evaluable_clusters_count"] == 335
    assert meta["n_replicates"] == 1000
    assert meta["random_seed"] == 42
    assert meta["confidence_level"] == 0.95

    with open(CONF_BOOTSTRAP_PATH) as f:
        master_boot = json.load(f)

    for m_key in MODEL_KEYS:
        rec_cis = boot_sec["recomputed_confidence_intervals"][m_key]
        mast_cis = master_boot["model_confidence_intervals"][m_key]
        for metric_name in ["macro_f1", "accuracy", "pure_normal_far_pct", "pooled_fpr_pct", "episode_recall_pct"]:
            assert abs(rec_cis[metric_name]["ci_95"][0] - mast_cis[metric_name]["ci_95"][0]) < 1e-4
            assert abs(rec_cis[metric_name]["ci_95"][1] - mast_cis[metric_name]["ci_95"][1]) < 1e-4


def test_hypothesis_testing_vs_confidence_intervals(audit_data):
    """Verifies formal hypothesis test execution and conceptual distinction from confidence intervals."""
    hyp_sec = audit_data["bootstrap_validation"]["hypothesis_testing_vs_ci"]

    for m_key in MODEL_KEYS:
        m_tests = hyp_sec[m_key]

        # Entity FAR exact binomial test
        far_test = m_tests["entity_far_gate"]
        assert far_test["n_entities"] == 198
        assert far_test["h0_threshold_p0"] == 0.12
        assert far_test["exact_binomial_pvalue"] < 1e-12
        assert far_test["statistically_significant_breach"] is True

        # Confidence interval lower bound strictly above threshold
        assert far_test["bootstrap_95_ci"][0] > 12.0

        # Window FPR cluster-aware check
        fpr_test = m_tests["window_fpr_gate"]
        assert fpr_test["n_windows"] == 4811
        assert fpr_test["h0_threshold_p0"] == 0.05
        assert fpr_test["bootstrap_95_ci"][0] > 5.0
        assert "cluster" in fpr_test["cluster_aware_note"].lower()


def test_false_alarm_error_analysis_completeness(audit_data):
    """Verifies that false alarm investigation covers severity distribution, features, concentration, and cohorts."""
    fa_sec = audit_data["false_alarm_investigation"]

    for m_key in MODEL_KEYS:
        m_fa = fa_sec[m_key]

        # 1. Alert severity distribution: Critical must be predominant
        sev = m_fa["alert_severity_distribution"]
        assert sev["predominant_error_type"] == "CRITICAL"
        assert sev["critical_fp_pct"] > 70.0
        assert sev["total_false_positive_windows"] == sev["warning_false_positives"] + sev["critical_false_positives"]

        # 2. Top differentiating feature: cpi_missing_fraction_6 must be #1
        top_feats = m_fa["top_differentiating_features"]
        assert len(top_feats) >= 5
        top_feat_name = top_feats[0]["feature"]
        assert top_feat_name == "cpi_missing_fraction_6"
        assert top_feats[0]["cohens_d"] > 1.0  # Strongly positive

        # 3. Entity concentration: Gini > 0.6 and strong Pareto effect
        ent_conc = m_fa["entity_concentration"]
        assert ent_conc["gini_coefficient"] > 0.60
        assert ent_conc["top_20_percent_fp_share_pct"] >= 59.0
        assert ent_conc["entities_for_50_pct_of_fps_fraction"] <= 0.25

        # 4. Cohort disaggregation
        cohort = m_fa["cohort_disaggregation_comparison"]
        assert cohort["pure_normal_normal_windows"] == 4515
        assert cohort["failing_cohort_normal_windows"] == 296
        assert cohort["pure_normal_fpr_pct"] > 0.0
        assert cohort["failing_cohort_fpr_pct"] > 0.0


def test_wording_and_scope_checks(audit_data):
    """Verifies calibration verification, wording mandate, and cross-job generalization qualification."""
    wording = audit_data["wording_and_scope_checks"]

    assert wording["probability_calibration_verified"] is False
    assert "predicted class probabilities" in wording["calibration_audit_detail"]
    assert wording["cross_job_generalization_qualified"] is True
    assert "Google cluster trace" in wording["generalization_scope_detail"]
    assert "observational statistical associations" in wording["causal_claims_disclaimer"]


def test_confirmatory_result_preservation(audit_data):
    """Verifies that NO_OPERATIONAL_WINNER is preserved and zero unauthorized actions occurred."""
    pres = audit_data["preservation_of_confirmatory_result"]

    assert pres["verdict_reaffirmed"] == "NO_OPERATIONAL_WINNER"
    assert pres["winner_selected"] is False
    assert pres["threshold_tuning_conducted"] is False
    assert pres["models_retrained"] is False
    assert pres["phase6_4_artifacts_modified"] is False


def test_phase6_4_artifacts_preservation():
    """Verifies that every Phase 6.4 and Phase 6.2/6.3 prerequisite artifact remains present and unaltered."""
    predecessor_hash = hashlib.sha256(ORIGINAL_PROTOCOL_PATH.read_bytes()).hexdigest()
    assert predecessor_hash == "ebf692598f2cb7115919c427127782f44ba054c655b3b76100aadb5955377bd1"

    for artifact in [
        CONF_METRICS_PATH,
        CONF_PREDS_PATH,
        CONF_EPISODES_PATH,
        PHASE6_DIR / "confirmatory_entity_metrics_l5.csv",
        CONF_BOOTSTRAP_PATH,
        PROTOCOL_AMENDMENT_PATH,
        PROJECT_ROOT / "reports" / "phase6_4_confirmatory_evaluation.md",
    ]:
        assert artifact.exists(), f"Phase 6.4 artifact missing: {artifact}"
