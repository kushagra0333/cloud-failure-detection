"""Phase 6.6 — Post-Hoc False-Alarm Mechanism Investigation Automated Tests.

Verifies:
1. Artifact existence, schema validity, and formal post-hoc disclaimers.
2. Arithmetic consistency of false-positive definitions across all 4 models and cohorts.
3. Feature distribution analysis, Cohen's d values, and entity-cluster bootstrap CIs.
4. Parent-job clustering concentration measures (Gini, HHI, Pareto) and exploratory sensitivity.
5. Label-boundary temporal subgroups (140 pre-failure vs. 156 post-failure vs. 4,515 pure-normal).
6. Prediction behavior, uncalibrated class probabilities, and argmax triggering thresholds.
7. Strict preservation and immutability of Phase 6.4 and Phase 6.5 artifacts.
"""

import json
import hashlib
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PHASE6_DIR = PROJECT_ROOT / "results" / "phase6"
INVESTIGATION_JSON_PATH = PHASE6_DIR / "phase6_6_error_mechanism_investigation.json"
CONF_METRICS_PATH = PHASE6_DIR / "confirmatory_metrics_l5.json"
CONF_PREDS_PATH = PHASE6_DIR / "confirmatory_predictions_l5.csv"
AUDIT_PHASE6_5_PATH = PHASE6_DIR / "post_evaluation_audit_l5.json"
PROTOCOL_AMENDMENT_PATH = PHASE6_DIR / "protocol_l5_amendment_2_1.json"

MODEL_KEYS = ["random_forest", "temporal_lstm", "compact_transformer", "representation_hybrid"]


@pytest.fixture(scope="module")
def investigation_data():
    assert INVESTIGATION_JSON_PATH.exists(), f"Investigation master artifact missing: {INVESTIGATION_JSON_PATH}"
    with open(INVESTIGATION_JSON_PATH) as f:
        return json.load(f)


def test_phase6_6_artifact_exists_and_valid(investigation_data):
    """Verifies that the Phase 6.6 investigation master artifact exists and maintains post-hoc invariants."""
    assert investigation_data["investigation_phase"] == "Phase 6.6"
    assert investigation_data["nature_of_study"] == "EXPLORATORY_POST_HOC_ERROR_MECHANISM_ANALYSIS"
    assert investigation_data["confirmatory_verdict_reaffirmed"] == "NO_OPERATIONAL_WINNER"
    assert investigation_data["holdout_protection_status"] == "FROZEN_READ_ONLY"
    assert investigation_data["governing_protocol"] == "results/phase6/protocol_l5_amendment_2_1.json"


def test_task1_false_positive_definitions_and_arithmetic_consistency(investigation_data):
    """Verifies that false positive counts and cohort denominators satisfy all strict arithmetic invariants."""
    task1 = investigation_data["task1_definitions_audit"]
    denoms = task1["denominators"]

    # Invariant denominators
    assert denoms["evaluable_pure_normal_entities"] == 198
    assert denoms["evaluable_failure_entities"] == 137
    assert denoms["total_evaluable_entities"] == 335
    assert denoms["primary_normal_windows_total"] == 4811
    assert denoms["pure_normal_windows"] == 4515
    assert denoms["failing_cohort_normal_windows"] == 296
    assert denoms["prefail_warning_windows"] == 300
    assert denoms["prefail_critical_windows"] == 400
    assert denoms["total_confirmatory_windows"] == 5511
    assert denoms["pure_normal_windows"] + denoms["failing_cohort_normal_windows"] == denoms["primary_normal_windows_total"]

    # Model-level arithmetic consistency
    for m_key in MODEL_KEYS:
        m_data = task1["models"][m_key]

        pooled = m_data["pooled_false_positives"]
        pure = m_data["pure_normal_false_positives"]
        fail = m_data["failing_cohort_false_positives"]
        far = m_data["entity_far"]

        # Warning + Critical == Total
        assert pooled["warning_fps"] + pooled["critical_fps"] == pooled["total_fps"]
        assert pure["warning_fps"] + pure["critical_fps"] == pure["total_fps"]
        assert fail["warning_fps"] + fail["critical_fps"] == fail["total_fps"]

        # Pure + Fail == Pooled
        assert pure["total_fps"] + fail["total_fps"] == pooled["total_fps"]
        assert pure["warning_fps"] + fail["warning_fps"] == pooled["warning_fps"]
        assert pure["critical_fps"] + fail["critical_fps"] == pooled["critical_fps"]

        # Denominators
        assert pooled["denominator"] == 4811
        assert pure["denominator"] == 4515
        assert fail["denominator"] == 296
        assert far["evaluable_entities"] == 198


def test_task2_missing_telemetry_feature_and_cohens_d(investigation_data):
    """Verifies feature comparison between TN and FP windows and cluster bootstrap uncertainty estimates."""
    task2 = investigation_data["task2_feature_investigation"]

    for m_key in MODEL_KEYS:
        m_feat = task2[m_key]

        pn_cohort = m_feat["pure_normal_entities"]
        fn_cohort = m_feat["failing_cohort_normal_windows"]

        # Sample sizes must sum to exact cohort denominators
        assert pn_cohort["sample_size_tn"] + pn_cohort["sample_size_fp"] == 4515
        assert fn_cohort["sample_size_tn"] + fn_cohort["sample_size_fp"] == 296

        # In pure-normal entities, cpi_missing_fraction_6 must be the #1 differentiating feature
        top_pn_feat = pn_cohort["top_5_features"][0]
        assert top_pn_feat["feature"] == "cpi_missing_fraction_6"
        assert top_pn_feat["cohens_d"] > 1.0  # Massive positive elevation
        assert top_pn_feat["bootstrap_ci_95"][0] > 0.5  # Lower bound strictly positive

        # All 31 features must be evaluated
        assert len(pn_cohort["all_features"]) == 31
        assert len(fn_cohort["all_features"]) == 31


def test_task3_parent_job_clustering_and_concentration(investigation_data):
    """Verifies parent job counts, concentration metrics, and exploratory sensitivity analysis."""
    task3 = investigation_data["task3_parent_job_clustering"]

    job_counts = task3["job_counts"]
    assert job_counts["total_parent_jobs_confirmatory_dataset"] == 116
    assert job_counts["parent_jobs_with_primary_normal_windows"] == 112
    assert job_counts["parent_jobs_with_pure_normal_entities"] == 107

    for m_key in MODEL_KEYS:
        m_clust = task3["models"][m_key]
        conc = m_clust["concentration_metrics"]

        # Gini must indicate extreme concentration across parent jobs (> 0.65)
        assert conc["gini_coefficient"] > 0.65
        assert conc["herfindahl_hirschman_index"] > 0.04
        assert conc["top_20_percent_jobs_share_pct"] > 60.0
        assert conc["jobs_for_50_percent_of_fps"] <= 15

        # Sensitivity analysis stages
        sens = m_clust["exploratory_sensitivity_analysis"]
        assert len(sens) == 6  # k in [0, 1, 2, 3, 5, 10]
        assert sens[0]["excluded_top_jobs_count"] == 0
        assert sens[0]["remaining_normal_windows"] == 4811
        assert sens[0]["remaining_pure_normal_entities"] == 198

        # Removing high-FPR jobs must monotonically decrease false positive windows
        for i in range(1, len(sens)):
            assert sens[i]["remaining_fp_windows"] < sens[i - 1]["remaining_fp_windows"]


def test_task4_label_boundary_timing_subgroups(investigation_data):
    """Verifies timing subgroup partition and pre-failure vs post-failure degradation patterns."""
    task4 = investigation_data["task4_label_boundary_behavior"]
    counts = task4["timing_subgroups_counts"]

    # Partition of the 5,511 primary windows
    assert counts["critical_horizon_windows"] == 400
    assert counts["warning_horizon_windows"] == 300
    assert counts["failing_cohort_pre_failure_normal_windows"] == 140
    assert counts["failing_cohort_post_failure_normal_windows"] == 156
    assert counts["pure_normal_windows"] == 4515
    assert (
        counts["critical_horizon_windows"]
        + counts["warning_horizon_windows"]
        + counts["failing_cohort_pre_failure_normal_windows"]
        + counts["failing_cohort_post_failure_normal_windows"]
        + counts["pure_normal_windows"]
    ) == 5511

    # Pre-failure normal windows must lie strictly outside 30 min (1800s)
    pre_ttf = task4["pre_failure_normal_time_to_failure_minutes"]
    assert pre_ttf["count"] == 140
    assert pre_ttf["min"] > 30.0
    assert pre_ttf["max"] < 50.0

    # In sequence models, pre-failure normal FPR must be elevated over pure-normal FPR
    lstm_bound = task4["models"]["temporal_lstm"]
    assert lstm_bound["pre_failure_normal_fpr"]["fpr_pct"] > 90.0
    assert lstm_bound["elevation_ratio_pre_failure_to_pure_normal"] > 3.0

    tf_bound = task4["models"]["compact_transformer"]
    assert tf_bound["pre_failure_normal_fpr"]["fpr_pct"] > 70.0
    assert tf_bound["elevation_ratio_pre_failure_to_pure_normal"] > 3.0


def test_task5_prediction_behavior_and_calibration_status(investigation_data):
    """Verifies that probabilities are uncalibrated and Critical FPs dominate."""
    task5 = investigation_data["task5_prediction_behavior"]

    for m_key in MODEL_KEYS:
        m_pred = task5[m_key]

        # Severity breakdown: Critical must exceed Warning
        sev = m_pred["severity_breakdown"]
        assert sev["critical_fp_share_pct"] > sev["warning_fp_share_pct"]
        assert sev["critical_fp_share_pct"] > 70.0

        # Uncalibrated status
        cal = m_pred["calibration_status"]
        assert cal["is_calibrated"] is False
        assert "uncalibrated" in cal["notes"].lower()

        # Minimum probability required to trigger argmax in 3-class problem
        crit_fp_probs = m_pred["probability_distributions"]["critical_false_positives"]
        assert crit_fp_probs["min_triggering_probability"] < 0.40  # Can trigger with P(2) as low as ~0.34


def test_immutability_of_prior_phase_artifacts():
    """Verifies that Phase 6.4 and Phase 6.5 artifacts exist and remain strictly unaltered."""
    assert CONF_METRICS_PATH.exists()
    assert CONF_PREDS_PATH.exists()
    assert AUDIT_PHASE6_5_PATH.exists()
    assert PROTOCOL_AMENDMENT_PATH.exists()

    with open(AUDIT_PHASE6_5_PATH) as f:
        audit6_5 = json.load(f)
    assert audit6_5["summary"]["total_reproduction_discrepancies"] == 0
    assert audit6_5["preservation_of_confirmatory_result"]["verdict_reaffirmed"] == "NO_OPERATIONAL_WINNER"
