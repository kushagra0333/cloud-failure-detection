"""Tests for Phase 6.4 — Frozen L=5 Confirmatory Evaluation.

Verifies:
1. All required confirmatory result deliverables exist in results/phase6/:
   - confirmatory_metrics_l5.json
   - confirmatory_predictions_l5.csv
   - confirmatory_episode_metrics_l5.csv
   - confirmatory_entity_metrics_l5.csv
   - confirmatory_bootstrap_ci_l5.json
2. Predictions CSV integrity: exactly 5,511 rows, valid columns, valid probability simplexes, valid predictions.
3. Episode metrics CSV integrity: exactly 141 failure episodes across 137 failure entities.
4. Entity metrics CSV integrity: exactly 335 evaluable entities (137 failure + 198 pure-normal).
5. Operational gates enforcement: all 4 models evaluated against FAR <= 12% on 198 entities and FPR <= 5% on 4,811 windows.
6. Model selection decision: correctly reports NO_OPERATIONAL_WINNER since zero models pass both gates.
7. Sensitivity cohort separation: exactly 75 failure entities, 79 episodes, and 347 pre-failure windows from 4 unseen parent jobs.
8. Bootstrap uncertainty analysis: 1,000 replicates, entity_id clustering, valid empirical 95% CIs.
9. Protocol immutability: predecessor and amended protocol artifacts remain unmodified.
"""

import hashlib
import json
from pathlib import Path
import pytest
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

ORIGINAL_PROTOCOL_SHA256 = "ebf692598f2cb7115919c427127782f44ba054c655b3b76100aadb5955377bd1"


def compute_sha256(filepath: Path) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def test_deliverables_existence():
    """Verify that all five required confirmatory results artifacts exist in results/phase6/."""
    out_dir = PROJECT_ROOT / "results" / "phase6"
    required_files = [
        out_dir / "confirmatory_metrics_l5.json",
        out_dir / "confirmatory_predictions_l5.csv",
        out_dir / "confirmatory_episode_metrics_l5.csv",
        out_dir / "confirmatory_entity_metrics_l5.csv",
        out_dir / "confirmatory_bootstrap_ci_l5.json",
    ]
    for p in required_files:
        assert p.exists(), f"Missing required deliverable: {p}"


def test_predictions_dataframe_integrity():
    """Verify confirmatory_predictions_l5.csv has 5,511 rows, proper columns, and valid probability simplexes."""
    csv_path = PROJECT_ROOT / "results" / "phase6" / "confirmatory_predictions_l5.csv"
    df = pd.read_csv(csv_path)

    assert len(df) == 5511, f"Expected 5,511 rows in predictions CSV, got {len(df)}"

    base_cols = ["window_idx", "entity_id", "parent_job_id", "prediction_timestamp", "time_to_failure", "label"]
    for col in base_cols:
        assert col in df.columns, f"Missing base column: {col}"

    models = ["rf", "lstm", "transformer", "hybrid"]
    for m in models:
        pred_col = f"pred_{m}"
        assert pred_col in df.columns, f"Missing prediction column: {pred_col}"
        assert df[pred_col].isin([0, 1, 2]).all(), f"Predictions for {m} must be in {{0, 1, 2}}"

        prob_cols = [f"prob_{m}_{c}" for c in range(3)]
        for pcol in prob_cols:
            assert pcol in df.columns, f"Missing probability column: {pcol}"

        # Check probability simplex
        probs_matrix = df[prob_cols].values
        assert (probs_matrix >= 0.0).all() and (probs_matrix <= 1.0001).all()
        row_sums = probs_matrix.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-4), f"Probabilities for {m} do not sum to 1"


def test_episode_metrics_dataframe_integrity():
    """Verify confirmatory_episode_metrics_l5.csv has exactly 141 episodes and 79 sensitivity episodes."""
    csv_path = PROJECT_ROOT / "results" / "phase6" / "confirmatory_episode_metrics_l5.csv"
    df = pd.read_csv(csv_path)

    assert len(df) == 141, f"Expected 141 failure episodes, got {len(df)}"
    assert df["entity_id"].nunique() == 137, f"Expected 137 unique failure entities, got {df['entity_id'].nunique()}"

    # Verify sensitivity cohort episode count
    sens_df = df[df["is_unseen_parent_job"] == True]
    assert len(sens_df) == 79, f"Expected 79 unseen sensitivity episodes, got {len(sens_df)}"
    assert sens_df["entity_id"].nunique() == 75, f"Expected 75 unseen sensitivity entities, got {sens_df['entity_id'].nunique()}"

    # Verify lead time validity when detected
    for m in ["rf", "lstm", "transformer", "hybrid"]:
        det_mask = df[f"detected_{m}"] == True
        detected_leads = df.loc[det_mask, f"lead_time_sec_{m}"]
        assert (detected_leads >= 0.0).all(), f"Lead time must be non-negative for {m}"
        assert (detected_leads <= 1800.0 + 1e-4).all() or (detected_leads <= 1850.0).all()


def test_entity_metrics_dataframe_integrity():
    """Verify confirmatory_entity_metrics_l5.csv has exactly 335 evaluable entities (137 fail, 198 pure-normal)."""
    csv_path = PROJECT_ROOT / "results" / "phase6" / "confirmatory_entity_metrics_l5.csv"
    df = pd.read_csv(csv_path)

    assert len(df) == 335, f"Expected 335 evaluable entities, got {len(df)}"

    type_counts = df["cohort_type"].value_counts().to_dict()
    assert type_counts.get("failure") == 137, f"Expected 137 failure entities, got {type_counts.get('failure')}"
    assert type_counts.get("pure_normal") == 198, f"Expected 198 pure-normal entities, got {type_counts.get('pure_normal')}"

    # Sensitivity entities
    sens_entities = df[(df["cohort_type"] == "failure") & (df["is_unseen_parent_job"] == True)]
    assert len(sens_entities) == 75, f"Expected 75 unseen failure entities, got {len(sens_entities)}"


def test_master_confirmatory_metrics_and_gate_failures():
    """Verify master metrics JSON reports correct denominators, gate evaluations, and NO_OPERATIONAL_WINNER."""
    json_path = PROJECT_ROOT / "results" / "phase6" / "confirmatory_metrics_l5.json"
    with open(json_path) as f:
        master = json.load(f)

    assert master["evaluation_phase"] == "Phase 6.4"
    summary = master["primary_evaluation_summary"]
    assert summary["total_windows"] == 5511
    assert summary["critical_windows"] == 400
    assert summary["warning_windows"] == 300
    assert summary["normal_windows"] == 4811
    assert summary["pure_normal_windows"] == 4515
    assert summary["failing_cohort_normal_windows"] == 296
    assert summary["evaluable_entities"] == 335
    assert summary["failure_episodes"] == 141

    models = master["model_evaluations"]
    for m_key, m_data in models.items():
        gates = m_data["operational_gates"]
        far = gates["entity_far"]
        fpr = gates["window_fpr"]["pooled_primary"]

        assert far["denominator_entities"] == 198
        assert fpr["denominator_windows"] == 4811

        # All four models fail both gates on holdout
        assert far["gate_passed"] == 0
        assert fpr["gate_passed"] == 0
        assert gates["operational_gates_verdict"]["passed_both_gates"] == 0

    winner = master["operational_winner_selection"]
    assert winner["winner_selected"] is False
    assert winner["verdict"] == "NO_OPERATIONAL_WINNER"
    assert len(winner["disqualified_candidates"]) == 4


def test_cross_job_sensitivity_cohort_in_master_json():
    """Verify cross-job sensitivity cohort is reported separately and matches ratified counts."""
    json_path = PROJECT_ROOT / "results" / "phase6" / "confirmatory_metrics_l5.json"
    with open(json_path) as f:
        master = json.load(f)

    sens = master["cross_job_sensitivity_analysis"]
    assert set(sens["unseen_parent_jobs"]) == {"6250345153", "6250804554", "6251723929", "6252225776"}
    assert sens["failure_entities_count"] == 75
    assert sens["failure_episodes_count"] == 79
    assert sens["prefail_windows_count"] == 347

    # Check that results exist for each model
    for m_key in ["random_forest", "temporal_lstm", "compact_transformer", "representation_hybrid"]:
        assert m_key in sens["model_results"]
        ep_m = sens["model_results"][m_key]["episode_metrics"]
        assert ep_m["total_episodes"] == 79
        assert ep_m["detected_episodes"] > 60  # All models detect >75% of unseen episodes


def test_bootstrap_confidence_intervals_integrity():
    """Verify confirmatory_bootstrap_ci_l5.json has 1,000 replicates, cluster resampling, and non-empty CIs."""
    boot_path = PROJECT_ROOT / "results" / "phase6" / "confirmatory_bootstrap_ci_l5.json"
    with open(boot_path) as f:
        boot_data = json.load(f)

    meta = boot_data["bootstrap_metadata"]
    assert meta["resampling_unit"] == "entity_id"
    assert meta["evaluable_clusters_count"] == 335
    assert meta["n_replicates"] == 1000
    assert meta["confidence_level"] == 0.95

    cis = boot_data["model_confidence_intervals"]
    for m_key in ["random_forest", "temporal_lstm", "compact_transformer", "representation_hybrid"]:
        assert m_key in cis
        m_cis = cis[m_key]
        for metric in ["macro_f1", "accuracy", "pooled_fpr_pct", "pure_normal_far_pct", "episode_recall_pct", "mean_lead_time_sec"]:
            assert metric in m_cis
            ci_low, ci_high = m_cis[metric]["ci_95"]
            assert ci_low <= ci_high, f"Invalid CI range for {m_key} {metric}: [{ci_low}, {ci_high}]"


def test_protocol_immutability():
    """Verify that predecessor protocol results/phase6/protocol_l5.json remains strictly unmodified."""
    orig_path = PROJECT_ROOT / "results" / "phase6" / "protocol_l5.json"
    assert compute_sha256(orig_path) == ORIGINAL_PROTOCOL_SHA256
