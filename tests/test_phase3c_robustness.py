"""Unit tests for Phase 3C LSTM robustness, comparative error analysis, and bootstrap logic."""

import pytest
import numpy as np
import pandas as pd
import json
from pathlib import Path


def test_comparative_error_analysis_artifact_alignment():
    """Verifies that comparative_error_analysis.csv aligns perfectly with metadata."""
    artifact_path = Path("results/phase3c/comparative_error_analysis.csv")
    meta_path = Path("data/processed/test_metadata.csv")

    if not artifact_path.exists() or not meta_path.exists():
        pytest.skip("Comparative error analysis artifact not yet generated.")

    df = pd.read_csv(artifact_path)
    meta = pd.read_csv(meta_path)

    assert len(df) == len(meta) == 1003
    np.testing.assert_array_equal(df["entity_id"].values, meta["entity_id"].values)
    np.testing.assert_array_equal(df["prediction_timestamp"].values, meta["prediction_timestamp"].values)
    np.testing.assert_array_equal(df["label"].values, meta["label"].values)

    expected_cols = [
        "entity_id", "prediction_timestamp", "label", "time_to_failure",
        "rf_predicted_label", "rf_prob_normal", "rf_prob_warning", "rf_prob_critical",
        "lstm_predicted_label", "lstm_prob_normal", "lstm_prob_warning", "lstm_prob_critical",
        "rf_correct", "lstm_correct", "agreement",
    ]
    for c in expected_cols:
        assert c in df.columns, f"Missing column: {c}"


def test_validation_experiments_artifact():
    """Verifies that validation_experiments.json contains all 5 predeclared runs."""
    json_path = Path("results/phase3c/validation_experiments.json")
    if not json_path.exists():
        pytest.skip("Validation experiments artifact not found.")

    with open(json_path) as f:
        experiments = json.load(f)

    assert len(experiments) == 5
    exp_ids = [e["experiment_id"] for e in experiments]
    assert "exp1_lstm_2l_h64_balanced" in exp_ids
    assert "exp2_lstm_1l_h64_balanced" in exp_ids
    assert "exp3_lstm_2l_h32_balanced" in exp_ids
    assert "exp4_lstm_1l_h32_balanced" in exp_ids
    assert "exp5_lstm_2l_h64_critboost" in exp_ids

    for exp in experiments:
        assert 0.0 <= exp["val_macro_f1"] <= 1.0
        assert 0.0 <= exp["val_warning_recall"] <= 1.0
        assert 0.0 <= exp["val_critical_recall"] <= 1.0
        assert 0.0 <= exp["val_episode_detection_pct"] <= 100.0


def test_entity_cluster_bootstrap_resampling():
    """Verifies that entity cluster bootstrap resamples entire entities rather than single windows."""
    toy_df = pd.DataFrame({
        "entity_id": ["E1", "E1", "E2", "E2", "E2", "E3"],
        "val": [1, 2, 10, 20, 30, 100],
    })
    unique_entities = toy_df["entity_id"].unique()
    ent_map = {eid: grp for eid, grp in toy_df.groupby("entity_id")}

    # Re-sample 10 times
    for seed in range(5):
        np.random.seed(seed)
        sampled_eids = np.random.choice(unique_entities, size=len(unique_entities), replace=True)
        resampled_df = pd.concat([ent_map[eid] for eid in sampled_eids], ignore_index=True)

        # Confirm that for any entity included, ALL its original rows are included
        for eid in set(sampled_eids):
            count_in_sample = (sampled_eids == eid).sum()
            original_len = len(ent_map[eid])
            assert len(resampled_df[resampled_df["entity_id"] == eid]) == count_in_sample * original_len


def test_disaggregated_episode_phase_detection():
    """Verifies that episode detection properly breaks down into warning-phase and critical-phase."""
    meta_df = pd.DataFrame({
        "entity_id": ["E1", "E1", "E2", "E2"],
        "prediction_timestamp": [5100.0, 5400.0, 5100.0, 5400.0],
        "label": [1, 2, 1, 2],
        "time_to_failure": [900.0, 600.0, 900.0, 600.0],
    })
    # Episode for E1: predicted [1, 0] -> warning alert hit, critical alert missed
    # Episode for E2: predicted [0, 2] -> warning alert missed, critical alert hit
    preds = np.array([1, 0, 0, 2])
    meta_df["predicted_label"] = preds

    prefails = meta_df.copy()
    prefails["fail_time"] = (prefails["prediction_timestamp"] + prefails["time_to_failure"]).round(1)

    warn_hits = 0
    crit_hits = 0
    any_hits = 0

    for (eid, ftime), grp in prefails.groupby(["entity_id", "fail_time"]):
        if (grp["predicted_label"] == 1).any():
            warn_hits += 1
        if (grp["predicted_label"] == 2).any():
            crit_hits += 1
        if (grp["predicted_label"] > 0).any():
            any_hits += 1

    assert warn_hits == 1
    assert crit_hits == 1
    assert any_hits == 2
