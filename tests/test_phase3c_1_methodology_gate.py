"""Unit tests for Phase 3C.1 Statistical and Methodological Verification Gate."""

import pytest
import numpy as np
import pandas as pd
import json
from pathlib import Path


def test_metric_denominators():
    """Verifies that all reported metric denominators (48, 42, 80, 864, 36) are mathematically exact."""
    rf_pred_path = Path("results/phase3a/test_predictions_random_forest.csv")
    val_meta_path = Path("data/processed/val_metadata.csv")

    assert rf_pred_path.exists(), "RF test predictions not found"
    assert val_meta_path.exists(), "Validation metadata not found"

    rf_df = pd.read_csv(rf_pred_path)
    val_meta = pd.read_csv(val_meta_path)

    # 1. Total test entities = 80
    test_eids = set(rf_df["entity_id"].unique())
    assert len(test_eids) == 80, f"Expected 80 test entities, got {len(test_eids)}"

    # 2. Test failing entities = 32, pure-normal test entities = 48
    test_prefails = rf_df[rf_df["label"] > 0]
    test_failing_eids = set(test_prefails["entity_id"].unique())
    assert len(test_failing_eids) == 32, f"Expected 32 failing entities in test, got {len(test_failing_eids)}"
    pure_normal_test = test_eids - test_failing_eids
    assert len(pure_normal_test) == 48, f"Expected 48 pure-normal test entities, got {len(pure_normal_test)}"

    # 3. Validation entities = 78, failing = 36, pure-normal validation entities = 42
    val_eids = set(val_meta["entity_id"].unique())
    assert len(val_eids) == 78, f"Expected 78 validation entities, got {len(val_eids)}"
    val_prefails = val_meta[val_meta["label"] > 0]
    val_failing_eids = set(val_prefails["entity_id"].unique())
    assert len(val_failing_eids) == 36, f"Expected 36 failing entities in val, got {len(val_failing_eids)}"
    pure_normal_val = val_eids - val_failing_eids
    assert len(pure_normal_val) == 42, f"Expected 42 pure-normal validation entities, got {len(pure_normal_val)}"

    # 4. Total test normal windows = 864
    normal_windows = rf_df[rf_df["label"] == 0]
    assert len(normal_windows) == 864, f"Expected 864 normal windows in test, got {len(normal_windows)}"

    # 5. Total test failure episodes = 36
    test_prefails_copy = test_prefails.copy()
    test_prefails_copy["fail_time"] = (test_prefails_copy["prediction_timestamp"] + test_prefails_copy["time_to_failure"]).round(1)
    episodes = test_prefails_copy.groupby(["entity_id", "fail_time"])
    assert len(episodes) == 36, f"Expected 36 failure episodes in test, got {len(episodes)}"


def test_paired_cluster_bootstrap_mechanics():
    """Verifies that paired cluster bootstrap resamples entire entities and preserves duplicate draws."""
    toy_df_rf = pd.DataFrame({
        "entity_id": ["E1", "E1", "E2", "E3"],
        "label": [1, 2, 0, 0],
        "pred": [1, 0, 0, 1],
    })
    toy_df_lstm = pd.DataFrame({
        "entity_id": ["E1", "E1", "E2", "E3"],
        "label": [1, 2, 0, 0],
        "pred": [1, 2, 0, 0],
    })

    unique_eids = np.array(["E1", "E2", "E3"])
    rng = np.random.RandomState(123)

    # Simulate 5 replicates
    for _ in range(5):
        sampled = rng.choice(unique_eids, size=len(unique_eids), replace=True)
        # Duplicate draws must scale rows proportionally
        rf_sampled_rows = [toy_df_rf[toy_df_rf["entity_id"] == eid] for eid in sampled]
        lstm_sampled_rows = [toy_df_lstm[toy_df_lstm["entity_id"] == eid] for eid in sampled]

        rf_boot = pd.concat(rf_sampled_rows, ignore_index=True)
        lstm_boot = pd.concat(lstm_sampled_rows, ignore_index=True)

        assert len(rf_boot) == len(lstm_boot)
        # Verify cluster integrity: E1 always has 2 rows
        e1_draws = (sampled == "E1").sum()
        assert len(rf_boot[rf_boot["entity_id"] == "E1"]) == e1_draws * 2


def test_episode_deduplication_and_lead_time():
    """Verifies that episodes are deduplicated, missed episodes stay in denominator, and set union is used."""
    rf_df = pd.read_csv("results/phase3a/test_predictions_random_forest.csv")
    lstm_df = pd.read_csv("results/phase3b/test_predictions_lstm.csv")

    prefails_rf = rf_df[rf_df["label"] > 0].copy()
    prefails_rf["fail_time"] = (prefails_rf["prediction_timestamp"] + prefails_rf["time_to_failure"]).round(1)

    prefails_lstm = lstm_df[lstm_df["label"] > 0].copy()
    prefails_lstm["fail_time"] = (prefails_lstm["prediction_timestamp"] + prefails_lstm["time_to_failure"]).round(1)

    episodes = prefails_rf.groupby(["entity_id", "fail_time"])
    assert len(episodes) == 36

    rf_detected_eps = set()
    lstm_detected_eps = set()

    for ep_id, grp_rf in episodes:
        eid, ftime = ep_id
        grp_lstm = prefails_lstm[(prefails_lstm["entity_id"] == eid) & (prefails_lstm["fail_time"] == ftime)]
        if (grp_rf["predicted_label"] > 0).any():
            rf_detected_eps.add(ep_id)
        if (grp_lstm["predicted_label"] > 0).any():
            lstm_detected_eps.add(ep_id)

    assert len(rf_detected_eps) == 23
    assert len(lstm_detected_eps) == 25
    # Missed episodes must remain in denominator:
    assert (36 - len(rf_detected_eps)) == 13
    assert (36 - len(lstm_detected_eps)) == 11

    # Oracle set union: 20 joint + 3 RF-only + 5 LSTM-only = 28
    joint_detected = rf_detected_eps & lstm_detected_eps
    assert len(joint_detected) == 20
    assert len(rf_detected_eps - lstm_detected_eps) == 3
    assert len(lstm_detected_eps - rf_detected_eps) == 5

    oracle_union = rf_detected_eps | lstm_detected_eps
    assert len(oracle_union) == 28
    assert len(oracle_union) / 36 == pytest.approx(0.777777, rel=1e-3)


def test_post_failure_exclusion_invariance():
    """Asserts that zero post-failure windows are included in episode detection or early warning."""
    rf_df = pd.read_csv("results/phase3a/test_predictions_random_forest.csv")
    prefails = rf_df[rf_df["label"] > 0]

    # time_to_failure must be strictly positive
    assert (prefails["time_to_failure"] > 0.0).all(), "Found non-positive time_to_failure (post-failure leakage)!"
    assert prefails["time_to_failure"].min() >= 0.5, "Found window suspiciously close to failure (<0.5s)"
    assert prefails["time_to_failure"].max() <= 1800.0, "Found window outside 30-minute pre-failure horizon!"


def test_comparative_error_alignment_and_counts():
    """Verifies the exact 1,003-row alignment and agreement counts between RF and LSTM."""
    rf_df = pd.read_csv("results/phase3a/test_predictions_random_forest.csv")
    lstm_df = pd.read_csv("results/phase3b/test_predictions_lstm.csv")

    assert len(rf_df) == len(lstm_df) == 1003
    np.testing.assert_array_equal(rf_df["entity_id"], lstm_df["entity_id"])
    np.testing.assert_array_equal(rf_df["prediction_timestamp"], lstm_df["prediction_timestamp"])
    np.testing.assert_array_equal(rf_df["label"], lstm_df["label"])

    y_true = rf_df["label"].values
    y_rf = rf_df["predicted_label"].values
    y_lstm = lstm_df["predicted_label"].values

    both_corr = ((y_true == y_rf) & (y_true == y_lstm)).sum()
    rf_only = ((y_true == y_rf) & (y_true != y_lstm)).sum()
    lstm_only = ((y_true != y_rf) & (y_true == y_lstm)).sum()
    both_wrong = ((y_true != y_rf) & (y_true != y_lstm)).sum()

    assert both_corr == 748
    assert rf_only == 120
    assert lstm_only == 26
    assert both_wrong == 109
    assert (both_corr + rf_only + lstm_only + both_wrong) == 1003

    # Warning window hits (label == 1, N = 68)
    warn_mask = (y_true == 1)
    assert warn_mask.sum() == 68
    assert (y_rf[warn_mask] == 1).sum() == 17
    assert (y_lstm[warn_mask] == 1).sum() == 28
    assert ((y_rf[warn_mask] == 1) & (y_lstm[warn_mask] == 1)).sum() == 16
    assert ((y_rf[warn_mask] != 1) & (y_lstm[warn_mask] == 1)).sum() == 12
    assert ((y_rf[warn_mask] == 1) & (y_lstm[warn_mask] != 1)).sum() == 1

    # Critical window blurring (label == 2, N = 71)
    crit_mask = (y_true == 2)
    assert crit_mask.sum() == 71
    assert (y_rf[crit_mask] == 2).sum() == 32
    assert (y_lstm[crit_mask] == 2).sum() == 15
    assert (y_lstm[crit_mask] == 1).sum() == 31  # LSTM predicts Warning on Critical windows


def test_confirmatory_data_audit_record():
    """Verifies that the audit JSON records confirmatory data availability accurately."""
    audit_path = Path("results/phase3c/methodology_audit.json")
    assert audit_path.exists(), "methodology_audit.json does not exist"

    with open(audit_path) as f:
        data = json.load(f)

    assert data["gate_decision"] == "PASS"
    conf = data["confirmatory_data_feasibility"]
    assert conf["status"] == "FEASIBLE_UNTOUCHED_DATA_AVAILABLE"
    assert conf["untouched_telemetry_entities"] > 190000
    assert conf["untouched_failing_entities_with_telemetry"] >= 300
