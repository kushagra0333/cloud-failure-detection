"""Unit tests for Phase 4.1 Independent Scientific Audit."""

import pytest
import numpy as np
import pandas as pd
import json
from pathlib import Path
from sklearn.metrics import f1_score, accuracy_score, precision_recall_fscore_support


def test_validation_denominators():
    """Verifies all stated validation denominators: 1,015 windows, 840 normal, 72 warn, 103 crit, 78 entities, 42 pure-normal, 61 episodes."""
    m_val = pd.read_csv("data/processed/val_metadata.csv")

    assert len(m_val) == 1015
    assert (m_val["label"] == 0).sum() == 840
    assert (m_val["label"] == 1).sum() == 72
    assert (m_val["label"] == 2).sum() == 103
    assert m_val["entity_id"].nunique() == 78

    prefails = m_val[m_val["label"] > 0].copy()
    prefails["fail_time"] = (prefails["prediction_timestamp"] + prefails["time_to_failure"]).round(1)
    episodes = prefails.groupby(["entity_id", "fail_time"])
    assert len(episodes) == 61

    failing_eids = set(prefails["entity_id"].unique())
    all_eids = set(m_val["entity_id"].unique())
    pure_normal = all_eids - failing_eids
    assert len(pure_normal) == 42


def test_post_failure_exclusion_in_validation():
    """Asserts that no pre-failure window leaks post-failure telemetry."""
    m_val = pd.read_csv("data/processed/val_metadata.csv")
    prefails = m_val[m_val["label"] > 0]

    assert (prefails["time_to_failure"] > 0.0).all(), "Found non-positive time_to_failure!"
    assert prefails["time_to_failure"].min() >= 1.0, "Suspicious window < 1.0s before failure"
    assert prefails["time_to_failure"].max() <= 1800.0, "Window exceeds 30-min horizon"


def test_all_four_validation_predictions_aligned():
    """Verifies that all 4 model prediction CSVs exist, have 1,015 rows, and match metadata exactly."""
    m_val = pd.read_csv("data/processed/val_metadata.csv")
    pred_files = [
        "results/phase4/validation_predictions_random_forest.csv",
        "results/phase4/validation_predictions_lstm.csv",
        "results/phase4/validation_predictions_transformer.csv",
        "results/phase4/validation_predictions_hybrid.csv",
    ]

    for p in pred_files:
        path = Path(p)
        assert path.exists(), f"Prediction file not found: {p}"
        df = pd.read_csv(path)
        assert len(df) == 1015, f"{p} has length {len(df)} != 1015"
        np.testing.assert_array_equal(df["entity_id"].values, m_val["entity_id"].values)
        np.testing.assert_array_equal(df["prediction_timestamp"].values, m_val["prediction_timestamp"].values)
        np.testing.assert_array_equal(df["label"].values, m_val["label"].values)

        probs = df[["prob_normal", "prob_warning", "prob_critical"]].values
        assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5), f"{p}: Probabilities do not sum to 1"
        assert (probs >= 0.0).all() and (probs <= 1.0).all(), f"{p}: Invalid probabilities"


def test_independent_metric_recomputation_exactness():
    """Recomputes validation metrics independently and asserts <1e-5 difference from model_comparison.json."""
    m_val = pd.read_csv("data/processed/val_metadata.csv")
    with open("results/phase4/model_comparison.json") as f:
        saved_bundle = json.load(f)["models"]

    models = [
        ("Random Forest", "results/phase4/validation_predictions_random_forest.csv"),
        ("Temporal LSTM", "results/phase4/validation_predictions_lstm.csv"),
        ("Compact Temporal Transformer", "results/phase4/validation_predictions_transformer.csv"),
        ("Representation Hybrid", "results/phase4/validation_predictions_hybrid.csv"),
    ]

    for name, path in models:
        df = pd.read_csv(path)
        y_true = df["label"].values
        y_pred = df["predicted_label"].values

        recomputed_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0.0)
        recomputed_acc = accuracy_score(y_true, y_pred)
        _, r, _, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1, 2], zero_division=0.0)

        saved = saved_bundle[name]["metrics"]
        assert abs(saved["macro_f1"] - recomputed_f1) < 1e-5, f"{name}: Macro-F1 mismatch"
        assert abs(saved["accuracy"] - recomputed_acc) < 1e-5, f"{name}: Accuracy mismatch"
        assert abs(saved["class_metrics"]["warning"]["recall"] - r[1]) < 1e-5, f"{name}: Warning Recall mismatch"
        assert abs(saved["class_metrics"]["critical"]["recall"] - r[2]) < 1e-5, f"{name}: Critical Recall mismatch"


def test_confirmatory_protocol_quarantine_integrity():
    """Verifies that confirmatory protocol remains untouched, frozen, and strictly isolated."""
    proto_path = Path("results/phase4/confirmatory_protocol.json")
    assert proto_path.exists(), "Confirmatory protocol missing"

    with open(proto_path) as f:
        proto = json.load(f)

    assert proto["status"] == "FROZEN_PRE_REGISTERED_UNTOUCHED"
    assert proto["quarantine_policy"]["phase4_evaluation"] is False
    assert proto["quarantine_policy"]["model_selection_use"] is False

    fails = set(proto["frozen_entity_ids"]["failure_entities"])
    normals = set(proto["frozen_entity_ids"]["normal_entities"])
    assert len(fails) == 100
    assert len(normals) == 200
    assert len(fails & normals) == 0

    train_meta = pd.read_csv("data/processed/train_metadata.csv")
    val_meta = pd.read_csv("data/processed/val_metadata.csv")
    test_meta = pd.read_csv("data/processed/test_metadata.csv")

    dev_eids = set(train_meta["entity_id"]) | set(val_meta["entity_id"]) | set(test_meta["entity_id"])
    assert len(fails & dev_eids) == 0, "Confirmatory fails contaminated with development data!"
    assert len(normals & dev_eids) == 0, "Confirmatory normals contaminated with development data!"


def test_independent_audit_json_artifact():
    """Verifies that independent_audit.json records PASS decision and complete recomputations."""
    audit_path = Path("results/phase4/independent_audit.json")
    assert audit_path.exists(), "independent_audit.json missing"

    with open(audit_path) as f:
        data = json.load(f)

    assert data["auditor_verdict"] == "PASS"
    assert "denominators" in data
    assert data["denominators"]["validation_windows"] == 1015
    assert data["denominators"]["normal_windows"] == 840
    assert data["denominators"]["warning_windows"] == 72
    assert data["denominators"]["critical_windows"] == 103
    assert data["denominators"]["failure_episodes"] == 61
    assert data["denominators"]["pure_normal_entities"] == 42
    assert data["post_failure_exclusion"]["verified"] is True
