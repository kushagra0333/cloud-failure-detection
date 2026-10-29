"""Unit tests for Phase 3A Baseline Modeling, evaluation metrics, and prediction alignment."""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from src.models.baselines import (
    MajorityClassBaseline,
    ClassWeightedLogisticRegression,
    ClassWeightedRandomForest,
    extract_feature_vector,
)
from src.evaluation.metrics import (
    compute_window_metrics,
    compute_episode_metrics,
)


def test_extract_feature_vector_strategies():
    """Tests that sequence-to-vector adapters return correct shapes."""
    N, L, F = 10, 15, 31
    X_seq = np.random.randn(N, L, F).astype(np.float32)

    vec_final = extract_feature_vector(X_seq, strategy="final_step")
    assert vec_final.shape == (N, F)
    np.testing.assert_array_equal(vec_final, X_seq[:, -1, :])

    vec_summary = extract_feature_vector(X_seq, strategy="summary")
    assert vec_summary.shape == (N, 3 * F)

    vec_flat = extract_feature_vector(X_seq, strategy="flattened")
    assert vec_flat.shape == (N, L * F)

    with pytest.raises(ValueError):
        extract_feature_vector(X_seq, strategy="invalid_strategy")  # type: ignore


def test_window_metrics_computation():
    """Verifies that window metrics handle per-class P/R/F1, Macro-F1, and PR-AUC."""
    y_true = np.array([0, 0, 0, 1, 1, 2, 2, 2])
    y_pred = np.array([0, 0, 1, 1, 0, 2, 2, 1])
    y_prob = np.array([
        [0.8, 0.1, 0.1],
        [0.7, 0.2, 0.1],
        [0.3, 0.5, 0.2],
        [0.1, 0.8, 0.1],
        [0.6, 0.3, 0.1],
        [0.1, 0.2, 0.7],
        [0.1, 0.1, 0.8],
        [0.2, 0.5, 0.3],
    ])

    metrics = compute_window_metrics(y_true, y_pred, y_prob)

    assert "macro_f1" in metrics
    assert "class_metrics" in metrics
    assert "NORMAL" in metrics["class_metrics"]
    assert "WARNING" in metrics["class_metrics"]
    assert "CRITICAL" in metrics["class_metrics"]

    # Check that macro_f1 is the average of the 3 class F1 scores
    f1_0 = metrics["class_metrics"]["NORMAL"]["f1"]
    f1_1 = metrics["class_metrics"]["WARNING"]["f1"]
    f1_2 = metrics["class_metrics"]["CRITICAL"]["f1"]
    assert np.isclose(metrics["macro_f1"], (f1_0 + f1_1 + f1_2) / 3.0)

    # Check confusion matrix shape and row sums
    cm = np.array(metrics["confusion_matrix"])
    assert cm.shape == (3, 3)
    np.testing.assert_array_equal(cm.sum(axis=1), [3, 2, 3])

    # Check PR-AUC is bounded between 0 and 1
    assert 0.0 <= metrics["pr_auc"]["pr_auc_warning"] <= 1.0
    assert 0.0 <= metrics["pr_auc"]["pr_auc_critical"] <= 1.0


def test_window_metrics_constant_probability_handling():
    """Verifies that constant predicted probabilities equal class prevalence."""
    y_true = np.array([0, 0, 0, 1])
    y_pred = np.array([0, 0, 0, 0])
    y_prob = np.array([[1.0, 0.0, 0.0]] * 4)

    metrics = compute_window_metrics(y_true, y_pred, y_prob)
    assert metrics["pr_auc"]["pr_auc_warning_is_prevalence"] is True
    assert np.isclose(metrics["pr_auc"]["pr_auc_warning"], 0.25)


def test_episode_metrics_computation():
    """Tests failure episode grouping, detection rate, and lead time logic."""
    meta_df = pd.DataFrame({
        "entity_id": ["E1", "E1", "E1", "E2", "E3", "E3"],
        "prediction_timestamp": [5100.0, 5400.0, 5700.0, 5100.0, 5100.0, 5400.0],
        "label": [1, 2, 2, 0, 1, 2],
        "time_to_failure": [900.0, 600.0, 300.0, np.nan, 1200.0, 900.0],
    })
    # Episode for E1: fail_time = 5100 + 900 = 6000.0
    # Episode for E3: fail_time = 5100 + 1200 = 6300.0
    # E2 is a pure normal task (never fails)

    # Prediction scenario:
    # E1: predicted [0, 2, 2] -> detected at step 2 (t=5400s), lead_time = 6000 - 5400 = 600s
    # E2: predicted [0] -> no false alarm
    # E3: predicted [0, 0] -> missed (detected = 0)
    y_pred = np.array([0, 2, 2, 0, 0, 0])

    ep_metrics = compute_episode_metrics(meta_df, y_pred)
    assert ep_metrics["total_episodes"] == 2
    assert ep_metrics["detected_episodes"] == 1
    assert np.isclose(ep_metrics["detection_rate_pct"], 50.0)
    assert np.isclose(ep_metrics["lead_time_mean_sec"], 600.0)
    assert ep_metrics["total_pure_normal_entities"] == 1
    assert ep_metrics["false_alarm_entities"] == 0
    assert ep_metrics["false_alarm_entity_rate_pct"] == 0.0


def test_baseline_models_fit_and_predict():
    """Verifies that all three baseline models fit and output valid probabilities."""
    X = np.random.randn(60, 15, 31).astype(np.float32)
    y = np.array([0] * 50 + [1] * 5 + [2] * 5)

    for ModelCls in [MajorityClassBaseline, ClassWeightedLogisticRegression, ClassWeightedRandomForest]:
        model = ModelCls(random_state=42)
        model.fit(X, y)
        preds = model.predict(X)
        probs = model.predict_proba(X)

        assert preds.shape == (60,)
        assert probs.shape == (60, 3)
        assert np.all(probs >= 0.0) and np.all(probs <= 1.0)
        np.testing.assert_allclose(probs.sum(axis=1), np.ones(60), atol=1e-5)


def test_prediction_artifacts_alignment():
    """Verifies that saved test predictions align index-for-index with test metadata."""
    pred_path = Path("results/phase3a/test_predictions_random_forest.csv")
    meta_path = Path("data/processed/test_metadata.csv")

    if pred_path.exists() and meta_path.exists():
        pred_df = pd.read_csv(pred_path)
        meta_df = pd.read_csv(meta_path)

        assert len(pred_df) == len(meta_df) == 1003
        np.testing.assert_array_equal(pred_df["entity_id"].values, meta_df["entity_id"].values)
        np.testing.assert_array_equal(pred_df["prediction_timestamp"].values, meta_df["prediction_timestamp"].values)
        np.testing.assert_array_equal(pred_df["label"].values, meta_df["label"].values)
        assert "predicted_label" in pred_df.columns
        assert "prob_normal" in pred_df.columns
        assert "prob_warning" in pred_df.columns
        assert "prob_critical" in pred_df.columns
