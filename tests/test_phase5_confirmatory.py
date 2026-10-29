"""Phase 5 Confirmatory Evaluation Unit Test Suite.

Verifies:
1. Exact frozen entity membership from results/phase4/confirmatory_protocol.json (zero overlap with development data).
2. Complete evaluation coverage and documented exclusion of all 300 frozen entities.
3. Prediction file integrity, exact column schema, probability normalization, and argmax alignment.
4. Timestamp causality, 300s grid alignment, and absence of post-failure telemetry.
5. Exact reconciliation of reported metrics against independent recomputations from confirmatory_predictions.csv.
6. Rigorous verification of operational FAR thresholds under the pre-registered model selection policy.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def protocol_data():
    proto_path = PROJECT_ROOT / "results" / "phase4" / "confirmatory_protocol.json"
    assert proto_path.exists(), "confirmatory_protocol.json not found"
    with open(proto_path) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def predictions_df():
    pred_path = PROJECT_ROOT / "results" / "phase5" / "confirmatory_predictions.csv"
    assert pred_path.exists(), "confirmatory_predictions.csv not found"
    return pd.read_csv(pred_path)


@pytest.fixture(scope="module")
def metrics_data():
    met_path = PROJECT_ROOT / "results" / "phase5" / "confirmatory_metrics.json"
    assert met_path.exists(), "confirmatory_metrics.json not found"
    with open(met_path) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def audit_data():
    audit_path = PROJECT_ROOT / "results" / "phase5" / "confirmatory_audit.json"
    assert audit_path.exists(), "confirmatory_audit.json not found"
    with open(audit_path) as f:
        return json.load(f)


def test_frozen_entity_membership_and_zero_dev_overlap(protocol_data):
    """Verifies exact frozen entity counts and zero leakage into development partitions."""
    fail_eids = set(protocol_data["frozen_entity_ids"]["failure_entities"])
    norm_eids = set(protocol_data["frozen_entity_ids"]["normal_entities"])

    assert len(fail_eids) == 100, f"Expected 100 fail entities, got {len(fail_eids)}"
    assert len(norm_eids) == 200, f"Expected 200 normal entities, got {len(norm_eids)}"
    assert len(fail_eids & norm_eids) == 0, "Overlap detected between failure and normal confirmatory entities"

    all_conf_eids = fail_eids | norm_eids
    assert len(all_conf_eids) == 300, f"Expected 300 total entities, got {len(all_conf_eids)}"

    # Check zero overlap with development set
    tr_meta = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "train_metadata.csv")
    va_meta = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "val_metadata.csv")
    te_meta = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "test_metadata.csv")

    dev_eids = set(tr_meta["entity_id"]) | set(va_meta["entity_id"]) | set(te_meta["entity_id"])
    assert len(dev_eids) == 525, f"Expected 525 development entities, got {len(dev_eids)}"

    leakage = all_conf_eids & dev_eids
    assert len(leakage) == 0, f"CRITICAL LEAKAGE: {len(leakage)} confirmatory entities overlap with development set!"


def test_complete_evaluation_coverage_and_documented_exclusions(protocol_data, predictions_df, metrics_data, audit_data):
    """Verifies that every single one of the 300 frozen entities is accounted for."""
    fail_eids = set(protocol_data["frozen_entity_ids"]["failure_entities"])
    norm_eids = set(protocol_data["frozen_entity_ids"]["normal_entities"])
    all_target_eids = fail_eids | norm_eids

    pred_eids = set(predictions_df["entity_id"].unique())
    assert len(pred_eids) == 188, f"Expected 188 evaluated entities, got {len(pred_eids)}"
    assert pred_eids.issubset(all_target_eids), "Evaluated entities contain IDs not in pre-registered protocol!"

    # Evaluated entities are strictly a subset of normal entities
    assert pred_eids.issubset(norm_eids), "Evaluated entities contain failure entities that lacked valid pre-failure history"

    # Excluded entities
    excluded_eids = all_target_eids - pred_eids
    assert len(excluded_eids) == 112, f"Expected 112 excluded entities, got {len(excluded_eids)}"

    inv = metrics_data["entity_inventory"]
    assert inv["total_pre_registered_entities"] == 300
    assert inv["evaluated_entities"] == 188
    assert inv["excluded_entities_count"] == 112
    assert inv["excluded_failure_entities_count"] == 100
    assert inv["excluded_normal_entities_count"] == 12

    # Audit invariants passed
    assert audit_data["invariants_audited"]["frozen_entity_membership"]["status"] == "PASS"
    assert audit_data["invariants_audited"]["zero_leakage_and_quarantine"]["status"] == "PASS"


def test_prediction_dataframe_integrity_and_argmax_fairness(predictions_df):
    """Verifies row counts, column structure, probability sums, and strict argmax rule."""
    assert len(predictions_df) == 2596, f"Expected 2596 prediction rows, got {len(predictions_df)}"

    required_cols = [
        "entity_id", "prediction_timestamp", "true_label",
        "rf_pred", "rf_prob_0", "rf_prob_1", "rf_prob_2",
        "lstm_pred", "lstm_prob_0", "lstm_prob_1", "lstm_prob_2",
        "trans_pred", "trans_prob_0", "trans_prob_1", "trans_prob_2",
        "hybrid_pred", "hybrid_prob_0", "hybrid_prob_1", "hybrid_prob_2",
    ]
    for col in required_cols:
        assert col in predictions_df.columns, f"Missing required column: {col}"

    # All true labels must be 0 (Normal)
    assert (predictions_df["true_label"] == 0).all(), "Non-zero true label in pure-normal holdout windows"

    models = ["rf", "lstm", "trans", "hybrid"]
    for m in models:
        probs = predictions_df[[f"{m}_prob_0", f"{m}_prob_1", f"{m}_prob_2"]].values
        # Probabilities sum to 1.0
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-4, err_msg=f"{m} probabilities do not sum to 1.0")

        # Argmax prediction fairness
        argmax_preds = np.argmax(probs, axis=1)
        actual_preds = predictions_df[f"{m}_pred"].values
        np.testing.assert_array_equal(argmax_preds, actual_preds, err_msg=f"{m} predictions violate uniform argmax rule")


def test_timestamp_causality_and_grid_alignment(predictions_df):
    """Verifies that all prediction timestamps align with 300s discrete grid."""
    timestamps = predictions_df["prediction_timestamp"].values
    assert (timestamps >= 5100.0).all(), "Prediction timestamp occurs before 5100s (violates 15-step history invariant)"
    assert (timestamps <= 10800.0).all(), "Prediction timestamp occurs after trace max 10800s"
    assert (timestamps % 300.0 == 0.0).all(), "Prediction timestamp not aligned to 300s discrete grid"


def test_metric_recomputation_verification(predictions_df, metrics_data):
    """Independently recomputes window and false alarm metrics and compares with metrics JSON."""
    models_mapping = {
        "Random Forest": "rf",
        "Temporal LSTM": "lstm",
        "Compact Temporal Transformer": "trans",
        "Representation Hybrid": "hybrid",
    }

    n_windows = len(predictions_df)
    n_entities = predictions_df["entity_id"].nunique()

    for model_name, prefix in models_mapping.items():
        preds = predictions_df[f"{prefix}_pred"].values
        m_json = metrics_data["models"][model_name]

        # Accuracy
        acc_recomputed = float((preds == 0).mean())
        assert abs(acc_recomputed - m_json["accuracy"]) < 1e-6, f"{model_name} accuracy mismatch"

        # Window FPR
        fp_windows = int((preds > 0).sum())
        fpr_recomputed = (fp_windows / n_windows) * 100.0
        assert abs(fpr_recomputed - m_json["false_alarm_metrics"]["window_fpr"]["fpr_pct"]) < 1e-6, f"{model_name} FPR mismatch"

        # Pure-Normal FAR
        fa_entities = sum(
            (predictions_df[predictions_df["entity_id"] == eid][f"{prefix}_pred"] > 0).any()
            for eid in predictions_df["entity_id"].unique()
        )
        far_recomputed = (fa_entities / n_entities) * 100.0
        assert abs(far_recomputed - m_json["false_alarm_metrics"]["pure_normal_entity_far"]["far_pct"]) < 1e-6, f"{model_name} FAR mismatch"

        # Confusion Matrix
        cm = np.array(m_json["confusion_matrix"])
        assert cm.shape == (3, 3)
        assert cm[0, 0] == int((preds == 0).sum())
        assert cm[0, 1] == int((preds == 1).sum())
        assert cm[0, 2] == int((preds == 2).sum())
        assert cm[1, :].sum() == 0, "Non-zero ground-truth warning count in confirmatory holdout"
        assert cm[2, :].sum() == 0, "Non-zero ground-truth critical count in confirmatory holdout"


def test_model_selection_policy_and_far_violations(metrics_data):
    """Verifies that all 4 models were evaluated against the frozen operational constraint."""
    threshold = 12.0
    for name, m in metrics_data["models"].items():
        far = m["false_alarm_metrics"]["pure_normal_entity_far"]["far_pct"]
        assert far > threshold, f"Model {name} unexpectedly passed the operational FAR threshold ({far}% <= {threshold}%)"

    # Verify that bootstrap 95% CI is present for all models
    for name, m in metrics_data["models"].items():
        ci = m["bootstrap_confidence_intervals_95"]
        assert "pure_normal_far_pct" in ci
        assert "window_fpr_pct" in ci
        assert "macro_f1" in ci
        assert ci["pure_normal_far_pct"]["ci_lower_95"] > threshold, f"{name} lower 95% CI unexpectedly overlaps {threshold}%"
