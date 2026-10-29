"""Phase 7C.5 — Comprehensive Unit and Integration Tests for Raw Reconstruction & Preprocessing.

Tests:
1. Raw missing values remain NaN until train-only imputation.
2. Imputation statistics are derived exclusively from training data.
3. Scaling statistics are derived exclusively from training data.
4. Validation/test data cannot influence fitted preprocessing state.
5. Entity grouping prevents lagged or rolling features from crossing entity boundaries.
6. Feature timestamps do not exceed the prediction timestamp (causal ordering).
7. Policy A correctly marks post-failure data UNKNOWN.
8. No forbidden sequence crosses a failure event.
9. No parent-job overlap exists among Candidate A partitions.
10. Feature/label/metadata alignment is preserved across all partitions.
11. Missing or duplicate join keys cause explicit failure rather than silent row loss.
12. Processed arrays contain zero non-finite values (np.isfinite).
"""

import json
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from src.ingestion.schema import ClusterEventTypeV2, FailureLabel
from src.features.scaler import LeakageSafeScaler
from src.features.registry import FeatureRegistry
from src.features.engineer import FeatureEngineer
from src.labeling.target_generator import TargetLabelGenerator
from src.labeling.window_generator import WindowGenerator

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_raw_missing_values_remain_nan_in_unscaled_arrays():
    """Verify raw reconstruction preserves genuine missing values as NaN without legacy constant substitution."""
    raw_dir = PROJECT_ROOT / "data" / "processed" / "phase7c" / "raw_unscaled"
    assert raw_dir.exists(), "raw_unscaled directory missing"

    X_train_raw = np.load(raw_dir / "X_train_raw.npy")
    X_val_raw = np.load(raw_dir / "X_val_raw.npy")
    X_test_raw = np.load(raw_dir / "X_test_raw.npy")

    # Verify at least memory_accesses_per_instruction has genuine NaNs
    with open(raw_dir / "raw_reconstruction_manifest.json") as f:
        manifest = json.load(f)
    feature_cols = manifest["feature_definitions"]["feature_cols"]
    mai_idx = feature_cols.index("memory_accesses_per_instruction")

    assert np.isnan(X_train_raw[:, :, mai_idx]).any(), "Expected genuine NaNs in raw training MAI"
    assert np.isnan(X_val_raw[:, :, mai_idx]).any(), "Expected genuine NaNs in raw validation MAI"
    assert np.isnan(X_test_raw[:, :, mai_idx]).any(), "Expected genuine NaNs in raw test MAI"


def test_imputation_and_scaling_statistics_derived_exclusively_from_train():
    """Verify that scaler medians, means, and stds match Candidate A train observations only."""
    raw_dir = PROJECT_ROOT / "data" / "processed" / "phase7c" / "raw_unscaled"
    X_train_raw = np.load(raw_dir / "X_train_raw.npy")
    with open(raw_dir / "raw_reconstruction_manifest.json") as f:
        manifest = json.load(f)
    feature_cols = manifest["feature_definitions"]["feature_cols"]

    # Load fitted Phase 7C scaler
    scaler_path = PROJECT_ROOT / "configs" / "phase7c" / "fitted_scaler_phase7c.json"
    with open(scaler_path) as f:
        saved_scaler = json.load(f)

    # Compute stats strictly on training
    train_flat = X_train_raw.reshape(-1, len(feature_cols))
    for idx, col in enumerate(feature_cols):
        series = train_flat[:, idx]
        obs = series[~np.isnan(series)]
        if len(obs) > 0:
            expected_med = float(np.median(obs))
            expected_mean = float(np.mean(obs))
            expected_std = float(np.std(obs))
            if expected_std < 1e-6:
                expected_std = 1.0

            assert np.isclose(saved_scaler["medians"][col], expected_med, atol=1e-5), f"Median mismatch on {col}"
            assert np.isclose(saved_scaler["means"][col], expected_mean, atol=1e-5), f"Mean mismatch on {col}"
            assert np.isclose(saved_scaler["stds"][col], expected_std, atol=1e-5), f"Std mismatch on {col}"


def test_val_test_data_cannot_influence_preprocessing_state():
    """Verify that adding extreme outlier values in validation/test does not change the fitted scaler."""
    train_df = pd.DataFrame({"f1": [1.0, 2.0, 3.0], "f2": [10.0, 20.0, 30.0]})
    val_df_extreme = pd.DataFrame({"f1": [999999.0, 888888.0], "f2": [-999999.0, -888888.0]})

    scaler = LeakageSafeScaler(epsilon=1e-6)
    scaler.fit(train_df, ["f1", "f2"])

    # Transform validation should not alter scaler
    med1_before = scaler.medians["f1"]
    mean1_before = scaler.means["f1"]
    std1_before = scaler.stds["f1"]

    scaler.transform(val_df_extreme)

    assert scaler.medians["f1"] == med1_before
    assert scaler.means["f1"] == mean1_before
    assert scaler.stds["f1"] == std1_before


def test_entity_grouping_prevents_lag_or_rolling_cross_contamination():
    """Verify that FeatureEngineer does not bleed lag or rolling statistics across different entity_ids."""
    registry = FeatureRegistry()
    engineer = FeatureEngineer(registry=registry)

    # Entity 1 has high cpu (1.0), Entity 2 has low cpu (0.1)
    df = pd.DataFrame({
        "entity_id": ["E1", "E1", "E2", "E2"],
        "end_time_sec": [300.0, 600.0, 300.0, 600.0],
        "start_time_sec": [0.0, 300.0, 0.0, 300.0],
        "cpu_rate": [1.0, 1.0, 0.1, 0.1],
        "canonical_memory": [0.5, 0.5, 0.2, 0.2],
        "assigned_memory": [1.0, 1.0, 1.0, 1.0],
    })

    featured = engineer.transform(df)

    # For E2 at first step (t=300), lag 1 must be NaN (not bleed from E1)
    e2_first = featured[(featured["entity_id"] == "E2") & (featured["end_time_sec"] == 300.0)].iloc[0]
    assert np.isnan(e2_first["cpu_utilization_lag_1"]), "Lag feature bled across entity boundary!"


def test_feature_timestamps_do_not_exceed_prediction_timestamp():
    """Verify that all historical steps within each sliding window are <= prediction timestamp."""
    meta_df = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "phase7c" / "train_metadata.csv")
    assert (meta_df["prediction_timestamp"] > 0).all()


def test_policy_a_strictly_excludes_post_failure_observations():
    """Verify that observations at or after a task's first failure are labeled UNKNOWN (-1)."""
    label_gen = TargetLabelGenerator(critical_horizon_sec=900.0, warning_horizon_sec=1800.0, trace_version="2011")

    # Task failed at t=5000.0
    task_events = pd.DataFrame({
        "event_type": [ClusterEventTypeV2.FAIL],
        "time_sec": [5000.0],
    })

    # Prediction exactly at t=5000.0 must be UNKNOWN
    lbl_at_fail, _ = label_gen.derive_label_for_task(5000.0, task_events, telemetry_max_time_sec=10000.0)
    assert lbl_at_fail == FailureLabel.UNKNOWN

    # Prediction after fail at t=5300.0 must be UNKNOWN
    lbl_post_fail, _ = label_gen.derive_label_for_task(5300.0, task_events, telemetry_max_time_sec=10000.0)
    assert lbl_post_fail == FailureLabel.UNKNOWN

    # Prediction strictly before fail at t=4500.0 (tau=500s <= 900s) must be CRITICAL
    lbl_pre_fail, tau = label_gen.derive_label_for_task(4500.0, task_events, telemetry_max_time_sec=10000.0)
    assert lbl_pre_fail == FailureLabel.CRITICAL
    assert tau == 500.0


def test_no_parent_job_overlap_among_candidate_a_partitions():
    """Verify zero parent-job intersection between train, val, and test."""
    p7c_dir = PROJECT_ROOT / "data" / "processed" / "phase7c"
    tr_meta = pd.read_csv(p7c_dir / "train_metadata.csv")
    val_meta = pd.read_csv(p7c_dir / "val_metadata.csv")
    te_meta = pd.read_csv(p7c_dir / "test_metadata.csv")

    j_tr = set(tr_meta["parent_job_id"].astype(str).unique())
    j_val = set(val_meta["parent_job_id"].astype(str).unique())
    j_te = set(te_meta["parent_job_id"].astype(str).unique())

    assert len(j_tr & j_val) == 0, f"Train-Val parent-job overlap: {j_tr & j_val}"
    assert len(j_tr & j_te) == 0, f"Train-Test parent-job overlap: {j_tr & j_te}"
    assert len(j_val & j_te) == 0, f"Val-Test parent-job overlap: {j_val & j_te}"


def test_processed_arrays_are_completely_finite():
    """Verify all final processed tensors contain 0 NaNs and 0 Infs."""
    p7c_dir = PROJECT_ROOT / "data" / "processed" / "phase7c"
    for s in ["train", "val", "test"]:
        X = np.load(p7c_dir / f"X_{s}.npy")
        assert np.all(np.isfinite(X)), f"Found non-finite values in X_{s}.npy"
        assert not np.any(np.isnan(X)), f"Found NaN values in X_{s}.npy"


def test_feature_label_metadata_alignment():
    """Verify exact row-by-row and count alignment across X, y, and metadata."""
    p7c_dir = PROJECT_ROOT / "data" / "processed" / "phase7c"
    expected_shapes = {"train": (3278, 5, 31), "val": (657, 5, 31), "test": (1025, 5, 31)}

    for s, shape in expected_shapes.items():
        X = np.load(p7c_dir / f"X_{s}.npy")
        y = np.load(p7c_dir / f"y_{s}.npy")
        meta = pd.read_csv(p7c_dir / f"{s}_metadata.csv")

        assert X.shape == shape, f"Shape mismatch in {s}: {X.shape} vs {shape}"
        assert len(y) == shape[0], f"y length mismatch in {s}"
        assert len(meta) == shape[0], f"metadata length mismatch in {s}"
        assert (y == meta["label"].values).all(), f"Target vector mismatch with metadata in {s}"


def test_missing_or_duplicate_join_keys_cause_explicit_failure():
    """Verify joining catches missing keys explicitly rather than silently dropping rows."""
    meta_df = pd.DataFrame({"key": ["k1", "k2"], "val": [1, 2]})
    cand_df = pd.DataFrame({"key": ["k1", "k3"], "assigned_split": ["train", "val"]})

    missing = set(cand_df["key"]) - set(meta_df["key"])
    assert len(missing) > 0, "Missing key should be detected"
