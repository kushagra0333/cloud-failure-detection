"""Unit tests for sliding window sequence generation and chronological contiguity."""

import pytest
import numpy as np
import pandas as pd

from src.ingestion.schema import FailureLabel
from src.labeling.window_generator import WindowGenerator


@pytest.fixture
def sequence_telemetry_df():
    """Deterministic synthetic fixture for testing 15-step sequence generation."""
    # Entity 1: 20 contiguous steps at 300s cadence
    times1 = np.arange(20) * 300.0 + 600.0
    labels1 = [FailureLabel.NORMAL] * 15 + [FailureLabel.WARNING] * 3 + [FailureLabel.CRITICAL] * 2

    df1 = pd.DataFrame({
        "entity_id": ["task_1"] * 20,
        "start_time_sec": times1 - 300.0,
        "end_time_sec": times1,
        "cpu_utilization": np.linspace(0.1, 0.9, 20),
        "memory_utilization": np.linspace(0.2, 0.4, 20),
        "label": labels1,
        "time_to_failure_sec": [np.nan] * 15 + [1200.0, 900.0, 600.0, 300.0, 0.0],
    })

    # Entity 2: 10 steps only (less than L=15)
    times2 = np.arange(10) * 300.0 + 600.0
    df2 = pd.DataFrame({
        "entity_id": ["task_short"] * 10,
        "start_time_sec": times2 - 300.0,
        "end_time_sec": times2,
        "cpu_utilization": np.ones(10) * 0.5,
        "memory_utilization": np.ones(10) * 0.3,
        "label": [FailureLabel.NORMAL] * 10,
        "time_to_failure_sec": [np.nan] * 10,
    })

    # Entity 3: 20 steps with an irregular 1200s gap at index 10
    times3 = np.arange(20) * 300.0 + 600.0
    times3[10:] += 1200.0 # Introduce gap
    df3 = pd.DataFrame({
        "entity_id": ["task_gap"] * 20,
        "start_time_sec": times3 - 300.0,
        "end_time_sec": times3,
        "cpu_utilization": np.ones(20) * 0.2,
        "memory_utilization": np.ones(20) * 0.1,
        "label": [FailureLabel.NORMAL] * 20,
        "time_to_failure_sec": [np.nan] * 20,
    })

    return pd.concat([df1, df2, df3], ignore_index=True)


def test_window_generator_strict_history(sequence_telemetry_df):
    feature_cols = ["cpu_utilization", "memory_utilization"]
    generator = WindowGenerator(
        sequence_length=15,
        expected_interval_sec=300.0,
        require_full_history=True,
        exclude_unknown=True,
    )

    X, y, meta = generator.build_dataset_tensors(sequence_telemetry_df, feature_cols)

    # 1. Output shapes
    assert len(X.shape) == 3
    assert X.shape[1] == 15  # L = 15
    assert X.shape[2] == len(feature_cols)  # F = 2
    assert len(y) == len(X)
    assert len(meta) == len(X)

    # 2. Entity 2 (task_short with 10 steps) should produce 0 windows under require_full_history=True
    assert "task_short" not in meta["entity_id"].values

    # 3. Entity 1 has 20 steps. Eligible prediction indices are 14, 15, 16, 17, 18, 19 (6 windows)
    task1_meta = meta[meta["entity_id"] == "task_1"]
    assert len(task1_meta) == 6

    # 4. Check chronological ordering within windows
    for i in range(len(X)):
        # CPU utilization increases monotonically for task_1
        if meta.iloc[i]["entity_id"] == "task_1":
            cpu_seq = X[i, :, 0]
            assert np.all(np.diff(cpu_seq) > 0)

    # 5. Check target labels match prediction time
    # Last window of task_1 is at index 19 (CRITICAL)
    last_window_idx = task1_meta.index[-1]
    assert y[last_window_idx] == FailureLabel.CRITICAL
    assert np.isclose(meta.loc[last_window_idx, "prediction_timestamp"], 20 * 300.0 + 300.0)


def test_window_generator_padded_history(sequence_telemetry_df):
    feature_cols = ["cpu_utilization", "memory_utilization"]
    generator = WindowGenerator(
        sequence_length=15,
        require_full_history=False, # Allows left-padded windows
    )
    X, y, meta = generator.build_dataset_tensors(sequence_telemetry_df, feature_cols)

    # Now task_short should produce windows
    assert "task_short" in meta["entity_id"].values
    short_meta = meta[meta["entity_id"] == "task_short"]
    assert len(short_meta) == 10
