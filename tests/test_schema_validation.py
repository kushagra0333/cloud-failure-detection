"""Unit tests for schema validation, duplicate detection, and time-series profiling."""

import pytest
import numpy as np
import pandas as pd

from src.ingestion.schema import ClusterEventType, FailureLabel
from src.ingestion.validator import TimeSeriesValidator, SchemaValidationError


@pytest.fixture
def valid_telemetry_df():
    """Deterministic synthetic fixture for schema testing (labeled synthetic)."""
    return pd.DataFrame({
        "entity_id": ["task_1", "task_1", "task_1", "task_2", "task_2"],
        "timestamp": [300.0, 600.0, 900.0, 300.0, 600.0],
        "cpu_utilization": [0.45, 0.52, 0.48, 0.12, 0.15],
        "memory_utilization": [0.30, 0.32, 0.31, 0.08, 0.09],
        "assigned_memory": [1.0, 1.0, 1.0, 0.5, 0.5],
        "cycles_per_instruction": [1.2, 1.3, 1.1, 0.9, 0.95],
        "memory_accesses_per_instruction": [0.005, 0.006, 0.005, 0.002, 0.002],
        "disk_io_time": [0.01, 0.02, 0.01, 0.0, 0.0],
    })


@pytest.fixture
def valid_lifecycle_df():
    """Deterministic synthetic fixture for lifecycle events."""
    return pd.DataFrame({
        "entity_id": ["task_1", "task_1", "task_2"],
        "timestamp": [0.0, 1200.0, 0.0],
        "event_type": [
            int(ClusterEventType.SCHEDULE),
            int(ClusterEventType.FAIL),
            int(ClusterEventType.SCHEDULE),
        ],
    })


def test_valid_schema_passes(valid_telemetry_df, valid_lifecycle_df):
    validator = TimeSeriesValidator()
    # Should not raise
    validator.validate_schema(valid_telemetry_df, is_lifecycle=False)
    validator.validate_schema(valid_lifecycle_df, is_lifecycle=True)


def test_missing_column_raises_error(valid_telemetry_df):
    validator = TimeSeriesValidator()
    invalid_df = valid_telemetry_df.drop(columns=["cpu_utilization"])
    with pytest.raises(SchemaValidationError) as exc_info:
        validator.validate_schema(invalid_df, is_lifecycle=False)
    assert "cpu_utilization" in str(exc_info.value)


def test_invalid_dtype_raises_error(valid_telemetry_df):
    validator = TimeSeriesValidator()
    invalid_df = valid_telemetry_df.copy()
    invalid_df["timestamp"] = "not_a_numeric_timestamp"
    with pytest.raises(SchemaValidationError) as exc_info:
        validator.validate_schema(invalid_df, is_lifecycle=False)
    assert "Timestamp column must be numeric" in str(exc_info.value)


def test_duplicate_detection_and_resolution(valid_telemetry_df):
    validator = TimeSeriesValidator()
    # Introduce duplicate entity-timestamp record
    dup_row = pd.DataFrame({
        "entity_id": ["task_1"],
        "timestamp": [300.0],
        "cpu_utilization": [0.55],
        "memory_utilization": [0.34],
        "assigned_memory": [1.0],
        "cycles_per_instruction": [1.4],
        "memory_accesses_per_instruction": [0.006],
        "disk_io_time": [0.02],
    })
    dirty_df = pd.concat([valid_telemetry_df, dup_row], ignore_index=True)
    assert len(dirty_df) == 6

    resolved_df, num_dups = validator.detect_and_resolve_duplicates(dirty_df, resolution="mean")
    assert num_dups == 2  # The 2 records matching (task_1, 300.0)
    assert len(resolved_df) == 5
    # The mean of 0.45 and 0.55 is 0.50
    resolved_val = resolved_df.loc[
        (resolved_df["entity_id"] == "task_1") & (resolved_df["timestamp"] == 300.0),
        "cpu_utilization"
    ].iloc[0]
    assert np.isclose(resolved_val, 0.50)


def test_timestamp_gaps_and_out_of_order():
    validator = TimeSeriesValidator(expected_interval_sec=300.0)
    # Task with out of order and gap
    df = pd.DataFrame({
        "entity_id": ["task_gap", "task_gap", "task_gap"],
        "timestamp": [600.0, 300.0, 1500.0], # 600 -> 300 is out of order, 300 -> 1500 has a 1200s gap
        "cpu_utilization": [0.1, 0.2, 0.3],
        "memory_utilization": [0.1, 0.2, 0.3],
    })
    report = validator.validate_timestamps_and_gaps(df)
    assert report["total_entities"] == 1
    assert report["out_of_order_entities"] == 1
    assert report["gap_count"] >= 1  # 1200s is > 1.5 * 300s


def test_missing_value_analysis(valid_telemetry_df):
    validator = TimeSeriesValidator()
    df = valid_telemetry_df.copy()
    df.loc[0, "cpu_utilization"] = np.nan
    df.loc[1, "cpu_utilization"] = np.nan
    report = validator.analyze_missingness(df, features=["cpu_utilization", "memory_utilization"])
    # 2 out of 5 are missing for cpu
    assert np.isclose(report["cpu_utilization"], 2.0 / 5.0)
    assert np.isclose(report["memory_utilization"], 0.0)


def test_outlier_analysis():
    validator = TimeSeriesValidator(cpu_upper_bound=1.0)
    # 10 values around 0.5, with 1 extreme outlier at 2.5 (violates domain bound) and 1 Tukey outlier
    data = [0.48, 0.50, 0.51, 0.49, 0.52, 0.50, 0.48, 0.51, 0.49, 0.50, 2.50, -0.10]
    df = pd.DataFrame({
        "entity_id": ["task_x"] * len(data),
        "timestamp": np.arange(len(data)) * 300.0,
        "cpu_utilization": data,
        "memory_utilization": [0.3] * len(data),
    })
    report = validator.analyze_outliers(df, columns=["cpu_utilization"])
    cpu_rep = report["cpu_utilization"]
    assert cpu_rep["physical_negative_count"] == 1  # -0.10
    assert cpu_rep["physical_exceed_upper_count"] == 1  # 2.50 > 1.0
    assert cpu_rep["iqr_outlier_count"] >= 2
