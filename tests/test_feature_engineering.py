"""Unit tests for feature engineering across Groups A through I."""

import pytest
import numpy as np
import pandas as pd

from src.features.registry import FeatureRegistry
from src.features.engineer import FeatureEngineer


@pytest.fixture
def synthetic_telemetry_df():
    """Deterministic synthetic fixture for unit testing feature transformations."""
    # 2 entities: entity_1 has 10 steps, entity_2 has 5 steps
    e1_times = np.arange(10) * 300.0 + 600.0
    e2_times = np.arange(5) * 300.0 + 600.0

    df1 = pd.DataFrame({
        "entity_id": ["task_1"] * 10,
        "start_time_sec": e1_times - 300.0,
        "end_time_sec": e1_times,
        "cpu_rate": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95],
        "canonical_memory": [0.2, 0.22, 0.24, 0.26, 0.28, 0.30, 0.32, 0.34, 0.36, 0.38],
        "assigned_memory": [0.5] * 10,
        "disk_io_time": [0.01 * i for i in range(10)],
        "cycles_per_instruction": [1.0, 1.1, np.nan, 1.2, 1.3, np.nan, 1.4, 1.5, 1.6, 1.7],
        "memory_accesses_per_instruction": [0.005] * 10,
    })

    df2 = pd.DataFrame({
        "entity_id": ["task_2"] * 5,
        "start_time_sec": e2_times - 300.0,
        "end_time_sec": e2_times,
        "cpu_rate": [0.8, 0.7, 0.6, 0.5, 0.4],
        "canonical_memory": [0.4, 0.38, 0.36, 0.34, 0.32],
        "assigned_memory": [0.8] * 5,
        "disk_io_time": [0.05] * 5,
        "cycles_per_instruction": [2.0, 2.1, 2.2, 2.3, 2.4],
        "memory_accesses_per_instruction": [0.01] * 5,
    })

    return pd.concat([df1, df2], ignore_index=True)


def test_registry_has_all_groups():
    registry = FeatureRegistry()
    features = registry.list_features()
    assert len(features) >= 20
    groups = {f.group for f in features}
    assert any("Group A" in g for g in groups)
    assert any("Group B" in g for g in groups)
    assert any("Group C" in g for g in groups)
    assert any("Group D" in g for g in groups)
    assert any("Group E" in g for g in groups)
    assert any("Group F" in g for g in groups)
    assert any("Group G" in g for g in groups)
    assert any("Group H" in g for g in groups)
    assert any("Group I" in g for g in groups)


def test_feature_engineering_transforms_correctly(synthetic_telemetry_df):
    engineer = FeatureEngineer()
    res = engineer.transform(synthetic_telemetry_df)

    # 1. Output length matches input
    assert len(res) == len(synthetic_telemetry_df)

    # 2. Check no cross-entity lag leakage
    # First row of task_2 should have NaN for lag_1
    task2_first_row = res[res["entity_id"] == "task_2"].iloc[0]
    assert np.isnan(task2_first_row["cpu_utilization_lag_1"])
    assert np.isnan(task2_first_row["cpu_diff_1"])

    # 3. Check exact lag calculation for task_1
    task1_rows = res[res["entity_id"] == "task_1"].reset_index(drop=True)
    # At step index 1 (cpu=0.2), lag_1 should be 0.1
    assert np.isclose(task1_rows.loc[1, "cpu_utilization_lag_1"], 0.1)
    assert np.isclose(task1_rows.loc[2, "cpu_utilization_lag_2"], 0.1)
    assert np.isclose(task1_rows.loc[1, "cpu_diff_1"], 0.1)

    # 4. Check rolling mean (w=3 at index 2 should be mean(0.1, 0.2, 0.3) = 0.2)
    assert np.isclose(task1_rows.loc[2, "cpu_rolling_mean_3"], 0.2)

    # 5. Check threshold exceedance (> 0.85)
    # cpu at index 8 is 0.9 (> 0.85) -> exceed=1.0
    assert task1_rows.loc[8, "cpu_exceed_85"] == 1.0
    # cpu at index 7 is 0.8 (<= 0.85) -> exceed=0.0
    assert task1_rows.loc[7, "cpu_exceed_85"] == 0.0

    # 6. Check allocation ratio
    # index 0: mem=0.2 / assigned=0.5 = 0.4
    assert np.isclose(task1_rows.loc[0, "memory_to_assigned_ratio"], 0.4, atol=1e-3)

    # 7. Check task age (minutes)
    assert np.isclose(task1_rows.loc[0, "task_age_minutes"], 5.0)
    assert np.isclose(task1_rows.loc[9, "task_age_minutes"], 50.0)
