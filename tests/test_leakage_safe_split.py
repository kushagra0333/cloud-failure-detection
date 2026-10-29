"""Unit tests for leakage-safe data splitting, horizon embargoes, and audit checks."""

import pytest
import numpy as np
import pandas as pd

from src.splitting.leakage_safe_split import (
    LeakageSafeSplitter,
    LeakageViolationError,
)


@pytest.fixture
def sample_metadata_df():
    """Deterministic synthetic metadata spanning 4 hours (14,400 seconds)."""
    # 100 examples spaced across 4 hours
    times = np.linspace(1000.0, 15400.0, 100)
    entities = [f"task_{i % 10}" for i in range(100)]
    return pd.DataFrame({
        "entity_id": entities,
        "prediction_timestamp": times,
        "label": [0] * 80 + [1] * 10 + [2] * 10,
        "time_to_failure": [np.nan] * 80 + [1200.0] * 10 + [600.0] * 10,
    })


def test_chronological_split_with_embargo(sample_metadata_df):
    embargo_sec = 1800.0 # 30 min
    splitter = LeakageSafeSplitter(
        strategy="chronological",
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        embargo_sec=embargo_sec,
    )
    res = splitter.split(sample_metadata_df)
    train_idx = res["train_idx"]
    val_idx = res["val_idx"]
    test_idx = res["test_idx"]

    # 1. Non-empty splits
    assert len(train_idx) > 0
    assert len(val_idx) > 0
    assert len(test_idx) > 0

    # 2. Embargo purged windows exist
    assert res["purged_train_embargo_count"] > 0

    # 3. Disjoint indices
    assert len(set(train_idx) & set(val_idx)) == 0
    assert len(set(train_idx) & set(test_idx)) == 0
    assert len(set(val_idx) & set(test_idx)) == 0

    # 4. Temporal ordering with embargo
    train_times = sample_metadata_df.iloc[train_idx]["prediction_timestamp"]
    val_times = sample_metadata_df.iloc[val_idx]["prediction_timestamp"]
    test_times = sample_metadata_df.iloc[test_idx]["prediction_timestamp"]

    max_train_time = train_times.max()
    min_val_time = val_times.min()
    assert (min_val_time - max_train_time) >= embargo_sec

    max_val_time = val_times.max()
    min_test_time = test_times.min()
    assert min_test_time > max_val_time

    # 5. Independent leakage audit passes
    audit_rep = splitter.audit_leakage(sample_metadata_df, res, embargo_sec=embargo_sec)
    assert not audit_rep["temporal_leakage_detected"]
    assert audit_rep["embargo_respected"]


def test_entity_grouped_split(sample_metadata_df):
    splitter = LeakageSafeSplitter(
        strategy="entity_grouped",
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        random_seed=42,
    )
    res = splitter.split(sample_metadata_df)

    train_entities = set(sample_metadata_df.iloc[res["train_idx"]]["entity_id"])
    val_entities = set(sample_metadata_df.iloc[res["val_idx"]]["entity_id"])
    test_entities = set(sample_metadata_df.iloc[res["test_idx"]]["entity_id"])

    assert len(train_entities & val_entities) == 0
    assert len(train_entities & test_entities) == 0
    assert len(val_entities & test_entities) == 0


def test_leakage_audit_catches_synthetic_contamination(sample_metadata_df):
    splitter = LeakageSafeSplitter(strategy="chronological", embargo_sec=1800.0)
    res = splitter.split(sample_metadata_df)

    # Intentionally contaminate train split with a test window
    contaminated_res = dict(res)
    contaminated_res["train_idx"] = np.append(res["train_idx"], res["test_idx"][0])

    with pytest.raises(Exception):
        splitter.audit_leakage(sample_metadata_df, contaminated_res, embargo_sec=1800.0)
