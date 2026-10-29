"""Unit tests for training-only standardization, median imputation, and scaler serialization."""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from src.features.scaler import LeakageSafeScaler


def test_scaler_fits_on_training_data_only(tmp_path):
    # Training set with mean 10.0, std 2.0
    train_df = pd.DataFrame({
        "feat1": [8.0, 10.0, 12.0],
        "feat2": [1.0, 2.0, 3.0],
        "constant_feat": [5.0, 5.0, 5.0], # Zero variance
    })

    # Validation set with vastly different distribution: mean 100.0
    val_df = pd.DataFrame({
        "feat1": [100.0, 110.0],
        "feat2": [20.0, 30.0],
        "constant_feat": [5.0, 5.0],
    })

    scaler = LeakageSafeScaler()

    # Calling transform before fit must raise error
    with pytest.raises(RuntimeError):
        scaler.transform(train_df)

    feature_cols = ["feat1", "feat2", "constant_feat"]
    scaler.fit(train_df, feature_cols)

    # 1. Assert training statistics learned
    assert np.isclose(scaler.means["feat1"], 10.0)
    assert np.isclose(scaler.stds["feat1"], np.std([8.0, 10.0, 12.0], ddof=0))
    # Zero variance feature should have std safe-handled to 1.0
    assert scaler.stds["constant_feat"] == 1.0

    # 2. Transform train
    scaled_train = scaler.transform(train_df)
    # Mean of scaled training data should be ~ 0.0
    assert np.isclose(scaled_train["feat1"].mean(), 0.0, atol=1e-5)

    # 3. Transform val using TRAINING statistics
    scaled_val = scaler.transform(val_df)
    # 100.0 transformed with train mean 10.0 and std ~1.63 should be ~ (100 - 10) / 1.63 ~ 55
    expected_val_0 = (100.0 - scaler.means["feat1"]) / (scaler.stds["feat1"] + 1e-6)
    assert np.isclose(scaled_val.loc[0, "feat1"], expected_val_0)

    # 4. Assert serialization roundtrip
    save_file = tmp_path / "scaler.json"
    scaler.save(save_file)
    loaded_scaler = LeakageSafeScaler.load(save_file)

    scaled_val_loaded = loaded_scaler.transform(val_df)
    assert np.allclose(scaled_val.values, scaled_val_loaded.values)


def test_training_median_imputation():
    # Training data with median = 5.0
    train_df = pd.DataFrame({
        "feat_miss": [1.0, 5.0, 9.0],
    })
    # Val data with NaN
    val_df = pd.DataFrame({
        "feat_miss": [np.nan, 20.0],
    })

    scaler = LeakageSafeScaler()
    scaler.fit(train_df, ["feat_miss"])
    assert np.isclose(scaler.medians["feat_miss"], 5.0)

    # Transformed NaN should equal transformed 5.0 (which is 0.0 since mean=5.0)
    scaled_val = scaler.transform(val_df)
    assert not np.isnan(scaled_val.loc[0, "feat_miss"])
    assert np.isclose(scaled_val.loc[0, "feat_miss"], 0.0, atol=1e-5)
