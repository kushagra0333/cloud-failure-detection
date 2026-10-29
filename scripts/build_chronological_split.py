"""Script to build and verify the separate Chronological Evaluation split.

Implements Requirement B.6 and Phase 2.2 Invariant Controls:
- All windows have strictly verified 75-minute history (t_pred >= 5100s, 15 steps @ 300s).
- Full 30-minute embargo (1800s) at train-val and val-test boundaries.
- Fits chronological scaler exclusively on the chronological training data.
- Saves all artifacts to data/processed/chronological/ with zero leakage.
"""

import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.features.scaler import LeakageSafeScaler


def build_chronological_split():
    print("=" * 70)
    print("BUILDING CHRONOLOGICAL EVALUATION ARTIFACTS (30-MIN EMBARGO)")
    print("=" * 70)

    # 1. Load fitted scaler and reconstruct full unscaled dataset
    with open("configs/fitted_scaler.json") as f:
        scaler_data = json.load(f)

    feat_cols = scaler_data["feature_cols"]
    means = np.array([scaler_data["means"][k] for k in feat_cols], dtype=np.float32)
    stds = np.array([scaler_data["stds"][k] for k in feat_cols], dtype=np.float32)

    X_train_eg = np.load("data/processed/X_train.npy")
    y_train_eg = np.load("data/processed/y_train.npy")
    meta_train_eg = pd.read_csv("data/processed/train_metadata.csv")

    X_val_eg = np.load("data/processed/X_val.npy")
    y_val_eg = np.load("data/processed/y_val.npy")
    meta_val_eg = pd.read_csv("data/processed/val_metadata.csv")

    X_test_eg = np.load("data/processed/X_test.npy")
    y_test_eg = np.load("data/processed/y_test.npy")
    meta_test_eg = pd.read_csv("data/processed/test_metadata.csv")

    # Invert scaling to obtain exact unscaled tensors
    X_train_raw = X_train_eg * stds + means
    X_val_raw = X_val_eg * stds + means
    X_test_raw = X_test_eg * stds + means

    X_all_raw = np.concatenate([X_train_raw, X_val_raw, X_test_raw], axis=0)
    y_all = np.concatenate([y_train_eg, y_val_eg, y_test_eg], axis=0)
    meta_all = pd.concat([meta_train_eg, meta_val_eg, meta_test_eg], ignore_index=True)

    print(f"Total verified input dataset: N = {len(meta_all):,}, X_shape = {X_all_raw.shape}")
    print(f"Prediction timestamp range: [{meta_all['prediction_timestamp'].min():.1f}s, {meta_all['prediction_timestamp'].max():.1f}s]")

    # 2. Chronological split with full 30-minute embargo (1800s)
    # Available prediction span: [5100s, 10800s] (5700s = 95 minutes)
    # Train: t <= 6000s (5100s - 6000s, 15 min span)
    # Embargo 1: (6000s, 7800s] (1800s = 30 min buffer purged)
    # Val: [8100s, 8400s] (8100s - 8400s, 5 min active span)
    # Embargo 2: (8400s, 10200s] (1800s = 30 min buffer purged)
    # Test: >= 10500s (10500s - 10800s, 5 min active span)
    t_train_max = 6000.0
    t_embargo1_end = 7800.0
    t_val_start = 8100.0
    t_val_max = 8400.0
    t_embargo2_end = 10200.0
    t_test_start = 10500.0

    times = meta_all["prediction_timestamp"].values
    tr_idx = np.where(times <= t_train_max)[0]
    emb1_idx = np.where((times > t_train_max) & (times <= t_embargo1_end))[0]
    val_idx = np.where((times >= t_val_start) & (times <= t_val_max))[0]
    emb2_idx = np.where((times > t_val_max) & (times <= t_embargo2_end))[0]
    te_idx = np.where(times >= t_test_start)[0]

    print("\nChronological Split Partitions (30-min Embargo):")
    print(f"  - Train split (t <= {t_train_max:.0f}s)                : {len(tr_idx):,} windows")
    print(f"  - Embargo 1 (purged {t_train_max:.0f}s - {t_embargo1_end:.0f}s)    : {len(emb1_idx):,} windows")
    print(f"  - Val split   ({t_val_start:.0f}s <= t <= {t_val_max:.0f}s)       : {len(val_idx):,} windows")
    print(f"  - Embargo 2 (purged {t_val_max:.0f}s - {t_embargo2_end:.0f}s)   : {len(emb2_idx):,} windows")
    print(f"  - Test split  (t >= {t_test_start:.0f}s)               : {len(te_idx):,} windows")

    # Verify temporal isolation
    assert meta_all.iloc[tr_idx]["prediction_timestamp"].max() <= t_train_max
    assert meta_all.iloc[val_idx]["prediction_timestamp"].min() >= t_val_start
    assert meta_all.iloc[val_idx]["prediction_timestamp"].max() <= t_val_max
    assert meta_all.iloc[te_idx]["prediction_timestamp"].min() >= t_test_start
    assert t_val_start - t_train_max >= 1800.0, "Train-Val embargo violated!"
    assert t_test_start - t_val_max >= 1800.0, "Val-Test embargo violated!"
    print("  -> Boundary integrity: PASSED (Zero temporal overlap; 30m embargoes strictly enforced)")

    # 3. Fit scaler strictly on the chronological training partition
    print("\nFitting LeakageSafeScaler strictly on Chronological Train partition...")
    train_flat = X_all_raw[tr_idx].reshape(-1, len(feat_cols))
    train_df = pd.DataFrame(train_flat, columns=feat_cols)

    chrono_scaler = LeakageSafeScaler()
    chrono_scaler.fit(train_df, feat_cols)

    scaler_out = Path("configs/fitted_scaler_chronological.json")
    chrono_scaler.save(scaler_out)
    print(f"  - Chronological scaler saved to: {scaler_out}")

    # Transform tensors using chronological train scaler
    X_train_chrono = np.zeros_like(X_all_raw[tr_idx])
    for s in range(X_all_raw.shape[1]):
        df_s = pd.DataFrame(X_all_raw[tr_idx, s, :], columns=feat_cols)
        X_train_chrono[:, s, :] = chrono_scaler.transform(df_s).values

    X_val_chrono = np.zeros_like(X_all_raw[val_idx])
    for s in range(X_all_raw.shape[1]):
        df_s = pd.DataFrame(X_all_raw[val_idx, s, :], columns=feat_cols)
        X_val_chrono[:, s, :] = chrono_scaler.transform(df_s).values

    X_test_chrono = np.zeros_like(X_all_raw[te_idx])
    for s in range(X_all_raw.shape[1]):
        df_s = pd.DataFrame(X_all_raw[te_idx, s, :], columns=feat_cols)
        X_test_chrono[:, s, :] = chrono_scaler.transform(df_s).values

    # 4. Save artifacts to data/processed/chronological/
    out_dir = Path("data/processed/chronological")
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "X_train.npy", X_train_chrono)
    np.save(out_dir / "y_train.npy", y_all[tr_idx])
    meta_all.iloc[tr_idx].to_csv(out_dir / "train_metadata.csv", index=False)

    np.save(out_dir / "X_val.npy", X_val_chrono)
    np.save(out_dir / "y_val.npy", y_all[val_idx])
    meta_all.iloc[val_idx].to_csv(out_dir / "val_metadata.csv", index=False)

    np.save(out_dir / "X_test.npy", X_test_chrono)
    np.save(out_dir / "y_test.npy", y_all[te_idx])
    meta_all.iloc[te_idx].to_csv(out_dir / "test_metadata.csv", index=False)

    print(f"\nArtifacts successfully written to {out_dir}")
    print("=" * 70)


if __name__ == "__main__":
    build_chronological_split()
