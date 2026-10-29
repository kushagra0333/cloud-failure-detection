"""Phase 7C.3 — Leakage-Free Preprocessing and Dataset Standardization.

Adheres strictly to Phase 7C protocol:
1. Fits LeakageSafeScaler strictly on Candidate A Training split (N=3,278 windows, 16,390 steps).
2. Preserves and logs genuine raw missingness before imputation.
3. Imputation medians, scaling means, and scaling stds are fitted EXCLUSIVELY on training data.
4. Serializes Candidate A scaler to configs/phase7c/fitted_scaler_phase7c.json.
5. Standardizes Train, Val, and Test splits using the training-only parameters without refitting.
6. Verifies all output arrays contain no non-finite values (np.isfinite).
7. Serializes clean dataset arrays to data/processed/phase7c/.
"""

import sys
import json
import time
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.features.scaler import LeakageSafeScaler

def compute_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def prepare_phase7c_leakage_free_data(
    raw_data_dir: Path,
    output_data_dir: Path,
    config_dir: Path,
    results_dir: Path,
) -> Dict[str, Any]:
    print("=" * 80)
    print("PHASE 7C.3: LEAKAGE-FREE PREPROCESSING & DATASET STANDARDIZATION")
    print("=" * 80)
    start_time = time.time()

    output_data_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load raw unscaled arrays and metadata
    print(f"\n[Step 1] Loading raw unscaled tensors from {raw_data_dir}...")
    manifest_path = raw_data_dir / "raw_reconstruction_manifest.json"
    with open(manifest_path, "r") as f:
        raw_manifest = json.load(f)

    feature_cols = raw_manifest["feature_definitions"]["feature_cols"]
    print(f"  - Features count: {len(feature_cols)}")

    X_train_raw = np.load(raw_data_dir / "X_train_raw.npy")
    y_train = np.load(raw_data_dir / "y_train.npy")
    meta_train = pd.read_csv(raw_data_dir / "train_metadata.csv")

    X_val_raw = np.load(raw_data_dir / "X_val_raw.npy")
    y_val = np.load(raw_data_dir / "y_val.npy")
    meta_val = pd.read_csv(raw_data_dir / "val_metadata.csv")

    X_test_raw = np.load(raw_data_dir / "X_test_raw.npy")
    y_test = np.load(raw_data_dir / "y_test.npy")
    meta_test = pd.read_csv(raw_data_dir / "test_metadata.csv")

    print(f"  - Loaded Train: X={X_train_raw.shape}, y={y_train.shape}, meta={len(meta_train)}")
    print(f"  - Loaded Val:   X={X_val_raw.shape}, y={y_val.shape}, meta={len(meta_val)}")
    print(f"  - Loaded Test:  X={X_test_raw.shape}, y={y_test.shape}, meta={len(meta_test)}")

    # 2. Record Pre-Imputation Missingness
    print("\n[Step 2] Auditing raw missingness across all partitions...")
    raw_missingness = {}
    for name, arr in [("train", X_train_raw), ("val", X_val_raw), ("test", X_test_raw)]:
        n_obs = arr.shape[0] * arr.shape[1]
        nans = np.isnan(arr).sum(axis=(0, 1))
        raw_missingness[name] = {
            "total_observations": n_obs,
            "features": {}
        }
        for idx, col in enumerate(feature_cols):
            cnt = int(nans[idx])
            raw_missingness[name]["features"][col] = {
                "nan_count": cnt,
                "nan_pct": float(cnt / n_obs * 100.0)
            }
            if cnt > 0:
                print(f"    * [{name:5s}] {col:32s}: {cnt:5,d} / {n_obs:,d} ({cnt/n_obs*100:5.2f}%) NaNs")

    # 3. Fit LeakageSafeScaler Strictly on Candidate A Training Partition
    print("\n[Step 3] Fitting LeakageSafeScaler strictly on Candidate A Train split...")
    n_train_obs = X_train_raw.shape[0] * X_train_raw.shape[1]
    train_flat_df = pd.DataFrame(
        X_train_raw.reshape(-1, len(feature_cols)),
        columns=feature_cols
    )

    scaler = LeakageSafeScaler(epsilon=1e-6)
    scaler.fit(train_flat_df, feature_cols)

    # Save fitted scaler artifact
    scaler_save_path = config_dir / "fitted_scaler_phase7c.json"
    scaler.save(scaler_save_path)
    scaler_sha256 = compute_sha256(scaler_save_path)
    print(f"  - Scaler successfully fitted and serialized to: {scaler_save_path}")
    print(f"  - Scaler SHA-256: {scaler_sha256}")

    # Log fitted parameters for critical features
    for col in ["cycles_per_instruction", "memory_accesses_per_instruction", "cpu_utilization", "memory_utilization"]:
        print(f"    * {col:32s}: median={scaler.medians[col]:.6f}, mean={scaler.means[col]:.6f}, std={scaler.stds[col]:.6f}")

    # 4. Standardize Splits Using Training-Only Parameters
    print("\n[Step 4] Applying training-only scaler to Train, Val, and Test partitions...")
    val_flat_df = pd.DataFrame(X_val_raw.reshape(-1, len(feature_cols)), columns=feature_cols)
    test_flat_df = pd.DataFrame(X_test_raw.reshape(-1, len(feature_cols)), columns=feature_cols)

    X_train_scaled = scaler.transform(train_flat_df).values.reshape(X_train_raw.shape).astype(np.float32)
    X_val_scaled = scaler.transform(val_flat_df).values.reshape(X_val_raw.shape).astype(np.float32)
    X_test_scaled = scaler.transform(test_flat_df).values.reshape(X_test_raw.shape).astype(np.float32)

    # 5. Numerical Integrity Verification
    print("\n[Step 5] Verifying numerical integrity and standardization properties...")
    for name, arr in [("Train", X_train_scaled), ("Val", X_val_scaled), ("Test", X_test_scaled)]:
        assert np.all(np.isfinite(arr)), f"FATAL: Non-finite values detected in {name} split!"
        assert not np.any(np.isnan(arr)), f"FATAL: NaN values detected in {name} split after imputation!"
        print(f"  - {name} Array finite integrity: PASS (0 NaNs, 0 Infs across {arr.size:,} floats)")

    # Verify train standardization mean 0, std 1
    tr_means = np.mean(X_train_scaled, axis=(0, 1))
    tr_stds = np.std(X_train_scaled, axis=(0, 1))
    for idx, col in enumerate(feature_cols):
        raw_feat = X_train_raw[:, :, idx]
        obs_mask = ~np.isnan(raw_feat)
        raw_obs = raw_feat[obs_mask]
        scaled_obs = X_train_scaled[:, :, idx][obs_mask]

        raw_std = np.std(raw_obs)
        if raw_std < 1e-6:
            assert np.abs(np.std(scaled_obs)) < 1e-4, f"Constant feature {col} std is not 0"
        else:
            assert np.abs(np.mean(scaled_obs)) < 1e-3, f"Feature {col} observed train mean is not 0: {np.mean(scaled_obs)}"
            assert np.abs(np.std(scaled_obs) - 1.0) < 1e-3, f"Feature {col} observed train std is not 1: {np.std(scaled_obs)}"

        # If there are missing values, verify imputed entries equal transformed median
        if (~obs_mask).any():
            expected_imputed = (scaler.medians[col] - scaler.means[col]) / (scaler.stds[col] + scaler.epsilon)
            actual_imputed = X_train_scaled[:, :, idx][~obs_mask]
            assert np.allclose(actual_imputed, expected_imputed, atol=1e-5), f"Feature {col} imputed values do not match transformed median!"
    print("  - Train standardization verified: observed features have mean=0.0, std=1.0; missing values imputed with transformed training median.")

    # 6. Save Processed Datasets and Manifests
    print(f"\n[Step 6] Saving standardized arrays to {output_data_dir}...")
    np.save(output_data_dir / "X_train.npy", X_train_scaled)
    np.save(output_data_dir / "y_train.npy", y_train)
    meta_train.to_csv(output_data_dir / "train_metadata.csv", index=False)

    np.save(output_data_dir / "X_val.npy", X_val_scaled)
    np.save(output_data_dir / "y_val.npy", y_val)
    meta_val.to_csv(output_data_dir / "val_metadata.csv", index=False)

    np.save(output_data_dir / "X_test.npy", X_test_scaled)
    np.save(output_data_dir / "y_test.npy", y_test)
    meta_test.to_csv(output_data_dir / "test_metadata.csv", index=False)

    # 7. Create Authoritative Phase 7C Manifests
    scaler_manifest = {
        "creation_timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "scaler_path": str(scaler_save_path),
        "scaler_sha256": scaler_sha256,
        "fitted_on_split": "train",
        "training_windows": len(meta_train),
        "training_observations": n_train_obs,
        "features_count": len(feature_cols),
        "feature_cols": feature_cols,
        "fitted_statistics": {
            col: {
                "median": float(scaler.medians[col]),
                "mean": float(scaler.means[col]),
                "std": float(scaler.stds[col]),
            } for col in feature_cols
        },
        "raw_missingness": raw_missingness,
    }

    scaler_manifest_path = results_dir / "scaler_manifest.json"
    with open(scaler_manifest_path, "w") as f:
        json.dump(scaler_manifest, f, indent=2)

    data_manifest = {
        "creation_timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "duration_seconds": round(time.time() - start_time, 2),
        "candidate_split": "Candidate A (Seed 48, Job-Grouped)",
        "features_count": len(feature_cols),
        "sequence_length": 5,
        "scaler_sha256": scaler_sha256,
        "partitions": {
            "train": {
                "windows": len(meta_train),
                "entities": int(meta_train["entity_id"].nunique()),
                "parent_jobs": int(meta_train["parent_job_id"].nunique()),
                "X_shape": list(X_train_scaled.shape),
                "X_sha256": compute_sha256(output_data_dir / "X_train.npy"),
                "y_sha256": compute_sha256(output_data_dir / "y_train.npy"),
                "meta_sha256": compute_sha256(output_data_dir / "train_metadata.csv"),
                "class_distribution": {int(k): int(v) for k, v in pd.Series(y_train).value_counts().sort_index().items()}
            },
            "val": {
                "windows": len(meta_val),
                "entities": int(meta_val["entity_id"].nunique()),
                "parent_jobs": int(meta_val["parent_job_id"].nunique()),
                "X_shape": list(X_val_scaled.shape),
                "X_sha256": compute_sha256(output_data_dir / "X_val.npy"),
                "y_sha256": compute_sha256(output_data_dir / "y_val.npy"),
                "meta_sha256": compute_sha256(output_data_dir / "val_metadata.csv"),
                "class_distribution": {int(k): int(v) for k, v in pd.Series(y_val).value_counts().sort_index().items()}
            },
            "test": {
                "windows": len(meta_test),
                "entities": int(meta_test["entity_id"].nunique()),
                "parent_jobs": int(meta_test["parent_job_id"].nunique()),
                "X_shape": list(X_test_scaled.shape),
                "X_sha256": compute_sha256(output_data_dir / "X_test.npy"),
                "y_sha256": compute_sha256(output_data_dir / "y_test.npy"),
                "meta_sha256": compute_sha256(output_data_dir / "test_metadata.csv"),
                "class_distribution": {int(k): int(v) for k, v in pd.Series(y_test).value_counts().sort_index().items()}
            }
        }
    }

    data_manifest_path = results_dir / "data_manifest.json"
    with open(data_manifest_path, "w") as f:
        json.dump(data_manifest, f, indent=2)

    print(f"\nPhase 7C Scaler Manifest: {scaler_manifest_path}")
    print(f"Phase 7C Data Manifest:   {data_manifest_path}")
    print("=" * 80)
    print("PREPROCESSING & STANDARDIZATION COMPLETE")
    print("=" * 80)
    return data_manifest

def main():
    raw_data_dir = PROJECT_ROOT / "data" / "processed" / "phase7c" / "raw_unscaled"
    output_data_dir = PROJECT_ROOT / "data" / "processed" / "phase7c"
    config_dir = PROJECT_ROOT / "configs" / "phase7c"
    results_dir = PROJECT_ROOT / "results" / "phase7c"

    prepare_phase7c_leakage_free_data(
        raw_data_dir=raw_data_dir,
        output_data_dir=output_data_dir,
        config_dir=config_dir,
        results_dir=results_dir,
    )

if __name__ == "__main__":
    main()
