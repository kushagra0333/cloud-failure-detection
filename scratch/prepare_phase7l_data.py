"""Phase 7L — Dataset Standardization and Preflight Manifest Generator."""
import json
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(".").resolve()
results_dir = PROJECT_ROOT / "results" / "phase7l_longer_context"
raw_dir = results_dir / "raw_features"
processed_dir = results_dir / "standardized_data"
results_dir.mkdir(parents=True, exist_ok=True)
processed_dir.mkdir(parents=True, exist_ok=True)

def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

print("=" * 80)
print("PHASE 7L: PREFLIGHT AUDIT & DATASET STANDARDIZATION")
print("=" * 80)

# 1. Load Raw Unscaled Arrays
X_tr_raw_l5 = np.load(raw_dir / "X_train_raw_l5.npy")
X_tr_raw_l12 = np.load(raw_dir / "X_train_raw_l12.npy")
X_val_raw_l5 = np.load(raw_dir / "X_val_raw_l5.npy")
X_val_raw_l12 = np.load(raw_dir / "X_val_raw_l12.npy")

y_tr = np.load(raw_dir / "y_train.npy")
y_val = np.load(raw_dir / "y_val.npy")

meta_tr = pd.read_csv(raw_dir / "train_metadata.csv")
meta_val = pd.read_csv(raw_dir / "val_metadata.csv")

# 2. Check Candidate A Split and Labels
split_path = PROJECT_ROOT / "results" / "phase7a_4" / "candidate_a_job_grouped_split.csv"
split_hash = sha256(split_path)
print(f"Candidate A Split Hash: {split_hash}")
assert split_hash == "f84bc4fe54f8e0583e2ee52f49b2fbe8e6fcafb4f020c656985a1acc31bb4946"

# Verify reconciled validation labels: Normal=560, Warning=48, Critical=49
val_counts = pd.Series(y_val).value_counts().to_dict()
print(f"Validation Label Counts: {val_counts}")
assert val_counts == {0: 560, 2: 49, 1: 48}
assert len(y_val) == 657
assert len(y_tr) == 3278

# 3. Fit Preprocessors Strictly on Candidate A Train Partition
# Using the 31 feature names from Phase 7C
feature_cols = [
    "cpu_utilization", "memory_utilization", "assigned_memory", "disk_io_time",
    "cycles_per_instruction", "memory_accesses_per_instruction", "cpu_utilization_lag_1",
    "memory_utilization_lag_1", "cpu_utilization_lag_2", "memory_utilization_lag_2",
    "cpu_utilization_lag_3", "memory_utilization_lag_3", "cpu_utilization_lag_6",
    "memory_utilization_lag_6", "cpu_diff_1", "cpu_rel_change_1", "memory_diff_1",
    "cpu_rolling_mean_3", "cpu_rolling_std_3", "cpu_rolling_range_3", "memory_rolling_mean_3",
    "cpu_rolling_mean_6", "cpu_rolling_std_6", "cpu_rolling_range_6", "memory_rolling_mean_6",
    "cpu_trend_slope_6", "cpu_exceed_85", "cpu_exceed_fraction_6", "memory_to_assigned_ratio",
    "task_age_minutes", "cpi_missing_fraction_6"
]
assert len(feature_cols) == 31

# Compute Train-only statistics across all training observations in X_tr_raw_l5
# To maintain identical feature scaling across L=5 and L=12, the scaling statistics
# are fitted on the training split and applied identically to both L=5 and L=12.
N_tr, L5, D = X_tr_raw_l5.shape
tr_flat = X_tr_raw_l5.reshape(-1, D)

medians = np.zeros(D, dtype=np.float32)
means = np.zeros(D, dtype=np.float32)
stds = np.zeros(D, dtype=np.float32)

for j in range(D):
    col_vals = tr_flat[:, j]
    valid = col_vals[~np.isnan(col_vals)]
    med = float(np.median(valid)) if len(valid) > 0 else 0.0
    mean = float(np.mean(valid)) if len(valid) > 0 else 0.0
    std = float(np.std(valid)) if len(valid) > 0 else 1.0
    if std < 1e-6 or np.isnan(std):
        std = 1.0
    medians[j] = med
    means[j] = mean
    stds[j] = std

# Standardize L=5
X_tr_l5_flat = np.nan_to_num(X_tr_raw_l5.reshape(-1, D), nan=medians)
X_tr_l5_scaled = ((X_tr_l5_flat - means) / (stds + 1e-6)).reshape(N_tr, 5, D).astype(np.float32)

X_val_l5_flat = np.nan_to_num(X_val_raw_l5.reshape(-1, D), nan=medians)
X_val_l5_scaled = ((X_val_l5_flat - means) / (stds + 1e-6)).reshape(len(X_val_raw_l5), 5, D).astype(np.float32)

# Standardize L=12 using IDENTICAL Train parameters
X_tr_l12_flat = np.nan_to_num(X_tr_raw_l12.reshape(-1, D), nan=medians)
X_tr_l12_scaled = ((X_tr_l12_flat - means) / (stds + 1e-6)).reshape(N_tr, 12, D).astype(np.float32)

X_val_l12_flat = np.nan_to_num(X_val_raw_l12.reshape(-1, D), nan=medians)
X_val_l12_scaled = ((X_val_l12_flat - means) / (stds + 1e-6)).reshape(len(X_val_raw_l12), 12, D).astype(np.float32)

# Verify trailing 5 steps match bit-for-bit
assert np.allclose(X_tr_l12_scaled[:, -5:, :], X_tr_l5_scaled, atol=1e-6)
assert np.allclose(X_val_l12_scaled[:, -5:, :], X_val_l5_scaled, atol=1e-6)
print("Trailing 5-step equality between L=12 and L=5: VERIFIED EXACT")

# Check finite values
assert np.all(np.isfinite(X_tr_l5_scaled))
assert np.all(np.isfinite(X_val_l5_scaled))
assert np.all(np.isfinite(X_tr_l12_scaled))
assert np.all(np.isfinite(X_val_l12_scaled))
print("Finite integrity check: PASS (0 NaNs, 0 Infs)")

# Save standardized tensors
np.save(processed_dir / "X_train_l5.npy", X_tr_l5_scaled)
np.save(processed_dir / "X_val_l5.npy", X_val_l5_scaled)
np.save(processed_dir / "X_train_l12.npy", X_tr_l12_scaled)
np.save(processed_dir / "X_val_l12.npy", X_val_l12_scaled)
np.save(processed_dir / "y_train.npy", y_tr)
np.save(processed_dir / "y_val.npy", y_val)
meta_tr.to_csv(processed_dir / "train_metadata.csv", index=False)
meta_val.to_csv(processed_dir / "val_metadata.csv", index=False)

# Save Preprocessors Manifest
preprocessor_manifest = {
    "manifest_name": "fitted_preprocessors_manifest",
    "phase": "Phase 7L",
    "fitted_split": "Candidate A Train only",
    "train_window_count": 3278,
    "train_observation_steps": 16390,
    "feature_count": 31,
    "feature_cols": feature_cols,
    "medians": [float(m) for m in medians],
    "means": [float(m) for m in means],
    "stds": [float(s) for s in stds]
}
with open(results_dir / "fitted_preprocessors_manifest.json", "w") as f:
    json.dump(preprocessor_manifest, f, indent=2)

# Save Preflight Data Manifest
data_manifest = {
    "manifest_name": "phase7l_data_manifest",
    "phase": "Phase 7L",
    "timestamp": "2026-10-29T00:45:00+05:30",
    "governance": {
        "candidate_a_split_path": str(split_path),
        "candidate_a_split_sha256": split_hash,
        "candidate_a_test_accessed": False,
        "candidate_a_test_historical_status": "INSPECTED_IN_PHASE_7D_NON_VIRGIN",
        "phase_6_holdout_quarantined": True,
        "train_size": 3278,
        "val_size": 657,
        "val_labels_breakdown": {"normal": 560, "warning": 48, "critical": 49}
    },
    "standardized_tensors": {
        "X_train_l5": {"shape": list(X_tr_l5_scaled.shape), "sha256": sha256(processed_dir / "X_train_l5.npy")},
        "X_val_l5": {"shape": list(X_val_l5_scaled.shape), "sha256": sha256(processed_dir / "X_val_l5.npy")},
        "X_train_l12": {"shape": list(X_tr_l12_scaled.shape), "sha256": sha256(processed_dir / "X_train_l12.npy")},
        "X_val_l12": {"shape": list(X_val_l12_scaled.shape), "sha256": sha256(processed_dir / "X_val_l12.npy")},
        "y_train": {"shape": list(y_tr.shape), "sha256": sha256(processed_dir / "y_train.npy")},
        "y_val": {"shape": list(y_val.shape), "sha256": sha256(processed_dir / "y_val.npy")},
        "train_metadata": {"rows": len(meta_tr), "sha256": sha256(processed_dir / "train_metadata.csv")},
        "val_metadata": {"rows": len(meta_val), "sha256": sha256(processed_dir / "val_metadata.csv")}
    }
}
with open(results_dir / "phase7l_data_manifest.json", "w") as f:
    json.dump(data_manifest, f, indent=2)

# Save Experiment Config
experiment_config = {
    "experiment_name": "Phase 7L Controlled Longer-Temporal-Context Experiment",
    "research_question": "Does increasing the causal input sequence length from L=5 steps (25 minutes) to L=12 steps (60 minutes) improve validation failure-prediction performance?",
    "conditions": {
        "baseline": {
            "name": "Baseline_L5",
            "seq_len": 5,
            "temporal_span_minutes": 25.0,
            "temporal_span_seconds": 1500.0,
            "feature_dim": 31
        },
        "treatment": {
            "name": "Treatment_L12",
            "seq_len": 12,
            "temporal_span_minutes": 60.0,
            "temporal_span_seconds": 3600.0,
            "feature_dim": 31
        }
    },
    "loss_weighting_conditions": {
        "cond_B": {"name": "Moderate [1,2,2] (Pre-declared Primary Baseline)", "weights": [1.0, 2.0, 2.0]},
        "cond_A": {"name": "Unweighted [1,1,1] (Reference Baseline)", "weights": [1.0, 1.0, 1.0]}
    },
    "seeds": [42, 123, 456],
    "hyperparameters": {
        "architecture": "TemporalLSTMClassifier (2-layer LSTM)",
        "hidden_dim": 64,
        "num_layers": 2,
        "num_classes": 3,
        "dropout": 0.2,
        "optimizer": "Adam",
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "batch_size": 64,
        "max_epochs": 25,
        "checkpoint_selection": "Minimum validation loss"
    },
    "evaluation_protocol": {
        "threshold_steps": 201,
        "threshold_range": [0.0, 1.0],
        "operational_gates": {
            "gate_1a_episode_recall": ">= 70.0%",
            "gate_1b_critical_recall": ">= 50.0%",
            "gate_2a_entity_far": "<= 10.0%",
            "gate_2b_window_fpr": "<= 5.0%"
        }
    }
}
with open(results_dir / "phase7l_experiment_config.json", "w") as f:
    json.dump(experiment_config, f, indent=2)

print("Preflight audit and standardization completed successfully!")
print(f"Artifacts saved in {results_dir}")
