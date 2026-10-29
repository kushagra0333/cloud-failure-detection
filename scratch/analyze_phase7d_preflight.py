"""Phase 7D Preflight — In-depth Scaler Weighting and Population Analysis.

Quantifies parameter differences between:
1. Window-weighted population (current Phase 7C scaler: N=16,390 window-steps)
2. Unique physical observation population (N=4,414 unique physical telemetry steps)
Using strictly Candidate A Training data.
"""

import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path("c:/Users/hp/Desktop/Research project")
sys.path.insert(0, str(PROJECT_ROOT))

def analyze_scaler_populations():
    print("=" * 80)
    print("PHASE 7D PREFLIGHT: SCALER POPULATION AND WEIGHTING ANALYSIS")
    print("=" * 80)

    # Load Phase 7C data
    p7c_dir = PROJECT_ROOT / "data" / "processed" / "phase7c"
    raw_dir = p7c_dir / "raw_unscaled"

    with open(raw_dir / "raw_reconstruction_manifest.json") as f:
        recon_man = json.load(f)
    feature_cols = recon_man["feature_definitions"]["feature_cols"]

    X_train_raw = np.load(raw_dir / "X_train_raw.npy") # (3278, 5, 31)
    meta_train = pd.read_csv(p7c_dir / "train_metadata.csv")

    with open(PROJECT_ROOT / "configs" / "phase7c" / "fitted_scaler_phase7c.json") as f:
        current_scaler = json.load(f)

    # 1. Window-Weighted Population (Current Phase 7C Scaler)
    N_windows, seq_len, n_features = X_train_raw.shape
    total_window_steps = N_windows * seq_len
    train_flat = X_train_raw.reshape(-1, n_features)

    # 2. Extract Unique Physical Steps Population
    # Map each (entity_id, step_timestamp) to its unique unscaled feature vector
    unique_step_dict = {}
    step_counts = {}
    for w_idx, row in meta_train.iterrows():
        eid = str(row["entity_id"])
        t_pred = float(row["prediction_timestamp"])
        for s_idx in range(seq_len):
            t_step = t_pred - (seq_len - 1 - s_idx) * 300.0
            k = (eid, t_step)
            step_counts[k] = step_counts.get(k, 0) + 1
            if k not in unique_step_dict:
                unique_step_dict[k] = X_train_raw[w_idx, s_idx, :]

    unique_steps_mat = np.array(list(unique_step_dict.values()))
    n_unique_steps = len(unique_steps_mat)

    print(f"Total Window-Step Observations: {total_window_steps:,} ({N_windows:,} windows x {seq_len} steps)")
    print(f"Total Unique Physical Telemetry Steps: {n_unique_steps:,}")
    print(f"Mean Multiplicity per Physical Step : {total_window_steps / n_unique_steps:.3f}x")

    # Analyze Multiplicity Distribution
    counts_series = pd.Series(list(step_counts.values()))
    print(f"Multiplicity Breakdown:")
    for mult, count in counts_series.value_counts().sort_index().items():
        print(f"  Appears {mult}x in windows: {count:5,d} physical steps ({count / n_unique_steps * 100:5.2f}%)")

    # 3. Compute Unique Physical Scaler Parameters
    unique_params = {}
    comparison = []

    for idx, col in enumerate(feature_cols):
        # Current window-weighted
        w_vals = train_flat[:, idx]
        w_obs = w_vals[~np.isnan(w_vals)]
        w_med = float(np.median(w_obs))
        w_mean = float(np.mean(w_obs))
        w_std = float(np.std(w_obs))
        if w_std < 1e-6:
            w_std = 1.0

        # Unique physical
        u_vals = unique_steps_mat[:, idx]
        u_obs = u_vals[~np.isnan(u_vals)]
        u_med = float(np.median(u_obs))
        u_mean = float(np.mean(u_obs))
        u_std = float(np.std(u_obs))
        if u_std < 1e-6:
            u_std = 1.0

        unique_params[col] = {"median": u_med, "mean": u_mean, "std": u_std}

        diff_med = abs(w_med - u_med)
        diff_mean = abs(w_mean - u_mean)
        diff_std = abs(w_std - u_std)

        rel_diff_mean = diff_mean / (abs(u_mean) + 1e-5)
        rel_diff_std = diff_std / (abs(u_std) + 1e-5)

        comparison.append({
            "feature": col,
            "window_mean": w_mean,
            "unique_mean": u_mean,
            "diff_mean": diff_mean,
            "rel_diff_mean": rel_diff_mean,
            "window_std": w_std,
            "unique_std": u_std,
            "diff_std": diff_std,
            "rel_diff_std": rel_diff_std,
            "window_median": w_med,
            "unique_median": u_med,
            "diff_median": diff_med,
        })

    comp_df = pd.DataFrame(comparison)

    print("\n" + "=" * 80)
    print("PARAMETER COMPARISON TABLE (Window-Weighted vs Unique-Physical)")
    print("=" * 80)
    print(f"{'Feature Name':32s} | {'Window Mean':11s} {'Unique Mean':11s} | {'Mean Diff':10s} | {'Window Std':11s} {'Unique Std':11s} | {'Std Diff':10s}")
    print("-" * 105)
    for _, r in comp_df.iterrows():
        print(f"{r['feature']:32s} | {r['window_mean']:11.5f} {r['unique_mean']:11.5f} | {r['diff_mean']:10.5f} | {r['window_std']:11.5f} {r['unique_std']:11.5f} | {r['diff_std']:10.5f}")

    # Summary of Shifts
    print("\nSummary of Parameter Shifts:")
    print(f"  Mean absolute shift in Mean: {comp_df['diff_mean'].mean():.5f} (max: {comp_df['diff_mean'].max():.5f} in {comp_df.loc[comp_df['diff_mean'].idxmax(), 'feature']})")
    print(f"  Mean absolute shift in Std : {comp_df['diff_std'].mean():.5f} (max: {comp_df['diff_std'].max():.5f} in {comp_df.loc[comp_df['diff_std'].idxmax(), 'feature']})")
    print(f"  Mean absolute shift in Med : {comp_df['diff_median'].mean():.5f} (max: {comp_df['diff_median'].max():.5f} in {comp_df.loc[comp_df['diff_median'].idxmax(), 'feature']})")

    # 4. Measure Effect on Standardized Distributions
    # If we standardize X_train_raw using current vs unique scaler:
    # How much do standardized values shift?
    print("\nStandardized Value Discrepancy on Training Windows:")
    # For each observation: z_w = (x - w_mean) / w_std, z_u = (x - u_mean) / u_std
    z_diffs_all = []
    for idx, col in enumerate(feature_cols):
        w_m, w_s = current_scaler["means"][col], current_scaler["stds"][col]
        u_m, u_s = unique_params[col]["mean"], unique_params[col]["std"]

        raw_feat = train_flat[:, idx]
        obs = raw_feat[~np.isnan(raw_feat)]

        z_w = (obs - w_m) / (w_s + 1e-6)
        z_u = (obs - u_m) / (u_s + 1e-6)

        z_diff = np.abs(z_w - z_u)
        z_diffs_all.append({
            "feature": col,
            "max_z_diff": float(np.max(z_diff)),
            "mean_z_diff": float(np.mean(z_diff)),
            "p95_z_diff": float(np.percentile(z_diff, 95))
        })

    z_df = pd.DataFrame(z_diffs_all).sort_values("mean_z_diff", ascending=False)
    print(f"Top 5 features with largest mean z-score shift:")
    for _, r in z_df.head(5).iterrows():
        print(f"  {r['feature']:32s}: mean |z_w - z_u| = {r['mean_z_diff']:.4f}, max = {r['max_z_diff']:.4f}, p95 = {r['p95_z_diff']:.4f}")

    print(f"Bottom 5 features with smallest mean z-score shift:")
    for _, r in z_df.tail(5).iterrows():
        print(f"  {r['feature']:32s}: mean |z_w - z_u| = {r['mean_z_diff']:.4f}, max = {r['max_z_diff']:.4f}, p95 = {r['p95_z_diff']:.4f}")

    # Output JSON for preflight ledger
    output = {
        "total_window_steps": total_window_steps,
        "unique_physical_steps": n_unique_steps,
        "multiplicity_mean": float(total_window_steps / n_unique_steps),
        "multiplicity_distribution": {int(k): int(v) for k, v in counts_series.value_counts().sort_index().items()},
        "top_z_shifts": z_df.head(10).to_dict(orient="records"),
        "parameter_comparison": comparison,
    }

    out_path = PROJECT_ROOT / "scratch" / "phase7d_scaler_analysis.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nAnalysis serialized to {out_path}")

if __name__ == "__main__":
    analyze_scaler_populations()
