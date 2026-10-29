"""Phase 7C.6 Timestamp Causality and Sequence Step Verification Probe."""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path("c:/Users/hp/Desktop/Research project")
sys.path.insert(0, str(PROJECT_ROOT))

def verify_historical_step_timestamps():
    print("Verifying that all sequence history steps are strictly causal (t_step <= t_pred)...")

    # Load telemetry
    p7c_dir = PROJECT_ROOT / "data" / "processed" / "phase7c"
    raw_unscaled_dir = p7c_dir / "raw_unscaled"

    # We can check task_age_minutes:
    # Feature 29 (or check index) is task_age_minutes.
    # In any window [step0, step1, step2, step3, step4], task_age_minutes must be strictly monotonically increasing!
    # And step delta must be exactly 5 minutes (300s = 5.0 min)!
    import json
    with open(raw_unscaled_dir / "raw_reconstruction_manifest.json") as f:
        man = json.load(f)
    feature_cols = man["feature_definitions"]["feature_cols"]
    age_idx = feature_cols.index("task_age_minutes")

    for s_name in ["train", "val", "test"]:
        X_raw = np.load(raw_unscaled_dir / f"X_{s_name}_raw.npy")
        meta = pd.read_csv(p7c_dir / f"{s_name}_metadata.csv")

        # Check task age across 5 steps: shape (N, 5)
        ages = X_raw[:, :, age_idx]

        # 1. Monotonically increasing
        diffs = np.diff(ages, axis=1) # shape (N, 4)
        is_increasing = (diffs > 0).all()
        # Nominal step delta in minutes is 300s / 60s = 5.0 minutes
        # With grid alignment, each step is exactly 5.0 minutes!
        mean_step_delta = np.mean(diffs)
        min_step_delta = np.min(diffs)
        max_step_delta = np.max(diffs)

        print(f"[{s_name.upper():5s}] N={len(meta):,d}")
        print(f"  - Ages monotonically increasing: {is_increasing}")
        print(f"  - Step delta: mean={mean_step_delta:.2f} min, min={min_step_delta:.2f} min, max={max_step_delta:.2f} min")

        # Step 4 (last step) is prediction timestamp.
        # Check that diffs are in [4.0, 7.5] minutes (matching 240s-450s window contiguity filter)
        assert (diffs >= 4.0).all(), f"Found sub-interval or backwards step in {s_name}!"
        assert (diffs <= 7.5).all(), f"Found gap exceeding allowed contiguity in {s_name}!"
        print(f"  - Strictly causal and contiguous: VERIFIED (All step deltas between 4.0 and 7.5 min)")

if __name__ == "__main__":
    verify_historical_step_timestamps()
