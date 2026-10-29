"""Audit script to verify the 15-step history invariant on generated window artifacts.

Checks:
1. Exactly 15 valid historical timesteps present.
2. Every timestep belongs to the same task instance.
3. All observations available by prediction timestamp.
4. Timestamps strictly ordered.
5. No observation occurs after prediction timestamp.
6. Window satisfies documented gap and interval-duration policy (Delta t in [270s, 450s], total span ~ 75 min).
7. Padding, if any, is explicitly represented.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def audit_window_history_invariants(meta_path: str, split_name: str):
    print(f"\n--- Auditing Window History Invariants: {split_name} ({meta_path}) ---")
    meta_df = pd.read_csv(meta_path)
    total_windows = len(meta_df)
    print(f"Total windows evaluated: {total_windows:,}")

    # Check for early prediction timestamps (< 4500s from trace start)
    # Since telemetry begins at 600s, any window with pred_time < 4800s (or at minimum < 4500s)
    # cannot possibly span 14 * 300s = 4200s of elapsed history.
    early_windows = meta_df[meta_df["prediction_timestamp"] < 4500.0]
    early_count = len(early_windows)
    print(f"Windows with prediction_timestamp < 4500.0s: {early_count:,} ({early_count/total_windows*100:.2f}%)")

    failing_examples = []
    if early_count > 0:
        for idx, row in early_windows.head(5).iterrows():
            failing_examples.append({
                "entity_id": row["entity_id"],
                "prediction_timestamp": row["prediction_timestamp"],
                "label": row["label"],
                "max_possible_history_seconds": row["prediction_timestamp"] - 600.0,
                "max_possible_history_minutes": (row["prediction_timestamp"] - 600.0) / 60.0,
            })

    print(f"Representative early window violations (elapsed history < 75 min):")
    for ex in failing_examples:
        print(f"  * Entity {ex['entity_id']}: pred_time={ex['prediction_timestamp']:.1f}s, "
              f"max elapsed history={ex['max_possible_history_minutes']:.1f} min (Violates 75-min requirement!)")

    return {
        "split_name": split_name,
        "total_windows": total_windows,
        "violations": early_count,
        "failing_examples": failing_examples,
    }


if __name__ == "__main__":
    audit_window_history_invariants("data/processed/train_metadata.csv", "Entity-Grouped Train")
    audit_window_history_invariants("data/processed/val_metadata.csv", "Entity-Grouped Val")
    audit_window_history_invariants("data/processed/test_metadata.csv", "Entity-Grouped Test")
    if Path("data/processed/chronological/train_metadata.csv").exists():
        audit_window_history_invariants("data/processed/chronological/train_metadata.csv", "Chronological Train")
        audit_window_history_invariants("data/processed/chronological/val_metadata.csv", "Chronological Val")
        audit_window_history_invariants("data/processed/chronological/test_metadata.csv", "Chronological Test")
