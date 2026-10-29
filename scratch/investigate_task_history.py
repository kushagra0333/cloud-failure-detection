import pandas as pd
import numpy as np

meta_train = pd.read_csv("data/processed/phase7c/train_metadata.csv")
meta_val = pd.read_csv("data/processed/phase7c/val_metadata.csv")

print("=== TASK HISTORICAL OBSERVATION DEPTH INVESTIGATION ===")

for name, meta in [("Train", meta_train), ("Val", meta_val)]:
    print(f"\n--- {name} Split (N_windows={len(meta)}, N_entities={meta['entity_id'].nunique()}) ---")
    
    # Windows per entity
    win_per_ent = meta.groupby("entity_id")["prediction_timestamp"].count()
    print(f"Windows per entity: min={win_per_ent.min()}, max={win_per_ent.max()}, median={win_per_ent.median():.1f}, mean={win_per_ent.mean():.1f}")
    print(f"Entities with exactly 1 window: {(win_per_ent == 1).sum()} ({(win_per_ent == 1).mean()*100:.1f}%)")
    print(f"Entities with <= 3 windows: {(win_per_ent <= 3).sum()} ({(win_per_ent <= 3).mean()*100:.1f}%)")
    print(f"Entities with >= 6 windows: {(win_per_ent >= 6).sum()} ({(win_per_ent >= 6).mean()*100:.1f}%)")
    print(f"Entities with >= 12 windows: {(win_per_ent >= 12).sum()} ({(win_per_ent >= 12).mean()*100:.1f}%)")
    
    # Check elapsed time from first prediction timestamp to current prediction timestamp
    first_time = meta.groupby("entity_id")["prediction_timestamp"].transform("min")
    elapsed_time_sec = meta["prediction_timestamp"] - first_time
    elapsed_steps = (elapsed_time_sec / 300.0).round().astype(int)
    
    print(f"Cumulative available windows before current window:")
    print(f"  Step 0 (first window of task): {(elapsed_steps == 0).sum()} windows ({(elapsed_steps == 0).mean()*100:.1f}%)")
    print(f"  Steps < 3 (insufficient history): {(elapsed_steps < 3).sum()} windows ({(elapsed_steps < 3).mean()*100:.1f}%)")
    print(f"  Steps >= 6 (sufficient history): {(elapsed_steps >= 6).sum()} windows ({(elapsed_steps >= 6).mean()*100:.1f}%)")
