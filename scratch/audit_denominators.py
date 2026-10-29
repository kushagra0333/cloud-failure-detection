import pandas as pd
import numpy as np

val_meta = pd.read_csv("data/processed/phase7c/val_metadata.csv")
y_val = np.load("data/processed/phase7c/y_val.npy")
val_meta["y_true"] = y_val

print(f"Total validation windows: {len(val_meta)}")
print(f"Normal-labelled windows (y_true == 0): {(val_meta['y_true'] == 0).sum()}")
print(f"Warning windows (y_true == 1): {(val_meta['y_true'] == 1).sum()}")
print(f"Critical windows (y_true == 2): {(val_meta['y_true'] == 2).sum()}")

failing_eids = set(val_meta[val_meta["y_true"] > 0]["entity_id"].unique())
all_eids = set(val_meta["entity_id"].unique())
pure_norm_eids = all_eids - failing_eids

print(f"\nTotal entities: {len(all_eids)}")
print(f"Failing entities: {len(failing_eids)}")
print(f"Pure-normal entities: {len(pure_norm_eids)}")

windows_pure_norm_entities = val_meta[val_meta["entity_id"].isin(pure_norm_eids)]
print(f"\nWindows belonging to pure-normal entities: {len(windows_pure_norm_entities)}")
print("Label distribution for windows of pure-normal entities:")
print(windows_pure_norm_entities["y_true"].value_counts().to_dict())

windows_failing_entities = val_meta[val_meta["entity_id"].isin(failing_eids)]
print(f"\nWindows belonging to failing entities: {len(windows_failing_entities)}")
print("Label distribution for windows of failing entities:")
print(windows_failing_entities["y_true"].value_counts().to_dict())

failing_jobs = set(val_meta[val_meta["y_true"] > 0]["parent_job_id"].unique())
all_jobs = set(val_meta["parent_job_id"].unique())
pure_norm_jobs = all_jobs - failing_jobs
print(f"\nTotal parent jobs: {len(all_jobs)}")
print(f"Failing parent jobs: {len(failing_jobs)}")
print(f"Pure-normal parent jobs: {len(pure_norm_jobs)}")

windows_pure_norm_jobs = val_meta[val_meta["parent_job_id"].isin(pure_norm_jobs)]
print(f"Windows belonging to pure-normal parent jobs: {len(windows_pure_norm_jobs)}")
print("Label distribution for windows of pure-normal jobs:")
print(windows_pure_norm_jobs["y_true"].value_counts().to_dict())

# Episode grouping inspection
val_meta["fail_timestamp"] = (val_meta["prediction_timestamp"] + val_meta["time_to_failure"]).round(1)
prefails = val_meta[val_meta["y_true"] > 0].copy()
episodes = list(prefails.groupby(["entity_id", "fail_timestamp"]))
print(f"\nTotal pre-failure episodes: {len(episodes)}")
print(f"Unique entities among episodes: {len(prefails['entity_id'].unique())}")

ep_lengths = [len(grp) for _, grp in episodes]
print(f"Episode length (windows) min={min(ep_lengths)}, max={max(ep_lengths)}, mean={np.mean(ep_lengths):.2f}")
print("Episode lengths breakdown:", pd.Series(ep_lengths).value_counts().to_dict())
