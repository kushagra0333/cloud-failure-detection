import pandas as pd
import numpy as np

val_meta = pd.read_csv("data/processed/phase7c/val_metadata.csv")
y_val = np.load("data/processed/phase7c/y_val.npy")
val_meta["y_true"] = y_val

# Raw fail timestamp without rounding
val_meta["fail_timestamp_raw"] = val_meta["prediction_timestamp"] + val_meta["time_to_failure"]
val_meta["fail_timestamp_round1"] = val_meta["fail_timestamp_raw"].round(1)
val_meta["fail_timestamp_round4"] = val_meta["fail_timestamp_raw"].round(4)

prefails = val_meta[val_meta["y_true"] > 0].copy()

# Grouping by raw vs round1 vs round4
raw_groups = list(prefails.groupby(["entity_id", "fail_timestamp_raw"]))
round1_groups = list(prefails.groupby(["entity_id", "fail_timestamp_round1"]))
round4_groups = list(prefails.groupby(["entity_id", "fail_timestamp_round4"]))
eid_only_groups = list(prefails.groupby("entity_id"))

print(f"Number of pre-failure windows: {len(prefails)}")
print(f"Number of groups by entity_id alone: {len(eid_only_groups)}")
print(f"Number of groups by (entity_id, fail_timestamp_raw): {len(raw_groups)}")
print(f"Number of groups by (entity_id, fail_timestamp_round1): {len(round1_groups)}")
print(f"Number of groups by (entity_id, fail_timestamp_round4): {len(round4_groups)}")

# Inspect the 20 entities
episodes_info = []
for (eid, ftime), grp in round1_groups:
    timestamps = grp["prediction_timestamp"].values
    ttfs = grp["time_to_failure"].values
    labels = grp["y_true"].values
    
    # Check if raw timestamps have any float differences
    raw_ftimes = grp["fail_timestamp_raw"].values
    spread = raw_ftimes.max() - raw_ftimes.min()
    
    episodes_info.append({
        "entity_id": eid,
        "fail_timestamp": ftime,
        "window_count": len(grp),
        "warning_count": int((labels == 1).sum()),
        "critical_count": int((labels == 2).sum()),
        "min_pred_time": float(timestamps.min()),
        "max_pred_time": float(timestamps.max()),
        "min_ttf_sec": float(ttfs.min()),
        "max_ttf_sec": float(ttfs.max()),
        "raw_fail_time_spread": float(spread)
    })

df_ep_info = pd.DataFrame(episodes_info)
print("\nEpisode summary table:")
print(df_ep_info.to_string())

# Check for entities that fail multiple times in validation:
fail_counts = df_ep_info["entity_id"].value_counts()
print(f"\nMax episodes per entity: {fail_counts.max()}")
print(f"Entities with >1 episode: {(fail_counts > 1).sum()}")

# What about the entire dataset (Train vs Val)? Does any entity fail multiple times in Train?
tr_meta = pd.read_csv("data/processed/phase7c/train_metadata.csv")
y_tr = np.load("data/processed/phase7c/y_train.npy")
tr_meta["y_true"] = y_tr
tr_meta["fail_timestamp_round1"] = (tr_meta["prediction_timestamp"] + tr_meta["time_to_failure"]).round(1)
tr_prefails = tr_meta[tr_meta["y_true"] > 0].copy()
tr_episodes = list(tr_prefails.groupby(["entity_id", "fail_timestamp_round1"]))
print(f"\nTrain pre-failure windows: {len(tr_prefails)}")
print(f"Train total pre-failure episodes: {len(tr_episodes)}")
print(f"Train unique entities with failure: {len(tr_prefails['entity_id'].unique())}")
tr_fail_counts = tr_prefails.groupby("entity_id")["fail_timestamp_round1"].nunique()
print(f"Train entities with >1 episode: {(tr_fail_counts > 1).sum()}")
if (tr_fail_counts > 1).sum() > 0:
    print("Multi-failure entities in Train:", tr_fail_counts[tr_fail_counts > 1].to_dict())
