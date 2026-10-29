"""Inspect repeated failures, lifecycle events, and window history boundaries in detail."""

import json
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent

with open(PROJECT_ROOT / "results" / "phase6" / "protocol_l5_amendment_2_1.json") as f:
    protocol = json.load(f)

fail_eids = set(protocol["frozen_entity_ids"]["primary_failure_entities"])
pure_norm_eids = set(protocol["frozen_entity_ids"]["primary_normal_entities_evaluable"])

# Load raw events
event_files = sorted((PROJECT_ROOT / "data" / "raw").glob("sample_task_events*.csv.gz"))
EVENT_COLS = [
    "timestamp", "missing_info", "job_id", "task_index", "machine_id",
    "event_type", "user_name", "scheduling_class", "priority",
    "req_cpu", "req_ram", "req_disk", "diff_machine_restriction"
]
event_dfs = []
for ep in event_files:
    df = pd.read_csv(ep, header=None, names=EVENT_COLS)
    df["entity_id"] = df["job_id"].astype(str) + "_" + df["task_index"].astype(str)
    matched = df[df["entity_id"].isin(fail_eids)]
    if not matched.empty:
        event_dfs.append(matched)

events = pd.concat(event_dfs, ignore_index=True)
events["time_sec"] = events["timestamp"] / 1e6
events = events.sort_values(["entity_id", "time_sec"])

fail_events = events[events["event_type"] == 3].copy()
print(f"Total FAIL events: {len(fail_events)} across {fail_events['entity_id'].nunique()} entities")

# How many entities have multiple FAIL events?
fail_counts = fail_events.groupby("entity_id").size()
multi_fail_eids = fail_counts[fail_counts > 1]
print(f"\nEntities with multiple FAIL events: {len(multi_fail_eids)}")
for eid, count in multi_fail_eids.items():
    times = fail_events[fail_events["entity_id"] == eid]["time_sec"].tolist()
    print(f"  {eid}: {count} failures at {times}")

# Load predictions
preds = pd.read_csv(PROJECT_ROOT / "results" / "phase6" / "confirmatory_predictions_l5.csv")

# For entities with multiple failures, inspect predictions
print("\n--- Predictions for entities with multiple failures ---")
for eid in multi_fail_eids.index:
    ent_preds = preds[preds["entity_id"] == eid].sort_values("prediction_timestamp")
    times = fail_events[fail_events["entity_id"] == eid]["time_sec"].tolist()
    print(f"\nEntity {eid} (Failures at {times}):")
    for _, r in ent_preds.iterrows():
        t_pred = r["prediction_timestamp"]
        label = int(r["label"])
        ttf = r["time_to_failure"]
        print(f"  t_pred={t_pred:8.1f}, label={label}, ttf={ttf}")
