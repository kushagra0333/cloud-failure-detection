"""Scratch script for Phase 6.8 Confirmatory Cohort Validity Audit.
Inspects all 5,511 confirmatory prediction windows against raw Google cluster trace events.
"""

import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 1. Load predictions
preds_path = PROJECT_ROOT / "results" / "phase6" / "confirmatory_predictions_l5.csv"
preds = pd.read_csv(preds_path)
print(f"Total prediction rows: {len(preds)}")
print(f"Entities in predictions: {preds['entity_id'].nunique()}")

# 2. Load protocol amendment
with open(PROJECT_ROOT / "results" / "phase6" / "protocol_l5_amendment_2_1.json") as f:
    protocol = json.load(f)

fail_eids = set(protocol["frozen_entity_ids"]["primary_failure_entities"])
pure_norm_eids = set(protocol["frozen_entity_ids"]["primary_normal_entities_evaluable"])

print(f"Protocol primary failure entities: {len(fail_eids)}")
print(f"Protocol pure-normal evaluable entities: {len(pure_norm_eids)}")

# 3. Load raw events
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
    matched = df[df["entity_id"].isin(fail_eids | pure_norm_eids)]
    if not matched.empty:
        event_dfs.append(matched)

events_all = pd.concat(event_dfs, ignore_index=True)
events_all["time_sec"] = events_all["timestamp"] / 1e6
print(f"Loaded {len(events_all)} events for cohort entities")

# Event map (Borg 2011: FAIL is event_type == 3)
fail_events = events_all[events_all["event_type"] == 3].sort_values(["entity_id", "time_sec"])
print(f"Total FAIL events in cohort: {len(fail_events)} across {fail_events['entity_id'].nunique()} entities")

# Check if any pure-normal entity has FAIL events
pure_norm_fails = fail_events[fail_events["entity_id"].isin(pure_norm_eids)]
print(f"FAIL events among pure-normal entities: {len(pure_norm_fails)}")

# Check FAIL events among failure entities
fail_ent_fails = fail_events[fail_events["entity_id"].isin(fail_eids)]
print(f"FAIL events among primary failure entities: {len(fail_ent_fails)} across {fail_ent_fails['entity_id'].nunique()} entities")

# Group fail times per entity
fail_times_by_eid = {}
for eid, g in fail_ent_fails.groupby("entity_id"):
    fail_times_by_eid[eid] = sorted(g["time_sec"].tolist())

# Now let's audit every window
results = []
for idx, row in preds.iterrows():
    eid = row["entity_id"]
    t_pred = float(row["prediction_timestamp"])
    stored_label = int(row["label"])
    stored_ttf = float(row["time_to_failure"]) if pd.notna(row["time_to_failure"]) else None
    
    # Is it pure-normal or failure entity?
    is_pure_norm = eid in pure_norm_eids
    is_fail_cohort = eid in fail_eids
    
    e_fails = fail_times_by_eid.get(eid, [])
    
    # Previous failures (strictly before t_pred)
    prev_fails = [ft for ft in e_fails if ft < t_pred]
    # Future failures (strictly after t_pred)
    next_fails = [ft for ft in e_fails if ft > t_pred]
    # Exact hits at t_pred
    exact_fails = [ft for ft in e_fails if ft == t_pred]
    
    prev_fail_t = prev_fails[-1] if prev_fails else None
    next_fail_t = next_fails[0] if next_fails else None
    
    # Recomputed time to next failure
    recomputed_ttf = (next_fail_t - t_pred) if next_fail_t is not None else None
    
    # Determine temporal relation
    # Check 5-step input history:
    # History starts at t_pred - 4 * 300 = t_pred - 1200
    t_start = t_pred - 1200.0
    history_crosses_fail = False
    if prev_fail_t is not None and prev_fail_t >= t_start:
        history_crosses_fail = True
    
    # Post failure window: t_pred > a prior failure
    is_post_failure = len(prev_fails) > 0
    
    results.append({
        "window_idx": int(row["window_idx"]),
        "entity_id": eid,
        "is_pure_normal": is_pure_norm,
        "is_fail_cohort": is_fail_cohort,
        "prediction_timestamp": t_pred,
        "stored_label": stored_label,
        "stored_ttf": stored_ttf,
        "prev_fail_t": prev_fail_t,
        "next_fail_t": next_fail_t,
        "recomputed_ttf": recomputed_ttf,
        "history_crosses_fail": history_crosses_fail,
        "is_post_failure": is_post_failure,
    })

res_df = pd.DataFrame(results)

print("\n--- SUMMARY OF WINDOWS ---")
print("Stored label counts:")
print(res_df["stored_label"].value_counts())

print("\nPost-failure status by stored label:")
print(pd.crosstab(res_df["stored_label"], res_df["is_post_failure"]))

print("\nPost-failure status among failure-cohort entities only:")
fail_res = res_df[res_df["is_fail_cohort"]]
print(pd.crosstab(fail_res["stored_label"], fail_res["is_post_failure"]))

print("\nHistory crosses failure by stored label:")
print(pd.crosstab(res_df["stored_label"], res_df["history_crosses_fail"]))
