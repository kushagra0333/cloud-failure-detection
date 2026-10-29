"""Comprehensive breakdown of the 296 failure-cohort normal windows and all 5,511 windows."""

import json
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load protocol
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
    matched = df[df["entity_id"].isin(fail_eids | pure_norm_eids)]
    if not matched.empty:
        event_dfs.append(matched)

events = pd.concat(event_dfs, ignore_index=True)
events["time_sec"] = events["timestamp"] / 1e6
fail_events = events[events["event_type"] == 3].copy()
finish_events = events[events["event_type"] == 4].copy()
kill_events = events[events["event_type"] == 5].copy()

fail_times_by_eid = fail_events.groupby("entity_id")["time_sec"].apply(list).to_dict()

# Load predictions
preds = pd.read_csv(PROJECT_ROOT / "results" / "phase6" / "confirmatory_predictions_l5.csv")

# Focus on failure cohort normal windows
fail_norm = preds[(preds["entity_id"].isin(fail_eids)) & (preds["label"] == 0)].copy()
print(f"Total failure-cohort normal windows: {len(fail_norm)} across {fail_norm['entity_id'].nunique()} entities")

# Classify each of the 296 windows
records = []
for idx, r in fail_norm.iterrows():
    eid = r["entity_id"]
    t_pred = float(r["prediction_timestamp"])
    stored_ttf = r["time_to_failure"]
    e_fails = sorted(fail_times_by_eid.get(eid, []))
    
    prev_fails = [ft for ft in e_fails if ft < t_pred]
    next_fails = [ft for ft in e_fails if ft > t_pred]
    
    prev_fail = prev_fails[-1] if prev_fails else None
    next_fail = next_fails[0] if next_fails else None
    
    # 5-step history span: [t_pred - 1200, t_pred]
    t_start = t_pred - 1200.0
    
    history_crosses = (prev_fail is not None and prev_fail >= t_start)
    entirely_post_failure = (prev_fail is not None and prev_fail < t_start)
    
    if next_fail is not None:
        tau = next_fail - t_pred
        category = "PRE_FAILURE_NORMAL" if tau > 1800.0 else "PRE_FAILURE_WITHIN_HORIZON_MISLABELED"
    else:
        tau = None
        category = "POST_FAILURE_NO_SUBSEQUENT_FAIL"
    
    records.append({
        "window_idx": int(r["window_idx"]),
        "entity_id": eid,
        "prediction_timestamp": t_pred,
        "stored_ttf": stored_ttf,
        "prev_fail": prev_fail,
        "next_fail": next_fail,
        "tau": tau,
        "category": category,
        "history_crosses": history_crosses,
        "entirely_post_failure": entirely_post_failure,
    })

df_rec = pd.DataFrame(records)

print("\nBreakdown of 296 failure-cohort normal windows:")
print(df_rec["category"].value_counts())

print("\nHistory position relative to prior failure for POST_FAILURE windows:")
post_df = df_rec[df_rec["category"] == "POST_FAILURE_NO_SUBSEQUENT_FAIL"]
print("Total post-failure windows:", len(post_df))
print("History crosses prior failure (t_start <= prev_fail <= t_pred):", post_df["history_crosses"].sum())
print("History entirely after prior failure (prev_fail < t_start):", post_df["entirely_post_failure"].sum())

print("\nPre-failure normal windows (category == PRE_FAILURE_NORMAL):")
pre_df = df_rec[df_rec["category"] == "PRE_FAILURE_NORMAL"]
print("Total pre-failure normal windows:", len(pre_df))
print("Pre-failure min tau:", pre_df["tau"].min(), "max tau:", pre_df["tau"].max())
print("Do any pre-failure normal windows have a prior failure?", (pre_df["prev_fail"].notna()).sum())
if (pre_df["prev_fail"].notna()).sum() > 0:
    print("Pre-failure with prior failure details:")
    print(pre_df[pre_df["prev_fail"].notna()][["entity_id", "prediction_timestamp", "prev_fail", "next_fail", "tau", "history_crosses"]])
