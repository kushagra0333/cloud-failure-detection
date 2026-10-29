"""Comprehensive audit of all 5,511 confirmatory prediction windows."""

import json
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 1. Load protocol
with open(PROJECT_ROOT / "results" / "phase6" / "protocol_l5_amendment_2_1.json") as f:
    protocol = json.load(f)

fail_eids = set(protocol["frozen_entity_ids"]["primary_failure_entities"])
pure_norm_eids = set(protocol["frozen_entity_ids"]["primary_normal_entities_evaluable"])

# 2. Load raw events
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
evict_events = events[events["event_type"] == 2].copy()

fail_times_by_eid = fail_events.groupby("entity_id")["time_sec"].apply(list).to_dict()
finish_times_by_eid = finish_events.groupby("entity_id")["time_sec"].apply(list).to_dict()
kill_times_by_eid = kill_events.groupby("entity_id")["time_sec"].apply(list).to_dict()
evict_times_by_eid = evict_events.groupby("entity_id")["time_sec"].apply(list).to_dict()

# 3. Load predictions
preds = pd.read_csv(PROJECT_ROOT / "results" / "phase6" / "confirmatory_predictions_l5.csv")

# Audit all 5,511 windows
print(f"Total prediction rows: {len(preds)}")

# Group windows by label
crit_windows = preds[preds["label"] == 2].copy()
warn_windows = preds[preds["label"] == 1].copy()
norm_windows = preds[preds["label"] == 0].copy()
pure_norm_windows = norm_windows[norm_windows["entity_id"].isin(pure_norm_eids)].copy()
fail_norm_windows = norm_windows[norm_windows["entity_id"].isin(fail_eids)].copy()

print(f"Critical windows: {len(crit_windows)}")
print(f"Warning windows: {len(warn_windows)}")
print(f"Normal windows: {len(norm_windows)} ({len(pure_norm_windows)} pure-norm, {len(fail_norm_windows)} fail-cohort-norm)")

# Check 1: Critical Horizon (0 < tau <= 900s)
crit_taus = []
crit_discrepancies = []
for idx, r in crit_windows.iterrows():
    eid = r["entity_id"]
    t_pred = float(r["prediction_timestamp"])
    e_fails = sorted(fail_times_by_eid.get(eid, []))
    future_fails = [ft for ft in e_fails if ft > t_pred]
    if not future_fails:
        crit_discrepancies.append((eid, t_pred, "NO_FUTURE_FAIL"))
    else:
        tau = future_fails[0] - t_pred
        crit_taus.append(tau)
        if not (0.0 < tau <= 900.0):
            crit_discrepancies.append((eid, t_pred, f"TAU_OUT_OF_BOUNDS_{tau}"))

print(f"\n[Check 1] Critical Horizon Check (N={len(crit_windows)}):")
print(f"  - Min tau: {min(crit_taus):.4f}s, Max tau: {max(crit_taus):.4f}s")
print(f"  - Discrepancies: {len(crit_discrepancies)}")

# Check 2: Warning Horizon (900s < tau <= 1800s)
warn_taus = []
warn_discrepancies = []
for idx, r in warn_windows.iterrows():
    eid = r["entity_id"]
    t_pred = float(r["prediction_timestamp"])
    e_fails = sorted(fail_times_by_eid.get(eid, []))
    future_fails = [ft for ft in e_fails if ft > t_pred]
    if not future_fails:
        warn_discrepancies.append((eid, t_pred, "NO_FUTURE_FAIL"))
    else:
        tau = future_fails[0] - t_pred
        warn_taus.append(tau)
        if not (900.0 < tau <= 1800.0):
            warn_discrepancies.append((eid, t_pred, f"TAU_OUT_OF_BOUNDS_{tau}"))

print(f"\n[Check 2] Warning Horizon Check (N={len(warn_windows)}):")
print(f"  - Min tau: {min(warn_taus):.4f}s, Max tau: {max(warn_taus):.4f}s")
print(f"  - Discrepancies: {len(warn_discrepancies)}")

# Check 3: Pure-Normal windows (N=4,515)
pure_norm_discrepancies = []
for idx, r in pure_norm_windows.iterrows():
    eid = r["entity_id"]
    t_pred = float(r["prediction_timestamp"])
    # Any fail events?
    if eid in fail_times_by_eid:
        pure_norm_discrepancies.append((eid, t_pred, "FAIL_EVENT_PRESENT"))
print(f"\n[Check 3] Pure-Normal Windows Check (N={len(pure_norm_windows)}):")
print(f"  - Discrepancies with failure events: {len(pure_norm_discrepancies)}")

# Check 4: Pre-failure Normal Windows among failure cohort (N=147)
pre_fail_norm = []
post_fail_norm = []
for idx, r in fail_norm_windows.iterrows():
    eid = r["entity_id"]
    t_pred = float(r["prediction_timestamp"])
    e_fails = sorted(fail_times_by_eid.get(eid, []))
    future_fails = [ft for ft in e_fails if ft > t_pred]
    prev_fails = [ft for ft in e_fails if ft < t_pred]
    if future_fails:
        tau = future_fails[0] - t_pred
        pre_fail_norm.append((eid, t_pred, tau, prev_fails))
    else:
        post_fail_norm.append((eid, t_pred, prev_fails))

print(f"\n[Check 4] Failure-Cohort Normal Windows Check (N={len(fail_norm_windows)}):")
print(f"  - Pre-failure normal windows: {len(pre_fail_norm)}")
print(f"    * Min tau: {min(t[2] for t in pre_fail_norm):.4f}s, Max tau: {max(t[2] for t in pre_fail_norm):.4f}s")
print(f"  - Post-failure normal windows (no future fail): {len(post_fail_norm)}")
