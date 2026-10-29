"""Phase 6: Comprehensive Feasibility Audit Script.

Investigates:
1. Exact history-window mathematics (70 min endpoint span vs 75 min collection interval).
2. For all 100 frozen failure entities: exact lifecycle events, failure timestamps,
   telemetry steps before first failure, retry attempts, and why zero pre-failure windows formed.
3. Full-trace census of all failing entities across Google Borg Trace shards:
   Distribution of pre-failure history (1 step, 2 steps, ..., 15+ steps).
   Proof/evidence of whether any unused entities with >= 15 pre-failure steps exist.
4. Metric audit of Phase 5 predictions CSV: exact replication of all window and entity metrics,
   clarification of entity FAR definition, and prevalence adjustment recalculation.
5. Feasibility of alternative designs:
   - Shorter sequence lengths L in {3, 5, 8, 10, 12, 15}
   - Sampling cadences (e.g. 60s, 300s)
   - Alternative datasets (Alibaba Cluster Trace 2018 / 2021)
   - Alternative target horizons
"""

import sys
import json
import glob
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.schema import ClusterEventTypeV2, FailureLabel

EVENT_COLS = [
    "timestamp", "missing_info", "job_id", "task_index", "machine_id",
    "event_type", "user_name", "scheduling_class", "priority",
    "req_cpu", "req_ram", "req_disk", "diff_machine_restriction"
]

USAGE_COLS = [
    "start_time", "end_time", "job_id", "task_index", "machine_id",
    "cpu_rate", "canonical_memory", "assigned_memory", "unmapped_page_cache",
    "total_page_cache", "max_memory", "disk_io_time", "local_disk_space",
    "max_cpu_rate", "max_disk_io_time", "cycles_per_instruction",
    "memory_accesses_per_instruction", "sample_rate", "aggregation_type",
    "sampled_cpu_usage"
]


def audit_frozen_failure_entities(protocol_path: str = "results/phase4/confirmatory_protocol.json"):
    print("=" * 80)
    print("TASK 1: AUDITING FROZEN FAILURE ENTITIES (N=100)")
    print("=" * 80)

    with open(protocol_path) as f:
        proto = json.load(f)

    fail_eids = proto["frozen_entity_ids"]["failure_entities"]
    norm_eids = proto["frozen_entity_ids"]["normal_entities"]

    # 1. Ingest events for frozen fail entities
    event_shards = proto["source_event_shards"]
    event_dfs = []
    for s in event_shards:
        p = Path(s)
        if p.exists():
            df = pd.read_csv(p, header=None, names=EVENT_COLS)
            df["entity_id"] = df["job_id"].astype(str) + "_" + df["task_index"].astype(str)
            m = df[df["entity_id"].isin(fail_eids)]
            if not m.empty:
                event_dfs.append(m)

    events_df = pd.concat(event_dfs, ignore_index=True)
    events_df["time_sec"] = events_df["timestamp"] / 1e6

    # 2. Ingest usage for frozen fail entities
    usage_shards = proto["source_usage_shards"]
    usage_chunks = []
    for s in usage_shards:
        p = Path(s)
        if p.exists():
            for chunk in pd.read_csv(p, header=None, names=USAGE_COLS, chunksize=250000):
                chunk["entity_id"] = chunk["job_id"].astype(str) + "_" + chunk["task_index"].astype(str)
                m = chunk[chunk["entity_id"].isin(fail_eids)].copy()
                if not m.empty:
                    usage_chunks.append(m)

    usage_df = pd.concat(usage_chunks, ignore_index=True)
    usage_df["start_time_sec"] = usage_df["start_time"] / 1e6
    usage_df["end_time_sec"] = usage_df["end_time"] / 1e6
    usage_df["grid_bin"] = np.ceil(usage_df["end_time_sec"] / 300.0) * 300.0

    print(f"Total event records for 100 fail entities: {len(events_df)}")
    print(f"Total raw telemetry records for 100 fail entities: {len(usage_df)}")

    # Detailed per-entity audit
    entity_audits = []
    for eid in sorted(fail_eids):
        e_events = events_df[events_df["entity_id"] == eid].sort_values("time_sec")
        e_usage = usage_df[usage_df["entity_id"] == eid].sort_values("end_time_sec")

        # Lifecycle events
        event_types = e_events["event_type"].tolist()
        event_times = e_events["time_sec"].tolist()

        fail_evs = e_events[e_events["event_type"] == ClusterEventTypeV2.FAIL]
        has_fail_event = not fail_evs.empty
        first_fail_time = float(fail_evs["time_sec"].min()) if has_fail_event else None
        fail_count = len(fail_evs)

        # Retry / restart detection:
        # A restart occurs if SCHEDULE(1) or SUBMIT(0) appears AFTER a FAIL(3) event
        has_restart_after_fail = False
        restart_event_times = []
        if has_fail_event:
            post_fail_events = e_events[e_events["time_sec"] > first_fail_time]
            restart_evs = post_fail_events[post_fail_events["event_type"].isin([ClusterEventTypeV2.SUBMIT, ClusterEventTypeV2.SCHEDULE])]
            if not restart_evs.empty:
                has_restart_after_fail = True
                restart_event_times = restart_evs["time_sec"].tolist()

        # Telemetry steps
        unique_bins = sorted(e_usage["grid_bin"].unique().tolist())
        n_raw_records = len(e_usage)
        n_grid_bins = len(unique_bins)

        # Steps strictly before first fail event
        bins_before_fail = [b for b in unique_bins if b < first_fail_time] if first_fail_time else []
        bins_after_fail = [b for b in unique_bins if b >= first_fail_time] if first_fail_time else []

        n_steps_before_fail = len(bins_before_fail)
        n_steps_after_fail = len(bins_after_fail)

        # Pre-failure history span (seconds)
        if n_steps_before_fail >= 2:
            pre_fail_span_sec = bins_before_fail[-1] - bins_before_fail[0]
        elif n_steps_before_fail == 1:
            pre_fail_span_sec = 0.0
        else:
            pre_fail_span_sec = None

        entity_audits.append({
            "entity_id": eid,
            "has_fail_event": has_fail_event,
            "fail_count": fail_count,
            "first_fail_time_sec": first_fail_time,
            "n_raw_usage_records": n_raw_records,
            "n_grid_bins_total": n_grid_bins,
            "n_steps_before_fail": n_steps_before_fail,
            "n_steps_after_fail": n_steps_after_fail,
            "pre_fail_span_sec": pre_fail_span_sec,
            "has_restart_after_fail": has_restart_after_fail,
            "restart_event_times": restart_event_times,
            "can_form_15step_pre_fail_window": (n_steps_before_fail >= 15),
            "can_form_10step_pre_fail_window": (n_steps_before_fail >= 10),
            "can_form_5step_pre_fail_window": (n_steps_before_fail >= 5),
            "can_form_3step_pre_fail_window": (n_steps_before_fail >= 3),
        })

    audit_df = pd.DataFrame(entity_audits)
    print("\nPre-Failure Step Distribution across the 100 Frozen Failure Entities:")
    step_counts = audit_df["n_steps_before_fail"].value_counts().sort_index()
    for s_val, count in step_counts.items():
        print(f"  - {s_val:2d} steps before failure: {count:2d} entities")

    print(f"\nEntities with >= 15 pre-failure steps: {audit_df['can_form_15step_pre_fail_window'].sum()} / 100")
    print(f"Entities with >= 10 pre-failure steps: {audit_df['can_form_10step_pre_fail_window'].sum()} / 100")
    print(f"Entities with >=  5 pre-failure steps: {audit_df['can_form_5step_pre_fail_window'].sum()} / 100")
    print(f"Entities with >=  3 pre-failure steps: {audit_df['can_form_3step_pre_fail_window'].sum()} / 100")
    print(f"Entities exhibiting task restarts after failure: {audit_df['has_restart_after_fail'].sum()} / 100")

    return audit_df


def audit_full_trace_failure_census():
    print("\n" + "=" * 80)
    print("TASK 1.2: FULL-TRACE CENSUS OF ALL FAILING ENTITIES")
    print("=" * 80)

    # Ingest ALL events from all 3 shards
    event_files = sorted(glob.glob("data/raw/sample_task_events*.csv.gz"))
    usage_files = sorted(glob.glob("data/raw/sample_task_usage*.csv.gz"))

    print(f"Event files: {event_files}")
    print(f"Usage files: {usage_files}")

    # Load dev partitions
    tr_meta = pd.read_csv("data/processed/train_metadata.csv")
    va_meta = pd.read_csv("data/processed/val_metadata.csv")
    te_meta = pd.read_csv("data/processed/test_metadata.csv")
    dev_entities = set(tr_meta["entity_id"]) | set(va_meta["entity_id"]) | set(te_meta["entity_id"])
    dev_fail_windows = pd.concat([tr_meta, va_meta, te_meta])[lambda df: df["label"] > 0]
    dev_failing_entities = set(dev_fail_windows["entity_id"].unique())

    print(f"Development entities total: {len(dev_entities)} (Failing entities with valid windows: {len(dev_failing_entities)})")

    # Ingest all FAIL events
    ev_dfs = []
    for f in event_files:
        df = pd.read_csv(f, header=None, usecols=[0, 2, 3, 5])
        df["entity_id"] = df[2].astype(str) + "_" + df[3].astype(str)
        df["time_sec"] = df[0] / 1e6
        df["event_type"] = df[5]
        fails = df[df["event_type"] == ClusterEventTypeV2.FAIL]
        if not fails.empty:
            ev_dfs.append(fails[["entity_id", "time_sec", "event_type"]])

    all_fails_df = pd.concat(ev_dfs, ignore_index=True)
    min_fail_times = all_fails_df.groupby("entity_id")["time_sec"].min().to_dict()
    total_failing_entities_in_trace = len(min_fail_times)
    print(f"Total unique entities with FAIL events in raw event shards: {total_failing_entities_in_trace:,}")

    # Scan telemetry shards for ALL failing entities
    fail_telemetry_bins = {}
    for uf in usage_files:
        for chunk in pd.read_csv(uf, header=None, usecols=[1, 2, 3], chunksize=250000):
            chunk["entity_id"] = chunk[2].astype(str) + "_" + chunk[3].astype(str)
            # Filter to failing entities
            matched = chunk[chunk["entity_id"].isin(min_fail_times)]
            if not matched.empty:
                matched["grid_bin"] = np.ceil(matched[1] / 1e6 / 300.0) * 300.0
                for eid, grp in matched.groupby("entity_id"):
                    if eid not in fail_telemetry_bins:
                        fail_telemetry_bins[eid] = set()
                    fail_telemetry_bins[eid].update(grp["grid_bin"].unique())

    print(f"Total failing entities found with telemetry in raw usage shards: {len(fail_telemetry_bins):,}")

    census_records = []
    for eid, bins in fail_telemetry_bins.items():
        f_time = min_fail_times[eid]
        sorted_bins = sorted(bins)
        bins_before = [b for b in sorted_bins if b < f_time]
        n_before = len(bins_before)
        is_dev = eid in dev_entities

        census_records.append({
            "entity_id": eid,
            "total_bins": len(sorted_bins),
            "bins_before_fail": n_before,
            "in_dev_split": is_dev,
            "in_dev_failing": eid in dev_failing_entities,
            "first_fail_time": f_time,
        })

    census_df = pd.DataFrame(census_records)

    print("\nCensus of ALL Failing Entities with Telemetry:")
    print("Pre-failure grid bin distribution:")
    hist_all = census_df["bins_before_fail"].value_counts().sort_index()
    for n_b, cnt in hist_all.items():
        in_dev_cnt = (census_df[census_df["bins_before_fail"] == n_b]["in_dev_split"]).sum()
        unused_cnt = cnt - in_dev_cnt
        print(f"  - {n_b:2d} bins before fail: Total={cnt:3d} (Dev={in_dev_cnt:3d}, Unused={unused_cnt:3d})")

    # Critical Question: How many entities with >= 15 bins before failure exist that were NOT used in dev?
    unused_ge15 = census_df[(census_df["bins_before_fail"] >= 15) & (~census_df["in_dev_split"])]
    print(f"\nCRITICAL VERIFICATION: Unused failing entities with >= 15 bins before failure = {len(unused_ge15)}")

    # Check for >= 10, >= 5, >= 3
    print(f"Unused failing entities with >= 14 bins before failure: {len(census_df[(census_df['bins_before_fail'] >= 14) & (~census_df['in_dev_split'])])}")
    print(f"Unused failing entities with >= 10 bins before failure: {len(census_df[(census_df['bins_before_fail'] >= 10) & (~census_df['in_dev_split'])])}")
    print(f"Unused failing entities with >=  5 bins before failure: {len(census_df[(census_df['bins_before_fail'] >=  5) & (~census_df['in_dev_split'])])}")
    print(f"Unused failing entities with >=  3 bins before failure: {len(census_df[(census_df['bins_before_fail'] >=  3) & (~census_df['in_dev_split'])])}")

    return census_df


def audit_phase5_metrics_and_wording():
    print("\n" + "=" * 80)
    print("TASK 2: AUDITING PHASE 5 METRICS, FORMULAS, AND WORDING")
    print("=" * 80)

    pred_path = PROJECT_ROOT / "results" / "phase5" / "confirmatory_predictions.csv"
    preds_df = pd.read_csv(pred_path)
    print(f"Loaded {len(preds_df):,} prediction rows across {preds_df['entity_id'].nunique()} entities")

    models = {
        "Random Forest": "rf",
        "Temporal LSTM": "lstm",
        "Compact Temporal Transformer": "trans",
        "Representation Hybrid": "hybrid",
    }

    n_windows = len(preds_df)
    n_entities = preds_df["entity_id"].nunique()

    recomputed = {}
    for name, prefix in models.items():
        y_p = preds_df[f"{prefix}_pred"].values
        # Window FPR: fraction of normal windows (all are normal) receiving alert y_p in {1, 2}
        n_fp_windows = int((y_p > 0).sum())
        win_fpr_pct = (n_fp_windows / n_windows) * 100.0

        # Entity FAR: fraction of pure-normal entities receiving at least one alert across their windows
        fa_entities = sum(
            (preds_df[preds_df["entity_id"] == eid][f"{prefix}_pred"] > 0).any()
            for eid in preds_df["entity_id"].unique()
        )
        entity_far_pct = (fa_entities / n_entities) * 100.0

        # Accuracy
        acc = (y_p == 0).mean()

        # Macro-F1 across 3 classes when true labels are all 0:
        # P0 = (y_p == 0).sum() / (y_p == 0).sum() = 1.0
        # R0 = (y_p == 0).sum() / n_windows = acc
        # F1_0 = 2 * acc / (1 + acc)
        # F1_1 = 0.0, F1_2 = 0.0
        # Macro-F1 = F1_0 / 3.0
        f1_0 = (2.0 * acc) / (1.0 + acc) if (1.0 + acc) > 0 else 0.0
        macro_f1 = f1_0 / 3.0

        recomputed[name] = {
            "accuracy": float(acc),
            "macro_f1": float(macro_f1),
            "window_fpr_pct": float(win_fpr_pct),
            "fp_windows_count": n_fp_windows,
            "entity_far_pct": float(entity_far_pct),
            "fa_entities_count": fa_entities,
        }
        print(f"\n[{name}]")
        print(f"  - Accuracy: {acc:.4f}, Macro-F1: {macro_f1:.4f}")
        print(f"  - Window FPR: {win_fpr_pct:.2f}% ({n_fp_windows} / {n_windows})")
        print(f"  - Entity FAR: {entity_far_pct:.2f}% ({fa_entities} / {n_entities})")

    return recomputed


def evaluate_alternative_experimental_designs(census_df: pd.DataFrame):
    print("\n" + "=" * 80)
    print("TASK 3: EVALUATING ALTERNATIVE EXPERIMENTAL DESIGNS")
    print("=" * 80)

    # Let's assess available pre-failure entities under different sequence lengths L
    lengths = [3, 5, 8, 10, 12, 14, 15]
    feasibility_by_L = []

    for L in lengths:
        # Entity can produce at least 1 pre-failure window if bins_before_fail >= L
        # In fact, if bins_before_fail >= L, it yields (bins_before_fail - L + 1) windows
        total_eligible = census_df[census_df["bins_before_fail"] >= L]
        dev_eligible = total_eligible[total_eligible["in_dev_split"]]
        unused_eligible = total_eligible[~total_eligible["in_dev_split"]]

        # Pre-failure window potential among unused entities
        total_unused_windows = sum(b - L + 1 for b in unused_eligible["bins_before_fail"])

        feasibility_by_L.append({
            "sequence_length_L": L,
            "endpoint_duration_min": (L - 1) * 5.0,
            "collection_duration_min": L * 5.0,
            "total_trace_failing_entities": len(total_eligible),
            "dev_failing_entities": len(dev_eligible),
            "unused_failing_entities": len(unused_eligible),
            "potential_unused_failure_windows": total_unused_windows,
        })

    f_df = pd.DataFrame(feasibility_by_L)
    print("\nFeasibility by Sequence Length L (Google Borg 2011 Trace):")
    print(f_df.to_string(index=False))

    return f_df


if __name__ == "__main__":
    audit_df = audit_frozen_failure_entities()
    census_df = audit_full_trace_failure_census()
    recomp_metrics = audit_phase5_metrics_and_wording()
    feasibility_df = evaluate_alternative_experimental_designs(census_df)
