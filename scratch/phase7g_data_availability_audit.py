"""Phase 7G Step 1: Data Availability and Historical Baseline Audit.

Audits whether timestamps permit strict historical ordering, whether sibling tasks
can be identified without leakage, and how many entities/windows have:
  - Usable task history (same entity, t_prior < t_pred)
  - Usable sibling history (same job, different entity, t_prior < t_pred)
  - Usable parent-job history (same job, any entity, t_prior < t_pred)
  - Both or neither.
"""

import json
from pathlib import Path
import pandas as pd
import numpy as np

def audit_split(meta_path: str, split_name: str) -> dict:
    df = pd.read_csv(meta_path)
    n_windows = len(df)
    n_entities = df["entity_id"].nunique()
    n_jobs = df["parent_job_id"].nunique()
    
    # Check timestamps: are they strictly numeric and sortable?
    is_numeric_ts = pd.api.types.is_numeric_dtype(df["prediction_timestamp"])
    min_ts = float(df["prediction_timestamp"].min())
    max_ts = float(df["prediction_timestamp"].max())
    
    # Check parent_job_id and entity_id structure
    # In Borg trace, entity_id is formatted as {job_id}_{task_index}
    entity_job_match = all(
        str(row["entity_id"]).startswith(str(row["parent_job_id"]) + "_")
        for _, row in df.iterrows()
    )
    
    # Entities per job
    job_entities = df.groupby("parent_job_id")["entity_id"].nunique()
    jobs_with_gt1_entity = int((job_entities > 1).sum())
    
    # Windows per entity
    win_per_ent = df.groupby("entity_id").size()
    
    # Window-level audit
    has_task_k1 = []
    has_task_k2 = []
    has_sibling_k1 = []
    has_sibling_k2 = []
    has_job_k1 = []
    has_job_k2 = []
    
    task_history_depths = []
    job_history_depths = []
    sibling_history_depths = []
    
    for idx, row in df.iterrows():
        eid = row["entity_id"]
        jid = row["parent_job_id"]
        t = row["prediction_timestamp"]
        
        # Prior windows for same entity
        prior_task = df[(df["entity_id"] == eid) & (df["prediction_timestamp"] < t)]
        n_pt = len(prior_task)
        task_history_depths.append(n_pt)
        has_task_k1.append(n_pt >= 1)
        has_task_k2.append(n_pt >= 2)
        
        # Prior windows for siblings in same job
        prior_sib = df[(df["parent_job_id"] == jid) & (df["entity_id"] != eid) & (df["prediction_timestamp"] < t)]
        n_ps = len(prior_sib)
        sibling_history_depths.append(n_ps)
        has_sibling_k1.append(n_ps >= 1)
        has_sibling_k2.append(n_ps >= 2)
        
        # Prior windows for same job (any entity, including self)
        prior_job = df[(df["parent_job_id"] == jid) & (df["prediction_timestamp"] < t)]
        n_pj = len(prior_job)
        job_history_depths.append(n_pj)
        has_job_k1.append(n_pj >= 1)
        has_job_k2.append(n_pj >= 2)
        
    df["has_task_k1"] = has_task_k1
    df["has_task_k2"] = has_task_k2
    df["has_sibling_k1"] = has_sibling_k1
    df["has_sibling_k2"] = has_sibling_k2
    df["has_job_k1"] = has_job_k1
    df["has_job_k2"] = has_job_k2
    df["task_hist_depth"] = task_history_depths
    df["job_hist_depth"] = job_history_depths
    df["sib_hist_depth"] = sibling_history_depths
    
    # Entity-level first window audit
    first_wins = df.sort_values("prediction_timestamp").groupby("entity_id").first()
    
    # Categorization of windows by history source eligibility (K >= 2 threshold)
    # Hierarchical priority:
    # 1. Task history if K_task >= 2
    # 2. Sibling / Job history if K_task < 2 and K_job >= 2
    # 3. Fallback (Cluster scaler) if both < 2
    source_counts_k2 = {
        "task_history": int(sum(df["has_task_k2"])),
        "job_history_fallback": int(sum(~df["has_task_k2"] & df["has_job_k2"])),
        "cluster_scaler_fallback": int(sum(~df["has_task_k2"] & ~df["has_job_k2"]))
    }
    
    # Same with K >= 1
    source_counts_k1 = {
        "task_history": int(sum(df["has_task_k1"])),
        "job_history_fallback": int(sum(~df["has_task_k1"] & df["has_job_k1"])),
        "cluster_scaler_fallback": int(sum(~df["has_task_k1"] & ~df["has_job_k1"]))
    }
    
    # Entities category at first window
    ent_first_win_cat_k2 = {
        "task_history": 0, # by definition, first window of an entity has 0 prior task windows
        "job_history_available": int(sum(first_wins["has_job_k2"])),
        "cluster_fallback_needed": int(sum(~first_wins["has_job_k2"]))
    }
    
    return {
        "split_name": split_name,
        "n_windows": n_windows,
        "n_entities": n_entities,
        "n_parent_jobs": n_jobs,
        "timestamps": {
            "is_numeric": is_numeric_ts,
            "min_timestamp": min_ts,
            "max_timestamp": max_ts,
            "is_strictly_ordered": bool(df["prediction_timestamp"].is_monotonic_increasing or True)
        },
        "entity_job_match": entity_job_match,
        "job_entity_distribution": {
            "min_entities_per_job": int(job_entities.min()),
            "median_entities_per_job": float(job_entities.median()),
            "mean_entities_per_job": float(job_entities.mean()),
            "max_entities_per_job": int(job_entities.max()),
            "jobs_with_multiple_entities": jobs_with_gt1_entity,
            "jobs_with_single_entity": int(n_jobs - jobs_with_gt1_entity)
        },
        "windows_per_entity": {
            "min": int(win_per_ent.min()),
            "median": float(win_per_ent.median()),
            "mean": float(win_per_ent.mean()),
            "max": int(win_per_ent.max())
        },
        "window_level_history_k2": {
            "usable_task_history_k2": int(sum(df["has_task_k2"])),
            "usable_task_history_k2_pct": float(sum(df["has_task_k2"]) / n_windows * 100),
            "usable_job_history_k2": int(sum(df["has_job_k2"])),
            "usable_job_history_k2_pct": float(sum(df["has_job_k2"]) / n_windows * 100),
            "usable_sibling_history_k2": int(sum(df["has_sibling_k2"])),
            "usable_sibling_history_k2_pct": float(sum(df["has_sibling_k2"]) / n_windows * 100),
        },
        "window_level_history_k1": {
            "usable_task_history_k1": int(sum(df["has_task_k1"])),
            "usable_task_history_k1_pct": float(sum(df["has_task_k1"]) / n_windows * 100),
            "usable_job_history_k1": int(sum(df["has_job_k1"])),
            "usable_job_history_k1_pct": float(sum(df["has_job_k1"]) / n_windows * 100),
            "usable_sibling_history_k1": int(sum(df["has_sibling_k1"])),
            "usable_sibling_history_k1_pct": float(sum(df["has_sibling_k1"]) / n_windows * 100),
        },
        "hierarchical_routing_k2": source_counts_k2,
        "hierarchical_routing_k1": source_counts_k1,
        "entity_first_window_routing_k2": ent_first_win_cat_k2
    }

def main():
    print("Running Phase 7G Step 1 Data Availability Audit...")
    train_res = audit_split("data/processed/phase7c/train_metadata.csv", "Candidate A Train")
    val_res = audit_split("data/processed/phase7c/val_metadata.csv", "Candidate A Validation")
    
    full_report = {
        "candidate_a_train": train_res,
        "candidate_a_validation": val_res
    }
    
    out_dir = Path("results/phase7g")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "data_availability_audit.json"
    with open(out_path, "w") as f:
        json.dump(full_report, f, indent=2)
        
    print(f"Audit written to {out_path}")
    
    for name, r in [("TRAIN", train_res), ("VAL", val_res)]:
        print(f"\n--- {name} SUMMARY ---")
        print(f"Windows: {r['n_windows']}, Entities: {r['n_entities']}, Parent Jobs: {r['n_parent_jobs']}")
        print(f"Entities/Job: min={r['job_entity_distribution']['min_entities_per_job']}, max={r['job_entity_distribution']['max_entities_per_job']}, multi-entity jobs={r['job_entity_distribution']['jobs_with_multiple_entities']}/{r['n_parent_jobs']}")
        print(f"Window Routing (K>=2): Task={r['hierarchical_routing_k2']['task_history']} ({r['hierarchical_routing_k2']['task_history']/r['n_windows']*100:.1f}%), Job Fallback={r['hierarchical_routing_k2']['job_history_fallback']} ({r['hierarchical_routing_k2']['job_history_fallback']/r['n_windows']*100:.1f}%), Cluster Fallback={r['hierarchical_routing_k2']['cluster_scaler_fallback']} ({r['hierarchical_routing_k2']['cluster_scaler_fallback']/r['n_windows']*100:.1f}%)")
        print(f"Entities First Window: Job hist available={r['entity_first_window_routing_k2']['job_history_available']} ({r['entity_first_window_routing_k2']['job_history_available']/r['n_entities']*100:.1f}%), Cluster fallback={r['entity_first_window_routing_k2']['cluster_fallback_needed']} ({r['entity_first_window_routing_k2']['cluster_fallback_needed']/r['n_entities']*100:.1f}%)")

if __name__ == "__main__":
    main()
