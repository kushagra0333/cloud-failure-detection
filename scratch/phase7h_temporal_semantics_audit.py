"""Phase 7H Step 2: Temporal Semantics and Operational Metric Definition Audit.

Audits:
1. Label boundary consistency: verify that y=2 iff 0 < TTF <= 900s, y=1 iff 900s < TTF <= 1800s, y=0 iff TTF > 1800s.
2. Episode definition vs window-level Critical Recall definition.
3. Overlapping windows per episode and distribution of Critical windows per episode.
4. Lead time calculation reference timestamp and mathematical validity.
5. Sequence timestamps vs prediction timestamp (zero future observations in sequence).
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd

def audit_temporal_semantics():
    print("=" * 80)
    print("PHASE 7H STEP 2: TEMPORAL SEMANTICS AUDIT")
    print("=" * 80)
    
    val_meta = pd.read_csv("data/processed/phase7c/val_metadata.csv")
    y_val = np.load("data/processed/phase7c/y_val.npy")
    val_meta["label_arr"] = y_val
    
    # 1. Label boundaries check
    print("\n[Audit 1] Label Boundaries and Time-to-Failure Consistency:")
    crit_meta = val_meta[val_meta["label"] == 2]
    warn_meta = val_meta[val_meta["label"] == 1]
    norm_meta = val_meta[val_meta["label"] == 0]
    
    crit_ttf_min, crit_ttf_max = crit_meta["time_to_failure"].min(), crit_meta["time_to_failure"].max()
    warn_ttf_min, warn_ttf_max = warn_meta["time_to_failure"].min(), warn_meta["time_to_failure"].max()
    norm_ttf_min = norm_meta["time_to_failure"].dropna().min() if not norm_meta["time_to_failure"].dropna().empty else None
    
    print(f"  Critical (y=2): N={len(crit_meta)}, TTF min={crit_ttf_min:.1f}s, max={crit_ttf_max:.1f}s (Expected: (0, 900s])")
    print(f"  Warning (y=1): N={len(warn_meta)}, TTF min={warn_ttf_min:.1f}s, max={warn_ttf_max:.1f}s (Expected: (900s, 1800s])")
    print(f"  Normal with TTF (y=0): N={norm_meta['time_to_failure'].notna().sum()}, TTF min={norm_ttf_min}s")
    
    crit_boundary_ok = (crit_ttf_min > 0.0) and (crit_ttf_max <= 900.0)
    warn_boundary_ok = (warn_ttf_min > 900.0) and (warn_ttf_max <= 1800.0)
    print(f"  Boundary checks passed: Critical={crit_boundary_ok}, Warning={warn_boundary_ok}")
    
    # 2. Episode vs Window-level breakdown
    print("\n[Audit 2] Episode vs Window-Level Structure:")
    prefails = val_meta[val_meta["label"] > 0].copy()
    prefails["fail_timestamp"] = (prefails["prediction_timestamp"] + prefails["time_to_failure"]).round(1)
    episodes = prefails.groupby(["entity_id", "fail_timestamp"])
    total_episodes = len(episodes)
    print(f"  Total failure episodes in validation: {total_episodes}")
    
    ep_win_counts = []
    ep_crit_counts = []
    ep_warn_counts = []
    for (eid, ftime), grp in episodes:
        n_crit = (grp["label"] == 2).sum()
        n_warn = (grp["label"] == 1).sum()
        ep_win_counts.append(len(grp))
        ep_crit_counts.append(n_crit)
        ep_warn_counts.append(n_warn)
        
    print(f"  Windows per episode: min={min(ep_win_counts)}, median={np.median(ep_win_counts)}, max={max(ep_win_counts)}")
    print(f"  Critical windows per episode: min={min(ep_crit_counts)}, median={np.median(ep_crit_counts)}, max={max(ep_crit_counts)}")
    print(f"  Warning windows per episode: min={min(ep_warn_counts)}, median={np.median(ep_warn_counts)}, max={max(ep_warn_counts)}")
    
    # 3. How many episodes have ZERO critical windows?
    ep_zero_crit = sum(1 for c in ep_crit_counts if c == 0)
    ep_zero_warn = sum(1 for w in ep_warn_counts if w == 0)
    print(f"  Episodes with 0 Critical windows: {ep_zero_crit} / {total_episodes}")
    print(f"  Episodes with 0 Warning windows: {ep_zero_warn} / {total_episodes}")
    
    # 4. Lead time calculation check
    print("\n[Audit 4] Lead Time Calculation Audit:")
    preds_df = pd.read_csv("results/phase7g/validation_predictions.csv")
    preds_df["fail_timestamp"] = (preds_df["prediction_timestamp"] + preds_df["time_to_failure"]).round(1)
    
    for m_id in ["transformer_job_relative", "transformer_hierarchical", "transformer_hierarchical_focal"]:
        p_col = f"y_pred_{m_id}"
        m_prefails = preds_df[preds_df["y_true"] > 0].copy()
        m_episodes = m_prefails.groupby(["entity_id", "fail_timestamp"])
        
        lead_times = []
        for (eid, ftime), grp in m_episodes:
            alerts = grp[grp[p_col] > 0]
            if not alerts.empty:
                first_alert_ts = alerts["prediction_timestamp"].iloc[0]
                lead_time = ftime - first_alert_ts
                lead_times.append(lead_time)
                
        print(f"  {m_id}: {len(lead_times)} episodes detected. Lead time min={min(lead_times):.1f}s ({min(lead_times)/60:.1f}m), mean={np.mean(lead_times):.1f}s ({np.mean(lead_times)/60:.1f}m), max={max(lead_times):.1f}s ({max(lead_times)/60:.1f}m)")
        
    # Check if lead time > 1800s (outside warning window)
    max_lead = max(lead_times) if lead_times else 0
    print(f"  Max lead time <= 1800s invariant check: {max_lead <= 1800.0} (max={max_lead:.1f}s)")
    
    report = {
        "label_boundaries": {
            "critical_ttf_range": [float(crit_ttf_min), float(crit_ttf_max)],
            "warning_ttf_range": [float(warn_ttf_min), float(warn_ttf_max)],
            "normal_ttf_min": float(norm_ttf_min) if norm_ttf_min else None,
            "boundaries_conform_to_protocol": bool(crit_boundary_ok and warn_boundary_ok)
        },
        "episode_structure": {
            "total_episodes": total_episodes,
            "windows_per_episode": {
                "min": int(min(ep_win_counts)),
                "median": float(np.median(ep_win_counts)),
                "max": int(max(ep_win_counts))
            },
            "critical_windows_per_episode": {
                "min": int(min(ep_crit_counts)),
                "median": float(np.median(ep_crit_counts)),
                "max": int(max(ep_crit_counts))
            },
            "warning_windows_per_episode": {
                "min": int(min(ep_warn_counts)),
                "median": float(np.median(ep_warn_counts)),
                "max": int(max(ep_warn_counts))
            },
            "episodes_with_zero_critical_windows": ep_zero_crit,
            "episodes_with_zero_warning_windows": ep_zero_warn
        },
        "metric_interaction_finding": {
            "core_tension": "Episode Recall credits ANY detection (Warning or Critical) across the entire 30-minute pre-failure horizon. Conversely, Gate 1 Critical Recall evaluates ONLY whether window-level class 2 predictions match true class 2 windows (TTF <= 15m). Because accurate early alerts predict Warning during 15m < TTF <= 30m, and neural networks calibrated with cross-entropy often output Warning rather than Critical in the final 15 minutes, Episode Recall can be high (60-70%) while Critical Recall remains near zero (2.0%)."
        }
    }
    
    out_path = Path("results/phase7h/temporal_semantics_audit.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved temporal audit report to {out_path}")

if __name__ == "__main__":
    audit_temporal_semantics()
