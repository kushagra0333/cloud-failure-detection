"""Entity-level cluster bootstrap for paired uncertainty reporting (RF vs LSTM).

Implements Phase 3C.1 Methodology Requirements:
- Resamples at the entity level, preserving all sequential observations and episodes belonging to each sampled entity.
- RF and LSTM metrics use identical entity samples in every paired replicate.
- Paired differences are calculated within each replicate as LSTM minus RF.
- Window-level metrics are recomputed from the resampled entity data (not averaged across entities).
- Episode detection uses the correct episode-level denominator and preserves entity clustering.
- Duplicate entity draws are preserved (each drawn entity instance contributes its full sequence and episodes).
- Computes non-parametric 95% empirical percentile confidence intervals [2.5%, 97.5%].
"""

import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, precision_recall_fscore_support

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def run_entity_cluster_bootstrap(
    rf_pred_path: str = "results/phase3a/test_predictions_random_forest.csv",
    lstm_pred_path: str = "results/phase3b/test_predictions_lstm.csv",
    n_replicates: int = 1000,
    random_seed: int = 42,
    output_json: str = "results/phase3c/entity_bootstrap_results.json",
):
    print("=" * 78)
    print("ENTITY-LEVEL CLUSTER BOOTSTRAP UNCERTAINTY ANALYSIS (PHASE 3C.1)")
    print("=" * 78)

    rf_df = pd.read_csv(rf_pred_path)
    lstm_df = pd.read_csv(lstm_pred_path)

    assert len(rf_df) == len(lstm_df) == 1003, "Test set length mismatch"
    assert (rf_df["entity_id"] == lstm_df["entity_id"]).all(), "Entity ID alignment mismatch"
    assert (rf_df["prediction_timestamp"] == lstm_df["prediction_timestamp"]).all(), "Timestamp alignment mismatch"
    assert (rf_df["label"] == lstm_df["label"]).all(), "Label alignment mismatch"

    unique_entities = np.array(sorted(rf_df["entity_id"].unique()))
    n_entities = len(unique_entities)  # 80 entities in test partition
    print(f"Total cluster entities in test partition: {n_entities}")
    print(f"Total sliding windows in test partition : {len(rf_df)}")
    print(f"Number of bootstrap replicates          : {n_replicates}")
    print(f"Random seed                             : {random_seed}")

    # 1. Precompute per-entity arrays for ultra-fast vectorization
    y_by_ent = {}
    rf_pred_by_ent = {}
    lstm_pred_by_ent = {}

    for eid in unique_entities:
        mask = (rf_df["entity_id"] == eid)
        y_by_ent[eid] = rf_df.loc[mask, "label"].values
        rf_pred_by_ent[eid] = rf_df.loc[mask, "predicted_label"].values
        lstm_pred_by_ent[eid] = lstm_df.loc[mask, "predicted_label"].values

    # 2. Precompute per-entity failure episodes
    prefails_rf = rf_df[rf_df["label"] > 0].copy()
    prefails_rf["fail_time"] = (prefails_rf["prediction_timestamp"] + prefails_rf["time_to_failure"]).round(1)

    prefails_lstm = lstm_df[lstm_df["label"] > 0].copy()
    prefails_lstm["fail_time"] = (prefails_lstm["prediction_timestamp"] + prefails_lstm["time_to_failure"]).round(1)

    ep_count_by_ent = {}
    rf_ep_det_by_ent = {}
    lstm_ep_det_by_ent = {}

    for eid in unique_entities:
        e_prefails_rf = prefails_rf[prefails_rf["entity_id"] == eid]
        if e_prefails_rf.empty:
            ep_count_by_ent[eid] = 0
            rf_ep_det_by_ent[eid] = 0
            lstm_ep_det_by_ent[eid] = 0
        else:
            e_prefails_lstm = prefails_lstm[prefails_lstm["entity_id"] == eid]
            
            grp_rf = e_prefails_rf.groupby("fail_time")
            grp_lstm = e_prefails_lstm.groupby("fail_time")

            n_eps = len(grp_rf)
            ep_count_by_ent[eid] = n_eps
            rf_ep_det_by_ent[eid] = sum((g["predicted_label"] > 0).any() for _, g in grp_rf)
            lstm_ep_det_by_ent[eid] = sum((g["predicted_label"] > 0).any() for _, g in grp_lstm)

    # 3. Precompute per-entity pure-normal status and false alarms
    failing_eids = set(prefails_rf["entity_id"].unique())
    pure_normal_eids = set(unique_entities) - failing_eids
    print(f"Failing entities: {len(failing_eids)}, Pure-normal entities: {len(pure_normal_eids)}")

    is_pure_normal = {}
    rf_pure_normal_fa = {}
    lstm_pure_normal_fa = {}

    for eid in unique_entities:
        if eid in pure_normal_eids:
            is_pure_normal[eid] = 1
            rf_pure_normal_fa[eid] = int((rf_pred_by_ent[eid] > 0).any())
            lstm_pure_normal_fa[eid] = int((lstm_pred_by_ent[eid] > 0).any())
        else:
            is_pure_normal[eid] = 0
            rf_pure_normal_fa[eid] = 0
            lstm_pure_normal_fa[eid] = 0

    rng = np.random.RandomState(random_seed)
    replicate_records = []

    for b in range(n_replicates):
        # Cluster bootstrap: sample entities with replacement
        sampled_eids = rng.choice(unique_entities, size=n_entities, replace=True)

        # Concatenate full window histories for all drawn entity instances
        y_boot = np.concatenate([y_by_ent[eid] for eid in sampled_eids])
        rf_boot = np.concatenate([rf_pred_by_ent[eid] for eid in sampled_eids])
        lstm_boot = np.concatenate([lstm_pred_by_ent[eid] for eid in sampled_eids])

        # Window-level metrics
        f1_rf = f1_score(y_boot, rf_boot, average="macro", zero_division=0.0)
        f1_lstm = f1_score(y_boot, lstm_boot, average="macro", zero_division=0.0)

        acc_rf = accuracy_score(y_boot, rf_boot)
        acc_lstm = accuracy_score(y_boot, lstm_boot)

        _, r_rf, _, _ = precision_recall_fscore_support(y_boot, rf_boot, labels=[0, 1, 2], zero_division=0.0)
        _, r_lstm, _, _ = precision_recall_fscore_support(y_boot, lstm_boot, labels=[0, 1, 2], zero_division=0.0)

        # Episode-level metrics (scaled across all drawn entity instances)
        tot_eps = sum(ep_count_by_ent[eid] for eid in sampled_eids)
        if tot_eps > 0:
            det_rf = sum(rf_ep_det_by_ent[eid] for eid in sampled_eids) / tot_eps
            det_lstm = sum(lstm_ep_det_by_ent[eid] for eid in sampled_eids) / tot_eps
        else:
            det_rf = np.nan
            det_lstm = np.nan

        # Pure-normal entity FAR (scaled across all drawn pure-normal entity instances)
        tot_pn = sum(is_pure_normal[eid] for eid in sampled_eids)
        if tot_pn > 0:
            far_rf = sum(rf_pure_normal_fa[eid] for eid in sampled_eids) / tot_pn
            far_lstm = sum(lstm_pure_normal_fa[eid] for eid in sampled_eids) / tot_pn
        else:
            far_rf = np.nan
            far_lstm = np.nan

        replicate_records.append({
            "macro_f1_rf": f1_rf,
            "macro_f1_lstm": f1_lstm,
            "delta_macro_f1": f1_lstm - f1_rf,
            "acc_rf": acc_rf,
            "acc_lstm": acc_lstm,
            "delta_acc": acc_lstm - acc_rf,
            "warn_recall_rf": r_rf[1],
            "warn_recall_lstm": r_lstm[1],
            "delta_warn_recall": r_lstm[1] - r_rf[1],
            "crit_recall_rf": r_rf[2],
            "crit_recall_lstm": r_lstm[2],
            "delta_crit_recall": r_lstm[2] - r_rf[2],
            "ep_det_rf_pct": det_rf * 100.0 if not np.isnan(det_rf) else np.nan,
            "ep_det_lstm_pct": det_lstm * 100.0 if not np.isnan(det_lstm) else np.nan,
            "delta_ep_det_pct": (det_lstm - det_rf) * 100.0 if not np.isnan(det_rf) else np.nan,
            "pure_normal_far_rf_pct": far_rf * 100.0 if not np.isnan(far_rf) else np.nan,
            "pure_normal_far_lstm_pct": far_lstm * 100.0 if not np.isnan(far_rf) else np.nan,
            "delta_pure_normal_far_pct": (far_lstm - far_rf) * 100.0 if not np.isnan(far_rf) else np.nan,
        })

    rep_df = pd.DataFrame(replicate_records)

    # Compute 95% Percentile Confidence Intervals
    summary = {}
    metrics_to_summarize = [
        ("macro_f1_rf", "macro_f1_lstm", "delta_macro_f1", "Window Macro-F1"),
        ("acc_rf", "acc_lstm", "delta_acc", "Window Accuracy"),
        ("warn_recall_rf", "warn_recall_lstm", "delta_warn_recall", "Warning Recall"),
        ("crit_recall_rf", "crit_recall_lstm", "delta_crit_recall", "Critical Recall"),
        ("ep_det_rf_pct", "ep_det_lstm_pct", "delta_ep_det_pct", "Episode Detection Rate (%)"),
        ("pure_normal_far_rf_pct", "pure_normal_far_lstm_pct", "delta_pure_normal_far_pct", "Pure-Normal Entity FAR (%)"),
    ]

    print("\n" + "=" * 78)
    print("BOOTSTRAP 95% CONFIDENCE INTERVALS (PERCENTILE METHOD, B = 1,000)")
    print("=" * 78)

    for k_rf, k_lstm, k_delta, label in metrics_to_summarize:
        s_rf = rep_df[k_rf].dropna()
        s_lstm = rep_df[k_lstm].dropna()
        s_delta = rep_df[k_delta].dropna()

        ci_rf = np.percentile(s_rf, [2.5, 97.5])
        ci_lstm = np.percentile(s_lstm, [2.5, 97.5])
        ci_delta = np.percentile(s_delta, [2.5, 97.5])

        mean_rf = float(s_rf.mean())
        mean_lstm = float(s_lstm.mean())
        mean_delta = float(s_delta.mean())

        is_significant = (ci_delta[0] > 0 and ci_delta[1] > 0) or (ci_delta[0] < 0 and ci_delta[1] < 0)

        summary[label] = {
            "rf": {"mean": mean_rf, "ci_95": [float(ci_rf[0]), float(ci_rf[1])]},
            "lstm": {"mean": mean_lstm, "ci_95": [float(ci_lstm[0]), float(ci_lstm[1])]},
            "delta": {"mean": mean_delta, "ci_95": [float(ci_delta[0]), float(ci_delta[1])]},
            "statistically_significant_at_0_05": bool(is_significant),
        }

        sig_str = "STATISTICALLY SIGNIFICANT" if is_significant else "NOT SIGNIFICANT"
        print(f"\n{label}:")
        print(f"  Random Forest : {mean_rf:.4f} [95% CI: {ci_rf[0]:.4f}, {ci_rf[1]:.4f}]")
        print(f"  Temporal LSTM : {mean_lstm:.4f} [95% CI: {ci_lstm[0]:.4f}, {ci_lstm[1]:.4f}]")
        print(f"  Paired Delta  : {mean_delta:+.4f} [95% CI: {ci_delta[0]:.4f}, {ci_delta[1]:.4f}] -> {sig_str}")

    out_path = Path(output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "methodology": {
                "unit_of_resampling": "entity_id (cluster bootstrap)",
                "cluster_size_preserved": True,
                "handling_of_duplicate_draws": "all drawn entity instances preserve full sequence and episodes independently",
                "paired_replicates": True,
                "n_replicates": n_replicates,
                "random_seed": random_seed,
                "interval_construction": "non-parametric empirical percentile (2.5%, 97.5%)",
            },
            "summary": summary,
        }, f, indent=2)

    print(f"\nSaved bootstrap uncertainty summary to {out_path}")
    return summary


if __name__ == "__main__":
    run_entity_cluster_bootstrap()
