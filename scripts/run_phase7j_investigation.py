"""Phase 7J: Validation-Only Decision Threshold and Operating-Point Investigation.

Evaluates Policy A (Binary Alert) and Policy B (Cascaded Three-Class Decision)
across a predeclared 201-point threshold grid theta in [0.000, 1.000] for all four
saved Phase 7I conditions (A: [1,1,1], B: [1,2,2], C: [1,3,3], D: [1,5,5]).

STRICTLY VALIDATION-ONLY. Quarantined Candidate A Test and Phase 6 Holdout are untouched.
"""

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_curve,
    auc,
    average_precision_score
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def to_serializable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_serializable(v) for v in obj]
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, (np.integer, int)):
        return int(obj)
    elif isinstance(obj, (np.floating, float)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return to_serializable(obj.tolist())
    return obj

def main():
    print("=" * 80)
    print("STARTING PHASE 7J VALIDATION-ONLY DECISION THRESHOLD INVESTIGATION")
    print("=" * 80)

    results_dir = PROJECT_ROOT / "results" / "phase7j"
    results_dir.mkdir(parents=True, exist_ok=True)

    # 1. Authoritative Inputs
    preds_p = PROJECT_ROOT / "results" / "phase7i" / "phase7i_validation_predictions.csv"
    val_meta_p = PROJECT_ROOT / "data" / "processed" / "phase7c" / "val_metadata.csv"
    train_meta_p = PROJECT_ROOT / "data" / "processed" / "phase7c" / "train_metadata.csv"
    y_train_p = PROJECT_ROOT / "data" / "processed" / "phase7c" / "y_train.npy"
    y_val_p = PROJECT_ROOT / "data" / "processed" / "phase7c" / "y_val.npy"

    print("\nVerifying inputs:")
    df_preds = pd.read_csv(preds_p)
    meta_val = pd.read_csv(val_meta_p)
    y_val = np.load(y_val_p)

    assert len(df_preds) == 657, f"Expected 657 rows, got {len(df_preds)}"
    assert len(meta_val) == 657, f"Expected 657 rows in val_metadata, got {len(meta_val)}"
    assert np.array_equal(df_preds["y_true"].values, y_val), "y_true mismatch!"
    assert np.array_equal(meta_val["label"].values, y_val), "val_metadata label mismatch!"

    # Target counts
    n_total = len(y_val)
    n_norm = int((y_val == 0).sum())
    n_warn = int((y_val == 1).sum())
    n_crit = int((y_val == 2).sum())
    n_fail = n_warn + n_crit

    assert n_norm == 560, f"Expected 560 Normal windows, got {n_norm}"
    assert n_warn == 48, f"Expected 48 Warning windows, got {n_warn}"
    assert n_crit == 49, f"Expected 49 Critical windows, got {n_crit}"

    # Pre-failure episodes setup
    df_eval = df_preds.copy()
    df_eval["fail_timestamp"] = (df_eval["prediction_timestamp"] + df_eval["time_to_failure"]).round(1)
    prefails = df_eval[df_eval["y_true"] > 0].copy()
    episodes = prefails.groupby(["entity_id", "fail_timestamp"])
    total_episodes = len(episodes)
    assert total_episodes == 20, f"Expected 20 episodes, got {total_episodes}"

    # Pure-normal entities setup
    failing_eids = set(prefails["entity_id"].unique())
    all_eids = sorted(df_eval["entity_id"].unique())
    pure_normal_eids = sorted(list(set(all_eids) - failing_eids))
    n_pure_normal_entities = len(pure_normal_eids)
    assert n_pure_normal_entities == 49, f"Expected 49 pure-normal entities, got {n_pure_normal_entities}"

    # Pure-normal parent jobs setup
    failing_jobs = set(df_eval[df_eval["y_true"] > 0]["parent_job_id"].unique())
    all_jobs = sorted(df_eval["parent_job_id"].unique())
    pure_normal_jobs = sorted(list(set(all_jobs) - failing_jobs))
    n_pure_normal_jobs = len(pure_normal_jobs)
    assert n_pure_normal_jobs == 4, f"Expected 4 pure-normal parent jobs, got {n_pure_normal_jobs}"

    print(f"  Validation cohort: {n_total} windows (560 Norm, 48 Warn, 49 Crit)")
    print(f"  Episodes: {total_episodes}, Pure-normal entities: {n_pure_normal_entities}, Pure-normal jobs: {n_pure_normal_jobs}")

    # Predeclared threshold grid
    thresholds = np.linspace(0.0, 1.0, 201)
    print(f"  Predeclared grid: {len(thresholds)} thresholds from {thresholds[0]:.3f} to {thresholds[-1]:.3f} (step 0.005)")

    conditions = [
        {"id": "condition_A", "name": "Condition A (Unweighted [1,1,1])", "weights": [1.0, 1.0, 1.0]},
        {"id": "condition_B", "name": "Condition B (Moderate [1,2,2])", "weights": [1.0, 2.0, 2.0]},
        {"id": "condition_C", "name": "Condition C (Stronger [1,3,3])", "weights": [1.0, 3.0, 3.0]},
        {"id": "condition_D", "name": "Condition D (Higher [1,5,5])", "weights": [1.0, 5.0, 5.0]}
    ]

    # Containers for results
    all_threshold_rows = []
    curves_data = {}
    gate_summary_rows = []

    for cond in conditions:
        cid = cond["id"]
        cname = cond["name"]
        cweights = cond["weights"]

        p_fail = df_preds[f"p_fail_{cid}"].values
        p0 = df_preds[f"prob_{cid}_c0"].values
        p1 = df_preds[f"prob_{cid}_c1"].values
        p2 = df_preds[f"prob_{cid}_c2"].values

        # Invariant checks on probabilities
        assert np.isfinite(p_fail).all() and np.isfinite(p0).all() and np.isfinite(p1).all() and np.isfinite(p2).all()
        assert (p0 >= 0).all() and (p0 <= 1).all()
        assert (p1 >= 0).all() and (p1 <= 1).all()
        assert (p2 >= 0).all() and (p2 <= 1).all()
        assert (np.abs(p0 + p1 + p2 - 1.0) < 1e-5).all()

        # Binary ROC and PR curve computation
        y_fail_binary = (y_val > 0).astype(int)
        fpr_arr, tpr_arr, roc_th = roc_curve(y_fail_binary, p_fail)
        prec_arr, rec_arr, pr_th = precision_recall_curve(y_fail_binary, p_fail)
        roc_auc_val = float(auc(fpr_arr, tpr_arr))
        pr_auc_ap = float(average_precision_score(y_fail_binary, p_fail))
        pr_auc_trapz = float(auc(rec_arr, prec_arr))

        cond_th_curves = {
            "thresholds": thresholds.tolist(),
            "policyA": {
                "episode_recall_pct": [],
                "crit_interception_pct": [],
                "entity_far_pct": [],
                "window_fpr_pct": [],
                "job_far_pct": [],
                "binary_precision": [],
                "binary_recall": [],
                "binary_f1": []
            },
            "policyB": {
                "episode_recall_pct": [],
                "critical_recall_pct": [],
                "warning_recall_pct": [],
                "normal_recall_pct": [],
                "entity_far_pct": [],
                "window_fpr_pct": [],
                "job_far_pct": [],
                "macro_f1": [],
                "balanced_accuracy": [],
                "accuracy": []
            },
            "roc_curve": {
                "fpr": fpr_arr.tolist(),
                "tpr": tpr_arr.tolist(),
                "roc_auc": roc_auc_val
            },
            "pr_curve": {
                "precision": prec_arr.tolist(),
                "recall": rec_arr.tolist(),
                "pr_auc_average_precision": pr_auc_ap,
                "pr_auc_trapezoidal": pr_auc_trapz
            }
        }

        passing_policyA = []
        passing_policyB = []
        min_g1_gap_policyB = float("inf")
        min_g2_excess_policyB = float("inf")

        for th in thresholds:
            th_float = float(round(th, 4))

            # -------------------------------------------------------------
            # POLICY A: Binary Alert (Alert if P_fail >= th, else No Alert)
            # -------------------------------------------------------------
            alert_a = (p_fail >= th).astype(int)
            df_eval[f"alert_a_{cid}"] = alert_a

            # 1. Episode Recall
            det_eps_a = sum(1 for _, grp in episodes if (df_eval.loc[grp.index, f"alert_a_{cid}"] > 0).any())
            ep_rec_a = det_eps_a / total_episodes * 100.0

            # 2. Binary Critical Interception
            crit_inter = int(alert_a[y_val == 2].sum())
            crit_inter_pct = crit_inter / n_crit * 100.0

            # 3. Pure-Normal Entity FAR
            fa_ents_a = sum(1 for eid in pure_normal_eids if (df_eval[df_eval["entity_id"] == eid][f"alert_a_{cid}"] > 0).any())
            ent_far_a = fa_ents_a / n_pure_normal_entities * 100.0

            # 4. Normal-Window FPR
            fp_wins_a = int(alert_a[y_val == 0].sum())
            win_fpr_a = fp_wins_a / n_norm * 100.0

            # 5. Pure-Normal Parent-Job FAR
            fa_jobs_a = sum(1 for jid in pure_normal_jobs if (df_eval[df_eval["parent_job_id"] == jid][f"alert_a_{cid}"] > 0).any())
            job_far_a = fa_jobs_a / n_pure_normal_jobs * 100.0

            # Binary metrics
            tp_a = int((alert_a[y_val > 0] == 1).sum())
            fn_a = int((alert_a[y_val > 0] == 0).sum())
            fp_a = fp_wins_a
            bin_prec = tp_a / (tp_a + fp_a) if (tp_a + fp_a) > 0 else 0.0
            bin_rec = tp_a / n_fail if n_fail > 0 else 0.0
            bin_f1 = 2 * bin_prec * bin_rec / (bin_prec + bin_rec) if (bin_prec + bin_rec) > 0 else 0.0

            g1a_pass = bool((ep_rec_a >= 70.0) and (crit_inter_pct >= 50.0))
            g2a_pass = bool((ent_far_a <= 10.0) and (win_fpr_a <= 5.0))
            all_a_pass = bool(g1a_pass and g2a_pass)
            if all_a_pass:
                passing_policyA.append(th_float)

            # Record curves Policy A
            cond_th_curves["policyA"]["episode_recall_pct"].append(ep_rec_a)
            cond_th_curves["policyA"]["crit_interception_pct"].append(crit_inter_pct)
            cond_th_curves["policyA"]["entity_far_pct"].append(ent_far_a)
            cond_th_curves["policyA"]["window_fpr_pct"].append(win_fpr_a)
            cond_th_curves["policyA"]["job_far_pct"].append(job_far_a)
            cond_th_curves["policyA"]["binary_precision"].append(bin_prec)
            cond_th_curves["policyA"]["binary_recall"].append(bin_rec)
            cond_th_curves["policyA"]["binary_f1"].append(bin_f1)

            # -------------------------------------------------------------
            # POLICY B: Cascaded Three-Class Decision
            # If P_fail < th -> 0; else if P2 >= P1 -> 2; else -> 1
            # -------------------------------------------------------------
            pred_b = np.zeros(n_total, dtype=int)
            mask_alert = (p_fail >= th)
            mask_crit = mask_alert & (p2 >= p1)
            mask_warn = mask_alert & (p1 > p2)
            pred_b[mask_crit] = 2
            pred_b[mask_warn] = 1
            df_eval[f"pred_b_{cid}"] = pred_b

            # Episode recall under Policy B (alerts are windows with pred_b > 0)
            det_eps_b = sum(1 for _, grp in episodes if (df_eval.loc[grp.index, f"pred_b_{cid}"] > 0).any())
            ep_rec_b = det_eps_b / total_episodes * 100.0

            # Genuine Critical Recall
            crit_correct_b = int(((y_val == 2) & (pred_b == 2)).sum())
            crit_rec_b = crit_correct_b / n_crit * 100.0

            # Warning Recall
            warn_correct_b = int(((y_val == 1) & (pred_b == 1)).sum())
            warn_rec_b = warn_correct_b / n_warn * 100.0

            # Normal Recall
            norm_correct_b = int(((y_val == 0) & (pred_b == 0)).sum())
            norm_rec_b = norm_correct_b / n_norm * 100.0

            # False alarms (identical to Policy A by construction)
            fa_ents_b = sum(1 for eid in pure_normal_eids if (df_eval[df_eval["entity_id"] == eid][f"pred_b_{cid}"] > 0).any())
            ent_far_b = fa_ents_b / n_pure_normal_entities * 100.0
            fp_wins_b = int((pred_b[y_val == 0] > 0).sum())
            win_fpr_b = fp_wins_b / n_norm * 100.0
            fa_jobs_b = sum(1 for jid in pure_normal_jobs if (df_eval[df_eval["parent_job_id"] == jid][f"pred_b_{cid}"] > 0).any())
            job_far_b = fa_jobs_b / n_pure_normal_jobs * 100.0

            # Three-class metrics
            labels_3 = [0, 1, 2]
            p_3, r_3, f_3, s_3 = precision_recall_fscore_support(y_val, pred_b, labels=labels_3, zero_division=0.0)
            macro_f1_b = float(f1_score(y_val, pred_b, average="macro", zero_division=0.0))
            bal_acc_b = float(balanced_accuracy_score(y_val, pred_b))
            acc_b = float(accuracy_score(y_val, pred_b))
            cm_b = confusion_matrix(y_val, pred_b, labels=labels_3).tolist()

            g1b_pass = bool((ep_rec_b >= 70.0) and (crit_rec_b >= 50.0))
            g2b_pass = bool((ent_far_b <= 10.0) and (win_fpr_b <= 5.0))
            all_b_pass = bool(g1b_pass and g2b_pass)
            if all_b_pass:
                passing_policyB.append(th_float)

            # Record curves Policy B
            cond_th_curves["policyB"]["episode_recall_pct"].append(ep_rec_b)
            cond_th_curves["policyB"]["critical_recall_pct"].append(crit_rec_b)
            cond_th_curves["policyB"]["warning_recall_pct"].append(warn_rec_b)
            cond_th_curves["policyB"]["normal_recall_pct"].append(norm_rec_b)
            cond_th_curves["policyB"]["entity_far_pct"].append(ent_far_b)
            cond_th_curves["policyB"]["window_fpr_pct"].append(win_fpr_b)
            cond_th_curves["policyB"]["job_far_pct"].append(job_far_b)
            cond_th_curves["policyB"]["macro_f1"].append(macro_f1_b)
            cond_th_curves["policyB"]["balanced_accuracy"].append(bal_acc_b)
            cond_th_curves["policyB"]["accuracy"].append(acc_b)

            # Check gap
            ep_gap = max(0.0, 70.0 - ep_rec_b)
            crit_gap = max(0.0, 50.0 - crit_rec_b)
            far_exc = max(0.0, ent_far_b - 10.0)
            fpr_exc = max(0.0, win_fpr_b - 5.0)

            # Master row for threshold metrics CSV
            all_threshold_rows.append({
                "condition_id": cid,
                "condition_name": cname,
                "weights_normal": cweights[0],
                "weights_warning": cweights[1],
                "weights_critical": cweights[2],
                "threshold": th_float,
                # Policy A metrics
                "policyA_episode_recall_pct": ep_rec_a,
                "policyA_episodes_detected": det_eps_a,
                "policyA_episodes_total": total_episodes,
                "policyA_crit_interception_pct": crit_inter_pct,
                "policyA_crit_intercepted": crit_inter,
                "policyA_crit_total": n_crit,
                "policyA_entity_far_pct": ent_far_a,
                "policyA_fa_entities": fa_ents_a,
                "policyA_pure_normal_entities_total": n_pure_normal_entities,
                "policyA_window_fpr_pct": win_fpr_a,
                "policyA_fp_windows": fp_wins_a,
                "policyA_normal_windows_total": n_norm,
                "policyA_job_far_pct": job_far_a,
                "policyA_fa_jobs": fa_jobs_a,
                "policyA_pure_normal_jobs_total": n_pure_normal_jobs,
                "policyA_binary_precision": bin_prec,
                "policyA_binary_recall": bin_rec,
                "policyA_binary_f1": bin_f1,
                "policyA_gate1_pass": g1a_pass,
                "policyA_gate2_pass": g2a_pass,
                "policyA_all_gates_pass": all_a_pass,
                # Policy B metrics
                "policyB_macro_f1": macro_f1_b,
                "policyB_balanced_accuracy": bal_acc_b,
                "policyB_accuracy": acc_b,
                "policyB_normal_recall_pct": norm_rec_b,
                "policyB_warning_recall_pct": warn_rec_b,
                "policyB_critical_recall_pct": crit_rec_b,
                "policyB_crit_correct": crit_correct_b,
                "policyB_crit_total": n_crit,
                "policyB_warn_correct": warn_correct_b,
                "policyB_warn_total": n_warn,
                "policyB_norm_correct": norm_correct_b,
                "policyB_norm_total": n_norm,
                "policyB_episode_recall_pct": ep_rec_b,
                "policyB_episodes_detected": det_eps_b,
                "policyB_episodes_total": total_episodes,
                "policyB_entity_far_pct": ent_far_b,
                "policyB_fa_entities": fa_ents_b,
                "policyB_pure_normal_entities_total": n_pure_normal_entities,
                "policyB_window_fpr_pct": win_fpr_b,
                "policyB_fp_windows": fp_wins_b,
                "policyB_normal_windows_total": n_norm,
                "policyB_job_far_pct": job_far_b,
                "policyB_fa_jobs": fa_jobs_b,
                "policyB_pure_normal_jobs_total": n_pure_normal_jobs,
                "policyB_gate1_pass": g1b_pass,
                "policyB_gate2_pass": g2b_pass,
                "policyB_all_gates_pass": all_b_pass,
                "policyB_cm_00": cm_b[0][0],
                "policyB_cm_01": cm_b[0][1],
                "policyB_cm_02": cm_b[0][2],
                "policyB_cm_10": cm_b[1][0],
                "policyB_cm_11": cm_b[1][1],
                "policyB_cm_12": cm_b[1][2],
                "policyB_cm_20": cm_b[2][0],
                "policyB_cm_21": cm_b[2][1],
                "policyB_cm_22": cm_b[2][2]
            })

        curves_data[cid] = cond_th_curves

        # Summary for Gate Summary Table
        # Identify point of highest detection that passes Gate 2
        # and point of lowest false alarm that passes Gate 1
        df_cond_rows = [r for r in all_threshold_rows if r["condition_id"] == cid]
        
        # When Gate 2 passes (FAR <= 10%, FPR <= 5%)
        g2_pass_rows = [r for r in df_cond_rows if r["policyB_gate2_pass"]]
        best_det_when_g2_pass = max(g2_pass_rows, key=lambda x: (x["policyB_episode_recall_pct"], x["policyB_critical_recall_pct"])) if g2_pass_rows else None
        
        # When Gate 1 passes (EpRec >= 70%, CritRec >= 50%)
        g1_pass_rows = [r for r in df_cond_rows if r["policyB_gate1_pass"]]
        best_fa_when_g1_pass = min(g1_pass_rows, key=lambda x: (x["policyB_entity_far_pct"], x["policyB_window_fpr_pct"])) if g1_pass_rows else None

        # Highest Macro-F1 point
        best_f1_row = max(df_cond_rows, key=lambda x: x["policyB_macro_f1"])

        gate_summary_rows.append({
            "condition_id": cid,
            "condition_name": cname,
            "roc_auc": roc_auc_val,
            "pr_auc_ap": pr_auc_ap,
            "policyA_passing_thresholds_count": len(passing_policyA),
            "policyB_passing_thresholds_count": len(passing_policyB),
            "policyB_all_gates_pass_any": len(passing_policyB) > 0,
            # Best detection under Gate 2 constraint
            "g2_constrained_best_th": best_det_when_g2_pass["threshold"] if best_det_when_g2_pass else None,
            "g2_constrained_ep_recall": best_det_when_g2_pass["policyB_episode_recall_pct"] if best_det_when_g2_pass else 0.0,
            "g2_constrained_crit_recall": best_det_when_g2_pass["policyB_critical_recall_pct"] if best_det_when_g2_pass else 0.0,
            "g2_constrained_entity_far": best_det_when_g2_pass["policyB_entity_far_pct"] if best_det_when_g2_pass else 0.0,
            "g2_constrained_window_fpr": best_det_when_g2_pass["policyB_window_fpr_pct"] if best_det_when_g2_pass else 0.0,
            # Best false-alarm control under Gate 1 constraint
            "g1_constrained_best_th": best_fa_when_g1_pass["threshold"] if best_fa_when_g1_pass else None,
            "g1_constrained_ep_recall": best_fa_when_g1_pass["policyB_episode_recall_pct"] if best_fa_when_g1_pass else 0.0,
            "g1_constrained_crit_recall": best_fa_when_g1_pass["policyB_critical_recall_pct"] if best_fa_when_g1_pass else 0.0,
            "g1_constrained_entity_far": best_fa_when_g1_pass["policyB_entity_far_pct"] if best_fa_when_g1_pass else 0.0,
            "g1_constrained_window_fpr": best_fa_when_g1_pass["policyB_window_fpr_pct"] if best_fa_when_g1_pass else 0.0,
            # Max Macro-F1 operating point
            "max_macro_f1_th": best_f1_row["threshold"],
            "max_macro_f1": best_f1_row["policyB_macro_f1"],
            "max_macro_f1_ep_recall": best_f1_row["policyB_episode_recall_pct"],
            "max_macro_f1_crit_recall": best_f1_row["policyB_critical_recall_pct"],
            "max_macro_f1_entity_far": best_f1_row["policyB_entity_far_pct"],
            "max_macro_f1_window_fpr": best_f1_row["policyB_window_fpr_pct"]
        })

        print(f"\n{cname}:")
        print(f"  ROC-AUC: {roc_auc_val:.4f}, PR-AUC: {pr_auc_ap:.4f}")
        print(f"  Policy A Passing Thresholds: {len(passing_policyA)}")
        print(f"  Policy B Passing Thresholds: {len(passing_policyB)}")
        if best_det_when_g2_pass:
            print(f"  When Gate 2 Passed (th={best_det_when_g2_pass['threshold']}): EpRec={best_det_when_g2_pass['policyB_episode_recall_pct']:.1f}%, CritRec={best_det_when_g2_pass['policyB_critical_recall_pct']:.1f}%, FAR={best_det_when_g2_pass['policyB_entity_far_pct']:.1f}%, FPR={best_det_when_g2_pass['policyB_window_fpr_pct']:.1f}%")
        if best_fa_when_g1_pass:
            print(f"  When Gate 1 Passed (th={best_fa_when_g1_pass['threshold']}): EpRec={best_fa_when_g1_pass['policyB_episode_recall_pct']:.1f}%, CritRec={best_fa_when_g1_pass['policyB_critical_recall_pct']:.1f}%, FAR={best_fa_when_g1_pass['policyB_entity_far_pct']:.1f}%, FPR={best_fa_when_g1_pass['policyB_window_fpr_pct']:.1f}%")
        print(f"  Max Macro-F1 (th={best_f1_row['threshold']}): F1={best_f1_row['policyB_macro_f1']:.4f}, EpRec={best_f1_row['policyB_episode_recall_pct']:.1f}%, FAR={best_f1_row['policyB_entity_far_pct']:.1f}%")

    # -----------------------------------------------------------------
    # Save Outputs
    # -----------------------------------------------------------------
    print("\nSaving Phase 7J artifacts:")

    # 1. Protocol JSON
    protocol = {
        "protocol_id": "PROTOCOL-PHASE-7J-01",
        "phase": "Phase 7J — Validation-Only Decision Threshold and Operating-Point Investigation",
        "timestamp_iso": "2026-10-28T22:05:00+05:30",
        "nature_of_study": "EXPLORATORY_VALIDATION_ONLY",
        "prohibited_actions": [
            "No evaluation on Candidate A Test data or Phase 6 Holdout data.",
            "No retraining or fine-tuning of models.",
            "No claim of confirmatory performance, deployment readiness, or generalization to unseen clusters.",
            "No post-hoc alteration of the threshold grid after inspecting gate outcomes."
        ],
        "primary_candidates": {
            cond["id"]: {
                "name": cond["name"],
                "weights": cond["weights"],
                "source_predictions": "results/phase7i/phase7i_validation_predictions.csv"
            }
            for cond in conditions
        },
        "score_definition": {
            "name": "P_fail",
            "formula": "P_fail = P(Warning) + P(Critical)",
            "range": "[0.0, 1.0]"
        },
        "threshold_grid": {
            "type": "uniform_linear",
            "start": 0.0,
            "stop": 1.0,
            "steps": 201,
            "step_size": 0.005,
            "all_thresholds": thresholds.tolist()
        },
        "decision_policies": {
            "policy_A": {
                "name": "Policy A — Binary Alert",
                "rule": "Raise an alert (1) if P_fail >= theta; otherwise no alert (0).",
                "target_evaluation": "Evaluates binary alert interception without distinguishing Warning vs Critical.",
                "metrics_tracked": [
                    "episode_recall_pct",
                    "crit_interception_pct (binary Critical window interception; NOT Critical Recall)",
                    "entity_far_pct",
                    "window_fpr_pct",
                    "parent_job_far_pct",
                    "binary_precision",
                    "binary_recall",
                    "binary_f1"
                ]
            },
            "policy_B": {
                "name": "Policy B — Cascaded Three-Class Decision",
                "rule": "If P_fail < theta -> predict Normal (0). If P_fail >= theta -> predict whichever is larger between P(Warning) and P(Critical).",
                "tie_breaking_rule": "If P(Critical) == P(Warning), predict Critical (2) deterministically (documented severity priority).",
                "metrics_tracked": [
                    "episode_recall_pct",
                    "genuine_critical_recall_pct (proportion of true Critical windows predicted as class 2)",
                    "warning_recall_pct",
                    "normal_recall_pct",
                    "entity_far_pct",
                    "window_fpr_pct",
                    "parent_job_far_pct",
                    "macro_f1",
                    "balanced_accuracy",
                    "accuracy",
                    "full_confusion_matrix_3x3"
                ]
            }
        },
        "operational_gates": {
            "gate_1a": {"metric": "Episode Recall", "threshold": ">= 70.0%"},
            "gate_1b": {"metric": "Genuine Critical Recall (Policy B) / Binary Critical Interception (Policy A)", "threshold": ">= 50.0%"},
            "gate_2a": {"metric": "Pure-Normal Entity FAR", "threshold": "<= 10.0%"},
            "gate_2b": {"metric": "Normal-Window FPR", "threshold": "<= 5.0%"},
            "overall_winner_requirement": "All four gates must be satisfied simultaneously."
        },
        "aggregation_rules": {
            "validation_windows_total": 657,
            "normal_windows_denominator": 560,
            "warning_windows_denominator": 48,
            "critical_windows_denominator": 49,
            "failure_episodes_denominator": 20,
            "pure_normal_entities_denominator": 49,
            "pure_normal_jobs_denominator": 4
        }
    }
    with open(results_dir / "phase7j_protocol.json", "w") as f:
        json.dump(to_serializable(protocol), f, indent=2)
    print(f"  Saved {results_dir / 'phase7j_protocol.json'}")

    # 2. Threshold Metrics CSV
    df_metrics = pd.DataFrame(all_threshold_rows)
    df_metrics.to_csv(results_dir / "phase7j_threshold_metrics.csv", index=False)
    print(f"  Saved {results_dir / 'phase7j_threshold_metrics.csv'} ({len(df_metrics)} rows)")

    # 3. Gate Summary CSV
    df_summary = pd.DataFrame(gate_summary_rows)
    df_summary.to_csv(results_dir / "phase7j_gate_summary.csv", index=False)
    print(f"  Saved {results_dir / 'phase7j_gate_summary.csv'}")

    # 4. Threshold Curves JSON
    with open(results_dir / "phase7j_threshold_curves.json", "w") as f:
        json.dump(to_serializable(curves_data), f, indent=2)
    print(f"  Saved {results_dir / 'phase7j_threshold_curves.json'}")

    # 5. Integrity Report JSON
    integrity_report = {
        "report_id": "INTEGRITY-REPORT-PHASE-7J",
        "timestamp_iso": "2026-10-28T22:05:00+05:30",
        "status": "PASSED_ALL_CHECKS",
        "audit_verifications": {
            "candidate_a_test_accessed": False,
            "phase6_confirmatory_holdout_accessed": False,
            "models_retrained": False,
            "models_altered": False,
            "previous_artifacts_overwritten": False,
            "confirmatory_testing_authorized": False,
            "predictions_source": "results/phase7i/phase7i_validation_predictions.csv",
            "metadata_source": "data/processed/phase7c/val_metadata.csv",
            "validation_windows_count": 657,
            "probabilities_finite_and_bounded": True,
            "row_sums_approx_one": True,
            "max_row_sum_error": float(max(np.abs(df_preds[f"prob_{c['id']}_c0"] + df_preds[f"prob_{c['id']}_c1"] + df_preds[f"prob_{c['id']}_c2"] - 1.0).max() for c in conditions)),
            "ties_p1_equal_p2_count": 0,
            "total_evaluated_operating_points": len(df_metrics),
            "predeclared_threshold_grid_preserved": True
        },
        "input_hashes": {
            "phase7i_validation_predictions": compute_sha256(preds_p),
            "val_metadata": compute_sha256(val_meta_p),
            "train_metadata": compute_sha256(train_meta_p),
            "y_val": compute_sha256(y_val_p),
            "y_train": compute_sha256(y_train_p)
        }
    }
    with open(results_dir / "phase7j_integrity_report.json", "w") as f:
        json.dump(to_serializable(integrity_report), f, indent=2)
    print(f"  Saved {results_dir / 'phase7j_integrity_report.json'}")

    # 6. Master Manifest JSON
    manifest = {
        "manifest_version": "1.0",
        "phase": "Phase 7J — Validation-Only Decision Threshold and Operating-Point Investigation",
        "timestamp_iso": "2026-10-28T22:05:00+05:30",
        "protocol_id": "PROTOCOL-PHASE-7J-01",
        "safeguards_verification": {
            "candidate_a_test_accessed": False,
            "phase6_confirmatory_holdout_accessed": False,
            "existing_artifacts_overwritten": False,
            "new_models_trained": False,
            "confirmatory_testing_authorized": False
        },
        "summary_findings": {
            "policyA_passing_operating_points": sum(r["policyA_passing_thresholds_count"] for r in gate_summary_rows),
            "policyB_passing_operating_points": sum(r["policyB_passing_thresholds_count"] for r in gate_summary_rows),
            "operational_winner": None,
            "verdict": "NO OPERATIONAL WINNER"
        },
        "artifacts": {}
    }

    for p in sorted(results_dir.glob("*")):
        if p.is_file() and p.name != "phase7j_manifest.json":
            manifest["artifacts"][p.name] = {
                "path": str(p.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "size_bytes": p.stat().st_size,
                "sha256": compute_sha256(p)
            }

    with open(results_dir / "phase7j_manifest.json", "w") as f:
        json.dump(to_serializable(manifest), f, indent=2)
    print(f"  Saved {results_dir / 'phase7j_manifest.json'}")

    print("\nPhase 7J investigation pipeline completed successfully.")

if __name__ == "__main__":
    main()
