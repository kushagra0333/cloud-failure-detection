"""Comprehensive Audit of Probability Separation, Ranking Metrics, and Metric Reproduction for Phase 7I.
Strictly TRAIN/VAL only. Uses results/phase7i/phase7i_validation_predictions.csv and data/processed/phase7c/val_metadata.csv.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
    auc
)

PROJECT_ROOT = Path(".").resolve()
preds_p = PROJECT_ROOT / "results" / "phase7i" / "phase7i_validation_predictions.csv"
meta_val_p = PROJECT_ROOT / "data" / "processed" / "phase7c" / "val_metadata.csv"
metrics_orig_p = PROJECT_ROOT / "results" / "phase7i" / "phase7i_metrics.json"

df_preds = pd.read_csv(preds_p)
df_meta = pd.read_csv(meta_val_p)
with open(metrics_orig_p, "r") as f:
    orig_metrics = json.load(f)

print(f"Loaded validation predictions: shape {df_preds.shape}")
assert len(df_preds) == 657, f"Expected 657 rows, got {len(df_preds)}"

# Verify y_true
y_true = df_preds["y_true"].values
y_meta = df_meta["label"].values
assert np.array_equal(y_true, y_meta), "Mismatch between prediction df y_true and val_metadata label!"

# Binary targets
y_fail_binary = (y_true > 0).astype(int)  # 1 for Warning or Critical, 0 for Normal
y_crit_binary = (y_true == 2).astype(int)
y_warn_binary = (y_true == 1).astype(int)

n_total = len(y_true)
n_norm = int((y_true == 0).sum())
n_warn = int((y_true == 1).sum())
n_crit = int((y_true == 2).sum())
n_fail = n_warn + n_crit

print(f"Validation class distribution: Normal={n_norm} ({n_norm/n_total*100:.2f}%), Warning={n_warn} ({n_warn/n_total*100:.2f}%), Critical={n_crit} ({n_crit/n_total*100:.2f}%)")
print(f"Binary failure prevalence: {n_fail}/{n_total} = {n_fail/n_total*100:.4f}%")

def compute_multiclass_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels)
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return float(ece)

def get_distribution_stats(arr: np.ndarray) -> dict:
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p25": float(np.percentile(arr, 25)),
        "median": float(np.median(arr)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "max": float(np.max(arr)),
        "iqr": float(np.percentile(arr, 75) - np.percentile(arr, 25))
    }

def compute_pairwise_ranking_accuracy(pos_scores: np.ndarray, neg_scores: np.ndarray) -> float:
    # Exact pairwise comparison: % of (pos, neg) pairs where pos > neg (ties count 0.5)
    # This equals Mann-Whitney U / (n_pos * n_neg)
    n_pos = len(pos_scores)
    n_neg = len(neg_scores)
    total_pairs = n_pos * n_neg
    concordant = 0.0
    for p in pos_scores:
        concordant += np.sum(p > neg_scores) + 0.5 * np.sum(p == neg_scores)
    return float(concordant / total_pairs)

conditions = ["condition_A", "condition_B", "condition_C", "condition_D"]
condition_names = {
    "condition_A": "Condition A (Unweighted [1,1,1])",
    "condition_B": "Condition B (Moderate [1,2,2])",
    "condition_C": "Condition C (Stronger [1,3,3])",
    "condition_D": "Condition D (Higher [1,5,5])"
}

ranking_audit = {
    "audit_metadata": {
        "phase": "Phase 7I-A",
        "description": "Probability and feature separation audit across 4 loss-weighted conditions",
        "validation_samples": n_total,
        "true_normal_windows": n_norm,
        "true_failure_windows": n_fail,
        "true_warning_windows": n_warn,
        "true_critical_windows": n_crit,
        "positive_class_prevalence_pct": n_fail / n_total * 100.0
    },
    "conditions": {}
}

reproduced_metrics = {
    "audit_metadata": {
        "phase": "Phase 7I-A",
        "description": "Deterministic metric reproduction and discrepancy verification"
    },
    "conditions": {}
}

# Episode setup
df_eval = df_preds.copy()
df_eval["fail_timestamp"] = (df_eval["prediction_timestamp"] + df_eval["time_to_failure"]).round(1)
prefails = df_eval[df_eval["y_true"] > 0].copy()
episodes = prefails.groupby(["entity_id", "fail_timestamp"])
total_episodes = len(episodes)

# Entity setup
failing_eids = set(prefails["entity_id"].unique()) if not prefails.empty else set()
all_eids = sorted(df_eval["entity_id"].unique())
pure_normal_eids = sorted(list(set(all_eids) - failing_eids))
n_pure_normal_entities = len(pure_normal_eids)

# Parent Job setup
failing_jobs = set(df_eval[df_eval["y_true"] > 0]["parent_job_id"].unique())
all_jobs = sorted(df_eval["parent_job_id"].unique())
pure_normal_jobs = sorted(list(set(all_jobs) - failing_jobs))
n_pure_normal_jobs = len(pure_normal_jobs)

for cid in conditions:
    cname = condition_names[cid]
    p0 = df_preds[f"prob_{cid}_c0"].values
    p1 = df_preds[f"prob_{cid}_c1"].values
    p2 = df_preds[f"prob_{cid}_c2"].values
    probs = np.column_stack([p0, p1, p2])
    p_fail = p1 + p2
    y_pred = df_preds[f"y_pred_{cid}"].values

    # Probability row sum check
    row_sums = probs.sum(axis=1)
    max_sum_err = float(np.max(np.abs(row_sums - 1.0)))
    mean_sum_err = float(np.mean(np.abs(row_sums - 1.0)))

    # Ranking metrics
    # ROC-AUC
    roc_auc_fail = float(roc_auc_score(y_fail_binary, p_fail))
    roc_auc_crit = float(roc_auc_score(y_crit_binary, p2))
    roc_auc_warn = float(roc_auc_score(y_warn_binary, p1))

    # PR-AUC
    pr_auc_ap = float(average_precision_score(y_fail_binary, p_fail))
    prec_curve, rec_curve, _ = precision_recall_curve(y_fail_binary, p_fail)
    pr_auc_trapz = float(auc(rec_curve, prec_curve))

    # Score distributions
    s_norm = p_fail[y_true == 0]
    s_fail = p_fail[y_true > 0]
    s_warn = p_fail[y_true == 1]
    s_crit = p_fail[y_true == 2]

    stats_norm = get_distribution_stats(s_norm)
    stats_fail = get_distribution_stats(s_fail)
    stats_warn = get_distribution_stats(s_warn)
    stats_crit = get_distribution_stats(s_crit)

    # Pairwise ranking percentage (% of fail windows ranked above normal windows)
    pairwise_pct = compute_pairwise_ranking_accuracy(s_fail, s_norm) * 100.0
    pairwise_pct_crit = compute_pairwise_ranking_accuracy(s_crit, s_norm) * 100.0

    # Probability summaries
    prob_summary = {
        "overall_mean_probs": {
            "P_Normal": float(np.mean(p0)),
            "P_Warning": float(np.mean(p1)),
            "P_Critical": float(np.mean(p2))
        },
        "on_true_normal_windows": {
            "mean_P_Normal": float(np.mean(p0[y_true == 0])),
            "mean_P_Warning": float(np.mean(p1[y_true == 0])),
            "mean_P_Critical": float(np.mean(p2[y_true == 0])),
            "median_P_Normal": float(np.median(p0[y_true == 0])),
            "mean_P_fail": float(np.mean(p_fail[y_true == 0]))
        },
        "on_true_warning_windows": {
            "mean_P_Normal": float(np.mean(p0[y_true == 1])),
            "mean_P_Warning": float(np.mean(p1[y_true == 1])),
            "mean_P_Critical": float(np.mean(p2[y_true == 1])),
            "mean_P_fail": float(np.mean(p_fail[y_true == 1]))
        },
        "on_true_critical_windows": {
            "mean_P_Normal": float(np.mean(p0[y_true == 2])),
            "mean_P_Warning": float(np.mean(p1[y_true == 2])),
            "mean_P_Critical": float(np.mean(p2[y_true == 2])),
            "median_P_Critical": float(np.median(p2[y_true == 2])),
            "mean_P_fail": float(np.mean(p_fail[y_true == 2]))
        },
        "separation_gap_critical_minus_normal": float(np.mean(p_fail[y_true == 2]) - np.mean(p_fail[y_true == 0])),
        "separation_gap_failure_minus_normal": float(np.mean(p_fail[y_true > 0]) - np.mean(p_fail[y_true == 0]))
    }

    pred_counts = {
        "pred_c0_count": int((y_pred == 0).sum()),
        "pred_c1_count": int((y_pred == 1).sum()),
        "pred_c2_count": int((y_pred == 2).sum()),
        "pred_c0_pct": float(np.mean(y_pred == 0) * 100.0),
        "pred_c1_pct": float(np.mean(y_pred == 1) * 100.0),
        "pred_c2_pct": float(np.mean(y_pred == 2) * 100.0)
    }

    ranking_audit["conditions"][cid] = {
        "condition_name": cname,
        "row_probability_sum_check": {
            "all_approx_equal_one": bool(max_sum_err < 1e-5),
            "max_absolute_error": max_sum_err,
            "mean_absolute_error": mean_sum_err
        },
        "ranking_discrimination": {
            "roc_auc_binary_failure_vs_normal": roc_auc_fail,
            "pr_auc_average_precision": pr_auc_ap,
            "pr_auc_trapezoidal_integral": pr_auc_trapz,
            "positive_prevalence_pct": n_fail / n_total * 100.0,
            "pr_auc_baseline_ratio": pr_auc_ap / (n_fail / n_total),
            "pairwise_fail_above_normal_pct": pairwise_pct,
            "pairwise_crit_above_normal_pct": pairwise_pct_crit,
            "roc_auc_critical_vs_rest": roc_auc_crit,
            "roc_auc_warning_vs_rest": roc_auc_warn
        },
        "failure_score_distributions": {
            "true_normal_windows": stats_norm,
            "true_failure_windows": stats_fail,
            "true_warning_windows": stats_warn,
            "true_critical_windows": stats_crit
        },
        "probability_diagnostics": prob_summary,
        "predicted_class_counts": pred_counts
    }

    # REPRODUCE ALL REPORTED METRICS
    labels = [0, 1, 2]
    prec, rec, f1, sup = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0.0)
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0.0))
    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()

    crit_correct = int(((y_true == 2) & (y_pred == 2)).sum())
    crit_total = n_crit
    critical_recall = float(crit_correct / crit_total) if crit_total > 0 else 0.0

    # Operational metrics
    df_eval[f"y_pred_{cid}"] = y_pred
    detected_episodes = sum(1 for _, grp in episodes if (grp[f"y_pred_{cid}"] > 0).any())
    episode_recall_pct = (detected_episodes / total_episodes * 100.0) if total_episodes > 0 else 0.0

    fa_entities = sum(1 for eid in pure_normal_eids if (df_eval[df_eval["entity_id"] == eid][f"y_pred_{cid}"] > 0).any())
    entity_far_pct = (fa_entities / n_pure_normal_entities * 100.0) if n_pure_normal_entities > 0 else 0.0

    fp_windows = int((y_pred[y_true == 0] > 0).sum())
    window_fpr_pct = (fp_windows / n_norm * 100.0) if n_norm > 0 else 0.0

    fa_jobs = sum(1 for jid in pure_normal_jobs if (df_eval[df_eval["parent_job_id"] == jid][f"y_pred_{cid}"] > 0).any())
    job_far_pct = (fa_jobs / n_pure_normal_jobs * 100.0) if n_pure_normal_jobs > 0 else 0.0

    # Calibration
    y_one_hot = np.zeros((n_total, 3))
    for i, y in enumerate(y_true):
        y_one_hot[i, y] = 1.0
    P_safe = np.clip(probs, 1e-12, 1.0 - 1e-12)
    P_safe = P_safe / P_safe.sum(axis=1, keepdims=True)

    ece = compute_multiclass_ece(P_safe, y_true, n_bins=10)
    brier = float(np.mean(np.sum((P_safe - y_one_hot)**2, axis=1)))
    nll = float(log_loss(y_true, P_safe))

    # Compare against original Phase 7I metrics
    orig_c = orig_metrics[cid]
    discrepancies = []
    
    def check_match(name, val_recomputed, val_orig, tol=1e-5):
        if abs(val_recomputed - val_orig) > tol:
            discrepancies.append(f"{name}: recomputed={val_recomputed}, original={val_orig}")

    check_match("accuracy", acc, orig_c["accuracy"])
    check_match("macro_f1", macro_f1, orig_c["macro_f1"])
    check_match("balanced_accuracy", bal_acc, orig_c["balanced_accuracy"])
    check_match("critical_recall", critical_recall, orig_c["critical_recall"])
    check_match("episode_recall_pct", episode_recall_pct, orig_c["episode_metrics"]["episode_recall_pct"])
    check_match("entity_far_pct", entity_far_pct, orig_c["false_alarm_metrics"]["entity_far_pct"])
    check_match("window_fpr_pct", window_fpr_pct, orig_c["false_alarm_metrics"]["window_fpr_pct"])
    check_match("job_far_pct", job_far_pct, orig_c["false_alarm_metrics"]["job_far_pct"])
    check_match("ece", ece, orig_c["calibration"]["expected_calibration_error_ece"])
    check_match("brier_score", brier, orig_c["calibration"]["brier_score"])
    check_match("nll", nll, orig_c["calibration"]["negative_log_likelihood"])

    reproduced_metrics["conditions"][cid] = {
        "condition_name": cname,
        "confusion_matrix": cm,
        "classification_metrics": {
            "accuracy": acc,
            "balanced_accuracy": bal_acc,
            "macro_f1": macro_f1,
            "per_class": {
                "normal": {"precision": float(prec[0]), "recall": float(rec[0]), "f1": float(f1[0]), "support": int(sup[0])},
                "warning": {"precision": float(prec[1]), "recall": float(rec[1]), "f1": float(f1[1]), "support": int(sup[1])},
                "critical": {"precision": float(prec[2]), "recall": float(rec[2]), "f1": float(f1[2]), "support": int(sup[2])}
            }
        },
        "operational_metrics": {
            "episode_recall": {
                "detected": detected_episodes,
                "total": total_episodes,
                "percentage": episode_recall_pct
            },
            "critical_recall": {
                "correct": crit_correct,
                "total": crit_total,
                "percentage": critical_recall * 100.0
            },
            "pure_normal_entity_far": {
                "fa_entities": fa_entities,
                "total_entities": n_pure_normal_entities,
                "percentage": entity_far_pct
            },
            "normal_window_fpr": {
                "fp_windows": fp_windows,
                "total_windows": n_norm,
                "percentage": window_fpr_pct
            },
            "pure_normal_parent_job_far": {
                "fa_jobs": fa_jobs,
                "total_jobs": n_pure_normal_jobs,
                "percentage": job_far_pct
            }
        },
        "calibration_metrics": {
            "brier_score": brier,
            "negative_log_likelihood": nll,
            "ece": ece
        },
        "discrepancies_with_original_report": discrepancies,
        "reproduction_exact_match": len(discrepancies) == 0
    }

# Save output JSONs
audit_out_dir = PROJECT_ROOT / "results" / "phase7i_audit"
audit_out_dir.mkdir(parents=True, exist_ok=True)

with open(audit_out_dir / "validation_ranking_metrics.json", "w") as f:
    json.dump(ranking_audit, f, indent=2)
print(f"Saved {audit_out_dir / 'validation_ranking_metrics.json'}")

with open(audit_out_dir / "metric_reproduction.json", "w") as f:
    json.dump(reproduced_metrics, f, indent=2)
print(f"Saved {audit_out_dir / 'metric_reproduction.json'}")

print("\n=== RANKING AUDIT SUMMARY ===")
for cid in conditions:
    r = ranking_audit["conditions"][cid]["ranking_discrimination"]
    p = ranking_audit["conditions"][cid]["probability_diagnostics"]
    print(f"\n{condition_names[cid]}:")
    print(f"  ROC-AUC (Fail vs Norm): {r['roc_auc_binary_failure_vs_normal']:.4f}")
    print(f"  PR-AUC (Average Precision): {r['pr_auc_average_precision']:.4f} (baseline prevalence: {r['positive_prevalence_pct']:.2f}%)")
    print(f"  Pairwise % fail ranked > norm: {r['pairwise_fail_above_normal_pct']:.2f}%")
    print(f"  Critical vs Normal gap: {p['separation_gap_critical_minus_normal']:.4f}")
    print(f"  Failure vs Normal gap: {p['separation_gap_failure_minus_normal']:.4f}")
    print(f"  Reproduction match: {reproduced_metrics['conditions'][cid]['reproduction_exact_match']}")
