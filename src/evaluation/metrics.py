"""Comprehensive evaluation metrics for early failure prediction.

Computes:
1. Window-level metrics:
   - Per-class precision, recall, F1, and support (NORMAL, WARNING, CRITICAL)
   - Macro-F1 (primary metric)
   - Accuracy (secondary metric)
   - Warning recall & Critical recall
   - Confusion matrix
   - One-vs-rest PR-AUC for WARNING and CRITICAL

2. Failure-episode-level metrics:
   - Episode grouping via (entity_id, fail_timestamp)
   - Episode detection rate
   - Lead time distribution (mean, median, min, max in seconds and minutes)
   - False alarm rate on pure normal entities
   - Handles duplicate alerts and documents independence caveats
"""

from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_recall_fscore_support,
    f1_score,
    accuracy_score,
    confusion_matrix,
    average_precision_score,
)

CLASS_NAMES = {0: "NORMAL", 1: "WARNING", 2: "CRITICAL"}


def compute_window_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Computes comprehensive window-level classification metrics."""
    labels = [0, 1, 2]
    precisions, recalls, f1s, supports = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0.0
    )
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0.0))
    acc = float(accuracy_score(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()

    class_metrics = {}
    for i, c in enumerate(labels):
        cname = CLASS_NAMES[c]
        class_metrics[cname] = {
            "precision": float(precisions[i]),
            "recall": float(recalls[i]),
            "f1": float(f1s[i]),
            "support": int(supports[i]),
        }

    # PR-AUC / Average Precision calculation
    pr_auc_metrics = {}
    if y_prob is not None and y_prob.ndim == 2:
        for c in [1, 2]:
            cname = CLASS_NAMES[c]
            y_bin = (y_true == c).astype(int)
            prob_c = y_prob[:, c] if y_prob.shape[1] > c else np.zeros(len(y_true))

            # Handle constant probability outputs
            if np.all(prob_c == prob_c[0]):
                prevalence = float(np.mean(y_bin))
                pr_auc_metrics[f"pr_auc_{cname.lower()}"] = prevalence
                pr_auc_metrics[f"pr_auc_{cname.lower()}_is_prevalence"] = True
            else:
                score = float(average_precision_score(y_bin, prob_c))
                pr_auc_metrics[f"pr_auc_{cname.lower()}"] = score
                pr_auc_metrics[f"pr_auc_{cname.lower()}_is_prevalence"] = False
    else:
        pr_auc_metrics["pr_auc_warning"] = None
        pr_auc_metrics["pr_auc_critical"] = None

    return {
        "macro_f1": macro_f1,
        "accuracy": acc,
        "class_metrics": class_metrics,
        "warning_recall": float(recalls[1]),
        "critical_recall": float(recalls[2]),
        "confusion_matrix": cm,
        "pr_auc": pr_auc_metrics,
        "total_windows": len(y_true),
    }


def compute_episode_metrics(
    metadata_df: pd.DataFrame,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Computes failure-episode-level detection metrics and lead times.
    
    Grouping Definition:
    A failure episode is uniquely identified by (entity_id, fail_timestamp), where:
        fail_timestamp = round(prediction_timestamp + time_to_failure, 1)
    
    Detection Definition:
    An episode is detected if AT LEAST ONE pre-failure window for that episode predicts
    an alert (y_pred in {1: WARNING, 2: CRITICAL}).
    
    Lead Time Definition:
    Lead time = fail_timestamp - prediction_timestamp of the EARLIEST triggered alert window.
    Subsequent alerts are treated as confirmations and do not double-count the episode.
    """
    df = metadata_df.copy().reset_index(drop=True)
    df["y_pred"] = y_pred

    prefails = df[df["label"] > 0].copy()
    if prefails.empty:
        return {
            "total_episodes": 0,
            "detected_episodes": 0,
            "detection_rate_pct": 0.0,
            "lead_time_mean_sec": 0.0,
            "lead_time_median_sec": 0.0,
            "false_alarm_entity_rate_pct": 0.0,
        }

    prefails["fail_timestamp"] = (prefails["prediction_timestamp"] + prefails["time_to_failure"]).round(1)
    episodes = prefails.groupby(["entity_id", "fail_timestamp"])
    total_episodes = len(episodes)

    detected_episodes = 0
    lead_times_sec = []

    for (eid, ftime), grp in episodes:
        grp_sorted = grp.sort_values("prediction_timestamp")
        alert_windows = grp_sorted[grp_sorted["y_pred"] > 0]
        if not alert_windows.empty:
            detected_episodes += 1
            earliest_pred_time = float(alert_windows["prediction_timestamp"].iloc[0])
            lead_time = float(ftime - earliest_pred_time)
            lead_times_sec.append(lead_time)

    det_rate_pct = (detected_episodes / total_episodes) * 100.0 if total_episodes > 0 else 0.0
    mean_lead_sec = float(np.mean(lead_times_sec)) if lead_times_sec else 0.0
    median_lead_sec = float(np.median(lead_times_sec)) if lead_times_sec else 0.0
    min_lead_sec = float(np.min(lead_times_sec)) if lead_times_sec else 0.0
    max_lead_sec = float(np.max(lead_times_sec)) if lead_times_sec else 0.0

    # False alarm analysis on pure normal tasks
    failed_entity_ids = set(prefails["entity_id"].unique())
    all_entity_ids = set(df["entity_id"].unique())
    pure_normal_entities = all_entity_ids - failed_entity_ids

    false_alarm_entities = 0
    for eid in pure_normal_entities:
        ent_windows = df[df["entity_id"] == eid]
        if (ent_windows["y_pred"] > 0).any():
            false_alarm_entities += 1

    total_pure_normal = len(pure_normal_entities)
    far_rate_pct = (false_alarm_entities / total_pure_normal) * 100.0 if total_pure_normal > 0 else 0.0

    return {
        "total_episodes": total_episodes,
        "detected_episodes": detected_episodes,
        "detection_rate_pct": det_rate_pct,
        "lead_time_mean_sec": mean_lead_sec,
        "lead_time_mean_min": mean_lead_sec / 60.0,
        "lead_time_median_sec": median_lead_sec,
        "lead_time_median_min": median_lead_sec / 60.0,
        "lead_time_min_sec": min_lead_sec,
        "lead_time_max_sec": max_lead_sec,
        "total_pure_normal_entities": total_pure_normal,
        "false_alarm_entities": false_alarm_entities,
        "false_alarm_entity_rate_pct": far_rate_pct,
    }


def format_metrics_table(window_metrics: Dict[str, Any], episode_metrics: Optional[Dict[str, Any]] = None) -> str:
    """Formats metrics as a human-readable text table."""
    lines = []
    lines.append("-" * 72)
    lines.append(f"{'Class':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    lines.append("-" * 72)
    cm_dict = window_metrics["class_metrics"]
    for cname in ["NORMAL", "WARNING", "CRITICAL"]:
        m = cm_dict[cname]
        lines.append(f"{cname:<12} {m['precision']:<12.4f} {m['recall']:<12.4f} {m['f1']:<12.4f} {m['support']:<10,d}")
    lines.append("-" * 72)
    lines.append(f"Macro-F1 (Primary) : {window_metrics['macro_f1']:.4f}")
    lines.append(f"Accuracy (Secondary): {window_metrics['accuracy']:.4f}")
    lines.append(f"Warning Recall      : {window_metrics['warning_recall']:.4f}")
    lines.append(f"Critical Recall     : {window_metrics['critical_recall']:.4f}")
    if window_metrics["pr_auc"]["pr_auc_warning"] is not None:
        lines.append(f"Warning PR-AUC      : {window_metrics['pr_auc']['pr_auc_warning']:.4f}")
    if window_metrics["pr_auc"]["pr_auc_critical"] is not None:
        lines.append(f"Critical PR-AUC     : {window_metrics['pr_auc']['pr_auc_critical']:.4f}")
    lines.append(f"Confusion Matrix    : {window_metrics['confusion_matrix']}")

    if episode_metrics is not None:
        lines.append("\nEpisode-Level Metrics:")
        lines.append(f"  Total Episodes    : {episode_metrics['total_episodes']}")
        lines.append(f"  Detected Episodes : {episode_metrics['detected_episodes']} ({episode_metrics['detection_rate_pct']:.1f}%)")
        lines.append(f"  Mean Lead Time    : {episode_metrics['lead_time_mean_sec']:.1f}s ({episode_metrics['lead_time_mean_min']:.1f} min)")
        lines.append(f"  Median Lead Time  : {episode_metrics['lead_time_median_sec']:.1f}s ({episode_metrics['lead_time_median_min']:.1f} min)")
        lines.append(f"  False Alarm Rate  : {episode_metrics['false_alarm_entities']}/{episode_metrics['total_pure_normal_entities']} entities ({episode_metrics['false_alarm_entity_rate_pct']:.1f}%)")

    return "\n".join(lines)
