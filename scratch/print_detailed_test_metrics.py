import json
from pathlib import Path

res = json.loads(Path("results/phase7d/test_evaluation_results.json").read_text())

for m_key, m_val in res["models"].items():
    print("=" * 60)
    print(f"MODEL: {m_key}")
    print("=" * 60)
    print(f"Macro-F1: {m_val['macro_f1']:.4f} | Accuracy: {m_val['accuracy']:.4f}")
    print("Class Metrics:")
    for cname, cmetrics in m_val["class_metrics"].items():
        print(f"  {cname:<10}: P={cmetrics['precision']:.4f}, R={cmetrics['recall']:.4f}, F1={cmetrics['f1']:.4f}, Support={cmetrics['support']}")
    print("PR AUC:")
    for k, v in m_val.get("pr_auc", {}).items():
        print(f"  {k}: {v:.4f}")
    print(f"Episode Recall: {m_val['episode_metrics']['detected_episodes']}/{m_val['episode_metrics']['total_episodes']} ({m_val['episode_metrics']['episode_recall_pct']:.2f}%)")
    print(f"Lead Time Mean: {m_val['episode_metrics']['lead_time_mean_sec']:.1f}s ({m_val['episode_metrics']['lead_time_mean_min']:.2f} min)")
    print(f"Entity FAR: {m_val['entity_far']['false_alarm_entities']}/{m_val['entity_far']['denominator_entities']} ({m_val['entity_far']['far_pct']:.2f}%)")
    print(f"Window FPR: {m_val['window_fpr']['false_positive_windows']}/{m_val['window_fpr']['denominator_windows']} ({m_val['window_fpr']['fpr_pct']:.2f}%)")
    print(f"Parent Job FAR: {m_val['parent_job_far']['false_alarm_jobs']}/{m_val['parent_job_far']['denominator_jobs']} ({m_val['parent_job_far']['far_pct']:.2f}%)")
