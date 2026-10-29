import json

data = json.loads(open("results/phase7e/validation_comparison_metrics.json").read())

print("=== DETAILED VALIDATION METRICS ACROSS ALL 6 MODELS ===")
for m_id, m in data.items():
    print("=" * 60)
    print(f"MODEL: {m_id}")
    print(f"Macro-F1: {m['macro_f1']:.4f} | Acc: {m['accuracy']:.4f} | BalAcc: {m['balanced_accuracy']:.4f}")
    print(f"Confusion Matrix: {m['confusion_matrix']}")
    print("Per-class metrics:")
    for cname, cvals in m["class_metrics"].items():
        print(f"  {cname:<10}: P={cvals['precision']:.4f}, R={cvals['recall']:.4f}, F1={cvals['f1']:.4f}, Support={cvals['support']}")
    print(f"Episode Recall: {m['episode_metrics']['detected_episodes']}/{m['episode_metrics']['total_episodes']} ({m['episode_metrics']['episode_recall_pct']:.2f}%)")
    print(f"Window FPR: {m['window_fpr']['false_positive_windows']}/{m['window_fpr']['denominator_windows']} ({m['window_fpr']['fpr_pct']:.2f}%)")
    print(f"Entity FAR: {m['entity_far']['false_alarm_entities']}/{m['entity_far']['denominator_entities']} ({m['entity_far']['far_pct']:.2f}%)")
    print(f"Parent-Job FAR: {m['parent_job_far']['false_alarm_jobs']}/{m['parent_job_far']['denominator_jobs']} ({m['parent_job_far']['far_pct']:.2f}%)")
