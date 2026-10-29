import json
from pathlib import Path

b_val = json.loads(Path("results/phase7b/validation_metrics.json").read_text())
d_val = json.loads(Path("results/phase7d/validation_metrics.json").read_text())
b_test = json.loads(Path("results/phase7b/test_evaluation_results.json").read_text())
d_test = json.loads(Path("results/phase7d/test_evaluation_results.json").read_text())

print("=== VALIDATION COMPARISON ===")
for m in ["random_forest", "temporal_lstm", "compact_transformer", "representation_hybrid"]:
    bm = b_val["models"][m]
    dm = d_val["models"][m]
    print(f"\nModel: {m}")
    print(f"  Macro-F1:       7B = {bm['macro_f1']:.4f}  -->  7D = {dm['macro_f1']:.4f}  (diff: {dm['macro_f1'] - bm['macro_f1']:+.4f})")
    print(f"  Accuracy:       7B = {bm['accuracy']:.4f}  -->  7D = {dm['accuracy']:.4f}  (diff: {dm['accuracy'] - bm['accuracy']:+.4f})")
    print(f"  Critical Rec:   7B = {bm['critical_recall']:.4f}  -->  7D = {dm['critical_recall']:.4f}  (diff: {dm['critical_recall'] - bm['critical_recall']:+.4f})")
    print(f"  Warning Rec:    7B = {bm['warning_recall']:.4f}  -->  7D = {dm['warning_recall']:.4f}  (diff: {dm['warning_recall'] - bm['warning_recall']:+.4f})")
    print(f"  Episode Rec %:  7B = {bm['episode_metrics']['episode_recall_pct']:.2f}% -->  7D = {dm['episode_metrics']['episode_recall_pct']:.2f}%")
    print(f"  Entity FAR %:   7B = {bm['entity_far']['far_pct']:.2f}% -->  7D = {dm['entity_far']['far_pct']:.2f}%")
    print(f"  Window FPR %:   7B = {bm['window_fpr']['fpr_pct']:.2f}% -->  7D = {dm['window_fpr']['fpr_pct']:.2f}%")
    print(f"  Parent Job FAR%:7B = {bm['parent_job_far']['far_pct']:.2f}% -->  7D = {dm['parent_job_far']['far_pct']:.2f}%")

print("\n=== TEST COMPARISON ===")
for m in ["random_forest", "temporal_lstm", "compact_transformer", "representation_hybrid"]:
    bm = b_test["models"][m]
    dm = d_test["models"][m]
    print(f"\nModel: {m}")
    print(f"  Macro-F1:       7B = {bm['macro_f1']:.4f}  -->  7D = {dm['macro_f1']:.4f}  (diff: {dm['macro_f1'] - bm['macro_f1']:+.4f})")
    print(f"  Accuracy:       7B = {bm['accuracy']:.4f}  -->  7D = {dm['accuracy']:.4f}  (diff: {dm['accuracy'] - bm['accuracy']:+.4f})")
    print(f"  Critical Rec:   7B = {bm['critical_recall']:.4f}  -->  7D = {dm['critical_recall']:.4f}  (diff: {dm['critical_recall'] - bm['critical_recall']:+.4f})")
    print(f"  Warning Rec:    7B = {bm['warning_recall']:.4f}  -->  7D = {dm['warning_recall']:.4f}  (diff: {dm['warning_recall'] - bm['critical_recall']:+.4f})")
    print(f"  Episode Rec %:  7B = {bm['episode_metrics']['episode_recall_pct']:.2f}% -->  7D = {dm['episode_metrics']['episode_recall_pct']:.2f}%")
    print(f"  Entity FAR %:   7B = {bm['entity_far']['far_pct']:.2f}% -->  7D = {dm['entity_far']['far_pct']:.2f}%")
    print(f"  Window FPR %:   7B = {bm['window_fpr']['fpr_pct']:.2f}% -->  7D = {dm['window_fpr']['fpr_pct']:.2f}%")
    print(f"  Parent Job FAR%:7B = {bm['parent_job_far']['far_pct']:.2f}% -->  7D = {dm['parent_job_far']['far_pct']:.2f}%")
