import json
from pathlib import Path

ci_data = json.loads(Path("results/phase7d/bootstrap_ci.json").read_text())

print("=== BOOTSTRAP CI FOR ALL MODELS (Parent Job Cluster Bootstrap) ===")
for m_key in ["random_forest", "temporal_lstm", "compact_transformer", "representation_hybrid"]:
    print(f"\nModel: {m_key}")
    m = ci_data["parent_job_cluster_bootstrap"][m_key]["metrics"]
    for k in ["macro_f1", "accuracy", "critical_recall", "episode_recall_pct", "entity_far_pct", "window_fpr_pct"]:
        ci = m[k]["ci_95"]
        ci_str = f"[{ci[0]:.4f}, {ci[1]:.4f}]" if ci else "None"
        mean_str = f"{m[k]['mean']:.4f}" if m[k]['mean'] is not None else "None"
        print(f"  {k:<20}: Mean = {mean_str} | 95% CI = {ci_str}")
