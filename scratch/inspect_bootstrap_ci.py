import json
from pathlib import Path

ci_data = json.loads(Path("results/phase7d/bootstrap_ci.json").read_text())

print("=== BOOTSTRAP CI FOR M* (Random Forest) ===")
for cluster_type in ["parent_job_cluster_bootstrap", "entity_cluster_bootstrap"]:
    print(f"\nCluster Type: {cluster_type}")
    m = ci_data[cluster_type]["random_forest"]["metrics"]
    for k, v in m.items():
        ci = v["ci_95"]
        ci_str = f"[{ci[0]:.4f}, {ci[1]:.4f}]" if ci else "None"
        mean_str = f"{v['mean']:.4f}" if v['mean'] is not None else "None"
        print(f"  {k:<20}: Mean = {mean_str} | 95% CI = {ci_str} | Valid Replicates = {v['valid_replicates']}/1000 ({v['valid_fraction']*100:.1f}%)")

print("\n=== REPLICATE WINDOW STATS FOR M* ===")
for cluster_type in ["parent_job_cluster_bootstrap", "entity_cluster_bootstrap"]:
    stats = ci_data[cluster_type]["random_forest"]["replicate_window_stats"]
    print(f"{cluster_type}: min={stats['min']}, max={stats['max']}, mean={stats['mean']:.1f}, median={stats['median']:.1f}")
