import json
from pathlib import Path

def inspect_model_metrics():
    # 1. Phase 7D metrics
    p7d_json = Path("results/phase7d/confirmatory_evaluation_results.json")
    if p7d_json.exists():
        with open(p7d_json) as f:
            data = json.load(f)
        print("=== Phase 7D Results ===")
        # Check validation candidates
        if "validation_summary" in data:
            print("Validation Summary:", json.dumps(data["validation_summary"], indent=2))
        elif "candidates" in data:
            for c in data["candidates"]:
                print(f"Candidate: {c.get('name')}, Val Macro-F1: {c.get('val_macro_f1')}")
        if "test_results" in data:
            print("Test Results:", json.dumps(data["test_results"], indent=2))

    # 2. Phase 7E metrics
    p7e_json = Path("results/phase7e/phase7e_metrics.json")
    if p7e_json.exists():
        with open(p7e_json) as f:
            data = json.load(f)
        print("\n=== Phase 7E Results ===")
        print(json.dumps(data, indent=2)[:500])

    # 3. Phase 7G metrics
    p7g_json = Path("results/phase7g/decision_evaluation_results.json")
    if p7g_json.exists():
        with open(p7g_json) as f:
            data = json.load(f)
        print("\n=== Phase 7G Results ===")
        print(json.dumps(data, indent=2)[:500])

    # 4. Phase 7K metrics (Ablation)
    p7k_json = Path("results/phase7k/feature_ablation_results.json")
    if p7k_json.exists():
        with open(p7k_json) as f:
            data = json.load(f)
        print("\n=== Phase 7K Results ===")
        print(json.dumps(data, indent=2)[:500])

    # 5. Phase 7S metrics (Relative Telemetry)
    p7s_json = Path("results/phase7s_relative_ablation/phase7s_final_metrics.json")
    if p7s_json.exists():
        with open(p7s_json) as f:
            data = json.load(f)
        print("\n=== Phase 7S Results ===")
        print(json.dumps(data, indent=2)[:500])

    # 6. Phase 7T metrics (Hybrid Risk Blend)
    p7t_json = Path("results/phase7t_hybrid_risk/phase7t_final_metrics.json")
    if p7t_json.exists():
        with open(p7t_json) as f:
            data = json.load(f)
        print("\n=== Phase 7T Results ===")
        print(json.dumps(data, indent=2)[:500])

    # 7. Phase 7U metrics (Fuzzy Inference)
    p7u_json = Path("results/phase7u_fuzzy_lstm/phase7u_final_metrics.json")
    if p7u_json.exists():
        with open(p7u_json) as f:
            data = json.load(f)
        print("\n=== Phase 7U Results ===")
        print(json.dumps(data, indent=2)[:500])

if __name__ == "__main__":
    inspect_model_metrics()

