import json
import pandas as pd
from pathlib import Path

def extract_all_gate_summaries():
    results_dir = Path("results")
    
    print("=== Phase 7D Test Results ===")
    p7d_test = results_dir / "phase7d" / "test_evaluation_results.json"
    if p7d_test.exists():
        with open(p7d_test) as f:
            d = json.load(f)
        for k, v in d.items():
            if isinstance(v, dict):
                print(f"Model: {k}")
                for mk, mv in v.items():
                    if any(x in mk.lower() for x in ["macro", "recall", "far", "fpr", "accuracy"]):
                        print(f"  {mk}: {mv}")

    print("\n=== Operational Gate Summaries across Phases ===")
    phases = [
        "phase7j", "phase7k", "phase7l_longer_context", "phase7m_multiresolution",
        "phase7n_critical_margin", "phase7p_failure_calibration", "phase7q_signal_audit",
        "phase7r_score_forensics", "phase7s_relative_ablation", "phase7t_hybrid_risk", "phase7u_fuzzy_lstm"
    ]
    for p in phases:
        d = results_dir / p
        csv_files = list(d.glob("*gate_summary*.csv"))
        for c in csv_files:
            print(f"\n--- {p} :: {c.name} ---")
            df = pd.read_csv(c)
            print(df.to_string(index=False))

if __name__ == "__main__":
    extract_all_gate_summaries()

