import json
import re
from pathlib import Path

def extract_experiment_info():
    reports_dir = Path("reports")
    
    experiments = [
        {"phase": "Phase 7D", "dir": "phase7d", "report": "phase7d_retraining_and_evaluation_report.md"},
        {"phase": "Phase 7E", "dir": "phase7e", "report": "phase7e_class_weighting_experiment_report.md"},
        {"phase": "Phase 7F", "dir": "phase7f", "report": "phase7f_failure_separation_report.md"},
        {"phase": "Phase 7G", "dir": "phase7g", "report": "phase7g_hierarchical_baseline_report.md"},
        {"phase": "Phase 7H", "dir": "phase7h", "report": "phase7h_critical_and_normalization_audit_report.md"},
        {"phase": "Phase 7I", "dir": "phase7i", "report": "phase7i_report.md"},
        {"phase": "Phase 7J", "dir": "phase7j", "report": "phase7j_report.md"},
        {"phase": "Phase 7K", "dir": "phase7k", "report": "phase7k_report.md"},
        {"phase": "Phase 7L", "dir": "phase7l_longer_context", "report": "phase7l_report.md"},
        {"phase": "Phase 7M", "dir": "phase7m_multiresolution", "report": "phase7m_final_report.md"},
        {"phase": "Phase 7N", "dir": "phase7n_critical_margin", "report": "phase7n_final_report.md"},
        {"phase": "Phase 7P", "dir": "phase7p_failure_calibration", "report": "phase7p_final_report.md"},
        {"phase": "Phase 7Q", "dir": "phase7q_signal_audit", "report": "phase7q_final_report.md"},
        {"phase": "Phase 7R", "dir": "phase7r_score_forensics", "report": "phase7r_final_report.md"},
        {"phase": "Phase 7S", "dir": "phase7s_relative_ablation", "report": "phase7s_final_report.md"},
        {"phase": "Phase 7T", "dir": "phase7t_hybrid_risk", "report": "phase7t_final_report.md"},
        {"phase": "Phase 7U", "dir": "phase7u_fuzzy_lstm", "report": "phase7u_final_report.md"},
    ]

    for exp in experiments:
        p = reports_dir / exp["dir"] / exp["report"]
        if not p.exists():
            print(f"Missing report for {exp['phase']}: {p}")
            continue
        text = p.read_text(encoding="utf-8")
        print(f"\n=======================================================")
        print(f"[{exp['phase']}] {p.name}")
        print(f"=======================================================")
        
        # Look for model, datasets, metrics in text
        for line in text.splitlines():
            if any(k in line.lower() for k in ["macro-f1", "episode recall", "critical recall", "far", "fpr", "threshold", "operational candidate", "winner", "verdict"]):
                if len(line.strip()) < 150:
                    print("  ", line.strip())

if __name__ == "__main__":
    extract_experiment_info()

