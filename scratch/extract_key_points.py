import pandas as pd
df = pd.read_csv('results/phase7j_audit/independent_threshold_metrics.csv')

key_thetas = {
    'condition_A': [0.140, 0.205, 0.215],
    'condition_B': [0.255, 0.325, 0.355],
    'condition_C': [0.410, 0.490],
    'condition_D': [0.510, 0.625]
}

for cond, ths in key_thetas.items():
    print(f"=== {cond} ===")
    sub = df[df['condition_id'] == cond]
    for th in ths:
        row = sub[abs(sub['threshold'] - th) < 1e-4]
        if not row.empty:
            r = row.iloc[0]
            print(f"th={th:.3f}:")
            print(f"  Policy A: EpRec={r['policyA_episode_recall_pct']:.2f}% ({r['policyA_episodes_detected']}/{r['policyA_episodes_total']}), CritInt={r['policyA_crit_interception_pct']:.2f}% ({r['policyA_crit_intercepted']}/{r['policyA_crit_total']}), EntFAR={r['policyA_entity_far_pct']:.2f}% ({r['policyA_fa_entities']}/{r['policyA_pure_normal_entities_total']}), WinFPR={r['policyA_window_fpr_pct']:.2f}% ({r['policyA_fp_windows']}/{r['policyA_normal_windows_total']}), Prec={r['policyA_binary_precision']:.4f}, Rec={r['policyA_binary_recall']:.4f}, F1={r['policyA_binary_f1']:.4f}")
            print(f"  Policy B: EpRec={r['policyB_episode_recall_pct']:.2f}% ({r['policyB_episodes_detected']}/{r['policyB_episodes_total']}), CritRec={r['policyB_critical_recall_pct']:.2f}% ({r['policyB_crit_correct']}/{r['policyB_crit_total']}), WarnRec={r['policyB_warning_recall_pct']:.2f}% ({r['policyB_warn_correct']}/{r['policyB_warn_total']}), EntFAR={r['policyB_entity_far_pct']:.2f}% ({r['policyB_fa_entities']}/{r['policyB_pure_normal_entities_total']}), WinFPR={r['policyB_window_fpr_pct']:.2f}% ({r['policyB_fp_windows']}/{r['policyB_normal_windows_total']}), MacroF1={r['policyB_macro_f1']:.4f}, BalAcc={r['policyB_balanced_accuracy']:.4f}")
            print(f"  Confusion Matrix B (rows=true 0,1,2; cols=pred 0,1,2):")
            print(f"    Normal:   [{r['policyB_cm_00']:3d}, {r['policyB_cm_01']:3d}, {r['policyB_cm_02']:3d}] (total {r['policyB_norm_total']})")
            print(f"    Warning:  [{r['policyB_cm_10']:3d}, {r['policyB_cm_11']:3d}, {r['policyB_cm_12']:3d}] (total {r['policyB_warn_total']})")
            print(f"    Critical: [{r['policyB_cm_20']:3d}, {r['policyB_cm_21']:3d}, {r['policyB_cm_22']:3d}] (total {r['policyB_crit_total']})")
            print(f"  Gate Status: Gate1_Pass={r['policyB_gate1_pass']}, Gate2_Pass={r['policyB_gate2_pass']}, All_Pass={r['policyB_all_gates_pass']}")
