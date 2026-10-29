"""Audit Macro-F1 values in phase7j_threshold_metrics.csv."""
import pandas as pd
import numpy as np

df = pd.read_csv("results/phase7j/phase7j_threshold_metrics.csv")

conditions = ["condition_A", "condition_B", "condition_C", "condition_D"]

for cid in conditions:
    df_c = df[df["condition_id"] == cid]
    # Find row with max policyB_macro_f1
    max_row = df_c.loc[df_c["policyB_macro_f1"].idxmax()]
    print(f"\n{cid} Peak Macro-F1 in phase7j_threshold_metrics.csv:")
    print(f"  Threshold: {max_row['threshold']}")
    print(f"  Macro-F1: {max_row['policyB_macro_f1']:.6f}")
    print(f"  Normal Recall: {max_row['policyB_normal_recall_pct']:.2f}%")
    print(f"  Warning Recall: {max_row['policyB_warning_recall_pct']:.2f}%")
    print(f"  Critical Recall: {max_row['policyB_critical_recall_pct']:.2f}%")
    print(f"  Episode Recall: {max_row['policyB_episode_recall_pct']:.2f}%")
    print(f"  Entity FAR: {max_row['policyB_entity_far_pct']:.2f}%")
    print(f"  Window FPR: {max_row['policyB_window_fpr_pct']:.2f}%")

    # Let's also check binary F1 (policy A)
    max_bin_row = df_c.loc[df_c["policyA_binary_f1"].idxmax()]
    print(f"  {cid} Peak Binary F1 (Policy A):")
    print(f"    Threshold: {max_bin_row['threshold']}")
    print(f"    Binary F1: {max_bin_row['policyA_binary_f1']:.6f}")

    # Check the specific thresholds mentioned in the prompt:
    # Condition A: 0.180 vs 0.205
    # Condition B: 0.300 vs 0.325
    # Condition C: 0.370 vs 0.410
    # Condition D: 0.485 vs 0.510
    th1 = 0.180 if cid == "condition_A" else 0.300 if cid == "condition_B" else 0.370 if cid == "condition_C" else 0.485
    th2 = 0.205 if cid == "condition_A" else 0.325 if cid == "condition_B" else 0.410 if cid == "condition_C" else 0.510
    
    r1 = df_c[df_c["threshold"] == th1].iloc[0]
    r2 = df_c[df_c["threshold"] == th2].iloc[0]
    print(f"  Comparison for {cid}:")
    print(f"    At th={th1:.3f}: Policy B Macro-F1 = {r1['policyB_macro_f1']:.4f}, Policy A Binary F1 = {r1['policyA_binary_f1']:.4f}")
    print(f"    At th={th2:.3f}: Policy B Macro-F1 = {r2['policyB_macro_f1']:.4f}, Policy A Binary F1 = {r2['policyA_binary_f1']:.4f}")
