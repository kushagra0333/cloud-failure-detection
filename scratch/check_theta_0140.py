import pandas as pd
import numpy as np

df = pd.read_csv("results/phase7i/phase7i_validation_predictions.csv")
df["fail_timestamp"] = (df["prediction_timestamp"] + df["time_to_failure"]).round(1)

y_true = df["y_true"].values
prefails = df[df["y_true"] > 0]
episodes = prefails.groupby(["entity_id", "fail_timestamp"])

failing_eids = set(prefails["entity_id"].unique())
all_eids = sorted(df["entity_id"].unique())
pure_norm_eids = sorted(list(set(all_eids) - failing_eids))

p_fail_A = df["p_fail_condition_A"].values
p1_A = df["prob_condition_A_c1"].values
p2_A = df["prob_condition_A_c2"].values

th = 0.140
alert = (p_fail_A >= th)

det_eps = sum(1 for _, grp in episodes if (p_fail_A[grp.index] >= th).any())
ep_rec = det_eps / len(episodes) * 100.0

crit_inter = int(alert[y_true == 2].sum())
crit_inter_pct = crit_inter / (y_true == 2).sum() * 100.0

fa_ents = sum(1 for eid in pure_norm_eids if (p_fail_A[df["entity_id"] == eid] >= th).any())
ent_far = fa_ents / len(pure_norm_eids) * 100.0

fp_wins = int(alert[y_true == 0].sum())
win_fpr = fp_wins / (y_true == 0).sum() * 100.0

# Cascaded policy B
mask_crit = alert & (p2_A >= p1_A)
crit_b = int(((y_true == 2) & mask_crit).sum())
crit_b_rec = crit_b / (y_true == 2).sum() * 100.0

print(f"Condition A at theta={th}:")
print(f"  Episode Recall: {ep_rec:.1f}% ({det_eps}/{len(episodes)})")
print(f"  Binary Critical Interception (Policy A): {crit_inter_pct:.1f}% ({crit_inter}/{(y_true == 2).sum()})")
print(f"  Cascaded Critical Recall (Policy B): {crit_b_rec:.1f}% ({crit_b}/{(y_true == 2).sum()})")
print(f"  Pure-Normal Entity FAR: {ent_far:.2f}% ({fa_ents}/{len(pure_norm_eids)})")
print(f"  Normal-Window FPR: {win_fpr:.2f}% ({fp_wins}/{(y_true == 0).sum()})")

# Now check saved CSV in results/phase7j/phase7j_threshold_metrics.csv
df_saved = pd.read_csv("results/phase7j/phase7j_threshold_metrics.csv")
row_saved = df_saved[(df_saved["condition_id"] == "condition_A") & (df_saved["threshold"] == 0.140)].iloc[0]
print("\nSaved in results/phase7j/phase7j_threshold_metrics.csv:")
print(f"  policyA_episode_recall_pct: {row_saved['policyA_episode_recall_pct']}")
print(f"  policyA_crit_interception_pct: {row_saved['policyA_crit_interception_pct']}")
print(f"  policyB_critical_recall_pct: {row_saved['policyB_critical_recall_pct']}")
print(f"  policyA_entity_far_pct: {row_saved['policyA_entity_far_pct']}")
print(f"  policyA_window_fpr_pct: {row_saved['policyA_window_fpr_pct']}")
print(f"  policyB_macro_f1: {row_saved['policyB_macro_f1']}")

# Check why preliminary sweep had:
# Episode Recall 85%, Critical Recall 59.2%, Entity FAR 63.3%, Window FPR 46.6%
# Could 59.2% be Binary Critical Interception? 29 / 49 = 59.18%!
print(f"\nNotice: 29 / 49 = {29 / 49 * 100.0:.2f}%!")
print(f"26 / 49 = {26 / 49 * 100.0:.2f}%!")
print(f"How many critical windows have alert at theta=0.140? crit_inter = {crit_inter} ({crit_inter/49*100:.2f}%)")
print(f"How many critical windows have alert AND p2 >= p1 at theta=0.140? crit_b = {crit_b} ({crit_b/49*100:.2f}%)")
# What about 29 critical windows? At what threshold is crit_inter == 29 or crit_b == 29?
for t in np.linspace(0.12, 0.20, 81):
    a = (p_fail_A >= t)
    ci = int(a[y_true == 2].sum())
    cb = int(((y_true == 2) & a & (p2_A >= p1_A)).sum())
    if ci == 29 or cb == 29:
        print(f"  t={t:.4f}: Binary Interception={ci}/49 ({ci/49*100:.2f}%), Cascaded Crit Recall={cb}/49 ({cb/49*100:.2f}%)")
