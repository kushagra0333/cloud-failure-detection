import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import f1_score, balanced_accuracy_score, accuracy_score, confusion_matrix
import hashlib
import json

PROJECT_ROOT = Path(".").resolve()
results_dir = PROJECT_ROOT / "results" / "phase7m_multiresolution"
reports_dir = PROJECT_ROOT / "reports" / "phase7m_multiresolution"

data_dir = results_dir / "processed_data"
y_val = np.load(data_dir / "y_val.npy")
val_meta = pd.read_csv(data_dir / "val_metadata.csv")
df_val_preds = pd.read_csv(results_dir / "validation_predictions.csv")

n_val = len(y_val)
n_norm = int((y_val == 0).sum())
n_warn = int((y_val == 1).sum())
n_crit = int((y_val == 2).sum())

# Identify failure episodes and pure-normal entities/jobs
# Failure episode: entity_id + failure_timestamp (excluding NaN)
fail_mask = val_meta["failure_timestamp"].notna() & (val_meta["failure_timestamp"] > 0)
episodes = list(val_meta[fail_mask].groupby(["entity_id", "failure_timestamp"]))
total_episodes = len(episodes) # 40

entity_max_label = val_meta.groupby("entity_id")["label"].max()
pure_normal_eids = set(entity_max_label[entity_max_label == 0].index)
n_pure_normal_entities = len(pure_normal_eids) # 59

job_max_label = val_meta.groupby("parent_job_id")["label"].max()
pure_normal_jobs = set(job_max_label[job_max_label == 0].index)
n_pure_normal_jobs = len(pure_normal_jobs) # 15

seeds = [42, 123, 456]
eval_model_tags = []
for mid in ["M0", "M1", "M2"]:
    for cid in ["cond_B", "cond_A"]:
        for s in seeds:
            eval_model_tags.append((mid, cid, s, f"{mid}_{cid}_seed{s}"))

threshold_sweep_records = []
thresholds = np.linspace(0.0, 1.0, 201)

for mid, cid, s, run_tag in eval_model_tags:
    p0 = df_val_preds[f"prob_{run_tag}_c0"].values
    p1 = df_val_preds[f"prob_{run_tag}_c1"].values
    p2 = df_val_preds[f"prob_{run_tag}_c2"].values
    p_fail = df_val_preds[f"p_fail_{run_tag}"].values

    for th in thresholds:
        th_val = round(float(th), 4)

        # Policy A: Binary alert
        alert_a = (p_fail >= th_val).astype(int)
        det_eps_a = sum(1 for _, grp in episodes if (alert_a[grp.index] > 0).any())
        ep_rec_a = det_eps_a / total_episodes * 100.0
        crit_inter_a = int(alert_a[y_val == 2].sum()) / n_crit * 100.0
        fa_ents_a = sum(1 for eid in pure_normal_eids if (alert_a[val_meta["entity_id"] == eid] > 0).any()) / n_pure_normal_entities * 100.0
        win_fpr_a = int(alert_a[y_val == 0].sum()) / n_norm * 100.0

        # Policy B: Cascaded 3-Class Decision
        pred_b = np.zeros(n_val, dtype=int)
        mask_alert = (p_fail >= th_val)
        mask_crit = mask_alert & (p2 >= p1)
        mask_warn = mask_alert & (p1 > p2)
        pred_b[mask_crit] = 2
        pred_b[mask_warn] = 1

        det_eps_b = sum(1 for _, grp in episodes if (pred_b[grp.index] > 0).any())
        ep_rec_b = det_eps_b / total_episodes * 100.0
        cm_b = confusion_matrix(y_val, pred_b, labels=[0, 1, 2])
        crit_rec_b = float(cm_b[2, 2] / n_crit * 100.0)
        warn_rec_b = float(cm_b[1, 1] / n_warn * 100.0)
        norm_rec_b = float(cm_b[0, 0] / n_norm * 100.0)

        fa_ents_b = sum(1 for eid in pure_normal_eids if (pred_b[val_meta["entity_id"] == eid] > 0).any()) / n_pure_normal_entities * 100.0
        win_fpr_b = int((pred_b[y_val == 0] > 0).sum()) / n_norm * 100.0
        fa_jobs_b = sum(1 for jid in pure_normal_jobs if (pred_b[val_meta["parent_job_id"] == jid] > 0).any()) / n_pure_normal_jobs * 100.0

        macro_f1_b = float(f1_score(y_val, pred_b, average="macro", zero_division=0.0))
        bal_acc_b = float(balanced_accuracy_score(y_val, pred_b))
        acc_b = float(accuracy_score(y_val, pred_b))

        g1_pass = bool(ep_rec_b >= 70.0 and crit_rec_b >= 50.0)
        g2_pass = bool(fa_ents_b <= 10.0 and win_fpr_b <= 5.0)
        all_pass = bool(g1_pass and g2_pass)

        row_th = {
            "run_tag": run_tag, "model_id": mid, "condition_id": cid, "seed": s, "threshold": th_val,
            "polA_ep_rec": ep_rec_a, "polA_crit_inter": crit_inter_a, "polA_ent_far": fa_ents_a, "polA_win_fpr": win_fpr_a,
            "polB_macro_f1": macro_f1_b, "polB_bal_acc": bal_acc_b, "polB_acc": acc_b,
            "polB_ep_rec": ep_rec_b, "polB_crit_rec": crit_rec_b, "polB_warn_rec": warn_rec_b, "polB_norm_rec": norm_rec_b,
            "polB_ent_far": fa_ents_b, "polB_win_fpr": win_fpr_b, "polB_job_far": fa_jobs_b,
            "gate1_pass": g1_pass, "gate2_pass": g2_pass, "all_gates_pass": all_pass,
            "cm_00": int(cm_b[0, 0]), "cm_01": int(cm_b[0, 1]), "cm_02": int(cm_b[0, 2]),
            "cm_10": int(cm_b[1, 0]), "cm_11": int(cm_b[1, 1]), "cm_12": int(cm_b[1, 2]),
            "cm_20": int(cm_b[2, 0]), "cm_21": int(cm_b[2, 1]), "cm_22": int(cm_b[2, 2]),
        }
        threshold_sweep_records.append(row_th)

df_th_sweep = pd.DataFrame(threshold_sweep_records)
print(f"Generated threshold sweep: {len(df_th_sweep)} rows")
df_th_sweep.to_csv(results_dir / "threshold_sweep.csv", index=False)
df_th_sweep.to_csv(reports_dir / "threshold_sweep.csv", index=False)

# Re-run manifest hash calculation
def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

manifest_path = results_dir / "phase7m_manifest.json"
with open(manifest_path, "r") as f:
    manifest = json.load(f)

for prefix, folder in [("results", results_dir), ("reports", reports_dir)]:
    target_rel = f"{prefix}/phase7m_multiresolution/threshold_sweep.csv"
    p = folder / "threshold_sweep.csv"
    manifest["deliverables"][target_rel]["sha256"] = sha256(p)
    manifest["deliverables"][target_rel]["size_bytes"] = p.stat().st_size

with open(results_dir / "phase7m_manifest.json", "w") as f:
    json.dump(manifest, f, indent=2)

with open(reports_dir / "phase7m_manifest.json", "w") as f:
    json.dump(manifest, f, indent=2)

print("Updated threshold_sweep.csv and phase7m_manifest.json successfully.")
