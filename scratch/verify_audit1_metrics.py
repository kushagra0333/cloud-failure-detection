import json
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, balanced_accuracy_score

df = pd.read_csv('results/phase7h/validation_predictions.csv')
print(f"Validation predictions rows: {len(df)}")
assert len(df) == 657, f"Expected 657 rows, got {len(df)}"

# Let's inspect unique entities and jobs
print(f"Unique entities: {df['entity_id'].nunique()}")
print(f"Unique parent jobs: {df['parent_job_id'].nunique()}")

# True labels
y_true = df['y_true'].values
assert (y_true == df['label'].values).all()

# Class supports
n_c0 = int((y_true == 0).sum())
n_c1 = int((y_true == 1).sum())
n_c2 = int((y_true == 2).sum())
print(f"Support: c0={n_c0}, c1={n_c1}, c2={n_c2}, total={n_c0 + n_c1 + n_c2}")

# Identify episodes
prefails = df[df['label'] > 0].copy()
prefails['fail_timestamp'] = (prefails['prediction_timestamp'] + prefails['time_to_failure']).round(1)
episodes = prefails.groupby(['entity_id', 'fail_timestamp'])
total_episodes = len(episodes)
print(f"Total failure episodes: {total_episodes}")

# Pure normal entities
failing_eids = set(prefails['entity_id'].unique())
all_eids = sorted(df['entity_id'].unique())
pure_normal_eids = sorted(list(set(all_eids) - failing_eids))
n_pure_normal_entities = len(pure_normal_eids)
print(f"Pure normal entities: {n_pure_normal_entities} (failing: {len(failing_eids)})")

# Pure normal parent jobs
failing_jobs = set(prefails['parent_job_id'].unique())
all_jobs = sorted(df['parent_job_id'].unique())
pure_normal_jobs = sorted(list(set(all_jobs) - failing_jobs))
n_pure_normal_jobs = len(pure_normal_jobs)
print(f"Pure normal jobs: {n_pure_normal_jobs} (failing: {len(failing_jobs)})")
print(f"Pure normal job list: {pure_normal_jobs}")

models = {
    'transformer_hierarchical_baseline': 'y_pred_transformer_hierarchical_baseline',
    'transformer_hierarchical_floored': 'y_pred_transformer_hierarchical_floored',
    'transformer_hierarchical_robust': 'y_pred_transformer_hierarchical_robust'
}

audit1_results = {}

for m_key, pred_col in models.items():
    y_pred = df[pred_col].values
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    assert cm.sum() == 657, f"CM sum {cm.sum()} != 657"
    
    p, r, f1, s = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1, 2], zero_division=0.0)
    macro_f1 = float(np.mean(f1))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    
    # Assert CM per-class recalls match
    cm_r = [cm[i, i] / cm[i].sum() for i in range(3)]
    for i in range(3):
        assert abs(cm_r[i] - r[i]) < 1e-9
        
    # Episode recall
    det_eps = 0
    for (eid, ftime), grp in episodes:
        if (y_pred[grp.index] > 0).any():
            det_eps += 1
    ep_rec = float(det_eps / total_episodes * 100.0)
    
    # Pure normal entity FAR
    fa_entities = sum(1 for eid in pure_normal_eids if (y_pred[df['entity_id'] == eid] > 0).any())
    ent_far = float(fa_entities / n_pure_normal_entities * 100.0)
    
    # Normal window FPR
    fp_windows = int((y_pred[y_true == 0] > 0).sum())
    win_fpr = float(fp_windows / n_c0 * 100.0)
    
    # Pure normal parent-job FAR
    fa_jobs = sum(1 for jid in pure_normal_jobs if (y_pred[df['parent_job_id'] == jid] > 0).any())
    job_far = float(fa_jobs / n_pure_normal_jobs * 100.0)
    
    print(f"\n--- Model: {m_key} ---")
    print(f"Confusion Matrix:\n{cm}")
    print(f"Macro-F1: {macro_f1:.4f}, Bal Acc: {bal_acc:.4f}")
    print(f"Recall per class: [0: {r[0]:.4f}, 1: {r[1]:.4f}, 2: {r[2]:.4f}]")
    print(f"Precision per class: [0: {p[0]:.4f}, 1: {p[1]:.4f}, 2: {p[2]:.4f}]")
    print(f"Episode Recall: {det_eps}/{total_episodes} = {ep_rec:.2f}%")
    print(f"Entity FAR: {fa_entities}/{n_pure_normal_entities} = {ent_far:.2f}%")
    print(f"Window FPR: {fp_windows}/{n_c0} = {win_fpr:.2f}%")
    print(f"Job FAR: {fa_jobs}/{n_pure_normal_jobs} = {job_far:.2f}%")
    
    audit1_results[m_key] = {
        'confusion_matrix': cm.tolist(),
        'cm_sum': int(cm.sum()),
        'support': s.tolist(),
        'precision': p.tolist(),
        'recall': r.tolist(),
        'f1': f1.tolist(),
        'macro_f1': macro_f1,
        'balanced_accuracy': bal_acc,
        'episode_metrics': {
            'total_episodes': total_episodes,
            'detected_episodes': det_eps,
            'episode_recall_pct': ep_rec
        },
        'false_alarm_metrics': {
            'pure_normal_entities': n_pure_normal_entities,
            'fa_entities': fa_entities,
            'entity_far_pct': ent_far,
            'normal_windows': n_c0,
            'fp_windows': fp_windows,
            'window_fpr_pct': win_fpr,
            'pure_normal_jobs': n_pure_normal_jobs,
            'fa_jobs': fa_jobs,
            'job_far_pct': job_far
        }
    }

with open('scratch/audit1_verification.json', 'w') as f:
    json.dump(audit1_results, f, indent=2)
print("\nAudit 1 script completed successfully.")
