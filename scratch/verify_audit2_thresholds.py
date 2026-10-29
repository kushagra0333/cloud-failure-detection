import json
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

df = pd.read_csv('results/phase7h/validation_predictions.csv')
y_true = df['y_true'].values
normal_mask = (y_true == 0)
warn_mask = (y_true == 1)
crit_mask = (y_true == 2)
n_c0 = int(normal_mask.sum())
n_c1 = int(warn_mask.sum())
n_c2 = int(crit_mask.sum())

prefails = df[df['label'] > 0].copy()
prefails['fail_timestamp'] = (prefails['prediction_timestamp'] + prefails['time_to_failure']).round(1)
episodes = prefails.groupby(['entity_id', 'fail_timestamp'])
total_episodes = len(episodes)

failing_eids = set(prefails['entity_id'].unique())
all_eids = sorted(df['entity_id'].unique())
pure_normal_eids = sorted(list(set(all_eids) - failing_eids))
n_pure_normal_entities = len(pure_normal_eids)

thresholds = np.round(np.arange(0.05, 0.96, 0.01), 2)

models = {
    'transformer_hierarchical_baseline': {
        'p_fail': df['p_fail_transformer_hierarchical_baseline'].values,
        'p_crit': df['prob_transformer_hierarchical_baseline_c2'].values,
        'p_warn': df['prob_transformer_hierarchical_baseline_c1'].values,
        'p_norm': df['prob_transformer_hierarchical_baseline_c0'].values
    },
    'transformer_hierarchical_floored': {
        'p_fail': df['p_fail_transformer_hierarchical_floored'].values,
        'p_crit': df['prob_transformer_hierarchical_floored_c2'].values,
        'p_warn': df['prob_transformer_hierarchical_floored_c1'].values,
        'p_norm': df['prob_transformer_hierarchical_floored_c0'].values
    },
    'transformer_hierarchical_robust': {
        'p_fail': df['p_fail_transformer_hierarchical_robust'].values,
        'p_crit': df['prob_transformer_hierarchical_robust_c2'].values,
        'p_warn': df['prob_transformer_hierarchical_robust_c1'].values,
        'p_norm': df['prob_transformer_hierarchical_robust_c0'].values
    }
}

# Load saved Phase 7H decision evaluation results
with open('results/phase7h/decision_evaluation_results.json', 'r') as f:
    saved_decision_results = json.load(f)

audit2_data = {}

for m_key, probs in models.items():
    p_fail = probs['p_fail']
    p_crit = probs['p_crit']
    p_warn = probs['p_warn']
    p_norm = probs['p_norm']
    
    saved_sweep = saved_decision_results[m_key]['threshold_sweep']
    
    # 1. Verify reproduction of Phase 7H binary sweep
    binary_records = []
    semantic_discrepancies = []
    
    for i, theta in enumerate(thresholds):
        bin_pred = (p_fail >= theta).astype(int)
        saved_row = saved_sweep[i]
        assert abs(saved_row['threshold'] - theta) < 1e-4
        
        # Recomputed metrics
        bin_crit_intercept = float(bin_pred[crit_mask].sum() / n_c2)
        reported_crit_rec = saved_row['critical_recall']
        
        # Check exact match
        assert abs(bin_crit_intercept - reported_crit_rec) < 1e-6, f"Mismatch at theta={theta}"
        
        # Genuine 3-class Critical prediction:
        # If policy uses P(Critical) >= theta to predict Critical (class 2)
        genuine_crit_pred = (p_crit >= theta).astype(int)
        genuine_crit_rec = float(genuine_crit_pred[crit_mask].sum() / n_c2)
        
        # Cascaded rule:
        # If P(Critical) >= theta_crit and P(fail) >= theta:
        # Let's check when P(Critical) > P(Warning) and P(fail) >= theta
        argmax_failure_pred = np.zeros(len(df), dtype=int)
        alert_mask = (p_fail >= theta)
        argmax_failure_pred[alert_mask & (p_crit > p_warn)] = 2
        argmax_failure_pred[alert_mask & (p_crit <= p_warn)] = 1
        cascaded_crit_rec = float((argmax_failure_pred[crit_mask] == 2).sum() / n_c2)
        
        fp_windows = int(bin_pred[normal_mask].sum())
        win_fpr = float(fp_windows / n_c0 * 100.0)
        fa_entities = sum(1 for eid in pure_normal_eids if (bin_pred[df['entity_id'] == eid] == 1).any())
        ent_far = float(fa_entities / n_pure_normal_entities * 100.0)
        
        det_eps = 0
        lead_times = []
        for (eid, ftime), grp in episodes:
            det_m = (bin_pred[grp.index] == 1)
            if det_m.any():
                det_eps += 1
                first_det_idx = grp.index[det_m][0]
                lead_times.append(ftime - df.loc[first_det_idx, 'prediction_timestamp'])
        ep_rec = float(det_eps / total_episodes * 100.0)
        
        rec = {
            'threshold': float(theta),
            'binary_critical_interception_rate': bin_crit_intercept,
            'genuine_critical_recall_at_theta': genuine_crit_rec,
            'cascaded_critical_recall_at_theta': cascaded_crit_rec,
            'phase7h_reported_as_critical_recall': reported_crit_rec,
            'episode_recall_pct': ep_rec,
            'entity_far_pct': ent_far,
            'window_fpr_pct': win_fpr,
            'mean_lead_time_sec': float(np.mean(lead_times)) if lead_times else 0.0
        }
        binary_records.append(rec)
        
        if abs(bin_crit_intercept - genuine_crit_rec) > 1e-4:
            semantic_discrepancies.append({
                'threshold': float(theta),
                'reported_critical_recall_was_binary': reported_crit_rec,
                'genuine_critical_recall': genuine_crit_rec,
                'gap': float(reported_crit_rec - genuine_crit_rec)
            })

    print(f"\nModel {m_key}: Verified {len(binary_records)} thresholds.")
    print(f"Max semantic gap between binary interception and genuine critical recall: "
          f"{max(d['gap'] for d in semantic_discrepancies):.4f}")
    
    # Check reported theta = 0.63 in Phase 7H
    row_63 = next(r for r in binary_records if abs(r['threshold'] - 0.63) < 1e-4)
    print(f"At theta=0.63: Reported 'Crit Recall' (Binary Interception): {row_63['phase7h_reported_as_critical_recall']:.4f}, "
          f"Genuine Crit Recall: {row_63['genuine_critical_recall_at_theta']:.4f}, "
          f"Cascaded Crit Recall: {row_63['cascaded_critical_recall_at_theta']:.4f}")
    print(f"Episode Recall: {row_63['episode_recall_pct']:.1f}%, Entity FAR: {row_63['entity_far_pct']:.1f}%, Window FPR: {row_63['window_fpr_pct']:.1f}%")
    
    # Check Gate-Constrained Maxima (under Gate 2: Entity FAR <= 10%, Window FPR <= 5%)
    gate2_records = [r for r in binary_records if r['entity_far_pct'] <= 10.0 and r['window_fpr_pct'] <= 5.0]
    if gate2_records:
        best_gate2_ep = max(r['episode_recall_pct'] for r in gate2_records)
        best_gate2_row = next(r for r in gate2_records if r['episode_recall_pct'] == best_gate2_ep)
        print(f"Gate 2 constrained maximum: Ep Recall={best_gate2_ep:.1f}% at theta={best_gate2_row['threshold']}")
    else:
        print("No thresholds satisfy Gate 2 constraints.")
        
    audit2_data[m_key] = {
        'binary_records': binary_records,
        'semantic_discrepancies_sample': semantic_discrepancies[:5]
    }

with open('scratch/audit2_summary.json', 'w') as f:
    json.dump(audit2_data, f, indent=2)
print("\nAudit 2 verification completed successfully.")
