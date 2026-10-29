import json
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, brier_score_loss

df_7e = pd.read_csv('results/phase7e/validation_predictions.csv')
df_7h = pd.read_csv('results/phase7h/validation_predictions.csv')

y_true = df_7h['y_true'].values
assert (y_true == df_7e['y_true'].values).all()

mask_norm = (y_true == 0)
mask_warn = (y_true == 1)
mask_crit = (y_true == 2)

pure_normal_eids = sorted(list(set(df_7h['entity_id'].unique()) - set(df_7h[df_7h['label'] > 0]['entity_id'].unique())))

models_to_audit = {
    'Phase 7E Cond A (Unweighted CE [1,1,1])': {
        'c0': df_7e['prob_transformer_condA_unweighted_c0'].values,
        'c1': df_7e['prob_transformer_condA_unweighted_c1'].values,
        'c2': df_7e['prob_transformer_condA_unweighted_c2'].values,
        'pred': df_7e['y_pred_transformer_condA_unweighted'].values
    },
    'Phase 7E Cond B (Ad-hoc CE [1, 1.5, 2])': {
        'c0': df_7e['prob_transformer_condB_adhoc_c0'].values,
        'c1': df_7e['prob_transformer_condB_adhoc_c1'].values,
        'c2': df_7e['prob_transformer_condB_adhoc_c2'].values,
        'pred': df_7e['y_pred_transformer_condB_adhoc'].values
    },
    'Phase 7E Cond C (Inv-Freq CE [1, 11.46, 11.80])': {
        'c0': df_7e['prob_transformer_condC_inverse_freq_c0'].values,
        'c1': df_7e['prob_transformer_condC_inverse_freq_c1'].values,
        'c2': df_7e['prob_transformer_condC_inverse_freq_c2'].values,
        'pred': df_7e['y_pred_transformer_condC_inverse_freq'].values
    },
    'Phase 7H Ablation A (Unfloored Baseline, Inv-Freq)': {
        'c0': df_7h['prob_transformer_hierarchical_baseline_c0'].values,
        'c1': df_7h['prob_transformer_hierarchical_baseline_c1'].values,
        'c2': df_7h['prob_transformer_hierarchical_baseline_c2'].values,
        'pred': df_7h['y_pred_transformer_hierarchical_baseline'].values
    },
    'Phase 7H Ablation B (Floored sigma=0.05, Inv-Freq)': {
        'c0': df_7h['prob_transformer_hierarchical_floored_c0'].values,
        'c1': df_7h['prob_transformer_hierarchical_floored_c1'].values,
        'c2': df_7h['prob_transformer_hierarchical_floored_c2'].values,
        'pred': df_7h['y_pred_transformer_hierarchical_floored'].values
    },
    'Phase 7H Ablation C (Robust Scaled, Inv-Freq)': {
        'c0': df_7h['prob_transformer_hierarchical_robust_c0'].values,
        'c1': df_7h['prob_transformer_hierarchical_robust_c1'].values,
        'c2': df_7h['prob_transformer_hierarchical_robust_c2'].values,
        'pred': df_7h['y_pred_transformer_hierarchical_robust'].values
    }
}

def compute_multiclass_ece(probs, labels, n_bins=10):
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels)
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return float(ece)

def dist_stats(arr):
    return {
        'mean': float(np.mean(arr)),
        'std': float(np.std(arr)),
        'median': float(np.median(arr)),
        'p25': float(np.percentile(arr, 25)),
        'p75': float(np.percentile(arr, 75)),
        'min': float(np.min(arr)),
        'max': float(np.max(arr))
    }

audit4_output = {}

# One-hot true labels for Brier score
y_one_hot = np.zeros((len(y_true), 3))
for i, y in enumerate(y_true):
    y_one_hot[i, y] = 1.0

for name, d in models_to_audit.items():
    P = np.stack([d['c0'], d['c1'], d['c2']], axis=1)
    # Clip for numerical stability in log loss
    P_safe = np.clip(P, 1e-12, 1.0 - 1e-12)
    P_safe = P_safe / P_safe.sum(axis=1, keepdims=True)
    
    # Calibration metrics
    ece = compute_multiclass_ece(P_safe, y_true, n_bins=10)
    brier = float(np.mean(np.sum((P_safe - y_one_hot)**2, axis=1)))
    nll = float(log_loss(y_true, P_safe))
    
    # False alarm metrics
    pred = d['pred']
    fp_windows = int((pred[mask_norm] > 0).sum())
    win_fpr = float(fp_windows / mask_norm.sum() * 100.0)
    fa_entities = sum(1 for eid in pure_normal_eids if (pred[df_7h['entity_id'] == eid] > 0).any())
    ent_far = float(fa_entities / len(pure_normal_eids) * 100.0)
    
    # Class probability distributions
    # 1. On True Normal windows
    p_norm_on_norm = dist_stats(d['c0'][mask_norm])
    p_warn_on_norm = dist_stats(d['c1'][mask_norm])
    p_crit_on_norm = dist_stats(d['c2'][mask_norm])
    
    # 2. On True Warning windows
    p_norm_on_warn = dist_stats(d['c0'][mask_warn])
    p_warn_on_warn = dist_stats(d['c1'][mask_warn])
    p_crit_on_warn = dist_stats(d['c2'][mask_warn])
    
    # 3. On True Critical windows
    p_norm_on_crit = dist_stats(d['c0'][mask_crit])
    p_warn_on_crit = dist_stats(d['c1'][mask_crit])
    p_crit_on_crit = dist_stats(d['c2'][mask_crit])
    
    # Pseudo-logits via log(P):
    # z_k = log(P_k) - mean(log(P))
    log_P = np.log(P_safe)
    logits = log_P - np.mean(log_P, axis=1, keepdims=True)
    
    audit4_output[name] = {
        'calibration': {
            'expected_calibration_error_ece': ece,
            'brier_score': brier,
            'negative_log_likelihood': nll
        },
        'false_alarm_metrics': {
            'window_fpr_pct': win_fpr,
            'entity_far_pct': ent_far
        },
        'true_normal_windows': {
            'P_Normal': p_norm_on_norm,
            'P_Warning': p_warn_on_norm,
            'P_Critical': p_crit_on_norm,
            'mean_logits': {
                'logit_Normal': float(np.mean(logits[mask_norm, 0])),
                'logit_Warning': float(np.mean(logits[mask_norm, 1])),
                'logit_Critical': float(np.mean(logits[mask_norm, 2]))
            }
        },
        'true_warning_windows': {
            'P_Normal': p_norm_on_warn,
            'P_Warning': p_warn_on_warn,
            'P_Critical': p_crit_on_warn,
            'mean_logits': {
                'logit_Normal': float(np.mean(logits[mask_warn, 0])),
                'logit_Warning': float(np.mean(logits[mask_warn, 1])),
                'logit_Critical': float(np.mean(logits[mask_warn, 2]))
            }
        },
        'true_critical_windows': {
            'P_Normal': p_norm_on_crit,
            'P_Warning': p_warn_on_crit,
            'P_Critical': p_crit_on_crit,
            'mean_logits': {
                'logit_Normal': float(np.mean(logits[mask_crit, 0])),
                'logit_Warning': float(np.mean(logits[mask_crit, 1])),
                'logit_Critical': float(np.mean(logits[mask_crit, 2]))
            }
        }
    }
    
    print(f"\n=== {name} ===")
    print(f"Calibration: ECE={ece:.4f}, Brier={brier:.4f}, NLL={nll:.4f}")
    print(f"False Alarms: Window FPR={win_fpr:.2f}%, Entity FAR={ent_far:.2f}%")
    print(f"True Normal windows -> Mean P: Norm={p_norm_on_norm['mean']:.4f}, Warn={p_warn_on_norm['mean']:.4f}, Crit={p_crit_on_norm['mean']:.4f}")
    print(f"True Critical windows -> Mean P: Norm={p_norm_on_crit['mean']:.4f}, Warn={p_warn_on_crit['mean']:.4f}, Crit={p_crit_on_crit['mean']:.4f}")

with open('scratch/audit4_results.json', 'w') as f:
    json.dump(audit4_output, f, indent=2)

print("\nAudit 4 verification completed successfully.")
