import json
import numpy as np
import pandas as pd

df = pd.read_csv('results/phase7h/validation_predictions.csv')
y_true = df['y_true'].values

prefails = df[df['label'] > 0].copy()
prefails['fail_timestamp'] = (prefails['prediction_timestamp'] + prefails['time_to_failure']).round(1)
episodes = prefails.groupby(['entity_id', 'fail_timestamp'])
total_episodes = len(episodes)

# Analyze episode characteristics
episode_details = []
for (eid, ftime), grp in episodes:
    n_crit = int((grp['label'] == 2).sum())
    n_warn = int((grp['label'] == 1).sum())
    t_min = float(grp['prediction_timestamp'].min())
    t_max = float(grp['prediction_timestamp'].max())
    ttf_min = float(grp['time_to_failure'].min())
    ttf_max = float(grp['time_to_failure'].max())
    episode_details.append({
        'entity_id': eid,
        'parent_job_id': int(grp['parent_job_id'].iloc[0]),
        'fail_timestamp': float(ftime),
        'total_windows': len(grp),
        'warning_windows': n_warn,
        'critical_windows': n_crit,
        'ttf_span_sec': [ttf_min, ttf_max],
        'has_zero_critical': bool(n_crit == 0),
        'has_zero_warning': bool(n_warn == 0)
    })

models = {
    'transformer_hierarchical_baseline': 'y_pred_transformer_hierarchical_baseline',
    'transformer_hierarchical_floored': 'y_pred_transformer_hierarchical_floored',
    'transformer_hierarchical_robust': 'y_pred_transformer_hierarchical_robust'
}

temporal_audit = {
    'total_episodes': total_episodes,
    'episodes_without_critical_windows': sum(1 for e in episode_details if e['has_zero_critical']),
    'episodes_without_warning_windows': sum(1 for e in episode_details if e['has_zero_warning']),
    'episode_details': episode_details,
    'model_evaluations': {}
}

for m_key, pred_col in models.items():
    y_pred = df[pred_col].values
    
    det_count = 0
    lead_times = []
    episode_records = []
    
    for (eid, ftime), grp in episodes:
        det_mask = (y_pred[grp.index] > 0)
        is_detected = bool(det_mask.any())
        
        if is_detected:
            det_count += 1
            first_idx = grp.index[det_mask][0]
            t_first = float(df.loc[first_idx, 'prediction_timestamp'])
            lt = float(ftime - t_first)
            lead_times.append(lt)
            first_class = int(y_pred[first_idx])
            true_class_at_first = int(y_true[first_idx])
        else:
            lt = None
            first_class = None
            true_class_at_first = None
            
        episode_records.append({
            'entity_id': eid,
            'fail_timestamp': float(ftime),
            'detected': is_detected,
            'lead_time_sec': lt,
            'first_alert_predicted_class': first_class,
            'true_class_at_first_alert': true_class_at_first
        })
        
    ep_rec = float(det_count / total_episodes * 100.0)
    
    lt_arr = np.array(lead_times) if lead_times else np.array([])
    if len(lt_arr) > 0:
        lt_stats = {
            'mean_sec': float(np.mean(lt_arr)),
            'std_sec': float(np.std(lt_arr)),
            'median_sec': float(np.median(lt_arr)),
            'p25_sec': float(np.percentile(lt_arr, 25)),
            'p75_sec': float(np.percentile(lt_arr, 75)),
            'iqr_sec': float(np.percentile(lt_arr, 75) - np.percentile(lt_arr, 25)),
            'min_sec': float(np.min(lt_arr)),
            'max_sec': float(np.max(lt_arr)),
            'pct_lead_ge_600s_10m': float(np.mean(lt_arr >= 600.0) * 100.0),
            'pct_lead_ge_900s_15m': float(np.mean(lt_arr >= 900.0) * 100.0),
            'pct_lead_ge_1200s_20m': float(np.mean(lt_arr >= 1200.0) * 100.0),
            'lead_times_all': [float(x) for x in lt_arr]
        }
    else:
        lt_stats = None
        
    temporal_audit['model_evaluations'][m_key] = {
        'episode_recall_pct': ep_rec,
        'detected_episodes': det_count,
        'total_episodes': total_episodes,
        'lead_time_statistics': lt_stats,
        'episodes': episode_records
    }
    
    print(f"\n--- {m_key} ---")
    print(f"Episode Recall: {det_count}/{total_episodes} = {ep_rec:.2f}%")
    if lt_stats:
        print(f"Lead Time Mean: {lt_stats['mean_sec']:.1f}s ({lt_stats['mean_sec']/60:.1f} min)")
        print(f"Lead Time Median: {lt_stats['median_sec']:.1f}s, IQR: {lt_stats['iqr_sec']:.1f}s [{lt_stats['p25_sec']:.1f}s, {lt_stats['p75_sec']:.1f}s]")
        print(f"Lead Time Min: {lt_stats['min_sec']:.1f}s, Max: {lt_stats['max_sec']:.1f}s")
        print(f"Alerts >= 10m: {lt_stats['pct_lead_ge_600s_10m']:.1f}%, >= 15m: {lt_stats['pct_lead_ge_900s_15m']:.1f}%, >= 20m: {lt_stats['pct_lead_ge_1200s_20m']:.1f}%")

with open('scratch/audit5_results.json', 'w') as f:
    json.dump(temporal_audit, f, indent=2)

print("\nAudit 5 verification completed successfully.")
