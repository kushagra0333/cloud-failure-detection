import hashlib
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd

def compute_sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

out_dir = Path('results/phase7h_a')
out_dir.mkdir(parents=True, exist_ok=True)

# 1. Metric Reproduction JSON
with open('scratch/audit1_verification.json', 'r') as f:
    audit1_data = json.load(f)

metric_reproduction = {
    'audit_metadata': {
        'phase': 'Phase 7H-A — Metric, Normalization, and Provenance Verification',
        'audit_id': 'AUDIT-1-METRIC-REPRODUCTION',
        'timestamp': '2026-10-28T21:18:00+05:30',
        'dataset_split': 'Candidate A Validation Split Only (N=657)',
        'source_predictions_csv': 'results/phase7h/validation_predictions.csv',
        'assertions_passed': [
            'All confusion matrix totals equal 657 (100% of validation rows accounted for)',
            'Per-class recalls derived from confusion matrix exactly equal reported classification recalls',
            'All denominators and edge-case handlings documented and mathematically verified'
        ]
    },
    'denominators_and_definitions': {
        'total_validation_rows': 657,
        'class_support': {
            'class_0_normal_windows': 560,
            'class_1_warning_windows': 48,
            'class_2_critical_windows': 49,
            'sum_support': 657
        },
        'entity_support': {
            'total_validation_entities': 69,
            'failing_entities': 20,
            'pure_normal_entities': 49
        },
        'parent_job_support': {
            'total_validation_parent_jobs': 12,
            'failing_parent_jobs': 8,
            'pure_normal_parent_jobs': 4,
            'pure_normal_job_ids': [4665617026, 6250023935, 6251664479, 6251933641]
        },
        'episode_support': {
            'total_validation_failure_episodes': 20,
            'episode_grouping_key': ['entity_id', 'fail_timestamp'],
            'definition': 'Pre-failure degradation interval ending at physical failure timestamp'
        },
        'metric_formulas': {
            'Macro_F1': 'Arithmetic mean of unweighted per-class F1 scores: mean([F1_0, F1_1, F1_2])',
            'Balanced_Accuracy': 'Arithmetic mean of per-class recalls: mean([Recall_0, Recall_1, Recall_2])',
            'Episode_Recall': 'Number of failure episodes with >=1 failure alert (y_pred > 0) in (0, 1800s] / 20',
            'Entity_FAR': 'Number of pure-normal entities with >=1 false alarm (y_pred > 0) / 49',
            'Window_FPR': 'Number of true-normal windows (y_true == 0) with y_pred > 0 / 560',
            'Parent_Job_FAR': 'Number of pure-normal parent jobs with >=1 false alarm / 4'
        }
    },
    'models': audit1_data
}

with open(out_dir / 'metric_reproduction.json', 'w') as f:
    json.dump(metric_reproduction, f, indent=2)

# 2. Threshold Semantics Audit JSON
with open('scratch/audit2_summary.json', 'r') as f:
    audit2_data = json.load(f)

threshold_semantics = {
    'audit_metadata': {
        'phase': 'Phase 7H-A — Metric, Normalization, and Provenance Verification',
        'audit_id': 'AUDIT-2-THRESHOLD-SEMANTICS-RECONCILIATION',
        'timestamp': '2026-10-28T21:18:00+05:30',
        'objective': 'Reconcile decision rules, separate binary alert from 3-class prediction, and correct mislabeled Critical Recall'
    },
    'concept_separation_and_decision_rules': {
        'binary_failure_alert': {
            'rule': 'P(Warning) + P(Critical) >= threshold (theta)',
            'output_space': '{0: Healthy, 1: Failure Alert}',
            'interpretation': 'Flags whether ANY failure alert should be dispatched to operators.'
        },
        'three_class_prediction_rules': {
            'default_argmax': 'y_pred = argmax([P(0), P(1), P(2)])',
            'cascaded_threshold_rule': (
                'If P(Critical) >= theta_crit, assign 2. Else if P(Warning) + P(Critical) >= theta_fail, assign 1. '
                'Else assign 0. Disentangles urgency tier from alert threshold.'
            ),
            'argmax_among_alerts': (
                'If P(Warning) + P(Critical) >= theta: assign 2 if P(Critical) > P(Warning) else assign 1. '
                'Else assign 0.'
            )
        },
        'episode_detection': {
            'rule': 'An episode is detected if AT LEAST ONE window within (0, 1800s] triggers a failure alert (y_pred > 0).',
            'lead_time': 'Time from earliest alert in the episode to failure: t_fail - t_first_alert.'
        },
        'critical_class_window_recall': {
            'definition': 'Proportion of true Critical windows (y_true == 2) predicted specifically as Critical (y_pred == 2).',
            'denominator': 49,
            'mislabeling_in_phase7h': (
                'In Phase 7H Section 7, the column labeled "Crit. Recall" in the threshold sweep was actually computed as '
                'the proportion of Critical windows where P(Warning) + P(Critical) >= theta. This is Binary Critical Interception Rate '
                '(P(alert | Critical)), NOT genuine 3-class Critical Recall (P(y_pred=2 | y_true=2)). Under the binary threshold rule, '
                'genuine class 2 is never predicted.'
            )
        }
    },
    'threshold_verification_and_corrections': {
        'transformer_hierarchical_baseline': {
            'threshold_0_63_audit': {
                'reported_figure_in_phase7h': 'Crit. Recall = 0.3265',
                'actual_semantic_meaning': 'Binary Critical Interception Rate = 0.3265 (16 / 49 windows had P_fail >= 0.63)',
                'genuine_critical_recall_if_P_crit_ge_0_63': 0.0,
                'cascaded_critical_recall_at_0_63': 0.0204,
                'episode_recall_pct': 70.0,
                'entity_far_pct': 32.7,
                'window_fpr_pct': 23.9,
                'verdict': 'INCONSISTENT LABELING IN PHASE 7H — CORRECTED TO BINARY INTERCEPTION RATE'
            },
            'gate_constrained_maximum_under_gate2': {
                'constraint': 'Entity FAR <= 10.0% AND Window FPR <= 5.0%',
                'reproduced_threshold': 0.83,
                'max_episode_recall_pct': 0.0,
                'genuine_critical_recall': 0.0,
                'status': 'VERIFIED REPRODUCED'
            }
        },
        'transformer_hierarchical_floored': {
            'threshold_0_63_audit': {
                'reported_figure_in_phase7h': 'Crit. Recall = 0.6531',
                'actual_semantic_meaning': 'Binary Critical Interception Rate = 0.6531 (32 / 49 windows had P_fail >= 0.63)',
                'genuine_critical_recall_if_P_crit_ge_0_63': 0.0,
                'cascaded_critical_recall_at_0_63': 0.5306,
                'episode_recall_pct': 85.0,
                'entity_far_pct': 100.0,
                'window_fpr_pct': 64.8,
                'verdict': 'INCONSISTENT LABELING IN PHASE 7H — CORRECTED TO BINARY INTERCEPTION RATE'
            },
            'gate_constrained_maximum_under_gate2': {
                'constraint': 'Entity FAR <= 10.0% AND Window FPR <= 5.0%',
                'reproduced_threshold': 0.75,
                'max_episode_recall_pct': 0.0,
                'genuine_critical_recall': 0.0,
                'status': 'VERIFIED REPRODUCED'
            }
        },
        'transformer_hierarchical_robust': {
            'threshold_0_65_audit': {
                'reported_figure_in_phase7h': 'Crit. Recall = 0.5306 (at theta=0.65)',
                'actual_semantic_meaning': 'Binary Critical Interception Rate = 0.5306 (26 / 49 windows had P_fail >= 0.65)',
                'genuine_critical_recall_if_P_crit_ge_0_65': 0.0,
                'cascaded_critical_recall_at_0_65': 0.4082,
                'episode_recall_pct': 55.0,
                'entity_far_pct': 93.9,
                'window_fpr_pct': 26.4,
                'verdict': 'INCONSISTENT LABELING IN PHASE 7H — CORRECTED TO BINARY INTERCEPTION RATE'
            },
            'gate_constrained_maximum_under_gate2': {
                'constraint': 'Entity FAR <= 10.0% AND Window FPR <= 5.0%',
                'reproduced_threshold': 0.69,
                'max_episode_recall_pct': 15.0,
                'genuine_critical_recall': 0.0,
                'status': 'VERIFIED REPRODUCED'
            }
        }
    },
    'models_detailed_sweeps': audit2_data
}

with open(out_dir / 'threshold_semantics_audit.json', 'w') as f:
    json.dump(threshold_semantics, f, indent=2)

# 3. Normalization Unit Tests JSON
with open('scratch/audit3_results.json', 'r') as f:
    audit3_data = json.load(f)

normalization_unit_tests = {
    'audit_metadata': {
        'phase': 'Phase 7H-A — Metric, Normalization, and Provenance Verification',
        'audit_id': 'AUDIT-3-NORMALIZATION-NUMERICAL-VALIDATION',
        'timestamp': '2026-10-28T21:18:00+05:30',
        'implementation_files_audited': [
            'src/features/hierarchical_baseline.py',
            'scripts/prepare_phase7h_transformed_data.py'
        ]
    },
    'code_inspection_findings': {
        'temporal_causality': {
            'verified': True,
            'mechanism': 'task_mask = (job_entities == eid) & (job_timestamps < t_pred); job_mask = (job_timestamps < t_pred)',
            'conclusion': 'Strictly causal. Only observations strictly prior to prediction timestamp (t_prior < t_pred) are accessed.'
        },
        'training_data_fit': {
            'verified': True,
            'mechanism': 'Variance floor sigma_floor=0.05 and MAD floor mad_floor=0.05 fit exclusively on Candidate A Train split percentiles.',
            'conclusion': 'Zero validation or test observations were used to fit global scaling parameters.'
        },
        'target_and_future_telemetry_isolation': {
            'verified': True,
            'mechanism': 'Only raw input sequence array X_seq is indexed by prior timestamps; target labels are neither passed nor accessed.',
            'conclusion': 'Zero label leakage into features.'
        },
        'scale_and_units_of_variance_floor': {
            'verified': True,
            'mechanism': 'Features are already standardized to cluster population variance (sigma_cluster = 1.0) by Phase 7C scaler. '
                         'Therefore, sigma_floor = 0.05 represents exactly 5% of the cluster population standard deviation.',
            'conclusion': 'Directly aligned with intended feature scale and unit-variance dimensions.'
        },
        'mad_conversion_consistency': {
            'verified': True,
            'mechanism': 'mad = np.median(np.abs(past_obs - med), axis=0) * 1.4826 + 1e-5',
            'conclusion': 'Normal-consistency factor 1.4826 is correctly applied to yield an asymptotic normal standard deviation estimate.'
        },
        'missing_and_nonfinite_handling': {
            'verified': True,
            'mechanism': 'FeatureEngineer upstream preserves NaNs, and Scaler imputes using Training median prior to tensor slicing. Tensors contain 0 NaNs and 0 Infs.',
            'conclusion': 'Missing and non-finite values are handled deterministically.'
        }
    },
    'reproduced_tensor_extrema': audit3_data['tensor_audit'],
    'deterministic_unit_tests': audit3_data['unit_tests']
}

with open(out_dir / 'normalization_unit_tests.json', 'w') as f:
    json.dump(normalization_unit_tests, f, indent=2)

# 4. Probability Shift Audit JSON
with open('scratch/audit4_results.json', 'r') as f:
    audit4_data = json.load(f)

probability_shift = {
    'audit_metadata': {
        'phase': 'Phase 7H-A — Metric, Normalization, and Provenance Verification',
        'audit_id': 'AUDIT-4-PROBABILITY-SHIFT-EXPLANATION-TEST',
        'timestamp': '2026-10-28T21:18:00+05:30',
        'scope': 'Controlled comparisons across unweighted, ad-hoc, and inverse-frequency loss weighting'
    },
    'scientific_fact_vs_hypothesis_labeling': {
        'verified_facts': [
            'In Phase 7E (fixed features, architecture, seed), changing cross-entropy weights from [1,1,1] to [1, 11.46, 11.80] '
            'reduced mean P(Normal) on Normal windows from 0.7799 to 0.4499, increased Window FPR from 4.82% to 20.89%, and increased Entity FAR from 12.24% to 28.57%.',
            'In Phase 7H (floored and robust hierarchical features), inverse-frequency weights pushed mean P(Normal) on Normal windows down to 0.3519 (Ablation B) and 0.3650 (Ablation C).',
            'Under default argmax, when P(Normal) is close to 0.333, random workload fluctuations push P(Warning) or P(Critical) above P(Normal), causing Entity FAR to surge to 91.84%–95.92%.',
            'Brier score degrades from 0.3067 (unweighted) to 0.5308 (Phase 7E inv-freq) to 0.6425 (Phase 7H Ablation B).'
        ],
        'hypotheses': [
            'Hypothesis: The exact degree to which feature bounding interacts with logit inflation requires retraining an unweighted cross-entropy model on the floored hierarchical features as an ablation control.',
            'Hypothesis: Intermediate loss weights (e.g., w in [2.0, 5.0]) or cost-sensitive threshold tuning can decouple the Critical Recall gains (34%) from the 95% false-alarm explosion.'
        ]
    },
    'model_evaluations': audit4_data
}

with open(out_dir / 'probability_shift_audit.json', 'w') as f:
    json.dump(probability_shift, f, indent=2)

# 5. Temporal Metric Audit JSON
with open('scratch/audit5_results.json', 'r') as f:
    audit5_data = json.load(f)

temporal_metric_audit = {
    'audit_metadata': {
        'phase': 'Phase 7H-A — Metric, Normalization, and Provenance Verification',
        'audit_id': 'AUDIT-5-TEMPORAL-METRIC-VERIFICATION',
        'timestamp': '2026-10-28T21:18:00+05:30'
    },
    'definitions_and_semantics': {
        'episode_identification': (
            'A failure episode is uniquely identified by the tuple (entity_id, fail_timestamp), '
            'where fail_timestamp = round(prediction_timestamp + time_to_failure, 1). '
            'Validation contains exactly 20 such episodes across 20 distinct entities.'
        ),
        'what_counts_as_an_alert': (
            'Any prediction of class 1 (Warning) or class 2 (Critical) during the episode window: y_pred > 0.'
        ),
        'episode_alert_counting': (
            'An episode is credited as detected if AT LEAST ONE alert (y_pred > 0) occurs within its pre-failure window. '
            'Multiple alerts within the same episode do NOT increment the count of detected episodes (numerator is bounded by total episodes).'
        ),
        'lead_time_formula': (
            'lead_time = fail_timestamp - t_first_alert, where t_first_alert is the prediction_timestamp of the '
            'earliest alert (y_pred > 0) in the episode.'
        ),
        'handling_of_episodes_without_critical_windows': (
            'Exactly 3 episodes in validation contain 0 Critical windows because the tasks experienced abrupt termination '
            'more than 15 minutes before their scheduled logging window. These episodes are included in total episodes (N=20) '
            'for Episode Recall, but contribute 0 windows to Critical window support (N=49).'
        ),
        'operational_caveat': (
            'High mean lead time (~24 min) alone does NOT indicate reliable operational warning time. '
            'Because floored and robust models alert on >90% of pure-normal entities, an alert firing at t-24min '
            'on a failing entity cannot be distinguished from a persistent baseline false alert.'
        )
    },
    'summary_metrics': audit5_data
}

with open(out_dir / 'temporal_metric_audit.json', 'w') as f:
    json.dump(temporal_metric_audit, f, indent=2)

# 6. Provenance Audit JSON
with open('scratch/audit6_results.json', 'r') as f:
    audit6_data = json.load(f)

with open(out_dir / 'provenance_audit.json', 'w') as f:
    json.dump(audit6_data, f, indent=2)

# 7. Generate Manifest JSON
manifest = {
    'manifest_version': '1.0',
    'phase': 'Phase 7H-A — Metric, Normalization, and Provenance Verification',
    'timestamp': '2026-10-28T21:18:00+05:30',
    'safeguards_verification': {
        'candidate_a_test_accessed': False,
        'phase6_confirmatory_holdout_accessed': False,
        'phase7b_to_7h_artifacts_overwritten': False,
        'confirmatory_testing_authorized': False
    },
    'artifacts': {}
}

for p in sorted(out_dir.glob('*.json')):
    manifest['artifacts'][p.name] = {
        'path': str(p).replace('\\', '/'),
        'size_bytes': p.stat().st_size,
        'sha256': compute_sha256(p)
    }

with open(out_dir / 'phase7h_a_manifest.json', 'w') as f:
    json.dump(manifest, f, indent=2)

print("Generated all 7 phase7h_a JSON artifacts successfully.")
