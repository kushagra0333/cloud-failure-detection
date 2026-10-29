"""Ratified L=5 Confirmatory Evaluation Framework for Phase 6.2.3.

Strict Scientific Invariants:
1. Ingests and strictly enforces Protocol Amendment 2.1 (results/phase6/protocol_l5_amendment_2_1.json).
2. Uses the ratified primary evaluation denominators:
   - Primary Total Windows: 5,511 (400 Critical, 300 Warning, 4,811 Normal).
   - Normal Window Breakdown: 4,515 Pure-Normal Windows + 296 Failure-Cohort Normal Windows.
   - Evaluable Pure-Normal Entities: 198.
3. Formally isolates and rejects the 61 non-pre-failing tasks from the expanded census.
4. Enforces the twin operational gates:
   - Entity FAR Gate: <= 12.0% on 198 evaluable pure-normal entities (<= 23 false alarm entities).
   - Window FPR Gate: <= 5.0% on 4,811 primary normal windows (<= 240 false positive windows).
5. Mandates disaggregated false positive reporting (pure-normal vs. failure-cohort normal).
6. Enforces the pre-registered model selection rule:
   - Must pass BOTH operational gates to be eligible.
   - Ranked by Episode Detection Recall; tie-breaker: Macro-F1.
   - If no model passes both gates -> declares NO OPERATIONAL WINNER.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Set

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def load_ratified_l5_protocol(
    protocol_path: Optional[str] = None
) -> Dict[str, Any]:
    """Loads and validates the ratified Protocol Amendment 2.1 artifact.
    
    Rejects legacy, unamended, or ambiguous protocol artifacts to prevent
    denominator drift or unratified population leakage.
    """
    if protocol_path is None:
        p_path = PROJECT_ROOT / "results" / "phase6" / "protocol_l5_amendment_2_1.json"
    else:
        p_path = Path(protocol_path)
        if not p_path.is_absolute():
            p_path = PROJECT_ROOT / p_path

    if not p_path.exists():
        raise FileNotFoundError(
            f"Amended protocol artifact not found at {p_path}. "
            "Downstream evaluation code must read results/phase6/protocol_l5_amendment_2_1.json."
        )

    with open(p_path) as f:
        proto = json.load(f)

    # Validate protocol status and version
    amendment_version = proto.get("amendment_version")
    if amendment_version != "2.1":
        raise ValueError(
            f"Expected amendment_version '2.1', got '{amendment_version}'. "
            "Downstream code must read the ratified Protocol Amendment 2.1."
        )

    status = proto.get("status")
    if status != "RATIFIED_PRE_REGISTERED_FROZEN":
        raise ValueError(
            f"Expected protocol status 'RATIFIED_PRE_REGISTERED_FROZEN', got '{status}'."
        )

    # Verify locked denominators are explicitly present
    directives = proto.get("downstream_code_integration_directive", {})
    denoms = directives.get("locked_evaluation_denominators", {})
    required_denoms = {
        "primary_total_windows": 5511,
        "primary_normal_windows": 4811,
        "pure_normal_windows": 4515,
        "failure_cohort_normal_windows": 296,
        "warning_windows": 300,
        "critical_windows": 400,
        "prefail_windows": 700,
        "evaluable_pure_normal_entities": 198,
        "evaluable_failure_entities": 137,
        "max_allowable_window_false_positives": 240,
        "max_allowable_entity_false_alarms": 23,
    }

    for key, expected_val in required_denoms.items():
        val = denoms.get(key)
        if val != expected_val:
            raise ValueError(
                f"Protocol denominator mismatch for '{key}': expected {expected_val}, got {val}"
            )

    return proto


def validate_evaluation_manifest(
    metadata_df: pd.DataFrame,
    protocol: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Validates that a candidate evaluation dataset strictly matches the ratified primary population.
    
    Guarantees:
    1. Zero presence of the 61 excluded census tasks.
    2. Exact primary window counts: 5,511 total (400 Critical, 300 Warning, 4,811 Normal).
    3. Exact normal breakdown: 4,515 pure-normal + 296 failing-cohort normal.
    4. Exact entity coverage: 137 failure entities + 198 pure-normal entities = 335 evaluable entities.
    """
    if protocol is None:
        protocol = load_ratified_l5_protocol()

    frozen_eids = protocol["frozen_entity_ids"]
    primary_failure_eids = set(frozen_eids["primary_failure_entities"])
    primary_normal_eids = set(frozen_eids["primary_normal_entities_sampled"])
    evaluable_normal_eids = set(frozen_eids["primary_normal_entities_evaluable"])
    excluded_census_eids = set(frozen_eids["excluded_census_failure_tasks"])

    df_eids = set(metadata_df["entity_id"].unique())

    # 1. Strict Quarantine Check: 61 excluded tasks must NEVER enter the primary evaluation
    census_leakage = df_eids & excluded_census_eids
    if census_leakage:
        raise ValueError(
            f"FATAL: Detected {len(census_leakage)} excluded census tasks in evaluation manifest! "
            f"Excluded task IDs: {sorted(list(census_leakage))[:5]}... "
            "These tasks are strictly quarantined by Protocol Amendment 2.1."
        )

    # 2. Entity Membership: df_eids must be subset of primary cohort
    allowed_eids = primary_failure_eids | primary_normal_eids
    unknown_eids = df_eids - allowed_eids
    if unknown_eids:
        raise ValueError(
            f"FATAL: Detected {len(unknown_eids)} unapproved entities in evaluation manifest: "
            f"{sorted(list(unknown_eids))[:5]}..."
        )

    # 3. Window Count Verification
    total_windows = len(metadata_df)
    labels = metadata_df["label"].value_counts().to_dict()
    n_critical = labels.get(2, 0)
    n_warning = labels.get(1, 0)
    n_normal = labels.get(0, 0)

    # Disaggregate normal windows
    is_pure_normal = metadata_df["entity_id"].isin(evaluable_normal_eids)
    n_pure_normal = int((is_pure_normal & (metadata_df["label"] == 0)).sum())
    n_failing_normal = int((~is_pure_normal & (metadata_df["label"] == 0)).sum())

    denoms = protocol["downstream_code_integration_directive"]["locked_evaluation_denominators"]

    if total_windows != denoms["primary_total_windows"]:
        raise ValueError(
            f"Primary window count mismatch: expected {denoms['primary_total_windows']}, got {total_windows}"
        )
    if n_critical != denoms["critical_windows"]:
        raise ValueError(
            f"Critical window count mismatch: expected {denoms['critical_windows']}, got {n_critical}"
        )
    if n_warning != denoms["warning_windows"]:
        raise ValueError(
            f"Warning window count mismatch: expected {denoms['warning_windows']}, got {n_warning}"
        )
    if n_normal != denoms["primary_normal_windows"]:
        raise ValueError(
            f"Normal window count mismatch: expected {denoms['primary_normal_windows']}, got {n_normal}"
        )
    if n_pure_normal != denoms["pure_normal_windows"]:
        raise ValueError(
            f"Pure-normal window count mismatch: expected {denoms['pure_normal_windows']}, got {n_pure_normal}"
        )
    if n_failing_normal != denoms["failure_cohort_normal_windows"]:
        raise ValueError(
            f"Failing-cohort normal window count mismatch: expected {denoms['failure_cohort_normal_windows']}, got {n_failing_normal}"
        )

    eval_failure_eids = df_eids & primary_failure_eids
    eval_normal_eids = df_eids & evaluable_normal_eids

    if len(eval_failure_eids) != denoms["evaluable_failure_entities"]:
        raise ValueError(
            f"Evaluable failure entities count mismatch: expected {denoms['evaluable_failure_entities']}, got {len(eval_failure_eids)}"
        )
    if len(eval_normal_eids) != denoms["evaluable_pure_normal_entities"]:
        raise ValueError(
            f"Evaluable pure-normal entities count mismatch: expected {denoms['evaluable_pure_normal_entities']}, got {len(eval_normal_eids)}"
        )

    return {
        "is_valid": True,
        "total_windows": total_windows,
        "critical_windows": n_critical,
        "warning_windows": n_warning,
        "normal_windows": n_normal,
        "pure_normal_windows": n_pure_normal,
        "failing_cohort_normal_windows": n_failing_normal,
        "evaluable_failure_entities": len(eval_failure_eids),
        "evaluable_pure_normal_entities": len(eval_normal_eids),
        "census_leakage_detected": False,
    }


def evaluate_l5_operational_gates(
    metadata_df: pd.DataFrame,
    y_pred: np.ndarray,
    protocol: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Computes operational gates according to the ratified Protocol Amendment 2.1 rules.
    
    Operational Gates:
    1. Entity FAR Gate: <= 12.0% across 198 evaluable pure-normal entities (<= 23 false alarm entities).
    2. Window FPR Gate: <= 5.0% across 4,811 primary normal windows (<= 240 false positive windows).
    
    Disaggregated Reporting:
    - Pure Normal Window FPR (FP / 4,515)
    - Failing Cohort Normal Window FPR (FP / 296)
    - Pooled Primary Normal Window FPR (FP / 4,811)
    """
    if protocol is None:
        protocol = load_ratified_l5_protocol()

    denoms = protocol["downstream_code_integration_directive"]["locked_evaluation_denominators"]
    evaluable_normal_eids = set(protocol["frozen_entity_ids"]["primary_normal_entities_evaluable"])

    df = metadata_df.copy().reset_index(drop=True)
    df["y_pred"] = y_pred
    df["is_alert"] = df["y_pred"].isin([1, 2])

    # 1. Entity False Alarm Rate on Pure-Normal Entities
    pure_normal_df = df[df["entity_id"].isin(evaluable_normal_eids)]
    fa_entity_count = 0
    for eid, grp in pure_normal_df.groupby("entity_id"):
        if grp["is_alert"].any():
            fa_entity_count += 1

    total_eval_normal_entities = len(evaluable_normal_eids)
    far_entity_pct = (fa_entity_count / total_eval_normal_entities) * 100.0
    far_gate_passed = (fa_entity_count <= denoms["max_allowable_entity_false_alarms"]) and (far_entity_pct <= 12.0)

    # 2. Window False Positive Rates
    primary_normal_df = df[df["label"] == 0]
    total_normal_windows = len(primary_normal_df)

    pure_normal_mask = primary_normal_df["entity_id"].isin(evaluable_normal_eids)
    pure_normal_windows_df = primary_normal_df[pure_normal_mask]
    failing_normal_windows_df = primary_normal_df[~pure_normal_mask]

    fp_pure_normal = int(pure_normal_windows_df["is_alert"].sum())
    fp_failing_normal = int(failing_normal_windows_df["is_alert"].sum())
    fp_pooled_primary = int(primary_normal_df["is_alert"].sum())

    denom_pure_normal = len(pure_normal_windows_df)
    denom_failing_normal = len(failing_normal_windows_df)
    denom_pooled_primary = len(primary_normal_df)

    fpr_pure_normal_pct = (fp_pure_normal / denom_pure_normal) * 100.0 if denom_pure_normal > 0 else 0.0
    fpr_failing_normal_pct = (fp_failing_normal / denom_failing_normal) * 100.0 if denom_failing_normal > 0 else 0.0
    fpr_pooled_primary_pct = (fp_pooled_primary / denom_pooled_primary) * 100.0 if denom_pooled_primary > 0 else 0.0

    fpr_gate_passed = (fp_pooled_primary <= denoms["max_allowable_window_false_positives"]) and (fpr_pooled_primary_pct <= 5.0)

    both_gates_passed = far_gate_passed and fpr_gate_passed

    return {
        "entity_far": {
            "false_alarm_entities": fa_entity_count,
            "denominator_entities": total_eval_normal_entities,
            "rate_pct": far_entity_pct,
            "threshold_pct": 12.0,
            "max_allowable_entities": denoms["max_allowable_entity_false_alarms"],
            "gate_passed": far_gate_passed
        },
        "window_fpr": {
            "pooled_primary": {
                "false_positive_windows": fp_pooled_primary,
                "denominator_windows": denom_pooled_primary,
                "rate_pct": fpr_pooled_primary_pct,
                "threshold_pct": 5.0,
                "max_allowable_windows": denoms["max_allowable_window_false_positives"],
                "gate_passed": fpr_gate_passed
            },
            "disaggregated": {
                "pure_normal": {
                    "false_positive_windows": fp_pure_normal,
                    "denominator_windows": denom_pure_normal,
                    "rate_pct": fpr_pure_normal_pct
                },
                "failing_cohort_normal": {
                    "false_positive_windows": fp_failing_normal,
                    "denominator_windows": denom_failing_normal,
                    "rate_pct": fpr_failing_normal_pct
                }
            }
        },
        "operational_gates_verdict": {
            "passed_both_gates": both_gates_passed,
            "eligible_for_operational_winner": both_gates_passed
        }
    }


def select_operational_winner(
    model_candidate_results: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Applies the ratified model selection rule to choose an operational winner.
    
    Rule:
    1. Filter to candidates passing BOTH operational gates (Entity FAR <= 12.0%, Window FPR <= 5.0%).
    2. If no candidate passes both gates -> declare NO OPERATIONAL WINNER.
    3. Rank eligible candidates by Episode Detection Recall (descending).
    4. Tie-breaker: Macro-F1 across all 3 classes (descending).
    """
    eligible_models = []
    disqualified_models = []

    for cand in model_candidate_results:
        m_name = cand["model_name"]
        gates = cand["operational_gates"]
        if gates["operational_gates_verdict"]["passed_both_gates"]:
            eligible_models.append(cand)
        else:
            disqualified_models.append({
                "model_name": m_name,
                "far_passed": gates["entity_far"]["gate_passed"],
                "fpr_passed": gates["window_fpr"]["pooled_primary"]["gate_passed"],
                "far_entities": gates["entity_far"]["false_alarm_entities"],
                "fpr_windows": gates["window_fpr"]["pooled_primary"]["false_positive_windows"]
            })

    if not eligible_models:
        return {
            "winner_selected": False,
            "winner_model_name": None,
            "verdict": "NO_OPERATIONAL_WINNER",
            "rationale": (
                "Zero candidate models satisfied both operational gates "
                "(Entity FAR <= 12.0% and Window FPR <= 5.0%). "
                "Per pre-registered protocol, no operational winner is declared."
            ),
            "disqualified_candidates": disqualified_models,
            "eligible_candidates": []
        }

    # Rank eligible models: Primary = episode_recall, Tie-breaker = macro_f1
    sorted_eligible = sorted(
        eligible_models,
        key=lambda m: (m.get("episode_recall", 0.0), m.get("macro_f1", 0.0)),
        reverse=True
    )

    winner = sorted_eligible[0]
    return {
        "winner_selected": True,
        "winner_model_name": winner["model_name"],
        "verdict": "OPERATIONAL_WINNER_SELECTED",
        "winning_metrics": {
            "episode_recall": winner.get("episode_recall"),
            "macro_f1": winner.get("macro_f1"),
            "entity_far_pct": winner["operational_gates"]["entity_far"]["rate_pct"],
            "window_fpr_pct": winner["operational_gates"]["window_fpr"]["pooled_primary"]["rate_pct"]
        },
        "ranked_eligible_models": [m["model_name"] for m in sorted_eligible],
        "disqualified_candidates": disqualified_models
    }


def run_l5_cluster_bootstrap(
    metadata_df: pd.DataFrame,
    y_pred_dict: Dict[str, np.ndarray],
    n_replicates: int = 1000,
    random_seed: int = 42,
    protocol: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Computes entity-level cluster bootstrap uncertainty intervals for L=5 models.
    
    Strict Invariants:
    1. Resampling unit is entity_id (N=335 evaluable clusters in primary manifest).
    2. Keeps all windows and failure episodes belonging to each sampled entity together.
    3. Duplicate draws are preserved independently (each drawn instance contributes full sequence).
    4. Denominators:
       - Window FPR uses primary normal windows in replicate.
       - Pure-Normal Entity FAR uses drawn pure-normal entities in replicate.
       - Episode Recall uses failure episodes across drawn entities in replicate.
       - Disaggregated window FPRs computed for pure-normal and failing-cohort normal windows.
    5. Computes non-parametric empirical percentile 95% confidence intervals [2.5%, 97.5%].
    """
    if protocol is None:
        protocol = load_ratified_l5_protocol()

    frozen_eids = protocol["frozen_entity_ids"]
    evaluable_normal_eids = set(frozen_eids["primary_normal_entities_evaluable"])
    primary_failure_eids = set(frozen_eids["primary_failure_entities"])

    unique_entities = np.array(sorted(metadata_df["entity_id"].unique()))
    n_entities = len(unique_entities)

    # Pre-index windows by entity for vectorized replicate generation
    df = metadata_df.copy().reset_index(drop=True)
    y_true = df["label"].values

    entity_idx_map = {eid: np.where(df["entity_id"].values == eid)[0] for eid in unique_entities}

    # Episode extraction if failure information is present
    has_episodes = ("prediction_timestamp" in df.columns) and ("time_to_failure" in df.columns)
    if has_episodes:
        df["fail_timestamp"] = (df["prediction_timestamp"] + df["time_to_failure"]).round(1)
        prefails_df = df[df["label"] > 0]
        entity_episodes_map = {}
        for eid in unique_entities:
            ent_prefails = prefails_df[prefails_df["entity_id"] == eid]
            if ent_prefails.empty:
                entity_episodes_map[eid] = []
            else:
                eps = []
                for ftime, grp in ent_prefails.groupby("fail_timestamp"):
                    eps.append({
                        "fail_timestamp": ftime,
                        "window_indices": grp.index.values,
                        "pred_times": grp["prediction_timestamp"].values,
                    })
                entity_episodes_map[eid] = eps
    else:
        entity_episodes_map = {eid: [] for eid in unique_entities}

    rng = np.random.RandomState(random_seed)
    models = list(y_pred_dict.keys())

    # Precompute per-entity statistics for each model to maximize bootstrap speed
    ent_stats = {}
    for eid in unique_entities:
        idx = entity_idx_map[eid]
        y_ent = y_true[idx]
        is_pn = eid in evaluable_normal_eids
        norm_mask = (y_ent == 0)
        n_norm = int(norm_mask.sum())
        n_windows = len(idx)
        eps = entity_episodes_map[eid]
        n_eps = len(eps)

        m_dict = {}
        for m in models:
            preds = y_pred_dict[m][idx]
            correct = int((preds == y_ent).sum())
            fp_norm = int(((preds > 0) & norm_mask).sum())
            has_alert = bool((preds > 0).any())

            # Episode detections and lead times
            det_eps = 0
            lead_times = []
            for ep in eps:
                ep_preds = y_pred_dict[m][ep["window_indices"]]
                if (ep_preds > 0).any():
                    det_eps += 1
                    first_alert_idx = np.where(ep_preds > 0)[0][0]
                    first_alert_time = ep["pred_times"][first_alert_idx]
                    lead_times.append(float(ep["fail_timestamp"] - first_alert_time))

            m_dict[m] = {
                "correct": correct,
                "fp_norm": fp_norm,
                "has_alert": has_alert,
                "detected_episodes": det_eps,
                "lead_times": lead_times,
                "preds": preds,
                "y_true": y_ent,
            }

        ent_stats[eid] = {
            "is_pure_normal": is_pn,
            "n_windows": n_windows,
            "n_norm": n_norm,
            "n_episodes": n_eps,
            "model_stats": m_dict,
        }

    replicate_metrics = {m: {
        "macro_f1": [],
        "accuracy": [],
        "pooled_fpr_pct": [],
        "pure_normal_fpr_pct": [],
        "failing_normal_fpr_pct": [],
        "pure_normal_far_pct": [],
        "episode_recall_pct": [],
        "mean_lead_time_sec": [],
    } for m in models}

    for b in range(n_replicates):
        sampled_eids = rng.choice(unique_entities, size=n_entities, replace=True)

        # Aggregate across drawn entity instances
        total_drawn_pn = sum(1 for eid in sampled_eids if ent_stats[eid]["is_pure_normal"])
        total_drawn_norm = sum(ent_stats[eid]["n_norm"] for eid in sampled_eids)
        total_drawn_pn_norm = sum(ent_stats[eid]["n_norm"] for eid in sampled_eids if ent_stats[eid]["is_pure_normal"])
        total_drawn_fail_norm = sum(ent_stats[eid]["n_norm"] for eid in sampled_eids if not ent_stats[eid]["is_pure_normal"])
        total_drawn_eps = sum(ent_stats[eid]["n_episodes"] for eid in sampled_eids)
        total_drawn_windows = sum(ent_stats[eid]["n_windows"] for eid in sampled_eids)

        for m in models:
            tot_correct = sum(ent_stats[eid]["model_stats"][m]["correct"] for eid in sampled_eids)
            tot_fp_norm = sum(ent_stats[eid]["model_stats"][m]["fp_norm"] for eid in sampled_eids)
            tot_fp_pn_norm = sum(ent_stats[eid]["model_stats"][m]["fp_norm"] for eid in sampled_eids if ent_stats[eid]["is_pure_normal"])
            tot_fp_fail_norm = sum(ent_stats[eid]["model_stats"][m]["fp_norm"] for eid in sampled_eids if not ent_stats[eid]["is_pure_normal"])
            tot_pn_alerts = sum(1 for eid in sampled_eids if ent_stats[eid]["is_pure_normal"] and ent_stats[eid]["model_stats"][m]["has_alert"])
            tot_det_eps = sum(ent_stats[eid]["model_stats"][m]["detected_episodes"] for eid in sampled_eids)

            all_leads = []
            for eid in sampled_eids:
                all_leads.extend(ent_stats[eid]["model_stats"][m]["lead_times"])

            # Compute Macro-F1 across concatenated windows
            y_boot = np.concatenate([ent_stats[eid]["model_stats"][m]["y_true"] for eid in sampled_eids])
            preds_boot = np.concatenate([ent_stats[eid]["model_stats"][m]["preds"] for eid in sampled_eids])
            mf1 = float(f1_score(y_boot, preds_boot, average="macro", zero_division=0.0))
            acc = float(tot_correct / total_drawn_windows) if total_drawn_windows > 0 else 0.0

            pooled_fpr = float(tot_fp_norm / total_drawn_norm * 100.0) if total_drawn_norm > 0 else 0.0
            pn_fpr = float(tot_fp_pn_norm / total_drawn_pn_norm * 100.0) if total_drawn_pn_norm > 0 else 0.0
            fail_fpr = float(tot_fp_fail_norm / total_drawn_fail_norm * 100.0) if total_drawn_fail_norm > 0 else 0.0
            pn_far = float(tot_pn_alerts / total_drawn_pn * 100.0) if total_drawn_pn > 0 else 0.0
            ep_rec = float(tot_det_eps / total_drawn_eps * 100.0) if total_drawn_eps > 0 else 0.0
            m_lead = float(np.mean(all_leads)) if all_leads else 0.0

            replicate_metrics[m]["macro_f1"].append(mf1)
            replicate_metrics[m]["accuracy"].append(acc)
            replicate_metrics[m]["pooled_fpr_pct"].append(pooled_fpr)
            replicate_metrics[m]["pure_normal_fpr_pct"].append(pn_fpr)
            replicate_metrics[m]["failing_normal_fpr_pct"].append(fail_fpr)
            replicate_metrics[m]["pure_normal_far_pct"].append(pn_far)
            replicate_metrics[m]["episode_recall_pct"].append(ep_rec)
            replicate_metrics[m]["mean_lead_time_sec"].append(m_lead)

    # Compute 95% Percentile Confidence Intervals
    summary = {}
    for m in models:
        m_sum = {}
        for metric_name, vals in replicate_metrics[m].items():
            arr = np.array(vals)
            ci_low, ci_high = np.percentile(arr, [2.5, 97.5])
            m_sum[metric_name] = {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "ci_95": [float(ci_low), float(ci_high)],
            }
        summary[m] = m_sum

    return {
        "bootstrap_metadata": {
            "resampling_unit": "entity_id",
            "evaluable_clusters_count": n_entities,
            "n_replicates": n_replicates,
            "random_seed": random_seed,
            "confidence_level": 0.95,
        },
        "model_confidence_intervals": summary,
    }

