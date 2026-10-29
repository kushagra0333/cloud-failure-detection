"""Build results/project_discovery/claims_evidence_matrix.csv."""

import csv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
out_p = PROJECT_ROOT / "results" / "project_discovery" / "claims_evidence_matrix.csv"

claims = [
    {
        "claim_id": "CLAIM_01",
        "scientific_claim": "Deep temporal sequence models (LSTM, Transformer) outperform non-temporal baselines (Random Forest) in cloud task failure prediction.",
        "phase_origin": "Phase 1 / Phase 3B / Phase 4",
        "supporting_or_refuting_evidence": "confirmatory_metrics_l5.json, confirmatory_predictions_l5.csv",
        "verification_status": "CONTRADICTED",
        "findings_and_limitations": "Random Forest achieved higher Macro-F1 (0.6111 vs 0.4425 LSTM, 0.4479 Transformer) and lower pure-normal false alarm rates. No deep model surpassed RF operationally."
    },
    {
        "claim_id": "CLAIM_02",
        "scientific_claim": "Representation-Fused Hybrid architecture combines sequential learning with tree decision boundaries to achieve superior deployment performance.",
        "phase_origin": "Phase 4",
        "supporting_or_refuting_evidence": "confirmatory_metrics_l5.json, confirmatory_entity_metrics_l5.csv",
        "verification_status": "CONTRADICTED",
        "findings_and_limitations": "Hybrid model had the worst window FPR (34.65%) and worst entity FAR (64.14%) among all models. In-sample fitting of Random Forest on uncalibrated neural embeddings caused severe overfitting."
    },
    {
        "claim_id": "CLAIM_03",
        "scientific_claim": "All four models failed pre-registered operational gates on the confirmatory holdout (NO_OPERATIONAL_WINNER).",
        "phase_origin": "Phase 6.4 / Phase 6.5",
        "supporting_or_refuting_evidence": "confirmatory_metrics_l5.json, confirmatory_bootstrap_ci_l5.json, post_evaluation_audit_l5.json",
        "verification_status": "VERIFIED",
        "findings_and_limitations": "Every model exceeded the 5.0% Window FPR ceiling (20.27% - 34.65%) and the 12.0% Entity FAR ceiling (31.82% - 64.14%). Result independently reproduced across multiple audits."
    },
    {
        "claim_id": "CLAIM_04",
        "scientific_claim": "Post-failure telemetry was completely excluded from early warning evaluation and labeled UNKNOWN (-1).",
        "phase_origin": "Phase 2 / Phase 3C / Phase 4 / Phase 5 / Phase 6.2",
        "supporting_or_refuting_evidence": "src/labeling/target_generator.py lines 73-77, confirmatory_predictions_l5.csv, train_metadata.csv, val_metadata.csv, test_metadata.csv",
        "verification_status": "CONTRADICTED",
        "findings_and_limitations": "TargetLabelGenerator lacked a check for prior failures when future failures were absent. 149 post-failure windows were included as Normal in confirmatory data; over 1,400 were included across train/val/test splits."
    },
    {
        "claim_id": "CLAIM_05",
        "scientific_claim": "False positives across all models predominantly trigger Critical alerts (73% - 97%) rather than Warning alerts.",
        "phase_origin": "Phase 6.6 / Phase 6.7",
        "supporting_or_refuting_evidence": "confirmatory_predictions_l5.csv, phase6_6_error_mechanism_investigation.json, phase6_7_independent_scientific_audit.json",
        "verification_status": "VERIFIED",
        "findings_and_limitations": "RF: 86.20% Critical, LSTM: 97.06%, Transformer: 92.72%, Hybrid: 73.13%. Models systematically skip intermediate Warning states during false alarms."
    },
    {
        "claim_id": "CLAIM_06",
        "scientific_claim": "Missing CPI telemetry (cpi_missing_fraction_6) represents a physical hardware failure mechanism causing container death.",
        "phase_origin": "Phase 6.6",
        "supporting_or_refuting_evidence": "phase6_6_error_mechanism_investigation.md, phase6_7_independent_scientific_audit.json",
        "verification_status": "CONTRADICTED",
        "findings_and_limitations": "Mathematical elevation (Cohen's d = +1.2 to +3.0) reproduced, but causal claim is invalid. Missing CPI reflects container idling, Borg descheduling, and batch workload characteristics, not physical processor failure."
    },
    {
        "claim_id": "CLAIM_07",
        "scientific_claim": "A small number of parent jobs (top 3) account for 50% to 61% of false positives in Random Forest and Compact Transformer.",
        "phase_origin": "Phase 6.6 / Phase 6.7",
        "supporting_or_refuting_evidence": "confirmatory_predictions_l5.csv, phase6_7_independent_scientific_audit.json",
        "verification_status": "VERIFIED",
        "findings_and_limitations": "Top 3 jobs account for 50.63% (RF) and 52.21% (Transformer) of FPs. Gini coefficients exceed 0.79 across all models. Workload heterogeneity is heavily clustered by parent job."
    },
    {
        "claim_id": "CLAIM_08",
        "scientific_claim": "Uniform 3-class argmax decision rule triggers Critical alerts at predicted probabilities as low as 0.336 to 0.359.",
        "phase_origin": "Phase 6.6 / Phase 6.7",
        "supporting_or_refuting_evidence": "confirmatory_predictions_l5.csv, phase6_7_independent_scientific_audit.json",
        "verification_status": "VERIFIED",
        "findings_and_limitations": "Mathematical property of argmax in 3-class classification when non-critical probability mass is split evenly between Normal and Warning."
    },
    {
        "claim_id": "CLAIM_09",
        "scientific_claim": "Compact Transformer outputs extreme softmax saturation on false positives (median P(Critical) > 0.99).",
        "phase_origin": "Phase 6.6 / Phase 6.7",
        "supporting_or_refuting_evidence": "confirmatory_predictions_l5.csv, best_transformer_checkpoint_l5.pt",
        "verification_status": "VERIFIED",
        "findings_and_limitations": "Median P(Critical) = 0.9934 on false alarms; 53.65% exceed 0.99. Driven by cross-entropy optimization without temperature scaling or probability calibration."
    },
    {
        "claim_id": "CLAIM_10",
        "scientific_claim": "All four models were trained with uniform ~10x loss weights on Critical and ~13x on Warning.",
        "phase_origin": "Phase 6.6",
        "supporting_or_refuting_evidence": "scripts/train_phase6_3_l5_models.py line 321 & 415, training_manifest_l5.json",
        "verification_status": "CONTRADICTED",
        "findings_and_limitations": "Neural models (LSTM, Transformer) were trained with explicit fixed weights [1.0, 1.5, 2.0]. Only Random Forest and Hybrid used balanced class weights (~11.5x / 17.0x)."
    },
    {
        "claim_id": "CLAIM_11",
        "scientific_claim": "Sequence length L=15 (75 minutes unbroken history) is viable for failure prediction in cloud datacenters.",
        "phase_origin": "Phase 1 / Phase 2",
        "supporting_or_refuting_evidence": "results/phase5/confirmatory_metrics.json, reports/phase5_confirmatory_results.md",
        "verification_status": "CONTRADICTED",
        "findings_and_limitations": "All 100 Phase 5 frozen failure entities failed before completing 15 pre-failure steps (< 75 min). 0 pre-failure windows met the invariant, forcing the protocol to pivot to L=5 (25 min)."
    },
    {
        "claim_id": "CLAIM_12",
        "scientific_claim": "Development splits (train, val, test) provide independent validation of failure generalization.",
        "phase_origin": "Phase 2 / Phase 3",
        "supporting_or_refuting_evidence": "data/processed/train_metadata.csv, val_metadata.csv, test_metadata.csv",
        "verification_status": "PARTIALLY_VERIFIED",
        "findings_and_limitations": "Entities are 100% disjoint, but parent jobs overlap by 75%-80% (24/32 val jobs and 25/31 test jobs appear in train). Models memorized parent-job-specific resource signatures."
    },
    {
        "claim_id": "CLAIM_13",
        "scientific_claim": "Operational gate failures and NO_OPERATIONAL_WINNER verdict are robust to cohort censoring corrections.",
        "phase_origin": "Phase 6.8",
        "supporting_or_refuting_evidence": "results/phase6/phase6_8_confirmatory_validity_audit.json, scratch/eval_counterfactuals.py",
        "verification_status": "VERIFIED",
        "findings_and_limitations": "Under all counterfactual scenarios (censoring 149 post-failure windows, censoring all 296 failure-cohort normal windows, or censoring cross-history windows), all 4 models fail both gates by wide margins."
    }
]

with open(out_p, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(claims[0].keys()))
    writer.writeheader()
    writer.writerows(claims)

print(f"Claims evidence matrix written to {out_p} ({len(claims)} claims)")
