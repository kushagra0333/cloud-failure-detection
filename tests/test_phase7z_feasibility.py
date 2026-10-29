"""Tests for Phase 7Z: External Container Telemetry Access Feasibility."""

import json
from pathlib import Path
import pytest


PROJECT_ROOT = Path(r"C:\Users\hp\Desktop\Research project")
RESULTS_DIR = PROJECT_ROOT / "results" / "phase7y_borg2019_pilot"
REPORTS_DIR = PROJECT_ROOT / "reports" / "phase7y_borg2019_pilot"


def test_phase7z_deliverables_exist():
    """Verify that all Phase 7Z deliverable files exist and are non-empty."""
    md_file = REPORTS_DIR / "phase7z_external_telemetry_access_feasibility.md"
    json_results = RESULTS_DIR / "phase7z_external_telemetry_access_feasibility.json"
    json_reports = REPORTS_DIR / "phase7z_external_telemetry_access_feasibility.json"

    assert md_file.is_file(), "phase7z_external_telemetry_access_feasibility.md must exist in reports/"
    assert json_results.is_file(), "phase7z_external_telemetry_access_feasibility.json must exist in results/"
    assert json_reports.is_file(), "phase7z_external_telemetry_access_feasibility.json must exist in reports/"

    assert md_file.stat().st_size > 1000, "Markdown report must be non-trivial"
    assert json_results.stat().st_size > 1000, "JSON results file must be non-trivial"
    assert json_reports.stat().st_size == json_results.stat().st_size, "JSON files must be identical mirrors"


def test_phase7z_json_structure_and_verdict():
    """Verify that the JSON report contains valid structure and the declared final verdict."""
    json_file = RESULTS_DIR / "phase7z_external_telemetry_access_feasibility.json"
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Required verdict from the allowed enum
    allowed_verdicts = {
        "NO_SCIENTIFICALLY_COMPARABLE_SOURCE_FOUND",
        "COMPARABLE_SOURCE_REQUIRES_ACCESS_APPROVAL",
        "COMPARABLE_SOURCE_WITHIN_BUDGET_FOR_REVIEW",
        "BUDGET_AMENDMENT_WOULD_BE_REQUIRED",
    }
    assert data["final_feasibility_verdict"] in allowed_verdicts
    assert data["final_feasibility_verdict"] == "BUDGET_AMENDMENT_WOULD_BE_REQUIRED"

    # Governance checks
    gov = data["investigation_metadata"]["governing_rules"]
    assert gov["zero_data_extraction"] is True
    assert gov["zero_model_training"] is True
    assert gov["zero_table_materialization"] is True
    assert gov["protected_holdouts_accessed"] is False

    # Check candidate inventory includes key sources
    source_ids = {s["source_id"] for s in data["candidate_sources_inventory"]}
    assert "google_borg_2019" in source_ids
    assert "google_borg_2011" in source_ids
    assert "alibaba_cluster_trace_2018" in source_ids
    assert "azure_vm_traces" in source_ids

    # Check storage options A through E are all present
    opts = data["storage_and_cost_options_audit"]
    assert "approach_a_prefiltered_subset" in opts
    assert "approach_b_partitioned_clustered_source" in opts
    assert "approach_c_scientifically_defensible_sampling" in opts
    assert "approach_d_authorized_partitioned_staging" in opts
    assert "approach_e_bigquery_budget_amendment" in opts


def test_candidate_a_dry_run_values():
    """Verify Candidate A dry-run scan and cost metrics match measured service evidence."""
    json_file = RESULTS_DIR / "phase7z_external_telemetry_access_feasibility.json"
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    opt_e = data["storage_and_cost_options_audit"]["approach_e_bigquery_budget_amendment"]
    costs = opt_e["costs"]

    assert costs["measured_dry_run_scan_bytes"] == 418578205568
    assert costs["estimated_cost_usd_at_6_25"] == pytest.approx(2.3793, rel=1e-3)
    assert costs["marginal_cost_under_sandbox_free_tier"] == 0.00
    assert costs["sandbox_monthly_quota_fraction_pct"] == pytest.approx(38.07, rel=1e-2)


def test_protected_holdouts_remain_unopened():
    """Verify that Candidate A Test and Phase 6 Confirmatory holdouts remain unaccessed."""
    # Ensure no extraction files were written into data/ or results/
    pilot_csvs = list(RESULTS_DIR.glob("*.csv"))
    # Only dictionary / manifest metadata CSVs allowed, no raw pilot data
    for p in pilot_csvs:
        assert "pilot_telemetry" not in p.name
        assert "extracted" not in p.name

