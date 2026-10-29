"""Tests for Phase 7Z.1: Borg 2019 Pre-Extraction Authorization Package."""

import json
from pathlib import Path
import pytest


PROJECT_ROOT = Path(r"C:\Users\hp\Desktop\Research project")
RESULTS_DIR = PROJECT_ROOT / "results" / "phase7y_borg2019_pilot"
REPORTS_DIR = PROJECT_ROOT / "reports" / "phase7y_borg2019_pilot"


def test_phase7z1_deliverables_exist():
    """Verify that all Phase 7Z.1 deliverable files exist and are non-empty."""
    md_file = REPORTS_DIR / "phase7z1_pre_extraction_authorization_package.md"
    json_results = RESULTS_DIR / "phase7z1_pre_extraction_authorization_package.json"
    json_reports = REPORTS_DIR / "phase7z1_pre_extraction_authorization_package.json"

    assert md_file.is_file(), "phase7z1_pre_extraction_authorization_package.md must exist in reports/"
    assert json_results.is_file(), "phase7z1_pre_extraction_authorization_package.json must exist in results/"
    assert json_reports.is_file(), "phase7z1_pre_extraction_authorization_package.json must exist in reports/"

    assert md_file.stat().st_size > 1000, "Markdown report must be non-trivial"
    assert json_results.stat().st_size > 1000, "JSON results file must be non-trivial"
    assert json_reports.stat().st_size == json_results.stat().st_size, "JSON files must be identical mirrors"


def test_phase7z1_json_verdict_and_checklist():
    """Verify that the JSON package has the declared verdict and 100% PASS on checklist."""
    json_file = RESULTS_DIR / "phase7z1_pre_extraction_authorization_package.json"
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Allowed enum
    allowed_verdicts = {
        "READY_FOR_OWNER_AUTHORIZATION_REVIEW",
        "NOT_READY_QUERY_OR_SCIENTIFIC_ISSUES",
        "NOT_READY_BILLING_OR_BUDGET_UNVERIFIED",
        "NOT_READY_GOVERNANCE_REQUIREMENTS",
    }
    assert data["metadata"]["verdict"] in allowed_verdicts
    assert data["metadata"]["verdict"] == "READY_FOR_OWNER_AUTHORIZATION_REVIEW"

    # All checklist items must be PASS
    checklist = data["checklist"]
    assert len(checklist) >= 15, "Checklist must contain comprehensive items"
    for item in checklist:
        assert item["status"] == "PASS", f"Checklist item failed or unverified: {item['item']}"

    # Governance checks
    holdouts = data["protected_holdouts_status"]
    assert holdouts["candidate_a_test_accessed"] is False
    assert holdouts["phase6_confirmatory_accessed"] is False
    assert holdouts["status"] == "PROTECTED_UNACCESSED"


def test_billing_and_quota_sanity():
    """Verify billing status and free tier utilization figures."""
    json_file = RESULTS_DIR / "phase7z1_pre_extraction_authorization_package.json"
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    gcp = data["gcp_environment_and_billing"]
    assert gcp["billing_enabled"] is False
    assert "ACTIVE_SANDBOX" in gcp["bigquery_sandbox_status"]
    assert gcp["cost_disambiguation"]["expected_billed_financial_charge_usd"] == 0.00
    assert gcp["free_tier_utilization_pct"] == pytest.approx(38.07, rel=1e-2)

    dry = data["service_dry_run_evidence"]
    assert dry["total_bytes_processed"] == 418578205568
    assert dry["service_validation_status"] == "VALIDATED_BY_SERVICE"


def test_mandatory_safety_statement_verbatim():
    """Verify the mandatory safety statement is present verbatim at end of report."""
    md_file = REPORTS_DIR / "phase7z1_pre_extraction_authorization_package.md"
    text = md_file.read_text(encoding="utf-8")

    expected_statement = (
        "No Borg 2019 data extraction, export, staging-table population, pilot dataset creation, "
        "model training, or protected holdout access was authorized or performed during this pre-extraction review. "
        "Any subsequent extraction requires a separate explicit authorization after the research owner reviews this package."
    )
    assert expected_statement in text, "Mandatory safety statement must be present verbatim in the markdown report"

