"""Tests for Phase 7Z.2: Independent Final Authorization Readiness Audit."""

import json
from pathlib import Path
import pytest


PROJECT_ROOT = Path(r"C:\Users\hp\Desktop\Research project")
RESULTS_DIR = PROJECT_ROOT / "results" / "phase7y_borg2019_pilot"
REPORTS_DIR = PROJECT_ROOT / "reports" / "phase7y_borg2019_pilot"


def test_phase7z2_deliverables_exist():
    """Verify that all Phase 7Z.2 deliverable files exist and are non-empty."""
    md_file = REPORTS_DIR / "phase7z2_final_authorization_readiness_audit.md"
    json_results = RESULTS_DIR / "phase7z2_final_authorization_readiness_audit.json"
    json_reports = REPORTS_DIR / "phase7z2_final_authorization_readiness_audit.json"

    assert md_file.is_file(), "phase7z2_final_authorization_readiness_audit.md must exist in reports/"
    assert json_results.is_file(), "phase7z2_final_authorization_readiness_audit.json must exist in results/"
    assert json_reports.is_file(), "phase7z2_final_authorization_readiness_audit.json must exist in reports/"

    assert md_file.stat().st_size > 1000, "Markdown report must be non-trivial"
    assert json_results.stat().st_size > 1000, "JSON results file must be non-trivial"
    assert json_reports.stat().st_size == json_results.stat().st_size, "JSON files must be identical mirrors"


def test_phase7z2_verdict_and_unverified_quota():
    """Verify that the JSON audit contains valid verdict and unverified quota classification."""
    json_file = RESULTS_DIR / "phase7z2_final_authorization_readiness_audit.json"
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Allowed enum
    allowed_verdicts = {
        "READY_FOR_OWNER_EXTRACTION_DECISION",
        "NOT_READY_BILLING_OR_QUOTA_UNVERIFIED",
        "NOT_READY_SQL_OR_SCIENTIFIC_VALIDITY",
        "NOT_READY_EXTRACTION_SAFEGUARDS",
        "NOT_READY_GOVERNANCE",
    }
    assert data["metadata"]["final_audit_verdict"] in allowed_verdicts
    assert data["metadata"]["final_audit_verdict"] == "NOT_READY_BILLING_OR_QUOTA_UNVERIFIED"

    # CHK-08 (Remaining Quota) must be UNVERIFIED
    chk08 = next(c for c in data["checkpoints"] if c["checkpoint_id"] == "CHK-08")
    assert chk08["status"] == "UNVERIFIED"
    assert "https://console.cloud.google.com/bigquery" in chk08["evidence"]

    # CHK-17 (.gitignore) must be UNVERIFIED
    chk17 = next(c for c in data["checkpoints"] if c["checkpoint_id"] == "CHK-17")
    assert chk17["status"] == "UNVERIFIED"

    # CHK-18 (Protected Holdouts) must be PASS
    chk18 = next(c for c in data["checkpoints"] if c["checkpoint_id"] == "CHK-18")
    assert chk18["status"] == "PASS"


def test_phase7z2_mandatory_safety_statement():
    """Verify the mandatory safety statement is present verbatim at end of report."""
    md_file = REPORTS_DIR / "phase7z2_final_authorization_readiness_audit.md"
    text = md_file.read_text(encoding="utf-8")

    expected_statement = (
        "No data-returning query, extraction, export, staging operation, pilot dataset creation, "
        "model training, or protected holdout access was authorized or performed during this audit. "
        "A separate explicit owner decision is required before any extraction."
    )
    assert expected_statement in text, "Mandatory safety statement must be present verbatim in the markdown report"

