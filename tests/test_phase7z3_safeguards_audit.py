"""Tests for Phase 7Z.3: Resolve Quota Verification and Git Data Protection."""

import json
from pathlib import Path
import subprocess
import pytest


PROJECT_ROOT = Path(r"C:\Users\hp\Desktop\Research project")
RESULTS_DIR = PROJECT_ROOT / "results" / "phase7y_borg2019_pilot"
REPORTS_DIR = PROJECT_ROOT / "reports" / "phase7y_borg2019_pilot"
GITIGNORE_FILE = PROJECT_ROOT / ".gitignore"


def test_phase7z3_deliverables_exist():
    """Verify that all Phase 7Z.3 deliverable files exist and are non-empty."""
    md_file = REPORTS_DIR / "phase7z3_quota_and_git_safeguards.md"
    json_results = RESULTS_DIR / "phase7z3_quota_and_git_safeguards.json"
    json_reports = REPORTS_DIR / "phase7z3_quota_and_git_safeguards.json"

    assert md_file.is_file(), "phase7z3_quota_and_git_safeguards.md must exist in reports/"
    assert json_results.is_file(), "phase7z3_quota_and_git_safeguards.json must exist in results/"
    assert json_reports.is_file(), "phase7z3_quota_and_git_safeguards.json must exist in reports/"

    assert md_file.stat().st_size > 1000, "Markdown report must be non-trivial"
    assert json_results.stat().st_size > 1000, "JSON results file must be non-trivial"
    assert json_reports.stat().st_size == json_results.stat().st_size, "JSON files must be identical mirrors"


def test_phase7z3_verdict_and_checkpoints():
    """Verify that the JSON audit contains valid verdict and checkpoint classifications."""
    json_file = RESULTS_DIR / "phase7z3_quota_and_git_safeguards.json"
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Allowed verdicts per prompt specification
    allowed_verdicts = {
        "READY_FOR_OWNER_EXTRACTION_DECISION",
        "NOT_READY_BILLING_OR_QUOTA_UNVERIFIED",
        "NOT_READY_GIT_DATA_PROTECTION",
        "NOT_READY_BOTH",
    }
    assert data["metadata"]["final_verdict"] in allowed_verdicts
    assert data["metadata"]["final_verdict"] == "NOT_READY_BILLING_OR_QUOTA_UNVERIFIED"

    # CHK-Z3-05 (Remaining Quota) must be UNVERIFIED
    chk05 = next(c for c in data["checkpoints"] if c["checkpoint_id"] == "CHK-Z3-05")
    assert chk05["status"] == "UNVERIFIED"
    assert "https://console.cloud.google.com/bigquery" in chk05["evidence"]

    # CHK-Z3-08 (.gitignore) must be PASS
    chk08 = next(c for c in data["checkpoints"] if c["checkpoint_id"] == "CHK-Z3-08")
    assert chk08["status"] == "PASS"

    # CHK-Z3-09 (Existing Tracked Raw Data) must be PASS
    chk09 = next(c for c in data["checkpoints"] if c["checkpoint_id"] == "CHK-Z3-09")
    assert chk09["status"] == "PASS"

    # CHK-Z3-10 (Protected Holdouts) must be PASS
    chk10 = next(c for c in data["checkpoints"] if c["checkpoint_id"] == "CHK-Z3-10")
    assert chk10["status"] == "PASS"


def test_phase7z3_gitignore_and_protection():
    """Verify that root .gitignore exists and properly ignores raw data and Parquet files."""
    assert GITIGNORE_FILE.is_file(), "Root .gitignore must exist"
    content = GITIGNORE_FILE.read_text(encoding="utf-8")
    assert "/data/raw/" in content or "data/raw/" in content
    assert "*.parquet" in content

    # Check git check-ignore functionality
    res_parquet = subprocess.run(
        ["git", "check-ignore", "data/raw/borg2019_pilot/pilot_telemetry.parquet"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    assert res_parquet.returncode == 0, "Raw parquet file must be ignored by git"

    res_code = subprocess.run(
        ["git", "check-ignore", "scripts/run_phase7z3_safeguards_audit.py"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    assert res_code.returncode == 1, "Script code must not be ignored by git"


def test_phase7z3_quota_mathematics():
    """Verify that quota arithmetic accurately derives the 634 GiB allowable prior usage threshold."""
    json_file = RESULTS_DIR / "phase7z3_quota_and_git_safeguards.json"
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    qm = data["quota_mathematics"]
    assert qm["total_applicable_quota_gib"] == 1024.0
    assert qm["estimated_query_scan_bytes"] == 418578205568

    # Check that maximum allowable prior usage is approximately 634.17 GiB
    assert round(qm["maximum_allowable_prior_usage_gib"], 2) == 634.17
    assert round(qm["max_prior_usage_with_safety_gib"], 2) == 595.19


def test_phase7z3_mandatory_safety_statement():
    """Verify the mandatory safety statement is present verbatim in report and JSON."""
    expected_statement = (
        "No data-returning BigQuery query, extraction, export, staging operation, pilot dataset creation, "
        "model training, or protected holdout access was authorized or performed during Phase 7Z.3. "
        "Completion of this phase does not authorize data extraction. A separate explicit owner decision is required."
    )

    md_file = REPORTS_DIR / "phase7z3_quota_and_git_safeguards.md"
    text = md_file.read_text(encoding="utf-8")
    assert expected_statement in text, "Mandatory safety statement must be present verbatim in the markdown report"

    json_file = RESULTS_DIR / "phase7z3_quota_and_git_safeguards.json"
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["mandatory_safety_statement"] == expected_statement

