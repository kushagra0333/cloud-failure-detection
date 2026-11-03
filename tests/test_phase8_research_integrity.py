"""Tests for Phase 8: Research Dataset Integrity, Leakage Audit & Final Testing Readiness."""

import json
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "phase8_research_integrity"
REPORTS_DIR = PROJECT_ROOT / "reports" / "phase8_research_integrity"

EXPECTED_SAFETY_STATEMENT = (
    "No Google Cloud query was executed. No Borg 2019 data was extracted. "
    "No Phase 6 Confirmatory holdout was accessed. No Candidate A Test data was newly accessed during this audit. "
    "No model training or evaluation was performed. No existing research results were overwritten."
)


def test_phase8_deliverables_exist():
    """Verify that all Phase 8 deliverables exist and are non-empty."""
    json_path = RESULTS_DIR / "phase8_dataset_integrity_audit.json"
    audit_md_path = REPORTS_DIR / "phase8_dataset_integrity_audit.md"
    rec_md_path = REPORTS_DIR / "phase8_final_evaluation_recommendation.md"

    assert json_path.is_file(), "phase8_dataset_integrity_audit.json must exist in results/"
    assert audit_md_path.is_file(), "phase8_dataset_integrity_audit.md must exist in reports/"
    assert rec_md_path.is_file(), "phase8_final_evaluation_recommendation.md must exist in reports/"

    assert json_path.stat().st_size > 5000, "JSON result must be non-trivial"
    assert audit_md_path.stat().st_size > 5000, "Audit Markdown report must be non-trivial"
    assert rec_md_path.stat().st_size > 2000, "Recommendation Markdown report must be non-trivial"


def test_phase8_verdict():
    """Verify that the final verdict is strictly NO_VALID_INDEPENDENT_TEST_SET_IDENTIFIED."""
    json_path = RESULTS_DIR / "phase8_dataset_integrity_audit.json"
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    allowed_verdicts = {
        "READY_FOR_FINAL_EVALUATION_PROTOCOL",
        "NO_VALID_INDEPENDENT_TEST_SET_IDENTIFIED",
        "DATA_LEAKAGE_REQUIRES_REMEDIATION",
        "AUDIT_INCOMPLETE",
    }
    verdict = data["metadata"]["final_audit_verdict"]
    assert verdict in allowed_verdicts, f"Verdict {verdict} not in allowed enums"
    assert verdict == "NO_VALID_INDEPENDENT_TEST_SET_IDENTIFIED", "Verdict must be NO_VALID_INDEPENDENT_TEST_SET_IDENTIFIED"


def test_candidate_a_test_exposure_status():
    """Verify Candidate A Test is properly identified as exposed/previously evaluated."""
    json_path = RESULTS_DIR / "phase8_dataset_integrity_audit.json"
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    roles = data["dataset_roles_and_contamination"]
    candidate_a_test = next(
        (r for r in roles if "phase7c/test" in r["dataset_identifier"]),
        None
    )
    assert candidate_a_test is not None, "Candidate A Test role must be categorized"
    assert "CONFIRMED_LEAKAGE" in candidate_a_test["contamination_status"] or "PREVIOUSLY_EVALUATED" in candidate_a_test["contamination_status"]


def test_protected_holdouts_quarantine():
    """Verify Phase 6 Confirmatory holdouts are marked as PROTECTED_UNACCESSED."""
    json_path = RESULTS_DIR / "phase8_dataset_integrity_audit.json"
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    holdouts = data["protected_holdouts"]
    assert holdouts["confirmatory_v1"]["status"] == "PROTECTED_UNACCESSED"
    assert holdouts["confirmatory_l5"]["status"] == "PROTECTED_UNACCESSED"


def test_operational_gate_verdict():
    """Verify that all compared models fail to achieve an operational winner."""
    json_path = RESULTS_DIR / "phase8_dataset_integrity_audit.json"
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    models = data["models_comparison"]
    assert len(models) >= 5, "At least 5 models must be compared"
    for m in models:
        assert m["operational_verdict"] == "NO_OPERATIONAL_WINNER", (
            f"Model {m['model_architecture']} should be marked NO_OPERATIONAL_WINNER"
        )


def test_mandatory_safety_statement():
    """Verify the mandatory safety statement is present verbatim in report and JSON."""
    json_path = RESULTS_DIR / "phase8_dataset_integrity_audit.json"
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["mandatory_safety_statement"] == EXPECTED_SAFETY_STATEMENT

    audit_md_path = REPORTS_DIR / "phase8_dataset_integrity_audit.md"
    assert EXPECTED_SAFETY_STATEMENT in audit_md_path.read_text(encoding="utf-8")

    rec_md_path = REPORTS_DIR / "phase8_final_evaluation_recommendation.md"
    assert EXPECTED_SAFETY_STATEMENT in rec_md_path.read_text(encoding="utf-8")
