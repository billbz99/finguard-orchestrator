import json

import pytest
from pydantic import ValidationError

from src.graph.pre_router import run_deterministic_ach_check
from src.graph.schemas import ComplianceReport, has_valid_assessment_status


def report_payload(status):
    return {
        "assessment_status": status,
        "risk_rating": "LOW",
        "flagged_wires": [],
        "applicable_regulations": [],
        "audit_summary": "Assessment summary.",
        "source_document_hashes": [],
    }


@pytest.mark.parametrize("status", ["COMPLETE", "INSUFFICIENT_EVIDENCE"])
def test_compliance_report_accepts_supported_assessment_status(status):
    report = ComplianceReport.model_validate(report_payload(status))

    assert report.assessment_status == status


def test_compliance_report_requires_assessment_status():
    payload = report_payload("COMPLETE")
    del payload["assessment_status"]

    with pytest.raises(ValidationError):
        ComplianceReport.model_validate(payload)


def test_compliance_report_rejects_unsupported_assessment_status():
    with pytest.raises(ValidationError):
        ComplianceReport.model_validate(report_payload("UNKNOWN"))


def test_deterministic_report_is_complete_and_schema_valid():
    report = run_deterministic_ach_check({"query": "Monthly payroll"})

    assert report["assessment_status"] == "COMPLETE"
    assert ComplianceReport.model_validate(report).model_dump() == report


def test_current_cached_report_preserves_assessment_status():
    cached_report = json.loads(json.dumps(report_payload("INSUFFICIENT_EVIDENCE")))

    assert has_valid_assessment_status(cached_report) is True
    assert cached_report["assessment_status"] == "INSUFFICIENT_EVIDENCE"


def test_legacy_cached_report_is_stale_and_not_interpreted_as_complete():
    legacy_report = report_payload("COMPLETE")
    del legacy_report["assessment_status"]
    original_report = dict(legacy_report)

    assert has_valid_assessment_status(legacy_report) is False
    assert legacy_report == original_report
    assert "assessment_status" not in legacy_report


@pytest.mark.parametrize("status", [None, "UNKNOWN", "complete"])
def test_invalid_cached_status_is_stale(status):
    cached_report = report_payload(status)

    assert has_valid_assessment_status(cached_report) is False
