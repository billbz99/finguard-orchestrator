import pytest
from pydantic import ValidationError

from src.graph.evidence_policy import DeficiencyType, evaluate_evidence_policy
from src.graph.schemas import AMLAssessment


@pytest.mark.parametrize(
    ("gaps", "expected_type", "expected_finalizable"),
    [
        ([], DeficiencyType.NONE, True),
        (["AMOUNT"], DeficiencyType.TRANSACTION, False),
        (["JURISDICTION"], DeficiencyType.TRANSACTION, False),
        (["REGULATORY_CONTEXT"], DeficiencyType.REGULATORY, False),
        (["MATERIAL_CONFLICT"], DeficiencyType.MATERIAL_CONFLICT, False),
        (
            ["REGULATORY_CONTEXT", "AMOUNT"],
            DeficiencyType.TRANSACTION,
            False,
        ),
        (
            ["AMOUNT", "MATERIAL_CONFLICT", "REGULATORY_CONTEXT"],
            DeficiencyType.MATERIAL_CONFLICT,
            False,
        ),
    ],
)
def test_evidence_policy_precedence(gaps, expected_type, expected_finalizable):
    decision = evaluate_evidence_policy(gaps)

    assert decision.deficiency_type == expected_type
    assert decision.is_finalizable is expected_finalizable
    assert decision.required_gaps == tuple(gaps)


def test_evidence_policy_deduplicates_gaps_without_reordering():
    decision = evaluate_evidence_policy(["AMOUNT", "AMOUNT", "TIMING"])

    assert decision.required_gaps == ("AMOUNT", "TIMING")


def test_aml_assessment_rejects_unknown_evidence_gap():
    with pytest.raises(ValidationError):
        AMLAssessment(
            risk_rating="Low",
            suspicious_patterns=[],
            flagged_transactions=[],
            applicable_regulations=[],
            required_evidence_gaps=["UNKNOWN"],
            reasoning_summary="Synthetic assessment.",
            insufficient_evidence=True,
        )


def test_required_evidence_gap_list_is_required_for_new_assessments():
    with pytest.raises(ValidationError):
        AMLAssessment(
            risk_rating="Low",
            suspicious_patterns=[],
            flagged_transactions=[],
            applicable_regulations=[],
            reasoning_summary="Synthetic assessment.",
            insufficient_evidence=False,
        )
