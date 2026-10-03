# src/graph/schemas.py

from collections.abc import Mapping
from typing import List, Literal
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator


AssessmentStatus = Literal["COMPLETE", "INSUFFICIENT_EVIDENCE"]
EvidenceGap = Literal[
    "AMOUNT",
    "TIMING",
    "COUNTERPARTIES",
    "JURISDICTION",
    "TRANSACTION_RELATIONSHIP",
    "TRANSACTION_HISTORY",
    "PURPOSE",
    "INTENT",
    "REGULATORY_CONTEXT",
    "MATERIAL_CONFLICT",
]
VALID_ASSESSMENT_STATUSES = frozenset({"COMPLETE", "INSUFFICIENT_EVIDENCE"})


def has_valid_assessment_status(report: object) -> bool:
    """Returns whether a report carries a current assessment-status value."""
    return (
        isinstance(report, Mapping)
        and report.get("assessment_status") in VALID_ASSESSMENT_STATUSES
    )


class TransactionExtraction(BaseModel):
    """Structured entities extracted from an AML audit request."""

    transaction_ids: List[str] = Field(
        default_factory=list,
        description="Transaction or wire reference IDs explicitly mentioned in the request"
    )

    amount: float | None = Field(
        default=None,
        description="Transaction amount explicitly mentioned in the request"
    )

    transaction_type: str | None = Field(
        default=None,
        description="Transaction type, such as wire, ACH, cash deposit, or transfer"
    )

    regulations: List[str] = Field(
        default_factory=list,
        description="Regulations or regulatory rules explicitly mentioned in the request"
    )

    suspected_patterns: List[str] = Field(
        default_factory=list,
        description="AML patterns mentioned or suspected in the request, such as structuring"
    )

    jurisdiction: str | None = Field(
        default=None,
        description="Jurisdiction explicitly mentioned in the request"
    )

class EvidenceAttribution(BaseModel):
    """Concise model-attributed support, not private reasoning or proof."""
    model_config = ConfigDict(extra="forbid")

    claim: str = Field(min_length=1, description="Exact existing suspicious pattern or applicable regulation")
    evidence_references: list[str] = Field(min_length=1, description="References of admitted evidence blocks supplied in this assessment pass")
    support_summary: str = Field(min_length=1, max_length=600, description="Concise analyst-facing support statement; no hidden reasoning or full document quotations")


class AssessmentAttribution(BaseModel):
    model_config = ConfigDict(extra="forbid")

    findings: list[EvidenceAttribution] = Field(default_factory=list)
    regulations: list[EvidenceAttribution] = Field(default_factory=list)


class EvidenceProvenance(BaseModel):
    """Allowlisted provenance for evidence actually supplied to an assessment."""
    model_config = ConfigDict(extra="forbid")

    evidence_reference: str
    indexed_document_id: str
    source_label: str | None = None
    document_type: str | None = None
    evidence_role: Literal["transaction_record", "regulatory_guidance", "unknown"]
    record_locator: str | None = None
    chunk_locator: str | int | None = None
    jurisdiction: str | None = None


class ReportExplainability(BaseModel):
    model_config = ConfigDict(extra="forbid")

    suspicious_patterns: list[str] = Field(default_factory=list)
    assessment_summary: str
    required_evidence_gaps: list[EvidenceGap] = Field(default_factory=list)
    admitted_evidence: list[EvidenceProvenance] = Field(default_factory=list)
    finding_attributions: list[EvidenceAttribution] = Field(default_factory=list)
    regulation_attributions: list[EvidenceAttribution] = Field(default_factory=list)
    attribution_status: Literal["available", "partial", "unavailable"]
    critic_action: str | None = None
    critic_failure_type: str | None = None
    refinement_occurred: bool | None = None


class ComplianceReport(BaseModel):
    """Pydantic schema for structured Suspicious Activity Report (SAR) generation."""
    assessment_status: AssessmentStatus = Field(
        description="Whether the AML assessment completed or lacked sufficient evidence"
    )
    risk_rating: str = Field(
        description="Low, Medium, or High Risk Assessment of transaction run"
    )
    flagged_wires: List[str] = Field(
        default_factory=list,
        description=(
            "Wire reference IDs requiring AML review based on available evidence; "
            "inclusion does not confirm illegal activity"
        )
    )
    applicable_regulations: List[str] = Field(
        default_factory=list,
        description="References to audited compliance sections and regulatory clauses"
    )
    audit_summary: str = Field(
        description="Markdown formatted detailed explanation of the analytical findings"
    )
    source_document_hashes: List[str] = Field(
        default_factory=list,
        description="Legacy field containing source labels, typically filenames; not content hashes or verified citations"
    )
    explainability: ReportExplainability | None = Field(default=None, description="Final-pass provenance and model-attributed support, separate from runtime telemetry")
    
class AMLAssessment(BaseModel):
    evidence_attribution: AssessmentAttribution | None = Field(
        default=None, description="Optional support links for existing findings/regulations to this pass's admitted references"
    )

    @field_validator("evidence_attribution", mode="before")
    @classmethod
    def optional_attribution(cls, value: object) -> AssessmentAttribution | None:
        """Malformed optional attribution must not invalidate core AML output."""
        if value is None:
            return None
        try:
            return AssessmentAttribution.model_validate(value)
        except (ValidationError, TypeError, ValueError):
            return None

    risk_rating: str = Field(
        description="Low, Medium, or High AML risk assessment"
    )

    suspicious_patterns: List[str] = Field(
        default_factory=list,
        description="AML patterns supported by the available evidence"
    )

    flagged_transactions: List[str] = Field(
        default_factory=list,
        description=(
            "Transaction IDs requiring AML review based on available evidence; "
            "inclusion does not confirm illegal activity"
        )
    )

    applicable_regulations: List[str] = Field(
        default_factory=list,
        description="Regulations supported by the retrieved context"
    )

    required_evidence_gaps: List[EvidenceGap] = Field(
        description=(
            "Evidence missing or unresolved that is required for the specific "
            "AML conclusion; do not list merely absent optional facts"
        )
    )

    reasoning_summary: str = Field(
        description="Evidence-grounded explanation of the AML assessment"
    )

    insufficient_evidence: bool = Field(
        description=(
            "Compatibility field normalized by the application from "
            "required_evidence_gaps"
        )
    )
    
class CriticAssessment(BaseModel):
    """Critiques the AML assessment and recommends the next workflow action."""

    is_sufficient: bool = Field(
        description="Whether the available evidence is sufficient to finalize the AML assessment"
    )

    missing_evidence: List[str] = Field(
        default_factory=list,
        description="Specific evidence that is still missing"
    )

    failure_type: str = Field(
        description=(
            "Reason evidence is insufficient. "
            "Use one of: NONE, MISSING_TRANSACTION_DATA, "
            "MISSING_REGULATORY_CONTEXT, or INCONSISTENT_ANALYSIS"
        )
    )

    recommended_action: str = Field(
        description=(
            "Next workflow action. "
            "Use one of: GENERATE, RETRIEVE_MORE, or STOP_INSUFFICIENT"
        )
    )

    critique: str = Field(
        description="Short explanation of why this action was selected"
    )
