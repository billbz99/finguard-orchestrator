# src/graph/schemas.py

from typing import List
from pydantic import BaseModel, Field


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

class ComplianceReport(BaseModel):
    """Pydantic schema for structured Suspicious Activity Report (SAR) generation."""
    risk_rating: str = Field(
        description="Low, Medium, or High Risk Assessment of transaction run"
    )
    flagged_wires: List[str] = Field(
        default_factory=list,
        description="List of suspicious wire reference IDs matching illegal AML patterns"
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
        description="Cryptographic or metadata IDs/sources of cited compliance records"
    )
    
class AMLAssessment(BaseModel):
    risk_rating: str = Field(
        description="Low, Medium, or High AML risk assessment"
    )

    suspicious_patterns: List[str] = Field(
        default_factory=list,
        description="AML patterns supported by the available evidence"
    )

    flagged_transactions: List[str] = Field(
        default_factory=list,
        description="Transaction IDs that should be flagged"
    )

    applicable_regulations: List[str] = Field(
        default_factory=list,
        description="Regulations supported by the retrieved context"
    )

    reasoning_summary: str = Field(
        description="Evidence-grounded explanation of the AML assessment"
    )

    insufficient_evidence: bool = Field(
        description="True when there is not enough evidence to make a reliable AML judgment"
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