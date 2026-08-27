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