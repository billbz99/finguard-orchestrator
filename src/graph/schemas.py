# src/graph/schemas.py

from typing import List
from pydantic import BaseModel, Field


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