# src/graph/state.py

from typing import Any, Dict, List, Optional, TypedDict


class AgentState(TypedDict):
    """Global state passing across LangGraph nodes."""
    raw_query: str
    doc_type: Optional[str]
    jurisdiction: Optional[str]
    extracted_entities: Dict[str, Any]
    retrieved_context: List[Dict[str, Any]]
    compliance_draft: Optional[str]
    confidence_score: float
    loop_count: int
    max_loops: int
    is_audit_complete: bool
    final_report: Optional[Dict[str, Any]]