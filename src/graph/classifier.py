# src/graph/classifier.py

from typing import Dict, Any
from src.graph.state import GraphState


def classify_query_intent(state: GraphState) -> Dict[str, Any]:
    """Inspects the raw input query and extracts dynamic metadata filters.
    
    Can be powered by a lightweight LLM call or regex/rule extraction.
    """
    query = state["query"].lower()
    updates = {}

    # Identify document type scope
    if any(k in query for k in ["swift", "transaction", "field :50k:", "field :59:", "message"]):
        updates["doc_type"] = "swift_log"
    elif any(k in query for k in ["finra", "fincen", "rule", "structuring", "ctr", "sar", "aml"]):
        updates["doc_type"] = "regulatory_pdf"

    # Identify jurisdiction scope
    if "ofac" in query or "us" in query:
        updates["jurisdiction"] = "US_OFAC"

    return updates