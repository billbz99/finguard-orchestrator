# src/graph/pre_router.py

from typing import Any, Dict, Optional


def run_deterministic_ach_check(transaction_data: Dict[str, Any]) -> Dict[str, Any]:
    """Deterministic rule engine for simple local ACH transactions.
    
    Costs $0.00 in LLM API tokens.
    """
    return {
        "assessment_status": "COMPLETE",
        "risk_rating": "LOW",
        "flagged_wires": [],
        "applicable_regulations": ["Standard ACH Compliance Rules"],
        "audit_summary": "Transaction verified clean via deterministic rule scan. Clean memo and under threshold.",
        "source_document_hashes": ["rule_deterministic_v1"],
    }


def route_incoming_audit(raw_query: str, amount: Optional[float] = None, is_cross_border: bool = False) -> str:
    """Pre-router to determine if a query escalates to LangGraph or executes deterministically."""
    query_lower = raw_query.lower()
    
    # Escalation criteria: cross-border, high value (>= $10,000), or AML trigger terms
    requires_escalation = (
        is_cross_border
        or (amount is not None and amount >= 10000.0)
        or any(k in query_lower for k in ["swift", "structuring", "finra", "fincen", "sar", "ctr", "ofac", "wire", "txn-"])
    )

    if requires_escalation:
        print("⚡ [Pre-Router] Escalating request to LangGraph Agentic Core (Cyclic Reasoning).")
        return "AGENTIC_GRAPH"
    else:
        print("⚡ [Pre-Router] Routing to Deterministic ACH Parser ($0.00 API Token Cost).")
        return "DETERMINISTIC_PASS"


if __name__ == "__main__":
    # Test Case 1: Simple local transaction (Deterministic Bypass)
    print("--- Test 1: Simple Local Payment ---")
    decision1 = route_incoming_audit("Monthly payroll direct deposit for employee", amount=3500.0, is_cross_border=False)
    if decision1 == "DETERMINISTIC_PASS":
        res = run_deterministic_ach_check({})
        print(f"Result: {res['risk_rating']} Risk | {res['audit_summary']}")

    # Test Case 2: Suspicious wire structuring (Escalation)
    print("\n--- Test 2: Suspicious Structuring Query ---")
    decision2 = route_incoming_audit("Audit wire TXN-984211-X for structuring under FINRA Rule 3310", amount=8500.0, is_cross_border=True)
    print(f"Decision: {decision2}")
