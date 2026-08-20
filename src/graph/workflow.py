# src/graph/workflow.py

import json
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from src.graph.state import AgentState
from src.graph.nodes import (
    extraction_node,
    aml_audit_node,
    auditor_critic_node,
    structured_generation_node,
)

# Load environment variables (LANGCHAIN_API_KEY, LANGCHAIN_TRACING_V2, etc.)
load_dotenv()


def should_continue_audit(state: AgentState) -> str:
    """Conditional routing function.
    
    Evaluates whether the audit should loop back for context refinement
    or proceed to final structured report generation.
    """
    if state.get("is_audit_complete", False):
        print("➡️ [Routing] Audit complete or max loops reached. Proceeding to Generation Node.")
        return "generate"
    else:
        print("🔄 [Routing] Confidence score < 0.80. Looping back to AML Audit Node for refinement...")
        return "refine"


def build_finguard_graph():
    """Assembles and compiles the cyclic LangGraph workflow for FinGuard Orchestrator."""
    workflow = StateGraph(AgentState)

    # 1. Add Nodes
    workflow.add_node("extraction", extraction_node)
    workflow.add_node("aml_audit", aml_audit_node)
    workflow.add_node("auditor_critic", auditor_critic_node)
    workflow.add_node("generation", structured_generation_node)

    # 2. Set Entry Point and Linear Edges
    workflow.set_entry_point("extraction")
    workflow.add_edge("extraction", "aml_audit")
    workflow.add_edge("aml_audit", "auditor_critic")

    # 3. Add Conditional Edge for Critic Loop Check
    workflow.add_conditional_edges(
        "auditor_critic",
        should_continue_audit,
        {
            "generate": "generation",
            "refine": "aml_audit",
        },
    )

    # 4. Final Edge to Exit
    workflow.add_edge("generation", END)

    # 5. Compile Graph
    return workflow.compile()


if __name__ == "__main__":
    app = build_finguard_graph()

    initial_state: AgentState = {
        "raw_query": "Audit wire TXN-984211-X for structuring and CTR threshold evasions under FINRA Rule 3310.",
        "doc_type": None,
        "jurisdiction": None,
        "extracted_entities": {},
        "retrieved_context": [],
        "compliance_draft": None,
        "confidence_score": 0.0,
        "loop_count": 0,
        "max_loops": 2,
        "is_audit_complete": False,
        "final_report": None,
    }

    # Injected corporate run-tags and metadata for LangSmith tracing
    config = {
        "tags": ["AML_AUDIT_RUN", "BATCH_PROD"],
        "metadata": {
            "client_tier": "VIP_Institutional",
            "audit_id": "aud-9988-xx",
            "batch_wire_count": len(initial_state["extracted_entities"].get("wires", [])),
        },
    }

    print("🚀 Executing FinGuard Agentic Graph Workflow with LangSmith Tracing...\n")
    output = app.invoke(initial_state, config=config)

    print("\n" + "=" * 50)
    print("📋 FINAL STRUCTURED COMPLIANCE REPORT (JSON)")
    print("=" * 50)
    print(json.dumps(output["final_report"], indent=2))